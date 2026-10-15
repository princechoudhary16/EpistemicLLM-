import re
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, revision=config.REVISION, dtype=torch.float32
    ).to(config.DEVICE)
    model.eval()
    return tokenizer, model

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(tokenizer, model, prompt: str, do_sample: bool, seed: int | None = None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
    gen_kwargs = dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=do_sample,
        top_k=config.SAMPLE_TOP_K,
        repetition_penalty=config.REPETITION_PENALTY,
    )
    if do_sample:
        gen_kwargs.update(temperature=config.SAMPLE_TEMPERATURE, top_p=config.SAMPLE_TOP_P)
    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def generate_for_questions(tokenizer, model, questions_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in questions_df.itertuples():
        prompt = build_prompt(tokenizer, row.problem)
        records.append(dict(
            question_id=row.question_id, sample_index=0, is_greedy=True,
            generation_seed=None, model_answer=generate(tokenizer, model, prompt, do_sample=False),
        ))
        for i in range(1, config.N_SAMPLES_PER_QUESTION + 1):
            seed = row.question_id * 1000 + i
            records.append(dict(
                question_id=row.question_id, sample_index=i, is_greedy=False,
                generation_seed=seed, model_answer=generate(tokenizer, model, prompt, do_sample=True, seed=seed),
            ))
    return pd.DataFrame(records)

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def contains_answer(answer: str, reference: str) -> bool:
    """Match a normalized reference as whole words, never as an arbitrary substring."""
    norm_answer = normalize(answer)
    norm_reference = normalize(reference)
    return bool(norm_reference) and f" {norm_reference} " in f" {norm_answer} "

def grade(model_answer: str, reference_answer: str) -> str:
    norm_ans = normalize(model_answer)
    if any(p in norm_ans for p in config.ABSTENTION_PHRASES):
        return "abstain"
    if contains_answer(model_answer, reference_answer):
        return "correct"
    return "incorrect"

def grade_responses(gen_df: pd.DataFrame, questions_df: pd.DataFrame) -> pd.DataFrame:
    gen_df = gen_df.merge(questions_df[["question_id", "answer"]], on="question_id")
    gen_df["outcome"] = gen_df.apply(lambda r: grade(r["model_answer"], r["answer"]), axis=1)
    gen_df["grading_version"] = config.GRADING_VERSION
    return gen_df

def knowledge_scores(graded_df: pd.DataFrame) -> pd.DataFrame:
    def k(group):
        sampled = group[~group["is_greedy"]] # rows where is_greedy is false
        return (sampled["outcome"] == "correct").mean() #checks if the outcome is correct, 1 if True, 0 is False. Gives us fraction of the 5 greedy answers that were correct
    return graded_df.groupby("question_id").apply(k).rename("K").reset_index()
