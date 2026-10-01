"""P1 Week-1 sanity run, self-contained for Colab (no repo access needed).

Fine-tunes Qwen2.5-0.5B-Instruct on the pilot_v0 synthetic corpus (facts shown
0/1/2/5/10/20 times) and evaluates with 1 greedy + 5 sampled generations per
question, exactly like P2's SimpleQA pilot. Reports the K(q) bucket
distribution overall and by exposure level.

Purpose: show whether the synthetic design can produce "inconsistent" (0<K<1)
examples where SimpleQA couldn't -- NOT a zero-shot check, since an untrained
model has zero chance of ever being correct on an invented fact (that would
be 100% K=0 by construction and prove nothing).

Run top to bottom in Colab with Runtime > Change runtime type > GPU (T4 is enough).
"""

# ============================= 1. SETUP =====================================
# !pip install -q -U torchao transformers peft accelerate
# (-U torchao matters: Colab's preinstalled torchao is older than what peft's
#  get_peft_model() requires, and peft raises ImportError without it.)

import json
import random
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

DATA_DIR = Path("/content/data/synthetic/pilot_v0")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

SYSTEM_PROMPT = (
    "Answer the question as briefly and factually as possible. "
    "If you do not know the answer, say \"I don't know\" instead of guessing."
)
MAX_NEW_TOKENS = 64
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9
N_SAMPLES_PER_QUESTION = 5
ABSTENTION_PHRASES = [
    "i dont know", "im not sure", "i am not sure",
    "cannot determine", "no information available",
]

# ===================== 2. GENERATOR (embedded from p1_exposure/) ============
# Verbatim from names.py / relations.py / generate.py / validate.py on branch
# p1/synthetic-generator, so this reproduces data/synthetic/pilot_v0 byte-for-byte.

ONSETS = ["b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "dr", "gl", "kr", "pl", "sh", "th", "tr", "vl", "zh"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ou"]
CODAS = ["", "", "", "n", "r", "l", "s", "th", "m", "k"]
ORG_NOUNS = ["Guild", "Institute", "Company", "Society", "Foundry", "League", "Trust"]
MIN_WORD_LEN = 6


def _word(rng, n_syllables):
    parts = []
    for i in range(n_syllables):
        coda = rng.choice(CODAS) if i == n_syllables - 1 else rng.choice(CODAS[:4])
        parts.append(rng.choice(ONSETS) + rng.choice(VOWELS) + coda)
    return "".join(parts).capitalize()


class NamePool:
    def __init__(self, seed):
        self._rng = random.Random(f"{seed}-names")
        self._used = set()

    def _fresh_word(self):
        while True:
            w = _word(self._rng, self._rng.choice([2, 3, 3]))
            lw = w.lower()
            if len(w) >= MIN_WORD_LEN and not any(lw in u or u in lw for u in self._used):
                self._used.add(lw)
                return w

    def place(self):
        return self._fresh_word()

    def person(self):
        return f"{self._fresh_word()} {self._fresh_word()}"

    def organization(self):
        return f"{self._fresh_word()} {self._rng.choice(ORG_NOUNS)}"


OCCUPATIONS = [
    "cartographer", "lighthouse keeper", "glassblower", "beekeeper", "astronomer",
    "shipwright", "cobbler", "harpsichord maker", "falconer", "tapestry weaver",
    "clockmaker", "salt merchant", "bell founder", "herbalist", "surveyor",
    "ropemaker", "violin maker", "bookbinder", "stonemason", "chandler",
    "dyer", "cooper", "fletcher", "tanner", "lens grinder",
    "potter", "ferryman", "mapmaker's apprentice", "orchardist", "wheelwright",
]
YEAR_MIN, YEAR_MAX, YEAR_BIN = 1820, 1989, 10


@dataclass(frozen=True)
class Relation:
    name: str
    subject_type: str
    answer_type: str
    make_subject: Callable
    make_answers: Callable
    train: tuple
    val: tuple
    test: tuple


def _years(pool, rng, n):
    bins = list(range(YEAR_MIN, YEAR_MAX, YEAR_BIN))
    out = []
    for i in range(n):
        start = bins[i % len(bins)]
        out.append((str(rng.randrange(start, start + YEAR_BIN)), start))
    rng.shuffle(out)
    return out


def _persons(pool, rng, n):
    return [(pool.person(), None) for _ in range(n)]


def _places(pool, rng, n):
    return [(pool.place(), None) for _ in range(n)]


def _occupations(pool, rng, n):
    order = OCCUPATIONS[:]
    rng.shuffle(order)
    return [(order[i % len(order)], order[i % len(order)]) for i in range(n)]


RELATIONS = {
    "birth_year": Relation(
        "birth_year", "person", "year", lambda p: p.person(), _years,
        train=("In what year was {s} born?", "{s} was born in which year?",
               "What is the birth year of {s}?", "Which year saw the birth of {s}?"),
        val=("When was {s} born? Give the year.", "Can you tell me the year {s} was born?"),
        test=("What year did {s} come into the world?", "The person named {s} was born in what year?"),
    ),
    "founder": Relation(
        "founder", "organization", "person", lambda p: p.organization(), _persons,
        train=("Who founded the {s}?", "Who was the founder of the {s}?",
               "The {s} was established by whom?", "Name the founder of the {s}."),
        val=("Which person is credited with founding the {s}?", "Who set up the {s}?"),
        test=("Who started the {s}?", "The {s} owes its founding to which person?"),
    ),
    "capital": Relation(
        "capital", "country", "place", lambda p: p.place(), _places,
        train=("What is the capital of {s}?", "Which city is the capital of {s}?",
               "{s} has which city as its capital?", "Name the capital city of {s}."),
        val=("What city serves as the seat of government of {s}?", "Where is the capital of {s}?"),
        test=("Which city does {s} govern from?", "What is the name of {s}'s capital?"),
    ),
    "headquarters": Relation(
        "headquarters", "organization", "place", lambda p: p.organization(), _places,
        train=("Where is the {s} headquartered?", "In which city is the {s} based?",
               "What city is home to the headquarters of the {s}?",
               "The {s} has its main office in which city?"),
        val=("Which city hosts the head office of the {s}?", "What is the home city of the {s}?"),
        test=("The {s} operates out of which city?", "Where is the main base of the {s}?"),
    ),
    "occupation": Relation(
        "occupation", "person", "category", lambda p: p.person(), _occupations,
        train=("What was the occupation of {s}?", "What did {s} do for a living?",
               "{s} worked as what?", "What was {s}'s profession?"),
        val=("What kind of work was {s} known for?", "What trade did {s} practise?"),
        test=("How did {s} earn a living?", "Which line of work was {s} in?"),
    ),
}
DEFAULT_RELATIONS = list(RELATIONS)
LEVELS = [0, 1, 2, 5, 10, 20]


def render(template, subject):
    return template.format(s=subject)


def build_facts(n_facts, relation_names, levels, dev_frac, seed):
    n_rel = n_facts // len(relation_names)
    pool = NamePool(seed)
    facts = []
    for rname in relation_names:
        rel = RELATIONS[rname]
        rng = random.Random(f"{seed}-{rname}")
        subjects = [rel.make_subject(pool) for _ in range(n_rel)]
        answers = rel.make_answers(pool, rng, n_rel)
        rel_facts = [{"relation": rname, "subject_type": rel.subject_type,
                      "answer_type": rel.answer_type, "subject": s,
                      "reference_answer": a, "stratum": st}
                     for s, (a, st) in zip(subjects, answers)]
        order = list(range(n_rel))
        rng.shuffle(order)
        order.sort(key=lambda i: str(rel_facts[i]["stratum"]))
        for b in range(0, n_rel, len(levels)):
            block_levels = levels[:]
            rng.shuffle(block_levels)
            for i, lvl in zip(order[b:b + len(levels)], block_levels):
                rel_facts[i]["exposure_level"] = lvl
        cells = defaultdict(list)
        for f in rel_facts:
            cells[f["exposure_level"]].append(f)
        for lvl in levels:
            members = cells[lvl]
            rng.shuffle(members)
            n_dev = round(dev_frac * len(members))
            for k, f in enumerate(members):
                f["split"] = "dev" if k < n_dev else "final"
        facts.extend(rel_facts)
    for n, f in enumerate(facts, start=1):
        f["fact_id"] = f"F{n:04d}"
    return facts


def build_train(facts, seed, version):
    rows = []
    for f in facts:
        e = f["exposure_level"]
        if e == 0:
            continue
        rel = RELATIONS[f["relation"]]
        offset = random.Random(f"{seed}-offset-{f['fact_id']}").randrange(len(rel.train))
        for j in range(e):
            t = (offset + j) % len(rel.train)
            rows.append({"fact_id": f["fact_id"], "relation": f["relation"],
                        "exposure_level": e, "split": f["split"],
                        "prompt": render(rel.train[t], f["subject"]),
                        "completion": f["reference_answer"]})
    random.Random(f"{seed}-train-order").shuffle(rows)
    return rows


def build_eval(facts, version):
    rows = []
    for f in facts:
        rel = RELATIONS[f["relation"]]
        kind, templates = ("val", rel.val) if f["split"] == "dev" else ("test", rel.test)
        for k, t in enumerate(templates):
            rows.append({"question_id": f"{f['fact_id']}-q{k}", "fact_id": f["fact_id"],
                        "split": f["split"], "relation": f["relation"],
                        "answer_type": f["answer_type"], "exposure_level": f["exposure_level"],
                        "question": render(t, f["subject"]), "reference_answer": f["reference_answer"]})
    return rows


def generate_pilot(n_facts=120, seed=0, dev_frac=0.5):
    facts = build_facts(n_facts, DEFAULT_RELATIONS, LEVELS, dev_frac, seed)
    train = build_train(facts, seed, "pilot_v0")
    evals = build_eval(facts, "pilot_v0")
    by_level = Counter(f["exposure_level"] for f in facts)
    assert set(by_level) == set(LEVELS) and len(set(by_level.values())) == 1, "generator drifted from repo version"
    return facts, train, evals


facts, train_rows, eval_rows = generate_pilot()
print(f"{len(facts)} facts, {len(train_rows)} train presentations, {len(eval_rows)} eval questions")
print("facts per exposure level:", dict(Counter(f["exposure_level"] for f in facts)))

# =========================== 3. LOAD MODEL + LORA ============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# float32 throughout: Colab's free-tier T4 has no native bf16 support and fp16
# risks silent NaNs during LoRA training on a model this small -- fp32 is slower
# but bulletproof, and 0.5B params on 760 examples is still fast on any GPU.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ================================ 4. TRAIN ===================================
# One pass over train_rows: exposure count is already encoded as repeated rows
# (a 10x fact appears 10 times), so one epoch here *is* "shown N times" per
# the team's data design -- do not add extra epochs without updating that contract.


def build_example(prompt, completion):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return input_ids, labels


def collate(batch):
    max_len = max(len(ids) for ids, _ in batch)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids).to(DEVICE), torch.tensor(labels).to(DEVICE),
            torch.tensor(attn).to(DEVICE))


examples = [build_example(r["prompt"], r["completion"]) for r in train_rows]
random.Random("train-shuffle").shuffle(examples)

BATCH_SIZE = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()
for step in range(0, len(examples), BATCH_SIZE):
    batch = examples[step:step + BATCH_SIZE]
    input_ids, labels, attn = collate(batch)
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 40 == 0:
        print(f"step {step}/{len(examples)}  loss {out.loss.item():.3f}")
print("training done.")

# =============================== 5. GENERATE =================================

model.eval()


@torch.no_grad()
def generate_one(prompt, do_sample, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P)
    out_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


responses = []
for i, q in enumerate(eval_rows):
    responses.append({"question_id": q["question_id"], "fact_id": q["fact_id"],
                      "exposure_level": q["exposure_level"], "is_greedy": True,
                      "model_answer": generate_one(q["question"], do_sample=False)})
    for s in range(1, N_SAMPLES_PER_QUESTION + 1):
        # hashlib, not builtin hash(): str hashing is salted per-process in
        # Python, so hash() would make these seeds (and results) non-reproducible.
        digest = hashlib.sha256(f"{q['question_id']}-{s}".encode()).hexdigest()
        seed = int(digest[:8], 16)
        responses.append({"question_id": q["question_id"], "fact_id": q["fact_id"],
                          "exposure_level": q["exposure_level"], "is_greedy": False,
                          "model_answer": generate_one(q["question"], do_sample=True, seed=seed)})
    if i % 20 == 0:
        print(f"generated {i}/{len(eval_rows)} questions")
print("generation done:", len(responses), "responses")

# ============================= 6. GRADE + REPORT ==============================

ref_by_q = {q["question_id"]: q["reference_answer"] for q in eval_rows}


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def grade(model_answer, reference_answer):
    norm_ans = normalize(model_answer)
    if any(p in norm_ans for p in ABSTENTION_PHRASES):
        return "abstain"
    if normalize(reference_answer) in norm_ans:
        return "correct"
    return "incorrect"


for r in responses:
    r["outcome"] = grade(r["model_answer"], ref_by_q[r["question_id"]])

by_fact = defaultdict(list)
for r in responses:
    if not r["is_greedy"]:
        by_fact[r["fact_id"]].append(r["outcome"] == "correct")

k_by_fact = {fid: sum(v) / len(v) for fid, v in by_fact.items()}
level_by_fact = {q["fact_id"]: q["exposure_level"] for q in eval_rows}


def bucket(k):
    if k == 0:
        return "K=0"
    if k >= 0.8:
        return "K>=0.8"
    return "0<K<1 (inconsistent)"


print("\n=== Overall K(q) bucket distribution (facts, not questions) ===")
overall = Counter(bucket(k) for k in k_by_fact.values())
for b, c in overall.items():
    print(f"  {b}: {c} / {len(k_by_fact)}")

print("\n=== K(q) bucket distribution by exposure level ===")
by_level_bucket = defaultdict(Counter)
for fid, k in k_by_fact.items():
    by_level_bucket[level_by_fact[fid]][bucket(k)] += 1
for lvl in sorted(by_level_bucket):
    print(f"  exposure {lvl}x: {dict(by_level_bucket[lvl])}")

import pandas as pd
pd.DataFrame(responses).to_csv("/content/p1_pilot_finetune_responses.csv", index=False)
pd.DataFrame([{"fact_id": f, "exposure_level": level_by_fact[f], "K": k}
             for f, k in k_by_fact.items()]).to_csv("/content/p1_pilot_finetune_k_scores.csv", index=False)
print("\nSaved /content/p1_pilot_finetune_responses.csv and p1_pilot_finetune_k_scores.csv")
