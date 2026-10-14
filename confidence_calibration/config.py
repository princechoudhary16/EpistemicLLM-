"""Pinned model, prompt and decoding settings.

Decoding values follow the team contract (plan.md §3). Every setting that can change a reported
number carries a version string, and each response row records it.
"""

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
BASE_CHECKPOINT_ID = "base"

# The prompt and grader are provisional until the shared runner/grader are merged; rows produced
# with them are development evidence only.
PROMPT_VERSION = "qa-short-fewshot-v1"
GRADING_VERSION = "alias-match-v0"
SIGNALS_VERSION = "signals-v0"

SYSTEM_PROMPT = "You are a helpful assistant."
_INSTRUCTION = (
    "Answer the following question with only the short answer (a few words), no explanation. "
    "If you do not know the answer, reply exactly \"I don't know\".\n\n"
    "Question: {question}"
)
PROMPTS = {
    # Zero-shot. The 0.5B model often restates the question as a full sentence.
    "qa-short-v0": {"template": _INSTRUCTION, "demos": []},
    # Same instruction plus three generic answer-only demos (not drawn from any evaluated dataset).
    "qa-short-fewshot-v1": {
        "template": _INSTRUCTION,
        "demos": [
            ("What is the capital of France?", "Paris"),
            ("Who wrote the novel Nineteen Eighty-Four?", "George Orwell"),
            ("In which year did the Berlin Wall fall?", "1989"),
        ],
    },
}

# Qwen's generation_config.json ships do_sample=True, repetition_penalty=1.1, top_k=20, top_p=0.8;
# generation.py replaces it so none of those leak into greedy or sampled decoding.
N_SAMPLES = 5
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 64
DECODING_CONFIG_ID = f"greedy+{N_SAMPLES}xT{TEMPERATURE}-p{TOP_P}-max{MAX_NEW_TOKENS}"
