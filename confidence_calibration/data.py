"""JSONL I/O, question loading, and the TriviaQA development sample.

Why TriviaQA for development: calibration needs both correct and incorrect answers, and a 0.5B
model gets almost nothing right on SimpleQA. TriviaQA has alias lists (fairer grading) and is one of
the datasets GeoDe evaluates on. Final claims use the shared SimpleQA splits and synthetic outputs.
"""

import json
from pathlib import Path

import numpy as np

REQUIRED_QUESTION_FIELDS = ("question_id", "question", "reference_answer")
# Copied from the question file onto every response row; null when absent (e.g. real-QA exposure).
PASSTHROUGH_FIELDS = ("fact_id", "source_dataset", "dataset_version", "split", "exposure_level")

TRIVIAQA_REPO = "mandarjoshi/trivia_qa"
TRIVIAQA_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
TRIVIAQA_FILE = "rc.nocontext/validation-00000-of-00001.parquet"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path, limit=None):
    """Load any question JSONL with question_id/question/reference_answer (+ optional aliases).

    Works for the synthetic eval files, the shared real-QA files and the TriviaQA sample. Without
    `aliases`, the reference answer is the only accepted string.
    """
    rows = read_jsonl(path)
    seen = set()
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED_QUESTION_FIELDS if k not in row]
        if missing:
            raise ValueError(f"{path}: row {i} missing {missing}")
        if row["question_id"] in seen:
            raise ValueError(f"{path}: duplicate question_id {row['question_id']}")
        seen.add(row["question_id"])
        row["aliases"] = list(dict.fromkeys([row["reference_answer"], *(row.get("aliases") or [])]))
    return rows[:limit] if limit else rows


def build_triviaqa_sample(n=300, seed=0):
    """Seeded sample of TriviaQA rc.nocontext validation, drawn before any model output is seen.

    Rows are assigned to calibration splits cal_train / cal_val / cal_test (50/25/25) so fitting,
    selection and evaluation never share questions.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TRIVIAQA_REPO, TRIVIAQA_FILE, repo_type="dataset", revision=TRIVIAQA_REVISION)
    df = pd.read_parquet(path).sort_values("question_id").reset_index(drop=True)
    picked = df.iloc[np.random.default_rng(seed).permutation(len(df))[:n]]
    splits = ["cal_train"] * (n // 2) + ["cal_val"] * (n // 4)
    splits += ["cal_test"] * (n - len(splits))
    version = f"triviaqa-rc.nocontext-validation@{TRIVIAQA_REVISION[:8]}-sample-n{n}-seed{seed}"
    rows = []
    for (_, r), split in zip(picked.iterrows(), splits):
        answer = r["answer"]
        rows.append({
            "question_id": f"triviaqa-{r['question_id']}",
            "fact_id": None,
            "source_dataset": "triviaqa",
            "dataset_version": version,
            "split": split,
            "exposure_level": None,
            "question": r["question"],
            "reference_answer": answer["value"],
            "aliases": sorted(set(answer["aliases"].tolist()) | {answer["value"]}),
        })
    return rows
