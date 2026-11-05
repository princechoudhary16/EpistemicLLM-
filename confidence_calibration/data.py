"""JSONL I/O, loading of shared response files, and the excluded TriviaQA development sample.

Final numbers come from the shared response files (data/real_qa/triviaqa_v0 and the synthetic worlds).
The n=300 sample only remains because p2_real_qa/data.py keeps its questions out of every shared split.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common.grading import GRADING_VERSION, regrade_rows

from . import config

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RESPONSE_FIELDS = ("question_id", "question", "reference_answer", "split", "is_greedy",
                            "sample_index", "model_answer", "outcome")
# A file set must agree on all of these, or it is not one model's behaviour under one protocol.
PROVENANCE_FIELDS = ("model_id", "model_revision", "checkpoint_id", "method", "training_seed", "source_dataset",
                     "dataset_version", "prompt_version", "decoding_config_id")

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


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path):
    """Repo-relative when possible, so committed artifacts carry no machine-specific paths."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else resolved.name


def load_responses(paths):
    """Read response files from any producer and bring them under the shared protocol.

    Rejects rows from another prompt or decoding protocol and files that disagree on provenance. Every
    row is regraded with the shared grader, whatever version it arrived with, so one definition of
    correct backs every number. Returns (rows, inputs); `inputs` records each file's hash and how many
    outcomes the regrade changed.
    """
    rows, inputs = [], []
    for path in paths:
        file_rows = read_jsonl(path)
        if not file_rows:
            raise ValueError(f"{path}: response file is empty")
        for field in REQUIRED_RESPONSE_FIELDS:
            if any(field not in row for row in file_rows):
                raise ValueError(f"{path}: every response must contain {field!r}")
        before = [row["outcome"] for row in file_rows]
        arrived_with = Counter(str(row.get("grading_version")) for row in file_rows)
        regrade_rows(file_rows)
        inputs.append({"path": display_path(path), "sha256": sha256_file(path), "rows": len(file_rows),
                       "run_ids": sorted({str(row.get("run_id")) for row in file_rows}),
                       "arrived_grading_versions": dict(arrived_with), "grading_version": GRADING_VERSION,
                       "outcomes_changed_by_regrade": sum(b != r["outcome"] for b, r in zip(before, file_rows))})
        rows.extend(file_rows)
    for field, expected in (("prompt_version", config.PROMPT_VERSION),
                            ("decoding_config_id", config.DECODING_CONFIG_ID)):
        found = {row.get(field) for row in rows}
        if found != {expected}:
            raise ValueError(f"responses use {field} {sorted(found, key=str)}; the shared protocol is {expected!r}")
    for field in PROVENANCE_FIELDS:
        found = {row.get(field) for row in rows}
        if len(found) > 1:
            raise ValueError(f"input mixes {field} values {sorted(found, key=str)}; use one model at a time")
    return rows, inputs


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
