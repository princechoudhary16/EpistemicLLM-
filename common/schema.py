"""Response-level output schema (plan.md section 9). One row per generated response.

Non-applicable fields (e.g. exposure_level/fact_id for real QA) are written as null,
not omitted, so every row has the same columns.
"""

import json

RESPONSE_FIELDS = (
    "run_id", "question_id", "fact_id", "source_dataset", "dataset_version", "split",
    "question", "reference_answer", "exposure_level",
    "model_id", "model_revision", "checkpoint_id", "method", "training_seed",
    "prompt_version", "decoding_config_id", "generation_seed", "sample_index", "is_greedy",
    "model_answer", "outcome", "grading_version", "review_status", "truncated",
    "avg_logprob", "entropy", "confidence_score", "latency_seconds",
)


def make_row(**fields):
    """Builds one response row, filling any unspecified schema field with null."""
    extra = set(fields) - set(RESPONSE_FIELDS)
    if extra:
        raise ValueError(f"not in the shared response schema: {sorted(extra)}")
    return {k: fields.get(k) for k in RESPONSE_FIELDS}


def write_jsonl(rows, path):
    """Writes standard JSON only — raises on NaN/Infinity instead of emitting them
    (Python's json module allows those non-standard literals by default, which
    other languages' JSON parsers then reject on read)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
