"""What the model is confidently wrong about, at a frozen threshold.

The pattern of each error uses the five independent samples, not the greedy answer being judged:
  stable_wrong     K=0 and the samples repeat the greedy answer (consistency >= 0.8)
  wavering_wrong   K=0 and the samples disagree with it
  knows_sometimes  K>0: some samples are right, so the greedy answer missed something within reach
Grader flags are counted because a confident "error" can be a grading artefact.
"""

import csv
from collections import Counter
from pathlib import Path

from common.grading import ALIAS_SEP
from common.metrics import CORRECT, INCORRECT

STABLE_CONSISTENCY = 0.8
PATTERNS = ("stable_wrong", "wavering_wrong", "knows_sometimes")
EXAMPLE_FIELDS = ("question_id", "exposure_level", "pattern", "probability_correct", "raw_score", "consistency",
                  "knowledge_score", "answer_n_tokens", "review_status", "question", "model_answer", "reference_answer")


def error_pattern(row):
    if (row.get("knowledge_score") or 0) > 0:
        return "knows_sometimes"
    return "stable_wrong" if (row.get("consistency") or 0) >= STABLE_CONSISTENCY else "wavering_wrong"


def confident_error_analysis(questions, calibrator, threshold):
    """Every incorrect answer kept by `threshold`, most confident first, with counts by pattern."""
    signal = calibrator.signal
    kept = [(row, calibrator.probability(row.get(signal))) for row in questions
            if row["outcome"] in (CORRECT, INCORRECT)]
    kept = [(row, probability) for row, probability in kept if probability is not None and probability >= threshold]
    errors = [(row, probability) for row, probability in kept if row["outcome"] == INCORRECT]
    errors.sort(key=lambda item: (-item[1], item[0]["question_id"]))
    examples = [{
        "question_id": row["question_id"], "exposure_level": row.get("exposure_level"),
        "pattern": error_pattern(row), "probability_correct": probability, "raw_score": row.get(signal),
        "consistency": row.get("consistency"), "knowledge_score": row.get("knowledge_score"),
        "answer_n_tokens": row.get("answer_n_tokens"), "review_status": row.get("review_status"),
        "question": row["question"], "model_answer": row["model_answer"],
        "reference_answer": row["reference_answer"].split(ALIAS_SEP)[0],
    } for row, probability in errors]
    return {
        "signal": signal, "threshold": threshold, "n_answers_kept": len(kept), "n_errors": len(errors),
        "by_pattern": {pattern: sum(e["pattern"] == pattern for e in examples) for pattern in PATTERNS},
        "by_exposure": {str(level): count for level, count in sorted(
            Counter(e["exposure_level"] for e in examples if e["exposure_level"] is not None).items())},
        "n_flagged_by_grader": sum(str(e["review_status"]).startswith("flagged") for e in examples),
        "examples": examples,
    }


def write_error_table(analysis, path):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(analysis["examples"])
