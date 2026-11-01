"""Small shared utilities; no external services or model calls."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from common import grading


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def normalize_answer(text):
    # Preserve internal punctuation, signs, and decimal points: 1.5 must not equal 15.
    text = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    return re.sub(r"\s+", " ", text).strip(' \t\r\n.\"“”')


def provisional_grade(answer, reference, question=""):
    """Use the project's shared grader for every P3 correctness label."""
    outcome, grading_version, review_status = grading.grade_fields(answer, reference, question)
    correct = {"correct": 1, "incorrect": 0, "abstain": 0, "invalid": None}[outcome]
    return {"outcome": outcome, "correct": correct, "review_status": review_status,
            "grading_version": grading_version}


def validate_questions(rows):
    if not rows:
        raise ValueError("Question file is empty")
    ids = [r["question_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")
    for row in rows:
        if not row.get("question", "").strip() or not row.get("group_id"):
            raise ValueError("Every question needs nonempty question and group_id fields")
        if row.get("split") not in {"development", "dev", "train_probe", "validation"}:
            raise ValueError("P3 extraction accepts development, train_probe, and validation rows only")
    # Identical normalized prompts must share a group even if they have different IDs.
    groups = {}
    for row in rows:
        key = normalize_answer(row["question"])
        if key in groups and groups[key] != row["group_id"]:
            raise ValueError("Duplicate question text has inconsistent group IDs")
        groups[key] = row["group_id"]
