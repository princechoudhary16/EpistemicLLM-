"""Select a development-only SimpleQA pilot before observing model responses."""

import argparse
import csv
import hashlib
import random
from pathlib import Path

from .utils import normalize_answer, sha256_file, validate_questions, write_json, write_jsonl

DATA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(".cache/p3/simple_qa_test_set.csv"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/p3/week1_questions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/p3_week1_selection.json"))
    args = parser.parse_args()
    source_hash = sha256_file(args.csv)
    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        source = list(csv.DictReader(stream))
    if not source or not {"problem", "answer"}.issubset(source[0]):
        raise ValueError("Expected SimpleQA CSV with problem and answer columns")
    # Exclude exact normalized duplicate prompts, before random selection.
    unique, seen = [], set()
    for index, row in enumerate(source):
        key = normalize_answer(row["problem"])
        if key not in seen:
            seen.add(key)
            unique.append((index, row, key))
    if not 0 < args.count <= len(unique):
        raise ValueError("count must be positive and no greater than the unique question count")
    selected = random.Random(args.seed).sample(unique, args.count)
    rows = []
    for index, row, key in selected:
        rows.append({
            "question_id": f"simpleqa-{index:05d}",
            "group_id": hashlib.sha256(key.encode()).hexdigest()[:16],
            "source_row_index": index,
            "source_dataset": "SimpleQA",
            "dataset_version": source_hash,
            "split": "development",
            "question": row["problem"],
            "reference_answer": row["answer"],
        })
    validate_questions(rows)
    write_jsonl(args.output, rows)
    write_json(args.manifest, {
        "source_url": DATA_URL, "source_sha256": source_hash,
        "source_rows": len(source), "unique_questions": len(unique),
        "seed": args.seed, "selected_count": len(rows),
        "question_ids": [r["question_id"] for r in rows],
        "questions_sha256": sha256_file(args.output),
        "selection": "Uniform fixed-seed sample of unique normalized questions, before model inference",
        "use": "P3 development-only pilot; reserve these IDs from the team's eventual final test",
        "grouping_limit": "Exact normalized duplicates grouped; related/paraphrased facts require P2 review",
    })
    print(f"Prepared {len(rows)} development questions: {args.output}")


if __name__ == "__main__":
    main()
