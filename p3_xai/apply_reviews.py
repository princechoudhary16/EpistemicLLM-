"""Apply explicit, response-bound label corrections without modifying original predictions."""

import argparse
import json
from pathlib import Path

from .utils import read_jsonl, sha256_file, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/p3/week1"))
    parser.add_argument("--reviews", type=Path, default=Path("data/p3/week1_label_reviews.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reviews = json.loads(args.reviews.read_text())
    if sha256_file(args.run_dir / "predictions.jsonl") != reviews["predictions_sha256"]:
        raise ValueError("Review file belongs to different responses; review the new run separately")
    labels = read_jsonl(args.run_dir / "labels.jsonl")
    updates = {r["question_id"]: r for r in reviews["corrections"]}
    if len(updates) != len(reviews["corrections"]) or not set(updates).issubset({r["question_id"] for r in labels}):
        raise ValueError("Duplicate or unknown review question IDs")
    for label in labels:
        if label["question_id"] in updates:
            correction = updates[label["question_id"]]
            if correction["outcome"] not in {"correct", "incorrect", "abstain"}:
                raise ValueError("Unknown reviewed outcome")
            label.update(outcome=correction["outcome"], correct=int(correction["outcome"] == "correct"),
                         review_status="assistant_reviewed_provisional", grading_version="p3-exact-match-v1+review",
                         review_note=correction["reason"])
    output = args.output or args.run_dir / "labels_reviewed.jsonl"
    if output.exists():
        raise ValueError("Reviewed label file already exists; use a new --output")
    write_jsonl(output, labels)
    print(f"Applied {len(updates)} provisional corrections: {output}")


if __name__ == "__main__":
    main()
