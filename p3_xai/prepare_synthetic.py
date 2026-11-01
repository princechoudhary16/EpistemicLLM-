"""Prepare P3's grouped, development-only view of the P1 synthetic evaluation set."""

import argparse
from pathlib import Path

from .utils import read_jsonl, sha256_file, validate_questions, write_json, write_jsonl


def prepare(source: Path, output: Path, manifest: Path):
    source_rows = read_jsonl(source)
    rows = []
    for row in source_rows:
        if row.get("split") != "dev":
            continue
        # Preserve P1 IDs and targets; fact_id keeps alternate templates together
        # when P3 makes its fixed grouped train/validation holdout.
        rows.append({**row, "group_id": row["fact_id"]})
    if not rows:
        raise ValueError(f"No dev rows found in {source}")
    validate_questions(rows)
    write_jsonl(output, rows)
    write_json(manifest, {
        "source_path": str(source), "source_sha256": sha256_file(source),
        "questions_path": str(output), "questions_sha256": sha256_file(output),
        "source_dataset_version": rows[0]["dataset_version"],
        "split": "dev", "question_count": len(rows),
        "fact_count": len({row["fact_id"] for row in rows}),
        "grouping": "fact_id; all templates for one fact remain in one probe partition",
        "use": "P3 probe development only; excludes P1 final rows",
    })
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/synthetic/main_v0/eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/p3/synthetic_main_v0_dev.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/p3_synthetic_main_v0.json"))
    args = parser.parse_args()
    count = prepare(args.source, args.output, args.manifest)
    print(f"Prepared {count} dev questions from {args.source}: {args.output}")


if __name__ == "__main__":
    main()
