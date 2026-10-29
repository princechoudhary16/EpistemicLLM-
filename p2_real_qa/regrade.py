"""Re-grade an existing response file with the CURRENT shared grader, without regenerating anything.
Use this whenever common/grading.py changes (bump GRADING_VERSION there first).

    python -m p2_real_qa.regrade --responses results/raw/p2_pilot_base.jsonl
        -> results/raw/p2_pilot_base__<grading_version>.jsonl      (the input file is never modified)
        -> results/audit/regrade_disagreements_<stem>.jsonl         (rows whose outcome changed -> audit these first)
"""
import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grading, schema  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    a = ap.parse_args()
    src = Path(a.responses)
    old = schema.read_jsonl(src)
    new = grading.regrade_rows(copy.deepcopy(old))

    print(f"{old[0]['grading_version']}  ->  {grading.GRADING_VERSION}")
    pairs = Counter((o["outcome"], n["outcome"]) for o, n in zip(old, new))
    for (o, n), c in sorted(pairs.items()):
        print(f"  {o:10s} -> {n:10s} {c:5d}{'   <-- changed' if o != n else ''}")

    out = src.with_name(f"{src.stem}__{grading.GRADING_VERSION}.jsonl")
    schema.write_jsonl(new, out)
    changed = [dict(n, previous_outcome=o["outcome"]) for o, n in zip(old, new) if o["outcome"] != n["outcome"]]
    d = REPO_ROOT / "results" / "audit"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"regrade_disagreements_{src.stem}.jsonl", "w", encoding="utf-8") as f:
        import json
        for r in changed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out} and {len(changed)} changed rows for audit")


if __name__ == "__main__":
    main()
