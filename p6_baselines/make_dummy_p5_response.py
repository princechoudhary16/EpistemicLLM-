"""Proves the Week 3 acceptance criterion 'common evaluator accepts P5 outputs' is
actually true, not just true-by-design, without needing to wait for P5 to produce a
real checkpoint.

Takes an existing response file (same question_ids, same schema) and relabels it as
a stand-in fourth method ("p5_dummy") -- the content is meaningless as a research
result (it's literally someone else's answers wearing a different label), but the
file is schema-valid and shares question_ids with the other methods, which is exactly
what compare_methods.py needs to accept and compare a fourth method with zero code
changes. This file's output must never be read as an actual P5 result.

Usage:
    python -m p6_baselines.make_dummy_p5_response \
        --source results/raw/p1_main_v0_base.jsonl \
        --out results/raw/p1_main_v0_p5dummy.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.schema import read_jsonl, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(REPO_ROOT / args.source)
    for r in rows:
        r["method"] = "p5_dummy"
        r["checkpoint_id"] = "p5_dummy_v0"
        r["run_id"] = r["run_id"] + "_p5dummy_relabel"

    out_path = REPO_ROOT / args.out
    write_jsonl(rows, out_path)
    print(f"Wrote {len(rows)} relabeled rows to {out_path} -- TEST FIXTURE ONLY, not a real P5 result.")


if __name__ == "__main__":
    main()
