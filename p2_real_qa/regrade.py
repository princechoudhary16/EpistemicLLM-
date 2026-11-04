"""Re-grade an existing response file with the CURRENT shared grader, without regenerating anything.
Use this whenever common/grading.py changes (bump GRADING_VERSION there first).

    python -m p2_real_qa.regrade --responses results/raw/p2_pilot_base.jsonl
        -> results/raw/p2_pilot_base__<grading_version>.jsonl      (the input file is never modified)
        -> results/audit/regrade_disagreements_<stem>.jsonl         (rows whose outcome changed -> audit these first)

Use --questions when the QUESTION FILE changed (e.g. new aliases) but the model answers did not: each row's
reference_answer is replaced by the one in the question file before grading, so nothing has to be regenerated.

    python -m p2_real_qa.regrade --responses results/raw/p2_triviaqa_v0_pilot_base.jsonl \
        --questions data/real_qa/triviaqa_v0/pilot.jsonl
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
    ap.add_argument("--questions", help="question JSONL whose reference_answer replaces the one in each response row")
    a = ap.parse_args()
    src = Path(a.responses)
    old = schema.read_jsonl(src)
    new = copy.deepcopy(old)
    if a.questions:
        refs = {q["question_id"]: q["reference_answer"] for q in schema.read_jsonl(a.questions)}
        missing = {r["question_id"] for r in new} - set(refs)
        if missing:
            raise SystemExit(f"{len(missing)} response question_ids are not in {a.questions} (first: {sorted(missing)[:3]})")
        n_ref = sum(r["reference_answer"] != refs[r["question_id"]] for r in new)
        for r in new:
            r["reference_answer"] = refs[r["question_id"]]
        print(f"reference_answer refreshed from {a.questions}: {n_ref} of {len(new)} rows changed")
    new = grading.regrade_rows(new)

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
