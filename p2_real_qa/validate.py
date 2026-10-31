"""Integrity checks for the P2 question files (same style as p1_exposure/validate.py).

    python -m p2_real_qa.validate data/real_qa/simpleqa_v0
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REQUIRED = ("question_id", "source_dataset", "dataset_version", "split", "question", "reference_answer", "group_id")


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def _norm(q):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", q.lower())).strip()


def run_checks(records_by_split: dict, expected_sizes: dict | None = None):
    out = []
    allrows = [r for rows in records_by_split.values() for r in rows]

    missing = [(r.get("question_id"), f) for r in allrows for f in REQUIRED if r.get(f) in (None, "")]
    out.append(_check("required_fields_present", not missing, f"{len(missing)} missing/empty (first: {missing[:3]})"))

    dup_ids = [k for k, c in Counter(r["question_id"] for r in allrows).items() if c > 1]
    out.append(_check("question_ids_unique_across_splits", not dup_ids, f"duplicates: {dup_ids[:5]}"))

    wrong_split = [r["question_id"] for name, rows in records_by_split.items() for r in rows if r["split"] != name]
    out.append(_check("split_field_matches_file", not wrong_split, f"mismatches: {wrong_split[:5]}"))

    names = list(records_by_split)
    shared = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = {r["group_id"] for r in records_by_split[a]} & {r["group_id"] for r in records_by_split[b]}
            if s:
                shared[f"{a}/{b}"] = len(s)
    out.append(_check("no_group_shared_across_splits", not shared, f"shared groups: {shared}"))

    seen = {}
    for name, rows in records_by_split.items():
        for r in rows:
            seen.setdefault(_norm(r["question"]), set()).add(name)
    dup_q = [q for q, sp in seen.items() if len(sp) > 1]
    out.append(_check("no_question_text_shared_across_splits", not dup_q, f"{len(dup_q)} question strings in >1 split"))

    if expected_sizes:
        bad = {k: (len(records_by_split.get(k, [])), n) for k, n in expected_sizes.items()
               if len(records_by_split.get(k, [])) != n}
        out.append(_check("split_sizes_as_expected", not bad, f"(actual, expected) mismatches: {bad}"))
    return out


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None):
    d = Path((argv or sys.argv[1:])[0])
    checks = run_checks({p.stem: _load(p) for p in sorted(d.glob("*.jsonl"))})
    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    sys.exit(0 if all(c["ok"] for c in checks) else 1)


if __name__ == "__main__":
    main()
