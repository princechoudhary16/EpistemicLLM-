"""Per-question labels for joint analyses (P3 probes, P4 calibration): plan.md Week 5, P1.

One row per (method, evaluation question). The file holds no reference answers, so it cannot leak a
test answer into a probe or a calibrator.

    python -m p1_exposure.analysis.question_labels --version main_v0 \
        --responses sft_matched=results/raw/p1_main_v0_sft_matched.jsonl base=results/raw/p1_main_v0_base.jsonl \
        --out p1_exposure/analysis/results/final/question_labels_main_v0.csv

Columns: method, question_id, fact_id, relation, exposure_level, split, greedy_outcome (correct, incorrect or
abstain), greedy_correct (1 or 0), k_question (fraction of this question's sampled responses that were correct),
n_sampled, fact_k (the same fraction pooled over both questions of the fact), fact_k_bucket (K=0, inconsistent
or K>=0.8). The bucket cut-offs match exposure_analysis.py.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from .exposure_analysis import bucket, check_coverage, load_eval, load_responses

COLUMNS = ["method", "question_id", "fact_id", "relation", "exposure_level", "split", "greedy_outcome",
           "greedy_correct", "k_question", "n_sampled", "fact_k", "fact_k_bucket"]


def build_rows(label, rows, eval_by_q):
    greedy = {r["question_id"]: r["outcome"] for r in rows if r["is_greedy"]}
    samp = defaultdict(list)
    for r in rows:
        if not r["is_greedy"]:
            samp[r["question_id"]].append(r["outcome"] == "correct")
    by_fact = defaultdict(list)
    for qid, v in samp.items():
        by_fact[eval_by_q[qid]["fact_id"]].extend(v)
    out = []
    for qid, q in eval_by_q.items():          # eval order, so every method's file lines up
        v = samp.get(qid, [])
        fv = by_fact.get(q["fact_id"], [])
        fact_k = sum(fv) / len(fv) if fv else None
        out.append({
            "method": label, "question_id": qid, "fact_id": q["fact_id"], "relation": q["relation"],
            "exposure_level": int(q["exposure_level"]), "split": q["split"],
            "greedy_outcome": greedy[qid], "greedy_correct": int(greedy[qid] == "correct"),
            "k_question": "" if not v else round(sum(v) / len(v), 6), "n_sampled": len(v),
            "fact_k": "" if fact_k is None else round(fact_k, 6),
            "fact_k_bucket": "" if fact_k is None else bucket(fact_k)})
    return out


def run(responses, data_dir, version, out_path):
    eval_by_q = load_eval(data_dir, version)
    all_rows = []
    for label, path in responses.items():
        rows = load_responses(path, eval_by_q)
        check_coverage(label, rows, eval_by_q)
        all_rows += build_rows(label, rows, eval_by_q)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)
    return all_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--responses", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--version", default="main_v0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    responses = {}
    for item in a.responses:
        if "=" not in item:
            sys.exit(f"--responses expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        responses[label] = path
    rows = run(responses, a.data_dir, a.version, a.out)
    print(f"wrote {len(rows)} rows for {len(responses)} methods to {a.out}")


if __name__ == "__main__":
    main()
