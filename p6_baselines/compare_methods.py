"""P6 Week 3: the common evaluator (plan.md section 2, P6: "ensure all methods are
evaluated on the same held-out examples"; Week 3 acceptance criterion: "common
evaluator accepts P5 outputs").

Takes any number of label=path response files in the shared schema and:
  1. validates every method was evaluated on the identical set of question_ids
     (refuses to compare otherwise, rather than silently comparing mismatched
     populations),
  2. computes the same behavior/knowledge/risk-coverage metrics for every method,
  3. computes a paired, cluster-bootstrapped accuracy difference against a baseline
     method (default: the first one given).

Takes response files, not method names -- so when P5 has checkpoint predictions in
this schema, they plug in with `--responses ... p5=results/raw/p1_pilot_v0_p5.jsonl`,
no code change required.

Usage:
    python -m p6_baselines.compare_methods \
        --responses base=results/raw/p1_pilot_v0_base.jsonl \
                     sft=results/raw/p1_pilot_v0_sft.jsonl \
                     idk=results/raw/p1_pilot_v0_idk.jsonl \
        --out results/final/baseline_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import (area_under_risk_coverage, behavior_metrics,
                             knowledge_scores, paired_bootstrap_diff, risk_coverage_curve)
from common.schema import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT_VERSION = "p6-eval-v1"  # frozen per plan.md section 3; bump deliberately, not silently.


def greedy_rows(rows):
    return [r for r in rows if r["is_greedy"]]


def accuracy_metric(rows):
    return behavior_metrics([r["outcome"] for r in greedy_rows(rows)])["accuracy"]


def method_summary(rows):
    greedy = greedy_rows(rows)
    bm = behavior_metrics([r["outcome"] for r in greedy])
    confidences = [r["confidence_score"] if r["confidence_score"] is not None else float("-inf") for r in greedy]
    rc = risk_coverage_curve([r["outcome"] for r in greedy], confidences)
    ks = knowledge_scores(rows)
    mean_k = sum(v["knowledge_score"] for v in ks.values()) / len(ks) if ks else None
    return {
        **bm,
        "aurc": area_under_risk_coverage(rc),
        "mean_knowledge_score": mean_k,
        "n_facts_scored": len(ks),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="+", required=True,
                     help="label=path pairs, e.g. base=results/raw/p1_pilot_v0_base.jsonl")
    ap.add_argument("--baseline-label", default=None, help="paired diffs vs this label (default: first given)")
    ap.add_argument("--out", default="results/final/baseline_comparison.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    methods = {}
    for item in args.responses:
        label, path = item.split("=", 1)
        methods[label] = read_jsonl(REPO_ROOT / path)

    # Matched-question-join check, before computing anything from the data.
    question_sets = {label: {r["question_id"] for r in greedy_rows(rows)} for label, rows in methods.items()}
    first_label = next(iter(question_sets))
    for label, qs in question_sets.items():
        if qs != question_sets[first_label]:
            raise ValueError(
                f"'{label}' and '{first_label}' were not evaluated on the same questions "
                f"({len(qs ^ question_sets[first_label])} mismatched question_ids) -- refusing to compare"
            )
    print(f"Matched-question-join check passed: all {len(methods)} methods share "
          f"{len(question_sets[first_label])} question_ids.", file=sys.stderr)

    summaries = {label: method_summary(rows) for label, rows in methods.items()}

    baseline = args.baseline_label or first_label
    diffs = {}
    for label, rows in methods.items():
        if label == baseline:
            continue
        diffs[f"{label}_vs_{baseline}"] = paired_bootstrap_diff(
            rows, methods[baseline], accuracy_metric, cluster_key="question_id", n_boot=args.n_boot,
        )

    result = {"eval_script_version": EVAL_SCRIPT_VERSION, "baseline": baseline,
              "summaries": summaries, "accuracy_diffs": diffs}

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Per-method summary (greedy accuracy/coverage/etc., all generations' knowledge score) ===")
    print(json.dumps(summaries, indent=2))
    print(f"\n=== Accuracy diffs vs '{baseline}' (95% CI, question-cluster bootstrap, n_boot={args.n_boot}) ===")
    print(json.dumps(diffs, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
