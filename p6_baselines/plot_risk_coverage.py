"""P6 Week 5 acceptance criterion: "risk-coverage plot" (plan.md section 8, Week 5,
P6 row). Plots selective risk vs. coverage for every method given, using the exact
same risk_coverage_curve function the master table's AURC numbers come from, so the
plot and the numbers can never silently disagree.

Usage:
    python -m p6_baselines.plot_risk_coverage \
        --responses base=results/raw/p1_main_v0_base.jsonl \
                     sft=results/raw/p1_main_v0_sft.jsonl \
                     idk=results/raw/p1_main_v0_idk.jsonl \
                     sft_matched=results/raw/p1_main_v0_sft_matched.jsonl \
                     p5_lam0=results/raw/p5_main_v0_lam0_seed0.jsonl \
                     p5_lam3=results/raw/p5_main_v0_lam3_seed0.jsonl \
        --out results/final/risk_coverage_main_v0.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import risk_coverage_curve
from common.schema import read_jsonl
from p6_baselines.compare_methods import greedy_rows, validate_methods

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--confidence-field", default="confidence_score")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    methods = {}
    for item in args.responses:
        label, path = item.split("=", 1)
        methods[label] = read_jsonl(REPO_ROOT / path)
    validate_methods(methods)  # same fairness check the master table uses

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, rows in methods.items():
        greedy = greedy_rows(rows)
        confidences = [r[args.confidence_field] if r.get(args.confidence_field) is not None else float("-inf")
                       for r in greedy]
        points = risk_coverage_curve([r["outcome"] for r in greedy], confidences)
        xs = [p["coverage"] for p in points]
        ys = [p["selective_risk"] for p in points]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=label)

    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Selective risk (error rate among answered questions)")
    ax.set_title(f"Risk-coverage curve, confidence = {args.confidence_field}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
