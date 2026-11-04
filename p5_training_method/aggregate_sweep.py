"""Collects the dev-split summaries from the lambda sweep into one table (mean +- std over seeds).

Selection rule (written down before the multi-seed results were seen; it was proposed after
one seed, so treat it as a stated rule, not a pre-registration): among the lambdas whose mean
dev accuracy is within --max-acc-drop points of lambda=0, pick the one with the highest mean
AUROC(head predicts correct). Only dev summaries are read; the final split is never opened.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVELS = ["0", "1", "2", "5", "10", "20"]


def load(root, dset, split="dev"):
    runs = defaultdict(list)
    for path in sorted(Path(root).glob(f"{dset}_lam*_seed*/eval_{dset}_{split}/summary.json")):
        m = re.search(r"_lam(\d+(?:\.\d+)?)_seed(\d+)/", str(path))
        if m is None:      # other experiments share the folder (lam3aux0, lam3kfinal, ...): not part of the lambda sweep
            continue
        s = json.loads(path.read_text())
        levels = s["by_exposure"]
        acc = {e: levels[e]["accuracy"] for e in LEVELS}
        runs[float(m.group(1))].append({
            "seed": int(m.group(2)),
            "acc_overall": float(np.mean([acc[e] for e in LEVELS])),  # equal n per level
            "acc": acc,
            "spearman": s["spearman_head_vs_exposure"],
            "auroc_ge5_vs_le2": s["auroc_head_exposure_ge5_vs_le2"],
            "auroc_correct": s["auroc_head_predicts_correct"],
        })
    return runs


def ms(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--dataset", default="main_v0")
    ap.add_argument("--max-acc-drop", type=float, default=0.05)
    ap.add_argument("--split", choices=["dev", "final"], default="dev",
                    help="final just tabulates the locked lambdas; the selection rule applies to dev only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/final/p5_lambda_sweep_{args.split}.json"

    runs = load(args.root, args.dataset, args.split)
    if 0.0 not in runs:
        raise SystemExit("No lambda=0 runs found; the selection rule needs them.")

    table = {}
    print(f"{'lambda':>7} {'n':>2} | {'overall acc':>13} | " + " ".join(f"{e+'x':>6}" for e in LEVELS)
          + f" | {'spearman':>13} | {'AUROC >=5v<=2':>14} | {'AUROC correct':>14}")
    for lam in sorted(runs):
        rs = runs[lam]
        row = {
            "n_seeds": len(rs),
            "acc_overall": ms([r["acc_overall"] for r in rs]),
            "acc_by_level": {e: ms([r["acc"][e] for r in rs]) for e in LEVELS},
            "spearman": ms([r["spearman"] for r in rs]),
            "auroc_ge5_vs_le2": ms([r["auroc_ge5_vs_le2"] for r in rs]),
            "auroc_correct": ms([r["auroc_correct"] for r in rs]),
        }
        table[str(lam)] = row
        pm = lambda t: f"{t[0]:.3f}+-{t[1]:.3f}"
        print(f"{lam:>7g} {len(rs):>2} | {pm(row['acc_overall']):>13} | "
              + " ".join(f"{row['acc_by_level'][e][0]:>6.2f}" for e in LEVELS)
              + f" | {pm(row['spearman']):>13} | {pm(row['auroc_ge5_vs_le2']):>14} | {pm(row['auroc_correct']):>14}")

    base = table["0.0"]["acc_overall"][0]
    choice = None
    if args.split == "dev":
        eligible = {lam: r for lam, r in table.items() if float(lam) > 0 and r["acc_overall"][0] >= base - args.max_acc_drop}
        choice = max(eligible, key=lambda k: eligible[k]["auroc_correct"][0]) if eligible else None
        print(f"\nlambda=0 mean accuracy {base:.3f}; allowed drop {args.max_acc_drop:.2f}")
        print(f"selected lambda (dev, by the rule): {choice}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"table": table, "selected_lambda": choice, "max_acc_drop": args.max_acc_drop}, indent=2))


if __name__ == "__main__":
    main()
