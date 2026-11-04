"""Scores the pre-stated 0x-prompt criteria (reports/p5_aux_zero.md) from saved results; reads only, changes nothing.

For one split (dev for the pass/fail judgement, final for reporting) and each world, compares the lambda=3 runs with
the lambda=3 + aux-zero runs (same seeds):
  a  head AUROC, 1x/2x vs 0x                    >= 0.60  (baseline about 0.35)
  b  head AUROC, >=5x vs <=2x                   >= 0.80 and not more than 0.03 below the baseline lambda=3
  c  overall accuracy within 2 points of the baseline lambda=3, and accuracy at 0x <= 2%
Criterion d (does the head's abstention net score close the gap to log-prob) is reported by abstain_eval.py.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

LEVELS = ["0", "1", "2", "5", "10", "20"]


def scores_path(world, split, aux0):
    stem = {"main_v0": "p5", "main_w1": "p5_main_w1"}[world]
    return Path("results/final") / (f"{stem}_signal_scores_{split}.jsonl" if not aux0 else f"p5_{world}_aux0_signal_scores_{split}.jsonl")


def load_scores(path):
    by = collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        by[r["seed"]].append(r)
    return by


def auc(rows, pos, neg):
    p = [r["scores"]["p5_head"] for r in rows if r["exposure_level"] in pos]
    n = [r["scores"]["p5_head"] for r in rows if r["exposure_level"] in neg]
    return roc_auc_score([1] * len(p) + [0] * len(n), p + n)


def accuracy(world, split, run_tag, seed, root):
    s = json.loads((Path(root) / f"{world}_{run_tag}_seed{seed}" / f"eval_{world}_{split}" / "summary.json").read_text())
    lv = {e: s["by_exposure"][e]["accuracy"] for e in LEVELS}
    return float(np.mean(list(lv.values()))), lv["0"]


def ms(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "final"], default="dev")
    ap.add_argument("--worlds", nargs="+", default=["main_v0", "main_w1"])
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result, verdicts = {}, []
    for world in args.worlds:
        base, aux = load_scores(scores_path(world, args.split, False)), load_scores(scores_path(world, args.split, True))
        seeds = sorted(set(base) & set(aux))
        row = {"seeds": seeds}
        for name, by, tag in (("baseline_lam3", base, "lam3"), ("aux0_lam3", aux, "lam3aux0")):
            row[name] = {
                "head_auroc_1x2x_vs_0x": ms([auc(by[s], {1, 2}, {0}) for s in seeds]),
                "head_auroc_ge5_vs_le2": ms([auc(by[s], {5, 10, 20}, {0, 1, 2}) for s in seeds]),
                "head_auroc_ge5_vs_0x": ms([auc(by[s], {5, 10, 20}, {0}) for s in seeds]),
                "overall_accuracy": ms([accuracy(world, args.split, tag, s, args.root)[0] for s in seeds]),
                "accuracy_at_0x": ms([accuracy(world, args.split, tag, s, args.root)[1] for s in seeds]),
            }
        b, a = row["baseline_lam3"], row["aux0_lam3"]
        crit = {
            "a": a["head_auroc_1x2x_vs_0x"][0] >= 0.60,
            "b": a["head_auroc_ge5_vs_le2"][0] >= 0.80 and a["head_auroc_ge5_vs_le2"][0] >= b["head_auroc_ge5_vs_le2"][0] - 0.03,
            "c": abs(a["overall_accuracy"][0] - b["overall_accuracy"][0]) <= 0.02 and a["accuracy_at_0x"][0] <= 0.02,
        }
        row["criteria"] = crit
        result[world] = row
        verdicts.append(crit)
        print(f"=== {world}, {args.split}, seeds {seeds} (mean ± std)")
        print(f"{'':28}{'baseline λ=3':>20}{'λ=3 + 0× prompts':>22}")
        for k, label in (("head_auroc_1x2x_vs_0x", "head AUROC 1×/2× vs 0×"), ("head_auroc_ge5_vs_0x", "head AUROC ≥5× vs 0×"),
                         ("head_auroc_ge5_vs_le2", "head AUROC ≥5× vs ≤2×"), ("overall_accuracy", "overall accuracy"),
                         ("accuracy_at_0x", "accuracy at 0×")):
            print(f"{label:28}{b[k][0]:>13.3f} ±{b[k][1]:.3f}{a[k][0]:>15.3f} ±{a[k][1]:.3f}")
        print("   criteria:", {k: ("PASS" if v else "FAIL") for k, v in crit.items()})
    if args.split == "dev":
        allpass = all(all(v.values()) for v in verdicts)
        print(f"\nDev verdict (a, b and c must pass in every world): {'ALL PASS' if allpass else 'NOT ALL PASS'}")
        result["dev_all_pass"] = allpass
    out = Path(args.out or f"results/final/p5_aux0_criteria_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
