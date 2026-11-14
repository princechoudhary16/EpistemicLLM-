"""Scores the name-recognition control (reports/p5_aux_zero_novel.md) from saved results; reads only.

Scored on split S (the 0x facts of S), per world, 3 seeds, four conditions:
  baseline     lambda=3, no 0x prompts                        (the head never saw any 0x example)
  seen_all     lambda=3, all 400 0x prompts                   (earlier experiment; saw the scored names, 2x the data)
  seen_half    lambda=3, the 200 0x prompts of S's own facts  (saw the scored names)
  novel_half   lambda=3, the 200 0x prompts of the OTHER split's facts (never saw the scored names)
seen_half and novel_half have identical amounts of 0x supervision and differ only in name novelty.
Criterion A (pass/fail, each world): novel_half head AUROC for 1x/2x vs 0x >= 0.60.
Retained gain (descriptive): (novel_half - baseline) / (seen_half - baseline); >= 0.5 -> "mostly generalises",
< 0.5 -> "mostly names"; if seen_half itself is below 0.60 the reading is "inconclusive (not enough 0x data)".
Accuracy cost is reported, not judged.
"""
import argparse
import json
from pathlib import Path

from aux0_criteria import accuracy, auc, load_scores, ms


def condition_files(world, split):
    stem = {"main_v0": "p5", "main_w1": "p5_main_w1"}[world]
    other = "final" if split == "dev" else "dev"
    f = lambda name: Path("results/final") / name
    return {
        "baseline": (f(f"{stem}_signal_scores_{split}.jsonl"), "lam3"),
        "seen_all": (f(f"p5_{world}_aux0_signal_scores_{split}.jsonl"), "lam3aux0"),
        "seen_half": (f(f"p5_{world}_aux0half-{split}_signal_scores_{split}.jsonl"), f"lam3half{split}"),
        "novel_half": (f(f"p5_{world}_aux0half-{other}_signal_scores_{split}.jsonl"), f"lam3half{other}"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "final"], default="dev")
    ap.add_argument("--worlds", nargs="+", default=["main_v0", "main_w1"])
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result, passes = {}, []
    for world in args.worlds:
        files = condition_files(world, args.split)
        by = {c: load_scores(p) for c, (p, _) in files.items()}
        seeds = sorted(set.intersection(*(set(v) for v in by.values())))
        row = {"seeds": seeds}
        for c, (_, tag) in files.items():
            row[c] = {
                "auroc_1x2x_vs_0x": ms([auc(by[c][s], {1, 2}, {0}) for s in seeds]),
                "auroc_ge5_vs_0x": ms([auc(by[c][s], {5, 10, 20}, {0}) for s in seeds]),
                "auroc_ge5_vs_le2": ms([auc(by[c][s], {5, 10, 20}, {0, 1, 2}) for s in seeds]),
                "overall_accuracy": ms([accuracy(world, args.split, tag, s, args.root)[0] for s in seeds]),
                "accuracy_at_0x": ms([accuracy(world, args.split, tag, s, args.root)[1] for s in seeds]),
            }
        b, seen, nov = (row[c]["auroc_1x2x_vs_0x"][0] for c in ("baseline", "seen_half", "novel_half"))
        retained = (nov - b) / (seen - b) if seen != b else float("nan")
        row["criterion_A_pass"] = nov >= 0.60
        row["retained_gain"] = retained
        row["reading"] = ("inconclusive (not enough 0x data)" if seen < 0.60
                          else "mostly generalises" if retained >= 0.5 else "mostly names")
        result[world] = row
        passes.append(row["criterion_A_pass"])
        print(f"=== {world}, {args.split}: 0× facts scored here; seeds {seeds}")
        print(f"{'':24}{'baseline':>15}{'seen (400)':>15}{'seen (200)':>15}{'NOVEL (200)':>15}")
        for k, label in (("auroc_1x2x_vs_0x", "AUROC 1×/2× vs 0×"), ("auroc_ge5_vs_0x", "AUROC ≥5× vs 0×"),
                         ("auroc_ge5_vs_le2", "AUROC ≥5× vs ≤2×"), ("overall_accuracy", "overall accuracy"),
                         ("accuracy_at_0x", "accuracy at 0×")):
            print(f"{label:24}" + "".join(f"{row[c][k][0]:>8.3f} ±{row[c][k][1]:.3f}" for c in files))
        print(f"   criterion A (novel ≥ 0.60): {'PASS' if row['criterion_A_pass'] else 'FAIL'} | "
              f"retained gain {retained:.2f} -> {row['reading']}")
    if args.split == "dev":
        print(f"\nDev verdict (criterion A in every world): {'ALL PASS' if all(passes) else 'NOT ALL PASS'}")
        result["dev_all_pass"] = all(passes)
    out = Path(args.out or f"results/final/p5_aux0nov_criteria_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
