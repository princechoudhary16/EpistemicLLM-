"""Scores the K(q)-target criteria (reports/p5_kq_target.md) from saved results; reads only.

On the scored split S (dev: sets labelled on final facts; final: sets labelled on dev facts), per world, 3 seeds:
  A  AUROC(head predicts the student's correct answers): K-head minus matched s(e)-head >= 0.03
  B  AURC of the K-head within 0.03 of the AURC using the same student's own average log-probability
  C  reported: head-vs-teacher-K Spearman on the scored facts; the all-labelled s(e) lambda=3 baseline; accuracy
--arm p1 scores the second arm (heads trained on P1's K, main_v0 only; tags pfinal/pdev) with the same criteria against
the same matched s(e) control (efinal/edev) and writes results/final/p5_kq_criteria_<split>_p1.json.
AURC = P6's risk-coverage area (lower is better), computed with the copies of P6's functions in abstain_eval.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from abstain_eval import area_under_risk_coverage, risk_coverage_curve
from aux0_criteria import accuracy, load_scores, ms


def scores_file(world, split, tag):
    stem = {"main_v0": "p5", "main_w1": "p5_main_w1"}[world]
    if tag == "baseline":
        return Path("results/final") / f"{stem}_signal_scores_{split}.jsonl"
    return Path("results/final") / f"p5_{world}_kq-{tag}_signal_scores_{split}.jsonl"


def per_seed_metrics(rows, labels):
    outcomes = [r["outcome_p5"] for r in rows]
    head = [r["scores"]["p5_head"] for r in rows]
    logp = [r["scores"]["p5_avg_logprob"] for r in rows]
    correct = [o == "correct" for o in outcomes]
    k = [labels[r["question_id"].rsplit("-", 1)[0]] for r in rows]
    return {
        "auroc_correct": float(roc_auc_score(correct, head)),
        "aurc_head": area_under_risk_coverage(risk_coverage_curve(outcomes, head)),
        "aurc_logprob": area_under_risk_coverage(risk_coverage_curve(outcomes, logp)),
        "spearman_vs_teacher_k": float(spearmanr(head, k).correlation),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "final"], default="dev")
    ap.add_argument("--arm", choices=["own", "p1"], default="own", help="label source of the K head (own = this branch's teacher)")
    ap.add_argument("--worlds", nargs="+", default=None, help="default: both worlds (own arm), main_v0 (p1 arm)")
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prefix = "k" if args.arm == "own" else "p"
    k_tag, e_tag = ((prefix + "final", "efinal") if args.split == "dev" else (prefix + "dev", "edev"))
    worlds = args.worlds or (["main_v0", "main_w1"] if args.arm == "own" else ["main_v0"])
    if args.arm == "p1" and worlds != ["main_v0"]:
        raise SystemExit("the p1 arm exists for main_v0 only")
    result, verdicts = {}, []
    for world in worlds:
        suffix = "" if args.arm == "own" else "_p1"
        labels = json.loads((Path("results/final") / f"p5_kq_labels_{world}{suffix}.json").read_text())["labels"]
        by = {c: load_scores(scores_file(world, args.split, t)) for c, t in
              (("k", k_tag), ("e", e_tag), ("baseline", "baseline"))}
        seeds = sorted(set.intersection(*(set(v) for v in by.values())))
        acc_tags = {"k": f"lam3{k_tag}", "e": f"lam3{e_tag}", "baseline": "lam3"}
        row = {"seeds": seeds}
        per = {c: [per_seed_metrics(by[c][s], labels) for s in seeds] for c in by}
        for c in by:
            row[c] = {m: ms([p[m] for p in per[c]]) for m in per[c][0]}
            row[c]["overall_accuracy"] = ms([accuracy(world, args.split, acc_tags[c], s, args.root)[0] for s in seeds])
        a_diff = float(np.mean([pk["auroc_correct"] - pe["auroc_correct"] for pk, pe in zip(per["k"], per["e"])]))
        b_gap = float(np.mean([pk["aurc_head"] - pk["aurc_logprob"] for pk in per["k"]]))
        row["A_auroc_k_minus_e"], row["B_aurc_k_minus_own_logprob"] = a_diff, b_gap
        row["criteria"] = {"A": a_diff >= 0.03, "B": b_gap <= 0.03}
        result[world] = row
        verdicts.append(row["criteria"])
        print(f"=== {world}, arm {args.arm}, scored on {args.split} (facts the head had no label for); seeds {seeds}")
        print(f"{'':30}{'K head':>16}{'s(e) head (matched)':>22}{'s(e), all labelled':>22}")
        for key, label in (("auroc_correct", "AUROC predicts correct"), ("aurc_head", "AURC head (lower=better)"),
                           ("aurc_logprob", "AURC own log-prob"), ("spearman_vs_teacher_k", "Spearman vs teacher K"),
                           ("overall_accuracy", "overall accuracy")):
            print(f"{label:30}" + "".join(f"{row[c][key][0]:>9.3f} ±{row[c][key][1]:.3f}  " for c in ("k", "e", "baseline")))
        print(f"   A (K - s(e) AUROC = {a_diff:+.3f}, need >= +0.03): {'PASS' if row['criteria']['A'] else 'FAIL'}"
              f" | B (K AURC - own log-prob AURC = {b_gap:+.3f}, need <= 0.03): {'PASS' if row['criteria']['B'] else 'FAIL'}")
    if args.split == "dev":
        ok = all(all(v.values()) for v in verdicts)
        print(f"\nDev verdict (A and B in every world): {'ALL PASS' if ok else 'NOT ALL PASS'}")
        result["dev_all_pass"] = ok
    out = Path(args.out or f"results/final/p5_kq_criteria_{args.split}{'' if args.arm == 'own' else '_p1'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
