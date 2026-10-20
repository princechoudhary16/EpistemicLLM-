"""Abstain-when-unsure test: does a confidence signal actually reduce hallucination?

Each policy = (which model's answers, which confidence score). An answer is kept if its score >= a
threshold, otherwise it becomes an abstention (P6's apply_threshold rule). The threshold is chosen
on the DEV scores of each seed by maximising  accuracy - penalty * hallucination_rate  (penalty fixed
in advance, default 1), then applied unchanged to that seed's FINAL scores. Nothing is tuned on final.

Reads the per-question scores written by probe_controls.py (dev and final).
The metric definitions below are copied from P6's common/metrics.py (behavior_metrics,
apply_threshold, risk_coverage_curve, area_under_risk_coverage) so the numbers are comparable.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORRECT, INCORRECT, ABSTAIN = "correct", "incorrect", "abstain"
LEVELS = [0, 1, 2, 5, 10, 20]

POLICIES = {
    "sft_logprob": ("outcome_sft", "sft_avg_logprob"),
    "sft_probe": ("outcome_sft", "probe_on_sft"),
    "p5_logprob": ("outcome_p5", "p5_avg_logprob"),
    "p5_head": ("outcome_p5", "p5_head"),
    "p5_probe": ("outcome_p5", "probe_on_p5_model"),
}
REFERENCES = {"sft_no_abstain": "outcome_sft", "p5_no_abstain": "outcome_p5"}


def _div(a, b):
    return a / b if b else None


def behavior_metrics(outcomes):
    k = Counter(outcomes)
    c, w, a = k[CORRECT], k[INCORRECT], k[ABSTAIN]
    n = c + w + a
    return {
        "accuracy": _div(c, n), "hallucination_rate": _div(w, n), "abstention_rate": _div(a, n),
        "coverage": _div(c + w, n), "selective_accuracy": _div(c, c + w),
    }


def apply_threshold(outcomes, scores, threshold):
    return [ABSTAIN if o in (CORRECT, INCORRECT) and s < threshold else o for o, s in zip(outcomes, scores)]


def risk_coverage_curve(outcomes, scores):
    n = len(outcomes)
    attempted = sorted(((s, o) for o, s in zip(outcomes, scores) if o in (CORRECT, INCORRECT)), key=lambda t: -t[0])
    points, kc, kw, i = [], 0, 0, 0
    while i < len(attempted):
        thr = attempted[i][0]
        while i < len(attempted) and attempted[i][0] == thr:
            kc, kw = (kc + 1, kw) if attempted[i][1] == CORRECT else (kc, kw + 1)
            i += 1
        points.append({"coverage": (kc + kw) / n, "selective_risk": kw / (kc + kw)})
    return points[::-1]


def area_under_risk_coverage(points):
    if not points:
        return None
    pts = sorted((p["coverage"], p["selective_risk"]) for p in points)
    area = pts[0][0] * pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def choose_threshold(outcomes, scores, penalty):
    """Threshold maximising accuracy - penalty*hallucination on the given (dev) rows. Ties -> more coverage."""
    best_thr, best_val = None, -np.inf
    for thr in sorted(set(scores)) + [np.inf]:
        m = behavior_metrics(apply_threshold(outcomes, scores, thr))
        val = m["accuracy"] - penalty * m["hallucination_rate"]
        if val > best_val + 1e-12:
            best_thr, best_val = thr, val
    return best_thr


def load(path):
    by_seed = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        by_seed[r["seed"]].append(r)
    return by_seed


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")) if vals else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/final/p5_signal_scores_dev.jsonl")
    ap.add_argument("--final", default="results/final/p5_signal_scores_final.jsonl")
    ap.add_argument("--penalty", type=float, default=1.0)
    ap.add_argument("--out", default="results/final/p5_abstain_eval.json")
    args = ap.parse_args()

    dev, final = load(args.dev), load(args.final)
    seeds = sorted(set(dev) & set(final))
    results = defaultdict(lambda: defaultdict(list))
    by_level = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        frows = final[seed]
        exposure = np.array([r["exposure_level"] for r in frows])
        for name, okey in REFERENCES.items():
            outs = [r[okey] for r in frows]
            for k, v in behavior_metrics(outs).items():
                results[name][k].append(v)
        for name, (okey, skey) in POLICIES.items():
            d_out = [r[okey] for r in dev[seed]]
            d_sc = [r["scores"][skey] for r in dev[seed]]
            thr = choose_threshold(d_out, d_sc, args.penalty)
            f_out = [r[okey] for r in frows]
            f_sc = [r["scores"][skey] for r in frows]
            kept = apply_threshold(f_out, f_sc, thr)
            m = behavior_metrics(kept)
            m["net"] = m["accuracy"] - args.penalty * m["hallucination_rate"]
            m["aurc"] = area_under_risk_coverage(risk_coverage_curve(f_out, f_sc))
            for k, v in m.items():
                results[name][k].append(v)
            for e in LEVELS:
                sel = [o for o, x in zip(kept, exposure) if x == e]
                by_level[name][e].append(sum(o == ABSTAIN for o in sel) / len(sel))
        for name, okey in REFERENCES.items():
            outs = [r[okey] for r in frows]
            m = behavior_metrics(outs)
            results[name]["net"].append(m["accuracy"] - args.penalty * m["hallucination_rate"])

    cols = ["accuracy", "hallucination_rate", "abstention_rate", "coverage", "selective_accuracy", "net", "aurc"]
    print(f"final split, thresholds from dev, seeds={seeds}, penalty={args.penalty}")
    print(f"{'policy':<16}" + "".join(f"{c[:11]:>14}" for c in cols))
    table = {}
    for name in list(REFERENCES) + list(POLICIES):
        table[name] = {c: mean_sd(results[name][c]) for c in cols if results[name][c]}
        cells = []
        for c in cols:
            m, s = table[name].get(c, (None, None))
            cells.append(f"{'-':>14}" if m is None else f"{m:>8.3f}+-{s:.2f}")
        print(f"{name:<16}" + "".join(cells))

    print("\nabstention rate by exposure level (higher at 0-2x, near 0 at 20x is what we want)")
    print(f"{'policy':<16}" + "".join(f"{str(e) + 'x':>8}" for e in LEVELS))
    lvl = {}
    for name in POLICIES:
        lvl[name] = {e: float(np.mean(by_level[name][e])) for e in LEVELS}
        print(f"{name:<16}" + "".join(f"{lvl[name][e]:>8.2f}" for e in LEVELS))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"table": table, "abstention_by_level": lvl, "penalty": args.penalty, "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
