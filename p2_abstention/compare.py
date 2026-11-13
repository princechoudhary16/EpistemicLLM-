"""Compare every abstention arm over seeds on validation: means, a log-prob filter baseline at matched coverage,
pooled paired bootstrap intervals, and a breakdown by the base model's K(q) bucket.

    python -m p2_abstention.compare --out results/final/abstention/comparison_own.json

Pieces:
1. mean ± sd over seeds of F1_rel and the shared metrics, plus N1/N2/N3 (known questions: right / wrong / refused).
2. "logprob filter": the UNTRAINED base model answers only its most confident questions (highest avg_logprob), the
   same number as the arm answered in that seed; the rest count as "I don't know". No labels are used.
3. F1_rel differences averaged over seeds, with a 95% interval from resampling questions (same draws for both sides).
4. By K bucket (from the base model's 5 validation samples): unknown K=0, partial 0<K<0.8, known K>=0.8;
   for each arm, the share of those questions answered right / wrong / refused.
"""
import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import behavior_metrics  # noqa: E402
from p2_abstention.evaluate import confusion, f1_scores  # noqa: E402

BASE = "results/raw/p2_triviaqa_v0_validation_base.jsonl"
PREFIX = "results/raw/p2_triviaqa_v0_validation_abstention"
ARMS = ("own_rtuning", "own_random", "own_geode", "own_kfilter", "own_kthresh", "own_ksoft")
PAIRS = (("own_kfilter", "own_geode"), ("own_ksoft", "own_geode"), ("own_kthresh", "own_geode"),
         ("own_ksoft", "own_kfilter"), ("own_kfilter", "own_random"), ("own_ksoft", "own_random"),
         ("own_geode", "own_random"))
SEEDS = (0, 1, 2)
COLS = ("F1_rel", "accuracy", "hallucination_rate", "abstention_rate", "coverage", "selective_accuracy")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def greedy(rows):
    return {r["question_id"]: r["outcome"] for r in rows if r["is_greedy"]}


def k_values(rows):
    k = defaultdict(list)
    for r in rows:
        if not r["is_greedy"] and r["outcome"] != "invalid":
            k[r["question_id"]].append(r["outcome"] == "correct")
    return {q: sum(v) / len(v) for q, v in k.items() if v}


def bucket(k):
    return "unknown" if k == 0 else "known" if k >= 0.8 else "partial"


def logprob_filter(base_rows, n):
    """The base model keeps its n most confident non-abstaining answers; everything else becomes an abstention."""
    g = {r["question_id"]: r for r in base_rows if r["is_greedy"]}
    ranked = sorted((q for q, r in g.items() if r["outcome"] != "abstain" and r.get("avg_logprob") is not None),
                    key=lambda q: -g[q]["avg_logprob"])
    keep = set(ranked[:n])
    return {q: (r["outcome"] if q in keep else "abstain") for q, r in g.items()}


def scores(base, run, qids):
    n = confusion(base, run, qids)
    return dict(**n, **f1_scores(n), **behavior_metrics([run[q] for q in qids]))


def answered(run, qids):
    return sum(run[q] != "abstain" for q in qids)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out")
    ap.add_argument("--split", default="validation", help="validation | locked_test")
    a = ap.parse_args(argv)

    base_rows = read(f"results/raw/p2_triviaqa_v0_{a.split}_base.jsonl")
    base = greedy(base_rows)
    qids = sorted(base)
    runs = {}
    for arm in a.arms:
        paths = [Path(f"results/raw/p2_triviaqa_v0_{a.split}_abstention_{arm}_seed{s}.jsonl") for s in SEEDS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f"skipping {arm}: missing {missing}", file=sys.stderr)
            continue
        runs[arm] = [greedy(read(p)) for p in paths]
    # the matched log-prob filter for each arm and seed answers exactly as many questions as the arm did
    filters = {arm: [logprob_filter(base_rows, answered(r, qids)) for r in rs] for arm, rs in runs.items()}

    out = {"n_questions": len(qids), "base_known": sum(base[q] == "correct" for q in qids), "arms": {}, "pairs": {}}
    print(f"{len(qids)} validation questions, {out['base_known']} known to the base model (greedy); mean ± sd over seeds\n")
    print(f"{'run':22s}" + "".join(f"{c[:12]:>15s}" for c in COLS) + "     N1    N2    N3")
    base_s = scores(base, base, qids)
    print(f"{'base':22s}" + "".join(f"{100 * base_s[c]:9.1f}      " for c in COLS) + f"{base_s['N1']:7d}{0:6d}{0:6d}")
    for arm, rs in runs.items():
        for name, versions in ((arm, rs), (f"  logprob filter @{arm[4:]}", filters[arm])):
            per = [scores(base, r, qids) for r in versions]
            mean = {c: st.mean(p[c] for p in per) for c in (*COLS, "N1", "N2", "N3")}
            sd = {c: st.stdev(p[c] for p in per) for c in COLS}
            out["arms"][name.strip()] = dict(mean=mean, sd=sd, per_seed=per)
            print(f"{name:22s}" + "".join(f"{100 * mean[c]:6.1f} ± {100 * sd[c]:4.1f} " for c in COLS)
                  + "".join(f"{mean[n]:6.1f}" for n in ("N1", "N2", "N3")))

    rng = random.Random(0)
    draws = [[rng.choice(qids) for _ in qids] for _ in range(a.n_boot)]

    def f1(versions, qs):
        return sum(f1_scores(confusion(base, r, qs))["F1_rel"] for r in versions) / len(versions)

    comparisons = [(x, y, runs[x], runs[y]) for x, y in PAIRS if x in runs and y in runs]
    comparisons += [(arm, "logprob filter", runs[arm], filters[arm]) for arm in runs]
    print("\nF1_rel difference, mean over seeds [95% interval]  (interval excluding 0 = reliable)")
    for x, y, A, B in comparisons:
        d = sorted(f1(A, q) - f1(B, q) for q in draws)
        est, lo, hi = f1(A, qids) - f1(B, qids), d[int(0.025 * a.n_boot)], d[int(0.975 * a.n_boot) - 1]
        out["pairs"][f"{x} - {y}"] = dict(estimate=est, ci95=[lo, hi])
        print(f"  {x:12s} - {y:15s}: {100 * est:+6.1f}  [{100 * lo:+6.1f}, {100 * hi:+6.1f}]")

    k = k_values(base_rows)
    groups = defaultdict(list)
    for q in qids:
        groups[bucket(k.get(q, 0.0))].append(q)
    print("\nBy the base model's K(q) on each validation question: % answered right / wrong / refused (mean over seeds)")
    print(f"{'run':14s}" + "".join(f"{b + f' (n={len(groups[b])})':>24s}" for b in ("known", "partial", "unknown")))
    out["by_k_bucket"] = {"sizes": {b: len(v) for b, v in groups.items()}}
    for name, versions in [("base", [base])] + list(runs.items()):
        row, rec = f"{name:14s}", {}
        for b in ("known", "partial", "unknown"):
            qs = groups[b]
            c = Counter(o for r in versions for o in (("abstain" if r[q] == "abstain" else
                                                       "right" if r[q] == "correct" else "wrong") for q in qs))
            tot = len(qs) * len(versions)
            rec[b] = {o: c[o] / tot for o in ("right", "wrong", "abstain")} if tot else {}
            row += f"{'':6s}{100 * rec[b].get('right', 0):5.1f} /{100 * rec[b].get('wrong', 0):5.1f} /{100 * rec[b].get('abstain', 0):5.1f}"
        out["by_k_bucket"][name] = rec
        print(row)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
