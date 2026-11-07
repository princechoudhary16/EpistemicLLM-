"""Compare abstention models on the same questions: GeoDe's metrics plus the team's shared ones.

    python -m p2_abstention.evaluate --base results/raw/p2_triviaqa_v0_validation_base.jsonl \
        --run rtuning=results/raw/p2_triviaqa_v0_validation_abstention_rtuning.jsonl \
        --run geode=results/raw/p2_triviaqa_v0_validation_abstention_geode.jsonl \
        --run random=results/raw/p2_triviaqa_v0_validation_abstention_random.jsonl \
        --reference rtuning --out results/final/abstention/validation_comparison.json

"Known" / "unknown" = whether the BASE model's greedy answer to that question was correct (GeoDe Table 1). Greedy answers:
    known:    N1 correct, N2 wrong, N3 abstained        unknown: N4 wrong, N5 abstained (N0 = correct, reported only)
    F1_ans = harmonic mean of N1/(N1+N2+N3) and N1/(N1+N2+N4)       how well it still answers what it knew
    F1_abs = harmonic mean of N5/(N4+N5) and N5/(N3+N5)             how well it abstains on what it did not know
    F1_rel = harmonic mean of F1_ans and F1_abs                      GeoDe's headline
Shared metrics (accuracy, hallucination, abstention, coverage, selective accuracy) come from common.metrics.
Differences against --reference get a 95% question-bootstrap interval (paired: same resampled questions).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import behavior_metrics  # noqa: E402

N_BOOT = 1000
SEED = 0


def greedy(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["question_id"]: r["outcome"] for r in map(json.loads, f) if r["is_greedy"]}


def hm(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def ratio(a, b):
    return a / b if b else 0.0


def confusion(base: dict, run: dict, qids) -> dict:
    n = dict.fromkeys(("N0", "N1", "N2", "N3", "N4", "N5"), 0)
    for q in qids:
        known, o = base[q] == "correct", run[q]
        if o == "invalid":
            o = "incorrect"                  # an empty answer is not an abstention
        if known:
            n["N1" if o == "correct" else "N3" if o == "abstain" else "N2"] += 1
        else:
            n["N0" if o == "correct" else "N5" if o == "abstain" else "N4"] += 1
    return n


def f1_scores(n: dict) -> dict:
    f_ans = hm(ratio(n["N1"], n["N1"] + n["N2"] + n["N3"]), ratio(n["N1"], n["N1"] + n["N2"] + n["N4"]))
    f_abs = hm(ratio(n["N5"], n["N4"] + n["N5"]), ratio(n["N5"], n["N3"] + n["N5"]))
    return dict(F1_ans=f_ans, F1_abs=f_abs, F1_rel=hm(f_ans, f_abs))


def evaluate(base: dict, runs: dict, reference: str | None = None, n_boot=N_BOOT, seed=SEED) -> dict:
    qids = sorted(set(base).intersection(*[set(r) for r in runs.values()]))
    if not qids:
        raise ValueError("no question is answered by the base model and every run")
    out = {"n_questions": len(qids), "base_known": sum(base[q] == "correct" for q in qids), "runs": {}}
    for name, run in {"base": base, **runs}.items():
        n = confusion(base, run, qids)
        out["runs"][name] = dict(**n, **f1_scores(n), **behavior_metrics([run[q] for q in qids]))
    if reference:
        rng = random.Random(seed)
        draws = [[rng.choice(qids) for _ in qids] for _ in range(n_boot)]
        for name, run in runs.items():
            if name == reference:
                continue
            diffs = sorted(f1_scores(confusion(base, run, d))["F1_rel"]
                           - f1_scores(confusion(base, runs[reference], d))["F1_rel"] for d in draws)
            point = out["runs"][name]["F1_rel"] - out["runs"][reference]["F1_rel"]
            out["runs"][name][f"F1_rel_minus_{reference}"] = dict(
                estimate=point, ci95=[diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True, help="base-model responses on the same split (defines known/unknown)")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=FILE")
    ap.add_argument("--reference", help="run name to report F1_rel differences against")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    runs = {k: greedy(v) for k, v in (x.split("=", 1) for x in a.run)}
    if a.reference and a.reference not in runs:
        raise SystemExit(f"--reference {a.reference} is not one of the runs {list(runs)}")
    res = evaluate(greedy(a.base), runs, a.reference)
    cols = ("F1_ans", "F1_abs", "F1_rel", "accuracy", "hallucination_rate", "abstention_rate", "coverage")
    print(f"{res['n_questions']} questions, {res['base_known']} known to the base model")
    print(f"{'run':10s} " + " ".join(f"{c[:13]:>13s}" for c in cols))
    for name, r in res["runs"].items():
        print(f"{name:10s} " + " ".join(f"{100 * r[c]:13.1f}" for c in cols))
    for name, r in res["runs"].items():
        for k, v in r.items():
            if k.startswith("F1_rel_minus_"):
                print(f"{name} {k}: {100 * v['estimate']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
