"""Per-exposure-level knowledge-score breakdown, for comparing against P1/P5's
own per-exposure tables (which this report's overall-accuracy numbers did not
originally include -- raised as a valid gap by a teammate review).

Usage:
    python -m p6_baselines.exposure_breakdown results/raw/p1_pilot_v0_sft.jsonl
    python -m p6_baselines.exposure_breakdown results/raw/p1_main_v0_sft.jsonl
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import knowledge_scores
from common.schema import read_jsonl


def main():
    path = sys.argv[1]
    rows = read_jsonl(path)
    ks = knowledge_scores(rows)  # {(run_id, checkpoint_id, question_id): {"knowledge_score":..., "n_samples":...}}

    # exposure_level per question_id, from the rows themselves (same field on every row).
    exposure_by_qid = {r["question_id"]: r["exposure_level"] for r in rows}

    by_level = defaultdict(list)
    for (run_id, checkpoint_id, qid), v in ks.items():
        by_level[exposure_by_qid[qid]].append(v["knowledge_score"])

    print(f"{path}")
    print(f"{'exposure':>8} | {'n_facts':>8} | {'mean_K':>8} | {'K=0':>6} | {'0<K<1':>6} | {'K>=0.8':>6}")
    for level in sorted(by_level, key=lambda x: (x is None, x)):
        scores = by_level[level]
        n = len(scores)
        mean_k = sum(scores) / n
        k0 = sum(1 for s in scores if s == 0)
        kmid = sum(1 for s in scores if 0 < s < 1)
        khigh = sum(1 for s in scores if s >= 0.8)
        print(f"{str(level):>8} | {n:>8} | {mean_k:>8.3f} | {k0:>6} | {kmid:>6} | {khigh:>6}")

    overall = [s for scores in by_level.values() for s in scores]
    print(f"{'overall':>8} | {len(overall):>8} | {sum(overall)/len(overall):>8.3f}")


if __name__ == "__main__":
    main()
