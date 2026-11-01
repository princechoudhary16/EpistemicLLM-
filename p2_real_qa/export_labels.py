"""Export per-question knowledge labels from a graded P2 response file, for P3 (probes) and P4 (calibration).

    python -m p2_real_qa.export_labels --responses results/raw/p2_triviaqa_v0_train_probe_base.jsonl
        -> results/features/p2_triviaqa_v0_train_probe_k_labels.jsonl

One row per question:
    question_id, split, dataset_version            which question
    K, n_correct, n_samples                        K = correct sampled answers / sampled answers (common.metrics)
    k_bucket                                       unknown (K = 0) | partial (0 < K < 0.8) | known (K >= 0.8)
    outcome, correct                               the GREEDY answer: correct=1, incorrect/abstain=0, invalid=null
                                                   (the same outcome/correct pair p3_xai.probe expects)
    run_id, model_revision, prompt_version, decoding_config_id, grading_version   provenance, copied from the rows
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import schema  # noqa: E402
from common.metrics import knowledge_scores  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_MIN = 0.8                      # same cut as the plan's thresholds and analyze.py
CORRECT_OF = {"correct": 1, "incorrect": 0, "abstain": 0, "invalid": None}
PROVENANCE = ("run_id", "model_revision", "prompt_version", "decoding_config_id", "grading_version")


def k_bucket(k: float) -> str:
    return "unknown" if k == 0 else ("known" if k >= KNOWN_MIN else "partial")


def build_labels(rows: list) -> list:
    """Rows of ONE run (one checkpoint). Raises if the file mixes runs or a question has no greedy answer."""
    for f in PROVENANCE:
        vals = {r.get(f) for r in rows}
        if len(vals) != 1:
            raise ValueError(f"response file mixes several values of {f}: {sorted(map(str, vals))[:3]}")
    ks = {g[-1]: v for g, v in knowledge_scores(rows).items()}
    greedy, n_correct, first = {}, defaultdict(int), {}
    for r in rows:
        q = r["question_id"]
        first.setdefault(q, r)
        if r["is_greedy"]:
            greedy[q] = r["outcome"]
        elif r["outcome"] == "correct":
            n_correct[q] += 1
    missing = [q for q in first if q not in greedy]
    if missing:
        raise ValueError(f"{len(missing)} questions have no greedy response (first: {missing[:3]})")
    out = []
    for q, r in first.items():
        k = ks.get(q, {"knowledge_score": None, "n_samples": 0})
        K = k["knowledge_score"]
        out.append(dict(question_id=q, split=r["split"], dataset_version=r["dataset_version"],
                        K=K, n_correct=n_correct[q], n_samples=k["n_samples"],
                        k_bucket=k_bucket(K) if K is not None else None,
                        outcome=greedy[q], correct=CORRECT_OF[greedy[q]],
                        **{f: r[f] for f in PROVENANCE}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", help="default results/features/<responses stem without _base>_k_labels.jsonl")
    a = ap.parse_args(argv)
    src = Path(a.responses)
    labels = build_labels(schema.read_jsonl(src))
    stem = src.stem.removesuffix("_base")
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "features" / f"{stem}_k_labels.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema.write_jsonl(labels, out)
    counts = defaultdict(int)
    for l in labels:
        counts[l["k_bucket"]] += 1
    print(f"wrote {len(labels)} labels -> {out}  |  " + ", ".join(f"{b}: {counts[b]}" for b in ("unknown", "partial", "known")))


if __name__ == "__main__":
    main()
