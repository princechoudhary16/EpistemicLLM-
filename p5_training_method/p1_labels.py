"""P1's K per fact -> the label file train.py reads (second arm of reports/p5_kq_target.md, Amendment 1).

P1 (p1_exposure/k_labels.py) measured K for each main_v0 fact: the share of 20 sampled answers (4 training wordings x 5)
that the shared grader marks correct, from P1's own float32 matched-recipe model. This script does not recompute anything;
it checks P1's file against the data and rewrites it as {"labels": {fact_id: K}, ...} plus provenance, the same shape
kq_labels.py writes, so `train.py --aux-target k --k-labels <this file>` works unchanged.

    python p5_training_method/p1_labels.py            # -> results/final/p5_kq_labels_main_v0_p1.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
P1_DIR = REPO / "p1_exposure" / "analysis" / "results" / "k_labels" / "main_v0_float32_seed0"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def check_and_convert(p1_rows, facts, meta):
    """Raises ValueError unless P1's rows match the world's facts and were made in float32 with grader v5."""
    if meta.get("dtype") != "torch.float32":
        raise ValueError(f"P1 K was not made in float32 (dtype={meta.get('dtype')!r})")
    expected = {f["fact_id"]: f for f in facts}
    got = {r["fact_id"]: r for r in p1_rows}
    if len(got) != len(p1_rows):
        raise ValueError("duplicate fact_id in P1's file")
    if set(got) != set(expected):
        raise ValueError(f"fact ids differ: {len(set(expected) - set(got))} missing, {len(set(got) - set(expected))} extra")
    for fid, r in got.items():
        f = expected[fid]
        if (r["exposure_level"], r["split"]) != (f["exposure_level"], f["split"]):
            raise ValueError(f"{fid}: exposure/split differ from facts.jsonl")
        if not 0.0 <= r["k"] <= 1.0:
            raise ValueError(f"{fid}: K={r['k']} outside [0, 1]")
        if r["n_sampled"] != 20:
            raise ValueError(f"{fid}: expected 20 samples, found {r['n_sampled']}")
        if r["grading_version"] != meta["grading_version"]:
            raise ValueError(f"{fid}: graded with {r['grading_version']}, meta says {meta['grading_version']}")
    return {fid: float(r["k"]) for fid, r in sorted(got.items())}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-dir", default=str(P1_DIR))
    ap.add_argument("--facts", default=str(REPO / "data" / "synthetic" / "main_v0" / "facts.jsonl"))
    ap.add_argument("--out", default=str(REPO / "results" / "final" / "p5_kq_labels_main_v0_p1.json"))
    args = ap.parse_args()

    p1_dir = Path(args.p1_dir)
    k_file, meta_file = p1_dir / "k_per_fact.jsonl", p1_dir / "k_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    try:
        labels = check_and_convert(read_jsonl(k_file), read_jsonl(args.facts), meta)
    except ValueError as e:
        sys.exit(f"P1 labels rejected: {e}")
    doc = {
        "dataset": "main_v0",
        "source": "P1 (p1_exposure/k_labels.py), not recomputed here",
        "teacher": "P1's own float32 run of the matched recipe, training seed 0 (not P6's sft_matched)",
        "labelled_questions": "the 4 training wordings of each fact (not eval.jsonl, not held-out wordings)",
        "n_sampled_responses_per_fact": 20,
        "dtype": "float32",
        "grading_version": meta["grading_version"],
        "decoding": meta["decoding"],
        "p1_k_per_fact_sha256": sha256(k_file),
        "p1_k_meta_sha256": sha256(meta_file),
        "facts_sha256": sha256(args.facts),
        "n_facts": len(labels),
        "labels": labels,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    by_level = {}
    facts = {f["fact_id"]: f["exposure_level"] for f in read_jsonl(args.facts)}
    for fid, k in labels.items():
        by_level.setdefault(facts[fid], []).append(k)
    print({e: round(sum(v) / len(v), 3) for e, v in sorted(by_level.items())})
    print(f"wrote {out} ({len(labels)} facts)")


if __name__ == "__main__":
    main()
