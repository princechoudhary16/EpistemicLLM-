"""P6 Week 3: the common evaluator (plan.md section 2, P6: "ensure all methods are
evaluated on the same held-out examples"; Week 3 acceptance criterion: "common
evaluator accepts P5 outputs").

Takes any number of label=path response files in the shared schema and:
  1. validates every method was evaluated on the identical set of question_ids
     (refuses to compare otherwise, rather than silently comparing mismatched
     populations),
  2. computes the same behavior/knowledge/risk-coverage metrics for every method,
  3. computes a paired, cluster-bootstrapped accuracy difference against a baseline
     method (default: the first one given).

Takes response files, not method names -- so when P5 has checkpoint predictions in
this schema, they plug in with `--responses ... p5=results/raw/p1_pilot_v0_p5.jsonl`,
no code change required.

Usage:
    python -m p6_baselines.compare_methods \
        --responses base=results/raw/p1_pilot_v0_base.jsonl \
                     sft=results/raw/p1_pilot_v0_sft.jsonl \
                     idk=results/raw/p1_pilot_v0_idk.jsonl \
        --out results/final/baseline_comparison.json

Risk-coverage/AURC defaults to the `confidence_score` field (avg log-probability --
the one signal every method, including the untouched base model, can produce).
Pass --confidence-field aux_confidence_score to use a method-specific signal instead
(e.g. an auxiliary head's own predicted score) -- note that's then only comparable
across methods that actually populate that field; methods that leave it null fall
back to the lowest-confidence sentinel, same as any other missing confidence value.
"""

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import (area_under_risk_coverage, behavior_metrics,
                             knowledge_scores, paired_bootstrap_diff, risk_coverage_curve)
from common.schema import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT_VERSION = "p6-eval-v1.1"  # v1 -> v1.1: added --confidence-field (additive,
# default-unchanged); frozen per plan.md section 3, bump deliberately, not silently.


def greedy_rows(rows):
    return [r for r in rows if r["is_greedy"]]


def accuracy_metric(rows):
    return behavior_metrics([r["outcome"] for r in greedy_rows(rows)])["accuracy"]


def method_summary(rows, confidence_field="confidence_score"):
    greedy = greedy_rows(rows)
    bm = behavior_metrics([r["outcome"] for r in greedy])
    confidences = [r[confidence_field] if r.get(confidence_field) is not None else float("-inf") for r in greedy]
    rc = risk_coverage_curve([r["outcome"] for r in greedy], confidences)
    ks = knowledge_scores(rows)
    mean_k = sum(v["knowledge_score"] for v in ks.values()) / len(ks) if ks else None
    return {
        **bm,
        "aurc": area_under_risk_coverage(rc),
        "confidence_field": confidence_field,
        "mean_knowledge_score": mean_k,
        "n_facts_scored": len(ks),
    }


def _response_layout(rows, label):
    """Validate per-question response identities and return sample/metadata layouts."""
    if not rows:
        raise ValueError(f"'{label}' has no response rows")
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        qid = row.get("question_id")
        if qid is None:
            raise ValueError(f"'{label}' has a response without question_id")
        sample_index = row.get("sample_index")
        if sample_index is None:
            raise ValueError(f"'{label}' has a response without sample_index for {qid}")
        if not isinstance(row.get("is_greedy"), bool):
            raise ValueError(f"'{label}' has a non-boolean is_greedy for {qid}")
        identity = (qid, str(sample_index))
        if identity in seen:
            raise ValueError(f"'{label}' has duplicate response {sample_index!r} for {qid}")
        seen.add(identity)
        grouped[qid].append(row)

    layout = {}
    for qid, question_rows in grouped.items():
        greedy = [row for row in question_rows if row.get("is_greedy") is True]
        if len(greedy) != 1:
            raise ValueError(f"'{label}' has {len(greedy)} greedy responses for {qid}; expected exactly one")
        samples = frozenset(str(row["sample_index"]) for row in question_rows if row.get("is_greedy") is not True)
        if not samples:
            raise ValueError(f"'{label}' has no sampled responses for {qid}")
        metadata_fields = (
            "question", "reference_answer", "source_dataset", "dataset_version", "split",
            "model_id", "model_revision", "prompt_version", "decoding_config_id",
        )
        metadata = tuple(greedy[0].get(field) for field in metadata_fields)
        if any(tuple(row.get(field) for field in metadata_fields) != metadata for row in question_rows):
            raise ValueError(f"'{label}' has inconsistent question/reference metadata across responses for {qid}")
        layout[qid] = (samples, metadata)
    return layout


def validate_methods(methods):
    """Require matching question sets, sample indices, and question/reference metadata."""
    if not methods:
        raise ValueError("at least one method is required")
    layouts = {label: _response_layout(rows, label) for label, rows in methods.items()}
    first_label = next(iter(layouts))
    expected = layouts[first_label]
    for label, actual in layouts.items():
        if set(actual) != set(expected):
            raise ValueError(
                f"'{label}' and '{first_label}' were not evaluated on the same questions "
                f"({len(set(actual) ^ set(expected))} mismatched question_ids) -- refusing to compare"
            )
        for qid in expected:
            if actual[qid][0] != expected[qid][0]:
                raise ValueError(f"'{label}' has different sampled response indices for {qid}")
            if actual[qid][1] != expected[qid][1]:
                raise ValueError(f"'{label}' has different question/reference metadata for {qid}")
    return set(expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="+", required=True,
                     help="label=path pairs, e.g. base=results/raw/p1_pilot_v0_base.jsonl")
    ap.add_argument("--baseline-label", default=None, help="paired diffs vs this label (default: first given)")
    ap.add_argument("--confidence-field", default="confidence_score",
                     help="schema field used for risk-coverage/AURC (default: confidence_score)")
    ap.add_argument("--out", default="results/final/baseline_comparison.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    methods = {}
    for item in args.responses:
        if "=" not in item:
            raise ValueError(f"response entry must be label=path, got {item!r}")
        label, path = item.split("=", 1)
        if not label or label in methods:
            raise ValueError(f"empty or duplicate method label: {label!r}")
        methods[label] = read_jsonl(REPO_ROOT / path)

    # Validate the full response layout before computing any comparisons.
    question_ids = validate_methods(methods)
    print(f"Matched-question-join check passed: all {len(methods)} methods share "
          f"{len(question_ids)} question_ids, with matching sample rows and references.", file=sys.stderr)

    summaries = {label: method_summary(rows, confidence_field=args.confidence_field) for label, rows in methods.items()}

    first_label = next(iter(methods))
    baseline = args.baseline_label or first_label
    if baseline not in methods:
        raise ValueError(f"baseline label {baseline!r} is not in the response list")
    diffs = {}
    for label, rows in methods.items():
        if label == baseline:
            continue
        diffs[f"{label}_vs_{baseline}"] = paired_bootstrap_diff(
            rows, methods[baseline], accuracy_metric, cluster_key="question_id", n_boot=args.n_boot,
        )

    result = {"eval_script_version": EVAL_SCRIPT_VERSION, "baseline": baseline,
              "summaries": summaries, "accuracy_diffs": diffs}

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Per-method summary (greedy accuracy/coverage/etc., all generations' knowledge score) ===")
    print(json.dumps(summaries, indent=2))
    print(f"\n=== Accuracy diffs vs '{baseline}' (95% CI, question-cluster bootstrap, n_boot={args.n_boot}) ===")
    print(json.dumps(diffs, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
