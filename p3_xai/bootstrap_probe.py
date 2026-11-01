"""Fact-cluster bootstrap intervals for a frozen grouped validation probe."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def fact_bootstrap(rows, groups_by_id, replicates=5000, seed=20261024):
    """Resample whole fact groups, preserving all held-out variants per draw."""
    if replicates < 100:
        raise ValueError("Use at least 100 replicates")
    groups = sorted({groups_by_id[row["question_id"]] for row in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two validation fact groups")
    by_group = {group: [] for group in groups}
    for row in rows:
        by_group[groups_by_id[row["question_id"]]].append(row)

    rng = np.random.default_rng(seed)
    metrics = {"auroc": [], "auprc": []}
    skipped_single_class = 0
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        draw = [row for group in sampled for row in by_group[group]]
        labels = np.asarray([int(row["correct_label"]) for row in draw])
        scores = np.asarray([float(row["probability"]) for row in draw])
        if len(np.unique(labels)) < 2:
            skipped_single_class += 1
            continue
        metrics["auroc"].append(float(roc_auc_score(labels, scores)))
        metrics["auprc"].append(float(average_precision_score(labels, scores)))

    result = {
        "method": "nonparametric percentile bootstrap resampling fact groups with replacement",
        "seed": seed,
        "requested_replicates": replicates,
        "valid_two_class_replicates": replicates - skipped_single_class,
        "skipped_single_class_replicates": skipped_single_class,
        "validation_fact_groups": len(groups),
        "confidence_level": 0.95,
        "interval_note": "Percentile interval over valid two-class replicates; exploratory because labels are provisional.",
        "metrics": {},
    }
    for name, values in metrics.items():
        if values:
            result["metrics"][name] = {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
        else:
            result["metrics"][name] = {"lower": None, "upper": None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/p3/synthetic_main_v0_sft_matched_seed0"))
    parser.add_argument("--probe-dir", type=Path, default=Path("results/p3/synthetic_main_v0_sft_matched_seed0/probe_reviewed"))
    parser.add_argument("--output", type=Path, default=Path("results/p3/synthetic_main_v0_sft_matched_seed0/probe_reviewed/fact_bootstrap.json"))
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()

    with np.load(args.run_dir / "features.npz", allow_pickle=False) as archive:
        ids = archive["question_ids"].astype(str).tolist()
        group_values = archive["group_ids"].astype(str).tolist()
    groups_by_id = dict(zip(ids, group_values))
    if len(groups_by_id) != len(ids):
        raise ValueError("Duplicate question IDs in feature archive")

    split = json.loads((args.probe_dir / "split.json").read_text())
    validation_ids = set(split["validation_ids"])
    if not validation_ids <= set(groups_by_id):
        raise ValueError("Validation split contains IDs absent from features")
    rows = []
    with (args.probe_dir / "validation_predictions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["probe"] == "layer_18":
                rows.append({
                    "question_id": row["question_id"],
                    "correct_label": int(row["correct_label"]),
                    "probability": float(row["probability"]),
                })
    if {row["question_id"] for row in rows} != validation_ids - set(split.get("excluded_unlabeled_ids", [])):
        raise ValueError("Layer 18 score IDs do not match labeled validation IDs")
    output = fact_bootstrap(rows, groups_by_id, args.replicates, args.seed)
    output["run_id"] = json.loads((args.run_dir / "manifest.json").read_text())["run_id"]
    output["probe"] = "layer_18"
    output["target"] = "assistant-reviewed provisional greedy exact-match correctness"
    output["n_validation_questions"] = len(rows)
    output["n_validation_correct"] = sum(row["correct_label"] == 1 for row in rows)
    output["n_validation_not_correct"] = sum(row["correct_label"] == 0 for row in rows)
    output["n_validation_groups"] = len({groups_by_id[row["question_id"]] for row in rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
