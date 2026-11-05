"""Development stage: fit every signal, select on validation, and write down what was chosen.

The frozen policy is the hand-off artifact. It holds plain numbers (no pickles), so anyone can apply
it, and records enough provenance to refuse responses produced under a different protocol.
"""

from common.grading import GRADING_VERSION

from . import config
from .calibration import (attempted_with_signal, cluster_key, confidences, development_splits, fit_calibrator,
                          outcome_label, select_threshold)
from .data import PROVENANCE_FIELDS
from .evaluation import compare_to_reference, evaluate_signal, threshold_stability
from .signals import SIGNAL_DIRECTIONS

POLICY_SCHEMA = "confidence-policy-v1"


def coverage_key(coverage):
    return f"{coverage:.2f}"


def _run_provenance(rows):
    provenance = {field: rows[0].get(field) for field in PROVENANCE_FIELDS}
    provenance["grading_version"] = GRADING_VERSION
    provenance["signals_version"] = rows[0].get("signals_version")
    return provenance


def _split_counts(rows):
    counts = {"n_questions": len(rows), "outcomes": {}}
    for outcome in ("correct", "incorrect", "abstain", "invalid"):
        counts["outcomes"][outcome] = sum(r["outcome"] == outcome for r in rows)
    return counts


def prediction_rows(rows, split_name, calibrators):
    """One row per question and signal for downstream merging; abstentions stay, with a null probability."""
    output = []
    for row in rows:
        for signal, calibrator in calibrators.items():
            attempted = row["outcome"] in ("correct", "incorrect")
            output.append({
                "question_id": row["question_id"], "fact_id": row.get("fact_id"), "run_id": row.get("run_id"),
                "checkpoint_id": row.get("checkpoint_id"), "method": row.get("method"),
                "training_seed": row.get("training_seed"), "dataset_version": row.get("dataset_version"),
                "exposure_level": row.get("exposure_level"), "split": split_name, "outcome": row["outcome"],
                "signal": signal, "raw_score": row.get(signal),
                "probability_correct": calibrator.probability(row.get(signal)) if attempted else None,
            })
    return output


def fit_policy(responses, inputs, seed=0, target_coverage=config.PRIMARY_COVERAGE, coverage_grid=config.COVERAGE_GRID):
    """Returns (policy, development summary, validation prediction rows)."""
    if not 0 < target_coverage <= 1 or any(not 0 < c <= 1 for c in coverage_grid):
        raise ValueError("coverage targets must be in (0, 1]")
    splits, split_metadata = development_splits(responses, seed)
    train, validation = splits["train"], splits["validation"]
    grid = sorted({*coverage_grid, target_coverage}, reverse=True)

    calibrators, fits = {}, {}
    for signal in SIGNAL_DIRECTIONS:
        calibrator, fits[signal] = fit_calibrator(train, signal)
        if calibrator is not None and attempted_with_signal(validation, signal):
            calibrators[signal] = calibrator
        elif calibrator is not None:
            fits[signal] = {**fits[signal], "status": "unavailable", "reason": "validation has no usable values"}
    if not calibrators:
        raise ValueError("no signal could be fitted: "
                         f"training split needs both outcomes with a usable signal; {fits}")

    labels = [outcome_label(r) for r in train if r["outcome"] in ("correct", "incorrect")]
    base_rate = sum(labels) / len(labels)
    outcomes = [r["outcome"] for r in validation]
    signals, results = {}, {}
    for signal, calibrator in calibrators.items():
        confidence = confidences(validation, calibrator)
        thresholds = {coverage_key(c): select_threshold(outcomes, confidence, c) for c in grid}
        thresholds = {key: value for key, value in thresholds.items() if value is not None}
        metrics = evaluate_signal(validation, calibrator, thresholds, base_rate, seed)
        metrics["threshold_stability_ci_95"] = threshold_stability(validation, calibrator, target_coverage, seed)
        signals[signal] = {"calibrator": calibrator.to_dict(), "thresholds": thresholds, "training": fits[signal]}
        results[signal] = {"training": fits[signal], "metrics": metrics}

    selected = min(results, key=lambda name: (results[name]["metrics"]["brier"], name))
    policy = {
        "schema": POLICY_SCHEMA,
        "protocol": {"fit_split": "train", "selection_split": "validation",
                     "selection_metric": "validation Brier score",
                     "held_out_rows_read": False, "torch_dtype": config.TORCH_DTYPE},
        "provenance": _run_provenance(validation),
        "inputs": inputs,
        "split_assignment": split_metadata,
        "train_base_rate": base_rate,
        "primary_coverage": coverage_key(target_coverage),
        "selected_signal": selected,
        "signals": signals,
        "unavailable_signals": {name: fit for name, fit in fits.items() if name not in signals},
        "development_clusters": sorted(str(cluster_key(r)) for r in train + validation),
    }
    summary = {
        "stage": "development", "evaluated_split": "validation", "provenance": policy["provenance"],
        "inputs": inputs, "split_assignment": split_metadata, "protocol": policy["protocol"],
        "split_counts": {"train": _split_counts(train), "validation": _split_counts(validation)},
        "train_base_rate": base_rate, "primary_coverage": policy["primary_coverage"],
        "selected_signal": selected, "signals": results,
        "unavailable_signals": policy["unavailable_signals"],
        "reference_comparison": compare_to_reference(validation, list(results), seed),
    }
    return policy, summary, prediction_rows(validation, "validation", calibrators)
