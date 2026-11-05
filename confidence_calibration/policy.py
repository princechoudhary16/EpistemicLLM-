"""Development stage: fit every signal, select on validation, and write down what was chosen.

The frozen policy is the hand-off artifact. It holds plain numbers (no pickles), so anyone can apply
it, and records enough provenance to refuse responses produced under a different protocol.
"""

import json
from pathlib import Path

from common.grading import GRADING_VERSION

from . import config
from .calibration import (ATTEMPTED_OUTCOMES, PlattCalibrator, attempted_with_signal, cluster_key, confidences,
                          development_splits, fit_calibrator, greedy_question_rows, outcome_label, select_threshold,
                          split_role)
from .data import PROVENANCE_FIELDS
from .error_analysis import confident_error_analysis
from .evaluation import compare_to_reference, evaluate_signal, threshold_stability
from .signals import SIGNAL_DIRECTIONS

POLICY_SCHEMA = "confidence-policy-v1"
# Must match for a calibrated probability to mean the same thing at all.
PROTOCOL_FIELDS = ("model_id", "model_revision", "prompt_version", "decoding_config_id", "grading_version")
# Must also match unless the evaluation is an explicit transfer test.
SAME_RUN_FIELDS = ("checkpoint_id", "method", "training_seed", "source_dataset", "dataset_version", "signals_version")


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


def load_policy(path):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"{path}: expected schema {POLICY_SCHEMA!r}, found {policy.get('schema')!r}")
    return policy


def check_compatible(policy, rows, allow_transfer=False):
    """Refuse rows the policy was not fitted for. Transfer keeps the protocol but lets the model or data differ."""
    for field in (PROTOCOL_FIELDS if allow_transfer else (*PROTOCOL_FIELDS, *SAME_RUN_FIELDS)):
        found = {row.get(field) for row in rows}
        if found != {policy["provenance"][field]}:
            raise ValueError(f"policy was fitted with {field}={policy['provenance'][field]!r} "
                             f"but the responses have {sorted(found, key=str)}")


def _check_unseen(policy, questions):
    overlap = {str(cluster_key(row)) for row in questions} & set(policy["development_clusters"])
    if overlap:
        raise ValueError(f"{len(overlap)} held-out groups were used in development; first: {sorted(overlap)[:5]}")


def evaluate_policy(policy, policy_sha256, responses, inputs, seed=0, allow_transfer=False):
    """Apply a frozen policy to held-out rows. Nothing is refitted or reselected.

    Returns (summary, prediction rows). With `allow_transfer`, signals the new rows do not record are
    listed as unavailable; otherwise a missing signal is an error, because it would silently abstain.
    """
    held_out = [row for row in responses if split_role(row) == "test"]
    if not held_out:
        raise ValueError("no held-out rows (locked_test, test, cal_test or final) in the responses")
    check_compatible(policy, held_out, allow_transfer)
    questions = greedy_question_rows(held_out)
    if not allow_transfer:
        _check_unseen(policy, questions)

    calibrators, results, unavailable = {}, {}, dict(policy["unavailable_signals"])
    for signal, frozen in policy["signals"].items():
        calibrator = PlattCalibrator(**frozen["calibrator"])
        attempted = [row for row in questions if row["outcome"] in ATTEMPTED_OUTCOMES]
        if len(attempted_with_signal(attempted, signal)) != len(attempted):
            missing = len(attempted) - len(attempted_with_signal(attempted, signal))
            if not allow_transfer:
                raise ValueError(f"{missing} attempted answers have no {signal!r}; score the responses first")
            unavailable[signal] = {"status": "unavailable", "reason": f"{missing} attempted answers have no value"}
            continue
        calibrators[signal] = calibrator
        metrics = evaluate_signal(questions, calibrator, frozen["thresholds"], policy["train_base_rate"], seed)
        results[signal] = {"training": frozen["training"], "metrics": metrics}
    if not results:
        raise ValueError("no signal in the policy can be applied to these responses")
    selected = policy["selected_signal"]
    summary = {
        "stage": "evaluation", "evaluated_split": "held-out test", "policy_sha256": policy_sha256,
        "transfer": allow_transfer, "policy_provenance": policy["provenance"],
        "provenance": _run_provenance(questions), "inputs": inputs, "protocol": policy["protocol"],
        "split_counts": {"held-out test": _split_counts(questions)},
        "train_base_rate": policy["train_base_rate"], "primary_coverage": policy["primary_coverage"],
        "selected_signal": selected if selected in results else None, "signals": results,
        "unavailable_signals": unavailable,
        "reference_comparison": compare_to_reference(questions, list(results), seed),
    }
    if selected in calibrators:
        summary["confident_errors"] = confident_error_analysis(
            questions, calibrators[selected],
            policy["signals"][selected]["thresholds"][policy["primary_coverage"]]["probability_correct"])
    return summary, prediction_rows(questions, "test", calibrators)
