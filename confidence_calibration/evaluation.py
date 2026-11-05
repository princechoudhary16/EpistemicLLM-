"""Metrics for one fitted signal on one split. The same code evaluates validation and held-out rows.

Probability metrics (Brier, ECE, AUROC) use attempted greedy answers with a usable signal. Policy
metrics keep abstentions in the denominator. Intervals resample facts on synthetic data and questions
on real QA, so paraphrases are never treated as independent.
"""

from collections import defaultdict

from common.metrics import (apply_threshold, area_under_risk_coverage, auprc, auroc, behavior_metrics, bootstrap_ci,
                            brier_score, confident_hallucinations, expected_calibration_error, reliability_bins,
                            risk_coverage_curve)

from .calibration import attempted_with_signal, cluster_key, confidences, outcome_label, select_threshold
from .signals import oriented, raw_probability

N_BOOTSTRAP = 1000
N_RELIABILITY_BINS = 10
REFERENCE_SIGNAL = "avg_logprob"


def _with_cluster(rows):
    return [{**row, "_cluster": cluster_key(row)} for row in rows]


def _interval(rows, metric, seed):
    result = bootstrap_ci(_with_cluster(rows), metric, "_cluster", n_boot=N_BOOTSTRAP, seed=seed)
    return {"low": result["ci_low"], "high": result["ci_high"], "n_clusters": result["n_clusters"],
            "replicates": result["n_boot"], "undefined_replicates": result["n_undefined"]}


def signal_auroc(rows, signal):
    return auroc([oriented(signal, r[signal]) for r in rows], [outcome_label(r) for r in rows])[0]


def signal_auprc(rows, signal):
    return auprc([oriented(signal, r[signal]) for r in rows], [outcome_label(r) for r in rows])


def _probability_metrics(rows, calibrator):
    return ([calibrator.probability(r[calibrator.signal]) for r in rows], [outcome_label(r) for r in rows])


def _raw_probability_metrics(rows, signal):
    """Reading a signal directly as a probability, which is what calibration is meant to improve on."""
    values = [raw_probability(signal, r[signal]) for r in rows]
    if not rows or any(v is None for v in values):
        return None
    labels = [outcome_label(r) for r in rows]
    return {"brier": brier_score(values, labels),
            "ece_10_bins": expected_calibration_error(values, labels, n_bins=N_RELIABILITY_BINS)}


def _group_metrics(members, calibrator, seed):
    usable = attempted_with_signal(members, calibrator.signal)
    probabilities, labels = _probability_metrics(usable, calibrator)
    discrimination = signal_auroc(usable, calibrator.signal) if usable else None
    interval = None
    if discrimination is not None:
        interval = _interval(usable, lambda rs: signal_auroc(rs, calibrator.signal), seed)
    return {"n_questions": len(members), "behavior": behavior_metrics([r["outcome"] for r in members]),
            "n_calibrated_attempts": len(usable), "n_correct": sum(labels), "n_incorrect": len(labels) - sum(labels),
            "auroc": discrimination, "auroc_ci_95": interval,
            "brier": brier_score(probabilities, labels) if usable else None,
            "ece_10_bins": expected_calibration_error(probabilities, labels, n_bins=N_RELIABILITY_BINS)}


def stability_slices(rows, calibrator, seed):
    """Does the signal keep working within each exposure level and knowledge group?

    Descriptive. Sampled K does not include the greedy response being judged, but it shares its samples
    with the consistency signals, so K slices for those two are partly by construction.
    """
    groups = defaultdict(list)
    for row in rows:
        if row.get("exposure_level") is not None:
            groups[("exposure_level", str(row["exposure_level"]))].append(row)
        score = row.get("knowledge_score")
        if score is not None:
            bucket = "K=0" if score == 0 else ("0<K<0.8" if score < 0.8 else "K>=0.8")
            groups[("sample_knowledge_score", bucket)].append(row)
    output = defaultdict(dict)
    for (dimension, label), members in sorted(groups.items(), key=lambda item: (item[0][0], _level_order(item[0][1]))):
        output[dimension][label] = _group_metrics(members, calibrator, seed)
    return dict(output)


def _level_order(label):
    return (0, int(label), "") if label.isdigit() else (1, 0, label)


def policy_metrics(rows, calibrator, thresholds):
    """Behaviour under each frozen threshold, plus the full risk-coverage curve for the figures."""
    outcomes = [row["outcome"] for row in rows]
    confidence = confidences(rows, calibrator)
    curve = risk_coverage_curve(outcomes, confidence)
    operating_points = {}
    for key, selected in thresholds.items():
        threshold = selected["probability_correct"]
        operating_points[key] = {
            "threshold": threshold, "selection_coverage": selected["selection_coverage"],
            "behavior": behavior_metrics(apply_threshold(outcomes, confidence, threshold)),
            "confident_hallucinations": confident_hallucinations(outcomes, confidence, threshold),
        }
    return {"answer_everything": behavior_metrics(outcomes), "operating_points": operating_points,
            "area_under_risk_coverage": area_under_risk_coverage(curve), "risk_coverage_curve": curve}


def threshold_stability(rows, calibrator, target_coverage, seed):
    """How much the selected threshold moves when the selection split is resampled."""
    def threshold(sample):
        selected = select_threshold([r["outcome"] for r in sample], confidences(sample, calibrator), target_coverage)
        return selected["probability_correct"] if selected else None
    return _interval(rows, threshold, seed)


def evaluate_signal(rows, calibrator, thresholds, train_base_rate, seed):
    """All metrics for one calibrated signal on one split."""
    signal = calibrator.signal
    usable = attempted_with_signal(rows, signal)
    probabilities, labels = _probability_metrics(usable, calibrator)
    metrics = {
        "n": len(usable), "n_correct": sum(labels), "n_incorrect": len(labels) - sum(labels),
        "n_attempted_without_signal": sum(r["outcome"] in ("correct", "incorrect") for r in rows) - len(usable),
        "auroc": signal_auroc(usable, signal) if usable else None,
        "auprc": signal_auprc(usable, signal) if usable else None,
        "brier": brier_score(probabilities, labels) if usable else None,
        "brier_base_rate": brier_score([train_base_rate] * len(usable), labels) if usable else None,
        "ece_10_bins": expected_calibration_error(probabilities, labels, n_bins=N_RELIABILITY_BINS),
        "reliability_bins": reliability_bins(probabilities, labels, n_bins=N_RELIABILITY_BINS),
        "uncalibrated": _raw_probability_metrics(usable, signal),
        "stability": stability_slices(rows, calibrator, seed),
        "policy": policy_metrics(rows, calibrator, thresholds),
    }
    if usable:
        metrics["auroc_ci_95"] = _interval(usable, lambda rs: signal_auroc(rs, signal), seed)
        metrics["auprc_ci_95"] = _interval(usable, lambda rs: signal_auprc(rs, signal), seed)
        metrics["brier_ci_95"] = _interval(usable, lambda rs: brier_score(*_probability_metrics(rs, calibrator)), seed)
        metrics["ece_10_bins_ci_95"] = _interval(usable, lambda rs: expected_calibration_error(
            *_probability_metrics(rs, calibrator), n_bins=N_RELIABILITY_BINS), seed)
    return metrics


def compare_to_reference(rows, signals, seed):
    """AUROC difference from the shared log-probability signal, on questions where both are usable."""
    if REFERENCE_SIGNAL not in signals:
        return {}
    comparisons = {}
    for signal in signals:
        if signal == REFERENCE_SIGNAL:
            continue
        shared = _with_cluster(attempted_with_signal(attempted_with_signal(rows, signal), REFERENCE_SIGNAL))
        comparisons[signal] = _paired_auroc(shared, signal, seed)
    return {"reference": REFERENCE_SIGNAL, "differences": comparisons}


def _paired_auroc(rows, signal, seed):
    def difference(sample):
        first, second = signal_auroc(sample, signal), signal_auroc(sample, REFERENCE_SIGNAL)
        return None if first is None or second is None else first - second
    result = bootstrap_ci(rows, difference, "_cluster", n_boot=N_BOOTSTRAP, seed=seed)
    return {"auroc_difference": result["estimate"], "low": result["ci_low"], "high": result["ci_high"],
            "n_clusters": result["n_clusters"], "undefined_replicates": result["n_undefined"]}
