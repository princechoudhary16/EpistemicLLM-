"""Leakage-safe probability calibration and confidence-policy evaluation.

Calibration targets the correctness of a greedy, substantive answer. Raw signals are mapped to
P(correct) with one-dimensional Platt scaling fitted on calibration-training rows only. Signal
selection and abstention thresholds use validation rows; locked-test rows are evaluated once.
"""

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.metrics import (
    ABSTAIN,
    CORRECT,
    INCORRECT,
    INVALID,
    apply_threshold,
    behavior_metrics,
    bootstrap_ci,
    brier_score,
    confident_hallucinations,
    expected_calibration_error,
    reliability_bins,
    risk_coverage_curve,
    auprc as common_auprc,
    auroc as common_auroc,
)

from .consistency import GROUP_KEYS, summarize_questions
from .signals import SIGNAL_DIRECTIONS, oriented

ATTEMPTED_OUTCOMES = {CORRECT, INCORRECT}
CANONICAL_SPLITS = {"cal_train": "train", "train_probe": "train", "train": "train",
                    "cal_val": "validation", "validation": "validation", "val": "validation",
                    "cal_test": "test", "locked_test": "test", "test": "test"}


def _cluster_key(row):
    return row.get("fact_id") or row["question_id"]


def _split_development_facts(rows, seed):
    """Turn P1's dev/final convention into train/validation/test without fact leakage."""
    if any(not row.get("fact_id") for row in rows):
        raise ValueError("dev/final data must include fact_id so paraphrases cannot cross splits")
    by_split = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "")).lower()].append(row)
    if set(by_split) != {"dev", "final"}:
        raise ValueError("expected either explicit train/validation/test splits or dev/final splits")

    groups = defaultdict(list)
    for row in by_split["dev"]:
        groups[_cluster_key(row)].append(row)
    final_groups = {_cluster_key(row) for row in by_split["final"]}
    overlap = set(groups) & final_groups
    if overlap:
        raise ValueError(f"{len(overlap)} facts/questions overlap dev and final splits; first: {sorted(overlap, key=str)[:5]}")
    group_ids = sorted(groups, key=str)
    if len(group_ids) < 4:
        raise ValueError("at least four distinct development facts are required for a 70/30 split")
    order = np.random.default_rng(seed).permutation(len(group_ids))
    n_train = min(len(group_ids) - 1, max(1, round(0.7 * len(group_ids))))
    train_ids = {group_ids[int(i)] for i in order[:n_train]}
    split_rows = {"train": [], "validation": [], "test": []}
    for gid, fact_rows in groups.items():
        split_rows["train" if gid in train_ids else "validation"].extend(fact_rows)
    split_rows["test"] = by_split["final"]
    for name, selected in split_rows.items():
        if not selected:
            raise ValueError(f"derived {name} split is empty")
    return split_rows, {"strategy": "dev facts shuffled with seed; 70/30 train/validation; final locked",
                        "seed": seed,
                        "groups": {name: len({_cluster_key(r) for r in selected})
                                   for name, selected in split_rows.items()}}


def prepare_splits(rows, seed=0):
    """Validate split assignments and return train/validation/test rows plus split metadata."""
    labels = {str(r.get("split", "")).lower() for r in rows}
    if labels == {"dev", "final"}:
        return _split_development_facts(rows, seed)
    unknown = labels - set(CANONICAL_SPLITS)
    if unknown:
        raise ValueError(f"unsupported or missing split labels: {sorted(unknown)}")
    selected = {name: [] for name in ("train", "validation", "test")}
    group_assignments = defaultdict(set)
    for row in rows:
        canonical = CANONICAL_SPLITS[str(row.get("split", "")).lower()]
        selected[canonical].append(row)
        group_assignments[_cluster_key(row)].add(canonical)
    leaked = [str(group) for group, assigned in group_assignments.items() if len(assigned) > 1]
    if leaked:
        raise ValueError(f"{len(leaked)} question/fact groups cross data splits; first: {leaked[:5]}")
    empty = [name for name, values in selected.items() if not values]
    if empty:
        raise ValueError(f"required data split(s) are empty: {empty}")
    return selected, {"strategy": "explicit split labels; question/fact group integrity checked",
                      "groups": {name: len({_cluster_key(r) for r in values})
                                 for name, values in selected.items()}}


def _greedy_question_rows(responses):
    """Join sampled-response consistency onto one greedy row per question."""
    if not responses:
        raise ValueError("response file is empty")
    for field in ("outcome", "is_greedy", "question_id", "split", "sample_index", "model_answer"):
        if any(field not in row for row in responses):
            raise ValueError(f"every response must contain {field!r}")
    for field in ("run_id", "checkpoint_id", "decoding_config_id"):
        values = {row.get(field) for row in responses}
        if len(values) > 1:
            raise ValueError(f"input contains multiple {field} values; calibrate one run at a time")
    sample_indices = defaultdict(set)
    for row in responses:
        if not isinstance(row["is_greedy"], bool):
            raise ValueError(f"is_greedy must be boolean for question {row['question_id']}")
        if not row["is_greedy"]:
            index = row["sample_index"]
            key = (row.get("run_id"), row.get("checkpoint_id"), row["question_id"])
            if index is None or index in sample_indices[key]:
                raise ValueError(f"missing or duplicate sample_index for question {row['question_id']}")
            sample_indices[key].add(index)
    summaries = summarize_questions(responses)
    summary_by_key = {tuple(summary[k] for k in GROUP_KEYS): summary for summary in summaries}
    question_metadata = defaultdict(lambda: {"splits": set(), "facts": set()})
    for row in responses:
        key = (row.get("run_id"), row.get("checkpoint_id"), row["question_id"])
        question_metadata[key]["splits"].add(str(row["split"]).lower())
        if row.get("fact_id") is not None:
            question_metadata[key]["facts"].add(row["fact_id"])
    inconsistent = [key for key, meta in question_metadata.items()
                    if len(meta["splits"]) > 1 or len(meta["facts"]) > 1]
    if inconsistent:
        raise ValueError(f"response rows disagree on split or fact_id for questions: {inconsistent[:5]}")
    greedy = []
    for original in responses:
        if not original.get("is_greedy"):
            continue
        row = dict(original)
        summary = summary_by_key[tuple(row.get(k) for k in GROUP_KEYS)]
        row.update({k: v for k, v in summary.items() if k not in GROUP_KEYS})
        greedy.append(row)
    seen = set()
    for row in greedy:
        key = (row.get("run_id"), row.get("checkpoint_id"), row["question_id"])
        if key in seen:
            raise ValueError(f"multiple greedy responses for question {key}")
        seen.add(key)
        if row["outcome"] not in {CORRECT, INCORRECT, ABSTAIN, INVALID}:
            raise ValueError(f"unknown outcome {row['outcome']!r} for question {row['question_id']}")
    return greedy


def _eligible(rows, signal):
    output = []
    for row in rows:
        if row.get("outcome") not in ATTEMPTED_OUTCOMES:
            continue
        raw = row.get(signal)
        score = oriented(signal, raw)
        if score is None or not np.isfinite(score):
            continue
        output.append((row, float(score), int(row["outcome"] == CORRECT)))
    return output


def _fit_calibrator(train, signal):
    eligible = _eligible(train, signal)
    labels = [label for _, _, label in eligible]
    if len(set(labels)) < 2:
        return None, {"status": "unavailable", "reason": "training split must contain both outcomes",
                      "n": len(eligible), "n_correct": sum(labels), "n_incorrect": len(labels) - sum(labels)}
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))
    x = np.asarray([[score] for _, score, _ in eligible], dtype=float)
    model.fit(x, labels)
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    parameters = {
        "mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0]),
        "coefficient": float(logistic.coef_[0][0]), "intercept": float(logistic.intercept_[0]),
        "formula": "sigmoid(coefficient * ((oriented_score - mean) / scale) + intercept)",
    }
    return model, {"status": "fitted", "method": "Platt scaling (standardized 1D logistic regression)",
                   "parameters": parameters,
                   "n": len(eligible), "n_correct": sum(labels), "n_incorrect": len(labels) - sum(labels)}


def _predict(model, rows, signal):
    eligible = _eligible(rows, signal)
    if not eligible:
        return {}
    x = np.asarray([[score] for _, score, _ in eligible], dtype=float)
    probabilities = model.predict_proba(x)[:, 1]
    return {id(row): float(probability) for (row, _, _), probability in zip(eligible, probabilities)}


def _metric_rows(rows, probabilities):
    return [dict(row, calibrated_probability=probabilities[id(row)])
            for row in rows if id(row) in probabilities]


def _calibration_metrics(rows):
    probs = [row["calibrated_probability"] for row in rows]
    labels = [int(row["outcome"] == CORRECT) for row in rows]
    return {"n": len(rows), "n_correct": sum(labels), "n_incorrect": len(labels) - sum(labels),
            "brier": brier_score(probs, labels) if probs else None,
            "ece_10_bins": expected_calibration_error(probs, labels, n_bins=10),
            "reliability_bins": reliability_bins(probs, labels, n_bins=10) if probs else []}


def _stratified_metrics(rows, probabilities):
    """Descriptive test slices; sample K is independent of the greedy response target."""
    groups = defaultdict(list)
    for row in rows:
        exposure = row.get("exposure_level")
        if exposure is not None:
            groups[("exposure_level", str(exposure))].append(row)
        score = row.get("knowledge_score")
        if score is not None:
            bucket = "K=0" if score == 0 else ("0<K<0.8" if score < 0.8 else "K>=0.8")
            groups[("sample_knowledge_score", bucket)].append(row)
    output = defaultdict(dict)
    for (dimension, label), members in groups.items():
        attempted = _metric_rows(members, probabilities)
        behavior = behavior_metrics([row["outcome"] for row in members])
        output[dimension][label] = {
            "n_questions": len(members), "behavior": behavior,
            "n_calibrated_attempts": len(attempted),
            "brier": brier_score([r["calibrated_probability"] for r in attempted],
                                  [int(r["outcome"] == CORRECT) for r in attempted]) if attempted else None,
            "ece_10_bins": expected_calibration_error(
                [r["calibrated_probability"] for r in attempted],
                [int(r["outcome"] == CORRECT) for r in attempted], n_bins=10) if attempted else None,
        }
    return dict(output)


def _best_threshold(rows, probabilities, target_coverage):
    """Choose the most conservative threshold that still reaches validation coverage target."""
    outcomes = [row["outcome"] for row in rows]
    # -1 marks an attempted answer with no usable score; every valid probability is in [0, 1],
    # so any non-negative threshold conservatively abstains on a missing confidence estimate.
    confidence = [probabilities.get(id(row), -1.0) for row in rows]
    max_coverage = behavior_metrics(outcomes)["coverage"]
    if max_coverage is None or max_coverage == 0:
        return None, {"status": "unavailable", "reason": "no substantive answers in validation split"}
    requested = min(float(target_coverage), max_coverage)
    candidates = sorted({0.0, *[confidence[i] for i, outcome in enumerate(outcomes)
                               if outcome in ATTEMPTED_OUTCOMES]})
    feasible = []
    for threshold in candidates:
        kept = apply_threshold(outcomes, confidence, threshold)
        coverage = behavior_metrics(kept)["coverage"]
        if coverage is not None and coverage + 1e-12 >= requested:
            feasible.append((threshold, coverage))
    if not feasible:
        return 0.0, {"status": "selected", "requested_coverage": target_coverage,
                      "attainable_coverage": max_coverage, "validation_coverage": max_coverage}
    threshold, coverage = max(feasible, key=lambda item: item[0])
    return float(threshold), {"status": "selected", "requested_coverage": target_coverage,
                              "effective_target_coverage": requested,
                              "attainable_coverage": max_coverage, "validation_coverage": coverage}


def evaluate_calibration(responses, seed=0, target_coverage=0.8, selected_signals=None):
    """Fit calibrators on train; select signal/threshold on validation; evaluate locked test.

    Returns a JSON-serializable summary and long-form probability rows for downstream merging.
    """
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    questions = _greedy_question_rows(responses)
    splits, split_metadata = prepare_splits(questions, seed)
    signals = list(selected_signals or SIGNAL_DIRECTIONS)
    unknown = set(signals) - set(SIGNAL_DIRECTIONS)
    if unknown:
        raise ValueError(f"unsupported confidence signal(s): {sorted(unknown)}")

    results, predictions = {}, []
    for signal in signals:
        model, fit_meta = _fit_calibrator(splits["train"], signal)
        result = {"signal": signal, "confident_when": "higher",
                  "training": fit_meta, "validation": None, "test": None, "threshold": None}
        if model is None:
            results[signal] = result
            continue
        probs = {name: _predict(model, values, signal) for name, values in splits.items()}
        val_rows = _metric_rows(splits["validation"], probs["validation"])
        test_rows = _metric_rows(splits["test"], probs["test"])
        if not val_rows:
            result["training"] = {**fit_meta, "status": "unavailable",
                                  "reason": "validation split has no usable signal values"}
            results[signal] = result
            continue
        result["validation"] = _calibration_metrics(val_rows)
        threshold, threshold_meta = _best_threshold(splits["validation"], probs["validation"], target_coverage)
        result["threshold"] = {"probability_correct": threshold, **threshold_meta}
        if test_rows:
            test_metrics = _calibration_metrics(test_rows)
            cluster_key = "fact_id" if all(row.get("fact_id") for row in test_rows) else "question_id"
            oriented_scores = [oriented(signal, row.get(signal)) for row in test_rows]
            test_labels = [int(row["outcome"] == CORRECT) for row in test_rows]
            roc, n_positive, n_negative = common_auroc(oriented_scores, test_labels)
            test_metrics["discrimination"] = {
                "auroc": roc, "auprc": common_auprc(oriented_scores, test_labels),
                "n_correct": n_positive, "n_incorrect": n_negative,
            }
            test_metrics["stratified"] = _stratified_metrics(splits["test"], probs["test"])
            for metric_name, metric_fn in (
                ("auroc", lambda rs: common_auroc(
                    [oriented(signal, r.get(signal)) for r in rs],
                    [int(r["outcome"] == CORRECT) for r in rs])[0]),
                ("auprc", lambda rs: common_auprc(
                    [oriented(signal, r.get(signal)) for r in rs],
                    [int(r["outcome"] == CORRECT) for r in rs])),
                ("brier", lambda rs: brier_score([r["calibrated_probability"] for r in rs],
                                                  [int(r["outcome"] == CORRECT) for r in rs])),
                ("ece_10_bins", lambda rs: expected_calibration_error(
                    [r["calibrated_probability"] for r in rs],
                    [int(r["outcome"] == CORRECT) for r in rs], n_bins=10)),
            ):
                ci = bootstrap_ci(test_rows, metric_fn, cluster_key, n_boot=1000, seed=seed)
                test_metrics[metric_name + "_ci_95"] = {
                    "low": ci["ci_low"], "high": ci["ci_high"],
                    "n_clusters": ci["n_clusters"], "replicates": ci["n_boot"],
                    "undefined_replicates": ci["n_undefined"]}
            confidence = [probs["test"].get(id(row), -1.0) for row in splits["test"]]
            if threshold is None:
                test_metrics["policy"] = {"status": "unavailable",
                                           "reason": "validation split has no usable attempted answers"}
            else:
                thresholded = apply_threshold([row["outcome"] for row in splits["test"]],
                                              confidence, threshold)
                test_metrics["policy"] = {
                    "target_coverage_selected_on": "validation",
                    "behavior": behavior_metrics(thresholded),
                    "confident_hallucinations": confident_hallucinations(
                        [row["outcome"] for row in splits["test"]], confidence, threshold),
                    "risk_coverage_curve": risk_coverage_curve(
                        [row["outcome"] for row in splits["test"]], confidence),
                }
            result["test"] = test_metrics
        else:
            result["test"] = {"n": 0, "status": "no correct/incorrect rows with this signal"}

        for split_name, split_rows in splits.items():
            for row in split_rows:
                probability = probs[split_name].get(id(row))
                if probability is None:
                    continue
                predictions.append({
                    "run_id": row.get("run_id"), "question_id": row["question_id"],
                    "checkpoint_id": row.get("checkpoint_id"), "method": row.get("method"),
                    "model_id": row.get("model_id"), "training_seed": row.get("training_seed"),
                    "source_dataset": row.get("source_dataset"),
                    "dataset_version": row.get("dataset_version"),
                    "decoding_config_id": row.get("decoding_config_id"),
                    "fact_id": row.get("fact_id"), "split": split_name,
                    "exposure_level": row.get("exposure_level"),
                    "outcome": row["outcome"], "signal": signal,
                    "raw_score": row.get(signal), "probability_correct": probability,
                })
        results[signal] = result

    valid = [(name, value) for name, value in results.items()
             if value["validation"] is not None and value["validation"].get("brier") is not None]
    selected = min(valid, key=lambda item: (item[1]["validation"]["brier"], item[0]))[0] if valid else None
    summary = {
        "protocol": {
            "target": "greedy substantive answer correctness",
            "probability_method": "one-dimensional Platt scaling per signal",
            "fit_split": "train", "selection_split": "validation", "evaluation_split": "test",
            "threshold_target_coverage": target_coverage,
            "reliability_bins": 10,
            "bootstrap": {"method": "percentile cluster bootstrap", "confidence_level": 0.95,
                          "replicates": 1000, "seed": seed,
                          "cluster": "fact_id when available, otherwise question_id"},
            "test_touched_for_selection": False,
            "excluded_from_probability_metrics": ["abstain", "invalid", "missing signal"],
            "missing_signal_policy": "missing confidence is treated as unavailable and abstained under non-negative thresholds",
            "behavior_policy_denominator": "all valid greedy responses, including abstentions",
        },
        "split_assignment": split_metadata,
        "run_metadata": {field: questions[0].get(field) for field in (
            "run_id", "model_id", "checkpoint_id", "method", "training_seed", "source_dataset",
            "dataset_version", "prompt_version", "grading_version", "decoding_config_id")},
        "split_counts": {name: {"n_questions": len(values),
                                 "outcomes": {outcome: sum(r["outcome"] == outcome for r in values)
                                              for outcome in (CORRECT, INCORRECT, ABSTAIN, INVALID)}}
                         for name, values in splits.items()},
        "selected_signal_on_validation": selected,
        "signals": results,
    }
    return summary, predictions
