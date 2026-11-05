"""Leakage-safe probability calibration: fit on train, select on validation, never read held-out rows.

Calibration targets the correctness of a greedy, substantive answer. Raw signals are mapped to
P(correct) with one-dimensional Platt scaling fitted on calibration-training rows only. Signal
selection and abstention thresholds use validation rows. Held-out rows are removed by split label
before any outcome or signal is read, so they cannot influence a choice made here.
"""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common.metrics import CORRECT, INCORRECT, INVALID, behavior_metrics

from .consistency import GROUP_KEYS, summarize_questions
from .signals import oriented

ATTEMPTED_OUTCOMES = {CORRECT, INCORRECT}
SPLIT_ROLES = {"cal_train": "train", "train_probe": "train", "train": "train",
               "cal_val": "validation", "validation": "validation", "val": "validation",
               "dev": "development",
               "cal_test": "test", "locked_test": "test", "test": "test", "final": "test"}
TRAIN_SHARE_OF_DEVELOPMENT = 0.7
# A confidence below every probability: an attempted answer without a usable score is never kept.
MISSING_CONFIDENCE = -1.0


def cluster_key(row):
    """Unit of independence: paraphrases of one fact are not separate observations."""
    return row.get("fact_id") or row["question_id"]


def split_role(row):
    label = str(row["split"]).lower()
    if label not in SPLIT_ROLES:
        raise ValueError(f"unsupported split label {row['split']!r} for question {row['question_id']}")
    return SPLIT_ROLES[label]


def _check_groups_stay_in_one_role(responses):
    """Identifiers only: this must stay safe to run on rows whose outcomes may not be read."""
    roles = defaultdict(set)
    for row in responses:
        roles[cluster_key(row)].add(split_role(row))
    leaked = sorted(str(group) for group, assigned in roles.items() if len(assigned) > 1)
    if leaked:
        raise ValueError(f"{len(leaked)} question/fact groups cross data splits; first: {leaked[:5]}")


def greedy_question_rows(responses):
    """Join sampled-response consistency onto one greedy row per question."""
    seen_samples = set()
    for row in responses:
        if not isinstance(row["is_greedy"], bool):
            raise ValueError(f"is_greedy must be boolean for question {row['question_id']}")
        if not row["is_greedy"]:
            key = (row["question_id"], row["sample_index"])
            if row["sample_index"] is None or key in seen_samples:
                raise ValueError(f"missing or duplicate sample_index for question {row['question_id']}")
            seen_samples.add(key)
    summary_by_key = {tuple(s[k] for k in GROUP_KEYS): s for s in summarize_questions(responses)}
    greedy, seen = [], set()
    for original in responses:
        if not original["is_greedy"]:
            continue
        if original["question_id"] in seen:
            raise ValueError(f"multiple greedy responses for question {original['question_id']}")
        seen.add(original["question_id"])
        summary = summary_by_key[tuple(original.get(k) for k in GROUP_KEYS)]
        greedy.append({**original, **{k: v for k, v in summary.items() if k not in GROUP_KEYS}})
    return greedy


def _split_development_facts(questions, seed):
    """P1's dev rows have no separate validation set: assign whole facts 70/30 within each exposure level."""
    if any(not row.get("fact_id") for row in questions):
        raise ValueError("dev data must include fact_id so paraphrases cannot cross splits")
    levels = defaultdict(set)
    for row in questions:
        levels[row.get("exposure_level")].add(row["fact_id"])
    rng = np.random.default_rng(seed)
    train_facts = set()
    for level in sorted(levels, key=str):
        facts = sorted(levels[level])
        order = rng.permutation(len(facts))
        n_train = round(TRAIN_SHARE_OF_DEVELOPMENT * len(facts))
        train_facts.update(facts[int(i)] for i in order[:n_train])
    splits = {"train": [r for r in questions if r["fact_id"] in train_facts],
              "validation": [r for r in questions if r["fact_id"] not in train_facts]}
    return splits, "dev facts assigned 70/30 to train/validation within each exposure level"


def development_splits(responses, seed=0):
    """Return ({train, validation} question rows, metadata). Held-out rows are dropped unread."""
    _check_groups_stay_in_one_role(responses)
    development = [row for row in responses if split_role(row) != "test"]
    roles = {split_role(row) for row in development}
    if "development" in roles and roles != {"development"}:
        raise ValueError("dev rows cannot be combined with explicit train/validation rows")
    questions = greedy_question_rows(development)
    if roles == {"development"}:
        splits, strategy = _split_development_facts(questions, seed)
    else:
        splits = {"train": [r for r in questions if split_role(r) == "train"],
                  "validation": [r for r in questions if split_role(r) == "validation"]}
        strategy = "explicit split labels"
    empty = [name for name, rows in splits.items() if not rows]
    if empty:
        raise ValueError(f"required data split(s) are empty: {empty}")
    metadata = {"strategy": strategy, "seed": seed,
                "groups": {name: len({cluster_key(r) for r in rows}) for name, rows in splits.items()},
                "held_out_rows_ignored": len(responses) - len(development)}
    return splits, metadata


def outcome_label(row):
    return int(row["outcome"] == CORRECT)


def attempted_with_signal(rows, signal):
    """Attempted answers that have a usable value for `signal`."""
    usable = []
    for row in rows:
        score = oriented(signal, row.get(signal))
        if row["outcome"] in ATTEMPTED_OUTCOMES and score is not None and math.isfinite(score):
            usable.append(row)
    return usable


@dataclass(frozen=True)
class PlattCalibrator:
    """sigmoid(coefficient * ((oriented_score - mean) / scale) + intercept), portable as plain numbers."""
    signal: str
    mean: float
    scale: float
    coefficient: float
    intercept: float

    def probability(self, raw_value):
        score = oriented(self.signal, raw_value)
        if score is None or not math.isfinite(score):
            return None
        logit = self.coefficient * ((score - self.mean) / self.scale) + self.intercept
        return 1.0 / (1.0 + math.exp(-logit))

    def to_dict(self):
        return asdict(self)


def fit_calibrator(train_rows, signal):
    """Returns (calibrator or None, metadata). A signal the producer did not record is unavailable, not an error."""
    usable = attempted_with_signal(train_rows, signal)
    labels = [outcome_label(r) for r in usable]
    counts = {"n": len(usable), "n_correct": sum(labels), "n_incorrect": len(labels) - sum(labels)}
    if len(set(labels)) < 2:
        return None, {"status": "unavailable", "reason": "training split must contain both outcomes", **counts}
    x = np.asarray([[oriented(signal, r[signal])] for r in usable], dtype=float)
    scaler = StandardScaler().fit(x)
    logistic = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(scaler.transform(x), labels)
    calibrator = PlattCalibrator(signal, float(scaler.mean_[0]), float(scaler.scale_[0]),
                                 float(logistic.coef_[0][0]), float(logistic.intercept_[0]))
    return calibrator, {"status": "fitted", "method": "Platt scaling (standardized 1D logistic regression)", **counts}


def confidences(rows, calibrator):
    """P(correct) per row, or MISSING_CONFIDENCE so a thresholded policy abstains on an unscored answer."""
    values = [calibrator.probability(row.get(calibrator.signal)) for row in rows]
    return [MISSING_CONFIDENCE if value is None else value for value in values]


def select_threshold(outcomes, confidence, target_coverage):
    """Highest threshold whose coverage still reaches the target (or the most that is attainable)."""
    attainable = behavior_metrics(outcomes)["coverage"]
    if not attainable:
        return None
    scored = sorted((c for o, c in zip(outcomes, confidence)
                     if o in ATTEMPTED_OUTCOMES and c != MISSING_CONFIDENCE), reverse=True)
    if not scored:
        return None
    n_valid = sum(o != INVALID for o in outcomes)
    requested = min(target_coverage, attainable)
    n_keep = min(len(scored), max(1, math.ceil(requested * n_valid - 1e-9)))
    threshold = scored[n_keep - 1]
    kept = sum(1 for o, c in zip(outcomes, confidence) if o in ATTEMPTED_OUTCOMES and c >= threshold)
    return {"probability_correct": threshold, "requested_coverage": target_coverage,
            "attainable_coverage": attainable, "selection_coverage": kept / n_valid}
