"""Behaviour and discrimination metrics (pure numpy). Definitions follow plan.md §6.

Undefined quantities are returned as None (reported as N/A), never silently as 0 or 0.5.
"""

import numpy as np

OUTCOMES = ("correct", "incorrect", "abstain", "invalid")


def _ratio(num, den):
    return num / den if den else None


def outcome_metrics(outcomes):
    """N counts valid responses only (correct + incorrect + abstain); invalid is reported separately."""
    counts = {o: 0 for o in OUTCOMES}
    for o in outcomes:
        counts[o] += 1
    c, w, a = counts["correct"], counts["incorrect"], counts["abstain"]
    n = c + w + a
    return {
        "N": n, "C": c, "W": w, "A": a, "invalid": counts["invalid"],
        "accuracy": _ratio(c, n),
        "hallucination_rate": _ratio(w, n),
        "abstention_rate": _ratio(a, n),
        "coverage": _ratio(c + w, n),
        "selective_accuracy": _ratio(c, c + w),
        "selective_risk": _ratio(w, c + w),
    }


def _clean(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    keep = ~np.isnan(scores)
    return scores[keep], labels[keep]


def auroc(scores, labels):
    """Probability a random positive outranks a random negative (ties count half).

    Mann-Whitney U with average ranks. None if either class is absent.
    """
    s, y = _clean(scores, labels)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auprc(scores, labels):
    """Average precision for the positive class (step-wise, sklearn convention). None if no positives."""
    s, y = _clean(scores, labels)
    if y.sum() == 0 or len(y) == 0:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def bootstrap_ci(metric, scores, labels, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI, resampling units (one row per question/fact). Skips resamples where metric is None."""
    s, y = _clean(scores, labels)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        v = metric(s[idx], y[idx])
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None, None
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
