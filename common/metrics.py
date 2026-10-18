"""Shared evaluation metrics (plan.md section 6). Standard library only.

Outcomes: "correct" (C), "incorrect" (W, substantive wrong answer), "abstain" (A),
"invalid" (technical failure; excluded from N and reported separately).
N = C + W + A. Every rate function returns None where plan.md says "N/A".
"""

import math
import random
from collections import Counter, defaultdict

CORRECT, INCORRECT, ABSTAIN, INVALID = "correct", "incorrect", "abstain", "invalid"
OUTCOMES = (CORRECT, INCORRECT, ABSTAIN, INVALID)


def _div(num, den):
    return num / den if den else None


def outcome_counts(outcomes):
    counts = Counter(outcomes)
    unknown = set(counts) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"unknown outcomes: {sorted(unknown)}")
    c, w, a = counts[CORRECT], counts[INCORRECT], counts[ABSTAIN]
    return {"C": c, "W": w, "A": a, "N": c + w + a, "invalid": counts[INVALID]}


def behavior_metrics(outcomes):
    """Accuracy, hallucination, abstention, coverage, selective accuracy/risk."""
    k = outcome_counts(outcomes)
    c, w, a, n = k["C"], k["W"], k["A"], k["N"]
    return {
        **k,
        "accuracy": _div(c, n),
        "hallucination_rate": _div(w, n),
        "abstention_rate": _div(a, n),
        "coverage": _div(c + w, n),
        "selective_accuracy": _div(c, c + w),
        "selective_risk": _div(w, c + w),
    }


def knowledge_scores(rows, key=("run_id", "checkpoint_id", "question_id")):
    """K(q) = correct sampled responses / sampled responses, per question.

    Greedy responses are excluded. Invalid samples are excluded from the denominator;
    `n_samples` records how many samples actually counted.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["is_greedy"] or r["outcome"] == INVALID:
            continue
        groups[tuple(r[k] for k in key)].append(r["outcome"] == CORRECT)
    return {
        g: {"knowledge_score": sum(v) / len(v), "n_samples": len(v)}
        for g, v in groups.items()
    }


def auroc(scores, labels):
    """P(score of a random positive > score of a random negative); ties count half.

    Returns (value, n_pos, n_neg); value is None if only one class is present.
    """
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, n_pos, n_neg
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1
    pos_rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    value = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return value, n_pos, n_neg


def auprc(scores, labels):
    """Average precision (step-wise, ties grouped). None if there are no positives."""
    n_pos = sum(1 for y in labels if y)
    if n_pos == 0:
        return None
    by_score = defaultdict(lambda: [0, 0])
    for s, y in zip(scores, labels):
        by_score[s][0 if y else 1] += 1
    tp = fp = 0
    ap = 0.0
    for s in sorted(by_score, reverse=True):
        pos, neg = by_score[s]
        tp += pos
        fp += neg
        ap += (pos / n_pos) * (tp / (tp + fp))
    return ap


def brier_score(probs, labels):
    _check_probs(probs)
    return _div(sum((p - float(y)) ** 2 for p, y in zip(probs, labels)), len(probs))


def reliability_bins(probs, labels, n_bins=10):
    """Equal-width bins on [0, 1]; p == 1.0 goes in the last bin. Empty bins are kept."""
    _check_probs(probs)
    bins = [{"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": 0, "sum_p": 0.0, "sum_y": 0.0}
            for b in range(n_bins)]
    for p, y in zip(probs, labels):
        b = bins[min(int(p * n_bins), n_bins - 1)]
        b["n"] += 1
        b["sum_p"] += p
        b["sum_y"] += float(y)
    for b in bins:
        b["mean_confidence"] = _div(b.pop("sum_p"), b["n"])
        b["accuracy"] = _div(b.pop("sum_y"), b["n"])
    return bins


def expected_calibration_error(probs, labels, n_bins=10):
    total = len(probs)
    if not total:
        return None
    return sum(
        b["n"] / total * abs(b["accuracy"] - b["mean_confidence"])
        for b in reliability_bins(probs, labels, n_bins) if b["n"]
    )


def _check_probs(probs):
    bad = [p for p in probs if not 0.0 <= p <= 1.0]
    if bad:
        raise ValueError("calibration metrics need probabilities in [0, 1] "
                         f"(map raw scores first); got e.g. {bad[:3]}")


def apply_threshold(outcomes, confidences, threshold):
    """Inference policy: attempted answers with confidence < threshold become abstentions.

    Existing abstentions and invalid rows are unchanged. This is a policy, not a training
    effect (plan.md section 7) — apply the same policy to every compared method.
    """
    return [
        ABSTAIN if o in (CORRECT, INCORRECT) and conf < threshold else o
        for o, conf in zip(outcomes, confidences)
    ]


def confident_hallucinations(outcomes, confidences, threshold):
    """Incorrect answers with confidence >= a frozen, validation-selected threshold."""
    k = outcome_counts(outcomes)
    count = sum(1 for o, c in zip(outcomes, confidences) if o == INCORRECT and c >= threshold)
    kept = apply_threshold(outcomes, confidences, threshold)
    return {
        "threshold": threshold,
        "count": count,
        "rate": _div(count, k["N"]),
        "coverage": behavior_metrics(kept)["coverage"],
    }


def risk_coverage_curve(outcomes, confidences):
    """One point per distinct confidence threshold among attempted answers.

    Coverage is over all N valid responses; risk is over answers kept by the threshold.
    Points are ordered from highest to lowest coverage.
    """
    k = outcome_counts(outcomes)
    attempted = sorted(
        ((c, o) for o, c in zip(outcomes, confidences) if o in (CORRECT, INCORRECT)),
        key=lambda t: -t[0],
    )
    points = []
    kept_c = kept_w = 0
    i = 0
    while i < len(attempted):
        thr = attempted[i][0]
        while i < len(attempted) and attempted[i][0] == thr:
            if attempted[i][1] == CORRECT:
                kept_c += 1
            else:
                kept_w += 1
            i += 1
        points.append({
            "threshold": thr,
            "coverage": _div(kept_c + kept_w, k["N"]),
            "selective_risk": kept_w / (kept_c + kept_w),
            "selective_accuracy": kept_c / (kept_c + kept_w),
        })
    return points[::-1]


def area_under_risk_coverage(points):
    """Trapezoidal area under selective risk vs coverage (lower is better)."""
    if not points:
        return None
    pts = sorted((p["coverage"], p["selective_risk"]) for p in points)
    area = pts[0][0] * pts[0][1]  # risk held flat from coverage 0 to the first point
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def risk_at_coverage(points, target):
    """Lowest selective risk among thresholds reaching at least `target` coverage."""
    ok = [p["selective_risk"] for p in points if p["coverage"] is not None and p["coverage"] >= target]
    return min(ok) if ok else None


def bootstrap_ci(rows, metric_fn, cluster_key, n_boot=1000, alpha=0.05, seed=0):
    """Percentile CI resampling whole clusters (question_id for real QA, fact_id for synthetic).

    `metric_fn(rows) -> float | None`. Resamples where the metric is None are dropped and
    counted in `n_undefined`.
    """
    clusters = defaultdict(list)
    for r in rows:
        clusters[r[cluster_key]].append(r)
    ids = sorted(clusters)
    rng = random.Random(seed)
    stats, undefined = [], 0
    for _ in range(n_boot):
        sample = [r for cid in rng.choices(ids, k=len(ids)) for r in clusters[cid]]
        v = metric_fn(sample)
        if v is None:
            undefined += 1
        else:
            stats.append(v)
    return {"estimate": metric_fn(rows), **_percentiles(stats, alpha),
            "n_boot": n_boot, "n_undefined": undefined, "n_clusters": len(ids)}


def paired_bootstrap_diff(rows_a, rows_b, metric_fn, cluster_key, n_boot=1000, alpha=0.05, seed=0):
    """CI for metric(A) - metric(B) on the same clusters, resampled jointly.

    Raises if the two methods were not evaluated on identical cluster IDs.
    """
    ca, cb = defaultdict(list), defaultdict(list)
    for r in rows_a:
        ca[r[cluster_key]].append(r)
    for r in rows_b:
        cb[r[cluster_key]].append(r)
    if set(ca) != set(cb):
        raise ValueError(f"cluster IDs differ: {len(set(ca) ^ set(cb))} mismatched")
    ids = sorted(ca)
    rng = random.Random(seed)
    stats, undefined = [], 0
    for _ in range(n_boot):
        pick = rng.choices(ids, k=len(ids))
        va = metric_fn([r for c in pick for r in ca[c]])
        vb = metric_fn([r for c in pick for r in cb[c]])
        if va is None or vb is None:
            undefined += 1
        else:
            stats.append(va - vb)
    ea, eb = metric_fn(rows_a), metric_fn(rows_b)
    est = None if ea is None or eb is None else ea - eb
    return {"estimate": est, **_percentiles(stats, alpha),
            "n_boot": n_boot, "n_undefined": undefined, "n_clusters": len(ids)}


def _percentiles(values, alpha):
    if not values:
        return {"ci_low": None, "ci_high": None}
    v = sorted(values)

    def q(p):
        pos = p * (len(v) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return {"ci_low": q(alpha / 2), "ci_high": q(1 - alpha / 2)}
