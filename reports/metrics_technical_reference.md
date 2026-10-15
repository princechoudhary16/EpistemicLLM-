# `common/metrics.py` — full technical reference

Complete statistical/ML documentation of every function, with formal definitions,
algorithmic complexity, and edge-case semantics. Companion to
`p6_code_explained_line_by_line.md` (plain-English line-by-line) and
`metrics_explained_no_jargon.md` (analogy-only, zero prerequisites). This one assumes
familiarity with basic probability, classification metrics, and Python.

## Outcome taxonomy and notation

Four mutually exclusive, jointly exhaustive outcome labels per response:
`CORRECT`, `INCORRECT`, `ABSTAIN`, `INVALID`. `INVALID` denotes a technical failure
(malformed/empty/timeout) rather than a semantic judgment on the answer, and is excluded
from `N`. Throughout, `C`, `W`, `A` denote counts of correct, incorrect (a substantive
wrong answer — "hallucination" in this project's operational sense, not a claim about
internal cause), and abstaining responses; `N = C + W + A`.

```python
def _div(num, den):
    return num / den if den else None
```
Total-safe division. Returns `None` (Python's null) rather than raising
`ZeroDivisionError` or fabricating `0.0`/`NaN`, matching `plan.md`'s explicit rule that
undefined ratios must surface as N/A, not as a number that looks meaningful but isn't.
Note `den` is falsy for `0`, `0.0`, and `None` alike, which is the intended coverage.

```python
def outcome_counts(outcomes):
    counts = Counter(outcomes)
    unknown = set(counts) - set(OUTCOMES)
    if unknown:
        raise ValueError(...)
    ...
```
`Counter` is an O(n) tally over the four-symbol alphabet. The `set` difference is a
closed-world validation: any label outside `{correct, incorrect, abstain, invalid}`
(e.g. a typo, or an un-migrated older schema value) raises immediately rather than being
silently coerced to zero-count, which would corrupt every downstream rate without any
visible symptom.

## Behavioral metrics

```python
def behavior_metrics(outcomes):
    ...
    "accuracy": _div(c, n),
    "hallucination_rate": _div(w, n),
    "abstention_rate": _div(a, n),
    "coverage": _div(c + w, n),
    "selective_accuracy": _div(c, c + w),
    "selective_risk": _div(w, c + w),
```

Standard definitions from the **selective classification** literature (a classifier
paired with a reject option):

- **Accuracy** `= C/N`, **hallucination rate** `= W/N`, **abstention rate** `= A/N`.
  These three sum to exactly 1 by construction (`C+W+A = N`), so any one is
  redundant given the other two — all three are still reported because each has an
  independent, direct interpretation stakeholders care about.
- **Coverage** `= (C+W)/N`: the fraction of the domain the model elects to make a
  prediction on — the standard selective-classification "coverage" term, `φ` in most
  papers' notation.
- **Selective accuracy** `= C/(C+W)`, i.e. accuracy conditioned on non-abstention —
  `1 − selective risk`, where **selective risk** `= W/(C+W)` is the standard term for
  error rate restricted to the accepted subset. Both are undefined (returned as `None`)
  at coverage 0.

This is precisely the accuracy-coverage decomposition used to prevent the "reward
hacking via refusal" failure mode: coverage and (accuracy or selective accuracy) must be
read together, since either alone is gameable (accuracy by attempting only easy
questions and abstaining elsewhere; coverage alone says nothing about correctness).

## Knowledge score K(q)

```python
def knowledge_scores(rows, key=(...)):
    groups = defaultdict(list)
    for r in rows:
        if r["is_greedy"] or r["outcome"] == INVALID:
            continue
        groups[tuple(r[k] for k in key)].append(r["outcome"] == CORRECT)
    return {g: {"knowledge_score": sum(v)/len(v), "n_samples": len(v)} for g, v in groups.items()}
```

`K(q)` is the empirical estimate of `P(correct | q)` under the sampling
distribution induced by the decoding config (temperature/top-p), computed as a sample
mean over `m` i.i.d. draws per question: `K(q) = (1/m) Σ 1[response_i is correct]`.
Two exclusions matter for validity:

1. **Greedy exclusion:** the greedy (temperature-0, argmax) decode is a degenerate,
   non-random draw and is not exchangeable with the sampled draws — mixing it in would
   bias the estimator (effectively double-weighting the mode of the distribution).
2. **Invalid exclusion:** invalid generations are missing data, not failed trials;
   `n_samples` (the realized denominator, ≤ the nominal 5) is reported alongside so a
   score based on `n_samples=2` isn't silently treated with the same precision as one
   based on 5.

The grouping key is `(run_id, checkpoint_id, question_id)` by default — a composite key
because `question_id` alone is not unique across runs/checkpoints, and treating it as if
it were would silently pool knowledge estimates for the base model and a fine-tuned
checkpoint into one meaningless average.

**Known limitation (documented, not fixed here):** any analysis that both computes `K(q)`
from a sample and then evaluates a predictor's ability to predict `K(q)` from that *same*
sample is circular — `plan.md` §6 flags this explicitly and requires an independent
held-out generation for actual predictive claims. This module computes `K(q)`; it does not
enforce sample-independence for its consumers, which remains a discipline imposed at the
call site.

## Ranking/discrimination metrics

### AUROC

```python
def auroc(scores, labels):
    n_pos = sum(1 for y in labels if y); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return None, n_pos, n_neg
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ...  # average-rank tie handling
    value = (pos_rank_sum - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
```

Implements AUROC via the **rank-sum (Mann–Whitney U) identity**:

```
AUROC = U / (n_pos · n_neg),   U = R_pos − n_pos(n_pos+1)/2
```

where `R_pos` is the sum of the mid-ranks of the positive class in the pooled,
ascending sort of all scores. This is algebraically identical to the probabilistic
definition `AUROC = P(score(+) > score(−)) + 0.5·P(score(+) = score(−))` for a randomly
drawn positive/negative pair, but computed in `O(n log n)` via one sort rather than the
naive `O(n_pos · n_neg)` pairwise comparison.

**Tie handling:** the inner `while` loop finds runs of equal scores and assigns every
tied element the **average (mid-)rank** of that run (`(i+j)/2 + 1`, 1-indexed) —
standard mid-rank tie correction. This is what makes a single tied pos/neg pair
contribute exactly `0.5` to the U-statistic rather than being arbitrarily broken by sort
order, which would otherwise make the result depend on input order for tied scores —
non-determinism that would be silently unreproducible.

**Degenerate case:** `n_pos == 0` or `n_neg == 0` (a single-class sample) makes AUROC
mathematically undefined (there is no pair to compare) — returns `(None, n_pos, n_neg)`
rather than an arbitrary constant like 0.5, and always returns the class counts so a
caller can log *why* it's undefined rather than just failing silently downstream.

### AUPRC (Average Precision)

```python
def auprc(scores, labels):
    ...
    for s in sorted(by_score, reverse=True):
        pos, neg = by_score[s]
        tp += pos; fp += neg
        ap += (pos/n_pos) * (tp/(tp+fp))
```

Implements **Average Precision** as the step-function integral of precision over recall,
walking thresholds from highest score to lowest and accumulating
`Σ_k precision(k) · ΔRecall(k)`. Because `ΔRecall` contributed by a group of `pos`
positives at a tied score is `pos/n_pos`, and grouping by distinct score value (rather
than one point per example) is required for correctness under ties — the naive
per-example step function would otherwise assign precision-at-k using an arbitrary
tie-break order between same-scored positives and negatives, which is not what AP is
defined to measure. `by_score` groups all examples sharing an exact score before
computing that group's joint contribution to `tp`/`fp`, avoiding that bias.

AUPRC is reported as a companion to AUROC specifically for **severe class imbalance**
(`plan.md` §6): AUROC can remain deceptively high under extreme imbalance because it's
invariant to the positive/negative base rate, whereas AUPRC is base-rate–sensitive and
surfaces poor performance on a rare positive class that AUROC would mask.

## Calibration

```python
def brier_score(probs, labels):
    return _div(sum((p - float(y))**2 for p, y in zip(probs, labels)), len(probs))
```
The Brier score is the mean squared error between a stated probability and the realized
binary outcome: `BS = (1/N) Σ (p_i − y_i)²`, a strictly proper scoring rule (minimized in
expectation only by reporting one's true belief) that decomposes into
calibration + refinement (discrimination) terms — unlike accuracy, it penalizes both
miscalibration and lack of sharpness simultaneously.

```python
def reliability_bins(probs, labels, n_bins=10):
    bins = [{"lo": b/n_bins, "hi": (b+1)/n_bins, ...} for b in range(n_bins)]
    for p, y in zip(probs, labels):
        b = bins[min(int(p*n_bins), n_bins-1)]
        ...
```
Fixed-width binning of `[0, 1]` into `n_bins` equal intervals (not equal-frequency/
quantile bins — a deliberate simplicity choice; quantile binning is a documented
possible refinement, not implemented here). `min(int(p*n_bins), n_bins-1)` is the
standard boundary-inclusion fix: `p = 1.0` would otherwise index bin `n_bins` (out of
range), so it's clamped into the last bin, making the top bin closed on both ends while
every other bin is half-open `[lo, hi)`. Each bin accumulates count, mean predicted
confidence, and empirical accuracy (mean of `y` in-bin) — the standard inputs to a
**reliability diagram**.

```python
def expected_calibration_error(probs, labels, n_bins=10):
    return sum(b["n"]/total * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["n"])
```
ECE is the bin-count-weighted mean absolute gap between predicted confidence and
observed accuracy: `ECE = Σ_b (n_b/N) · |acc_b − conf_b|`. This is the `ℓ1`, binned
approximation to calibration error most commonly reported in the literature (as opposed
to Maximum Calibration Error, `max_b |acc_b − conf_b|`, which is not implemented here).
Empty bins are excluded from the sum via `if b["n"]` (their weight is 0 anyway, but this
also avoids reading `None` out of an empty bin's `accuracy`/`mean_confidence` fields).

```python
def _check_probs(probs):
    bad = [p for p in probs if not 0.0 <= p <= 1.0]
    if bad: raise ValueError(...)
```
Both calibration functions require inputs already mapped into `[0, 1]` — i.e. actual
probability estimates, not raw confidence scores like log-probability or negative
entropy (which are unbounded or differently scaled). This guard enforces that
precondition at the boundary rather than silently computing a nonsensical Brier score on
out-of-range values; `plan.md` §6 explicitly assigns the raw-score → probability mapping
to a calibration-training-split-fitted model (e.g. isotonic regression or a logistic
link), which is out of scope for this module and must happen upstream of these calls.

## Threshold policies and confident hallucinations

```python
def apply_threshold(outcomes, confidences, threshold):
    return [ABSTAIN if o in (CORRECT, INCORRECT) and conf < threshold else o
            for o, conf in zip(outcomes, confidences)]
```
Implements a **selective classifier's reject rule** as a pure function over already-
generated outcomes: any attempted answer (`CORRECT`/`INCORRECT`) with confidence below
`threshold` is remapped to `ABSTAIN`; existing `ABSTAIN`/`INVALID` rows pass through
unchanged (an abstention can't become "more abstained," and invalid rows aren't a
confidence question). This models an **inference-time policy**, deliberately separate
from any training-time intervention — `plan.md` §7's core methodological guardrail is that
this exact function must be applied identically to every compared method's confidence
scores, so that an apparent improvement can't actually be "we just filtered more," which
would be a benefit of the policy, not the training objective.

```python
def confident_hallucinations(outcomes, confidences, threshold):
    count = sum(1 for o, c in zip(...) if o == INCORRECT and c >= threshold)
    kept = apply_threshold(outcomes, confidences, threshold)
    return {"threshold": threshold, "count": count, "rate": _div(count, k["N"]),
            "coverage": behavior_metrics(kept)["coverage"]}
```
Operationalizes "confident hallucination" (`plan.md` §4's definition: an incorrect
answer accompanied by a strong confidence signal) as a simple count/rate above a single
frozen threshold, reported jointly with the coverage that same threshold would induce —
so a low confident-hallucination rate can be immediately checked against whether it was
bought by also reducing coverage substantially.

## Risk–coverage curve

```python
def risk_coverage_curve(outcomes, confidences):
    attempted = sorted(((c, o) for o, c in zip(...) if o in (CORRECT, INCORRECT)),
                        key=lambda t: -t[0])
    ...
    while i < len(attempted):
        thr = attempted[i][0]
        while i < len(attempted) and attempted[i][0] == thr:
            ...  # accumulate kept_c, kept_w
        points.append({"threshold": thr, "coverage": _div(kept_c+kept_w, k["N"]),
                       "selective_risk": kept_w/(kept_c+kept_w),
                       "selective_accuracy": kept_c/(kept_c+kept_w)})
    return points[::-1]
```
Constructs the full **selective-risk-vs-coverage curve** by sweeping the confidence
threshold through every distinct value observed, from highest to lowest, accumulating
cumulative kept-correct/kept-incorrect counts as the threshold relaxes and admits more
attempted answers. This is the standard tool from the selective prediction literature
(cf. Geifman & El-Yaniv) for characterizing a confidence signal's *entire* operating
range rather than one arbitrary cutoff. Ties (identical confidence values) are grouped
into a single point (the inner `while`) — the curve only advances at each *distinct*
score value, since a threshold strictly between two identical scores is not a meaningful
distinct operating point. `coverage` is computed over all `N` valid responses (so an
abstention-heavy method's curve correctly starts from a lower ceiling), while
`selective_risk`/`selective_accuracy` are conditioned on the *kept* (attempted-and-passed-
threshold) subset only. `points[::-1]` reverses to present highest-coverage-first, since
the accumulation loop naturally produces lowest-coverage (highest threshold) first.

```python
def area_under_risk_coverage(points):
    pts = sorted((p["coverage"], p["selective_risk"]) for p in points)
    area = pts[0][0] * pts[0][1]
    for (x0,y0), (x1,y1) in zip(pts, pts[1:]):
        area += (x1-x0) * (y0+y1) / 2
    return area
```
**AURC** (Area Under the Risk-Coverage curve), computed by trapezoidal quadrature over
the coverage-sorted points, lower being better (less risk at equivalent coverage). The
leading term `pts[0][0] * pts[0][1]` extends the curve down to coverage 0 by holding risk
constant at the lowest observed point's risk — a rectangle rule for the
`[0, first_coverage]` segment, since no data exists below the most restrictive observed
threshold. This is a documented modeling choice, not a derived value: an alternative
convention (e.g. assuming risk → 0 as coverage → 0) would give a different, generally
smaller, AURC. Note a subtlety this module deliberately does **not** paper over: even a
perfectly rank-ordered confidence signal (every correct answer scored above every
incorrect one) has strictly positive AURC once coverage is forced to 1, because full
coverage necessarily admits the incorrect answers too — AURC=0 is only achievable if
`W=0` outright, not merely from perfect ranking.

```python
def risk_at_coverage(points, target):
    ok = [p["selective_risk"] for p in points if p["coverage"] is not None and p["coverage"] >= target]
    return min(ok) if ok else None
```
The pointwise inverse query: the best (lowest) achievable risk among all operating
points whose coverage is at least `target` — i.e. "if I require the model to answer at
least X% of questions, what's the best error rate I can get." Returns `None` if no
observed threshold reaches that coverage (rather than extrapolating).

## Bootstrap inference

```python
def bootstrap_ci(rows, metric_fn, cluster_key, n_boot=1000, alpha=0.05, seed=0):
    clusters = defaultdict(list)
    for r in rows: clusters[r[cluster_key]].append(r)
    ids = sorted(clusters)
    rng = random.Random(seed)
    for _ in range(n_boot):
        sample = [r for cid in rng.choices(ids, k=len(ids)) for r in clusters[cid]]
        v = metric_fn(sample)
        ...
    return {"estimate": metric_fn(rows), **_percentiles(stats, alpha), ...}
```
A **cluster (block) nonparametric bootstrap**, not the textbook i.i.d.-row bootstrap.
The resampling unit is a *cluster* — all rows sharing one `cluster_key` (e.g. all 6
generations for one `question_id`, or all rows for one `fact_id`) — resampled with
replacement `k=len(ids)` times via `rng.choices` (sampling with replacement, matching the
bootstrap's defining operation), and every row belonging to a chosen cluster is included
in full. This is the correct design when observations within a cluster are not
independent: 5 sampled responses to the same question share a common latent "does the
model know this" state, so treating them as 5 independent Bernoulli draws when computing
a variance estimate would understate the true sampling variability — the classic
"pseudoreplication" error corrected for by the block bootstrap. `random.Random(seed)`
(a local RNG instance, not the global `random` module state) makes the specific set of
`n_boot` resamples deterministic and reproducible independent of anything else in the
process's random state. `metric_fn(sample) is None` cases are excluded from the interval
and separately counted as `n_undefined` — e.g. a resample with zero attempted answers
makes selective accuracy undefined, and silently dropping vs. reporting that rate are
very different epistemic claims; this module does the latter.

The point estimate returned (`metric_fn(rows)`) is the metric computed on the **original,
unresampled data** — standard bootstrap practice; the resamples are used *only* to
characterize the sampling distribution's spread, not to average into a different point
estimate (that would be closer to a bagging estimator, a different and unintended
quantity).

```python
def paired_bootstrap_diff(rows_a, rows_b, metric_fn, cluster_key, n_boot=1000, alpha=0.05, seed=0):
    ...
    if set(ca) != set(cb): raise ValueError(...)
    ids = sorted(ca)
    for _ in range(n_boot):
        pick = rng.choices(ids, k=len(ids))
        va = metric_fn([r for c in pick for r in ca[c]])
        vb = metric_fn([r for c in pick for r in cb[c]])
        ...  # stats.append(va - vb)
```
A **paired bootstrap** for the difference of two methods' metrics evaluated on the
*same* underlying cluster IDs. The critical design point is that **the same resampled
cluster-ID list (`pick`) is applied to both A and B in every iteration** — this preserves
the pairing (cluster `c`'s difficulty/idiosyncrasy affects both methods' scores in that
resample identically), which is what makes this a paired rather than independent-samples
test, and is why the CI on the difference is typically much tighter than what you'd get
by independently bootstrapping A and B and subtracting their separate CIs (independent
bootstraps would double-count unrelated noise from each method's own resampling). The
`set(ca) != set(cb)` precondition enforces that this pairing is actually meaningful —
comparing two methods evaluated on different question sets would make "the same cluster
in both" undefined, so this raises rather than silently comparing unmatched populations.

```python
def _percentiles(values, alpha):
    v = sorted(values)
    def q(p):
        pos = p * (len(v) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)
    return {"ci_low": q(alpha/2), "ci_high": q(1 - alpha/2)}
```
The **percentile bootstrap CI** (Efron's original, simplest bootstrap interval method —
as opposed to the bias-corrected-and-accelerated, BCa, interval, which is not implemented
here and would be a documented possible refinement for skewed statistics). `q(p)` is
**linear interpolation between order statistics** — the same convention as NumPy's
default `interpolation='linear'` percentile method — rather than a nearest-rank
selection, giving a continuous estimate even for `n_boot` values that don't divide evenly
into percentile positions. Default `alpha=0.05` yields the standard two-sided 95% CI
(`[q(0.025), q(0.975)]`).

## Deliberately out of scope for this module

- Fitting the raw-score → probability mapping (isotonic regression, Platt scaling, or
  the frozen calibrator `plan.md` §6 assigns to a separate validation-fit step).
- The confidence signals themselves (log-probability, entropy, consistency) — computed
  in `common/model.py` and owned analytically by P4.
- Multiple-comparison correction across the several method pairs the final table will
  report — `paired_bootstrap_diff` gives one pairwise interval at a time; correcting
  across many such comparisons (e.g. Bonferroni/Holm) is a call-site decision for
  whoever assembles the master table (P6, per `plan.md` §2, at final-comparison time).
