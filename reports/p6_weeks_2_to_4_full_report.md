# P6 — Weeks 2 to 4: full report

**Owner:** P6 (shared infrastructure, baselines, common evaluation — `plan.md` §2)
**Scope:** everything built and run for `plan.md` §8's Week 2, 3 and 4 P6 rows.
**Status:** P6's own technical work for Weeks 2–4 is complete, at both pilot and
main scale. Two items remain genuinely open and require the team, not more code —
see §7.

---

## 0. Plain-language summary

Think of the AI model as a student, and this whole exercise as a controlled test
of three different ways of teaching it:

- **Version A ("base")** — the untouched student, never studied anything.
- **Version B ("SFT")** — studied a stack of flashcards made of invented, made-up
  facts (so we can be 100% sure whether it actually learned them or is just
  guessing).
- **Version C ("IDK")** — studied the same flashcards, *plus* extra practice
  specifically on saying "I don't know" instead of guessing when it's unsure.

We ran this twice: once on a small practice set (120 facts) to work the bugs out
cheaply, then again for real on the full set (600 facts).

**What happened:**
- **Studying worked.** Version B went from knowing 0% of the facts to a real,
  solidly-measured improvement — genuinely learned something, not a fluke.
- **But studying worked less well on the bigger set.** With 5x more facts to
  learn in the same amount of study time, each individual fact got less
  attention, so Version B's score actually dropped on the full set compared to
  the small practice set.
- **The "teach it to say I don't know" version broke, both times.** Instead of
  learning to say "I don't know" only when it genuinely didn't know something,
  it collapsed into saying "I don't know" to almost everything — even facts it
  had studied repeatedly. We tested whether this was because it was "distracted"
  by so few "I don't know" examples sitting in a now-much-bigger pile of
  flashcards. It wasn't — the exact same collapse happened again, meaning the
  problem is in *how* it was taught, not *how much* other material it had to
  compete with. We flagged this for the team rather than trying to fix it
  ourselves, since fixing it is someone else's assignment on this project.
- **Along the way, we caught and fixed a grading mistake** before trusting it:
  the very first run made it look like Version C had failed completely, when it
  had actually just started saying "I don't know" to everything — our automatic
  grader just wasn't recognizing that phrase correctly. We fixed the grader and
  re-checked its own already-saved answers rather than making it retake the test.

---

## 1. What P6 was asked to deliver, Weeks 2–4

| Week | Required of P6 | Acceptance criteria |
|---|---|---|
| 2 | Produce pilot base/SFT/IDK predictions; integrate common grading and confidence fields; reserve training slots | Three-baseline validation table; IDK training set documented; main run queue agreed |
| 3 | Complete canonical SFT/IDK runs; implement risk–coverage evaluation; validate matched-question joins | Main baseline checkpoints and validation comparison; common evaluator accepts P5 outputs |
| 4 | Check comparison fairness, exposure budgets, IDK data, and missing runs; freeze evaluation scripts/configs | Signed-off run manifest, model selection, metric definitions, and final-test execution checklist |

Everything below maps back to one of these rows.

---

## 2. Infrastructure built

| File | Purpose |
|---|---|
| `common/model.py` | `ModelRunner` — reproducible generation (greedy + sampled) from a pinned model revision, with per-token log-probability/entropy extraction |
| `common/schema.py` | The shared 28-column response-row format every method's output must follow |
| `common/metrics.py` (+ `common/tests/test_metrics.py`, 25 tests) | Accuracy, hallucination, coverage, selective accuracy/risk, knowledge score, AUROC/AUPRC, Brier/ECE, threshold policy, risk-coverage curve + AURC, bootstrap and paired-bootstrap confidence intervals |
| `common/evaluate.py` | Shared generate+grade loop (`run_eval`), used identically by the smoke test and every real baseline run |
| `common/train.py` | Minimal answer-only-loss LoRA trainer |
| `p6_baselines/generate_base.py` | Week 1 smoke-test script (refactored in Week 2 to use `common/evaluate.py`, re-verified to give identical results after the refactor) |
| `p6_baselines/run_canonical_baselines.py` | Trains + evaluates base / standard-SFT / IDK, parametrized by `--dataset-version`, with `--skip-base` to resume without re-running a completed phase |
| `p6_baselines/compare_methods.py` | The common evaluator: validates matched question-ID joins, computes per-method metrics, computes paired cluster-bootstrap accuracy diffs vs. a baseline. Takes `label=path` response-file arguments — no code change needed to add a new method |
| `p6_baselines/make_dummy_p5_response.py` | One-off verification tool: relabels an existing response file as a stand-in fourth method to prove the evaluator genuinely accepts a new method, without waiting for P5 |

---

## 3. Data

- **`pilot_v0`** (120 facts): P1's own pilot dataset, checked out from `origin/p1/synthetic-generator`.
- **`main_v0`** (600 facts): did not exist yet when Week 2–4 work started. Generated by
  running P1's own, already-tested generator code directly —
  `python -m p1_exposure.generate --n-facts 600 --version main_v0` — after confirming
  all 17 of P1's unit tests pass (Python 3.11 required; the generator's
  `Path.write_text(newline=...)` call needs 3.10+). All 11 of P1's own validation
  checks pass on the generated data (balanced exposure groups, no 0× leakage, no
  contradictory prompts, disjoint eval/train templates, etc.).

| Dataset | Facts | Train presentations | Eval questions | IDK examples |
|---|---|---|---|---|
| pilot_v0 | 120 | 760 | 240 | 30 |
| main_v0 | 600 | 3,800 | 1,200 | 30 |

---

## 4. A grading bug caught before being reported as a result

The first pilot run showed the IDK-trained checkpoint at **0% accuracy and 0%
abstention** — which made no sense next to a training loss that had converged
*better* than standard SFT's. Inspecting the actual generated text showed the model
saying `"I don't know."` on essentially every question.

**Root cause:** the grader's text-normalizer strips punctuation (so `"Paris."`
matches `"Paris"`), which also stripped the apostrophe out of the model's answer —
turning `"I don't know."` into `"i dont know"`. The phrase list being checked
against still had the apostrophe (`"i don't know"`), so the substring match silently
never fired. Every abstention was being misgraded as "incorrect."

**Fix:** normalize the phrase list through the same function before comparing
(`common/evaluate.py`), bump `grading_version` from `placeholder_exact_v0` to
`placeholder_exact_v1`, and re-grade the **already-generated text** — no need to
re-run the model, since the bug was entirely downstream of generation (the saved
`model_answer` text was always correct; only the automatic judgment of it was
wrong). 1,439 of 1,440 pilot IDK rows changed outcome once fixed; standard SFT: 0
changed; base: 88 changed.

This is also why the matched-question-join check in `compare_methods.py` matters:
a method that looks like it "failed" can be the grader failing to recognize it,
not the model.

---

## 5. Results

### 5.0 Clean summary table (both scales, side by side)

| Method | Scale | Accuracy | Hallucination | Abstention | Coverage | vs. base (95% CI) |
|---|---|---:|---:|---:|---:|---|
| Base (untouched) | Pilot (120 facts) | 0.0% | 96.7% | 3.3% | 96.7% | — |
| Base (untouched) | Main (600 facts) | 0.25% | 95.8% | 3.9% | 96.1% | — |
| **Standard SFT** | Pilot (120 facts) | **15.4%** | 84.6% | 0.0% | 100% | +15.4pp [+10.4, +20.0] |
| **Standard SFT** | Main (600 facts) | **8.2%** | 91.8% | 0.0% | 100% | +7.9pp [+6.4, +9.5] |
| **IDK** | Pilot (120 facts) | 0.0% | 0.0% | **100%** | 0.0% | 0.0pp [0.0, 0.0] |
| **IDK** | Main (600 facts) | 0.0% | 0.3% | **99.7%** | 0.3% | -0.25pp [-0.58, 0.0] |

**Read this as:** SFT is the only method that reliably learns facts (bold accuracy
column, confidence interval excludes zero both times). IDK is the only method
that drives hallucination near zero — but only by refusing to answer almost
anything (bold abstention/coverage columns), which is a failure, not a win; see
§5.3 for why.

### 5.1 Pilot scale (120 facts, 240 eval questions) — full detail

| Method | C | W | A | Accuracy | Hallucination | Abstention | Coverage |
|---|---|---|---|---|---|---|---|
| base (untouched) | 0 | 232 | 8 | 0.0% | 96.7% | 3.3% | 96.7% |
| standard SFT | 37 | 203 | 0 | 15.4% | 84.6% | 0.0% | 100% |
| IDK | 0 | 0 | 240 | 0.0% | 0.0% | 100% | 0.0% |

Accuracy diffs vs. base (paired, question-cluster bootstrap, n_boot=1000):

| Comparison | Estimate | 95% CI |
|---|---|---|
| SFT vs base | +15.4pp | [+10.4pp, +20.0pp] |
| IDK vs base | 0.0pp | [0.0pp, 0.0pp] |

### 5.2 Main scale (600 facts, 1,200 eval questions) — full detail

| Method | C | W | A | Accuracy | Hallucination | Abstention | Coverage |
|---|---|---|---|---|---|---|---|
| base (untouched) | 3 | 1150 | 47 | 0.25% | 95.8% | 3.9% | 96.1% |
| standard SFT | 98 | 1102 | 0 | 8.2% | 91.8% | 0.0% | 100% |
| IDK | 0 | 4 | 1196 | 0.0% | 0.3% | 99.7% | 0.3% |

Accuracy diffs vs. base (paired, question-cluster bootstrap, n_boot=1000):

| Comparison | Estimate | 95% CI |
|---|---|---|
| SFT vs base | +7.9pp | [+6.4pp, +9.5pp] |
| IDK vs base | -0.25pp | [-0.58pp, 0.0pp] |

### 5.3 Two findings from comparing pilot vs. main

**Finding 1 — SFT accuracy dropped at scale (15.4% → 8.2%), but the improvement
over base is still real.** Both confidence intervals clearly exclude zero, so
fine-tuning measurably teaches facts at both scales. The drop between scales isn't
a bug: training ran for a fixed single epoch at both scales (as `plan.md` requires
for comparability), but main_v0 has 5× as many *distinct* facts competing for that
same single pass of gradient updates. Each fact still gets the same number of
*presentations* its exposure level specifies — the dilution is across facts, not
within any one fact's exposure.

**Finding 2 — the IDK abstention collapse is not a small-data artifact.** After
the pilot run, the working hypothesis was that IDK's 30 extra "I don't know"
examples caused the collapse because they were a large share (3.8%) of a small
(790-example) training mix, and that diluting them to 0.8% of a 3,830-example main
corpus might fix it. **It didn't.** IDK collapsed to 99.7% abstention at main
scale — essentially identical to the pilot's. This rules out "more fact data" as a
fix and points at the training recipe itself (same learning rate/epoch count used
for standard SFT, applied unchanged to any mix containing IDK examples) as the
actual cause. **Not pursued further here, per instruction** — candidate fixes
(down-weighting IDK examples in the loss, interleaved mixed-batch training, a
lower learning rate specifically for the IDK run) are flagged for the team/P5,
not attempted.

---

## 6. Week-by-week completion

### Week 2

| Item | Status |
|---|---|
| Produce pilot base/SFT/IDK predictions | ✅ Done |
| Integrate common grading and confidence fields | ✅ Done (`common/evaluate.py`) |
| Reserve training slots | ❌ Needs team coordination — not something P6 can do unilaterally |
| *Acceptance:* three-baseline validation table | ✅ Done |
| *Acceptance:* IDK training set documented | ✅ Done |
| *Acceptance:* main run queue agreed | ❌ Needs team coordination |

### Week 3

| Item | Status |
|---|---|
| Complete canonical SFT/IDK runs | ✅ Done at both pilot and main (600-fact) scale |
| Implement risk–coverage evaluation | ✅ Done (`risk_coverage_curve`, AURC) |
| Validate matched-question joins | ✅ Done (`compare_methods.py` refuses mismatched question sets) |
| *Acceptance:* main baseline checkpoints + comparison | ✅ Done |
| *Acceptance:* common evaluator accepts P5 outputs | ✅ **Verified by test**, not just asserted — `make_dummy_p5_response.py` proved a fourth method plugs in with zero code changes |

### Week 4

| Item | Status |
|---|---|
| Training-parity / fairness audit (SFT vs IDK) | ✅ Done — identical settings except IDK's disclosed +30 examples |
| Exposure/data-budget audit | ✅ Done |
| Missing-runs audit | ✅ Done, explicit (see §7) |
| Freeze evaluation scripts/configs | ✅ Done — tagged `p6-eval-v1` |
| *Acceptance:* signed-off run manifest + checklist | ✅ Done (`reports/p6_week4_fairness_and_freeze.md`) |

---

## 7. What's still missing — and who it's needed from

Two items can't be resolved by more work from P6 alone:

1. **Reserving training slots / agreeing a run queue with the team** — the cluster
   allows 2 concurrent jobs and 1 GPU per person; six people training independently
   will eventually collide. Needs an actual conversation.
2. **The shared contract isn't formally ratified** — `plan.md` §3 assigns prompt
   format and training-parity settings (LoRA config) as **joint** decisions, several
   with P5 specifically. These were chosen unilaterally here because nobody else had
   frozen them yet. If the team adopts different settings later, every result in
   this report needs regenerating.

Three more items are blocked on specific teammates' deliverables, not on P6:

3. **P5's actual method** — the comparison table's most important column is empty.
4. **P2's real grader** — every number here uses the placeholder text-matcher,
   explicitly not meant to be final.
5. **The model/dataset decision** — P2's SimpleQA pilot triggered `plan.md` §4's
   warning thresholds; until the team decides, **no real-world-QA baselines can run
   at all**, only the synthetic track covered in this report.

---

## 8. Where to look for more detail

- `reports/p6_week2_4_results.md` — the results narrative this report consolidates
- `reports/p6_week4_fairness_and_freeze.md` — the full fairness/freeze audit
- `p6_baselines/README.md` — how to actually run any of this
- `reports/p6_code_explained_line_by_line.md`, `metrics_technical_reference.md` — the code itself, at different levels of depth
