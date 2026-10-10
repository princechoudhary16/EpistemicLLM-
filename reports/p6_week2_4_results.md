# P6 Weeks 2–4 — canonical base/SFT/IDK baseline results

First real run of P6's infrastructure against actual project data (P1's `pilot_v0`
synthetic dataset: 120 facts, 760 train presentations, 240 eval questions, 30 IDK
examples). Pilot scale, single seed — not the frozen main-experiment result. See
`reports/p6_week4_fairness_and_freeze.md` for the training-parity/freeze audit.

## A grading bug caught before it could corrupt the result

The first run of `compare_methods.py` reported the IDK-trained checkpoint at **0%
accuracy, 0% abstention** — identical-looking to a model that just fails outright,
indistinguishable from the base model's own failure. That was surprising enough to
check by hand: the IDK checkpoint's actual text output was `"I don't know."`,
word-for-word, on every single question.

The grader's text-normalizer strips punctuation (so "Paris." matches "Paris"), which
also strips the apostrophe out of the model's answer -- turning `"I don't know."`
into `"i dont know"`. The phrase list being checked against still had the apostrophe
(`"i don't know"`), so the substring check silently never matched. Every abstention
was being misgraded as "incorrect." Fixed in `common/evaluate.py` by normalizing the
phrase list through the identical function before comparing (`grading_version`
bumped `placeholder_exact_v0` -> `placeholder_exact_v1`), then re-applied to the
already-generated text (no need to re-run the model) -- 1,439 of 1,440 IDK rows
changed outcome once fixed; SFT: 0 changed; base: 88 changed.

This is exactly why the join/fairness checks in this pipeline matter: a method that
looks like it "failed" can actually be the grader failing to recognize success.

## Results (corrected grading)

| Method | C | W | A | Accuracy | Hallucination | Abstention | Coverage |
|---|---|---|---|---|---|---|---|
| base (untouched) | 0 | 232 | 8 | 0.0% | 96.7% | 3.3% | 96.7% |
| standard SFT | 37 | 203 | 0 | **15.4%** | 84.6% | 0.0% | 100% |
| IDK | 0 | 0 | 240 | 0.0% | **0.0%** | **100%** | **0.0%** |

(N=240 greedy responses per method; all three evaluated on the identical 240
question_ids -- `compare_methods.py`'s join check confirmed this before computing
anything.)

**Standard SFT learned real facts.** 15.4% accuracy vs base's 0%, with a bootstrapped
95% CI of **[10.4%, 20.0%]** that excludes zero -- a statistically real improvement
given the evaluation's own sampling variability, not noise. (This is diluted by
design: the eval set spans all exposure levels including 0x/1x facts the model
shouldn't be able to learn well, so 15.4% overall is consistent with P1's own
fact-level finding that only high-exposure facts converge reliably.)

**IDK training collapsed into always abstaining.** With correct grading: 0%
hallucination sounds like a win, but coverage is also 0% -- it said "I don't know"
to every question, including facts it was shown 20 times during training. This is
exactly the failure mode `plan.md` section 10's risk table names explicitly: *"Low
hallucination comes from refusal → report coverage/selective accuracy and
matched-coverage comparisons."* Selective accuracy and AURC are both reported as
`null` for IDK, correctly, since nothing was ever attempted to compute them from.

**This is very likely a training-recipe problem, not evidence that IDK training
doesn't work.** 30 IDK examples mixed into 760 fact examples, at the same learning
rate and single epoch used for standard SFT, pushed the model into the "always
abstain" regime visible in its own training loss (IDK's final loss, 0.0114, was
*lower* than standard SFT's, 0.1597 -- near-perfect fit to a training mix the model
resolved by special-casing one easy, frequent-looking pattern rather than learning
the actual facts too). Candidate fixes for a follow-up run, not yet tried:
- Down-weight the IDK examples in the loss, or reduce how many epochs they're seen
- Use a true mixed-batch training loop (interleaving fact and IDK presentations)
  instead of this run's simplest-possible append-then-single-pass approach
- Lower learning rate specifically for the IDK run, since standard SFT only needed
  it to fit 760 harder, more varied targets

## Accuracy differences (paired, question-cluster bootstrap, n_boot=1000)

| Comparison | Estimate | 95% CI |
|---|---|---|
| SFT vs base | +15.4pp | [+10.4pp, +20.0pp] |
| IDK vs base | 0.0pp | [0.0pp, 0.0pp] |

IDK's CI is a point mass at 0 because every single resample has 0 correct answers in
both arms -- there's no sampling variability left to estimate once every answer is
identical to "I don't know."

## What this pilot run does and doesn't establish

**Solid:**
- Standard SFT measurably learns facts from this corpus, with real statistical
  support, not just an eyeballed number.
- The evaluation pipeline (schema, matched-question-join check, bootstrap CIs,
  risk-coverage/AURC) ran against real data end-to-end and caught a real bug before
  it was reported as a finding.

**Not yet established:**
- Whether IDK training *can* work well with a better recipe -- this result shows
  one specific (naive) recipe collapsing, not that abstention training is futile.
- Anything about main_v0 (600 facts) -- this is pilot_v0 (120 facts) only.
- Anything using real-world QA data -- blocked on the team's pending model/dataset
  review (P2's SimpleQA pilot).
- Training-seed robustness -- single seed (0) for both SFT and IDK.

## Files

- `results/raw/p1_pilot_v0_{base,sft,idk}.jsonl` -- 1,440 response rows each
- `results/final/baseline_comparison.json` -- full summary + bootstrap output
- `results/final/sft_pilot_v0_adapter/`, `idk_pilot_v0_adapter/` -- trained LoRA
  adapters (cluster only, not committed -- see `.gitignore`, plan.md section 9)
