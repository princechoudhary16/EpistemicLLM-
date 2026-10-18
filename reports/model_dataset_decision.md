# Model/Dataset Decision — Qwen2.5-0.5B-Instruct + SimpleQA

**Owner:** P2
**Status:** All 3 warning thresholds triggered — team decision required (see "Decisions requested")

## Pilot setup (shared protocol)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, bfloat16, run on Apple MPS |
| Prompt / decoding | `prompt_v0` / `decoding_v1` (`configs/decoding.yaml`): max 64 new tokens, greedy + 5 sampled (T=0.7, top-p 0.9, top-k 0, repetition penalty 1.0) |
| Dataset | SimpleQA snapshot `simpleqa_snapshot_2026-09-28.csv`, 4,326 questions |
| Pilot | 200 questions, seed 20260928 (identical questions to the first pilot); question file sha256 `7e7a234e…788b9` |
| Generations | 1,200 (6 per question), via `common.model.ModelRunner` + `common.evaluate.run_eval`, ~13 min |
| Grader | `common/grading.py`, version `p2_whole_word_v3` (whole-word matching, numeric guard, full-answer abstention) |
| Seeds | greedy = 0; sample *i* = *i* (shared protocol: seed + i + 1) |

## K(q) distribution (K = fraction of the 5 sampled responses graded correct)

| Bucket | Count | % of 200 |
|---|---|---|
| K = 0 | 198 | 99.0% |
| 0 < K < 1 | 2 | 1.0% |
| K ≥ 0.8 | 0 | 0.0% |

## Warning threshold check

| Threshold | Result | Triggered? |
|---|---|---|
| ≥ 90% of questions have K = 0 | 99.0% | **Yes** |
| Fewer than 20 questions with 0 < K < 1 | 2 | **Yes** |
| Fewer than 20 questions with K ≥ 0.8 | 0 | **Yes** |

**3 of 3 thresholds triggered.** (The previous version of this document said "3 of 4" while listing three thresholds; if a fourth threshold exists in the guide, it needs to be added here.)

## Behaviour of the base model (shared `common.metrics`)

| | Greedy (200) | Sampled (1,000) |
|---|---|---|
| Correct / Wrong / Abstain | 1 / 172 / 27 | 3 / 793 / 204 |
| Accuracy | 0.5% | 0.3% |
| Hallucination rate (W/N) | 86.0% | 79.3% |
| Abstention rate | 13.5% | 20.4% |
| Coverage | 86.5% | 79.6% |
| Selective accuracy (C/(C+W)) | 0.6% | 0.4% |

Truncated responses: 4 / 1,200 (0.3%). Invalid responses: 0.

## The two "inconsistent" questions are chance hits, not partial knowledge

- `sqa-04312` (K = 0.2): six different years were given across the six runs (2009, 1994, 2014, 2007, 2008, 2018); one happened to equal the reference.
- `sqa-03322` (K = 0.4): the answer "English" (a near coin-flip for the language of a figure in an English-language paper); the model abstained in 3 of 5 sampled runs.

Neither is evidence of a partially-known fact, so even the 2 "inconsistent" questions are not the behaviour P3/P4/P5 need.

## Error taxonomy (200 questions; heuristic, see limitations)

| Type | Questions | % |
|---|---|---|
| inconsistent_guessing (different wrong answers across samples) | 111 | 55.5 |
| partial_or_alias_candidate (weak flag: shares a token with the reference) | 42 | 21.0 |
| refusal (≥ 3 of 5 samples abstain) | 29 | 14.5 |
| unsupported_confident_answer (consistently wrong and high confidence) | 8 | 4.0 |
| consistently_wrong | 4 | 2.0 |
| unclassified (too few attempts) | 4 | 2.0 |
| unstable_partial_knowledge (0 < K < 0.8) | 2 | 1.0 |

## Grader audit

- **Sample:** 96 rows — all 4 "correct" rows, the 2 hedged-attempt rows, 30 weak-hint rows (one per question), and 30 random controls each for "incorrect" and "abstain". Single annotator.
- **Result under v2:** abstain 30/30 agree, correct 4/4 agree, incorrect 60/62 agree. Both errors were abstentions graded incorrect: a generic sentence followed by "I don't know <the thing asked>" (`sqa-00334`, `sqa-02533`).
- **Fix (v3):** an "IDK restating the question" rule (documented in `common/grading.py`). Re-scoring the same human labels against v3 gives 96/96 agreement. This is **not independent evidence** (the rule was written from those two rows). The meaningful checks are the regression results: the rule changed exactly these 2 of 1,200 rows in this run and 0 of the 1,200 rows from the earlier pilot.
- **Not estimated:** grader recall (the sample is stratified by predicted class). A separate check found no incorrect response containing all content tokens of its reference, which argues against missed correct answers, but it is not a recall estimate.
- K is unaffected by the v3 change (it counts only correct responses); coverage and hallucination numbers move by 2 rows.

## How this differs from the first pilot

| Run | K = 0 | 0 < K < 1 | K ≥ 0.8 |
|---|---|---|---|
| First pilot, v1 grader, old prompt (reported earlier) | 192 | 7 | 1 |
| Same responses, regraded with whole-word grader | 196 | 4 | 0 |
| This run: shared prompt/decoding protocol, v3 grader | 198 | 2 | 0 |

12 of the first pilot's 17 "correct" responses were substring false positives (e.g. reference `20` matching `2006`). The shared placeholder grader in `common/evaluate.py` gives the identical 17 on those responses; results graded with it on real QA must not be reported.

## Limitations

- 200 questions, one annotator, no grader-recall estimate.
- Generated on Apple MPS in bfloat16; other team members may use CUDA. Device and dtype are recorded in `results/raw/p2_pilot_base.meta.json`.
- Consistency (taxonomy) is a named-entity/number overlap heuristic; "confident" uses terciles of average log-probability and entropy over the whole response. The exploratory confident-hallucination count (265 / 1,000) is high by construction and is not a finding. P4 owns the validated thresholds.
- The v3 abstention rule treats a wrong name stated before a restating "I don't know" as an abstention (flagged `idk_restates_question`).
- SimpleQA references are single strings; alias handling (`" || "`) is implemented but untested on real alias data.

## Recommendation

On SimpleQA this model sits entirely at the "unknown" end of the knowledge continuum. It cannot provide the inconsistent or reliably-correct examples needed by P3 (probe training), P4 (calibration with positives) or P5 (preserving correct answers), and a different grader or prompt does not change that (three independent checks above).

Proposed handling:
1. **Keep SimpleQA as a documented negative-result baseline:** 86% hallucination with 13.5% abstention shows the problem is real.
2. **Pilot an easier real-QA dataset with answer aliases** (candidates: TriviaQA, PopQA) on the **same 0.5B model**, 200 questions (~13 min). Suggested acceptance rule, from the plan's thresholds: below 90% K = 0, and at least 20 questions in each of 0 < K < 1 and K ≥ 0.8.
3. **Optional side check:** SimpleQA pilot on a 1.5B model via `--model-yaml` (not yet run). Expectation: still low, since SimpleQA was built adversarially against strong models — a hypothesis, not a measurement.
4. **Primary controlled experiments stay on P1's synthetic exposure data** (reported by P1 as 18% inconsistent in a Colab pilot, graded with the grader in use at the time; to be confirmed after regrading with the shared grader); real QA becomes the transfer/validation arm.

## Decisions requested

| # | Decision | Owner(s) |
|---|---|---|
| A | Real-QA dataset to pilot next (and acceptance rule above) | all |
| B | Whether the real-QA arm may use a different model size than 0.5B | all, P6 |
| C | Approve `common/grading.py` (v3) as the shared grader; regrade any final results that used the placeholder | P6, P1 |
| D | Abstention definition (incl. "IDK restating the question") — same rule for P6's IDK baseline | P6, P2 |
| E | Freeze or defer the group-aware SimpleQA splits (depends on A) | P3, P2 |
| F | Question ID convention `sqa-00004` and consistency/confidence rules for the taxonomy | P3, P4 |
