# Model/Dataset Decision — Qwen2.5-0.5B-Instruct: SimpleQA → TriviaQA

**Owner:** P2
**Status:** SimpleQA triggered all 3 warning thresholds. The TriviaQA replacement pilot passes all 3 (section "TriviaQA replacement pilot"). **Decided on 31 Oct: TriviaQA is the real-QA dataset; SimpleQA is kept as a negative-result baseline; v5 is the shared grader; float32 throughout.**

## Pilot setup (shared protocol)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, bfloat16, run on Apple MPS |
| Prompt / decoding | `prompt_v0` / `decoding_v1` (`configs/decoding.yaml`): max 64 new tokens, greedy + 5 sampled (T=0.7, top-p 0.9, top-k 0, repetition penalty 1.0) |
| Dataset | SimpleQA snapshot `simpleqa_snapshot_2026-09-28.csv`, 4,326 questions |
| Pilot | 200 questions, seed 20260928 (identical questions to the first pilot); question file sha256 `7e7a234e…788b9` |
| Generations | 1,200 (6 per question), via `common.model.ModelRunner` + `common.evaluate.run_eval`, ~13 min |
| Grader | `common/grading.py`, version `p2_whole_word_v5` (whole-word matching, numeric guard, full-answer abstention, negation guard, wrong-candidate guard, generic-hedge abstention) |
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
| Correct / Wrong / Abstain | 1 / 172 / 27 | 3 / 794 / 203 |
| Accuracy | 0.5% | 0.3% |
| Hallucination rate (W/N) | 86.0% | 79.4% |
| Abstention rate | 13.5% | 20.3% |
| Coverage | 86.5% | 79.7% |
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
- **v4 (PR review fixes):** (1) a reference mention with a negator in the 3 words before it ("It is not Paris, it is Rome.") is no longer credited (`negated_reference`); (2) the "IDK restating the question" rule no longer applies when another sentence names a candidate (a capitalised word not in the question), so "The medal went to John Smith. I don't know who won…" is a hallucination, not an abstention. Regression tests for both are in `p2_real_qa/tests/test_grading.py`.
- **v4 effect on this run:** the negation guard changes 0 of 1,200 rows. The candidate guard changes 1: `sqa-00334` (sample 3) goes back to incorrect, because "the Sun" in its generic first sentence counts as a candidate. This is accepted on purpose: hiding a hallucination as an abstention is the worse error for this project. Audit agreement under v4 is **95/96** (the one disagreement is that row). K, the K buckets and the taxonomy are unchanged; sampled hallucination 79.3% → 79.4%.
- **v5:** a hedge built only from category words and the question's own words ("X was a trade or profession, but I don't know which one.") is an abstention (`idk_generic_hedge`); one that names a specific answer ("an Egyptian musician, but I don't know…", "folding and carving I don't know") stays incorrect. Found by regrading P1's synthetic base-model file: of 13 rows the placeholder grader called abstentions and v4 called incorrect, v5 makes 7 abstentions and keeps 6 incorrect. Changes 0 of the 1,200 SimpleQA pilot rows, so every number in this report is unchanged.

## How this differs from the first pilot

| Run | K = 0 | 0 < K < 1 | K ≥ 0.8 |
|---|---|---|---|
| First pilot, v1 grader, old prompt (reported earlier) | 192 | 7 | 1 |
| Same responses, regraded with whole-word grader | 196 | 4 | 0 |
| This run: shared prompt/decoding protocol, v3 grader (identical under v4) | 198 | 2 | 0 |

12 of the first pilot's 17 "correct" responses were substring false positives (e.g. reference `20` matching `2006`). The shared placeholder grader in `common/evaluate.py` gives the identical 17 on those responses; results graded with it on real QA must not be reported.

## Limitations

- 200 questions, one annotator, no grader-recall estimate.
- Generated on Apple MPS in bfloat16; other team members may use CUDA. Device and dtype are recorded in `results/raw/p2_pilot_base.meta.json`.
- Consistency (taxonomy) is a named-entity/number overlap heuristic; "confident" uses terciles of average log-probability and entropy over the whole response. The exploratory confident-hallucination count (265 / 1,000) is high by construction and is not a finding. P4 owns the validated thresholds.
- The v4 candidate check is a capitalised-word heuristic: it can treat a generic capitalised noun ("the Sun") as a candidate (counted incorrect, the safe direction), and it misses a wrong candidate written in lower case. The negation guard looks only 3 words back.
- SimpleQA references are single strings; alias handling (`" || "`) is implemented but untested on real alias data.

## Recommendation

On SimpleQA this model sits entirely at the "unknown" end of the knowledge continuum. It cannot provide the inconsistent or reliably-correct examples needed by P3 (probe training), P4 (calibration with positives) or P5 (preserving correct answers), and a different grader or prompt does not change that (three independent checks above).

Proposed handling:
1. **Keep SimpleQA as a documented negative-result baseline:** 86% hallucination with 13.5% abstention shows the problem is real.
2. **Pilot an easier real-QA dataset with answer aliases** (candidates: TriviaQA, PopQA) on the **same 0.5B model**, 200 questions (~13 min). Suggested acceptance rule, from the plan's thresholds: below 90% K = 0, and at least 20 questions in each of 0 < K < 1 and K ≥ 0.8.
3. **Optional side check:** SimpleQA pilot on a 1.5B model via `--model-yaml` (not yet run). Expectation: still low, since SimpleQA was built adversarially against strong models — a hypothesis, not a measurement.
4. **Primary controlled experiments stay on P1's synthetic exposure data** (reported by P1 as 18% inconsistent in a Colab pilot, graded with the grader in use at the time; to be confirmed after regrading with the shared grader); real QA becomes the transfer/validation arm.

## TriviaQA replacement pilot

Same model, prompt, decoding and grader as SimpleQA; only the dataset changes. **All TriviaQA numbers below are float32** (team decision of 31 Oct; the earlier bfloat16 runs are superseded and kept only in git history).

| Item | Value |
|---|---|
| Dataset | TriviaQA `rc.nocontext` validation, `mandarjoshi/trivia_qa@0f7faf33` (the pin P4 uses), snapshot `data/raw/triviaqa_rc_nocontext_validation_0f7faf33.jsonl` |
| Release | `data/real_qa/triviaqa_v0/` built by `python -m p2_real_qa.data --source triviaqa --version v0`; all 7 integrity checks pass (`data/manifests/real_qa_triviaqa_v0/audit.json`) |
| Unique questions | 9,960 (the rc file lists 7,984 questions twice, once per evidence source; the first copy is kept, 2 copies differed slightly) |
| Splits | pilot 200, train_probe 500, validation 200, locked_test 300; grouped by canonical answer, so questions with the same answer never straddle splits |
| P4 overlap | P4's 300 calibration questions, and the 303 that share an answer with one, are excluded from every split (check `excluded_ids_absent`) |
| Answers | about 12.5 aliases per question, joined with ` \|\| `. Bracketed labels also add a bare form ("Roots (book)" → "Roots"). Dropped: 1,021 aliases already written in the question, also after removing a possessive "'s" (e.g. "Maurice Cole" for Kenny Everett, "The Elephant Man" for "the Elephant Man's real name": restating the question must not earn credit) and 31 one-letter aliases; digits are kept |
| Model / precision | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760`, **float32**, Apple MPS (recorded in each `.meta.json`) |
| Prompt / decoding / grader | `prompt_v0` (zero-shot, the shared prompt) / `decoding_v1` / `p2_whole_word_v5` |
| Responses | `results/raw/p2_triviaqa_v0_pilot_base.jsonl`, graded at generation time with the final aliases |

**Warning thresholds: all 3 pass.**

| Threshold | SimpleQA | TriviaQA | Passes? |
|---|---|---|---|
| < 90% of questions with K = 0 | 99.0% | 70.5% (141) | Yes |
| ≥ 20 questions with 0 < K < 1 | 2 | 46 (35 strictly below 0.8) | Yes |
| ≥ 20 questions with K ≥ 0.8 | 0 | 24 | Yes |

**Behaviour (shared `common.metrics`).**

| | Greedy (200) | Sampled (1,000) |
|---|---|---|
| Correct / Wrong / Abstain | 41 / 140 / 19 | 177 / 729 / 94 |
| Accuracy | 20.5% | 17.7% |
| Hallucination rate (W/N) | 70.0% | 72.9% |
| Abstention rate | 9.5% | 9.4% |
| Coverage | 90.5% | 90.6% |
| Selective accuracy | 22.7% | 19.5% |

**Hallucination falls steadily with K** (sampled responses, by question bucket):

| Correct out of 5 | Questions | Hallucination | Abstention |
|---|---|---|---|
| 0/5 | 141 | 90.1% | 9.9% |
| 1/5 | 16 | 63.8% | 16.2% |
| 2/5 | 5 | 52.0% | 8.0% |
| 3/5 | 14 | 34.3% | 5.7% |
| 4/5 | 11 | 10.9% | 9.1% |
| 5/5 | 13 | 0.0% | 0.0% |

The real-world curve has the same shape as P1's synthetic exposure curve: gradual, with no single threshold. On questions it never answers correctly, the model still answers 90.1% of the time.

**Error taxonomy (200 questions; same heuristics as above).** inconsistent guessing 62 · partial/alias candidate 45 · unstable partial knowledge 35 · reliable (K ≥ 0.8) 24 · unclassified 12 · consistently wrong 10 · refusal 7 · unsupported confident answer 5.

**Grading spot-check.** All 66 questions flagged `ref_partial_overlap` in the first pilot were read: nearly all are genuine wrong answers sharing a word with the reference ("Obama" for Lincoln, "Hackney" for Whitechapel, "Apollo 11" for Apollo 17). The one systematic miss (bracketed aliases) is fixed above. The stratified audit on the main set follows below.

**Limitations.** Known questions are scarce (24 of 200 here, 77 of 700 in the main set below). The pilot uses the shared zero-shot prompt; P4's few-shot TriviaQA run is not directly comparable. One annotator.

## TriviaQA main development set and grading audit

Same model (float32), prompt, decoding and grader (`p2_whole_word_v5`), final aliases. Responses: `results/raw/p2_triviaqa_v0_{train_probe,validation}_base.jsonl`; tables and figures: `results/final/p2_triviaqa_train_probe_*`.

**K labels handed to P3 and P4** (`results/features/p2_triviaqa_v0_{train_probe,validation}_k_labels.jsonl`, written by `p2_real_qa/export_labels.py`):

| Split | Questions | Unknown (K = 0) | Partial (0 < K < 0.8) | Known (K ≥ 0.8) |
|---|---|---|---|---|
| train_probe | 500 | 364 | 87 | 49 |
| validation | 200 | 132 | 40 | 28 |
| Total | 700 | 496 (70.9%) | 127 | 77 |

**Behaviour.**

| | train_probe greedy | train_probe sampled | validation greedy | validation sampled |
|---|---|---|---|---|
| Correct / Wrong / Abstain | 88 / 378 / 34 | 380 / 1,937 / 183 | 47 / 135 / 18 | 199 / 720 / 81 |
| Accuracy | 17.6% | 15.2% | 23.5% | 19.9% |
| Hallucination rate (W/N) | 75.6% | 77.5% | 67.5% | 72.0% |
| Abstention rate | 6.8% | 7.3% | 9.0% | 8.1% |
| Coverage | 93.2% | 92.7% | 91.0% | 91.9% |
| Selective accuracy | 18.9% | 16.4% | 25.8% | 21.7% |

**The pilot's main finding replicates on 500 new questions** (train_probe, sampled responses):

| Correct out of 5 | Questions | Hallucination | Abstention |
|---|---|---|---|
| 0/5 | 364 | 91.2% | 8.8% |
| 1/5 | 44 | 74.1% | 5.9% |
| 2/5 | 22 | 57.3% | 2.7% |
| 3/5 | 21 | 37.1% | 2.9% |
| 4/5 | 16 | 16.2% | 3.8% |
| 5/5 | 33 | 0.0% | 0.0% |

**Error taxonomy (train_probe, 500 questions).** inconsistent guessing 145 · partial/alias candidate 134 · unstable partial knowledge 87 · reliable 49 · unclassified 46 · consistently wrong 16 · refusal 14 · unsupported confident answer 9.

**Grading audit.** 142 train_probe rows, stratified by predicted class (`python -m p2_real_qa.audit make --n-correct 40 --n-weak 40 --n-control 30`), labelled by one annotator; sheet: `results/audit/audit_sheet_p2_triviaqa_v0_train_probe_base.csv`.

| Predicted | Rows | Human agrees |
|---|---|---|
| correct | 38 | 100% |
| incorrect | 74 | 100% |
| abstain | 30 | 100% |
| All | 142 | **142 / 142** |

How it got there:
- **First pass (bfloat16 answers): 140 / 142.** One real bug: "The real name of the Elephant Man was William Henry Biggs" was credited because "The Elephant Man" was an alias and the question says "Elephant Man's". Aliases are now also checked against the question with possessives removed (6 train_probe rows and 1 pilot row changed, all wrong names). The other disagreement was a vague hedge ("Brown can help people with various tasks, but I don't know the specific slogan…") graded incorrect rather than abstain.
- **After the float32 rerun:** 10 audited answers changed wording, so their old labels no longer applied (`audit score --responses` now detects this and writes them to `…__relabel.csv`). They were labelled again: all 10 are wrong answers, and the grader agrees on all 10. The vague-hedge row became a plain wrong answer under float32.
- **Known limitation, still true:** v5 can count a vague refusal with filler words as incorrect instead of abstain (the error is in the safe direction: counted as a hallucination, not hidden). It no longer appears in this sample.
- **Not estimated:** grader recall (the sample is stratified by predicted class), and agreement between annotators (one annotator).

## Decisions

Team meeting, 31 Oct 2026 (the full team plan is in the shared "Remaining Work and Order" doc).

| # | Decision | Outcome |
|---|---|---|
| A | Real-QA dataset | **Decided: TriviaQA (`triviaqa_v0`)**; SimpleQA kept as a negative-result baseline |
| B | Model size for the real-QA arm | Same 0.5B model throughout (not revisited) |
| C | Shared grader | **Decided: `common/grading.py` v5** for every final number |
| D | Abstention definition (incl. "IDK restating the question") — same rule for P6's IDK baseline | Follows from C: v5's abstention rules |
| E | Freeze the TriviaQA splits | **Decided: pilot, train_probe and validation frozen.** The locked test is still an open question for the 5 Nov meeting |
| F | Question ID convention (`sqa-00004`, `triviaqa-bb_1357` = P4's IDs) and the K-label file format (`p2_real_qa/export_labels.py`) | Proposed; confirm on 5 Nov |
| — | Precision | **Decided: float32** for all training and evaluation (`configs/model.yaml`) |
