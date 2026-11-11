# P1 Week 5 protocol (written before the final tables are produced)

Plan.md, Week 5, P1: "Produce final exposure figures for all required methods; supply fact/group labels
for joint analyses." This file fixes the reporting choices so they cannot drift to suit a result.

## Choices fixed here

1. **Headline set: the `final` split.** In `main_v0` every fact is either `dev` (asked with the *val*
   wordings) or `final` (asked with the *test* wordings), half each within every exposure level. The
   headline numbers use the 300 `final` facts (50 per exposure level, 100 questions per level). The `dev`
   split is reported separately as the development set. The `all` view (all 600 facts, 1,200 questions)
   is also reported so the numbers line up with P6's master table; it is not the headline.
2. **Methods, all on identical `main_v0` questions, greedy responses, decoding_v1:**
   `base`, `sft` (P6 original recipe), `idk` (P6 original recipe), `sft_matched`, `p5_lam0`, `p5_lam3`.
   The response files are the ones on `main`; nothing is rerun or retrained.
3. **Metrics per exposure level:** accuracy, hallucination, abstention, coverage, selective accuracy,
   mean K and the K buckets, each with a 95% fact-level cluster bootstrap interval (4,000 draws, seed 0).
   Confidence versus exposure uses each response file's `confidence_score` (average log-probability).
4. **Grading:** the grader recorded in each response file (`placeholder_exact_v1`). Regrading with the
   shared v3 grader moved accuracy by at most 0.1 points on P1 data, so it is not applied here.
   **Amendment 2026-10-31:** the team's decision 2 makes the shared grader `p2_whole_word_v5` the one
   grader for every final number, so all six response files were regraded with it
   (`python -m p2_real_qa.regrade`) and the tables rebuilt. This is the only change to the protocol. The decision to
   regrade was taken before any regraded table was produced (only the counts of changed rows, at most 13 of
   7,200 per file, had been seen); this note was written after the tables.
   **Amendment 2026-11-05:** team decision 6 (float32 for all training and evaluation) requires the bfloat16
   `sft_matched` rows to be replaced once P6's float32 retrain exists, and it now does
   (`results/raw/p1_main_v0_sft_matched.jsonl`, already graded with v5). I replaced that one response file and
   rebuilt `final`, `dev`, `all`, the Week 3 `main_v0` tables and the labels file. The split, metrics, intervals,
   bootstrap seed and method list are unchanged, and the replacement was planned before I looked at the new
   numbers; the headline `sft_matched` accuracy on the final facts moved from 47.0% to 48.3%. `base`, `sft` and
   `idk` remain bfloat16 until P6 reruns them. On 2026-11-11 I also replaced the two P5 files with P5's float32
   re-export (PR #27, about 420 of 7,200 sampled answers per file changed): final-facts accuracy moved from 47.5
   to 47.7% (`p5_lam0`) and from 48.8 to 49.2% (`p5_lam3`).
5. **No choice above changes after the final tables exist.** A defect found later is logged and every
   affected method is rerun the same way.

## What "locked" does and does not mean

This is a reporting lock, not an untouched hold-out. The `final` facts were already included in the
Week 3 and Week 4 analyses, and in P5's and P6's results, so the final split has been seen. The lock
guarantees only that the headline split, metrics, intervals and method list were chosen before this
week's final tables and are applied identically to every method.

## Labels supplied for joint analyses

One row per evaluation question for each method: `question_id`, `fact_id`, `relation`,
`exposure_level`, `split`, the greedy outcome, the question's K (fraction of its 5 sampled responses
that were correct) and the fact's K bucket. The file contains no reference answers, so it cannot leak
a test answer into a probe or calibrator. Labels from a model that answers correctly (for example
`sft_matched`) are needed for probing; labels from `base` are almost all "not correct".
