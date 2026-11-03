# K per fact for P5 (main_v0, float32, training seed 0)

> **Superseded for P5's use:** `../main_v0_float32_p6_adapter/` has K from P6's float32 `sft_matched` adapter, which
> is the source the plan names. This file (my own run of the same recipe) is kept as a second, independent run for
> checking how much K varies between training runs; the comparison is in that folder's README.

`k_per_fact.jsonl` has one row per fact (600). It is the target P5 needs for the K(q) version (team decision 4,
2026-10-31): how often the standard fine-tuned model actually gets a fact right, instead of how many times the
fact was shown.

**K of a fact** is the share of the model's sampled answers that are correct. The fact is asked in each of its 4
training wordings, and the model answers each wording 5 times (temperature 0.7, top-p 0.9, top_k 0, repetition
penalty 1.0, the project's decoding_v1 settings), so K rests on 20 samples. An answer is correct if the shared
grader (`p2_whole_word_v5`) says so.

| Column | Meaning |
|---|---|
| `fact_id`, `relation`, `exposure_level`, `split` | Same as `facts_manifest.jsonl` (`split` is dev or final) |
| `is_trained` | false for the 100 facts at exposure 0 (never in the fine-tuning data), true otherwise |
| `k` | Share of the 20 samples that are correct (0 to 1) |
| `k_by_template` | The same share for each of the 4 training wordings |
| `greedy_correct_by_template`, `greedy_correct_rate` | Whether the single most likely answer was correct, per wording and averaged |
| `k_bucket` | `K=0` (never right), `inconsistent` (0 < K < 0.8), `K>=0.8` (reliably right); same cut-offs as the analysis |
| `n_templates`, `n_samples_per_template`, `n_sampled`, `n_correct`, `grading_version` | Counts and the grader used |

The rows contain no answers and no response text.

## Which model

This is **my own float32 run of the matched recipe, not P6's float32 `sft_matched`**: Qwen2.5-0.5B-Instruct
(revision `7ae55760`), LoRA r16, alpha 32, dropout 0.05 on all 7 linear layers, batch 4, lr 2e-4, one epoch over
`main_v0/train.jsonl` in file order, training seed 0, float32, run on a Colab T4. The plan names P6's float32
retrain as the source; the tool takes P6's adapter directly (`--adapter`), so this file can be replaced by running
`python -m p1_exposure.k_labels --version main_v0 --out-dir <dir> --adapter <P6 adapter dir>` and nothing else changes.
`k_meta.json` records the model, precision, recipe, seed, grader hash and the hashes of the inputs. The trained adapter
(35 MB) and the raw responses (4.6 MB, they hold reference answers) are not committed; their sha256 is in
`uncommitted_files_sha256.json`.

## What K looks like

| Exposure | Mean K | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|
| 0x | 0.001 | 99 / 1 / 0 |
| 1x | 0.062 | 84 / 15 / 1 |
| 2x | 0.200 | 57 / 33 / 10 |
| 5x | 0.614 | 19 / 28 / 53 |
| 10x | 0.800 | 5 / 21 / 74 |
| 20x | 0.945 | 2 / 3 / 95 |

Exposure explains 64% of the variation in K across facts (R squared 0.64), so K is not just a relabelling of the
exposure count: at 5x, 19 facts are never answered correctly and 53 are answered reliably, and `birth_year` facts at
5x average K 0.27 against 0.61 to 0.81 for the other relations. About 36% of K varies within an exposure level.

## Design choices and cautions for P5

1. **Training wording, not evaluation wording.** K comes from the 4 training question wordings, so the target is
   never computed on the questions in `eval.jsonl` that the head is later scored on.
2. **K is still a property of the same facts.** The head would be trained on K of facts that it is then evaluated
   on (with different wording). That can let the head memorise a fact's K instead of reading evidence in the model.
   A fact-disjoint check (train the head on K of some facts, score others) is the control that rules this out.
3. **The 0x facts get a K too** (almost all 0), which gives the head the never-seen examples the PDF asks for. They
   are asked by name, so the head sees those entity names; `aux-zero-novel` already showed that 0x gains can come
   from recognising names, so treat any 0x improvement with the same control.
4. **One seed, one model.** The three training seeds I ran earlier differ by up to 4 points overall, so K for an
   individual fact carries training-seed noise as well as sampling noise (20 samples).

## Checks (all pass, also re-run independently from the raw responses: 0 mismatches)

One row per fact; exposure and split match `facts.jsonl`; K in [0, 1]; every fact has 4 wordings x 5 samples; no
answer or text field in any row; `is_trained` matches exposure; every response graded by `p2_whole_word_v5`.

Reproduce: `python -m p1_exposure.ablation.make_colab_notebook --profile k --repo-root . --out P1_k_labels_colab.ipynb`,
then run the notebook on a T4 (about 12 minutes), or run `python -m p1_exposure.k_labels` on any GPU.
