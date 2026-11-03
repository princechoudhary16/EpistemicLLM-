# K per fact for P5 from P6's float32 `sft_matched` adapter (main_v0)

`k_per_fact.jsonl` has one row per fact (600), with the same columns and definition of K as in
`../main_v0_float32_seed0/README.md`: the share of the model's sampled answers that are correct when the fact is
asked in each of its 4 training wordings, 5 samples per wording (20 samples), decoding_v1 settings, graded with the
shared grader `p2_whole_word_v5`. The rows contain no answers and no response text.

**This is the file the plan asks for** ("K for each synthetic training fact using the float32 `sft_matched` model").
It replaces my own run in `../main_v0_float32_seed0/`, which was a stand-in made before P6's adapter was shared.

## Which model

The adapter Sahana shared on 2026-11-03, loaded on top of Qwen2.5-0.5B-Instruct (revision `7ae55760`) in float32 on a
Colab T4. No training happened in this run. What the files themselves show (`k_meta.json`):

| Property | Value |
|---|---|
| Weights | `adapter_model.safetensors`, 35,237,104 bytes, sha256 `6ed089468ba45df2a3f6b05cd5e34a0c25b0f83e965d482dc6d255e37d3fc8db` |
| Tensors | 336, all `F32` (float32), 8,798,208 parameters; same tensor names and shapes as my adapter |
| LoRA settings | r 16, alpha 32, dropout 0.05, all 7 linear layers (q, k, v, o, gate, up, down), PEFT 0.21.0 |

These match the matched recipe, and float32 weights fit a float32 training run. The files do **not** say which
dataset, training seed, data order or code produced the adapter (its `README.md` is the empty PEFT template), so those
four things are Sahana's to confirm: main_v0 `train.jsonl` in file order, seed 0, one epoch, lr 2e-4, batch 4. The
adapter and the raw responses (hold reference answers) are not committed; their hashes are in
`uncommitted_files_sha256.json`.

## What K looks like, and how it compares with my own run

| Exposure | Mean K (P6 adapter) | Facts K=0 / inconsistent / K>=0.8 | Mean K (my run) |
|---|---|---|---|
| 0x | 0.004 | 98 / 2 / 0 | 0.001 |
| 1x | 0.079 | 79 / 17 / 4 | 0.062 |
| 2x | 0.243 | 60 / 23 / 17 | 0.199 |
| 5x | 0.742 | 12 / 17 / 71 | 0.614 |
| 10x | 0.856 | 5 / 15 / 80 | 0.800 |
| 20x | 0.961 | 0 / 6 / 94 | 0.945 |

Two independent float32 runs of the same recipe agree on which facts are known, but not exactly:

- K correlates at 0.87 across all 600 facts (Spearman 0.88) and 0.84 across the 500 trained facts.
- The part of K that goes beyond the exposure level (each fact's K minus its level's mean) correlates only 0.62.
- 83% of facts land in the same K bucket in both runs; the average absolute difference in K is 0.10.
- P6's adapter has higher K at every exposure level and for every relation (at 5x `birth_year` 0.49 against 0.27).

So the K target carries run-to-run noise on top of sampling noise: a fact's K is partly a property of the training run,
not only of the fact. If P5 trains on K from one run, a second K file (from another seed) is the way to check that a
result does not depend on that one run.

## Cautions for P5 (same as the other K file)

1. K is computed on training wordings, so it is not taken from the questions in `eval.jsonl`; it is still a property of
   the same facts the head is evaluated on (different wording), so a fact-disjoint check is the control for memorising
   a fact's K.
2. The 0x facts get a K too (almost all 0). They are asked by name, so the head sees those names; `aux-zero-novel`
   already showed that 0x gains can come from recognising names.

## Checks (all pass; K also recomputed from the raw responses: 0 mismatches, 20 samples per fact)

One row per fact; exposure and split match `facts.jsonl`; K in [0, 1]; 4 wordings x 5 samples for every fact; no answer
or text field in any row; `is_trained` matches exposure; every response graded by `p2_whole_word_v5`.

Reproduce: `python -m p1_exposure.ablation.make_colab_notebook --profile k --repo-root . --adapter-dir <adapter folder>
--split-to <parts folder> --out P1_k_shared_colab.ipynb`, upload the parts to a Colab T4 runtime, and run the notebook
(about 8 minutes of sampling); or `python -m p1_exposure.k_labels --version main_v0 --out-dir <dir> --adapter <adapter folder>`
on any GPU.
