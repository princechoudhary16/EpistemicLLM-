# P3 synthetic dev probe run

**Date:** 21 October 2026  
**Status:** Historical exploratory results only. The matched-SFT extraction used bfloat16 and P3 exact-match labels; the team later selected float32 and shared grader v5. Do not cite these metrics as final evidence. Re-run using `p3_xai/README.md` and the shared TriviaQA splits.

## Run

- Source: P1 `synthetic-main_v0`, dev split only; 600 questions across 300 fact groups, with 100 questions at each exposure level (0, 1, 2, 5, 10, 20).
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`, CPU, float32, greedy decoding, shared `prompt_v0` system prompt.
- Features: last non-padding prompt token at depths 6, 12, 18, and 24; each matrix is `600 × 896`.
- Group split: 450 training and 150 validation questions before filtering labels; all templates for a fact stay together. Six truncated responses were excluded, leaving 446 training and 148 validation labels.
- Runtime: 336.9 seconds; feature archive: 7,986,354 bytes.

## Labels and probe outcome

P3's provisional normalized exact-match proxy labeled 589 answers incorrect and 11 abstentions; there were no truncated responses and no correct greedy answers. The 450 training labels and 150 validation labels are therefore all not-correct. Each of the four linear probes and the prompt-length baseline reported `skipped: training labels contain fewer than two classes`; AUROC, AUPRC, and balanced accuracy are N/A. The majority baseline accuracy is 1.0 solely because every validation label is negative.

The feature extraction, ID alignment, and output serialization completed. The run does **not** establish a useful probe result: the greedy target and base checkpoint produced no positive examples, even at the 20-exposure level. Labels need review/regrading under the shared project grader. P3 should next consider a same-prompt repeated-response `K(q)` target or an SFT checkpoint with useful dev label diversity before fitting a correctness probe. The earlier run at `results/p3/synthetic_main_v0_base/` used P3's provisional prompt and is retained only as an initial pipeline check; this report describes the shared-prompt run. Do not use final rows to remedy this class imbalance.

## Matched-SFT run

P1's `sft_matched_main_v0` adapter was supplied locally and loaded on top of the same pinned Qwen base model. It uses the P1 adapter checkpoint, BF16 inference, shared `prompt_v0`, and the same 600 dev prompts. The adapter files are stored under the ignored `.cache/p3/sft_matched_main_v0_adapter/`; their hashes and checkpoint metadata are recorded in the run manifest. Extraction completed on CPU in 690.4 seconds.

The provisional exact-match target has both classes: training has 189 correct and 260 not-correct examples (one invalid response excluded); validation has 70 correct and 80 not-correct examples. Greedy exact-match accuracy rises with exposure: 0/100 at 0×, 5/100 at 1×, 25/100 at 2×, 67/100 at 5×, 78/100 at 10×, and 84/100 at 20×. With the existing fact-group split, layer 18 achieved 0.769 AUROC, 0.777 AUPRC, and 0.702 balanced accuracy. Layers 6, 12, and 24 achieved AUROC 0.603, 0.640, and 0.626. Prompt length achieved 0.519 AUROC; majority accuracy was 0.533. The observed exposure trend and layer 18 signal are encouraging but exploratory: labels are auto-generated normalized exact matches (one invalid response needs review), the validation set has only 150 items, and there is one split/checkpoint. They are not yet benchmark claims.

Artifacts: `results/p3/synthetic_main_v0_sft_matched_seed0/`. The config is `configs/p3_synthetic_main_v0_sft_matched.json`.

### Independent answer review

I reviewed all 600 generated SFT answers against their synthetic references. Exact matches remain correct; mismatched years and fictional names remain incorrect; the one truncated response remains invalid. I corrected 11 occupation answers where the model named the corresponding trade or activity (for example, “cartography” for “cartographer”). The original `labels.jsonl` is unchanged. Corrections are recorded with reasons in `data/p3/sft_matched_main_v0_label_review.json`, bound to the predictions checksum, and applied to `labels_reviewed.jsonl`.

The reviewed labels contain 270 correct, 329 incorrect, and 1 invalid response. This was an independent assistant review, not a human/team audit or an agreed shared grader.

After those corrections, SFT exact-match/provisional accuracy by exposure is 0/100, 5/100, 27/100, 70/100, 81/100, and 87/100 at 0×, 1×, 2×, 5×, 10×, and 20×, respectively. The differences from the earlier counts are the 11 reviewed occupation aliases.

The probe was subsequently refit using `labels_reviewed.jsonl` with the original fact-group split and hyperparameters. Training had 199 correct and 250 not-correct labels; validation had 71 correct and 79 not-correct labels; one invalid response was excluded. Layer 18 achieved 0.763 AUROC, 0.756 AUPRC, and 0.745 balanced accuracy. Layer 6/12/24 AUROCs were 0.589/0.606/0.639; prompt length was 0.526 AUROC. The result is similar to the original-label run (layer 18 AUROC 0.769). Metrics and validation predictions are saved in `results/p3/synthetic_main_v0_sft_matched_seed0/probe_reviewed/`. These remain provisional assistant-reviewed results.

For the fixed layer 18 probe, a 5,000-replicate percentile bootstrap resampled the 75 held-out fact groups, keeping the two question variants together. The 95% interval was **0.657–0.863 AUROC** and **0.611–0.877 AUPRC** (all replicates retained both classes). This quantifies sampling variation on the current development split only; it does not include grading uncertainty, training-seed variation, or model-selection uncertainty, and it is not a locked-test interval. Reproduce with `.venv-p3/bin/python -m p3_xai.bootstrap_probe`; machine-readable output is `results/p3/synthetic_main_v0_sft_matched_seed0/probe_reviewed/fact_bootstrap.json`.

### Base vs SFT representations

For a matched comparison, base prompt states were re-extracted in BF16 with batch size 1 to match the SFT run. The two runs use the same pinned base revision, prompt, 600 questions, layer depths, pooling, and inference precision. The comparison fits one joint PCA per layer on the base and SFT training-question vectors from the seed-42 fact-group split, then projects the 150 held-out questions. The [PCA figure](p3_representation_comparison/base_vs_sft_joint_pca.svg), [paired metrics](p3_representation_comparison/paired_representation_metrics.csv), and [comparison notes](p3_representation_comparison/README.md) are saved.

Mean paired cosine similarity for base-to-SFT prompt vectors was 0.916, 0.890, 0.741, and 0.653 at layers 6, 12, 18, and 24. Mean relative L2 shift was 0.419, 0.492, 0.718, and 0.778. This shows the adapter changes the prompt representations, with larger shifts at deeper layers; these geometry measures do not show that one representation is better. At 0× exposure, both models had zero correct answers. The base had zero correct answers at every exposure under its automatic exact-match labels, so a base correctness probe still cannot be fit meaningfully. SFT uses assistant-reviewed labels, so cross-model correctness comparisons remain provisional.

As a separate feasibility check, the existing P1 dev response files contain five sampled responses per question. Under their recorded placeholder exact grader, base has only 3/600 questions with `K>0` and no `K≥0.8`; SFT has 70/600 with `K>0` and 25/600 with `K≥0.8`; IDK has none with `K>0`. These counts are exploratory because they use the placeholder grader. The SFT adapter is now available locally, so the matched-SFT hidden-state extraction above has been run.

## Why this run had zero correct answers

The configuration loaded the pinned **base** checkpoint. Its only input was each dev question; it did not load P1's synthetic training examples or a fine-tuned checkpoint. The names and facts are fictional, so the base model had no reason to know which specific year, founder, or location had been assigned. Exposure level is metadata on the question rows; it does not alter base-model weights. The zero greedy hits therefore reflect running the base checkpoint before synthetic exposure training.

This is consistent with the stored P1 responses. The table uses P1's five sampled outputs per dev question, grouped by the question's exposure level. Each level has 100 questions. Counts use P1's `placeholder_exact_v1` labels and are exploratory:

| Exposure | Base questions with K>0 | SFT questions with K>0 | SFT questions with K≥0.8 |
|---:|---:|---:|---:|
| 0× | 1 | 2 | 0 |
| 1× | 0 | 3 | 0 |
| 2× | 2 | 0 | 0 |
| 5× | 0 | 2 | 2 |
| 10× | 0 | 11 | 4 |
| 20× | 0 | 52 | 19 |

The P1 SFT responses show increasing answerability at the higher exposure levels, while the base responses do not. That is evidence to pursue the P3 probe on the trained checkpoint, not a probe result from this base-only run.

The provisional exact-match proxy also was not the cause of this zero: applying PR #7's shared grader as a diagnostic to the 600 P3 answers produced the same 589 incorrect and 11 abstentions, with no newly correct answers. This diagnostic does not change the saved labels or constitute approval of that grader for P3. A precision difference remains between runs: P1's configured inference uses bfloat16 while this P3 extraction used float32; for `F0073-q0`, P1's greedy answer happened to match the reference and P3's did not. That single discrepancy does not explain the absence of hits across the other dev questions.

## Artifacts and reproduction

- Primary run manifest, predictions, labels, features, and probe report: `results/p3/synthetic_main_v0_base_prompt_v0/`
- Matched-SFT run manifest, predictions, labels, features, and probe report: `results/p3/synthetic_main_v0_sft_matched_seed0/`
- Initial provisional-prompt run: `results/p3/synthetic_main_v0_base/`
- Prepared dev input and selection manifest: `data/p3/synthetic_main_v0_dev.jsonl` and `data/manifests/p3_synthetic_main_v0.json`
- Reproduce with the three commands in [the P3 README](../p3_xai/README.md), using a fresh output directory if rerunning.
