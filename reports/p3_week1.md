# P3 Week 1 completion report

**Date:** 20 September 2026  
**Status:** Week 1 engineering deliverables completed and run on the actual Qwen model. Labels remain provisional; scientific evaluation awaits audited labels and adequate class diversity.

## Completed work

| Week 1 requirement | Evidence | Status |
|---|---|---|
| Load Qwen and record environment | [Run manifest](../results/p3/week1/manifest.json), [dependency lock](../p3_xai/requirements-lock.txt) | Complete |
| Prepare approximately 100 development questions | [100 questions](../data/p3/week1_questions.jsonl), [selection manifest](../data/manifests/p3_week1_selection.json) | Complete; selected before inference |
| Extract four prompt-only depths | Depths 6, 12, 18, 24; each array is `100 × 896` | Complete |
| Document pooling and preserve ID alignment | Last non-padding prompt token; IDs saved inside feature archive | Complete |
| Build and run a small linear probe | Four logistic probes plus majority and prompt-length baselines | Complete; metrics limited by label imbalance |
| Save/reload features and probes | Exact feature-array round trip and unchanged reloaded probe predictions | Passed |
| Verify critical failure cases | 10 offline tests, including hook equivalence against Hugging Face outputs | Passed |
| Document reproduction and handoff | [P3 README](../p3_xai/README.md) | Complete |

## What actually ran

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Source: public SimpleQA CSV, versioned by SHA-256; 100 unique normalized questions sampled with seed 42. These IDs are development-only and must be reserved from later final testing.
- Prompt: brief factual answer with explicit permission to say “I don't know.” This is a local provisional prompt, not a P6-approved shared contract.
- Inference: CPU, float32, four threads, batch size four, greedy decoding, maximum 64 new tokens. No fine-tuning or external judge.
- Features: one pooled 896-dimensional vector at each of four depths per question, captured before generation. Final depth includes final RMS normalization.
- Full extraction/generation/artifact run: **69.7 seconds**, excluding dependency/model downloads. Compressed feature archive: **1,334,177 bytes** (about 1.27 MiB).
- Split: 75 development-training questions and 25 development-validation questions, grouped and fixed without looking at labels. One truncated training response has a null label, leaving 74 training examples for fitting.
- Original-label and corrected-label probes both trained successfully. All saved probe reload checks passed.

The complete raw output is in [predictions.jsonl](../results/p3/week1/predictions.jsonl). The [manifest](../results/p3/week1/manifest.json) records hashes, environment versions, shapes, and settings. Weights, feature arrays, and serialized probes are local artifacts ignored by Git; reproduction commands regenerate them.

## Provisional labels and an identified grading error

| Label category | Initial strict matching | After one explicit assistant correction |
|---|---:|---:|
| Correct | 1 | 2 |
| Incorrect substantive answer | 88 | 87 |
| Literal abstention | 10 | 10 |
| Unresolved truncated response | 1 | 1 |
| Total | 100 | 100 |

These are **provisional label counts, not an official SimpleQA accuracy estimate**. A whole-string matcher misses answers expressed in sentences and can misclassify less literal refusals. The unreviewed nonmatches must still be audited by P2.

One observed false negative, `simpleqa-02278`, had reference “Billy Graham”; Qwen correctly named Billy Graham in a complete sentence. This was corrected in a separate label file with an explicit reason. Original predictions and labels were preserved. The [correction record](../data/p3/week1_label_reviews.json) is bound to the original response-file checksum. It records an assistant review, not a human audit.

## Probe smoke results

With the corrected provisional labels, training has **2 positive and 72 negative** examples; validation has **0 positive and 25 negative** examples. This split was not changed after inspecting outcomes.

| Probe / baseline | Validation accuracy | AUROC | Interpretation |
|---|---:|---|---|
| Training-majority baseline | 1.00 | N/A | Predicting every validation answer as not correct already scores perfectly |
| Depth 6 | 1.00 | N/A | Successful pipeline execution only |
| Depth 12 | 1.00 | N/A | Successful pipeline execution only |
| Depth 18 | 0.92 | N/A | Successful pipeline execution only |
| Depth 24 | 0.96 | N/A | Successful pipeline execution only |
| Prompt-length baseline | 0.40 | N/A | No knowledge-discrimination conclusion possible |

Balanced accuracy and AUPRC are also reported as N/A because validation contains only one class. The underlying [metrics](../results/p3/week1/probe_reviewed/metrics.json), [split](../results/p3/week1/probe_reviewed/split.json), and [validation predictions](../results/p3/week1/probe_reviewed/validation_predictions.csv) are saved. [Original exact-match results](../results/p3/week1/probe/report.md) remain available for comparison.

**No best layer is selected.** The apparent high accuracy reflects the validation class distribution, not demonstrated uncertainty detection. Two positive training labels for 896 input features are far too few for a credible knowledge probe. Regularization and scaling do not fix the lack of data.

## Checks and limitations

The offline tests cover left/right padding, zero-token masks, depth indexing, selected-hook equivalence to full hidden-state outputs on a tiny Qwen2 model, hook removal, ID-based joins, missing/duplicate/stale labels, conservative grading, development-only inputs, duplicate grouping, training-only standardization, group isolation, and single-class metrics/fitting. Ten test methods cover these cases.

The real run additionally verifies feature save/reload and probe save/reload. All 100 questions retain representations, including the response excluded from supervised probe fitting for truncation. References and generated answers are never passed into the prompt-state extraction function.

A final two-question run reproduced both answer texts with batch size two instead of four. Vectors were numerically close, not bitwise identical: relative L2 differences were below `6e-6` at all depths. An initial strict elementwise comparison failed at final depth (maximum absolute difference approximately `0.00069`), so cross-batch numerical identity is not claimed. Within-run feature save/reload was exact. Details and source hashes are in the [verification record](p3_week1_verification.json).

Limitations to carry forward:

1. Labels are provisional and not a completed P2 audit; only one explicit semantic correction was made.
2. This is one greedy response per question, not P2's five-sample knowledge mapping. It does not measure `K(q)` or satisfy the team's model-switch gate by itself.
3. The prompt and development IDs require coordination with P2/P6 before shared comparisons.
4. Only exact normalized duplicate prompts are automatically grouped; related facts/paraphrases need dataset-level review.
5. No causal XAI claim, layer ranking, calibrated-confidence claim, or final-test result is supported here.

## Week 2 handoff

- **To P2:** audit these responses or provide audited labels for these exact outputs. Review the severe lack of positive labels alongside the full repeated-generation pilot. Do not balance a benchmark by keeping only questions Qwen answers correctly.
- **To P6:** confirm the shared model revision, chat prompt, decoding, and development/final-test manifests. Re-extract if these inputs change.
- **For P3:** once label diversity is sufficient, run the four-layer comparison on the shared train/validation sets and report AUROC with class counts. Keep related questions/facts grouped. Synthetic data is the planned fallback for continuum analysis if real-QA outcomes remain nearly constant.
- **To P4:** the validation-score CSV has question IDs for later joins. Current sigmoid outputs are uncalibrated and the pilot is too imbalanced for confidence conclusions.

The team has not switched models or datasets as part of this P3 task. The working pipeline can be reused after the team's model/data decision.

## P3 completion status against the team plan

### Completed on current development artifacts

- Prompt-only hidden states were extracted at depths 6, 12, 18, and 24 for 600 synthetic development questions on both the untouched base checkpoint and the matched-SFT checkpoint.
- The matched-SFT run has a reproducible grouped probe fit, training-only scaling, majority and prompt-length baselines, and assistant-reviewed provisional labels. The base run correctly records that a correctness probe cannot be fit because every automatic label is negative.
- A same-question base/SFT representation comparison and train-fit joint PCA are saved. The fixed layer 18 SFT probe now also has fact-cluster bootstrap intervals; the command and artifact are documented in `p3_xai/README.md`.
- The P3 offline regression suite passes (10 tests).

### Still blocked for plan-level final acceptance

1. **P2 grading and real-QA split:** current SimpleQA and synthetic SFT labels are P3 provisional/assistant-reviewed labels, not P2-approved labels. P2's split proposal is explicitly not frozen and lacks manual family grouping.
2. **Final-set execution:** no approved locked test hidden states/labels are present for applying frozen probes; the P3 synthetic runs intentionally use dev rows only.
3. **P5 cross-method representation comparison:** no P5 method activation extraction is present, so the required base vs standard SFT vs proposed-method comparison cannot be completed yet.
4. **Shared handoff agreement:** P6/P2 need to confirm the prompt, grader, dataset version, and group assignments before these probes feed final cross-track claims.

These are external handoffs rather than pipeline defects. Do not interpret the exploratory SFT AUROC or bootstrap interval as final evidence until the labels, split, and checkpoint set are frozen and the locked evaluation is run.
