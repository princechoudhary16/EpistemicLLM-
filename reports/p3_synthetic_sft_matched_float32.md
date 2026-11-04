# P3 synthetic `sft_matched` float32 probe

**Status:** Exploratory development result. This run uses the 600-question `dev` split only; no `final` questions were loaded.

## Run

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Adapter: Sahana's float32 `sft_matched`, training seed 0; adapter model SHA-256 `6ed089468ba45df2a3f6b05cd5e34a0c25b0f83e965d482dc6d255e37d3fc8db`.
- Runtime: PyTorch 2.14.1, Transformers 5.20.0, PEFT 0.21.2, Python 3.13.13; MPS device.
- Generation: shared `common.model.ModelRunner`, greedy decoding, `top_k=0`, `repetition_penalty=1.0`, maximum 64 new tokens.
- Labels: generated and graded in the same P3 run using `p2_whole_word_v5`; 600 questions across 300 fact groups. Grouped split: 450 train questions and 150 validation questions; no fact crosses the split.

## Probe results

The target is greedy answer correctness. Features are last non-padding prompt-token representations with 896 dimensions at each listed layer. Probes use a training-only StandardScaler and balanced logistic regression (`C=1`, threshold 0.5); the split was fixed before fitting.

| Layer | Validation AUROC | Validation AUPRC | Balanced accuracy |
|---:|---:|---:|---:|
| 6 | 0.402 | 0.466 | 0.478 |
| 12 | 0.569 | 0.552 | 0.535 |
| 18 | 0.722 | 0.739 | 0.648 |
| 24 | 0.489 | 0.520 | 0.522 |
| Prompt length baseline | 0.517 | 0.506 | 0.526 |

The training split has 221 correct and 229 not-correct labels; validation has 72 correct and 78 not-correct labels. Layer 18 performs best in this single split, but this is exploratory evidence and does not show that layer 18 is reliably best or that its representations cause model behavior.

## Artifacts

- Run directory: `results/p3/synthetic_main_v0_sft_matched_seed0_float32_modelrunner/`
- Feature SHA-256: `97ed5e56a77e6308e711edd05ce3c02084c75a5c17e65c99a9ee1ab69e56b8fa`
- Label SHA-256: `6ca94b7318355d87360b7f0c34e1b3d827125f4eadfeace09ad0fa8081126fa2`
- Probe metrics: `results/p3/synthetic_main_v0_sft_matched_seed0_float32_modelrunner/probe/metrics.json`
