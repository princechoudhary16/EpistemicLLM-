# P3 TriviaQA probe run

**Status:** Exploratory development results. The probes use P2's frozen questions and splits, P3-generated greedy responses, and shared grader `p2_whole_word_v5`. They do not use `locked_test`.

## Setup

- Dataset: `triviaqa_v0`; 500 `train_probe` questions and 200 `validation` questions. P2 grouping was retained and no groups cross the split.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; float32 inference on CPU, shared `prompt_v0`.
- SFT: P6 `sft_matched`, training seed 0. Adapter file SHA-256: `6ed089468ba45df2a3f6b05cd5e34a0c25b0f83e965d482dc6d255e37d3fc8db`.
- Runtime: PyTorch 2.14.1, Transformers 5.20.0, PEFT 0.21.2, Accelerate 1.15.0, Datasets 5.1.0; `pip check` reported no broken requirements.
- Probe: fixed P2 train/validation split, StandardScaler fitted only on training rows, balanced logistic regression (`C=1`), threshold 0.5. No layer or hyperparameter selection.

## Results

| Checkpoint | Correct labels (train / validation) | Layer | AUROC | AUPRC | Validation balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Base | 92 / 45 | 6 | 0.706 | 0.370 | 0.612 |
| Base | 92 / 45 | 12 | 0.696 | 0.351 | 0.629 |
| Base | 92 / 45 | 18 | 0.707 | 0.504 | 0.660 |
| Base | 92 / 45 | 24 | 0.712 | 0.448 | 0.650 |
| sft_matched | 15 / 6 | 6 | 0.765 | 0.112 | 0.643 |
| sft_matched | 15 / 6 | 12 | 0.820 | 0.257 | 0.578 |
| sft_matched | 15 / 6 | 18 | 0.786 | 0.395 | 0.747 |
| sft_matched | 15 / 6 | 24 | 0.812 | 0.298 | 0.581 |

The base model had 137 correct, 12 abstain, 9 invalid and 542 incorrect responses. The SFT model had only 21 correct answers out of 700; the other 679 were incorrect. SFT validation AUROC values are based on six positive labels and are highly uncertain. They are not final evidence that one layer is better.

## Runtime comparability

The P3 runs now use the same recorded package versions as P2's float32 run. On the 700 shared questions, however, P3 and P2 produced identical answer strings for 245 questions (455 differed), and their graded outcomes disagree on 78. So matching package versions alone did not reproduce P2's greedy outputs. Keep P3 probes paired with P3-generated labels, as done here; do not present the answer-level results as a reproduction of P2's response metrics. Differences in runtime settings or generation behavior still need to be isolated before treating the responses as directly comparable.

## Artifacts

- Base features, predictions, labels, manifest and probe: `results/p3/triviaqa_v0_base_features_torch214_transformers520/`
- SFT features, predictions, labels, manifest and probe: `results/p3/triviaqa_v0_sft_matched_features_torch214_transformers520/`
- Prepared frozen question file: `results/p3/triviaqa_v0_base_inputs/questions.jsonl`

Describe these results as information **decodable from** prompt representations. They do not establish that the representation causes the model's behavior.
