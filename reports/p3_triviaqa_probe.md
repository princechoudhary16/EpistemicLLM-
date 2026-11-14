# P3 TriviaQA probe run

**Status:** Exploratory development results. The probes use P2's frozen questions and splits, greedy responses generated through the shared `common.model.ModelRunner`, and shared grader `p2_whole_word_v5`. They do not use `locked_test`.

## Setup

- Dataset: `triviaqa_v0`; 500 `train_probe` questions and 200 `validation` questions. P2 grouping was retained and no groups cross the split.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; float32 inference on CPU, shared `prompt_v0`.
- SFT: Sahana's `sft_matched` adapter, training seed 0. Adapter file SHA-256: `6ed089468ba45df2a3f6b05cd5e34a0c25b0f83e965d482dc6d255e37d3fc8db`.
- Runtime: PyTorch 2.14.1, Transformers 5.20.0, PEFT 0.21.2, Accelerate 1.15.0, Datasets 5.1.0; `pip check` reported no broken requirements.
- Greedy settings passed through `ModelRunner`: `do_sample=false`, `top_k=0`, `repetition_penalty=1.0`, `max_new_tokens=64`.
- Probe: fixed P2 train/validation split, StandardScaler fitted only on training rows, balanced logistic regression (`C=1`), threshold 0.5. No layer or hyperparameter selection.

## Results

| Checkpoint | Correct labels (train / validation) | Layer | AUROC | AUPRC | Validation balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Base | 87 / 47 | 6 | 0.724 | 0.406 | 0.582 |
| Base | 87 / 47 | 12 | 0.716 | 0.464 | 0.667 |
| Base | 87 / 47 | 18 | 0.735 | 0.536 | 0.671 |
| Base | 87 / 47 | 24 | 0.717 | 0.479 | 0.681 |
| sft_matched | 15 / 6 | 6 | 0.855 | 0.195 | 0.727 |
| sft_matched | 15 / 6 | 12 | 0.900 | 0.314 | 0.578 |
| sft_matched | 15 / 6 | 18 | 0.841 | 0.495 | 0.664 |
| sft_matched | 15 / 6 | 24 | 0.844 | 0.468 | 0.583 |

The base model had 134 correct, 52 abstain, 512 incorrect, and 2 truncated/invalid responses. The SFT model had 21 correct answers out of 700; the other 679 were incorrect. SFT validation AUROC values are based on six positive labels and are highly uncertain. They are not final evidence that one layer is better.

## Generation parity with P2

P3 previously loaded the model directly and called `generate()` without overriding the checkpoint's `repetition_penalty=1.1`. The shared `ModelRunner` explicitly uses `repetition_penalty=1.0` for greedy decoding. P3 now uses that shared runner for answer generation while retaining its prompt-state extraction.

On all 700 shared base questions, P3's answer strings now match P2's saved greedy answer strings exactly (0/700 answer mismatches). Two stored outcome fields differ because those answers hit the 64-token limit: P3 marks truncated answers `invalid` / `needs_review_truncated`; P2's raw response rows contain a graded outcome despite `truncated=true`. This is a label-policy difference, not an answer-generation difference. The P3 probe excludes those two unlabeled rows.

## Artifacts

- Base features, predictions, labels, manifest and probe: `results/p3/triviaqa_v0_base_features_modelrunner_torch214_transformers520/`
- SFT features, predictions, labels, manifest and probe: `results/p3/triviaqa_v0_sft_matched_features_modelrunner_torch214_transformers520/`
- Previous pre-ModelRunner runs remain in their original directories.
- Prepared frozen question file: `results/p3/triviaqa_v0_base_inputs/questions.jsonl`

Describe these results as information **decodable from** prompt representations. They do not establish that the representation causes the model's behavior.
