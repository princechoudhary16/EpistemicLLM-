# P3: hidden-state and probe pipeline

**Implemented:** hidden-state extraction and grouped probes for the pinned Qwen model. Earlier SimpleQA and synthetic runs are exploratory; use the shared TriviaQA procedure below for comparable P3 results.

Probes estimate what is decodable from prompt representations. They do not show that a representation causes model behavior. The TriviaQA workflow uses P2's frozen train/validation sets and labels graded by `common/grading.py` v5; the locked test is never loaded.

## What is implemented

| File | Purpose |
|---|---|
| `download.py` | Download pinned public assets; verify SHA-256 checksums |
| `prepare.py` | Select 100 unique SimpleQA questions with seed 42 before model inference |
| `extract.py` | Load frozen Qwen with an optional PEFT LoRA adapter, capture four pooled prompt states, generate greedy responses, label with the shared grader |
| `apply_reviews.py` | Apply explicit corrections tied to the exact original response file |
| `probe.py` | Join labels by ID, split by group, fit and reload probes, write metrics and validation scores |
| `bootstrap_probe.py` | Compute fact-cluster percentile intervals for a frozen validation probe; no probe refitting or model selection |
| `tests/test_pipeline.py` | Offline pooling, layer mapping, ID alignment, leakage, and degenerate-label checks |

## Run on this workspace

Run commands from the repository root. The isolated `.venv-p3` environment and model/data downloads already exist locally. No API key or paid judge is needed.

```bash
# Verify the cached public assets and run offline regression checks.
.venv-p3/bin/python -m p3_xai.download --verify-only
.venv-p3/bin/python -m unittest discover -s p3_xai/tests -v

# Tiny real-model run. Output directories must be new/empty.
.venv-p3/bin/python -m p3_xai.extract --limit 4 --output-dir results/p3/my_smoke4
.venv-p3/bin/python -m p3_xai.probe --run-dir results/p3/my_smoke4

# Full 100-question reproduction using the existing frozen selection.
.venv-p3/bin/python -m p3_xai.extract --output-dir results/p3/week1_rerun
.venv-p3/bin/python -m p3_xai.probe --run-dir results/p3/week1_rerun
```

The tiny split may have only one training class, in which case the probe records a skipped fit. The 100-question run did fit all four probes. Neither command retries splits until it gets a favorable result. Use another output directory for another run; previous evidence is preserved.

## Fresh environment

Use **Python 3.11–3.13**; the tested interpreter is Python 3.13.13 on Apple Silicon. The machine's default Homebrew Python 3.14 was not used. On this machine, the interpreter used to create the environment was `/opt/homebrew/Caskroom/miniconda/base/bin/python3`.

```bash
# Replace python3.13 with your installed Python 3.11–3.13 interpreter if necessary.
python3.13 -m venv .venv-p3
.venv-p3/bin/python -m pip install -r p3_xai/requirements-lock.txt
.venv-p3/bin/python -m p3_xai.download
.venv-p3/bin/python -m p3_xai.prepare
.venv-p3/bin/python -m p3_xai.extract
.venv-p3/bin/python -m p3_xai.probe
```

`requirements.txt` pins the four main libraries; `requirements-lock.txt` records the complete tested environment. Downloads include about 1 GB of model weights, tokenizer/config files, and the SimpleQA CSV. All network access is confined to downloads/installations; model inference and grading run locally. The downloader refuses changed source content rather than silently updating the dataset snapshot. Model weights, virtual environments, caches, feature arrays, and serialized probes are ignored by Git; the latter two remain available locally and can be regenerated or shared through the team's artifact storage.

## Representation definition

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- The loaded model has 24 decoder blocks and hidden width 896. Fractions 25%, 50%, 75%, 100% map to **depths 6, 12, 18, 24**.
- These are **one-based depths**, matching Hugging Face `hidden_states[depth]`; index 0 would be embeddings. Intermediate vectors are decoder-block outputs; depth 24 includes the model's final RMS normalization.
- Input is the system instruction plus the question, formatted with the tokenizer's chat template and assistant generation prefix. Neither the reference answer nor the generated answer is part of the extraction input.
- Pool at the last non-padding token of the full prompt, including that generation prefix. Find its index from the attention mask; `sum(mask)-1` would be wrong for left padding.
- Selected hooks copy only pooled vectors. Hooks are removed before generation; the model runs in evaluation/inference mode with all parameters frozen.
- Each layer array has shape `[question_count, 896]`, saved as float32. IDs and group IDs are stored as non-object arrays, so loading does not require pickle.
- Reference answers are used only after generation for provisional grading.

The [model configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json) defines the architecture. The [Qwen2 documentation](https://huggingface.co/docs/transformers/model_doc/qwen2) describes model outputs; the tests independently verify hook results against `output_hidden_states=True` on a tiny Qwen2 model.

## Data and label contract

`data/p3/week1_questions.jsonl` contains:

```json
{"question_id": "simpleqa-00912", "group_id": "39c162ccdd52e8b7", "source_dataset": "SimpleQA", "dataset_version": "<CSV SHA-256>", "split": "development", "question": "<question>", "reference_answer": "<reference>"}
```

The selection manifest records the CSV checksum and row IDs. IDs are stable only with that dataset snapshot. Exact normalized duplicate questions are grouped; P2 must group related facts/paraphrases when supplying richer data. The runner rejects non-development rows and inconsistent groups for identical prompts. Reserve these 100 IDs from the eventual locked test before the team creates its final split.

Default labels compare the entire normalized output to the reference. They normalize case, Unicode, whitespace, and terminal punctuation, without substring matching or removing internal decimal points. Nonmatches can be semantically correct, so **these are provisional exact-match proxies, not official SimpleQA grades**. Empty/truncated outputs are kept in predictions with null correctness and excluded, explicitly, from probe fitting. Ten literal IDK outputs were recognized in the completed run; less literal refusals still need P2 review.

The [official SimpleQA evaluator](https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py) uses a model-based grading rubric. This P3 smoke pipeline does not reproduce that grading or report official benchmark scores.

### Replace labels without extracting again

Provide one JSONL row for every extracted question, including null for unresolved labels:

```json
{"question_id": "simpleqa-00912", "correct": 1, "outcome": "correct", "review_status": "p2_audited", "grading_version": "p2-v1", "run_id": "week1", "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775", "prompt_version": "p3-week1-provisional-v1"}
```

Allowed outcome/label pairs: `correct/1`, `incorrect/0`, `abstain/0`, `invalid/null`. The labels must concern **these exact greedy responses** under the same model and prompt, not another generation or K-score target. Missing, duplicate, extra, inconsistent, and mismatched-run labels cause errors.

```bash
.venv-p3/bin/python -m p3_xai.probe \
  --run-dir results/p3/week1 \
  --labels path/to/p2_audited_labels.jsonl \
  --output-dir results/p3/week1/probe_p2_audited
```

For the existing run, one obvious false negative is recorded in `data/p3/week1_label_reviews.json`: a sentence correctly naming Billy Graham failed whole-string exact matching. `apply_reviews.py` generated `labels_reviewed.jsonl`; original predictions/labels remain unchanged. This is an assistant correction, **not** a completed human/P2 audit. The correction is bound to the original response-file checksum and must not be blindly applied to a new run.

## Probe definition and safeguards

- One fixed `GroupShuffleSplit`, seed 42, with 25% validation groups. Split before dropping null labels and without looking at class labels. All variants of a group remain together.
- `StandardScaler` fits only on training rows, followed by logistic regression (`C=1`, balanced class weights, maximum 2,000 iterations). No hyperparameter/layer selection this week.
- Target: whether the independent greedy answer is correct, using provisional labels for now. This is not a knowledge-continuum/K-score probe yet.
- Report training and validation class counts, accuracy, balanced accuracy, AUROC, and AUPRC. Single-class validation gives N/A for discrimination metrics. Single-class training skips fitting and records why.
- Include a training-majority baseline and a prompt-length baseline. The saved sigmoid outputs are **not calibrated confidence estimates**.
- Save/reload each fitted probe and check that its validation predictions are unchanged. Only load trusted locally generated joblib artifacts.

## Outputs and handoff

| Artifact | Contents |
|---|---|
| `results/p3/week1/manifest.json` | Model/config/data hashes, environment, depths, pooling, device, timings, array dimensions |
| `results/p3/week1/features.npz` | Four feature matrices, question/group IDs, prompt lengths; local, ignored by Git |
| `results/p3/week1/predictions.jsonl` | All 100 greedy outputs, references, provisional labels, truncation flags |
| `results/p3/week1/labels.jsonl` | Original automatic provisional labels |
| `results/p3/week1/labels_reviewed.jsonl` | Same labels with one documented assistant correction |
| `results/p3/week1/probe/` | Original-label probe results |
| `results/p3/week1/probe_reviewed/` | Corrected-label results, grouped split, validation scores, local serialized probes |
| `reports/p3_week1.md` | Completion checklist, actual observations, and next-week handoffs |

P2 should audit the response labels and provide richer label diversity. P6 should review the provisional prompt/model configuration for alignment with shared runs. Re-extract if the model, prompt, or questions change; re-fit only if labels change for the same responses. P4 can consume the ID-linked validation scores once label quality is sufficient.

## P3 on shared TriviaQA splits

The team decision is float32 for training and evaluation, matching `configs/model.yaml`. These configs use the shared `prompt_v0`, P2's frozen `train_probe`/`validation` question files, and never load `locked_test`.

The extraction writes one greedy answer and grades it with `common/grading.py` v5. Use the labels from that same extraction when fitting the probe so answers and prompt features come from the same inference runtime:

```bash
python -m p3_xai.prepare_triviaqa --method base --output-dir results/p3/triviaqa_v0_base_inputs
python -m p3_xai.extract --config configs/p3_triviaqa_v0_base.json
python -m p3_xai.probe --run-dir results/p3/triviaqa_v0_base_features

# Set adapter_path in the config to the P6-supplied float32 sft_matched adapter.
python -m p3_xai.extract --config configs/p3_triviaqa_v0_sft_matched.json
python -m p3_xai.probe --run-dir results/p3/triviaqa_v0_sft_matched_features
```

Install the pinned P3 environment with `python -m pip install -r p3_xai/requirements-lock.txt`. Copy P6's float32 adapter to `.cache/p3/sft_matched_main_v0_adapter` or update `adapter_path` in the SFT config. The tested runtime is PyTorch 2.14.1, Transformers 5.20.0, PEFT 0.21.2, Accelerate 1.15.0, and Datasets 5.1.0.

P3 uses the shared `common.model.ModelRunner` for greedy answer generation, with `top_k=0` and `repetition_penalty=1.0` from the P2 decoding configuration. On the frozen 700 base questions, P3's generated answer strings match P2's saved greedy strings exactly. Two truncated responses have different outcome fields: P3 marks them invalid for review, while P2's raw response rows retain graded outcomes even though `truncated=true`. The current exploratory results are recorded in [the TriviaQA probe report](../reports/p3_triviaqa_probe.md).

## P3 development on P1 synthetic data

The SimpleQA pilot has too few positive correctness labels to support a useful probe. PR #7's P2 pilot also reports 198/200 questions at `K=0`, so it does not solve that problem. The project plan names the synthetic exposure experiment as the fallback for P3/P4 continuum analysis. A dev-only adapter is provided for that path:

```bash
.venv-p3/bin/python -m p3_xai.prepare_synthetic
.venv-p3/bin/python -m p3_xai.extract --config configs/p3_synthetic_main_v0.json
.venv-p3/bin/python -m p3_xai.probe \
  --run-dir results/p3/synthetic_main_v0_base_prompt_v0
```

For the matched SFT representation, use Sahana's float32 seed-0 adapter. P3 generation uses the shared `ModelRunner` with `top_k=0` and `repetition_penalty=1.0`. Run on the 600 dev questions only, with a fresh output directory:

```bash
.venv-p3-aligned/bin/python -m p3_xai.extract \
  --config configs/p3_synthetic_main_v0_sft_matched.json \
  --adapter-path new_weights \
  --output-dir results/p3/synthetic_main_v0_sft_matched_seed0_float32_modelrunner
.venv-p3-aligned/bin/python -m p3_xai.probe \
  --run-dir results/p3/synthetic_main_v0_sft_matched_seed0_float32_modelrunner
```

The prepared input reads only `split=dev` rows from P1's `data/synthetic/main_v0/eval.jsonl`, preserves question/fact IDs and exposure levels, and uses `fact_id` as the probe grouping key so alternate templates cannot cross the grouped holdout. The extractor rejects every other split, including P1's `final` rows. It records adapter hashes and grades its generated answers with the shared v5 grader. Probe performance remains decodability evidence, not causal evidence about model internals.

The float32 SFT run and its exploratory probe metrics are documented in [the matched-SFT report](../reports/p3_synthetic_sft_matched_float32.md). The earlier base-only synthetic dev run is documented in [the historical run report](../reports/p3_synthetic_dev_probe.md); it found no correct greedy labels and could not fit a probe.

### Fact-cluster uncertainty for the exploratory SFT probe

The reviewed-label layer 18 SFT probe has a fixed grouped train/validation split and no layer search. To summarize validation uncertainty without treating the two templates per fact as independent, run:

```bash
.venv-p3/bin/python -m p3_xai.bootstrap_probe
```

This resamples the 75 held-out fact groups with replacement, keeps each sampled fact's question variants together, and reports percentile intervals over 5,000 two-class draws. It does not refit the probe or select a layer. The saved result is `results/p3/synthetic_main_v0_sft_matched_seed0/probe_reviewed/fact_bootstrap.json`. Since labels are assistant-reviewed and provisional, intervals are descriptive uncertainty for this development split, not final-test confidence intervals.

P2's PR #7 report recommends a different real-QA dataset and keeping SimpleQA as a negative-result baseline. That is a team decision; this synthetic path is independently runnable and does not assume that decision has been approved. Once a shared grader or prompt changes, regenerate/regrade on the same dev questions and record the matching versions before interpreting probe scores.
