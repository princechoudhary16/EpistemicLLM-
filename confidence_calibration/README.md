# Confidence and calibration

Implements the confidence/calibration track (P4 in `plan.md`). It asks one question: which cheap, output-level confidence signal best predicts whether an LLM answer is correct, and how well calibrated is it?

## Design

- **One scoring path.** Generation produces only text and token IDs. A separate teacher-forced pass (`scoring.py`) computes every signal from the model's raw logits. That gives one definition per signal, and it lets us score another producer's responses (synthetic exposure runs, fine-tuned checkpoints) with the matching model, without regenerating.
- **Pure core.** `signals.py` and `metrics.py` have no I/O or model code. They are unit-tested against hand-computed values and can move into the shared `common/` package unchanged.
- **Provisional components are labelled.** `generation.py` and `grading.py` stand in for the shared runner and grader until those are merged. Their rows carry `prompt_version` and `grading_version`, and results produced with them are development evidence only.

| Module | Responsibility |
|---|---|
| `config.py` | pinned model revision, prompts, decoding settings, version strings |
| `data.py` | JSONL I/O, question loading, seeded TriviaQA development sample |
| `modeling.py` | device selection, model/adapter loading, chat-prompt encoding |
| `generation.py` | greedy + sampled decoding into schema rows (provisional) |
| `grading.py` | correct / incorrect / abstain / invalid, with `match_type` for audits (provisional) |
| `signals.py` | log-probability and entropy statistics from raw logits |
| `scoring.py` | teacher-forced scoring of any response file |
| `consistency.py` | per-question sample agreement, cluster statistics, knowledge score |
| `metrics.py` | behaviour metrics, AUROC/AUPRC, bootstrap intervals |
| `report.py` | per-run report (`report.md`, `report.json`, `signals.csv`) |
| `calibration.py` | train-only probability mapping, validation selection, locked-test evaluation |
| `calibration_report.py` | calibration tables and dependency-free SVG figures |
| `jobs/tc2_eval.sbatch` | SLURM job for the TC2 GPU cluster |

## Signal definitions (`signals-v0`)

All statistics use the model's **unmodified** next-token distribution: float32 logits (`configs/model.yaml`), temperature 1, no top-k/top-p. That is not the sampling distribution. Answer tokens are the generated tokens before the first `<|im_end|>`/`<|endoftext|>`; the end token, prompt and padding never enter a statistic.

| Signal | Definition | Confident when |
|---|---|---|
| `avg_logprob` | mean log p(token) over answer tokens | higher |
| `sum_logprob`, `min_logprob` | sum / minimum of the same | higher |
| `entropy` | mean full-vocabulary entropy (nats) over answer positions | lower |
| `first_token_entropy` | entropy at the first answer position | lower |
| `consistency` | share of the 5 samples whose normalised answer equals the greedy answer | higher |
| `majority_frac` | largest normalised-answer cluster among the samples / 5 | higher |

- **Target and population.** The target being judged is the greedy response. Discrimination is measured on greedy responses that attempted an answer (correct/incorrect), with label = correct. Abstentions and invalid outputs are reported separately.
- **No circularity.** The samples are independent of the greedy target, so consistency predicting greedy correctness is not circular.
- **Knowledge score is descriptive only.** `knowledge_score` = correct samples / 5 shares its samples with `consistency`.
- **Nulls are reported.** Empty answers get null signals and are counted, not dropped.

## Decoding and prompt

- **Decoding** follows `plan.md` §3: one greedy response plus 5 samples at T=0.7, top-p 0.9, 64 new tokens.
- **Qwen's shipped sampling defaults are overridden.** Qwen2.5 ships `do_sample=True, repetition_penalty=1.1, top_k=20, top_p=0.8`. `generation.py` replaces that config and passes `use_model_defaults=False`, so greedy really is greedy and sampling uses only the settings above.
- **Prompt.** The default is `qa-short-fewshot-v1`: a short-answer instruction that allows "I don't know", plus three generic answer-only demos.
  - The zero-shot `qa-short-v0` made the 0.5B model restate the question in full sentences, which polluted the token statistics.
  - The demos almost eliminate abstention. The shared prompt decision should weigh that.

## Usage

**Setup** (Python 3.11):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r confidence_calibration/requirements.txt
```

**Unit tests** (CPU only, no model download):

```bash
.venv/bin/python -m unittest discover -s confidence_calibration/tests -t .
```

**End-to-end smoke test.** `--verify-scoring` checks that teacher-forced scoring reproduces generation:

```bash
.venv/bin/python -m confidence_calibration run \
    --questions data/triviaqa/validation_sample_n300_seed0.jsonl --run-id smoke-n20 --limit 20 --verify-scoring
```

**Score another producer's graded responses** under their checkpoint:

```bash
python -m confidence_calibration score --responses graded.jsonl --output scored.jsonl \
    --adapter path/to/adapter --checkpoint-id <checkpoint_id>
python -m confidence_calibration report --responses scored.jsonl --output-dir results/raw/confidence_calibration/<name>
```

**Fit and evaluate correctness probabilities** from a single run's graded, scored responses:

```bash
python -m confidence_calibration calibrate \
    --responses results/raw/confidence_calibration/<name>/responses.jsonl \
    --output-dir results/raw/confidence_calibration/<name>/calibration
```

This fits one-dimensional Platt scaling separately for each available signal using only the training
split. It selects the headline signal by validation Brier score and selects each signal's abstention
threshold on validation for the requested coverage (default 80%). The locked test is used once for
the final Brier/ECE, reliability diagram, confident-hallucination and risk/coverage results. The
output contains `calibration.json`, `calibration_report.md`, `calibrated_predictions.jsonl`,
`reliability_diagram.svg` and `risk_coverage.svg`; SVG output requires no plotting dependency.
It also reports locked-test AUROC/AUPRC and cluster-bootstrap intervals alongside Brier/ECE, and
adds descriptive calibration slices by exposure level or independent sampled knowledge score when
those labels exist.

Supported explicit split names are `cal_train` / `cal_val` / `cal_test`,
`train_probe` / `validation` / `locked_test`, or `train` / `val` / `test`. For P1's `dev` / `final`
convention, the command deterministically assigns development **facts** 70/30 to train/validation
and keeps `final` locked as test. If a fact has multiple evaluation paraphrases, they stay together.
The command rejects missing/unsupported split labels, cross-split fact/question IDs, mixed runs,
or a training split without both correct and incorrect attempted answers. Abstentions and invalid
outputs are excluded from correctness-probability fitting, but remain in the behavior and coverage
denominators. The saved probability rows preserve question/fact IDs and split labels for integration.

**Rebuild the TriviaQA sample** (committed; seed 0, pinned dataset revision):

```bash
python -m confidence_calibration build-dataset
```

- Any question JSONL with `question_id`, `question` and `reference_answer` works; `aliases` is optional.
- Calibration accepts only response JSONL with explicit split labels and requires graded outcomes,
  greedy/sample markers, and the confidence signals being evaluated.
- `run` writes `responses.jsonl`, `questions.jsonl`, `report.md`, `report.json`, `signals.csv` and `run_config.json` to `results/raw/confidence_calibration/<run_id>/`.
- A healthy run reports `greedy_argmax_match_rate` ≈ 1.0 and `generation_logprob_max_abs_gap` ≈ 0.

## TC2 GPU cluster

The rules from the CCDS TC2 user guide that matter here:
- Compute only through `sbatch`, never on the head node.
- QoS `normal` allows 1 GPU, 10 CPUs, 30 GB, 6 h and 2 concurrent jobs.
- The home quota is 100 GB, with no backups.

1. **Connect.** Use NTU VPN when off campus, then `ssh <ntu_id>@10.96.189.12`.
2. **Get the code.** Use `git clone` if the head node reaches GitHub. Otherwise, from your machine: `rsync -av --exclude .venv --exclude results ./ <ntu_id>@10.96.189.12:~/EpistemicLLM-/`.
3. **Create the environment** (one time, on the head node):
   ```bash
   module load anaconda
   conda create -n epistemic python=3.11 -y && conda activate epistemic
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -r confidence_calibration/requirements.txt
   ```
4. **Pre-download the model** (about 1 GB). Compute nodes run with `HF_HUB_OFFLINE=1`:
   ```bash
   HF_HOME=$HOME/hf_cache huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
       --revision 7ae557604adf67be50417f59c2c2f167def9a775
   ```
5. **Submit** from the repo root:
   ```bash
   mkdir -p logs && sbatch confidence_calibration/jobs/tc2_eval.sbatch
   ```
   Use `QUESTIONS=… RUN_ID=…` to evaluate another file.
6. **Monitor.** Use `squeue -u $USER` and `tail -f logs/confidence_eval_<jobid>.out`. After the job, check `seff <jobid>`.
7. **Copy results back** with `scp -r <ntu_id>@10.96.189.12:~/EpistemicLLM-/results/raw/confidence_calibration/<run_id> results/raw/confidence_calibration/`.
