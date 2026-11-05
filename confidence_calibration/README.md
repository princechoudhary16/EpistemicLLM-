# Confidence and calibration

Implements the confidence/calibration track (P4 in `plan.md`). It asks one question: which cheap, output-level confidence signal best predicts whether an LLM answer is correct, and how well calibrated is it? Results and their interpretation are in [`reports/confidence_calibration.md`](../reports/confidence_calibration.md).

## Design

- **One protocol.** The model pin, float32 precision, prompt and decoding come from `configs/`; correctness comes from `common.grading`; metrics come from `common.metrics`. This package holds no second copy of any of them.
- **One scoring path.** Responses are generated once by the team's shared runner. A separate teacher-forced pass (`scoring.py`) computes every signal from the model's raw logits, so any producer's responses can be scored with the matching model, and the result is checked against the values recorded at generation time.
- **Two stages, so held-out rows cannot influence a choice.** `calibrate` drops held-out rows by split label before reading any outcome or signal, then writes every choice into `frozen_policy.json`. `evaluate` applies that file unchanged, and refuses to run twice. Tests prove that scrambling held-out rows leaves the policy byte-identical.

| Module | Responsibility |
|---|---|
| `config.py` | reads the shared configs; signal version; coverage targets |
| `data.py` | JSONL I/O; `load_responses` validates provenance and regrades with the shared grader |
| `modeling.py` | device selection, float32 model/adapter loading, prompt encoding |
| `signals.py` | log-probability and entropy statistics from raw logits |
| `scoring.py` | teacher-forced scoring of any response file; agreement with generation |
| `consistency.py` | per-question sample agreement, cluster statistics, knowledge score |
| `calibration.py` | split roles, Platt calibrators as plain numbers, threshold selection |
| `evaluation.py` | every metric for one signal on one split |
| `policy.py` | the two stages: `fit_policy` and `evaluate_policy` |
| `error_analysis.py` | high-confidence wrong answers and their patterns |
| `method_comparison.py` | methods on the same questions at matched coverage |
| `calibration_report.py` | Markdown tables and dependency-free SVG figures |
| `jobs/tc2_eval.sbatch` | SLURM job for the TC2 GPU cluster |

## Signal definitions (`signals-v1`)

All statistics use the model's **unmodified** next-token distribution: float32 logits (`configs/model.yaml`), temperature 1, no top-k/top-p. That is not the sampling distribution. Answer tokens are the tokens of the generated text; the end token, prompt and padding never enter a statistic.

| Signal | Definition | Confident when |
|---|---|---|
| `avg_logprob` | mean log p(token) over answer tokens; the team's shared `confidence_score` | higher |
| `sum_logprob`, `min_logprob` | sum / minimum of the same | higher |
| `entropy` | mean full-vocabulary entropy (nats) over answer positions | lower |
| `first_token_entropy` | entropy at the first answer position | lower |
| `consistency` | share of the 5 samples whose normalised answer equals the greedy answer | higher |
| `majority_frac` | largest normalised-answer cluster among the samples / 5 | higher |
| `aux_confidence_score` | P5's evidence-head score, where a checkpoint has one | higher |

- **Target and population.** The target being judged is the greedy response. Discrimination is measured on greedy responses that attempted an answer (correct/incorrect), with label = correct. Abstentions and invalid outputs are reported separately and stay in the policy denominators.
- **No circularity.** The samples are independent of the greedy target, so consistency predicting greedy correctness is not circular.
- **Normalised agreement** uses the shared grader's normalisation with articles removed; the shared abstention rule puts all abstentions in one cluster.
- **Knowledge score is descriptive only.** `knowledge_score` = correct samples / valid samples, matching `common.metrics.knowledge_scores`. It shares samples with `consistency`.
- **Nulls are reported.** Empty answers get null signals and are counted, not dropped.

## Usage

**Setup** (Python 3.11, run everything from the repo root):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r confidence_calibration/requirements.txt
```

**Unit tests** (CPU only, no model download):

```bash
.venv/bin/python -m unittest discover -s confidence_calibration/tests -t .
```

**1. Score** a model's responses, for the signals the shared runner does not record. Pass `--adapter` and `--checkpoint-id` for a fine-tuned model. The `.meta.json` beside the output records how closely scoring reproduces generation:

```bash
python -m confidence_calibration score \
    --responses results/raw/p2_triviaqa_v0_train_probe_base.jsonl results/raw/p2_triviaqa_v0_validation_base.jsonl \
    --output results/raw/confidence_calibration/triviaqa_v0_base/scored_responses.jsonl
```

**2. Calibrate** on development rows. This writes `frozen_policy.json` (the hand-off for abstention thresholds), `calibration.json`, `calibration_report.md`, `calibrated_predictions.jsonl`, `confident_errors.csv` and two SVG figures:

```bash
python -m confidence_calibration calibrate \
    --responses results/raw/confidence_calibration/triviaqa_v0_base/scored_responses.jsonl \
    --output-dir results/final/confidence_calibration/triviaqa_v0_base/development
```

**3. Evaluate** held-out rows, once. Nothing is refitted or reselected. It refuses another model or protocol, questions used in development, a missing signal, and an output directory that already holds an evaluation:

```bash
python -m confidence_calibration evaluate \
    --policy results/final/confidence_calibration/main_v0_sft_matched/development/frozen_policy.json \
    --responses results/raw/p1_main_v0_sft_matched.jsonl \
    --output-dir results/final/confidence_calibration/main_v0_sft_matched/held_out
```

`--allow-transfer` applies a policy to another model or dataset under the same prompt, decoding, grader and model pin, as an explicit transfer test.

**4. Compare** methods on the same held-out questions. `--probe-predictions` adds a hidden-state probe on the questions both scored:

```bash
python -m confidence_calibration compare --reference sft_matched \
    --run sft_matched=results/final/confidence_calibration/main_v0_sft_matched/held_out@avg_logprob \
    --run p5_lam3=results/final/confidence_calibration/main_v0_p5_lam3/held_out@avg_logprob \
    --output-dir results/final/confidence_calibration/main_v0_comparison
```

Supported split labels are `train_probe` / `validation` / `locked_test`, `cal_train` / `cal_val` / `cal_test`, `train` / `val` / `test`, and P1's `dev` / `final`. Dev **facts** are divided 70/30 into train/validation within each exposure level, and paraphrases of a fact stay together. Each policy holds a threshold per coverage target (default 90, 80, 70, 60 and 50%, with 80% primary).

`build-dataset` and `data/triviaqa/validation_sample_n300_seed0.jsonl` remain only because `p2_real_qa/data.py` keeps those questions out of every shared split. `results/raw/confidence_calibration/triviaqa-n300-base-mps/` used a provisional prompt and grader and is superseded; the code that produced it is at commit `3948aaf`.

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
5. **Submit** from the repo root. For the locked test, once its responses exist:
   ```bash
   mkdir -p logs && NAME=triviaqa_v0_base RESPONSES=results/raw/p2_triviaqa_v0_locked_test_base.jsonl \
       POLICY=results/final/confidence_calibration/triviaqa_v0_base/development/frozen_policy.json \
       sbatch confidence_calibration/jobs/tc2_eval.sbatch
   ```
6. **Monitor.** Use `squeue -u $USER` and `tail -f logs/confidence_eval_<jobid>.out`. After the job, check `seff <jobid>`.
7. **Copy results back** with `scp -r` from `results/final/confidence_calibration/<name>` and `results/raw/confidence_calibration/<name>`.
