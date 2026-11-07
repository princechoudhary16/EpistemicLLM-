# Real-data abstention baselines (R-Tuning and GeoDe style)

**Owner:** P2. **Status:** exploratory extension, protocol written before any result.

## Why

The project's "I don't know" baseline (`p6_baselines/run_idk_shuffled.py`) adds 30 refusal examples about unrelated
invented names to 3,800 synthetic facts and abstains on 0–1% of questions, so the comparison the plan requires
("does the method beat simply teaching the model to say I don't know?") has no working baseline. The abstention methods
that work in the literature (R-Tuning; GeoDe, the paper the proposal positions itself against) instead label **real**
questions by whether the model itself gets them right, and train "I don't know" on the ones it gets wrong. This folder
reproduces that at 0.5B on TriviaQA, and adds the comparison the proposal promised: remove the gray zone (GeoDe) or not.

## Arms (identical recipe; only the training rows differ)

| Arm | Training rows (from the same 20,000-question pool D1) | Target |
|---|---|---|
| `rtuning` | all of D1 | base model's greedy answer correct → reference answer; wrong or abstained → "I don't know." |
| `geode` | the 25% of D1 farthest from the probe's hyperplane (largest \|d\|) | d > 0 → reference answer; d < 0 → "I don't know." (GeoDe TBG) |
| `random` | a random 25% of D1 | as `rtuning` (controls for GeoDe's smaller set) |

Recipe: `sft_matched`'s LoRA settings (r 16, alpha 32, dropout 0.05, all 7 linear layers, batch 4, lr 2e-4, 1 epoch),
float32, shared system prompt, answer-only loss, rows shuffled by seed. Probe: L2 logistic regression (C = 1) on the
final-layer hidden state of the last prompt token, standardised on D0; labels = base model's greedy correctness on D0.

## Data

- Questions: TriviaQA `rc.nocontext` **train** file, same pinned revision as the shared release (`0f7faf33`). The
  shared splits come from the *validation* file.
- Removed before sampling: every question whose wording **or** canonical answer group appears in any shared TriviaQA
  split (pilot, train_probe, validation, locked_test) or in P4's 300-question sample. Checked again after sampling
  (`no_overlap_with_evaluation`).
- D0 = 5,000 questions (labels for the probe), D1 = 20,000 (fine-tuning pool); disjoint; seed 20261107.
- IDs are `triviaqa_train-<official id>`, so they can never collide with evaluation IDs.

## Protocol (written before any result; do not change after seeing one)

1. All choices (arms, recipe, X = 25%, C = 1, 1 epoch) are fixed above. Development uses the **validation** split
   (200 questions) only. The **locked test** is run once, after the team's freeze, for every arm and the base model.
2. Seeds: one seed (0) for development; the final comparison uses seeds 0, 1, 2 (mean ± sd).
3. Reported for every arm, greedy answers, same questions: GeoDe's F1_ans / F1_abs / F1_rel (known/unknown defined by
   the base model's greedy answer to each question) **and** the team's accuracy, hallucination rate, abstention rate,
   coverage and selective accuracy. Hallucination is never reported without coverage.
4. Questions this answers (stated as hypotheses, not expected results):
   - **H1** abstention fine-tuning on real data lowers hallucination relative to the base model (expected to cost coverage);
   - **H2** GeoDe's F1_rel exceeds R-Tuning's (the paper's claim, here at 0.5B);
   - **H3** GeoDe's F1_rel exceeds the `random` control's (the gain comes from *which* questions, not from fewer of them).
   Each difference is reported with a 95% paired question-bootstrap interval; an interval containing 0 is reported as
   "no difference shown".
5. Known caveat, stated in advance: the base model gets about 20% of TriviaQA right, so about 80% of R-Tuning's targets
   are "I don't know." Heavy abstention is the expected outcome, which is why results are read at matched coverage.
6. Probe variant for the `geode` arm, chosen from the D1 probe reports only (before any training or validation run):
   - layer: the one with the higher `apply_auroc` (plain fits); within 0.01 → final layer (the paper's setting);
   - weighting: at that layer, the higher `kept.balanced_accuracy`; within 0.01 → plain (the paper's setting).
   The chosen file is copied to `results/abstention/geode_distances.jsonl`; the other three reports stay for the write-up.

## Pipeline

From the repository root (on TC2: `module load anaconda && conda activate epistemic` first; see
`confidence_calibration/README.md` → "TC2 GPU cluster" for set-up and the pre-downloaded model).

```bash
# 1. data (laptop or TC2 head node; needs network once)
python -m p2_abstention.data --download data/raw/triviaqa_rc_nocontext_train_0f7faf33.jsonl
python -m p2_abstention.data                          # all checks must PASS

# 2. labels: base-model greedy answers, graded with v5 (TC2, resumable)
mkdir -p logs
SET=probe sbatch p2_abstention/jobs/label.sbatch
SET=sft   sbatch p2_abstention/jobs/label.sbatch

# 3. GeoDe probe and distances (after both label jobs)
sbatch p2_abstention/jobs/probe.sbatch

# 4. training sets (seconds; laptop or TC2)
python -m p2_abstention.build_sets --questions data/abstention/triviaqa_train_v0/sft.jsonl \
    --responses results/raw/p2_abstention_sft_base.jsonl \
    --distances results/abstention/geode_distances.jsonl \
    --out-dir data/abstention/triviaqa_train_v0/sets

# 5. train + validation answers, one job per arm (seed 0 for development)
for ARM in rtuning geode random; do ARM=$ARM SEED=0 sbatch p2_abstention/jobs/train.sbatch; done

# 6. compare on validation
python -m p2_abstention.evaluate --base results/raw/p2_triviaqa_v0_validation_base.jsonl \
    --run rtuning=results/raw/p2_triviaqa_v0_validation_abstention_rtuning_seed0.jsonl \
    --run geode=results/raw/p2_triviaqa_v0_validation_abstention_geode_seed0.jsonl \
    --run random=results/raw/p2_triviaqa_v0_validation_abstention_random_seed0.jsonl \
    --reference rtuning --out results/final/abstention/validation_seed0.json
```

Not committed (large, rebuilt from pinned sources; see `.gitignore`): the raw train snapshot, hidden-state features,
adapters. Committed: question pools, audit, labels, distances, training sets, summaries, response files, comparisons.

## Files

| File | Purpose |
|---|---|
| `data.py` | download the train split; build D0/D1 with evaluation overlap removed; audit |
| `probe.py` | hidden-state features (`extract`); logistic probe and signed distances (`fit`), with diagnostics |
| `build_sets.py` | the three arms' training files and `sets_summary.json` |
| `train.py` | LoRA training for one arm and seed; `train_meta.json` beside the adapter |
| `evaluate.py` | GeoDe metrics + shared metrics, paired bootstrap against a reference arm |
| `jobs/*.sbatch` | TC2 jobs: `label`, `probe`, `train` (+ validation answers) |
| `tests/test_abstention.py` | everything above on fake data, no GPU |

`p2_real_qa/generate_responses.py` gained `--n-samples` (0 = greedy only, used for labelling); the decoding id records
it (`decoding_v1_greedy_only`), so these rows cannot be mixed up with the shared protocol's.

## Amendment 1 (7 Nov 2026): K(q) arms, declared after the seed-0 validation results, before any K arm is run
Seed 0 showed every arm over-refusing, and selective accuracy no higher than base. These arms test the proposal's
knowledge-continuum idea and are a new experiment, not a change to the arms above (whose protocol is unchanged).
- Labels: base model, greedy + 5 samples (decoding_v1), on a random 8,000 of D1 (`sft_k8000.jsonl`, seed 20261107).
  K = correct / valid samples (common.metrics.knowledge_scores).
- Arms (`build_k_sets.py`), each cut to 5,000 rows to match geode/random:
  `kfilter` (K ≥ 0.8 answer, K = 0 IDK, rest dropped), `ksoft` (answer with probability K), `kthresh` (K > 0.5 answer).
- Same recipe, seeds, validation and locked-test rules and metrics as above.
- Hypotheses: H4 kfilter's selective accuracy exceeds rtuning's and geode's; H5 ksoft's F1_rel exceeds kfilter's
  (keeping the gray zone helps). Caveat: greedy decoding turns ksoft into roughly "answer if K > 0.5".

## Amendment 2 (7 Nov 2026): own-answer targets, declared after seeds 0–2, before any own-answer run
Seeds 0–2 lost most known answers (N2 + N3 = 35–46 of 47), and the wrong answers were short, plausible but wrong
facts ("Boise" for Washington's capital) where the base model had answered correctly in a full sentence. Hypothesis:
short reference targets remove the sentence the 0.5B model uses to recall the answer.
- Same arms, questions, labels, probe distances, recipe and seeds; only the known-row target changes
  (`build_sets --target own`): the base model's own correct greedy answer. Known rows with no usable own answer
  (cut off, or geode's probe says known but the greedy answer was wrong) are dropped.
- Prediction: N2 falls and N1 rises compared with the reference-target arms, at similar hallucination.
