# P6 — baselines, abstention and common evaluation

Shared infrastructure and baselines per `plan.md` section 2 (P6). Owns the model
runner, metrics, standard/IDK SFT baselines, and the final common evaluator.
This README is the Week 6 command reference — see
`reports/p6_week6_reproducibility_appendix.md` for the full narrative and
`reports/p6_week5_master_table.md` for the frozen master table.

## Setup

```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate torch      # python 3.11, torch 2.13 (CUDA 12.8), transformers 4.57.6,
                           # peft 0.21.0, accelerate 1.15.0, datasets 5.0.1
module load cuda/12.8.0
export CPATH="/apps/cuda_12.8.0/include:$CPATH"   # Triton's JIT needs cuda.h on the
                                                    # compiler's include path; `module
                                                    # load cuda` alone does not add it.
```

Run the tests first on any fresh checkout:

```bash
python -m pytest common/tests/ p6_baselines/tests/ -q
```

## 1. Smoke test (no dataset needed, ~45s on an A40)

```bash
sbatch p6_baselines/job_smoke.sh
# or locally (CPU, slower):
python -m p6_baselines.generate_base \
    --fixture p6_baselines/smoke_fixture.jsonl \
    --out results/raw/smoke_base_v0.jsonl --run-id smoke_base_v0 --seed 0
```

5 questions x 6 generations, graded, metrics printed. Proves the pipeline works
end-to-end without needing P1/P2's data. **Re-verified Week 6** (job 46319,
COMPLETED, 44s) after a full resync of `common/`/`p6_baselines/`/`configs/` to
confirm nothing drifted between the individual file uploads made across
Weeks 1-5.

## 2. Tiny SFT sanity check

```bash
sbatch p6_baselines/job_sft_smoke.sh
```

Trains a LoRA adapter on 3 fictional facts, saves it, reloads it fresh from
disk, confirms the reload actually recovers the facts (not just the in-memory
trained object). Proves the training+checkpoint pipeline, separately from
generation.

## 3. Canonical baselines (base / standard SFT / IDK)

```bash
# pilot_v0 (120 facts, fast) or main_v0 (600 facts, ~20-90 min depending on load)
sbatch p6_baselines/job_canonical_baselines.sh          # pilot_v0
sbatch p6_baselines/job_canonical_baselines_main.sh      # main_v0

# resume after phase 1 only (skip re-generating an existing base file):
python -m p6_baselines.run_canonical_baselines --dataset-version main_v0 --skip-base
```

LoRA recipe here: r=8, `q_proj`+`v_proj` only, batch 1, lr 2e-4, 1 epoch.
**This recipe is under-capacity** (Week 4 finding, see appendix) — kept for
history, not the headline standard-SFT reference.

## 4. Matched-recipe SFT (the actual standard-SFT reference)

```bash
sbatch p6_baselines/job_matched_sft.sh                           # pilot_v0
DATASET_VERSION=main_v0 sbatch p6_baselines/job_matched_sft.sh    # main_v0
```

LoRA recipe: r=16, alpha=32, dropout=0.05, batch=4, all 7 linear layers
(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`), lr 2e-4, 1 epoch
— P1/Aryaman's published recipe, adopted for training parity with P5's
auxiliary-loss run (`reports/p6_training_parity_fix.md`).

## 5. The common evaluator

```bash
python -m p6_baselines.compare_methods \
    --responses base=results/raw/p1_main_v0_base.jsonl \
                 sft=results/raw/p1_main_v0_sft.jsonl \
                 idk=results/raw/p1_main_v0_idk.jsonl \
                 sft_matched=results/raw/p1_main_v0_sft_matched.jsonl \
                 p5_lam0=results/raw/p5_main_v0_lam0_seed0.jsonl \
                 p5_lam3=results/raw/p5_main_v0_lam3_seed0.jsonl \
    --baseline-label sft_matched \
    --out results/final/baseline_comparison_main_v0_combined.json
```

Validates every method shares identical question_ids, sample indices, and
question/reference metadata (refuses to compare otherwise — this is what
caught the `decoding_v0`/`decoding_v1` mismatch in Week 5). Computes behavior
metrics, knowledge scores, AURC, and paired cluster-bootstrap accuracy diffs.
Add `--confidence-field aux_confidence_score` to rank by a method-specific
signal (e.g. P5's auxiliary head) instead of the default `confidence_score`
(average log-probability — the one signal every method, including the
untouched base model, can produce).

## 6. Risk-coverage plot

```bash
python -m p6_baselines.plot_risk_coverage \
    --responses base=... sft=... idk=... sft_matched=... p5_lam0=... p5_lam3=... \
    --out results/final/risk_coverage_main_v0.png
```

Same `risk_coverage_curve` function the AURC numbers come from — plot and
numbers can't silently disagree. Supports `--confidence-field` the same way.

## 7. Dataset generation (P1's code, run here only because main_v0 didn't exist yet)

```bash
python -m p1_exposure.generate --n-facts 600 --version main_v0
python -m p1_exposure.validate data/synthetic/main_v0
```

## Key results

See `reports/p6_week5_master_table.md` for the frozen master table and plots.
Headline: `sft_matched` (46.8%), `p5_lam0` (48.7%), `p5_lam3` (48.6%) are
statistically indistinguishable from each other, and all three clearly beat
`base` (0.0%) and the under-capacity original `sft`/`idk` recipe.

## Known, documented limitations (not fixed, intentionally)

- Grading is `placeholder_exact_v1` (P6's own text-match grader), not P2's —
  every number in this project is provisional pending that.
- `idk` is a known-degenerate result (100% abstention, a training-order
  artifact P1's ablation traced precisely) — left as-is, not retrained, since
  fixing it would mean introducing new training after the Week 5 freeze.
- Real-QA baselines don't exist — blocked on the team's model/dataset decision
  (`reports/model_dataset_decision.md`).
