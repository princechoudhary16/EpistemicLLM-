# P6 — Baselines, abstention and common evaluation

Shared infrastructure and baselines per `plan.md` section 2 (P6). Owns the model
runner, metrics, standard/IDK SFT baselines, and the final common evaluator.

## What's implemented so far (Week 1)

- `configs/model.yaml`, `configs/decoding.yaml` — pinned model revision, prompt and
  decoding settings (plan.md section 3).
- `common/model.py` — `ModelRunner`: reproducible generation from
  (model revision, decoding config, seed); records package/GPU versions.
- `common/schema.py` — the shared response-level JSONL schema (plan.md section 9).
- `common/metrics.py` (+ `common/tests/test_metrics.py`) — accuracy, hallucination,
  coverage, selective accuracy/risk, knowledge score, AUROC/AUPRC, Brier/ECE,
  threshold policy, risk-coverage curve, bootstrap CIs. Tested on hand-calculated
  fixtures per the Week 1 acceptance criteria.
- `p6_baselines/generate_base.py` + `smoke_fixture.jsonl` — the "one reproducible
  command for generation/evaluation" required by Week 1. Self-contained: does not
  depend on P1/P2 data yet, per the dependency table ("P6 needs: dummy prediction
  files + base model").

- `common/train.py` + `train_sft_smoke.py` + `toy_sft_corpus.jsonl` — the tiny
  standard-SFT sanity run: trains a LoRA adapter on 3 fictional facts the base model
  cannot already know, saves it, reloads it fresh from disk, and confirms the reloaded
  model actually recovers all 3 (verified on the cluster: loss 7.26 → 0.007 over 90
  steps; 0/3 correct before training, 3/3 after). Placeholder LoRA config (r=8,
  `q_proj`/`v_proj` only) — not the frozen main-experiment training settings.

**Not yet done:** the IDK/abstention baseline, and the real grader (P2 owns grading —
the smoke run's `placeholder_exact_v0` grading is only for wiring the pipeline together
and must not be used for any reported result).

## Smoke run

```bash
# locally (CPU is fine for 5 tiny questions, just slow)
python -m p6_baselines.generate_base \
    --fixture p6_baselines/smoke_fixture.jsonl \
    --out results/raw/smoke_base_v0.jsonl \
    --run-id smoke_base_v0 --seed 0

# on the cluster (GPU)
sbatch p6_baselines/job_smoke.sh
```

Produces `results/raw/smoke_base_v0.jsonl` (30 rows: 5 questions x 6 generations)
and prints greedy behavior metrics + per-question knowledge scores to stdout.

## Tiny SFT sanity run

```bash
sbatch p6_baselines/job_sft_smoke.sh
```

Trains a LoRA adapter, saves it under `results/final/toy_sft_adapter/` (not committed
to git — see `.gitignore` — per plan.md section 9's "do not commit model weights"),
and prints before/after answers proving the adapter actually changed behavior.

## Compute environment

See `reports/compute_budget.md` for measured GPU/CPU/memory and the QoS limits
this project must stay within.
