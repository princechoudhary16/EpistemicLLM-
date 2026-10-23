# P5 — second synthetic world (`main_w1`): protocol and replication criteria

Status: **protocol written before any `main_w1` result exists.** Results section is filled in afterwards.

## Purpose
All `main_v0` results (reports/p5_evidence_aware_training.md) come from one generated world, shared by all three
seeds. P1 generated a second world (`data/synthetic/main_w1/`, PR #9, `python -m p1_exposure.generate --n-facts 600
--seed 1 --version main_w1`): same design (600 facts, 100 per exposure level, 3,800 training rows, 1,200 eval
questions, dev/final split by fact), different invented names (0 of 600 shared) and a different fact-to-exposure
assignment (119/600 facts keep the same exposure, about chance). This repeats the P5 protocol on it to test whether
the `main_v0` findings are robust or specific to that one world.

## Protocol (no re-tuning)
- Same code, same config (`config.yaml`), same recipe. **λ is fixed in advance at 3** (chosen on `main_v0` dev only);
  λ=0 is the plain-SFT control through the same code path. No λ search, no threshold or probe-setting changes.
- Seeds 0, 1, 2 for both λ. Train on `main_w1/train.jsonl` (file order, one epoch).
- Dev split scored first, final split scored once (`--allow-final`), exactly as for `main_v0`.
- Fact-disjoint probes (fit on the other split's facts), same fixed probe settings.
- Abstention policies: thresholds chosen on dev by `accuracy − penalty·hallucination` (penalty 1 and 3), applied once to final.
- Run: `DATA_DIR=data/synthetic/main_w1 sbatch p5_training_method/jobs/p5_world.sbatch`, then
  `aggregate_sweep.py --dataset main_w1`, `abstain_eval.py` and `regrade_check.py --dataset main_w1`
  with `p5_main_w1_*` output names so no `main_v0` result is overwritten.

## What would count as "replicates" (my thresholds, written before the run)
Each is the `main_v0` finding restated; a miss is reported as a miss, not explained away.

| # | `main_v0` finding | Replicates on `main_w1` if (final split, mean over 3 seeds) |
|---|---|---|
| 1 | λ=3 costs no accuracy | accuracy of λ=3 and λ=0 differ by ≤ 2 points |
| 2 | The head carries a signal; λ=0 head does not | λ=3 head AUROC (≥5× vs ≤2×) ≥ 0.80, λ=0 head ≤ 0.60 |
| 3 | The aux loss makes exposure more readable than plain SFT (fact-disjoint probes) | probe on λ=3 model beats probe on SFT model by > 0.05 AUROC |
| 4 | The model's own log-prob beats the head for abstaining | log-prob net score and AURC better than the head's at penalty 1 **and** 3 |
| 5 | The head ranks never-trained 0× facts above 1×/2× facts (a flaw) | AUROC for 1×/2× vs 0× is below 0.5 for the head |

## Results
_Pending._
