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

## Results (`main_w1`, λ=3 fixed, 3 seeds, final split unless stated)

Provenance: the protocol and criteria above were committed after the job was submitted but **before any `main_w1`
result had been seen** by me or the reviewers. The first job (45479) trained and evaluated everything but crashed in the
probe step (a missing environment activation in my own script, fixed; no results depended on it); job 45886 re-ran
only the probe step. No code, λ, threshold or probe setting was changed between `main_v0` and `main_w1`.

### Verdict on the five criteria

| # | Criterion | `main_w1` result | Verdict |
|---|---|---|---|
| 1 | λ=3 vs λ=0 accuracy within 2 points | final 45.1% vs 46.1% (−1.0); dev 46.8% vs 46.4% (+0.4) | **replicates** |
| 2 | λ=3 head AUROC ≥ 0.80, λ=0 head ≤ 0.60 | 0.863 vs 0.497 (dev 0.878 vs 0.489) | **replicates** |
| 3 | probe on λ=3 model beats probe on SFT model by > 0.05 | 0.835 vs 0.707 (+0.128; dev 0.839 vs 0.726, +0.113); base-model probe at chance (0.482) | **replicates** |
| 4 | log-prob beats the head for abstaining at penalty 1 and 3 | net 0.374 / 0.371 vs 0.230 (penalty 1), 0.326 / 0.327 vs 0.172 (penalty 3); AURC 0.202 / 0.208 vs 0.281 | **replicates** |
| 5 | the head ranks never-trained 0× facts above 1×/2× facts (AUROC < 0.5) | 1×/2× vs 0×: head 0.351 (dev 0.379); probe on λ=3 model 0.399; log-prob at chance (0.476 / 0.524) | **replicates** (the flaw is real) |

### Side by side (final split, mean over 3 seeds)

| | `main_v0` | `main_w1` |
|---|---|---|
| accuracy λ=0 / λ=3 | 47.4% / 47.4% | 46.1% / 45.1% |
| head AUROC ≥5× vs ≤2×, λ=0 / λ=3 | 0.523 / 0.848 | 0.497 / 0.863 |
| fact-disjoint probe AUROC: λ=3 model / SFT model / base model | 0.807 / 0.680 / 0.472 | 0.835 / 0.707 / 0.482 |
| log-prob AUROC (≥5× vs ≤2×): P5 model / SFT model | 0.902 / 0.888 | 0.919 / 0.908 |
| head AUROC, 1×/2× vs 0× | 0.349 | 0.351 |
| head AUROC, 5× vs 1×/2× | 0.668 | 0.703 |

Abstention (thresholds from dev, penalty 1, net score / AURC ↓): SFT+log-prob 0.364 / 0.200 → 0.374 / 0.202;
P5+log-prob 0.352 / 0.202 → 0.371 / 0.208; SFT+probe 0.112 / 0.380 → 0.146 / 0.346; P5+probe 0.197 / 0.292 →
0.214 / 0.288; P5+head 0.222 / 0.285 → 0.230 / 0.281. Penalty 3 net scores (`main_v0` → `main_w1`): SFT+log-prob
0.288 → 0.326, P5+log-prob 0.313 → 0.327, SFT+probe 0.017 → 0.052, P5+probe 0.105 → 0.128, P5+head 0.122 → 0.172.
Without abstention both worlds hallucinate on 53–55% of questions. Files: `results/final/p5_main_w1_*`.

### What this does and does not show
- All five `main_v0` findings hold on a second world with different names and a different exposure assignment, with
  the λ chosen on `main_v0` only. The numbers agree closely, so the effects are not an accident of one world.
- It is **not** evidence about real questions, a different generator design, other model sizes or other graders:
  both worlds come from the same generator and the same placeholder grader, and the seeds are 3 per world.
- The thresholds in the criteria table were mine; the margins are comfortable (e.g. the probe gap is +0.13 against
  a bar of +0.05), so none of the verdicts is close to a borderline.
- The conclusion in `p5_evidence_aware_training.md` stands unchanged: the auxiliary loss makes the evidence
  signal more readable at no accuracy cost, but the model's own log-probability remains the better abstention signal.
