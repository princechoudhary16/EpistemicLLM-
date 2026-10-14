# Model/Dataset Decision — Qwen2.5-0.5B-Instruct + SimpleQA

**Owner:** P2
**Status:** Warning thresholds triggered — team review required

## Pilot setup

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset: SimpleQA, snapshot `simpleqa_snapshot_2026-09-28.csv`
- Pilot size: 200 questions, fixed seed [PILOT_SEED 20260928]
- Generations: 1 greedy + 5 sampled per question (temperature 0.7, top-p 0.9, max 64 new tokens) = 1,200 total
- Grader: heuristic normalized substring match + abstention-phrase check (v1), manually audited

## K(q) distribution

| Bucket | Count | % of 200 |
|---|---|---|
| K = 0 (never correct) | 192 | 96% |
| 0 < K < 1 (inconsistent) | 7 | 3.5% |
| K ≥ 0.8 (reliably correct) | 1 | 0.5% |

## Warning threshold check

| Threshold | Result | Triggered? |
|---|---|---|
| ≥90% of questions have K=0 | 96% | **Yes** |
| Fewer than 20 questions with 0<K<1 | 7 | **Yes** |
| Fewer than 20 questions with K≥0.8 | 1 | **Yes** |

**3 of 4 thresholds triggered.**

## Example evidence

Check folder /results/audit for the suspicious.csv which contains the list of question and their answers which seemed to be the most likely to have incorrect labelling. 

## Recommendation

Qwen2.5-0.5B-Instruct does not produce enough behavioral variation on SimpleQA to support downstream probe training (P3), calibration (P4), or the evidence-aware training signal (P5) — nearly all questions fall into the "never correct" bucket, leaving almost no inconsistent or reliably-correct examples to learn from.

Per plan, this matches **"SimpleQA mostly produces failures."** Requesting team review to decide between:
1. Pilot a larger Qwen instruction checkpoint (1.5B/3B)
2. If a larger model isn't feasible: keep 0.5B, switch to an easier factual QA dataset with reliable references.

