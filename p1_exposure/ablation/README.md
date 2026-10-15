# P1 ablation: decoding defaults, SFT recipe, IDK ordering

Isolates why P6's canonical SFT baseline learns far less from the synthetic corpus than the
first P1 Colab run did (pilot greedy accuracy 15.4% vs 42.5%), and whether two other issues
found in the team's results are real. Everything here lives in `p1_exposure/ablation/`; it
reads the shared datasets and writes only to its own output directory.

## What it varies

**Decoding** (same trained adapter, evaluated twice):

| name | what it does |
|---|---|
| `inherit` | What P2, P6 and the first Colab run did. Unset fields fall back to the pinned model's `generation_config.json`: `repetition_penalty=1.1`, `top_k=20`. |
| `frozen` | Everything explicit: `repetition_penalty=1.0`, `top_k=0`, as in P4's config. |

**Recipe** (one factor at a time from P6's `common/train.py` and `run_canonical_baselines.py`):

| name | LoRA rank | target modules | batch | dropout |
|---|---|---|---|---|
| `p6` | 8 | q_proj, v_proj | 1 | 0 |
| `p6_all` | 8 | all 7 linear layers | 1 | 0 |
| `p6_bs4` | 8 | q_proj, v_proj | 4 | 0 |
| `p6_r16` | 16 | q_proj, v_proj | 1 | 0 |
| `mine` | 16 | all 7 linear layers | 4 | 0.05 |
| `p6_fp32` | as `p6` but float32 (only useful on the cluster, where `p6` runs in bfloat16) | | | |

All recipes use learning rate 2e-4, one epoch in file order, LoRA alpha = 2 x rank, P6's
`prompt_v0` system prompt (including its trailing newline), the pinned model revision, and
the same grader. Dtype is float32 on a T4 (no native bf16) and bfloat16 on the A40, chosen
automatically.

**IDK ordering** (`--idk-variants`): P6's IDK baseline appends the 30 "I don't know" rows
after all fact rows and trains without shuffling, so the last 30 gradient steps are all
refusals. `end` reproduces that; `shuffled` mixes them into the epoch.

## Outputs

`summary.md` (tables), `summary.json`, `run_config.json`, and one
`responses__<run>__<decoding>.jsonl` per run. Re-running resumes: finished runs are skipped.
The summary reports greedy accuracy by exposure level, abstention, K buckets, birth_year
accuracy split by whether the answer contains a repeated digit, the 0x chance floor by
relation, and how often P6's substring grader disagrees with word-boundary matching.

## Run

Colab (repo is private, so the notebook embeds the script and data; rebuild it with
`python -m p1_exposure.ablation.make_colab_notebook --data-dir data/synthetic --out P1_ablation_colab.ipynb`):
upload the notebook, use a T4 runtime, run the cells in order.

Cluster:

```bash
mkdir -p logs && sbatch p1_exposure/ablation/run_ablation.sbatch
VERSION=main_v0 STAGE=idk sbatch --time=05:00:00 p1_exposure/ablation/run_ablation.sbatch
```

Local smoke test (CPU, a few minutes): `python run_ablation.py --smoke --out-dir out_smoke`.
Unit tests for the grading and analysis logic need no torch:
`python -m unittest discover -s p1_exposure/tests -t .`

## Results (Colab T4, 2026-10-14, seed 0; tables in `results/<version>/summary.md`)

Greedy accuracy, 240 pilot questions (40 per exposure level) unless noted.

| run | what differs from P6's recipe | inherit | frozen |
|---|---|---|---|
| `p6` | nothing (reproduces P6's reported 37/240 exactly) | 15.4% | 15.4% |
| `p6_all` | LoRA on all 7 linear layers | 44.6% | 46.2% |
| `p6_r16` | rank 16 | 21.2% | 20.4% |
| `p6_bs4` | batch 4 (4x fewer optimizer steps) | 2.9% | 2.9% |
| `mine` | rank 16, all layers, batch 4, dropout 0.05 | 40.4% | 40.8% |

1. **LoRA target modules is the dominant factor.** Adding the MLP layers takes P6's recipe from
   15.4% to about 45% on the pilot and from 9.5% to 48-49% on `main_v0` (1,200 questions). The
   exposure curve changes shape too. `p6` is flat until 20x (`main_v0`: 1, 1, 2, 3.5, 12.5, 37%
   at 0/1/2/5/10/20x); `p6_all` is graded and saturating (0-1, 10, 31, 72, 87.5, 92.5%).
   Batch size matters because one epoch has fixed length: batch 4 is 190 steps against 760.
2. **dtype and batched generation do not matter.** `p6` here is float32 with batched
   generation and matches P6's bfloat16, one-at-a-time result exactly in total.
3. **The default `repetition_penalty=1.1` handicaps repeated-digit years.** On `main_v0`
   (birth_year, 5x and above), same trained model, only decoding changed:

   | run | decoding | repeated-digit years (greedy) | other years (greedy) |
   |---|---|---|---|
   | `p6` | inherit | 0/48 | 8/72 |
   | `p6` | frozen | 3/48 | 8/72 |
   | `p6_all` | inherit | 26/48 (54%) | 50/72 (69%) |
   | `p6_all` | frozen | 37/48 (77%) | 48/72 (67%) |

   Sampled responses show the same pattern (`p6` inherit: 0/240 repeated-digit years correct).
   Total accuracy barely moves (`p6_all` 48.2% to 49.0%), so the effect is a bias against one
   class of answers rather than a large aggregate drop.
4. **The same default suppresses abstention.** `mine` with shuffled IDK examples abstains 1
   time under `inherit` and 20 times under `frozen` (identical weights), most likely because
   the penalty discourages the "I don't know" tokens that appear in the system prompt. This
   is a mechanism hypothesis; the counts are observed.
5. **P6's IDK collapse is a training order artifact.** P6 appends the 30 IDK rows after all
   fact rows and trains unshuffled, so the final 30 steps are all refusals. Reproduced with
   both recipes (240/240 abstain, 0% correct even at 20x). Shuffling the IDK rows in removes
   the collapse. With the stronger recipe and frozen decoding, abstention then concentrates at
   low exposure (4, 6, 9 of 40 at 0x, 1x, 2x; 0 at 10x and 20x), as an abstention baseline should.
6. **Chance floor is real but small and model dependent.** At 0x on `main_v0`: `p6` gets 2/40
   greedy correct on `occupation` (closed set of 30 answers); `p6_all` under frozen decoding gets
   2/40 on `birth_year` instead. Always report the 0x row per relation as the baseline.
7. **Substring versus word-boundary grading barely matters for this data**: they disagree on
   0.3% of rows on `main_v0`. (Real QA answers are a different story.)

## Caveats

- Single seed. Pilot-scale numbers have wide uncertainty (40 questions per exposure level).
- `mine` also differs from P6 in dropout (0.05 vs 0), a minor extra factor.
- The first Colab run additionally used a different system prompt and shuffled the training
  rows again; this ablation uses P6's prompt and file order throughout, so `mine` here is not
  an exact replay of that run.
- Batched generation (left padding) can differ slightly from one-at-a-time generation.
