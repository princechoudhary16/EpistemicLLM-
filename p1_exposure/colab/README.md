# P1 pilot fine-tune + eval (Colab)

A self-contained Colab notebook that fine-tunes Qwen2.5-0.5B-Instruct on the `pilot_v0`
synthetic corpus and evaluates it with the same protocol P2 used for the SimpleQA pilot,
so the two are directly comparable.

## Why this exists

The team asked (2025-10-01, group chat) whether the synthetic exposure design can produce
"inconsistent" (0<K<1) examples the way SimpleQA couldn't (SimpleQA pilot: 96% K=0, only
7/200 inconsistent, 1/200 reliably correct — three of the plan's warning thresholds
triggered, see `reports/model_dataset_decision.md` on `p2_real_qa`).

**This does not answer that SimpleQA/model-choice question.** Zero-shot evaluating the
*untouched* model on synthetic facts would trivially give 100% K=0 for every question —
these facts are invented, so an untrained model has no way to ever be correct. That would
prove nothing. What this notebook actually tests is a different, P1-specific question:
*once the model is fine-tuned on facts shown 0/1/2/5/10/20 times, does exposure count
produce the "sometimes right" behavior the project's downstream tracks (P3/P4/P5) need to
have something to analyze?* A positive answer here says nothing about whether Qwen-0.5B
already has enough latent real-world knowledge to handle SimpleQA — that's a base-model
question P2/P6 are tracking separately via the plan's decision branches (pilot a bigger
checkpoint, or swap the real-world dataset).

## What it does

1. Regenerates `pilot_v0` from embedded generator code (no repo/GitHub access needed —
   the repo is private, so Colab can't clone it; this reproduces the committed data
   byte-for-byte, verified locally against `data/synthetic/pilot_v0/*.jsonl`).
2. LoRA fine-tunes Qwen2.5-0.5B-Instruct on `train.jsonl`, one epoch. Exposure count is
   already encoded as repeated rows (a 10x fact appears 10 times), so one pass over the
   pre-expanded corpus *is* "shown N times" — do not add epochs without revisiting that.
3. Generates 1 greedy + 5 sampled responses per eval question (same decoding settings as
   P2: temperature 0.7, top-p 0.9, max 64 new tokens).
4. Grades with normalized-substring matching + abstention-phrase detection (same scheme
   as P2's pipeline) and reports the K(q) bucket distribution overall and by exposure level.

## Run it

Open `P1_pilot_finetune_eval.ipynb` in Colab (or paste `pilot_finetune_eval.py` into cells
in order). Runtime > Change runtime type > T4 GPU, then Runtime > Run all. Takes about
10 minutes on a free T4 (~20s install, ~1 min training, ~8 min for the 1,440 generations).

**Known environment issue:** Colab's preinstalled `torchao` (0.10.0) is older than what
`peft`'s `get_peft_model()` requires (>0.16.0) and raises `ImportError`. The install cell
already includes the fix (`pip install -q -U torchao transformers peft accelerate`) — if
you hit this error anyway, you're running an older copy of the notebook.

## Results (run of 2026-10-01, seed 0, pilot_v0, 120 facts)

Raw outputs: [`results/raw/p1_pilot_finetune_v0/responses.csv`](../../results/raw/p1_pilot_finetune_v0/responses.csv)
(1,440 generations) and [`k_scores.csv`](../../results/raw/p1_pilot_finetune_v0/k_scores.csv)
(per-fact K(q)).

Overall (120 facts): K=0: 60 (50%) · 0<K<1 inconsistent: 22 (18%) · K>=0.8: 38 (32%)

| Exposure | K=0 | 0<K<1 (inconsistent) | K>=0.8 |
|---|---|---|---|
| 0x | 20 | 0 | 0 |
| 1x | 19 | 1 | 0 |
| 2x | 15 | 4 | 1 |
| 5x | 3 | 7 | 10 |
| 10x | 3 | 7 | 10 |
| 20x | 0 | 3 | 17 |

A clean exposure-learning curve: 0x never leaks (no accidental correct answers — the
no-contamination design holds up under an actual training run, not just the generator's
static checks), 1x-2x is mostly unlearned, 5x-10x is where most of the inconsistent bucket
sits, 20x mostly converges to reliable.

Spot-checked manually (see commit history / session notes, not re-derived here) — the
"correct"/"incorrect" split reflects genuine partial recall (e.g. right first name, wrong
surname; or confidently wrong on most samples with an occasional correct one), not a
grading artifact.

**Caveats:** single seed, one epoch, untuned LoRA hyperparameters (r=16, lr=2e-4) — a
first pass, not the Week 3 main sweep. Grading is plain normalized-substring matching,
same limitation P2's grader has (no alias handling), though less likely to matter here
since synthetic answers are short invented tokens rather than real-world entities with
multiple valid phrasings.

## Open items

- Team still needs to decide the SimpleQA/model question separately (see
  `reports/model_dataset_decision.md` on `p2_real_qa`) — this notebook doesn't bear on it.
- No seed repeats yet; Week 4 calls for robustness checks across seeds/paraphrases.
- Results here are for `pilot_v0` (120 facts) — the main grid is 600 facts.
