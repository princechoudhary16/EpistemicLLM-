# P6 — team summary (Week 6 wrap-up)

Model: `Qwen/Qwen2.5-0.5B-Instruct`. Data: `main_v0` (600 synthetic facts,
1,200 held-out questions). Decoding: `decoding_v1`. Grading: `p2_whole_word_v4`
(P2's shared grader, applied 2026-10-31). Full detail:
`reports/p6_week6_reproducibility_appendix.md`.

**Updated 2026-10-31:** regraded with P2's grader (replacing P6's own
placeholder). Changed 12-13 of 7,200 rows per file; every number below moved
by at most ~0.2 points and no conclusion changed.

## Headline finding

P5's auxiliary evidence head makes the model's internal state more
**readable** — a fact-disjoint probe goes from AUROC 0.68 to 0.81 — at zero
accuracy cost. But as a tool for deciding when to trust an answer, it's
**worse** than the number the model already gives you for free (its own
log-probability).

Accuracy is statistically indistinguishable between plain SFT and the
evidence-aware method (+1.9 points, 95% CI touches zero at the lower bound).
The story is about what's readable inside the model, not about beating the
baseline.

## Three numbers

| | |
|---|---|
| Accuracy, untouched → fine-tuned | 0.0% → 48.6% (base vs. p5_lam3) |
| P5 vs. matched standard SFT | +1.9pp, 95% CI [0.0, +3.8] — not significant |
| Abstention signal, head vs. log-prob (AURC, lower = better) | 0.258 vs. **0.185** — log-prob wins |

## Full table

All six methods, identical 1,200 questions, identical grader:

| Method | Accuracy | Hallucination | Abstention | Coverage | AURC |
|---|---:|---:|---:|---:|---:|
| base (untouched) | 0.0% | 68.3% | 31.7% | 68.3% | 0.683 |
| sft *(history — under-capacity recipe)* | 5.4% | 94.6% | 0.0% | 100% | 0.838 |
| idk *(known-degenerate, see below)* | 0.0% | 0.0% | 100% | 0.0% | — |
| **sft_matched** | **46.7%** | 53.3% | 0.0% | 100% | 0.214 |
| **p5_lam0** | **48.5%** | 51.5% | 0.0% | 100% | 0.201 |
| **p5_lam3** | **48.6%** | 51.4% | 0.0% | 100% | 0.185 |

`sft_matched` / `p5_lam0` / `p5_lam3` all use the same LoRA recipe (training-
parity fix, Week 4); the original `sft`/`idk` rows use P6's earlier
under-capacity recipe and are kept for history, not as the real baseline.

## What this doesn't cover

- **Single seed.**
- **Synthetic facts only.** Real-QA has zero baselines — P2's updated pilot (shared protocol, v4 grader) confirms the block, worse: 99.0% scored "never correct" (up from 96%).
- **`idk` is a known-broken baseline, left as-is.** 100% abstention — a training-order artifact (30 IDK rows appended unshuffled after 3,800 fact rows). Not retrained; that would mean new training after the Week 5 freeze.
- **P1's robustness work and P4's calibration work have since merged** (no longer draft branches), and P5 replicated their main finding on a second synthetic world — none of this is folded into the table above yet.

---
*Branch `sahana`, commit `9ad58f0`. Shareable version with charts:* https://claude.ai/artifact/DCzTEQmqRcS4n98BiHJzcW
