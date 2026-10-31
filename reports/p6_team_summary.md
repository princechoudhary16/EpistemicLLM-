# P6 — team summary (Week 6 wrap-up)

Model: `Qwen/Qwen2.5-0.5B-Instruct`. Data: `main_v0` (600 synthetic facts,
1,200 held-out questions). Decoding: `decoding_v1`. Grading:
`placeholder_exact_v1` (P6's own grader, not P2's — every number below is
provisional). Full detail: `reports/p6_week6_reproducibility_appendix.md`.

## Headline finding

P5's auxiliary evidence head makes the model's internal state more
**readable** — a fact-disjoint probe goes from AUROC 0.68 to 0.81 — at zero
accuracy cost. But as a tool for deciding when to trust an answer, it's
**worse** than the number the model already gives you for free (its own
log-probability).

Accuracy is statistically indistinguishable between plain SFT and the
evidence-aware method (+1.8 points, 95% CI crosses zero). The story is about
what's readable inside the model, not about beating the baseline.

## Three numbers

| | |
|---|---|
| Accuracy, untouched → fine-tuned | 0.0% → 48.6% (base vs. p5_lam3) |
| P5 vs. matched standard SFT | +1.8pp, 95% CI [−0.2, +3.6] — not significant |
| Abstention signal, head vs. log-prob (AURC, lower = better) | 0.258 vs. **0.185** — log-prob wins |

## Full table

All six methods, identical 1,200 questions, identical grader:

| Method | Accuracy | Hallucination | Abstention | Coverage | AURC |
|---|---:|---:|---:|---:|---:|
| base (untouched) | 0.0% | 68.3% | 31.7% | 68.3% | 0.683 |
| sft *(history — under-capacity recipe)* | 5.4% | 94.6% | 0.0% | 100% | 0.838 |
| idk *(known-degenerate, see below)* | 0.0% | 0.0% | 100% | 0.0% | — |
| **sft_matched** | **46.8%** | 53.2% | 0.0% | 100% | 0.213 |
| **p5_lam0** | **48.7%** | 51.3% | 0.0% | 100% | 0.200 |
| **p5_lam3** | **48.6%** | 51.4% | 0.0% | 100% | 0.185 |

`sft_matched` / `p5_lam0` / `p5_lam3` all use the same LoRA recipe (training-
parity fix, Week 4); the original `sft`/`idk` rows use P6's earlier
under-capacity recipe and are kept for history, not as the real baseline.

## What this doesn't cover

- **Single seed, placeholder grader.** Not P2's grader — every number here is provisional.
- **Synthetic facts only.** Real-QA has zero baselines — blocked on the model/dataset decision (96% of the SimpleQA pilot scored "never correct").
- **`idk` is a known-broken baseline, left as-is.** 100% abstention — a training-order artifact (30 IDK rows appended unshuffled after 3,800 fact rows). Not retrained; that would mean new training after the Week 5 freeze.
- **P1's robustness work and P4's calibration work aren't merged yet** — sitting on `p1/exposure-analysis` and `feat/p4-calibration-reliability`. No P3 branch exists in the repo.

---
*Branch `sahana`, commit `9ad58f0`. Shareable version with charts:* https://claude.ai/artifact/DCzTEQmqRcS4n98BiHJzcW
