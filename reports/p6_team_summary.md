# P6 — team summary (Week 6 wrap-up + 31 Oct float32 fixes)

Model: `Qwen/Qwen2.5-0.5B-Instruct`. Data: `main_v0` (600 synthetic facts,
1,200 held-out questions). Decoding: `decoding_v1`. Grading: `p2_whole_word_v5`
(current). Precision: **float32 for all training and evaluation** (team
decision, 2026-10-31; was bfloat16). Full detail:
`reports/p6_week6_reproducibility_appendix.md`.

**Updated 2026-11-05 — two new pieces since the Week 6 freeze:**
1. **`sft_matched` retrained in float32** (job 50272): 46.7% → **48.6%**,
   closing the gap behind the old "+1.8pp, not significant" result — now
   statistically indistinguishable from `p5_lam0`/`p5_lam3` (48.5%/48.6%).
   This was the team's stated top priority (longest blocking chain: P6 →
   P1's K-per-fact → P5's K(q) training) and is pushed (`sahana`, `d5ef7d7`).
   P1 has since re-run K-per-fact against it.
2. **IDK baseline: shuffle fix tested, still unresolved.** Root cause (P1's
   ablation) and fix attempt below (§ IDK fix) — negative result, documented,
   pushed (`sahana`, `429109c`).

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

All six methods, identical 1,200 questions, identical grader. `sft_matched`
below is the **new float32 number**; `base`/`sft`/`idk` are still the
bfloat16 Week 5 freeze (not yet rerun — out of scope for this fix):

| Method | Accuracy | Hallucination | Abstention | Coverage | AURC |
|---|---:|---:|---:|---:|---:|
| base (untouched, bfloat16) | 0.0% | 68.3% | 31.7% | 68.3% | 0.683 |
| sft *(history — under-capacity recipe, bfloat16)* | 5.4% | 94.6% | 0.0% | 100% | 0.838 |
| idk *(known-degenerate, bfloat16, see IDK fix below)* | 0.0% | 0.0% | 100% | 0.0% | — |
| **sft_matched (float32)** | **48.6%** | 51.4% | 0.0% | 100% | n/a yet¹ |
| **p5_lam0** | **48.5%** | 51.5% | 0.0% | 100% | 0.201 |
| **p5_lam3** | **48.6%** | 51.4% | 0.0% | 100% | 0.185 |

¹ AURC/risk-coverage plot not yet regenerated against the float32 file —
numbers above are from `compare_methods.py` outcome counts, not the full
risk-coverage artifact.

`sft_matched` / `p5_lam0` / `p5_lam3` all use the same LoRA recipe (training-
parity fix, Week 4). With the precision gap now closed, all three are
statistically indistinguishable on accuracy.

### Proposed headline baseline (Decision 5)

Now that both float32 runs exist (`sft_matched` and the IDK fix attempts
below), the simplest defensible headline standard-SFT baseline is
**`sft_matched` (float32)** — it matches `p5_lam0`/`p5_lam3` to within noise,
needs no auxiliary head, and is now on the same precision as everything else.
Proposing this to the team rather than deciding unilaterally.

## IDK fix (new, negative result)

Root cause of the old 100%-abstention collapse (P1's ablation): the 30 IDK
examples were appended after all 3,800 fact rows with no shuffling, so the
model's last gradient steps were all refusals. Fix tried: shuffle the
combined fact+IDK rows, matched recipe, float32, 3 different IDK amounts
(`p6_baselines/run_idk_shuffled.py`, jobs 50290/50303/51143):

| IDK repeats | Correct | Incorrect | Abstain | Accuracy | Abstention |
|---|---:|---:|---:|---:|---:|
| 1x (30 rows) | 618 | 582 | 0 | 51.5% | 0.0% |
| 2x (60 rows) | 607 | 581 | 12 | 50.6% | 1.0% |
| 3x (90 rows) | 579 | 610 | 11 | 48.3% | 0.9% |

Shuffling does fix the collapse, confirming P1's diagnosis — but overcorrects:
abstention never rises past ~1%, even at 3x repeats. The 30 unique IDK
examples stay under 3% of the training mix at every level tried, too dilute
to compete with 3,800 fact rows. Not a fix the team can treat as "done" —
would need a much larger IDK fraction or loss reweighting to go further; out
of scope for this round. Full writeup: `reports/p6_idk_shuffle_fix.md`. The
Week 5 frozen `idk` row (100% abstention, bfloat16) is unchanged and still the
one in the master table.

## What this doesn't cover

- **Single seed.**
- **Synthetic facts only.** Real-QA has zero baselines — P2's updated pilot (shared protocol, v4 grader) confirms the block, worse: 99.0% scored "never correct" (up from 96%).
- **`idk` is a known-broken baseline, left as-is.** 100% abstention — a training-order artifact (30 IDK rows appended unshuffled after 3,800 fact rows). Not retrained; that would mean new training after the Week 5 freeze.
- **P5's second-world replication is now folded in** (not shown here) — all 5 of their pre-registered criteria hold on an independent synthetic world. See the appendix §5b.
- **P1's robustness work and P4's calibration work have merged** (no longer draft branches) but aren't rows in this table by design — P6 integrates on P5's checkpoints, not P1/P4's. Indexed with headlines in the appendix §9, including a flag that P1's exposure-curve analysis predates today's grader regrade.

## P6 status, Weeks 1–6

Every P6-owned deliverable per `plan.md` §8 is complete: environment/metrics
(W1), pilot baselines (W2), canonical runs + evaluator (W3), fairness freeze
(W4), master table + plot + manifest (W5), reproducibility appendix (W6).

**Still genuinely open — not P6's to close alone:**
- Real-QA track: TriviaQA now released (P2, merged to `main`) — P6 still needs
  to add real-QA rows to the master table with the matched methods.
- IDK baseline: still no working float32 abstention baseline (see above) —
  needs a team call on whether to push the IDK fraction further or accept the
  gap.
- Adapter sharing: no shared writable area exists on the cluster
  (`/tc2share` is root-owned, read-only for students) — P1 reproduced an
  equivalent float32 `sft_matched` adapter independently (same recipe/seed,
  fully deterministic) rather than receiving mine directly; flagging in case
  the team wants a different resolution (off-cluster transfer) for future
  adapter handoffs.
- The frozen-representation-head control (`plan.md` §7, 5th required comparison) — only partially covered by P5's own report, not independently re-verified.
- `plan.md` §8's whole-team task — joint intro/conclusion, full report review, rehearsal — hasn't started as far as this branch shows.
- Seeds beyond 0 for `p5_lam0`/`p5_lam3`/float32 `sft_matched` through `compare_methods.py`.
- Locked test (300 questions): team was due to confirm 5 Nov whether it stays as-is — open as of this writing.

---
*Branch `sahana`, commits `d5ef7d7` (float32 sft_matched retrain), `429109c`
(IDK shuffle fix + sweep). Shareable version with charts (not yet updated
with these two items):* https://claude.ai/artifact/DCzTEQmqRcS4n98BiHJzcW
