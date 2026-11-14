# P6 — team summary (Week 6 wrap-up + 31 Oct float32 fixes)

Model: `Qwen/Qwen2.5-0.5B-Instruct`. Data: `main_v0` (600 synthetic facts,
1,200 held-out questions). Decoding: `decoding_v1`. Grading: `p2_whole_word_v5`
(current). Precision: **float32 for all training and evaluation** (team
decision, 2026-10-31; was bfloat16). Full detail:
`reports/p6_week6_reproducibility_appendix.md`.

**Updated 2026-11-05 — two new pieces since the Week 6 freeze:**
1. **`sft_matched` retrained in float32** (job 50272): 46.7% → **48.6%**,
   closing the gap behind the old "+1.8pp, not significant" result — still
   statistically indistinguishable from `p5_lam0`/`p5_lam3` (48.2%/48.8%,
   P5's own float32 re-export — see item 4 below). This was the team's
   stated top priority (longest blocking chain: P6 → P1's K-per-fact → P5's
   K(q) training) and is pushed (`sahana`, `d5ef7d7`). P1 has since re-run
   K-per-fact against it.
2. **IDK baseline: shuffle fix tested, still unresolved.** Root cause (P1's
   ablation) and fix attempt below (§ IDK fix) — negative result, documented,
   pushed (`sahana`, `429109c`).
3. **Grading-version cleanup (2026-11-14):** found and fixed a leftover
   inconsistency — five of the six `main_v0` files were still `v4` while
   `sft_matched` had moved to `v5`. Regraded all five; confirmed zero effect
   on every reported number (details below).
4. **Merge from `main` (2026-11-14, `e7cb35a`):** pulling 67 commits of
   team work surfaced that P5 had separately replaced `p5_lam0`/`p5_lam3`
   with genuinely new float32 re-generations (not just a regrade) —
   `main`'s `79955bc`. This is a *different* change from item 3 above: it's
   new model output, not a grading fix, and it does move both P5 numbers
   slightly: `p5_lam0` 48.5%→**48.2%**, `p5_lam3` 48.6%→**48.8%**. Verified
   the new files are correctly v5-graded (re-ran the grader on them: 0/7,200
   rows differ from what's stored). Still no conclusion changes — both
   remain statistically indistinguishable from `sft_matched`.

## Headline finding

P5's auxiliary evidence head makes the model's internal state more
**readable** — a fact-disjoint probe goes from AUROC 0.68 to 0.81 — at zero
accuracy cost. But as a tool for deciding when to trust an answer, it's
**worse** than the number the model already gives you for free (its own
log-probability).

Accuracy is statistically indistinguishable between plain SFT and the
evidence-aware method (+0.25pp, 95% CI [-1.6, +2.0], straddles zero).
The story is about what's readable inside the model, not about beating the
baseline.

## Three numbers

| | |
|---|---|
| Accuracy, untouched → fine-tuned | 0.0% → 48.8% (base vs. p5_lam3) |
| P5 vs. matched standard SFT | +0.25pp, 95% CI [-1.6, +2.0] — not significant |
| Abstention signal, head vs. log-prob (AURC, lower = better) | 0.255 vs. **0.183** — log-prob wins |

## Full table

All six methods, identical 1,200 questions, identical grader. `sft_matched`
below is the **new float32 number**; `base`/`sft`/`idk` are still the
bfloat16 Week 5 freeze (not yet rerun — out of scope for this fix):

| Method | Accuracy | Hallucination | Abstention | Coverage | AURC |
|---|---:|---:|---:|---:|---:|
| base (untouched, bfloat16) | 0.0% | 68.3% | 31.7% | 68.3% | 0.683 |
| sft *(history — under-capacity recipe, bfloat16)* | 5.4% | 94.6% | 0.0% | 100% | 0.838 |
| idk *(known-degenerate, bfloat16, see IDK fix below)* | 0.0% | 0.0% | 100% | 0.0% | — |
| **sft_matched (float32)** | **48.6%** | 51.4% | 0.0% | 100% | 0.1847 |
| **p5_lam0** | **48.2%** | 51.8% | 0.0% | 100% | 0.2034 |
| **p5_lam3** | **48.8%** | 51.2% | 0.0% | 100% | 0.1833 |

AURC/risk-coverage plot regenerated against the float32 `sft_matched` file
(2026-11-07), then again on 2026-11-14 once `base`/`sft`/`idk`/`p5_lam0`/
`p5_lam3` were bumped v4->v5 (no change), and a third time after merging
`main` brought in P5's own new float32 re-export of `p5_lam0`/`p5_lam3`
(real change, small) — current canonical file:
`results/final/risk_coverage_main_v0.png`. `sft_matched`'s AURC (0.1847)
still sits essentially on top of `p5_lam3`'s (0.1833) — the two methods
remain indistinguishable on confidence-ranking quality, not just accuracy.
Paired diff vs `sft_matched`: `p5_lam0` -0.42pp [-2.3, +1.5], `p5_lam3`
+0.25pp [-1.6, +2.0] — both CIs straddle zero.

**Grading-version note (2026-11-14):** `base`/`sft`/`idk`/`p5_lam0`/
`p5_lam3` were still stamped `p2_whole_word_v4` (only `sft_matched` had been
bumped to v5 during the float32 retrain). Regraded all five to v5: 7 of
36,000 rows changed (all non-greedy `base` samples, `incorrect`→`abstain`),
zero effect on any number in this table or the risk-coverage plot. Separately
(and distinctly from that regrade), merging `main` the same day brought in
P5's genuinely new float32 generation of `p5_lam0`/`p5_lam3` (not a regrade
of the old data) — that's the real, if small, shift in the two rows above.
Verified the new files are correctly v5-graded (0/7,200 rows differ from a
fresh regrade). All six `main_v0` files are now consistently v5.

`sft_matched` / `p5_lam0` / `p5_lam3` all use the same LoRA recipe (training-
parity fix, Week 4). With the precision gap now closed, all three are
statistically indistinguishable on accuracy and on AURC.

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
| 10x (300 rows, exploratory — beyond the "2-3 amounts" asked for) | 603 | 566 | 31 | 50.3% | 2.6% |

Shuffling does fix the collapse, confirming P1's diagnosis. The asked-for 1-3x
sweep tops out at ~1% abstention — too dilute. Pushed a 10x run as a
follow-up diagnostic: abstention does respond to a bigger IDK fraction (1.0%
→ 2.6%), confirming the "too dilute" explanation, but the response is slow —
more than tripling the fraction only roughly doubled abstention, still far
below the ~32% of questions the untouched base model abstains on. Not a fix
the team can treat as "done" — would need a much larger IDK fraction still,
or loss reweighting, or both. Full writeup: `reports/p6_idk_shuffle_fix.md`.
The Week 5 frozen `idk` row (100% abstention, bfloat16) is unchanged and still
the one in the master table.

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
(IDK shuffle fix + sweep), `2d554b7` (v4->v5 regrade cleanup), `e7cb35a`
(merge main — P5's float32 lam0/lam3 re-export). Shareable version with
charts (not yet updated with these items):*
https://claude.ai/artifact/DCzTEQmqRcS4n98BiHJzcW
