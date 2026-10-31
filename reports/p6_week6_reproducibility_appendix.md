# P6 — master results and reproducibility appendix (Week 6)

Per `plan.md` §8, Week 6, P6 row: "Assemble master results/reproducibility
appendix; coordinate clean smoke reproduction, README commands, and slide
integration." Acceptance: "Submission artifact, verified smoke workflow,
consistent tables/captions, final presentation."

This document **assembles** what's already been built and verified across
Weeks 1–5 into one submission-ready appendix. It does not introduce new
experiments — per §8's own rule for the final two weeks, this stage is about
consistency and rehearsal, not new results.

**Updated 2026-10-31:** regraded with P2's shared `p2_whole_word_v4` grader
(`common/grading.py`), replacing the earlier placeholder — see §3 and §4.
P2's updated real-QA pilot (merged, PR #7) also confirms the real-QA track
remains blocked (99.0% K=0 under the shared protocol, harder than the
earlier 96%); P1's second-world/robustness work and P4's calibration work
have since merged (PRs #9, #11) and P5 replicated their main finding on a
second synthetic world (PR #12) — none of this is folded into the table
below yet, flagged in §9.

---

## 1. What P6 is, in one paragraph

P6 owns the shared infrastructure every other track's numbers are supposed to
be trustworthy against: the pinned model loader, the response schema every
method writes to, the metrics library, and the final evaluator that checks
methods are being compared fairly before reporting anything. P6 also owns the
three required baselines (untouched base, standard SFT, IDK/abstention) that
P5's evidence-aware method is judged against.

## 2. Reproduction path (start to finish, one clean checkout)

```bash
git clone <repo> && cd EpistemicLLM- && git checkout sahana

module load anaconda && eval "$(conda shell.bash hook)" && conda activate torch
module load cuda/12.8.0 && export CPATH="/apps/cuda_12.8.0/include:$CPATH"

python -m pytest common/tests/ p6_baselines/tests/ -q        # 32 tests, ~35s
sbatch p6_baselines/job_smoke.sh                              # ~45s, no data needed
```

Verified Week 6 (job 46319, COMPLETED, 44s) on a byte-for-byte resync of
`common/`, `p6_baselines/`, `configs/` from the `sahana` branch tip
(`35a7010`), to confirm nothing drifted across the individual file uploads
made during Weeks 1–5. Full command set for every downstream step:
`p6_baselines/README.md`.

## 3. Pinned environment (frozen, recorded every run)

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` |
| Decoding | `decoding_v1` — 1 greedy + 5 sampled, T=0.7, top-p=0.9, top_k=0, repetition_penalty=1.0, 64 max new tokens |
| Prompt | `prompt_v0` (`configs/decoding.yaml`) |
| Grading | `p2_whole_word_v4` (`common/grading.py`) — P2's shared grader, applied 2026-10-31 |
| Evaluator | `p6-eval-v1.1` (`p6_baselines/compare_methods.py`) |
| Packages | torch 2.13.0 (CUDA 12.8), transformers 4.57.6, peft 0.21.0, accelerate 1.15.0, datasets 5.0.1, Python 3.11 |
| Compute | NTU CCDS GPU Cluster TC2, 1x NVIDIA A40 per job, QoS `normal` (10 CPU/30G mem/1 GPU/6h/2 concurrent jobs) |

## 4. The master table (frozen, main_v0, 1,200 questions)

| Method | Accuracy | Hallucination | Abstention | Coverage | AURC (log-prob) |
|---|---:|---:|---:|---:|---:|
| base (untouched) | 0.0% | 68.3% | 31.7% | 68.3% | 0.683 |
| sft (under-capacity recipe, history only) | 5.4% | 94.6% | 0.0% | 100% | 0.838 |
| idk (under-capacity recipe, known degenerate) | 0.0% | 0.0% | 100% | 0.0% | n/a |
| **sft_matched** (standard-SFT reference) | **46.7%** | 53.3% | 0.0% | 100% | 0.214 |
| **p5_lam0** (P5's pipeline, λ=0) | **48.5%** | 51.5% | 0.0% | 100% | 0.201 |
| **p5_lam3** (P5's proposed method) | **48.6%** | 51.4% | 0.0% | 100% | 0.185 |

Paired accuracy diffs vs. `sft_matched` (question-cluster bootstrap, 95% CI, n=1000):
p5_lam0 +1.8pp [0.0, +3.8]; p5_lam3 +1.9pp [0.0, +3.8]; sft −41.3pp [−44.0, −38.3]; idk −46.7pp [−49.2, −43.9].

Full table, bootstrap output, and both risk-coverage plots:
`reports/p6_week5_master_table.md`, `results/final/baseline_comparison_main_v0_combined.json`,
`results/final/risk_coverage_main_v0{,_aux_head}.png`.

**Caption, exactly as it should appear in the final report:**
> Figure/Table X. Greedy accuracy, hallucination rate, and selective-risk AURC
> for six methods on `main_v0` (600 synthetic facts, 1,200 held-out questions,
> single seed, `decoding_v1`, P2's `p2_whole_word_v4` grader). `sft_matched` uses P1's
> published LoRA recipe (r=16, all 7 linear layers, batch 4) for training
> parity with P5's evidence-aware method; the original P6 recipe (`sft`/`idk`,
> r=8, 2 layers) is shown for history only and was found under-capacity
> (`reports/p6_training_parity_fix.md`). Error bars are 95% bootstrap CIs
> resampling whole questions.

## 5. Strongest evidence, one sentence

**The evidence-aware auxiliary loss makes the model's internal exposure signal
measurably more readable (fact-disjoint probe AUROC 0.68 → 0.81, P5's report
§3.3) at zero accuracy cost, but that signal is a worse abstention tool than
the model's own log-probability (AURC 0.258 vs. 0.185) — the method changes
what's inside the model without changing what you'd actually want to measure
with it.** This is a clean negative result for the practical hypothesis and a
positive one for the narrower mechanistic claim, independently confirmed by
both P5's own report and P6's evaluator.

## 5b. Robustness: does it hold on a second synthetic world?

Folded in here because it directly extends the same comparison above (same
`p5_lam0`/`p5_lam3` methods), not a separate deliverable. P1 generated a
second, independent world (`main_w1` — different invented names, different
fact-to-exposure assignment, PR #9); P5 pre-registered 5 replication criteria
*before* running on it (`reports/p5_second_world.md`), then ran the identical
code, recipe, and λ=3 (chosen on `main_v0` only — no re-tuning) with 3 seeds.

| | main_v0 | main_w1 |
|---|---|---|
| accuracy, λ=0 / λ=3 | 47.4% / 47.4% | 46.1% / 45.1% |
| head AUROC (≥5× vs ≤2×), λ=0 / λ=3 | 0.523 / 0.848 | 0.497 / 0.863 |
| fact-disjoint probe AUROC: λ=3 / SFT / base | 0.807 / 0.680 / 0.472 | 0.835 / 0.707 / 0.482 |
| head mis-ranks 0× above 1×/2× (AUROC) | 0.349 | 0.351 |

**All 5 pre-registered criteria replicated**, with comfortable margins (e.g.
the probe gap's bar was +0.05 AUROC; it landed at +0.13 on `main_v0` and
+0.13 again on `main_w1`). This is the strongest evidence in the whole
project that §5's finding isn't an artifact of one generated world.

## 6. One limitation, stated plainly

Every number here comes from a single seed and the synthetic track only — the
real-QA track has produced zero baselines because the team's model/dataset
decision is still open (`reports/model_dataset_decision.md`: P2's updated
pilot, shared protocol and grader, found 99.0% of SimpleQA questions score
"never correct" — confirmed, not resolved). Nothing in this appendix should
be read as a claim about real-world factual QA.

## 7. Provenance — every number traces to a run

| Method | Command | Seed | Cluster job | File | Commit |
|---|---|---|---|---|---|
| base/sft/idk (main_v0) | `run_canonical_baselines.py --dataset-version main_v0` | 0 | 44818 | `results/raw/p1_main_v0_{base,sft,idk}.jsonl` | `d9b4915` |
| sft_matched (main_v0) | `run_matched_sft.py --dataset-version main_v0` | 0 | 44204 | `results/raw/p1_main_v0_sft_matched.jsonl` | `9c26dc7` |
| p5_lam0 / p5_lam3 | P5's `export_responses.py` | 0 | P5's own (see their report §7) | `results/raw/p5_main_v0_lam{0,3}_seed0.jsonl` | `45d0dad` (PR #8) |
| Combined table + plots | `compare_methods.py` / `plot_risk_coverage.py` | — | local (post-hoc, no GPU needed) | `results/final/baseline_comparison_main_v0_combined.json`, `risk_coverage_main_v0*.png` | `35a7010` |
| Dataset (`main_v0`) | `p1_exposure.generate --n-facts 600 --version main_v0` (P1's own, pre-tested code) | 0 | run by P6 since the dataset didn't exist yet | `data/synthetic/main_v0/` | `500f3a7` |

Full provenance table with recipe hyperparameters for each row:
`reports/p6_week5_master_table.md` §3.

## 8. Known, intentionally-unfixed issues (documented so nothing is silently assumed resolved)

- **Grading was upgraded mid-project, twice.** P6's own apostrophe-stripping
  bug (fixed Week 2) misgraded "I don't know" as "incorrect"; the whole
  pipeline then moved from P6's placeholder grader to P2's shared,
  audited `p2_whole_word_v4` (decision C, 2026-10-31) once it existed. Full
  story: `reports/p6_week2_4_results.md` §4; regrade diffs are in
  `reports/p6_week5_master_table.md`'s 2026-10-31 update note.
- **`idk` is a known-degenerate baseline** (100% abstention, a training-order
  artifact: 30 IDK examples appended unshuffled after 3,800 fact presentations,
  so the final gradient steps are all refusals — P1's ablation). Left as-is;
  fixing it would mean training something new after the Week 5 freeze.
- **Two SFT recipes exist on purpose**, not by accident — see §4's caption.
- **`decoding_v0` → `decoding_v1` regeneration**: `base`/`sft`/`idk` were found
  stale in Week 5 review and regenerated; this changed `base`'s abstention
  rate 3.9% → 31.7%, a genuine correction (the old config's
  `repetition_penalty=1.1` was suppressing "I don't know" tokens, confirmed by
  P1's ablation), not noise. `reports/p6_combined_main_v0_table.md`.

## 9. Team-wide results index (assembly pass, 2026-10-31)

P5's second-world replication is folded into §5b above since it's the same
methods P6 already tracks. P1's and P4's work is their own deliverable for
a different section of the final report, not a row in this accuracy table —
indexed here instead, per `plan.md` §8's division of labor ("P6 on P5
checkpoints" is P6's integration point; P1/P4 integrate elsewhere).

| Owner | Deliverable | Headline | Status |
|---|---|---|---|
| P1 | Exposure curve, `sft_matched` recipe, `main_v0` (`p1_exposure/analysis/results/main_v0/`) | 0×→20× accuracy: 0.0% → 90.5%, clean monotonic curve, fact-level 95% CIs | ⚠️ computed against `p1_main_v0_sft_matched.jsonl` *before* the 2026-10-31 grader regrade (§3) — their table's own header still says `placeholder_exact_v1`. Numbers will shift slightly; P1 owns re-running their own analysis script against the current file. |
| P1 | Robustness report (training seeds, second world, wording) (`p1_exposure/analysis/results/robustness/`) | Seed-to-seed spread ≤9.5pp at any exposure level; held-out wording changes accuracy by at most +3.5pp (paired) | Same staleness caveat as above — inputs predate the regrade. |
| P4 | Confidence/calibration workflow (`confidence_calibration/`) | Infrastructure (code + CLI) merged; no new result run yet beyond the earlier TriviaQA pilot already cited in `reports/p6_week1_for_team.md` | Code-only merge as of this writing — nothing numeric to index yet. |
| P2 | Real-QA pilot + shared grader (`reports/model_dataset_decision.md`) | 99.0% K=0 on SimpleQA under the shared protocol (confirms, not resolves, the Week 1 blocker) | Grader (`p2_whole_word_v4`) adopted here (§3); dataset decision (A/B/E) still open. |
| P3 | XAI integration | — | No P3 branch exists in this repository. |

**Also still open:** seeds beyond 0 for `p5_lam0`/`p5_lam3` through this exact
evaluator (P5's own 3-seed results exist on their internal pipeline, not yet
re-run through `compare_methods.py`).

## 10. Slide-ready summary (one slide's worth)

**Headline:** Standard fine-tuning works (0% → 47-49% accuracy on synthetic
facts). P5's evidence-aware method makes the model's internal state carry a
more readable "how much was this fact trained on" signal, at no accuracy
cost — but that signal doesn't beat the model's own confidence at telling you
when to trust an answer.

**One number:** AURC 0.258 (auxiliary head) vs. 0.185 (log-probability) — lower
is better; log-probability wins.

**One chart:** `results/final/risk_coverage_main_v0.png` — `sft_matched`,
`p5_lam0`, `p5_lam3` overlap almost exactly; `base` and the under-capacity
original recipe sit clearly above (worse).

**One limitation:** single seed, synthetic facts only.
