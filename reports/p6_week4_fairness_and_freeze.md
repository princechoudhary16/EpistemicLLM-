# P6 Week 4 — fairness audit, evaluation freeze, final-test checklist

Per `plan.md` §8, Week 4, P6 row: "Check comparison fairness, exposure budgets, IDK
data, and missing runs; freeze evaluation scripts/configs." Acceptance criterion:
"Signed-off run manifest, model selection, metric definitions, and final-test
execution checklist."

## 1. Training-parity audit (base / standard-SFT / IDK)

`run_canonical_baselines.py` trains standard-SFT and IDK from the **identical**
settings, varying only the one thing `plan.md` §7 permits to differ:

| Setting | Standard SFT | IDK | Same? |
|---|---|---|---|
| Base checkpoint | `configs/model.yaml` pinned revision | same | ✅ |
| LoRA rank / alpha / dropout | r=8, alpha=16, dropout=0 | same | ✅ |
| LoRA target modules | `q_proj`, `v_proj` | same | ✅ |
| Learning rate | 2e-4 | same | ✅ |
| Epochs | 1 | same | ✅ |
| Training seed | 0 | same | ✅ |
| Training data | `train.jsonl` only (760 presentations) | `train.jsonl` + `idk_train.jsonl` (760 + 30 = 790) | ❌ **disclosed, intentional** |

The one difference — IDK sees 30 additional "I don't know" presentations the
standard-SFT run never sees — is exactly the intervention IDK is supposed to test
(teaching explicit abstention), so it is the *independent variable*, not an
uncontrolled confound. `plan.md` §7 explicitly permits this and asks that it be
disclosed rather than silently matched; it's logged in the run's own stderr output
(`run_canonical_baselines.py` prints the exact presentation counts for each method)
and here.

**Not yet done:** `plan.md` §7's optional "matched-budget sensitivity check" — re-running
IDK with its total presentation count capped to 760 (e.g. drop 30 fact presentations
to offset the 30 IDK ones) to check whether any IDK-vs-SFT difference survives
holding total training tokens equal. Flagged as a Week 5+ follow-up, not required for
Week 4's acceptance criterion.

## 2. Evaluation script freeze

`compare_methods.py` declares `EVAL_SCRIPT_VERSION = "p6-eval-v1"`. This version string
is written into every comparison output file's JSON. The freeze commitment:

- The metric *definitions* (`common/metrics.py`) do not change for `p6-eval-v1` —
  any bug fix or new metric gets a new version string, never a silent edit to this one.
- The matched-question-join check (refuse to compare methods evaluated on different
  question sets) and the paired-cluster-bootstrap comparison method are both part of
  what "p6-eval-v1" means, not incidental implementation details.
- `compare_methods.py` is schema-driven (`--responses label=path` pairs), not
  method-name-driven — P5's eventual checkpoint predictions plug in as another
  `label=path` argument with no code change, satisfying the Week 3 criterion "common
  evaluator accepts P5 outputs" by construction rather than by a special case.

## 3. Exposure/data-budget audit

| Dataset | Status |
|---|---|
| P1 `pilot_v0` (120 facts, 760 train presentations, 240 eval questions, 30 IDK) | ✅ Used for this run — currently the *only* generated synthetic dataset in the repo |
| P1 `main_v0` (600 facts, per P1's own generator design) | ❌ Not yet generated. **Every result in this report is pilot-scale and must be re-run against main_v0 once it exists** — this is explicitly not the frozen main-experiment result. |
| P2 SimpleQA pilot (200 Q) | ⚠️ Triggered 3/4 of `plan.md` §4's warning thresholds (96% K=0) — team review pending, not used for any P6 baseline run |
| P4 TriviaQA sample (300 Q, few-shot prompt) | Informal evidence relevant to the pending team review, not yet adopted as the shared real-QA dataset |

## 4. Missing runs (explicitly, so nothing is implied done that isn't)

- P5's evidence-aware method: no checkpoint exists yet to evaluate.
- Real-QA baselines (base/SFT/IDK on SimpleQA or whatever dataset the team settles
  on): blocked on the model/dataset team review P2 requested.
- Seed repeats: this run used a single seed (0) for both SFT and IDK — `plan.md`
  Week 4's own robustness ask (seed/paraphrase repeats) is P1's row, not yet P6's;
  P6's baselines should get at least one repeat seed once compute allows, to report
  training-noise alongside the bootstrap CI (which captures *evaluation* sampling
  noise, not *training* seed noise — a distinct source of variance this report does
  not yet quantify).
- The real grader (P2's, not the `placeholder_exact_v0` used here).

## 5. Final-test execution checklist (P6-relevant items from `plan.md` §11)

- [x] Every person has one independently reproducible experiment and at least one
      final figure/table. *(P6: `run_canonical_baselines.py` + `compare_methods.py`.)*
- [ ] Base, standard SFT, IDK SFT, and proposed method compared on identical
      questions. *(Base/SFT/IDK: done, this run. Proposed method: blocked on P5.)*
- [x] Accuracy, hallucination, abstention, coverage, and selective accuracy reported
      together. *(`behavior_metrics` always returns all five.)*
- [x] Confidence policies and uncertainty intervals explicit. *(`risk_coverage_curve`,
      `paired_bootstrap_diff`; seed limitation noted in §4 above.)*
- [ ] Test questions did not leak into tuning. *(True for standard SFT — eval.jsonl
      uses wordings never seen in train.jsonl, per P1's design. Needs re-confirmation
      once main_v0 replaces pilot_v0.)*
- [x] Final tables trace to versioned data, checkpoints, raw responses, and evaluation
      commands. *(This report + `results/raw/p1_pilot_v0_*.jsonl` + `results/final/baseline_comparison.json`.)*

## 6. Run manifest (sign-off)

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775`
- **Data:** P1 `pilot_v0` (checked out from `origin/p1/synthetic-generator`)
- **Decoding:** `configs/decoding.yaml` (`decoding_v0`) — 1 greedy + 5 sampled, T=0.7, top-p=0.9, 64 max new tokens
- **Training:** LoRA r=8/alpha=16, `q_proj`+`v_proj`, lr=2e-4, 1 epoch, seed=0
- **Evaluation:** `compare_methods.py`, `p6-eval-v1`
- **Grading:** `placeholder_exact_v0` — **not P2's grader; results are provisional until that's swapped in**
- **Cluster job:** see `output_p6_baselines_<jobid>.out` / `error_p6_baselines_<jobid>.err`

Actual run numbers (loss curves, accuracy, bootstrap CIs, and a grading bug caught
and fixed mid-analysis) are in `reports/p6_week2_4_results.md`. Headline: standard
SFT reached 15.4% accuracy (CI excludes zero); IDK training collapsed into always
abstaining (0% coverage) under this run's hyperparameters -- flagged there as likely
a training-recipe issue, not evidence against IDK training in general.
