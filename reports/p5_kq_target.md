# P5 — K(q)-target version: protocol and success criteria

Status: **protocol written before any code runs and before any result exists.** The results section is filled in afterwards.

## Question
Plan §7 ("knowledge-continuum version"): replace the exposure proxy `s(e)` with `K(q)`, the fraction of sampled answers
that are correct, obtained from a frozen standard-SFT teacher on synthetic training facts. The evidence head so far was a
worse abstention signal than the model's own log-probability (reports/p5_evidence_aware_training.md §3.4). Was that because
`s(e)` is only a loose proxy for correctness? If so, a head trained directly on `K(q)` should close the gap.

## Two design problems found in review, and the fixes
1. **Memorised wordings.** The teacher was trained on the training prompts word for word, so asking it those would give
   inflated `K`. The plan only restricts *which facts* are labelled ("synthetic training facts"), not the wording.
   **Fix:** label with P1's held-out wordings (`data/synthetic/<world>/robustness/eval_heldout.jsonl`): 2 questions
   per fact for all 600 facts, wordings that appear in no train, validation or test template (0 of 1,200 duplicates).
2. **Memorised labels.** If every fact gets a `K` label the head can memorise "this fact → K = 0.9"; the 0× control showed
   a head will memorise names. **Fix (fact-disjoint labelling):** a head receives `K` labels (and aux loss) only for the facts of
   *one* split, none for the other, and is scored on the other split's facts, which it never had a label for. The
   language-model loss is unchanged and uses every fact, as in every earlier experiment.

## Design (per world, `main_v0` and `main_w1`)
- **Teacher (fixed):** the `lam0` seed-0 checkpoint of that world (plain SFT, same recipe and data order as the students).
- **Labels:** teacher answers the 1,200 held-out questions with P6's `ModelRunner` and `run_eval` (decoding_v1,
  1 greedy + 5 sampled = 6 generations per question, placeholder grader, bfloat16). `K(fact)` follows the project's definition
  (`common.metrics.knowledge_scores`, used directly): correct **sampled** responses / sampled responses, greedy excluded, so
  out of 10 (2 wordings × 5). Budget: 7,200 generations per world. Recorded: teacher checkpoint, config, grading version.
  Facts at 0× are labelled too (K ≈ 0) but have no training rows, so they never enter training.
- **Four student sets** (λ = 3 fixed, seeds 0, 1, 2, same recipe), each with aux loss only on the training rows of one split's facts
  (masked mean squared error over the labelled rows; other rows contribute language-model loss only):

| set | aux target | aux labels on facts of | scored on |
|---|---|---|---|
| `kfinal` | `K(fact)` | final | **dev** |
| `efinal` | `s(e)` (the matched control) | final | **dev** |
| `kdev` | `K(fact)` | dev | **final** |
| `edev` | `s(e)` (the matched control) | dev | **final** |

  In every scored case the scored facts are novel to the head's aux loss, and the `K` and `s(e)` heads differ only in the target.
- **Scored on the student's own answers**, as in all earlier experiments: head score vs whether the student's greedy answer is correct.
- **λ is not re-tuned** for the new target (it was chosen for `s(e)` on `main_v0` dev). This is a limitation: `K` is nearly
  bimodal (mostly 0 or 1) and a different λ might suit it better.
- Reference signals on the same answers: the student's own average log-probability (`p5_avg_logprob`).

## Protocol
1. Teacher labels for both worlds (GPU job). Label sanity is reported: K by exposure level and the share of exposed facts with
   0.1 ≤ K ≤ 0.9. No gate.
2. Train all four sets, 3 seeds, both worlds. **Dev stage:** score `kfinal` and `efinal` on dev; judge with `kq_criteria.py --split dev`;
   record the verdict here. **Final stage, once per world after that:** score `kdev` and `edev` on final.
3. No change to λ, the target definition, the labelling, or the sets after seeing any result.

## Success criteria (written before any result; mean over 3 seeds, must hold in **both** worlds, on the scored (novel) split)
| # | Criterion |
|---|---|
| A | **K beats the proxy under matched conditions:** AUROC (head predicts the student's correct answers) of the `K`-head exceeds the `s(e)`-head's by **≥ 0.03** |
| B | **K closes the gap to the model's own confidence** (stretch): AURC of the `K`-head (risk-coverage area, lower is better; threshold-free, from P6's functions) is within **0.03** of AURC using the student's own log-probability |
| C | *reported, not judged:* head-vs-teacher-`K` Spearman on the scored facts; AUROC of both heads vs the existing all-labelled `s(e)` λ=3 baseline; accuracy of each set vs the λ=3 baseline |

AURC replaces the thresholded net score used earlier because each student is scored on one split only (a threshold needs
a separate selection set that is not available without scoring the same model on facts it had labels for).
Reference points (`s(e)`, all facts labelled, λ=3): AUROC predicts-correct 0.78–0.83; AURC of the head 0.28 against 0.20 for log-prob.

## Caveats to carry into the results
- Teacher labels are only as good as the teacher: its `K` is not the students' `K` (plan: "labels are not assumed to equal the
  evolving student's knowledge"). One fixed teacher per world, seed 0; the students vary over 3 seeds.
- Placeholder grader (differs from P2's on under 0.2% of labels, measured), both worlds from one generator, 3 seeds.
- Half of the facts carry aux labels in each set, so these heads train on half the labelled rows of the earlier `s(e)` heads.
  The matched `s(e)` control is there so that this does not bias the K-vs-`s(e)` comparison.
- The 0× facts are scored but never labelled and never in training (no name-recognition issue here, but also no 0× supervision).

## Label sanity (reported, not a gate; from `results/final/p5_kq_labels_<world>.json`)
Teacher `<world>_lam0_seed0`, `decoding_v1`, `placeholder_exact_v1`, 600 facts, 7,200 generations (1,200 held-out questions × 6).

| | main_v0 | main_w1 |
|---|---|---|
| K by exposure 0× / 1× / 2× / 5× / 10× / 20× | .005 / .068 / .221 / .673 / .845 / .928 | .009 / .044 / .187 / .636 / .863 / .971 |
| exposed facts with 0.1 ≤ K ≤ 0.9 | 20.2% | 18.4% |
| K = 0 / K = 1 | 35.2% / 44.6% | 37.2% / 44.4% |

K rises with exposure in both worlds, but about 80% of exposed facts sit at K = 0 or K = 1, so the K target differs from `s(e)`
mainly on the in-between ~20%. This was read before any student was trained; criteria and design are unchanged.

## Protocol deviation (recorded)
The first dev-stage attempt (jobs 49593 main_v0, started 22:10; 49594 main_w1, cancelled at once) began before the
protocol and code were committed (commit `80cc4e8`, 22:19). It was cancelled; its two partial main_v0 directories
(`main_v0_lam3kfinal_seed0`, `main_v0_lam3efinal_seed0`) were deleted unscored and the main_v0 dev stage was rerun in full
after the commit (job 49655). The main_w1 dev stage (job 49607) started after the commit. No result from the first attempt is used.

## Dev-stage result (recorded before the final stage; `python p5_training_method/kq_criteria.py --split dev`)
Heads are scored on dev facts they had no label for (`kfinal` / `efinal` sets: labels on final facts only). Mean ± sd over 3 seeds.

| main_v0, dev | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.571 ± 0.043 | 0.674 ± 0.034 | 0.829 ± 0.004 |
| AURC head (lower = better) | 0.490 ± 0.054 | 0.383 ± 0.054 | 0.270 ± 0.029 |
| AURC own log-prob | 0.207 ± 0.009 | 0.201 ± 0.041 | 0.199 ± 0.027 |
| Spearman vs teacher K | 0.143 ± 0.069 | 0.257 ± 0.024 | 0.552 ± 0.013 |
| overall accuracy | 0.463 ± 0.009 | 0.473 ± 0.044 | 0.467 ± 0.027 |

| main_w1, dev | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.686 ± 0.022 | 0.716 ± 0.036 | 0.817 ± 0.022 |
| AURC head (lower = better) | 0.425 ± 0.074 | 0.353 ± 0.039 | 0.273 ± 0.026 |
| AURC own log-prob | 0.234 ± 0.075 | 0.202 ± 0.021 | 0.200 ± 0.014 |
| Spearman vs teacher K | 0.386 ± 0.036 | 0.405 ± 0.058 | 0.630 ± 0.044 |
| overall accuracy | 0.444 ± 0.063 | 0.470 ± 0.025 | 0.468 ± 0.018 |

**Criterion A (K − matched s(e) AUROC ≥ +0.03): FAIL in both worlds** (main_v0 −0.103, main_w1 −0.030). **Criterion B (K-head AURC within 0.03 of
own log-prob): FAIL in both worlds** (+0.283, +0.191). Dev verdict: not all pass. Reading, limited to what was measured:
- The K target did not help; in main_v0 the K head is clearly worse than the matched `s(e)` head (−0.10 AUROC, more than 2 sd of either), in main_w1 it
  is lower by 0.03, inside seed noise but in the wrong direction. The K head also predicts the teacher's own K worse on unlabelled facts than the
  `s(e)` head does (Spearman 0.14 vs 0.26 in main_v0).
- Criterion B was far out of reach for the matched `s(e)` head as well (AURC 0.38 / 0.35 against ~0.20 for log-prob): on facts it has no label for,
  neither head approaches the model's own confidence.
- The all-labelled reference (0.83 AUROC) is **not** a fact-disjoint number: that head was trained on the scored facts' own labels, and has twice the
  labelled rows. The 0.83 → 0.67 drop to the matched control is therefore not an effect of the K target and should not be quoted as one.
  The matched control is the fair comparison.
- Accuracy: no clear cost at the means; main_w1's K set has a large seed spread (0.444 ± 0.063).
- Not tested: *why* the K head generalises worse. A possible reason is that K is near-binary (about 80% of exposed facts at 0 or 1) and a fact
  labelled 0 or 1 can be fitted by recognising the fact, whereas exposure count varies smoothly with how familiar the fact feels; this would need its own control.

Per the protocol nothing is changed after this result: the final stage (`kdev` / `edev`, scored on final facts) runs once as written.
**Superseded by Amendment 1 below: this dev stage was produced with bfloat16, placeholder-graded teacher labels, so the final stage has not been run.**

## Amendment 1 (1 Nov 2026; written before any rerun): float32 labels, shared grader v5
The team decided on 31 Oct (`configs/model.yaml`, `common/grading.py`) that all training and evaluation is float32 and that every reported number
is graded with the shared grader (`p2_whole_word_v5`). Checked against this experiment:
- All heads and LoRA adapters were trained in float32 (`p5_training_method/config.yaml`), so no training run is affected.
- The **teacher K labels were not**: `kq_labels.py` generated them through `ModelRunner` using this branch's copy of `configs/model.yaml`, which was still
  bfloat16 (the float32 change was on `main` only), and graded them with `run_eval`'s placeholder grader (`grading_version: placeholder_exact_v1`).

What changes and what does not:
- **Superseded:** the dev-stage tables above (bf16 / placeholder labels). They stay in this report as a record and are not a K(q) result.
- **Unchanged:** the teacher (`lam0`, seed 0, plain SFT), the held-out wordings, 2 wordings x 5 samples = 10 per fact, the K definition, the four label
  sets and their masks, 3 seeds, both worlds, criteria A and B, and the rule that the final stage runs once.
- **Changed:** `kq_labels.py` now stops unless dtype is float32 and regrades with `common/grading.py` before computing K; the label file records both
  `dtype` and `grading_version`. The dev stage is rerun on the new labels and **that** run is the dev result. The superseded bf16 run is reported
  beside it, so a reader can see whether the verdict depends on label precision. This is a correction of a setup error, not a second try at a failed
  test: nothing about the criteria, designs or thresholds is touched, and the new dev verdict is reported whichever way it falls.
- **Second arm, adopted 1 Nov 2026 (before any run): "team-spec labels".** The K target named in team decision 4 is P1's K per fact for `main_v0`
  (`p1_exposure/analysis/results/k_labels/main_v0_float32_seed0/`: float32, grader v5, the 4 training wordings x 5 samples = 20 per fact, from P1's own
  float32 run of the matched recipe, seed 0; P6's `sft_matched` was not yet available). It is run as an extra arm, never a replacement:
  - **Labels:** `p5_training_method/p1_labels.py` checks P1's file against `facts.jsonl` (600 facts, same exposure and split, K in [0, 1], 20 samples,
    float32, v5) and writes `results/final/p5_kq_labels_main_v0_p1.json`; nothing is recomputed. K is P1's, so it is measured on training wordings,
    where the primary arm uses held-out wordings; the two arms therefore differ in model, wordings and sample count, and a difference between them
    cannot be attributed to any one of these.
  - **Design:** the same masks and the same fact-disjoint scheme as the primary arm: `pfinal` (labels on final facts, scored on dev) and `pdev`
    (labels on dev facts, scored on final), 3 seeds, lambda 3. The matched s(e) controls are the primary arm's `efinal` / `edev` runs, not retrained.
  - **Criteria:** A and B exactly as above, scored by `kq_criteria.py --arm p1`, `main_v0` only (P1 has no `main_w1` file). The dev stage is run first and
    its verdict recorded here before the final stage, as for the primary arm.
  - **Reading rule:** both arms are reported whatever they show. The primary arm stays the headline; the P1 arm is a robustness check on how K is
    measured, and cannot be promoted after the results are seen. One caution carried from P1's own README: with training-wording K the head could memorise
    a fact's K instead of reading evidence; the fact-disjoint masks are the control for that, and any gain at 0x is read with the name-recognition caveat
    from `reports/p5_aux_zero_novel.md`.

## Dev-stage result after Amendment 1 (float32 labels, grader v5; recorded 2 Nov 2026, before any final-stage run)
Teacher labels: `results/final/p5_kq_labels_main_v0.json` and `_main_w1.json` (dtype float32, `p2_whole_word_v5`, 600 facts each; mean K by exposure
0 / 1 / 2 / 5 / 10 / 20: main_v0 0.005 / 0.071 / 0.223 / 0.673 / 0.848 / 0.926, main_w1 0.010 / 0.044 / 0.183 / 0.641 / 0.863 / 0.972).
Scored on dev facts the head had no label for; mean ± sd over 3 seeds. Commands: `kq_criteria.py --split dev` (primary arm, both worlds) and
`kq_criteria.py --split dev --arm p1` (P1 arm, main_v0).

**Grader scope (added 4 Nov 2026, before any v5 regrade of the scored side).** "v5" in this report applies to the **teacher K labels** only (`kq_labels.py`
regrades with `common/grading.py`; P1's labels are v5 too). The *scored* side, i.e. which of a student's own answers count as correct for "AUROC predicts
correct", AURC, accuracy and the log-prob columns, comes from `eval_head.py`, which grades by normalised substring containment ("p5_placeholder_v0"), not v5.
Training ran in float32 throughout. An earlier check (`regrade_check.py`, whole-word instead of substring, 13,200 answers) changed 0.098% of labels, but that
is not the v5 grader, so the tables below are provisional on this point until the scored side is regraded with v5 and the criteria rerun.

**Primary arm (held-out-wordings teacher labels), float32 / v5. The bfloat16 / placeholder values from the superseded run are in brackets.**

| main_v0, dev | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.601 ± 0.032 [0.571] | 0.674 ± 0.034 | 0.829 ± 0.004 |
| AURC head (lower = better) | 0.474 ± 0.020 [0.490] | 0.383 ± 0.054 | 0.270 ± 0.029 |
| AURC own log-prob | 0.201 ± 0.013 [0.207] | 0.201 ± 0.041 | 0.199 ± 0.027 |
| Spearman vs teacher K | 0.145 ± 0.050 [0.143] | 0.260 ± 0.026 | 0.556 ± 0.012 |
| overall accuracy | 0.469 ± 0.015 [0.463] | 0.473 ± 0.044 | 0.467 ± 0.027 |

| main_w1, dev | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.699 ± 0.016 [0.686] | 0.716 ± 0.036 | 0.817 ± 0.022 |
| AURC head (lower = better) | 0.412 ± 0.022 [0.425] | 0.353 ± 0.039 | 0.273 ± 0.026 |
| AURC own log-prob | 0.213 ± 0.006 [0.234] | 0.202 ± 0.021 | 0.200 ± 0.014 |
| Spearman vs teacher K | 0.420 ± 0.051 [0.386] | 0.406 ± 0.060 | 0.629 ± 0.045 |
| overall accuracy | 0.450 ± 0.010 [0.444] | 0.470 ± 0.025 | 0.468 ± 0.018 |

**P1 arm (P1's K on the 4 training wordings, 20 samples per fact), main_v0 only, float32 / v5:**

| main_v0, dev | K head (P1 labels) | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.609 ± 0.095 | 0.674 ± 0.034 | 0.829 ± 0.004 |
| AURC head (lower = better) | 0.443 ± 0.074 | 0.383 ± 0.054 | 0.270 ± 0.029 |
| AURC own log-prob | 0.201 ± 0.008 | 0.201 ± 0.041 | 0.199 ± 0.027 |
| Spearman vs teacher K | 0.255 ± 0.083 | 0.277 ± 0.048 | 0.556 ± 0.012 |
| overall accuracy | 0.464 ± 0.015 | 0.473 ± 0.044 | 0.467 ± 0.027 |

**Verdicts (criteria unchanged):**

| arm, world | A: K − matched s(e) AUROC (need ≥ +0.03) | B: K AURC − own log-prob AURC (need ≤ 0.03) |
|---|---|---|
| primary, main_v0 | −0.073 FAIL [−0.103] | +0.273 FAIL [+0.283] |
| primary, main_w1 | −0.017 FAIL [−0.030] | +0.199 FAIL [+0.191] |
| P1, main_v0 | −0.066 FAIL | +0.242 FAIL |

Dev verdict: **not all pass; criteria A and B fail in every arm and world.** Reading, limited to what was measured:
- Precision and grader did not change the verdict. Under float32 / v5 the K head moved by at most 0.03 AUROC against the superseded run (main_v0 +0.030,
  main_w1 +0.013), the criteria A and B gaps moved by at most 0.03, and every one still fails. The matched `s(e)` control is unchanged to three decimals
  (it never depended on K labels and was always float32), so the K-head change is the only thing that moved.
- The K target did not beat exposure as a head target. K minus `s(e)` is −0.073 / −0.017 (primary) and −0.066 (P1) in AUROC; in main_w1 the gap is
  inside seed noise, in main_v0 it is about two sd of the `s(e)` control, and for the P1 arm the K head's seed spread (±0.095) is wider than the gap, so
  that arm shows "not better", not "worse".
- Swapping the label source did not rescue K. The P1 labels (different model, training wordings, 20 samples) give 0.609 against 0.601 for the held-out-wording
  labels, a difference far inside seed noise. The two arms differ in several ways at once, so this is a robustness check and not a comparison of label recipes.
- Criterion B is out of reach for every head here, including the matched `s(e)` control (AURC 0.35 to 0.38 against about 0.20 for the model's own
  log-prob): on facts it had no label for, no head approaches the model's own confidence. The 0.83 all-labelled AUROC was trained on the scored facts'
  own labels and is not a fact-disjoint number; the matched control is the fair comparison.
- Accuracy: the K sets sit at 0.450 to 0.469 against 0.467 to 0.473 for the `s(e)` sets, which is within seed spread at the means.
- Not tested, as before: *why* K generalises no better than exposure. The near-binary shape of K (a fact labelled 0 or 1 can be fitted by recognising
  the fact) remains a hypothesis and would need its own control.

Per the protocol nothing is changed after this result. The final stage (`kdev` / `edev`, and `pdev` for the P1 arm, scored on final facts) runs once per arm as written.

## Results (final stage, run once per arm; recorded 3 Nov 2026)
Final stage: the `kdev` / `edev` / `pdev` sets (labels on dev facts) scored on the final split, whose facts the heads had no label for. Float32; teacher labels graded with v5. The tables below were scored before
the Amendment 2 regrade; the v5-scored values are in "Amendment 2: result" at the end of this report (no verdict changed). 3 seeds, mean ± sd. The dev verdicts above were committed and pushed before these runs. Jobs: 50314 (main_v0, primary arm), 50315 (main_w1, primary arm),
and the P1-arm `pdev` run (main_v0). Commands: `kq_criteria.py --split final` and `kq_criteria.py --split final --arm p1`.

| main_v0, final, primary arm | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.562 ± 0.040 | 0.659 ± 0.054 | 0.784 ± 0.027 |
| AURC head (lower = better) | 0.493 ± 0.028 | 0.400 ± 0.077 | 0.285 ± 0.027 |
| AURC own log-prob | 0.203 ± 0.021 | 0.212 ± 0.034 | 0.202 ± 0.015 |
| Spearman vs teacher K | 0.114 ± 0.029 | 0.214 ± 0.070 | 0.497 ± 0.041 |
| overall accuracy | 0.469 ± 0.027 | 0.456 ± 0.041 | 0.474 ± 0.016 |

| main_w1, final, primary arm | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.734 ± 0.021 | 0.803 ± 0.022 | 0.834 ± 0.026 |
| AURC head (lower = better) | 0.362 ± 0.014 | 0.302 ± 0.021 | 0.281 ± 0.028 |
| AURC own log-prob | 0.212 ± 0.025 | 0.206 ± 0.022 | 0.208 ± 0.011 |
| Spearman vs teacher K | 0.417 ± 0.048 | 0.537 ± 0.023 | 0.602 ± 0.045 |
| overall accuracy | 0.449 ± 0.024 | 0.452 ± 0.025 | 0.451 ± 0.015 |

| main_v0, final, P1 arm | K head (P1 labels) | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.626 ± 0.068 | 0.659 ± 0.054 | 0.784 ± 0.027 |
| AURC head (lower = better) | 0.455 ± 0.052 | 0.400 ± 0.077 | 0.285 ± 0.027 |
| AURC own log-prob | 0.204 ± 0.008 | 0.212 ± 0.034 | 0.202 ± 0.015 |
| Spearman vs teacher K | 0.275 ± 0.129 | 0.262 ± 0.070 | 0.520 ± 0.040 |
| overall accuracy | 0.459 ± 0.009 | 0.456 ± 0.041 | 0.474 ± 0.016 |

| arm, world | A: K − matched s(e) AUROC (need ≥ +0.03) | B: K AURC − own log-prob AURC (need ≤ 0.03) | dev A / B |
|---|---|---|---|
| primary, main_v0 | −0.096 FAIL | +0.291 FAIL | −0.073 / +0.273 |
| primary, main_w1 | −0.069 FAIL | +0.151 FAIL | −0.017 / +0.199 |
| P1, main_v0 | −0.033 FAIL | +0.251 FAIL | −0.066 / +0.242 |

**Verdict: criteria A and B fail in every arm and world on the final split, as on dev. The K target does not improve on the exposure target
`s(e) = log(1+e)/log(21)` as a head label, and no head reaches the model's own log-prob as an abstention signal.** Reading, limited to what was measured:
- Dev and final agree in sign and rough size for all three arm/world pairs, so the result was not a dev-split accident.
- In the primary arm the K head is below the matched `s(e)` head on final by 0.096 (main_v0) and 0.069 (main_w1) AUROC. That is about two sd of the
  `s(e)` control in main_v0 and about three in main_w1, so there it is a worse head, not just "not better". In the P1 arm the gap (−0.033) is inside the K head's
  seed spread (±0.068): "not better", with no evidence of "worse".
- The K head also predicts the teacher's own K worse than the `s(e)` head does on unlabelled facts in the primary arm (Spearman 0.114 vs 0.214 in main_v0,
  0.417 vs 0.537 in main_w1); in the P1 arm they are equal (0.275 vs 0.262, large spread).
- Criterion B stays out of reach for every head, including the matched `s(e)` control (AURC 0.30 to 0.40 against about 0.20 for log-prob). The
  all-labelled column was trained on the scored facts' own labels and is a reference, not a fact-disjoint result.
- Accuracy is unchanged within seed spread (K sets 0.449 to 0.469, `s(e)` sets 0.452 to 0.456), so the K aux loss did not cost answer quality.
- **What this does not show.** One teacher (plain SFT, seed 0), two synthetic worlds, a 0.5B model, 3 seeds, and heads trained on half the facts (the masking
  that keeps train and score facts disjoint). The explanation that K is near-binary and so is fitted by recognising a fact stays an untested hypothesis. The
  K target was also not tried at other λ values or as a replacement for the model's own signal, since the pre-stated question was whether it beats `s(e)` as a head target.

**Consequence for the P5 recipe.** The K(q) target is a negative result and is not adopted: the headline recipe keeps `s(e)` with λ = 3, and the model's average
log-prob remains the better abstention signal (see `reports/p5_aux_zero.md` and the earlier abstention results). The K(q) item in team decision 4 is therefore closed for P5 with this report.

## Amendment 2 (4 Nov 2026; written before the regrade is applied): scored-side grading moves to v5
The scored side of every P5 result (which of a model's own answers count as correct) was graded by `eval_head.py`'s substring grader, not by the shared v5
grader (see the grader scope note above). This applies to the K(q) results here and equally to the sweep, abstention, probe-control and aux-zero reports.
- **What is re-scored:** the same saved answers, nothing is generated or retrained. `p5_training_method/regrade_v5.py` rewrites each `predictions.jsonl` with the
  v5 label (originals kept as `predictions_containment.jsonl`); `eval_head.py` now grades with v5 for any future run.
- **Measured before applying (dry run, 102 prediction files, 60,480 answers):** 48 labels change (0.079%), all `correct` → `incorrect`, none involve abstentions;
  the largest accuracy shift in any one file is −0.83 points. The cases are longer words that contain the reference ("pottery" for "potter",
  "Drouzhakrou" for "Drouzhak"), which v5 does not accept.
- **Re-run after applying:** `probe_controls.py` and the criteria scripts for every experiment (K(q) own and P1 arms, aux-zero, aux-zero novel-name control,
  sweep, abstention). The criteria, thresholds, designs and the one-shot rule are untouched. The final-split scoring is re-scored once as a grader
  correction, not as a new test run.
- **Reporting rule:** each verdict is reported under v5, with the previous substring-graded value beside it, and any verdict that changes is said to have changed.
  The v5 numbers replace the old ones as the result; the old ones stay as the record.

### Amendment 2: result (4 Nov 2026; regrade applied, K(q) re-scored under v5)
`regrade_v5.py --apply` changed 48 of 60,480 saved labels (0.079%, all `correct` → `incorrect`, none abstentions, largest accuracy shift in one file −0.83 points).
The K(q) jobs were then rerun with training and eval skipped (checkpoints and summaries exist; the logs show 0 train / eval lines), so only `probe_controls.py` re-scored,
followed by `kq_criteria.py`. Criteria, thresholds and designs are those of the original protocol. Values are v5, with the earlier substring-graded value in brackets.

| arm, world, split | K-head AUROC | K-head AURC | A: K − s(e) AUROC (≥ +0.03) | B: K AURC − log-prob AURC (≤ 0.03) |
|---|---|---|---|---|
| primary, main_v0, dev | 0.601 [0.601] | 0.475 [0.474] | −0.074 FAIL [−0.073] | +0.273 FAIL [+0.273] |
| primary, main_w1, dev | 0.698 [0.699] | 0.413 [0.412] | −0.018 FAIL [−0.017] | +0.200 FAIL [+0.199] |
| P1 labels, main_v0, dev | 0.609 [0.609] | 0.443 [0.443] | −0.066 FAIL [−0.066] | +0.242 FAIL [+0.242] |
| primary, main_v0, final | 0.561 [0.562] | 0.495 [0.493] | −0.098 FAIL [−0.096] | +0.292 FAIL [+0.291] |
| primary, main_w1, final | 0.734 [0.734] | 0.362 [0.362] | −0.070 FAIL [−0.069] | +0.151 FAIL [+0.151] |
| P1 labels, main_v0, final | 0.627 [0.626] | 0.457 [0.455] | −0.032 FAIL [−0.033] | +0.252 FAIL [+0.251] |

No K(q) verdict changed: every A and B value moved by at most 0.002, and all twelve criteria still fail. The Spearman columns do not depend on grading and did not change.
The v5 values above replace the earlier ones as the result; the earlier ones, and the pre-regrade result files, are kept (git history for the K(q) files, and
`results/superseded_containment_grader_final/` on the cluster for all of `results/final`). The v5 summaries of the sweep, abstention and aux-zero reports are separate and follow in their own reports.

## Amendment 3 (4 Nov 2026; written before any run of this arm): third arm on K from P6's `sft_matched` adapter
The team plan names the K source: "the float32 `sft_matched` model" (decision 4, and Phase 2 of the work order). The two arms run so far do not use it: the primary arm uses this branch's own
teacher (λ=0, seed 0, held-out wordings) and the P1 arm uses P1's own float32 run of the recipe (`main_v0_float32_seed0`), which P1's README now marks "superseded for P5's use". P1 has since
published K from P6's adapter (`p1_exposure/analysis/results/k_labels/main_v0_float32_p6_adapter/`). This arm tests exactly that source, so the negative result can be stated for the labels the plan names.
- **Labels:** `p1_labels.py --source p6_adapter` checks P1's file against `facts.jsonl` (600 facts, same exposure and split, K in [0, 1], 20 samples per fact = 4 training wordings × 5, float32, v5) and writes
  `results/final/p5_kq_labels_main_v0_p6.json` with the adapter's weights hash; nothing is recomputed. Mean K by exposure 0 / 1 / 2 / 5 / 10 / 20: 0.004 / 0.079 / 0.243 / 0.742 / 0.856 / 0.961
  (higher than the P1 own-run labels at every level). P1 reports that the two runs' K agree on bucket for 83% of facts and that the part of K beyond the exposure level correlates only 0.62 between them.
- **Provenance limit:** the adapter's own files do not say which data, seed, data order or code produced it (its README is the empty PEFT template). Matching the recipe (main_v0 `train.jsonl`, seed 0, one epoch, lr 2e-4, batch 4) is P6's to confirm.
  K is also one run of one model, and K differs between runs; a second K file from another seed would be needed to separate that from the target's effect. Not available.
- **Design:** the same masks and fact-disjoint scheme: `sfinal` (labels on final facts, scored on dev) and `sdev` (labels on dev facts, scored on final), λ = 3, 3 seeds, `main_v0` only (P1 made no such file for
  `main_w1`). The matched s(e) controls are the primary arm's `efinal` / `edev`, not retrained. `p5_kq.sbatch` with `ARM=p6`; criteria via `kq_criteria.py --arm p6`.
- **Criteria:** A and B exactly as in the protocol (K − matched s(e) AUROC ≥ +0.03; K-head AURC within 0.03 of the student's own log-prob AURC). Scoring uses the v5 grader on both the labels and the students' answers
  (`eval_head.py` now grades with v5). Dev stage first, its verdict recorded and committed here, then the final stage once.
- **Reading rule:** reported whatever it shows. The primary arm stays the headline and this arm cannot replace it after the results are seen. Expectation, written now: it will agree with the other two arms. A pass of both A and B here
  would be a new finding and would need a second K file and a second look before it is believed.

### Amendment 3: dev result (job 51135, `kq_criteria.py --split dev --arm p6`; recorded 4 Nov 2026, before the final stage of this arm)
`main_v0`, scored on dev facts the head had no label for (`sfinal` students), λ = 3, 3 seeds, mean ± sd; v5 grader throughout.

| main_v0, dev, P6-adapter labels | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.589 ± 0.013 | 0.674 ± 0.034 | 0.829 ± 0.004 |
| AURC head (lower = better) | 0.466 ± 0.040 | 0.383 ± 0.054 | 0.270 ± 0.029 |
| AURC own log-prob | 0.216 ± 0.033 | 0.201 ± 0.041 | 0.199 ± 0.027 |
| Spearman vs teacher K | 0.189 ± 0.027 | 0.276 ± 0.048 | 0.566 ± 0.022 |
| overall accuracy | 45.2% ± 4.2 | 47.3% ± 4.4 | 46.7% ± 2.7 |

**Criterion A (K − matched s(e) AUROC ≥ +0.03): FAIL** (−0.086). **Criterion B (K-head AURC within 0.03 of own log-prob): FAIL** (+0.250). Dev verdict: not all pass, as expected in Amendment 3.
Reading, limited to what was measured:
- The K head is below the matched `s(e)` head by 0.086 AUROC, about 2.5 sd of the control's seed spread (0.034), the same direction and a similar size as the primary arm on `main_v0` dev (−0.074) and the P1 own-run arm (−0.066).
  Using the K file the plan names did not change the picture.
- Criterion B stays far out of reach (+0.250), as for every other head; the matched `s(e)` control is at +0.18.
- This arm's students have a wider accuracy spread (45.2% ± 4.2) and a higher own-log-prob AURC (0.216 ± 0.033) than the other `main_v0` sets, so a single seed moves its means more; none of that changes a verdict.
- The final stage of this arm (`sdev` students scored on final) runs once, next.

### Amendment 3: final result (job 51149, `kq_criteria.py --split final --arm p6`; run once, recorded 4 Nov 2026)
`main_v0`, `sdev` students (labels on dev facts) scored on final facts they had no label for, λ = 3, 3 seeds, mean ± sd; v5 grader throughout.

| main_v0, final, P6-adapter labels | K head | s(e) head (matched) | s(e), all labelled (reference) |
|---|---|---|---|
| AUROC predicts correct | 0.596 ± 0.021 | 0.659 ± 0.054 | 0.784 ± 0.027 |
| AURC head (lower = better) | 0.469 ± 0.036 | 0.400 ± 0.077 | 0.285 ± 0.027 |
| AURC own log-prob | 0.201 ± 0.019 | 0.212 ± 0.034 | 0.202 ± 0.015 |
| Spearman vs teacher K | 0.178 ± 0.024 | 0.254 ± 0.068 | 0.502 ± 0.046 |
| overall accuracy | 46.7% ± 1.4 | 45.6% ± 4.1 | 47.4% ± 1.6 |

**Criterion A: FAIL** (−0.062). **Criterion B: FAIL** (+0.268). The final result agrees with the dev result for this arm (−0.086 / +0.250) in sign and rough size.
- On final the K head is below the matched `s(e)` head by 0.062 AUROC, about one sd of the control's seed spread (0.054), so this arm shows "not better" and, as for the P1 own-run arm, no clear evidence of "worse".
  The primary arm's gaps are larger relative to their spread (−0.098 and −0.070).
- Criterion B again fails by a wide margin, and accuracy is unchanged within seed spread (46.7% against 45.6%).

**All arms, both splits (v5; A = K − matched s(e) AUROC, B = K AURC − own log-prob AURC):**

| arm, world | dev A / B | final A / B |
|---|---|---|
| primary (own teacher), main_v0 | −0.074 / +0.273 | −0.098 / +0.292 |
| primary (own teacher), main_w1 | −0.018 / +0.200 | −0.070 / +0.151 |
| P1 own-run labels, main_v0 | −0.066 / +0.242 | −0.032 / +0.252 |
| P6 `sft_matched` adapter labels, main_v0 | −0.086 / +0.250 | −0.062 / +0.268 |

All 16 criterion checks (4 arm-world pairs × dev and final × A and B) fail. **Conclusion: with each of three label sources, including the one the plan names, in both worlds
where the label exists, a head trained on K does not beat a head trained on the exposure target, and no head approaches the model's own log-probability as an abstention signal.**
Still not tested: why K does no better than exposure (the near-binary shape of K remains a hypothesis); a second K file from another training seed for the P6-adapter model; any K source for `main_w1` other than this branch's own teacher.
