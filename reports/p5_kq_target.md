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
- **Proposed second arm (not yet adopted):** P1's K per fact for `main_v0` (`p1_exposure/analysis/results/k_labels/main_v0_float32_seed0/`, float32,
  v5, 4 training wordings x 5 samples, P1's own matched-recipe model) is the target named in team decision 4. If run, it is an extra arm labelled "team-spec
  labels", under the same criteria and masks, `main_v0` only (P1 has no `main_w1` file). It cannot replace the primary arm after results are seen.

## Results
_Final-stage result pending._
