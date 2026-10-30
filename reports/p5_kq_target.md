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

## Results
_Pending._
