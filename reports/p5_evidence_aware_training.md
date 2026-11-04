# P5 — Evidence-aware fine-tuning: results and controls

Owner: P5. Status: first complete pass on `main_v0` (3 seeds), re-scored with the shared v5 grader on 4 Nov 2026 (see the
amendment at the end of this report; no conclusion changed). Section 3.5 was rebuilt from the float32 exports and P6's float32 `sft_matched`.

## 1. What was tested

Hypothesis (plan.md, P5): fine-tuning with a small auxiliary head that predicts an evidence target,
`L = L_answer + λ·L_aux`, produces a model whose internal state carries "how much evidence did I see
for this fact", and that signal can reduce hallucination (e.g. by abstaining when it is low).

- **Model:** Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, float32.
- **LoRA:** r=16, alpha=32, dropout 0.05, all 7 linear modules (q/k/v/o/gate/up/down), lr 2e-4, batch 4,
  1 epoch, answer-only loss, data in file order. This is P1's recipe, the one P6 used for `sft_matched`
  after the training-parity fix (`reports/p6_training_parity_fix.md`).
- **Head:** non-affine LayerNorm + Linear(896, 1) + sigmoid, in float32, reading the final-layer hidden
  state at the last prompt token. Target `s(e) = log(1+e)/log(21)`, MSE loss. λ=0 is plain SFT run through
  the same code path (the head exists but receives no gradient).
- **Data:** `data/synthetic/main_v0` (P1; origin/main commit `6edf030`): 600 facts, 3,800 training
  presentations, 100 facts per exposure level (0/1/2/5/10/20×). The eval questions use wordings never seen
  in training and are split by fact into `dev` (300 facts) and `final` (300 facts), 100 questions per
  exposure level in each.
- **Decoding for evaluation:** true greedy (`top_k=None`, `repetition_penalty=1.0`), matching `decoding_v1`.
- **System prompt:** loaded verbatim from `config.yaml`, byte-identical to P6's `configs/decoding.yaml`
  (including the trailing newline).

## 2. Protocol (what was chosen on what)

1. λ was swept on the **dev** split only: λ ∈ {0, 0.3, 1, 3, 10} × seeds {0, 1, 2}, plus λ=100 once.
2. Selection rule: among λ whose mean dev accuracy is within 5 points of λ=0, take the highest mean
   AUROC(head predicts correct). **The rule was written after seeing the seed-0 results**, so it is a
   stated rule, not a pre-registered one. It picked λ=3.
3. The **final** split was scored once for λ=0 and λ=3 (3 seeds each), reusing the sweep's checkpoints
   (training does not depend on the split). λ was not revisited after seeing any final number.
4. Controls and the abstention test were specified (probe settings, threshold rule, penalty) before their
   results were seen. Probe settings: Adam, lr 1e-3, 1000 full-batch steps. Abstention threshold: chosen
   per seed on dev by maximising `accuracy − penalty·hallucination`, then applied to that seed's final scores.
5. **Probes are fact-disjoint.** A first version fitted the probes on the training rows of all 600 facts,
   including the facts they were then scored on, so they could memorise per-fact exposure labels (found in
   review). Probes scored on the final split are now fitted only on dev facts' training rows, and the reverse,
   with an assertion that no fact is on both sides. All probe and abstention numbers below are from the
   corrected run; the first version's numbers are superseded (they are in the git history only).
   The models and the jointly-trained head are **not** changed by this: they train on all facts by the
   experiment's design (the evaluation questions are new wordings of facts the model was trained on).

## 3. Results

### 3.1 λ sweep (dev, mean ± std over seeds)

| λ | seeds | overall acc | 2× | 5× | 10× | head Spearman | AUROC ≥5× vs ≤2× | AUROC predicts correct |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 48.2 ± 0.4 | .24 | .77 | .87 | 0.01 ± .09 | 0.503 ± .057 | 0.529 ± .029 |
| 0.3 | 3 | 44.9 ± 1.4 | .20 | .66 | .85 | 0.49 ± .07 | 0.733 ± .048 | 0.713 ± .025 |
| 1 | 3 | 45.7 ± 0.9 | .24 | .73 | .78 | 0.61 ± .03 | 0.808 ± .021 | 0.747 ± .010 |
| **3** | 3 | 46.7 ± 2.7 | .19 | .71 | .90 | 0.69 ± .01 | 0.871 ± .009 | 0.829 ± .004 |
| 10 | 3 | 42.7 ± 1.4 | .12 | .65 | .82 | 0.67 ± .01 | 0.879 ± .007 | 0.834 ± .014 |
| 100 | 1 | 24.8 | .02 | .06 | .41 | 0.66 | 0.881 | 0.957* |

\*At λ=100 the model only answers correctly at 20× and part of 10×, so "predicts correct" mostly means
"is 20×". Not a meaningful figure. λ=10 fails the 5-point rule (−5.5); λ=3 is within it (−1.5, inside seed noise).

### 3.2 Held-out test (final split, λ=3 locked, 3 seeds)

| | overall acc | 5× | 10× | head Spearman | AUROC ≥5× vs ≤2× | AUROC predicts correct |
|---|---|---|---|---|---|---|
| λ=0 (plain SFT) | 47.3 ± 1.3 | .64 | .88 | 0.04 ± .09 | 0.523 ± .047 | 0.533 ± .038 |
| λ=3 | 47.4 ± 1.6 | .66 | .92 | 0.66 ± .04 | 0.848 ± .024 | 0.784 ± .027 |

No accuracy cost. The λ=0 head is untrained and is the no-skill reference. Neither model abstains on its
own (abstention 0%): both answer every never-seen (0×) question, and are right on at most ~1% of them.

### 3.2b What the headline AUROC is made of (pairwise breakdown, final split, mean ± std over 3 seeds)

The headline "≥5× vs ≤2×" AUROC pools the never-trained 0× facts with 1× and 2×. Split by pair (suggested in review;
the head never trains on 0× because no 0× presentations exist). AUROC 0.5 = chance; **below 0.5 = the signal ranks the
"more evidence" group lower**.

| comparison | P5 head | probe on λ=3 model | probe on SFT model | P5 model log-prob | SFT model log-prob |
|---|---|---|---|---|---|
| ≥5× vs ≤2× (headline) | 0.848 ± .024 | 0.807 ± .014 | 0.680 ± .011 | 0.902 ± .004 | 0.888 ± .014 |
| ≥5× vs 0× only | 0.811 ± .031 | 0.785 ± .009 | 0.693 ± .033 | 0.910 ± .010 | 0.898 ± .025 |
| ≥5× vs 1×/2× only | 0.867 ± .021 | 0.818 ± .016 | 0.673 ± .004 | 0.898 ± .001 | 0.883 ± .010 |
| 5× vs 1×/2× only | 0.668 ± .034 | 0.589 ± .034 | 0.600 ± .028 | 0.769 ± .008 | 0.755 ± .030 |
| 1×/2× vs 0× | **0.349 ± .009** | 0.426 ± .017 | 0.514 ± .035 | 0.496 ± .043 | 0.486 ± .020 |

Dev gives the same pattern for the head (≥5× vs 0× 0.826, ≥5× vs 1×/2× 0.894, 5× vs 1×/2× 0.733, 1×/2× vs 0× 0.342).

- The headline is **not** driven by 0×: the head separates ≥5× from 0× (0.81) and from 1×/2× (0.87) alike.
- The gradation is weaker than the headline suggests: 5× vs 1×/2× is only 0.67 (log-prob: 0.77).
- At the bottom of the scale the head is **systematically wrong**: it scores never-trained 0× facts *higher* than
  1×/2× facts (mean head score 0.66 at 0× vs 0.60 and 0.59 at 1× and 2×; AUROC 0.35, well below chance), so it
  abstains less on completely unknown facts than on barely-seen ones (section 3.4). The log-prob signals are at chance
  there (0.49–0.50). Hypothesised cause: the head never sees a 0× example, so it has no
  reason to place "no evidence" inputs at the low end. This was tested with P1's answer-masked 0× prompts as
  head-only rows (`reports/p5_aux_zero.md`): the ranking improved for the 0× names the head was shown, but a matched
  control (`reports/p5_aux_zero_novel.md`) found no improvement for 0× names it had never seen (AUROC 0.30–0.39,
  the baseline level). So the flaw stands: the head memorised the names instead of learning to recognise unknown entities.

### 3.3 Control: does the auxiliary loss make the evidence signal readable? (fact-disjoint probes)

Same head architecture and target, fitted on frozen hidden states of facts **disjoint** from the facts scored.
Final split, mean ± std over 3 seeds (dev in the last column for the headline metric):

| signal | Spearman | AUROC ≥5× vs ≤2× | AUROC predicts correct | dev: AUROC ≥5× vs ≤2× |
|---|---|---|---|---|
| P5 head (jointly trained, λ=3) † | 0.657 ± .043 | 0.848 ± .024 | 0.784 ± .027 | 0.871 ± .009 |
| probe on the λ=3 model | 0.616 ± .022 | **0.807 ± .014** | 0.774 ± .008 | 0.813 ± .019 |
| probe on the plain-SFT model | 0.347 ± .055 | 0.680 ± .011 | 0.668 ± .024 | 0.713 ± .043 |
| probe on the frozen base model | −0.046 ± .002 | 0.472 ± .002 | 0.481 ± .006 | 0.444 ± .003 |
| P5 model's average answer log-prob | 0.728 ± .019 | 0.902 ± .004 | 0.939 ± .005 | 0.931 ± .018 |
| SFT model's average answer log-prob | 0.706 ± .024 | 0.888 ± .014 | **0.946 ± .012** | 0.914 ± .010 |
| untrained head (no-skill) | 0.035 ± .092 | 0.523 ± .047 | 0.533 ± .038 | 0.503 ± .057 |

"Predicts correct" uses the answers of the model each signal belongs to (λ=3 model for the P5 head, the P5
probe and the P5 log-prob; plain-SFT model for the rest); both models score about 47%.
† The head is trained jointly on all facts (see limitations), so it is **not** a like-for-like comparison with
the fact-disjoint probes; do not read the head-vs-probe gap as "the head is better".

- The base model's probe is at chance (0.47): it has never seen these facts, so there is nothing to read.
- Plain LoRA SFT yields a hidden state from which exposure is only weakly readable across facts (0.68).
- The λ=3 model's hidden state is **much** more readable (0.81): **+0.13 AUROC on final, +0.10 on dev**, far
  larger than the seed spread (~0.02). So the auxiliary loss does make the evidence signal more readable.
- Caveat on that claim: the λ=3 model was itself trained with the exposure labels of these facts, so a
  fact-disjoint probe can read labels the *model* stored. The result shows the aux-trained model stores a
  readable evidence signal for facts it has seen; it does **not** show the signal transfers to facts with no
  training (for which the true exposure is 0 by definition, so that test is not well defined here).

### 3.4 Abstention policies (final split, thresholds from dev, mean over 3 seeds)

Penalty 1.0:

| policy | accuracy | hallucination | abstention | selective acc | net | AURC ↓ |
|---|---|---|---|---|---|---|
| SFT, never abstains | 47.3 | 52.7 | 0 | 47.3 | −0.05 | – |
| P5, never abstains | 47.4 | 52.6 | 0 | 47.4 | −0.05 | – |
| SFT + log-prob | 40.6 | 4.2 | 55.2 | 90.6 | 0.364 | 0.200 |
| P5 + log-prob | 41.3 | 6.2 | 52.5 | 87.5 | 0.352 | 0.202 |
| SFT + probe (fact-disjoint) | 26.3 | 15.2 | 58.5 | 63.2 | 0.111 | 0.381 |
| P5 + probe (fact-disjoint) | 29.8 | 10.1 | 60.1 | 74.6 | 0.197 | 0.292 |
| P5 + jointly-trained head † | 31.7 | 9.5 | 58.8 | 77.0 | 0.222 | 0.285 |

Penalty 3.0 (sensitivity):

| policy | accuracy | hallucination | abstention | selective acc | net | AURC ↓ |
|---|---|---|---|---|---|---|
| SFT + log-prob | 40.0 | 3.7 | 56.3 | 91.6 | 0.288 | 0.200 |
| P5 + log-prob | 37.0 | 1.9 | 61.1 | 95.2 | 0.313 | 0.202 |
| SFT + probe (fact-disjoint) | 6.2 | 1.5 | 92.3 | 82.5 | 0.017 | 0.381 |
| P5 + probe (fact-disjoint) | 22.3 | 3.9 | 73.7 | 85.6 | 0.105 | 0.292 |
| P5 + jointly-trained head † | 25.4 | 4.4 | 70.2 | 85.3 | 0.122 | 0.285 |

† trained on all facts, so not like-for-like with the fact-disjoint probes. Full tables:
`results/final/p5_abstain_eval.json`, `results/final/p5_abstain_eval_pen3.json`.

### 3.5 Hand-off to P6: matched comparison through P6's `compare_methods.py` (seed 0, all 1,200 eval questions; float32, v5)

Updated 4 Nov 2026. `export_responses.py` generates with P6's own `ModelRunner` and `run_eval` (**float32**, `decoding_v1`, 1 greedy + 5 sampled per question), grades with the
shared v5 grader, and writes one P6-schema file per model: `confidence_score` = average answer log-prob (the signal every method has), `aux_confidence_score` = the evidence head's output,
one value per question (null for λ=0). P6's float32 `sft_matched` responses (`p1_main_v0_sft_matched.jsonl`, shared 4 Nov 2026, v5, `decoding_v1`) are the baseline. `compare_methods.py`
accepted all three files (matched-question check passed on 1,200 question_ids). Dev and final questions are pooled, one seed.

| method label | file | accuracy | mean K | AURC ↓ (confidence = log-prob) |
|---|---|---|---|---|
| `sft_matched` (P6's float32 run, their code) | `p1_main_v0_sft_matched.jsonl` | 48.6% | 0.465 | 0.186 |
| `sft_p5code` (λ=0, P5 code path) | `p5_main_v0_lam0_seed0.jsonl` | 48.2% | 0.457 | 0.203 |
| `p5` (λ=3) | `p5_main_v0_lam3_seed0.jsonl` | 48.8% | 0.454 | 0.183 |

Paired accuracy differences (question-cluster bootstrap, 95% CI), baseline `sft_matched`: `sft_p5code` −0.4 points [−2.3, +1.5]; `p5` +0.3 points [−1.6, +2.0]. Within the P5
pipeline, λ=3 vs λ=0 is +0.7 point (no accuracy cost). Full output: `results/final/p5_compare_methods_seed0_float32.json` (head AURC: `…_float32_head.json`).

- **P5 vs P6's plain SFT: no difference.** Both P5 variants are within about 0.5 point of `sft_matched`, with intervals that comfortably include zero. The earlier bfloat16 table showed +1.8 points for both
  P5 variants; that gap is gone now that both pipelines are float32 and v5-graded. That fits the earlier explanation (dtype) but cannot separate it from the grader change, since both moved at once.
- **Head vs log-prob as confidence, same λ=3 answers:** AURC of `aux_confidence_score` is **0.255** vs **0.183** for log-prob (`compare_methods.py --confidence-field aux_confidence_score`). Same ordering as section 3.4.
- **AURC across methods is one seed, pooled dev+final, with no interval.** The spread (0.183 to 0.203) is about the size of the 3-seed spread in section 3.4 (0.200 vs 0.202 on final), so do not read it as an effect of P5 training.
- Superseded for the record: the earlier bfloat16 / placeholder-graded table had `sft_matched` 46.8% (AURC 0.213), `sft_p5code` 48.7% (0.200), `p5` 48.6% (0.185), and head AURC 0.258.

**For P6:** to include the head in the shared table, `compare_methods.py` needs an option to rank by `aux_confidence_score` (the `--confidence-field` option already does this one method at a time). A combined base / SFT / IDK /
P5 table needs P6's base and IDK files rerun under the current config (`decoding_v1`, float32).

## 4. Interpretation

1. **Abstaining reduces hallucination a lot, whichever signal is used** (52.6% → 1.5–15%).
2. **The model's own average log-probability is the best abstention signal here**, and P5 training neither
   helps nor hurts it (SFT vs P5 net score flips order between penalty 1 and 3, within seed spread). The
   ranking of signals is the same at both penalties.
3. **The `s(e)`-trained head is a worse abstention signal than log-probability.** It estimates how much
   evidence the model saw for a fact, not whether this particular answer is right. At 5× exposure the model
   is right ~65% of the time, yet the head/probes abstain on 77–95% of those questions at penalty 1 and
   89–100% at penalty 3.
4. **The auxiliary loss does make the evidence signal more readable** (section 3.3: a fact-disjoint probe
   reaches AUROC 0.81 on the λ=3 model vs 0.68 on the plain-SFT model), at no accuracy cost (47.4% both).
5. **The practical benefit over the model's own confidence is not shown.** Log-probability is a stronger
   abstention signal than anything the evidence head or probes provide, with or without P5 training. So the
   hypothesis is supported as "the model can be trained to carry a readable evidence signal", and not
   supported as "that signal improves hallucination control beyond what log-probability already gives".

## 5. Limitations (read before citing any number)

- Grader: the tables above were first produced with P5's own substring grader (normalized containment) and re-scored with the shared v5 grader on 4 Nov 2026 ; see the amendment. Synthetic facts only; no real-QA evaluation.
  Impact check of the first, whole-word comparison (`regrade_check.py`, all 22 saved prediction files, 13,200 answers): whole-word matching changes
  **13 labels (0.098%)**, and at most 0.83 accuracy points in any single file. The changed cases are answers
  that extend the reference (e.g. "Drouzhakrou" for "Drouzhak", "tapestry weavers" for "tapestry weaver").
  Numeric references are all 4-digit years, and no reference is a substring of another.
- The models and the jointly-trained head train on all 600 facts (the experiment's design: evaluation asks new
  wordings of facts the model was trained on; 0× facts are never trained). The held-out test is therefore new
  wordings of known facts, not new facts, and the head may partly memorise per-fact exposure. Only the
  separate probe controls are made fact-disjoint.
- 100 questions per exposure level per split, 3 seeds. Per-level numbers are noisy (±~5 points).
- The λ selection rule was stated after seeing seed-0 dev results.
- The head cannot distinguish 0×, 1× and 2× (they are mostly unlearned, 0–26% accuracy), and ranks 0× facts
  above 1×/2× facts (AUROC 0.35, section 3.2b). It separates "well seen" from "barely seen", not the bottom of the scale.
- Dev and final differ in **question wording** as well as facts (dev uses P1's `val` templates, final uses `test`
  templates; training uses a third set of `train` templates). So the dev-chosen λ and abstention thresholds are
  tested on both new facts-to-score and new wording; shifts between dev and final numbers mix the two effects.
- All three seeds share one fact-to-exposure assignment (one generated synthetic world), so seeds vary training
  and probe initialisation, not the world. A second generated world (`main_w1`, from P1) was run with the same
  protocol and no re-tuning, and all five `main_v0` findings replicated (`reports/p5_second_world.md`).
- Each fact-disjoint probe is fitted on 1,900 training rows (250 facts, one half of the data) for 1000
  full-batch steps; its hyperparameters were not tuned. The head-vs-probe comparison is not like-for-like.
- The abstention penalty (1 and 3) is a judgement call. Absolute numbers depend on it; the ranking did not.
- Cross-check against P6's `sft_matched` (same recipe, float32, v5): on main_v0, P6 reports 48.6% (1,200 questions, seed 0) vs 48.2% (λ=0) and 48.8% (λ=3) for this pipeline through P6's runner (section 3.5);
  paired differences −0.4 [−2.3, +1.5] and +0.3 [−1.6, +2.0] points. One seed each; consistent within noise. (Earlier bfloat16 comparison, superseded: 46.8% vs 48.7%.)
- P5 did not run the IDK baseline. The `K(q)`-target version was run (4 Nov 2026) and is a negative result: `reports/p5_kq_target.md`.

## 6. Compute (A40, TC2 cluster, partition MGPU-TC2, qos normal)

- One training run on `main_v0` (950 steps): ~67–69 s wall, ~0.07 s/step, ~3.5k tokens/s, 3.3 GB peak GPU memory
  (λ=0 and λ=1 seed-0 logs). Eval, sweep, probe and final jobs each finished well inside their requested
  time limits (30 min for evals, 45 min for probes, 2 h for the sweep); exact wall times are in `sacct`.
- Total runs: 18 training runs (λ sweep: 5 values × 3 seeds = 15, plus λ=100 once, plus two earlier
  `pilot_v0` checkpoints that are not used in any table above), 3-seed dev and final evals, probe controls
  (dev and final).

## 7. Reproduction

Code: `p5_training_method/` (PR #8). Per-run provenance (`run_config.json`
with a hash of the code files and of the training data, versions, GPU) is under
`results/raw/p5/<run_id>/` on the cluster; the aggregate tables are under `results/final/p5_*.json`.

```bash
python -m unittest discover -s p5_training_method/tests -v      # gradient flow + save/reload (GPU job: jobs/p5_test.sbatch)
sbatch p5_training_method/jobs/p5_sweep.sbatch                   # train + dev-eval λ ∈ {0,0.3,1,3,10} × seeds {0,1,2}
python p5_training_method/aggregate_sweep.py                     # dev table + selection rule
sbatch p5_training_method/jobs/p5_final_eval.sbatch              # one-shot final split, λ ∈ {0,3}
python p5_training_method/aggregate_sweep.py --split final
sbatch p5_training_method/jobs/p5_probe.sbatch                   # probe controls (dev); PROBE_ARGS="--split final --allow-final" for final
python p5_training_method/abstain_eval.py [--penalty 3 --out ...]
python p5_training_method/regrade_check.py                       # substring vs whole-word grading on saved predictions
python p5_training_method/regrade_v5.py [--apply]                # regrade saved predictions with the shared v5 grader (dry run by default, originals kept)
sbatch p5_training_method/jobs/p5_rescore_v5.sbatch              # re-score probe controls, sweep, abstention and aux-zero tables after a regrade
sbatch p5_training_method/jobs/p5_export.sbatch                  # P6-schema response files (lambda 0 and 3, seed 0); ~1 h
python -m p6_baselines.compare_methods --responses ...           # see section 3.5
```

## 8. Open items

- Hand-off to P6: float32 / v5 export done for seed 0, and the matched `compare_methods.py` table rebuilt against P6's float32 `sft_matched` (4 Nov 2026, section 3.5). Still open: seeds 1 and 2
  (`EXPORT_ARGS="--seeds 1 2"`), and a combined table with P6's base and IDK files once they are rerun in float32.
- `K(q)`-target version: done (4 Nov 2026), negative: it did not beat the exposure target and no head approached the model's own log-prob (`reports/p5_kq_target.md`);
  conclusion 5 stands.
- Re-grade with the shared grader: done (4 Nov 2026, v5). More seeds if any claim above is to be quoted as significant.

## Amendment (4 Nov 2026): re-scored with the shared v5 grader
Sections 3.1 to 3.4 were first produced with `eval_head.py`'s own substring grader. The same saved answers were regraded with the shared v5 grader (`p5_training_method/regrade_v5.py`: 48 of
60,480 labels changed, all `correct` → `incorrect`, none abstentions, at most 0.83 accuracy points in any one file) and the sweep, probe-control and abstention tables rebuilt
(`p5_rescore_v5.sbatch`; plan and reporting rule in `reports/p5_kq_target.md`, Amendment 2). Nothing was retrained or regenerated. **No conclusion, verdict or selected λ changed**, and the tables above hold the v5 values. Section 3.5 was rebuilt separately (float32 exports against P6's float32 `sft_matched`); its old +1.8-point gap to P6 disappeared.
What moved, old → v5 (accuracy-type cells only; the exposure-based AUROCs, the pairwise breakdown in 3.2b and the Spearman values are unchanged):
- 3.1 dev sweep: λ=0.3 accuracy 45.0 ± 1.3 → 44.9 ± 1.4; λ=1 46.0 ± 1.2 → 45.7 ± 0.9; λ=10 42.8 ± 1.3 → 42.7 ± 1.4 (gap to λ=0 −5.4 → −5.5, still fails the 5-point rule); λ=100 25.2 → 24.8.
  λ=3 (46.7 ± 2.7, gap −1.5) is unchanged, and λ=3 is still the selected value.
- 3.2 final: λ=0 accuracy 47.4 ± 1.4 → 47.3 ± 1.3; its "predicts correct" AUROC 0.531 → 0.533. λ=3 is unchanged.
- 3.3 "AUROC predicts correct": plain-SFT probe 0.667 → 0.668, base probe ±.008 → ±.006, SFT log-prob 0.945 ± .013 → 0.946 ± .012, untrained head 0.531 → 0.533.
- 3.4 abstention: SFT never-abstains 47.4 / 52.6 → 47.3 / 52.7; SFT + probe selective accuracy 63.4 → 63.2, net 0.112 → 0.111, AURC 0.380 → 0.381 (penalty 3: AURC 0.380 → 0.381). Policy ranking unchanged at both penalties.
- Rerun noise: the probes are retrained on the GPU and are not bit-repeatable, so differences below about 0.001 are not effects.
