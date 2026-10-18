# P5 — Evidence-aware fine-tuning: results and controls

Owner: P5. Status: first complete pass on `main_v0` (3 seeds). Grading is P6's placeholder containment
grader, not P2's, so every number below is **provisional**.

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

## 3. Results

### 3.1 λ sweep (dev, mean ± std over seeds)

| λ | seeds | overall acc | 2× | 5× | 10× | head Spearman | AUROC ≥5× vs ≤2× | AUROC predicts correct |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 48.2 ± 0.4 | .24 | .77 | .87 | 0.01 ± .09 | 0.503 ± .057 | 0.529 ± .029 |
| 0.3 | 3 | 45.0 ± 1.3 | .20 | .66 | .85 | 0.49 ± .07 | 0.733 ± .048 | 0.713 ± .025 |
| 1 | 3 | 46.0 ± 1.2 | .25 | .73 | .78 | 0.61 ± .03 | 0.808 ± .021 | 0.747 ± .010 |
| **3** | 3 | 46.7 ± 2.7 | .19 | .71 | .90 | 0.69 ± .01 | 0.871 ± .009 | 0.829 ± .004 |
| 10 | 3 | 42.8 ± 1.3 | .12 | .66 | .82 | 0.67 ± .01 | 0.879 ± .007 | 0.834 ± .014 |
| 100 | 1 | 25.2 | .02 | .08 | .41 | 0.66 | 0.881 | 0.954* |

\*At λ=100 the model only answers correctly at 20× and part of 10×, so "predicts correct" mostly means
"is 20×". Not a meaningful figure. λ=10 fails the 5-point rule (−5.4); λ=3 is within it (−1.5, inside seed noise).

### 3.2 Held-out test (final split, λ=3 locked, 3 seeds)

| | overall acc | 5× | 10× | head Spearman | AUROC ≥5× vs ≤2× | AUROC predicts correct |
|---|---|---|---|---|---|---|
| λ=0 (plain SFT) | 47.4 ± 1.4 | .64 | .88 | 0.04 ± .09 | 0.523 ± .047 | 0.531 ± .036 |
| λ=3 | 47.4 ± 1.6 | .66 | .92 | 0.66 ± .04 | 0.848 ± .024 | 0.784 ± .027 |

No accuracy cost. The λ=0 head is untrained and is the no-skill reference. Neither model abstains on its
own (abstention 0%): both answer every never-seen (0×) question, and are right on at most ~1% of them.

### 3.3 Control: is the auxiliary loss needed? (same head architecture and target, frozen features)

Final split, mean ± std over 3 seeds:

| signal | Spearman | AUROC ≥5× vs ≤2× | AUROC predicts correct |
|---|---|---|---|
| P5 head (jointly trained, λ=3) | 0.657 ± .043 | 0.848 ± .024 | 0.784 ± .027 |
| probe on the λ=3 model | 0.757 ± .029 | **0.935 ± .019** | 0.858 ± .012 |
| probe on the plain-SFT model | 0.729 ± .010 | 0.899 ± .011 | 0.843 ± .011 |
| probe on the frozen base model | 0.264 ± .009 | 0.616 ± .005 | 0.597 ± .017 |
| SFT model's average answer log-prob | 0.706 ± .024 | 0.888 ± .014 | **0.945 ± .013** |
| untrained head (no-skill) | 0.035 ± .092 | 0.523 ± .047 | 0.531 ± .036 |

Dev gives the same ordering (probe on λ=3 model 0.943; on SFT 0.908; on base 0.619).
"Predicts correct" uses the answers of the model each signal belongs to (λ=3 model for the P5 head and the
P5 probe, plain-SFT model for the rest); both models score about 47%.

- LoRA SFT itself is what puts the evidence signal into the hidden state (base probe: 0.62).
- The auxiliary loss makes it **3.5 AUROC points easier to read** (0.935 vs 0.899, consistent on dev and final,
  larger than the seed spread). That is partly expected since the model was trained toward the target. On
  "predicts correct" the final-split gap (0.858 vs 0.843) is within noise.
- The jointly-trained head (0.848) underuses the representation compared with a probe on the same model
  (0.935): it sees the data once, at the model's learning rate.

### 3.4 Abstention policies (final split, thresholds from dev, mean over 3 seeds)

Penalty 1.0:

| policy | accuracy | hallucination | abstention | selective acc | net | AURC ↓ |
|---|---|---|---|---|---|---|
| SFT, never abstains | 47.4 | 52.6 | 0 | 47.4 | −0.05 | – |
| P5, never abstains | 47.4 | 52.6 | 0 | 47.4 | −0.05 | – |
| SFT + log-prob | 40.6 | 4.2 | 55.2 | 90.6 | 0.364 | 0.200 |
| P5 + log-prob | 41.3 | 6.2 | 52.5 | 87.5 | 0.352 | 0.202 |
| SFT + probe | 36.2 | 12.7 | 51.2 | 74.4 | 0.235 | 0.261 |
| P5 + probe | 36.6 | 9.9 | 53.5 | 79.5 | 0.266 | 0.251 |
| P5 + jointly-trained head | 31.7 | 9.5 | 58.8 | 77.0 | 0.222 | 0.285 |

Penalty 3.0 (sensitivity): net scores SFT+log-prob 0.288, P5+log-prob 0.313, P5+probe 0.179, SFT+probe 0.130,
P5+head 0.122; AURC unchanged. Same ranking of signals. (Full table: `results/final/p5_abstain_eval_pen3.json`.)

## 4. Interpretation

1. **Abstaining reduces hallucination a lot, whichever signal is used** (52.6% → 4–13%).
2. **The model's own average log-probability is the best abstention signal here**, and P5 training neither
   helps nor hurts it (SFT vs P5 net score flips order between penalty 1 and 3, within seed spread).
3. **The `s(e)`-trained head is a worse abstention signal than log-probability.** It estimates how much
   evidence the model saw for a fact, not whether this particular answer is right. At 5× exposure the model
   is right ~65% of the time, yet the head/probes abstain on 55–100% of those questions.
4. **The main P5 hypothesis is therefore not supported with the `s(e)` target on this data.** What is
   supported: the method costs no accuracy, and evidence-aware training does make the evidence signal more
   linearly readable than plain SFT.

## 5. Limitations (read before citing any number)

- Placeholder grader (normalized containment), not P2's. Synthetic facts only; no real-QA evaluation.
- 100 questions per exposure level per split, 3 seeds. Per-level numbers are noisy (±~5 points).
- The λ selection rule was stated after seeing seed-0 dev results.
- The head cannot distinguish 0×, 1× and 2× (they are mostly unlearned, 0–26% accuracy). It separates
  "well seen" from "barely seen", not the bottom of the scale.
- Probes were trained for 1000 full-batch steps on the same 3,800 training prompts, a deliberately
  generous baseline; their hyperparameters were not tuned.
- The abstention penalty (1 and 3) is a judgement call. Absolute numbers depend on it; the ranking did not.
- Runs differ from P6's `sft_matched` in dtype (float32 vs bfloat16) and seeding order (seed set before
  LoRA attach here). P6's `sft_matched` on pilot_v0 (43.3%) vs this pipeline's λ=0 on pilot_v0 (44.2%,
  before the system-prompt newline fix) agree within single-seed noise; no `main_v0` `sft_matched` run exists yet.
- The IDK baseline and `K(q)`-target version were not run.

## 6. Compute (A40, TC2 cluster, partition MGPU-TC2, qos normal)

- One training run on `main_v0` (950 steps): ~67–69 s wall, ~0.07 s/step, ~3.5k tokens/s, 3.3 GB peak GPU memory
  (λ=0 and λ=1 seed-0 logs). Eval, sweep, probe and final jobs each finished well inside their requested
  time limits (30 min for evals, 45 min for probes, 2 h for the sweep); exact wall times are in `sacct`.
- Total runs: 18 training runs (λ sweep: 5 values × 3 seeds = 15, plus λ=100 once, plus two earlier
  `pilot_v0` checkpoints that are not used in any table above), 3-seed dev and final evals, probe controls
  (dev and final).

## 7. Reproduction

Code: `p5_training_method/` (uncommitted when this was written). Per-run provenance (`run_config.json`
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
```

## 8. Open items

- Hand-off to P6: export P5 responses in the shared schema (1 greedy + 5 sampled per question, `confidence_score`
  = the chosen signal) so `compare_methods.py` can include P5 beside base/SFT/IDK. Needs `common/` from main.
- `K(q)`-target version (plan Week 4): the one variant that could change conclusion 4.
- Re-grade with P2's grader once available; more seeds if any claim above is to be quoted as significant.
