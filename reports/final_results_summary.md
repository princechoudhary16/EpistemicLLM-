# EpistemicLLM — final results summary

Model: `Qwen/Qwen2.5-0.5B-Instruct` (pinned revision), float32, shared prompt (`prompt_v0`), decoding (`decoding_v1`:
1 greedy + 5 sampled, T 0.7, top-p 0.9, 64 tokens), shared grader `p2_whole_word_v5`. Greedy answers for every
behaviour table; question-level bootstrap intervals (plan §6).

## 1. Research question (plan §1)

> How do factual exposure and empirical knowledge relate to accuracy, hallucination, confidence, and hidden
> representations — and does training on a continuous knowledge/evidence target improve reliability at comparable
> answer coverage?

The plan answers it in three parts: **measure knowledge** (P1 synthetic exposure, P2 real QA), **detect uncertainty**
(P3 hidden-state probes, P4 confidence and calibration) and **intervene and compare** (P5 evidence-aware training
against P6's standard and abstention baselines). A negative intervention result is acceptable (plan §1).

## 2. How the plan structures the results

| Track | Data | Methods compared on identical questions | Owners |
|---|---|---|---|
| A. Controlled (synthetic) | P1's `main_v0`: 600 invented facts at 0/1/2/5/10/20× exposure | base, standard SFT (`sft_matched`), IDK SFT, P5 evidence-aware SFT (λ=3), frozen-representation head control | P1, P5, P6 |
| B. Real QA | TriviaQA `rc.nocontext` (replaced SimpleQA after the Week 1 pilot): 300-question locked test | base; abstention baselines trained on TriviaQA's **train** file (this extension); confidence filters | P2 (+ P3, P4) |

Rules applied throughout: report accuracy, hallucination, abstention, coverage and selective accuracy together;
compare abstention methods **at matched coverage**; compare any confidence-threshold policy against the same policy on
baseline confidence (plan §7); fit and choose everything on development data and run the locked test once.

## 3. Track A — synthetic (owned by P5/P6; see `reports/p5_evidence_aware_training.md`, `reports/p6_week5_master_table.md`)

Float32 / v5, seed 0, 1,200 evaluation questions (P5 report §3.5):

| Method | Accuracy | AURC (log-prob) ↓ |
|---|---:|---:|
| sft_matched (standard SFT) | 48.6% | 0.186 |
| P5, λ=3 (evidence-aware) | 48.8% | 0.183 |
| P5 head as the confidence signal | — | 0.255 |

- Evidence-aware training makes exposure readable from the hidden state (fact-disjoint probe AUROC 0.81 vs 0.68 for
  plain SFT) at no accuracy cost.
- It does not improve reliability beyond the model's own log-probability: the head is a weaker abstention signal than
  log-prob (AURC 0.255 vs 0.183), and K(q) as the head target did not help (`reports/p5_kq_target.md`).

(Confirm with P6 which version of the synthetic master table is canonical before the final report.)

## 4. Track B — real QA (TriviaQA locked test, 300 questions)

**Plan context.** Real QA was evaluation-only: no answer fine-tuning on its development or test questions (plan §5).
This extension keeps that rule: abstention models are trained on TriviaQA's separate **train** file, with every
question sharing wording or a canonical answer with any evaluation split removed. It supplies what the real-QA track
lacked: a working "I don't know" baseline (P6's synthetic IDK baseline abstained on 0% or 100% of questions).
**R-Tuning-SA is the real-data IDK baseline** required by plan §7 ("IDK/abstention SFT"): it trains "I don't know" on
every training question the model gets wrong. The base + log-prob filter is the second required control (plan §7:
compare any abstention policy against the same policy on baseline confidence).

**Methods** (display names; file names in brackets):

| Name | What it is |
|---|---|
| **R-Tuning-SA** (`rtuning_own`) — **IDK baseline** | R-Tuning: all 20,000 training questions; right → the model's **own** correct answer, wrong → "I don't know." |
| **GeoDe-SA** (`geode_own`) | GeoDe: a probe on the last-prompt-token hidden state picks the 25% of questions it is surest about (gray zone dropped); "known" → the model's own correct answer, "unknown" → "I don't know." |
| Random-SA (`random_own`) | Control: a random 25%, labelled like R-Tuning-SA |
| base + log-prob filter | No training: the base model answers only its most confident questions (inference policy, plan §7) |

SA = self-answer: known questions are trained on the model's own correct answer, not the reference text. Training
recipe identical to `sft_matched` (LoRA r16, α32, all 7 linear layers, lr 2e-4, batch 4, 1 epoch).

### Table B1 — behaviour (mean ± sd over 3 seeds; base is one deterministic run)

| Method | Accuracy | Hallucination | Abstention | Coverage | Selective acc. | F1_rel |
|---|---:|---:|---:|---:|---:|---:|
| base | 17.7% | 76.0% | 6.3% | 93.7% | 18.9% | 19.7 |
| R-Tuning-SA (IDK baseline) | 1.9 ± 0.5% | 4.7 ± 1.7% | 93.4 ± 2.2% | 6.6 ± 2.2% | 29.5 ± 3.6% | 26.3 ± 4.3 |
| Random-SA (control) | 2.1 ± 0.8% | 2.2 ± 0.4% | 95.7 ± 1.2% | 4.3 ± 1.2% | 47.6 ± 6.8% | 28.4 ± 9.9 |
| **GeoDe-SA** | **6.0 ± 0.9%** | 7.3 ± 1.2% | 86.7 ± 1.9% | **13.3 ± 1.9%** | 45.0 ± 3.2% | **47.9 ± 1.3** |

F1_rel: GeoDe's balance of answering what the base model knew and refusing what it did not (formulas checked against
the GeoDe paper; "known" = base model's greedy answer correct).

### Table B2 — matched coverage (each row answers the same number of questions as GeoDe-SA, ~40 of 300)

| Method | Accuracy | Hallucination | Selective acc. | F1_rel |
|---|---:|---:|---:|---:|
| base + log-prob filter | 5.8 ± 0.8% | 7.6 ± 1.0% | 43.3 ± 0.4% | 52.3 ± 3.1 |
| **GeoDe-SA** (refuses by itself) | 6.0 ± 0.9% | 7.3 ± 1.2% | 45.0 ± 3.2% | 47.9 ± 1.3 |

(The filter cannot change a base answer, so it never turns a known answer wrong; that alone lifts its F1_rel.
Hallucination and selective accuracy are not affected by this.)

### Key comparisons (F1_rel difference, mean over 3 seeds, 95% question-bootstrap interval)

| | Comparison | Difference | Reading |
|---|---|---|---|
| **Required (plan §7)** | **GeoDe-SA − IDK baseline (R-Tuning-SA)** | **+21.6 [+13.3, +31.6]** | **reliably beats the IDK baseline** (validation: +24.2 [+10.8, +37.9]) |
| **Required (plan §7)** | **GeoDe-SA − base + log-prob filter** | **−4.4 [−16.1, +8.9]** | **no reliable difference: a tie**; training adds no reliability beyond the model's own confidence, but refuses without a score |
| Supporting | GeoDe-SA − Random-SA | +19.5 [+9.6, +29.3] | which questions are kept matters, not only how many |
| Supporting | IDK baseline − base + log-prob filter | −19.9 [−31.6, −6.8] | the IDK baseline loses to the free filter |
| Supporting | SA targets vs reference targets (validation) | GeoDe +27.1 [+16.3, +37.9], R-Tuning +14.3 [+0.1, +26.7] | training on the model's own answers is what makes abstention tuning work |

## 5. Findings

1. **Measuring knowledge.** The 0.5B model reliably knows few TriviaQA facts: about 18% greedy accuracy; 11% of
   training questions are right in ≥4 of 5 samples, 73% in none. A last-prompt-token probe predicts the model's
   correctness on unseen questions (AUROC 0.74, matching P3's 0.72–0.74).
2. **Training the model to refuse works only with its own answers.** Training known questions on one-word reference
   answers made the model lose facts it knew ("Boise" for Washington's capital); its own full-sentence answers fixed
   this (+14 to +27 F1_rel).
3. **GeoDe-SA is the best trained abstention method.** On the locked test it cuts hallucination from 76% to 7% and makes
   answers 2.4× as reliable (45% vs 19%), at 13% coverage, stably across seeds (F1_rel ± 1.3). It reliably beats the IDK
   baseline (R-Tuning-SA, +21.6 F1_rel [+13.3, +31.6]) and a same-size random subset.
4. **Nothing beats the model's own confidence.** At matched coverage GeoDe-SA ties a log-prob filter on the untouched
   base model, the only trained method to do so, and it refuses without needing a score or threshold at answer time.
   This matches Track A, where P5's head was a weaker signal than log-prob.

**Answer to the research question.** Exposure and empirical knowledge are readable from hidden states (P3, P5, probe
above), and training can make a model act on that knowledge by refusing. But neither an evidence target (P5) nor
abstention tuning on real data improves reliability beyond the model's own output log-probability at comparable
coverage; the best trained method matches it.

## 6. Limitations

- One model size (0.5B); results may differ for the paper-scale models (GeoDe: 1.7B–8B; R-Tuning: 3B–13B).
- 300 locked-test questions, 53 known to the base model: intervals are wide for close comparisons.
- The SA targets were chosen after the first validation run (a declared amendment); the untouched locked test confirms them.
- Known rows of GeoDe-SA are where the probe *and* actual correctness agree (probe-known but wrong rows had no own
  answer and were dropped), so it is GeoDe plus a correctness check, not GeoDe alone.
- F1_rel uses the base model's single greedy answer to define "known".

## 7. How this fits the final report (plan §11 checklist)

| Checklist item | Covered by |
|---|---|
| Controlled exposure and real factual QA both represented | Track A (P1/P5/P6) and Track B (P2 + this extension) |
| Base, standard SFT, IDK SFT and proposed method on identical questions within each experiment | Track A: P6/P5 table. Track B: base, IDK baseline (R-Tuning-SA), GeoDe-SA and the log-prob filter on the same 300 questions (no standard SFT on real QA: real QA is evaluation-only in the plan) |
| Accuracy, hallucination, abstention, coverage, selective accuracy reported together | Tables B1, B2 |
| Confidence policies compared fairly; uncertainty intervals; seed limits | Table B2 (matched coverage), bootstrap intervals, 3 seeds |
| No test leakage | Training from TriviaQA's train file; overlap check: 0 shared questions or canonical answers (2.7% alias-level overlap reported) |
| Traceable to data, checkpoints, responses, commands | `p2_abstention/README.md`, `results/raw/`, `results/final/abstention/` |

## 8. Reproduction

```bash
python -m p2_abstention.compare --split locked_test --arms own_rtuning own_random own_geode \
    --out results/final/abstention/comparison_locked_test_final.json
```
(`PAIRS` in `p2_abstention/compare.py` includes `("own_geode", "own_rtuning")`, which gives the GeoDe-SA − R-Tuning-SA
interval in the table above.)
Full pipeline: `p2_abstention/README.md`. Synthetic track: P5 and P6 reports.
