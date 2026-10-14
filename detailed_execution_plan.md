# Detailed Execution Plan: LLM Knowledge, Hallucination and Evidence-Aware Fine-Tuning

**6 members | ~6 weeks | Parallel workstreams with minimal critical dependencies**

## 1. Project Objective

The project investigates how an LLM moves from not knowing a fact to learning it as the amount of training evidence changes. We will study this from three complementary perspectives: model behavior (correct answers, hallucinations and abstention), confidence/calibration, and internal hidden representations. In parallel, we will test whether an evidence-aware fine-tuning intervention can improve reliability on weakly supported knowledge.

**Core research question:** How does factual exposure affect what an LLM knows, how confidently it behaves, and how that knowledge is represented internally—and can we use evidence strength during fine-tuning to reduce confident hallucinations?

## 2. Why This Scope Fits a Six-Week Project

- The project does not depend on a single risky new loss succeeding. Behavioral, calibration and XAI tracks each produce standalone results.
- Each member has a self-contained experiment and can generate a small local dataset to validate their code before shared data is available.
- A single small open model keeps compute and engineering manageable while still allowing hidden-state access and LoRA/QLoRA fine-tuning.
- The controlled synthetic experiment gives causal evidence because the group decides exactly how often each fictional fact is shown.
- Real factual QA provides an external check so conclusions are not based only on synthetic data.

## 3. Shared Setup to Freeze in Days 1–3

The only major coordination point should happen at the start. The following decisions should be frozen early so that results from all six tracks remain comparable.

| Shared Decision | Recommended Starting Choice | Reason |
|---|---|---|
| Base model | One 1–3B instruction-tuned open LLM | Small enough for repeated inference, hidden-state extraction and PEFT. |
| Generation settings | Greedy run + 5 sampled runs/question for knowledge mapping | Separates deterministic performance from response stability. |
| Synthetic exposure levels | 0×, 1×, 2×, 5×, 10×, 20× | Creates a controlled evidence gradient. |
| Common answer outcomes | Correct / Incorrect non-abstaining / Abstain | Prevents 'I don't know' from being counted as hallucination. |
| Core metrics | Accuracy, hallucination rate, answer coverage, selective accuracy | Measures usefulness and reliability together. |
| Result format | CSV/JSONL with fixed IDs and experiment metadata | Allows independent outputs to be merged in Week 5–6. |
| Version control | Separate folder/branch ownership + frozen common utilities | Reduces merge conflicts and accidental dependencies. |

## 4. Common Definitions

- **Knowledge score K(q):** For repeated generations, the fraction of runs that produce a correct answer. Example: 4 correct answers out of 5 gives K(q)=0.8.
- **Hallucination:** An incorrect substantive answer. An explicit abstention such as 'I do not know' is tracked separately.
- **Answer coverage:** Fraction of questions for which the model attempts a substantive answer rather than abstaining.
- **Selective accuracy:** Accuracy among only the questions the model chooses to answer.
- **Confident hallucination:** An incorrect answer accompanied by a strong confidence signal—for example low entropy, high log-probability, high response consistency, or high predicted knowledge confidence.

## 5. Member-wise Workstreams

### P1 — Controlled Synthetic Knowledge Acquisition

**Research question:** How does the number of times a fact is observed during fine-tuning affect whether the model learns it, guesses it, or abstains?

P1 creates a synthetic factual world using fictional names, places, organizations and attributes. Because these facts are invented, the group can control their presence in the fine-tuning data. Facts are assigned to exposure groups such as 0×, 1×, 2×, 5×, 10× and 20×. The model is then evaluated on questions about those facts.

- Create a fact generator with templates covering several relation types (birth date, founder, capital, location, color, occupation, etc.).
- Create train/test paraphrases so evaluation does not simply repeat the training sentence verbatim.
- Balance exposure groups and relation types to avoid one group being easier for unrelated reasons.
- Run standard SFT/LoRA on the controlled corpus or use the shared baseline checkpoint from P6 when available; P1 must also be able to run a small independent version.
- Measure accuracy, hallucination, abstention and output confidence for every exposure level.
- Repeat important runs with multiple seeds if compute permits.
- Analyze whether learning is gradual or whether there is a sharp exposure threshold.

**Primary figures:** Exposure vs accuracy; exposure vs hallucination; exposure vs confidence; relation-type breakdown.

**Fallback result:** Even if no new method is used, this track independently produces a controlled study of factual exposure and hallucination.

### P2 — Real-World Knowledge Boundary and Error Taxonomy

**Research question:** On real factual questions, where does the base model behave as if it knows, partially knows, or does not know the answer?

P2 evaluates the unchanged model on a real factual QA benchmark. Each question is sampled several times. Repeated correctness provides an empirical knowledge score, while answer consistency reveals whether errors are unstable guesses or stable misconceptions.

- Select a short-answer factual QA dataset with reliable reference answers.
- Implement answer normalization and a transparent grading rule; manually audit a sample of automatic grades.
- Generate approximately five sampled responses per question plus an optional deterministic response.
- Group questions by empirical knowledge score (e.g., 0/5 through 5/5) rather than forcing only binary known/unknown labels.
- Separate wrong answers from explicit abstentions.
- Create an error taxonomy: inconsistent guessing, consistently wrong answer, partially correct/alias issue, unsupported confident answer, and refusal.
- Analyze whether question characteristics such as entity popularity or answer type correlate with knowledge score, if metadata is available.

**Primary figures:** Distribution of K(q); hallucination rate by K(q); examples of confidently wrong vs unstable questions; error taxonomy table.

### P3 — Hidden-State / XAI Analysis

**Research question:** Do the model's internal representations contain a measurable signal of whether a question is known, uncertain or unknown?

P3 studies internal hidden states without changing the model. A fixed labeled question set can be created independently on Day 1 so this work does not wait for P1 or P2. Later, shared labels can be substituted to connect the XAI analysis to the main experiments.

- Extract hidden states at selected transformer depths (for example 25%, 50%, 75% and final layer) rather than storing every activation initially.
- Choose and document a pooling strategy such as final prompt token or mean prompt representation.
- Train lightweight logistic/linear probes to predict correctness or high-vs-low knowledge score from frozen representations.
- Report AUROC/accuracy by layer to identify where knowledge-state information is strongest.
- Visualize representations using PCA and optionally UMAP, colored by knowledge score/exposure group.
- If time permits, compare question-state representations before and after standard SFT or the proposed P5 method.
- Avoid claiming causal interpretability; the probe shows information is decodable, not necessarily that the layer causes the behavior.

**Primary figures:** Layer vs probe AUROC; PCA/UMAP plots; representation distance vs empirical knowledge score.

### P4 — Confidence and Calibration

**Research question:** Which readily available confidence signal best predicts whether an LLM answer is actually correct?

P4 works primarily with output-level signals and can therefore operate independently of P3. The track compares several notions of confidence and identifies cases where the model appears certain but is wrong.

- Compute answer/token log-probability using a clearly defined aggregation method.
- Compute token-distribution entropy over generated positions or selected answer positions.
- Compute response consistency across repeated generations (exact/normalized answer agreement or semantic grouping if needed).
- Optionally elicit verbalized confidence and test whether self-reported confidence is calibrated.
- Evaluate each confidence signal as a predictor of correctness using AUROC and calibration metrics such as ECE/Brier score.
- Produce reliability diagrams and identify high-confidence error slices.
- Later compare P3's hidden-state probe confidence against these simpler output-level baselines.

**Primary figures:** Reliability diagrams; confidence-signal AUROC table; confident-hallucination examples; calibration by knowledge/exposure group.

### P5 — Evidence-Aware Fine-Tuning Intervention

**Research question:** Can the fine-tuning objective explicitly account for evidence strength so that rare facts do not become as overconfident as strongly supported facts?

P5 owns the experimental intervention. To avoid blocking on other members, Version 1 uses a locally generated synthetic corpus where exposure frequency is known by construction. The exact loss should remain modest and testable; the project should not depend on inventing a complex architecture.

- Implement ordinary LoRA/QLoRA SFT first as a sanity check.
- Define an evidence-strength target *s* based on exposure frequency (for example a normalized/log-scaled value).
- Add a small confidence/evidence prediction head or representation projection that predicts *s* or empirical reliability.
- Train with a combined objective: standard generation loss plus a weighted evidence/calibration loss.
- Test at least two or three lambda values so conclusions do not depend on one arbitrary weight.
- Compare a simple frequency target with a behavioral target (e.g., repeated correctness) only if time permits.
- Run ablations: SFT only; SFT + evidence head without backpropagating into the backbone if practical; full proposed objective.
- Document negative results. The key question is whether calibration/hallucination improves at comparable answer coverage—not whether raw accuracy alone increases.

**Primary figures:** Proposed method vs SFT across exposure levels; hallucination/coverage trade-off; lambda ablation; calibration comparison.

> **Important scope control:** Do not wait for P3 to discover a special layer. Start with a fixed selected layer/projection. Incorporating P3's best layer later is optional.

### P6 — Baselines, Abstention and Common Evaluation

**Research question:** Does the proposed method beat simple alternatives, or could the same improvement be obtained by ordinary SFT or teaching the model to say IDK?

P6 owns the evaluation backbone and baseline models. This track prevents misleading conclusions such as reporting low hallucination from a model that simply refuses every question.

- Implement common metrics and test them on dummy prediction files before shared experiments are ready.
- Evaluate the untouched base model.
- Train/evaluate standard SFT using the same PEFT configuration as the proposed method where possible.
- Train/evaluate an abstention baseline containing explicit unknown/IDK examples.
- Produce risk-coverage/selective-accuracy curves by thresholding a confidence signal.
- If feasible after core baselines are complete, implement a simplified GeoDe-inspired data-filtering baseline; this is optional, not critical-path.
- Maintain the final comparison table and ensure all methods are evaluated on the same held-out examples.

**Primary figures:** Base/SFT/IDK/proposed comparison; risk-coverage curves; selective accuracy; final benchmark table.

## 6. Explicit Dependency Design

| Track | Needs to Start | Optional Later Input | Must NOT Wait For |
|---|---|---|---|
| P1 | Base model + synthetic templates | P6 common evaluator | P2/P3/P5 |
| P2 | Base model + real QA dataset | P4 confidence utilities | P1/P3/P5 |
| P3 | Base model + self-contained labeled questions | P1 exposure labels; P2 K(q) labels | P1/P2 completion |
| P4 | Base model + any QA set | P2 real outputs; P1 synthetic outputs | P3 |
| P5 | Base model + locally generated synthetic data | P3 best layer; P4 best confidence signal | P1/P3 findings |
| P6 | Dummy prediction files + base model | P1/P2 final test sets; P5 predictions | Any member's completed experiment |

## 7. Data and Result Interfaces

To make integration easy, every experiment should export a row-level file with a minimum schema such as:

```text
question_id
source_dataset
exposure_level
question
reference_answer
model_answer
outcome            # correct / incorrect / abstain
knowledge_score
avg_logprob
entropy
consistency
model_checkpoint
method
seed
```

Hidden states should be stored separately (for example NumPy/PyTorch arrays) and linked through `question_id` rather than inserted into CSV files.

## 8. Six-Week Detailed Timeline

| Week | Group milestone | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|
| 1 | Freeze contract; every pipeline runs on tiny data | Generate 100-fact pilot | Load QA + grade 100 Qs | Extract states for 100 Qs | Compute 2 confidence signals | SFT + loss prototype on toy data | Metrics + base/SFT toy run |
| 2 | Every member has ≥1 quantitative result | Run 2–3 exposure groups | Repeated sampling + K(q) | Probe 3–4 layers | Calibration/AUROC pilot | First proposed-method result | Base + SFT + IDK results |
| 3 | Scale main experiments | Full exposure grid | Full real QA mapping | Main probe/XAI runs | Full confidence comparison | Main method sweep | Risk-coverage pipeline |
| 4 | Robustness/error analysis | Seeds + paraphrases | Error taxonomy | Before/after or exposure XAI | High-confidence error slices | Lambda/target ablations | Optional GeoDe-inspired baseline |
| 5 | Freeze experimental results | Final plots/tables | Final plots/tables | Final plots/tables | Final plots/tables | Final method table | Master baseline table |
| 6 | Integration, report, presentation | Write controlled-study section | Write real-QA section | Write XAI section | Write calibration section | Write method section | Consolidate metrics/comparison |

## 9. Integration Experiments (Week 5 Only)

Cross-member experiments should be treated as final integration rather than prerequisites. Choose only the highest-value combinations after independent results are stable.

- P3 applies its hidden-state probe to P1's exposure groups to test whether latent knowledge changes monotonically with exposure.
- P4 applies its best confidence measure to P1 and P2 outputs to compare synthetic and real-world calibration.
- P5 uses P4's best confidence target or P3's best layer only if doing so is a small change and Version 1 is already complete.
- P6 evaluates P5's final predictions using the exact same risk-coverage pipeline as all baselines.
- One joint figure can connect exposure → internal signal → confidence → behavior, but only after each component works independently.

## 10. Evaluation Matrix

| Method | Accuracy | Hallucination | Coverage | Calibration | XAI/Knowledge Signal |
|---|:---:|:---:|:---:|:---:|:---:|
| Base model | ✓ | ✓ | ✓ | ✓ | ✓ |
| Standard SFT | ✓ | ✓ | ✓ | ✓ | Optional |
| IDK/abstention SFT | ✓ | ✓ | ✓ | ✓ | Optional |
| GeoDe-style baseline | ✓ | ✓ | ✓ | ✓ | If implemented |
| Evidence-aware method | ✓ | ✓ | ✓ | ✓ | ✓ |

## 11. Risk Management and Fallbacks

| Risk | Impact | Fallback |
|---|---|---|
| New loss does not improve results | Main intervention is negative | Report controlled exposure + calibration + XAI findings; negative method result is still interpretable. |
| Compute is limited | Cannot run many checkpoints/seeds | Use 1–3B model, LoRA/QLoRA, smaller datasets, prioritize exposure groups and one main seed + targeted repeats. |
| Automatic QA grading is noisy | Knowledge scores become unreliable | Use short-answer data, normalization/aliases and manual audit of a stratified sample. |
| Hidden-state storage is too large | P3 pipeline becomes heavy | Store selected layers and pooled vectors only. |
| GeoDe reproduction takes too long | P6 gets blocked | Treat it as optional; base/SFT/IDK are mandatory baselines. |
| Synthetic facts leak semantic cues | Exposure result is confounded | Randomize names/attributes, balance relation types and use paraphrased evaluation questions. |
| Model says IDK too often | Low hallucination looks artificially good | Always report answer coverage and selective accuracy with hallucination. |

## 12. Suggested Repository and Ownership

```text
project/
├── common/                 # frozen shared utilities
│   ├── model.py
│   ├── generation.py
│   ├── grading.py
│   ├── metrics.py
│   └── config.yaml
├── p1_exposure/
├── p2_real_qa/
├── p3_xai/
├── p4_calibration/
├── p5_training_method/
├── p6_baselines/
├── data/
│   ├── synthetic/
│   └── real_qa/
├── results/
│   ├── raw/
│   └── final/
└── README.md
```

## 13. Final Story for the Report / Presentation

- **Part A — Why/when does the model hallucinate?** P1 provides controlled evidence and P2 provides real-world evidence.
- **Part B — Can we detect weak knowledge?** P3 examines hidden representations and P4 examines observable confidence/calibration.
- **Part C — Can training be improved?** P5 tests an evidence-aware intervention and P6 determines whether it beats simple baselines at comparable coverage.

**Desired final conclusion format:**

> "Increasing factual exposure changes [behavioral/internal findings]. Standard confidence signals [do/do not] reliably identify weak knowledge. Our evidence-aware intervention [improves/does not improve] hallucination and calibration relative to SFT/abstention baselines at comparable answer coverage."

## 14. Definition of Done

- Each member has one independently reproducible experiment and at least one final figure/table.
- All final methods use the same evaluation definitions and held-out examples where comparisons are claimed.
- At least one controlled synthetic exposure result and one real-world factual QA result are included.
- Hallucination is never reported without answer coverage/selective accuracy.
- The proposed training method is compared against standard SFT and explicit abstention training.
- XAI claims are phrased as decodability/association unless causal evidence is actually established.
- The final report clearly separates confirmed findings from hypotheses and future work.

## 15. Week-by-Week Individual and Group Deliverables

The following milestones are intended as hard weekly completion targets. From Week 2 onward, every member should finish each week with at least one quantitative result, plot, table, or validated experiment—not only literature review or setup.

### Week 1 — Freeze Design and Build Independent Pipelines

**Group deliverable:** Finalize the research question and scope; select one base 1–3B LLM; freeze prompt format, grading definitions and core metrics; create the Git/repository structure and common result schema; complete one tiny end-to-end run from question → generation → grading → metrics → saved output.

| Owner | Must be finished by end of week |
|---|---|
| P1 — Controlled Exposure | Generate ~100–200 pilot synthetic facts; implement exposure groups and train/test paraphrases; complete a tiny fine-tuning/evaluation sanity run. |
| P2 — Real QA | Select and load the factual QA dataset; implement normalization/grading; manually audit roughly 50–100 graded examples. |
| P3 — XAI | Extract hidden states from selected layers for ~100 questions; verify storage/pooling; run a first PCA or simple linear probe. |
| P4 — Calibration | Implement log-probability, entropy and repeated-generation consistency; validate all signals on ~100 questions. |
| P5 — New Method | Get ordinary LoRA/SFT working; define Version 1 of the evidence-aware objective; verify the extra loss backpropagates and training is stable. |
| P6 — Baselines | Implement common metrics; run the untouched base model; build standard SFT baseline and risk/coverage evaluation code. |

**End-of-week milestone:** The full experimental plumbing works and every member can proceed without waiting for another member.

### Week 2 — Produce First Quantitative Results

**Group deliverable:** Hold a short results review where every member presents hypothesis → experiment → first result → problem → next step. Freeze the base model, primary datasets, evaluation definitions and main generation settings after this review unless a genuine bug is found.

| Owner | Must be finished by end of week |
|---|---|
| P1 | Run at least three exposure levels (e.g. 0×, 2×, 10×) and produce the first exposure → accuracy/hallucination plot. |
| P2 | Run repeated generations on a meaningful QA subset; calculate empirical K(q); produce the first known/uncertain/unknown distribution. |
| P3 | Extract 3–4 representative layers; train simple probes; produce the first layer → AUROC/accuracy result. |
| P4 | Compare entropy, log-probability and consistency; produce a first correctness-prediction/calibration comparison. |
| P5 | Train ordinary SFT and the proposed method on a small controlled dataset; produce the first side-by-side result even if the proposed method is worse. |
| P6 | Complete Base + Standard SFT + IDK/abstention baselines on a small common dataset; generate the first comparison table. |

**End-of-week milestone:** All six tracks have at least one real quantitative result and the group can judge whether the project is experimentally feasible.

### Week 3 — Run the Main Experiments

**Group deliverable:** Finalize the exact experiments intended for the report; create a shared experiment tracker; centralize stable outputs and configs. Target 60–70% completion of all main experiments.

| Owner | Must be finished by end of week |
|---|---|
| P1 | Complete the main exposure grid: 0×, 1×, 2×, 5×, 10×, 20×. Measure accuracy, hallucination and confidence at each level; begin seed repeats if feasible. |
| P2 | Run the full selected real-QA subset with repeated generations; calculate continuous K(q); identify confidently wrong questions and begin the error taxonomy. |
| P3 | Complete the main layer-wise extraction/probing runs; produce PCA/UMAP; test whether representation patterns correlate with knowledge score. |
| P4 | Complete the main confidence comparison using AUROC and calibration metrics such as ECE/Brier where appropriate; generate reliability diagrams and high-confidence error slices. |
| P5 | Run the main proposed-method experiments; compare with SFT; begin a small λ sweep such as 0.1, 0.5 and 1.0. |
| P6 | Complete mandatory Base, SFT and SFT+IDK baselines and risk–coverage/selective-accuracy curves. Start a GeoDe-style baseline only if core baselines are stable. |

**End-of-week milestone:** The group has enough main data that the likely project conclusions are becoming visible.

### Week 4 — Robustness, Ablations and Research-Question Testing

**Group deliverable:** Review what the Week 3 results actually support. Decide which claims are strong enough for the final report and which need one additional experiment. Stop adding unrelated ideas.

| Owner | Must be finished by end of week |
|---|---|
| P1 | Repeat important exposure results across seeds and/or paraphrases; check whether the trend survives relation-type changes; analyze unexpected exposure thresholds. |
| P2 | Finish the real-world error taxonomy; analyze confidently wrong vs inconsistent questions; manually inspect representative cases and grading errors. |
| P3 | Test robustness of the strongest probe across layers/label definitions; if feasible compare representations before vs after SFT or across P1 exposure groups. |
| P4 | Analyze high-confidence hallucinations in detail; compare calibration across known/uncertain/unknown or exposure groups; verify the best confidence signal is stable. |
| P5 | Finish λ/target ablations; test whether gains come from the new objective rather than merely extra parameters; compare hallucination at similar answer coverage. |
| P6 | Finish risk–coverage comparisons; add the optional GeoDe-inspired baseline only if time allows; verify every model is evaluated with identical definitions. |

**End-of-week milestone:** Every major claim has supporting evidence or is explicitly downgraded to a hypothesis.

### Week 5 — Freeze Results and Integrate the Six Tracks

**Group deliverable:** Stop changing the core methodology. Select final checkpoints, tables and plots. Run only high-value cross-track integration experiments and begin writing the report in parallel.

| Owner | Must be finished by end of week |
|---|---|
| P1 | Freeze final exposure results and figures; provide P3/P4 with stable question IDs/exposure labels if needed for integration. |
| P2 | Freeze real-QA knowledge-boundary results, error examples and figures; provide stable K(q) labels for optional cross-track analysis. |
| P3 | Produce final XAI figures and probe table; optionally apply the best probe to P1/P2 data for one integrated result. |
| P4 | Produce final reliability diagrams and confidence table; optionally apply the best confidence signal to P1/P2 outputs. |
| P5 | Select the final proposed-method checkpoint; finish the essential ablation table; export predictions in the common format for P6. |
| P6 | Produce the master comparison table and final risk–coverage plots using Base/SFT/IDK/proposed/optional GeoDe-style results. |

**End-of-week milestone:** Experimental results are frozen; the project has a single coherent evidence chain from exposure → knowledge signal → behavior → intervention.

### Week 6 — Report, Presentation and Reproducibility

**Group deliverable:** Integrate all sections into one story, verify that every claimed comparison uses compatible data/metrics, rehearse the presentation, and prepare likely professor questions. No new major experiment should start unless it fixes a critical gap.

| Owner | Must be finished by end of week |
|---|---|
| P1 | Write the controlled-exposure methodology/results section; finalize captions and interpretation. |
| P2 | Write the real-world knowledge-boundary/error-analysis section and provide representative examples. |
| P3 | Write the hidden-state/XAI section, including limitations on interpreting probe results causally. |
| P4 | Write the confidence/calibration section and explain the chosen metrics clearly. |
| P5 | Write the proposed-method formulation, training setup, ablations, results and limitations. |
| P6 | Write the baseline/evaluation section; consolidate the final quantitative comparison and reproducibility details. |

**End-of-week milestone:** Final report, presentation, code/results package and a rehearsed 2–3 minute project summary are complete.

## 16. Weekly Group Review Format

To prevent hidden blockers, hold one fixed weekly review. Each member should use the same five-part update and keep it short:

1. What was the hypothesis/question for this week?
2. What experiment was actually completed?
3. What quantitative result/plot/table was produced?
4. What failed or remains uncertain?
5. What exactly will be finished before the next review?

> **Recommended rule:** If a member is blocked by another track for more than one day, they should switch to their independent fallback dataset/pipeline rather than wait. Cross-member inputs should improve an already working experiment, not be required to start it.
