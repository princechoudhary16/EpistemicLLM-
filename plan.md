# EpistemicLLM: team execution plan

## 1. Scope and starting decisions

**Team:** six people, identified as P1–P6 until names are assigned.  
**Schedule:** six core weeks, plus two optional extension weeks. Week 1 starts when the team begins execution; no calendar deadline was supplied.  
**Initial model:** `Qwen/Qwen2.5-0.5B-Instruct`.  
**Initial real-world dataset:** SimpleQA.  
**Controlled experiment:** fictional facts with deliberately varied training exposure.

The six-week document supplies the core schedule; the proposal supplies the knowledge-continuum hypothesis. The team's Qwen 0.5B and SimpleQA decision supersedes the documents' suggested 1–3B starting model. All quantities and decision thresholds below are proposed planning defaults, not measured results or requirements from the source documents.

**Research question:** How do factual exposure and empirical knowledge relate to accuracy, hallucination, confidence, and hidden representations—and does training on a continuous knowledge/evidence target improve reliability at comparable answer coverage?

The work has three connected parts:

1. **Measure knowledge:** controlled synthetic exposure (P1) and real factual QA (P2).
2. **Detect uncertainty:** hidden-state probes (P3) and output confidence/calibration (P4).
3. **Intervene and compare:** evidence/knowledge-continuum fine-tuning (P5), evaluated against standard and abstention baselines (P6).

A negative intervention result is acceptable. Success means a reproducible, fairly evaluated answer to the research question, not a guaranteed performance improvement.

## 2. People, ownership, and handoffs

| Person | Primary responsibility | Owns | Required final output | Main handoff |
|---|---|---|---|---|
| P1 | Controlled factual exposure | Synthetic generator, exposure manifests, paraphrases, exposure analysis | Validated synthetic dataset and exposure-versus-behavior figures | Data/manifests to P5/P6; exposure labels to P3/P4 |
| P2 | SimpleQA knowledge mapping | Dataset snapshot, question splits, grading, repeated-response labels, error taxonomy | Knowledge-score distribution, grading audit, real-QA analysis | Graded responses and question labels to P3/P4/P6 |
| P3 | Hidden representations and probes | Activation extraction, pooled vectors, probes, representation plots | Held-out probe results by layer and representation analysis | Probe scores to P4/P6; optional findings to P5 |
| P4 | Confidence and calibration | Log-probability/entropy/consistency signals, calibration fits, reliability plots | Confidence comparison and high-confidence error analysis | Scores and validation-selected thresholds to P6 |
| P5 | Proposed training method | Auxiliary head/loss, method configurations, lambda sweep, ablations | Reproducible intervention, checkpoint, method/ablation table | Checkpoints and inference interface to P6 |
| P6 | Shared infrastructure and baselines | Model runner, metric implementation, standard SFT, IDK SFT, comparison table | Reproducible baselines and common final evaluation | Baseline checkpoints/predictions to all members |

P6 coordinates shared utilities but does not implement every track. P2 maintains shared grading; P4 maintains confidence extraction; P5 contributes the training extension. P1 may run a tiny independent SFT for debugging, but P6 owns the canonical standard-SFT checkpoint to avoid duplicate training.

Every person delivers code/configuration, raw results, at least one figure or table, a short interpretation, and a reproduction command. Every result must identify its dataset version, model/checkpoint, prompt, decoding configuration, and seed.

## 3. Days 1–3: freeze the shared contract

| Decision | Starting default | Owner(s) | Completion evidence |
|---|---|---|---|
| Model loading | Exact Qwen model above, pinned revision and tokenizer; use its chat template | P6 | One batch generates reproducibly; environment versions recorded |
| Compute | Measure available accelerator memory, throughput, training step time, and activation storage | P6 + P5 | Written run budget and queue; no assumed hardware capacity |
| Prompt | Short factual answer; explicitly allow “I don't know”; no retrieved context in primary real-QA runs | P2 + P6 | Versioned prompt used by every compared method |
| Decoding | One greedy response plus five sampled responses/question; sampled temperature 0.7, top-p 0.9, max 64 new tokens | P2 + P6 | Saved settings, sample seeds, and truncation flag; tune only during pilot |
| Synthetic exposure | Actual presentations per fact: 0, 1, 2, 5, 10, 20 | P1 + P5 + P6 | Manifest and training-loader exposure audit |
| Split rules | Separate development and final evaluation; group paraphrases/duplicates by fact or question family | P1 + P2 | Immutable ID manifests and overlap check |
| Outcomes | Correct, incorrect substantive answer, abstain; separately flag invalid/empty/truncated generations | P2 + P6 | Grading examples and handling rules |
| Training parity | Same initial model, LoRA configuration, data order, and exposure budget where methods are comparable | P5 + P6 | Shared baseline configuration; differences explicitly documented |
| Output contract | One row per generated response; separate question summaries and hidden-state arrays | All; P6 coordinates | Every track reads/writes a tiny compatible example |

Start with ordinary LoRA if memory permits. Use quantization only if needed and supported by the actual hardware; record it and keep compared runs consistent. Freeze learning rate, adapter rank/targets, precision, batch size, accumulation, and optimizer after toy validation. Hardware-dependent choices should follow measurements rather than guessed settings.

## 4. Early decision: is Qwen overwhelmed by SimpleQA?

**Owner:** P2, supported by P3/P4 for label usefulness and P6 for runtime.  
**Deadline:** initial decision at the end of Week 1; any replacement pilot completed by the middle of Week 2.

### Pilot procedure

1. Select 200 SimpleQA questions with a fixed random seed before inspecting model answers. These are development questions, excluded from the final held-out evaluation.
2. Produce one greedy and five sampled responses per question: 1,200 responses total.
3. Grade responses, manually auditing at least 100 responses spread across predicted outcomes, with additional review of ambiguous grading cases. Report audit strata and disagreements.
4. Report greedy accuracy, sampled knowledge scores, hallucination, abstention, coverage, response failures, latency, and projected full-run cost.
5. Count questions in `K=0`, `0<K<1`, and `K=1`; also count correct/incorrect examples available for downstream probes and calibration.
6. Check prompt formatting, truncation, and grader errors before attributing failure to model capacity.

### Proposed warning thresholds

Trigger a team review if **any** of these hold:

- At least 90% of pilot questions have `K=0`.
- Fewer than 20 questions have intermediate scores (`0<K<1`).
- Fewer than 20 questions have high scores (`K>=0.8`).
- Runtime/memory projections exceed the team's measured compute budget.

These are practical warnings about usable label diversity, not statistical proof that the model is unsuitable. A hard dataset can still provide a useful failure analysis. P3/P4 must also check whether their planned train/validation/test sets contain enough examples of both outcomes; otherwise they reduce claims or use the synthetic study.

### Decision branches

| Pilot result | Action | What must be preserved |
|---|---|---|
| Enough behavioral variation and feasible compute | Keep Qwen 0.5B + SimpleQA | Freeze configuration and split IDs |
| SimpleQA mostly produces failures; larger model is feasible | Pilot one larger Qwen instruction checkpoint selected for available memory | Use the same pilot questions, prompt, grader, and sampling settings |
| Larger model is not feasible or still lacks useful variation | Keep 0.5B; pilot an easier factual short-answer dataset with reliable references | Select data before seeing answers; document provenance, grading, and new splits |
| Real-QA labels remain nearly constant | Keep SimpleQA as a stress test; make synthetic data the main continuum analysis | Report the lack of real-QA label diversity; do not manufacture balanced benchmark results |

Change one factor at a time, run at most one larger-model pilot and one replacement-dataset pilot, and record the decision in `reports/model_dataset_decision.md`. Do not select a test subset just because the model gets it right. If the primary model changes, regenerate model-dependent knowledge labels, hidden states, and baseline results. Earlier outputs remain separately labeled pilot evidence.

## 5. Data design and experimental safeguards

### P1: synthetic experiment

- **Pilot:** approximately 120 facts, balanced across six exposure levels and several relation types.
- **Main target:** 600 facts, 100 per exposure level; reduce to 300 if the measured budget requires it.
- Use fictional entity names and randomized attributes with balanced answer distributions. Avoid clues in names and avoid contradictory facts.
- Produce distinct training, validation-question, and test-question templates. Keep evaluation templates out of training.
- Partition entities/facts into development and final-analysis groups before generating variants. Exposed facts in both groups may appear in the factual training corpus by design; their evaluation questions do not. This tests recall/generalization of taught facts, not entirely unseen-fact generalization.
- For probes and calibration, group all variants of a fact into the same partition. Keep separate training entities for explicit IDK examples.
- Keep 0× reference answers in evaluation-only storage; never pass them into training examples. A 0× group is an unseen control, not a source of answer-supervised targets.
- Count actual training presentations. Repeating an exposure-expanded corpus for three epochs triples exposure and must not still be labeled 1×/2×/5×. Prefer one pass over an explicitly expanded, shuffled corpus.
- Standard SFT and the proposed method use the same fact presentations. Additional IDK examples and auxiliary-loss opportunities are documented separately.
- Audit accidental exact/alias matches on the untouched model. Fictional generation reduces contamination risk but does not prove that guessing is impossible.

### P2: real-QA experiment

- SimpleQA remains evaluation-only for the core six-week plan: do not answer-fine-tune on its development or final questions.
- Proposed allocation: 200 pilot questions, then 1,000 separate main questions split into 500 probe/calibration training, 200 validation, and 300 locked test questions. If resources permit, enlarge the final test before inspecting its outputs.
- These are project-defined splits, not claims about official dataset splits. Save IDs and check duplicates/related variants across partitions.
- Use references and conservative normalization/aliases to accept clear matches. Review ambiguous answers; substring matching alone is insufficient.
- Read and document the official grading approach. If using a local heuristic/manual grader, label results as a project evaluation rather than an exact reproduction of official benchmark scores. If using an external judge, record its model, prompt, version, and cost, and audit its decisions.
- Separate explicit abstentions from incorrect substantive answers; a tentative wrong claim still counts as incorrect under the project's rule. Freeze treatment of mixed answers before final evaluation.
- Preserve all five sampled correctness labels and `K(q)`; 0/5 does not prove the model has no underlying knowledge.
- Use only training/validation labels to fit probes, calibrators, and thresholds. Final test labels become available after analysis choices are frozen.

## 6. Common metrics and interpretation

Let `N` be the number of valid evaluated responses and `C`, `W`, and `A` their correct, incorrect non-abstaining, and abstaining counts; `N=C+W+A`. Report invalid generation counts separately and rerun technical failures under the same configuration before final comparisons. Do not silently discard model-produced empty or truncated outputs; use the frozen grading rule and disclose their frequency.

| Metric | Definition | Reporting rule |
|---|---|---|
| Accuracy | `C/N` | Abstentions are not correct |
| Hallucination rate | `W/N` | Operationally an incorrect substantive answer, not proof about its internal cause |
| Abstention rate | `A/N` | Must accompany accuracy/hallucination |
| Answer coverage | `(C+W)/N` | Prevents refusal-only improvements from looking useful |
| Selective accuracy | `C/(C+W)` | Undefined when no answers are attempted; report N/A |
| Selective risk | `W/(C+W)` | Compare methods at matched achievable coverage |
| Knowledge score | Correct sampled responses / 5, per question | Exclude the greedy response from this score |
| Correctness discrimination | AUROC, with class counts | N/A if only one class is present; add AUPRC for severe imbalance |
| Calibration | Brier score, ECE, reliability diagram | Use correctness probabilities, not raw entropy/log-probability |
| Confident hallucination | Incorrect answer above a validation-selected confidence threshold | Freeze threshold; report count, rate, and associated coverage |

P4 defines mean generated-answer token log-probability, mean predictive entropy, and normalized-answer agreement explicitly. Compute token statistics from the model's unmodified logits, distinguishing them from temperature/top-p sampling probabilities. Exclude prompt and padding tokens and document treatment of end tokens. Never use reference-answer likelihood as deployable confidence.

Fit any mapping from raw scores to correctness probabilities on the calibration-training split, choose settings on validation, and evaluate once on the locked test. Report metrics for all outputs and, where relevant, for attempted answers, with denominators clear.

Use greedy responses for the primary per-method behavior table; sampled runs characterize knowledge and consistency. Report uncertainty using question-level bootstrap intervals (fact-level for synthetic data), so repeated responses/paraphrases are not treated as independent observations. Fix the resampling procedure before final analysis.

Avoid circular findings: a relationship between sampled correctness and `K` computed from those same samples is descriptive by construction. To test prediction, predict a separate greedy response or an independent generation batch. Likewise, compare consistency-based confidence against an independently generated target response.

## 7. Proposed method and fair baselines

P5 starts with a small scalar prediction head on the final prompt-token hidden state, with a sigmoid output. This is a fixed initial choice; P5 does not wait for P3 to identify a layer.

**Core evidence-aware version:** for exposed synthetic training facts, use `s(e)=log(1+e)/log(21)` for exposure `e` in `{1,2,5,10,20}`. Train with:

`L = L_answer + lambda * L_aux`, where `L_aux = mean((predicted_s - s(e))^2)`.

Use answer-only language-model loss, excluding prompt/padding tokens. Ensure the auxiliary loss reaches trainable adapters; verify gradients and save the head with the adapter. Train each method from the same base checkpoint. Zero-exposure facts receive no reference-answer training. Because high-exposure facts appear more frequently, state whether the auxiliary objective is presentation-weighted or reweighted per fact; use presentation weighting for the initial implementation.

**Interpretation:** frequency is an evidence proxy, not a correctness probability. Calibrate the head before claiming calibrated confidence. This core version tests exposure-aware training, not the full empirical-knowledge hypothesis.

**Knowledge-continuum version:** after the core works, obtain repeated-response `K(q)` labels from a frozen standard-SFT teacher on synthetic training facts only, then train a separate version with `K(q)` replacing `s(e)`. Record the teacher checkpoint and label-generation budget. Keep the teacher fixed; its labels are not assumed to equal the evolving student's knowledge. This is the preferred Week 4 extension if feasible and directly tests the proposal's behavioral target. If omitted, clearly narrow the final claim to evidence-aware training.

Required comparisons:

| Method | Owner | Purpose |
|---|---|---|
| Untouched base | P6 | Initial behavior |
| Standard LoRA SFT | P6 | Effect of teaching the factual corpus |
| IDK/abstention SFT | P6 | Effect of explicit unknown-answer training, using separate training-only unknown entities |
| Evidence-aware SFT | P5 | Added auxiliary objective on identical factual training data |
| Head fitted on frozen standard-SFT representations | P5 + P4 | Tests whether post-hoc prediction explains benefits without changing representations |

Use `lambda=0` as the shared standard-SFT reference and initially try `0.1` and `1.0`; inspect loss scales on development data before freezing these values. Select the final lambda on validation only. Record training tokens, steps, exposure counts, and wall time; IDK SFT has extra examples, so disclose the difference and use a matched-budget sensitivity check if practical.

Evaluate both raw generated answers and confidence-thresholded answers. The auxiliary head alone does not teach the decoder to say IDK. If a head-based threshold creates abstention at inference, explicitly call that an inference policy and compare it against the same policy applied to baseline confidence scores. Never credit the training loss for a benefit caused solely by extra filtering.

## 8. Detailed weekly assignments

### Week 1 — Working pipelines and model/dataset decision

**Group achievement:** everyone has a runnable tiny experiment; shared interfaces are frozen; the initial SimpleQA suitability report exists.

| Person | Tasks this week | End-of-week acceptance criteria |
|---|---|---|
| P1 | Build fictional-fact generator; create ~120 balanced pilot facts; generate non-identical train/evaluation wording; export exposure and split manifests | Generator reruns from a seed; all six groups represented; no 0× answers in training; duplicate/contradiction checks pass |
| P2 | Load/version SimpleQA; select 200 pilot questions; implement grading and outcome rules; run greedy + five samples; audit >=100 responses | 1,200 response records, audited labels, K distribution, and keep/switch recommendation |
| P3 | Extract pooled prompt states from ~100 development questions at four depths; build a tiny linear-probe pipeline using provisional labels | Arrays join to IDs; pooling/layer indices documented; end-to-end smoke result, explicitly not a substantive finding |
| P4 | Implement mean answer log-probability and predictive entropy; start normalized-answer consistency; verify score extraction on tiny examples | Two signals exported with definitions and ID alignment; no full-vocabulary logits stored unnecessarily |
| P5 | Attach auxiliary head; implement combined loss; run a tiny synthetic training job and gradient check | Checkpoint reload works; answer and auxiliary losses logged; adapter gradient from auxiliary objective confirmed |
| P6 | Set up environment/config/model runner; implement metrics on hand-calculated fixtures; run base and tiny standard SFT; measure resources | One reproducible command for generation/evaluation; metric edge cases checked; compute budget and shared schema committed |

**Dependencies:** P3/P4 can use development examples while P2 audits the main pilot. P5 uses a local toy corpus until P1's validated pilot arrives. Provisional labels must not enter final plots.

### Week 2 — First interpretable result from every person

**Group achievement:** finalize the model/dataset by midweek; obtain preliminary behavior, probe, confidence, and training comparisons.

| Person | Tasks this week | End-of-week acceptance criteria |
|---|---|---|
| P1 | Freeze main synthetic generator/manifests; inspect 0×/5×/20× pilot behavior with shared SFT; audit actual presentations | Versioned main dataset ready; preliminary exposure plot and explanation of confounds |
| P2 | Complete any replacement pilot; freeze real-QA split IDs; grade main training/validation questions; begin error taxonomy | Decision record complete; train/validation K labels handed to P3/P4; at least 30 reviewed error examples |
| P3 | Train regularized probes at four selected depths on approved training labels; validate regularization; compare majority and simple prompt-length baselines | Validation AUROC/class counts by layer, or explicit insufficient-class finding; no test tuning |
| P4 | Compare log-probability, entropy, consistency; fit preliminary probability mappings; define independent target response | Validation signal table and first reliability diagram; threshold-selection protocol documented |
| P5 | Run one evidence-aware pilot against matching standard SFT; inspect head output and loss balance; test save/reload | Preliminary comparison on validation data, including coverage; final sweep configuration prepared |
| P6 | Produce pilot base/SFT/IDK predictions; integrate common grading and confidence fields; reserve training slots | Three-baseline validation table; IDK training set documented; main run queue agreed |

**Decision:** if high-exposure synthetic facts are not learned, debug data, masking, optimizer, and training budget before interpreting the auxiliary method. Adjust development settings once and record any exposure changes honestly.

### Week 3 — Main experiments

**Group achievement:** main datasets and training configurations are fixed; primary runs are completed or queued with realistic completion times.

| Person | Tasks this week | End-of-week acceptance criteria |
|---|---|---|
| P1 | Analyze all six exposure levels for base and standard SFT; break down by relation and paraphrase; inspect chance successes | Complete exposure table with denominators and uncertainty; validated input for intervention comparison |
| P2 | Finish main train/validation grading and audit; analyze K buckets, stable wrong answers, and unstable guesses; lock final-test evaluation procedure | Versioned development label release, taxonomy definitions, and final-test manifest; ambiguous cases adjudicated |
| P3 | Run selected-layer probes on full development data; group splits by fact/question; fit PCA on training vectors | Reproducible layer comparison; trained probe artifacts; validation representations plotted without test-driven layer selection |
| P4 | Complete development confidence comparison; select calibration settings and thresholds; compare probe scores if ready | Frozen calibrators and thresholds, reliability plots, and independent-response discrimination results |
| P5 | Run lambda sweep from the same initialization/data schedule; evaluate on validation; select finalist | SFT reference plus two nonzero-lambda results; selected checkpoint and written selection rationale |
| P6 | Complete canonical SFT/IDK runs; implement risk–coverage evaluation; validate matched-question joins | Main baseline checkpoints and validation comparison; common evaluator accepts P5 outputs |

**Scope rule:** one full primary training seed is mandatory. Reserve resources for selected repeats before expanding to extra models, datasets, or optional methods.

### Week 4 — Robustness, ablations, and final protocol freeze

**Group achievement:** determine whether initial patterns survive key controls; lock every analysis choice before opening final test results.

| Person | Tasks this week | End-of-week acceptance criteria |
|---|---|---|
| P1 | Repeat exposure study with another assignment/training seed if feasible; test held-out paraphrases; audit relation imbalance | Robustness plot or documented budget limitation; no unsupported causal claim beyond the controlled setup |
| P2 | Expand stratified error audit; separate alias/grading issues from real errors; review consistent wrong answers and abstentions | Audited taxonomy table with counts and representative examples; grading rules frozen |
| P3 | Compare base vs SFT vs proposed representations using the same question sets and split rules; audit leakage and label imbalance | Before/after development comparison; probe decodability claims separated from causal claims |
| P4 | Analyze high-confidence wrong answers; evaluate signal stability by exposure; compare baseline and head confidence under equal policies | Validation risk–coverage curves and high-confidence error table with fixed definitions |
| P5 | Fit frozen-representation head control; implement K-target version if budget permits; repeat finalist and SFT with a second training seed if possible | Required control complete; ablation table; clear status of evidence-target versus behavioral-target experiments |
| P6 | Check comparison fairness, exposure budgets, IDK data, and missing runs; freeze evaluation scripts/configs | Signed-off run manifest, model selection, metric definitions, and final-test execution checklist |

**Optional only after required work:** a simplified boundary-filtering baseline. Do not call it a faithful GeoDe reproduction without reading and matching the original method; removing ambiguous examples alone is a limited comparison.

### Week 5 — Locked evaluation and integration

**Group achievement:** final test runs, tables, figures, and uncertainty estimates are complete and frozen. No hyperparameter changes based on final-test outcomes.

| Person | Tasks this week | End-of-week acceptance criteria |
|---|---|---|
| P1 | Produce final exposure figures for all required methods; supply fact/group labels for joint analyses | Accuracy, hallucination, coverage, and confidence versus exposure, with raw supporting tables |
| P2 | Run/grade locked real-QA test with P6; report K distribution and taxonomy; verify dataset/prompt versions | Final real-QA results and audit notes; stress-test and replacement-dataset results kept distinct |
| P3 | Apply frozen probes to locked test states; complete cross-method representation analysis and synthetic exposure integration | Final per-layer/selected-probe table and PCA figure; class counts and uncertainty reported |
| P4 | Apply frozen calibrators/thresholds; report confidence comparison, reliability, and confident hallucinations | Final calibration/discrimination tables; matched-coverage analysis passed to P6 |
| P5 | Deliver finalist/ablation artifacts; explain gains or failures and whether filtering drives results | Method table, loss/config documentation, checkpoint/head loading instructions, limitations |
| P6 | Run all required methods through the identical evaluator; bootstrap comparisons; resolve missing/mismatched IDs | Master table and risk–coverage plot; reproduction manifest links every number to a run |

Limit integrations to shared-data analyses: P3 on P1 exposure labels, P4 on P1/P2 predictions, and P6 on P5 checkpoints. Do not introduce a new training objective in Week 5. If a genuine code/grading defect appears, log it and rerun all affected methods consistently.

### Week 6 — Report, reproducibility, and presentation

**Group achievement:** one coherent submission with a tested reproduction path and clearly bounded conclusions.

| Person | Tasks this week | End-of-week acceptance criteria |
|---|---|---|
| P1 | Write synthetic data and controlled-study sections; describe exposure accounting; review P5's claims against data | Final section, figure captions, dataset card, and clean generator command |
| P2 | Write real-QA setup/results/error analysis; document initial pairing and any fallback decision | Final section, dataset/grader description, representative examples, and audit artifact |
| P3 | Write representation methods/results; document pooling, splits, and probe controls; review P4 | Final section and XAI figures with association/decodability language |
| P4 | Write confidence/calibration sections; explain denominators and policy effects; review P3 | Final section, calibration figures, metric interpretation, and threshold artifacts |
| P5 | Write objective/implementation/ablation sections; distinguish tested method from proposal extensions; review P1 | Final method section, limitations, saved configurations, and runnable method example |
| P6 | Assemble master results/reproducibility appendix; coordinate clean smoke reproduction, README commands, and slide integration | Submission artifact, verified smoke workflow, consistent tables/captions, final presentation |

All six jointly write the introduction/conclusion, review the full report, and rehearse. Each person should explain their own experiment, its strongest evidence, and one limitation. Reserve the final two working days for consistency fixes and rehearsal rather than new experiments.

### Optional Weeks 7–8 — If the two-month schedule is available

| Person | Week 7: strengthen evidence | Week 8: finish extended submission |
|---|---|---|
| P1 | Repeat synthetic-world assignment and training seeds | Update robustness analysis and dataset release |
| P2 | Expand held-out real-QA sample or add a separately reported second dataset | Finalize dataset comparison and expanded audit |
| P3 | Test probe transfer across worlds/checkpoints with clearly separated training data | Update generalization claims and representation section |
| P4 | Test calibration transfer and threshold stability across settings | Update reliability/coverage comparison |
| P5 | Prioritize empirical-K targets if absent; otherwise additional controlled seeds | Finalize extended method/ablation results |
| P6 | Run matching baselines for extensions; optional justified filtering baseline | Rebuild master results, reproduce final artifact, coordinate final review |

These weeks improve robustness; they are not required to rescue an unfinished six-week core. New experiments use fresh held-out data if previous test results informed their design.

## 9. Result interfaces and repository layout

Proposed structure (to be implemented by the team; this plan does not create the experiment code):

```text
common/                 # P6 coordinates; P2 grading, P4 confidence
configs/                # pinned model, decoding, training, split configurations
p1_exposure/
p2_real_qa/
p3_xai/
p4_calibration/
p5_training_method/
p6_baselines/
data/manifests/         # versioned IDs, exposure counts, split membership
results/raw/            # per-generation predictions; large files stored externally
results/features/       # hidden vectors and ID/layer manifests
results/final/          # frozen tables and figures
reports/                # decisions, weekly notes, report sections
```

Minimum **response-level** JSONL fields:

```text
run_id, question_id, fact_id, source_dataset, dataset_version, split
question, reference_answer, exposure_level
model_id, model_revision, checkpoint_id, method, training_seed
prompt_version, decoding_config_id, generation_seed, sample_index, is_greedy
model_answer, outcome, grading_version, review_status, truncated
avg_logprob, entropy, confidence_score, latency_seconds
```

Use null for non-applicable values (for example, real-QA exposure and fact ID). Save question-level `knowledge_score`, `n_samples`, and `consistency` in a separate summary keyed by run/checkpoint/question and sampling configuration. Store hidden-state arrays separately with question IDs, layer indices, pooling strategy, and dtype. Keep raw text and grading revisions so disputed results can be reconstructed.

Do not commit model weights, full activation dumps, credentials, or caches. Record artifact locations/checksums and dependency versions. Each track's README documents a tiny smoke run and its full experiment command.

## 10. Coordination, budget, and fallback priorities

- **Start of week:** 20-minute meeting to commit to tasks, handoffs, and compute slots.
- **Midweek:** flag blocked dependencies and inspect one representative artifact per track.
- **End of week:** show results against the acceptance criteria above; update `reports/week_N.md` with completed work, metrics, blockers, and next actions.
- **Review pairs:** P1/P5, P2/P6, and P3/P4. Shared-schema changes require notifying all consumers and updating the schema version.
- **Compute priority:** pilot → main standard/IDK baselines → proposed sweep → locked evaluation → selected repeats → optional extensions.
- **Approximate base knowledge-mapping load:** 200 pilot + 1,000 main real questions at six generations each = 7,200 responses before any replacement pilots. A 600-fact synthetic evaluation at six generations adds 3,600 responses per checkpoint per question template. Extra templates, methods, and seeds multiply this budget; P6 estimates time from Week 1 measurements.
- Cache shared predictions and pooled states so members do not independently regenerate the same outputs. Do not reuse output-dependent features across different checkpoints.

| Risk | Owner | Response |
|---|---|---|
| Qwen/SimpleQA has almost no correct answers | P2 + P6 | Apply Week 1 gate; preserve stress-test result; use a documented model/data fallback |
| Fine-tuning fails to learn even high-exposure facts | P1 + P5 + P6 | Debug ordinary SFT first; do not blame the auxiliary objective before baseline learning works |
| Grading dominates the measured effect | P2 | Expand independent manual audit; adjudicate ambiguities; rerun affected comparisons |
| Probe/calibration labels contain only one outcome | P3 + P4 | Report N/A and class counts; use synthetic labels or the approved alternate dataset |
| Auxiliary method gives no benefit | P5 | Report negative results and controls; retain behavioral/calibration/XAI contributions |
| Low hallucination comes from refusal | P4 + P6 | Report coverage/selective accuracy and matched-coverage comparisons |
| Compute is insufficient | P6 | Reduce question/fact counts before inspecting outcomes; use fewer optional layers/ablations; preserve required baselines and test integrity |
| Shared output is delayed | Relevant producer | Consumers use local fixtures for engineering; final claims wait for validated shared artifacts |

## 11. Final completion checklist

- [ ] Starting Qwen–SimpleQA pilot and any fallback decision are documented.
- [ ] Every person has one independently reproducible experiment and at least one final figure/table.
- [ ] Controlled exposure and real-world factual QA are both represented.
- [ ] Base, standard SFT, IDK SFT, and proposed method are compared on identical questions within each reported experiment.
- [ ] Proposed method includes a frozen-representation head control and a documented lambda comparison.
- [ ] Evidence-frequency training and empirical-knowledge training are distinguished; omitted extensions are acknowledged.
- [ ] Test questions/reference answers did not leak into tuning, probes, calibration, or unknown-fact training.
- [ ] Accuracy, hallucination, abstention, coverage, and selective accuracy are reported together.
- [ ] Confidence policies, probability calibration, uncertainty intervals, and seed limitations are explicit.
- [ ] XAI results demonstrate prediction/association without unsupported causal interpretation.
- [ ] Final tables trace to versioned data, checkpoints, raw responses, grading, and evaluation commands.
- [ ] Report, slides, README reproduction steps, and a clean smoke run are complete.

## 12. Technical source anchors

The local project documents establish the research scope. These primary sources support implementation checks; the schedule, sample sizes, and decision rules above are team-planning proposals.

- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): checkpoint identity, chat-template usage, architecture, and loading guidance. Derive probe depths from the loaded configuration rather than hard-coding layer indices across model changes.
- [OpenAI simple-evals repository](https://github.com/openai/simple-evals): source to inspect and pin for SimpleQA evaluation and grading implementation. Document deviations when using the project's local grader or custom splits.
