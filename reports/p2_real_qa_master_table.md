# Real-QA master table — TriviaQA locked test

300 locked-test questions (`data/real_qa/triviaqa_v0/locked_test.jsonl`), run once after the freeze.
Qwen2.5-0.5B-Instruct (pinned revision), float32, prompt_v0, decoding_v1, grader `p2_whole_word_v5`; greedy answers.
Real-QA track, reported separately from the synthetic `main_v0` table (plan.md). sft_matched and P5 were trained on
synthetic facts, so their rows are a transfer test. On real data, rtuning_own and geode_own are the working IDK
baselines (P6's synthetic IDK baseline was degenerate: 100% abstention).

## Table 1 — master table (seed 0)

| Method | Trained on | Accuracy | Hallucination | Abstention | Coverage | Selective acc. | AURC* |
|---|---|---:|---:|---:|---:|---:|---:|
| base | — | 17.7% | 76.0% | 6.3% | 93.7% | 18.9% | 0.633 |
| sft_matched | synthetic | 3.0% | 97.0% | 0.0% | 100% | 3.0% | 0.940 |
| p5_lam0 | synthetic | 4.7% | 95.3% | 0.0% | 100% | 4.7% | 0.928 |
| p5_lam3 | synthetic | 3.7% | 96.3% | 0.0% | 100% | 3.7% | 0.923 |
| rtuning_own | TriviaQA train | 2.0% | 5.7% | 92.3% | 7.7% | 26.1% | 0.051 |
| random_own | TriviaQA train | 2.0% | 2.0% | 96.0% | 4.0% | 50.0% | 0.021 |
| **geode_own** | TriviaQA train | **6.3%** | 6.7% | 87.0% | **13.0%** | **48.7%** | 0.052 |
| kfilter_own | TriviaQA train | 0.0% | 0.3% | 99.7% | 0.3% | — | 0.003 |
| kthresh_own | TriviaQA train | 1.7% | 2.3% | 96.0% | 4.0% | 41.7% | 0.022 |
| ksoft_own | TriviaQA train | 3.0% | 3.7% | 93.3% | 6.7% | 45.0% | 0.027 |

\*AURC (log-prob confidence) covers only the coverage range each model reaches, so it is comparable only between
models with similar coverage; use Table 2 to compare across them. Accuracy differences vs base (question-cluster
bootstrap, 95% CI) are in `results/final/master_table_triviaqa_locked_test.json`; all methods are below base on
accuracy by design (abstention trades answers for fewer hallucinations); sft_matched / P5 lose 13–15 points.

## Table 2 — matched coverage (every method answers 40 of 300 questions = geode_own's coverage)

Means over seeds where available (geode_own and p5_lam3: 3 seeds; base and sft_matched: 1 run).

| Method | How it picks what to answer | Accuracy | Hallucination | Selective acc. |
|---|---|---:|---:|---:|
| base | log-prob filter | 6.0% | 7.3% | 45.0% |
| **geode_own** | **refuses by itself** | **6.0%** | **7.3%** | **45.0%** |
| sft_matched | log-prob filter | 1.0% | 12.3% | 7.5% |
| p5_lam3 | evidence head | 0.4% | 12.9% | 3.3% |
| p5_lam3 | log-prob filter | 1.1% | 12.2% | 8.3% |

## Table 3 — supplementary

3-seed means ± sd, F1_rel, pooled bootstrap intervals and the known / partial / unknown breakdown for all abstention
arms: `results/final/abstention/comparison_locked_test.json` (`python -m p2_abstention.compare --split locked_test`).

## Figures

- Risk–coverage, log-prob confidence: `results/final/risk_coverage_triviaqa_locked_test.png`
- Risk–coverage, P5 evidence head: `results/final/risk_coverage_triviaqa_locked_test_aux_head.png`

## Findings

1. Models fine-tuned on synthetic facts lose most real-world knowledge (accuracy 17.7% → 3–5%, hallucination 95–97%).
2. geode_own (GeoDe selection + the model's own answers as targets) cuts hallucination from 76% to 7% with answers
   2.4× as reliable, and at matched coverage it equals a log-prob filter on the base model while refusing by itself.
3. P5's evidence head is a weaker abstention signal than log-prob, also on real data.
4. K(q)-based training labels underperform the probe (see `p2_abstention/README.md`).
