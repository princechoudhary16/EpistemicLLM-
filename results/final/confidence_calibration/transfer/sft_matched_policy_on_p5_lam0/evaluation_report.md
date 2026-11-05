# Transfer: sft_matched policy on P5 lambda 0, final split

Model `Qwen/Qwen2.5-0.5B-Instruct` at `7ae55760`, checkpoint `p5_main_v0_lam0_seed0`; prompt `prompt_v0`, decoding `decoding_v1`, grader `p2_whole_word_v5`. Calibrators are fitted on train only and every choice is made on validation. Signals not available: `sum_logprob`, `min_logprob`, `first_token_entropy`, `aux_confidence_score`.

Frozen policy `cb392b57de50`, fitted on `sft_matched_v0` / `synthetic-main_v0`, applied unchanged as a transfer test: this model or dataset differs from the one it was fitted on.

## Inputs

| file | rows | sha256 | arrived graded with | outcomes changed by regrade |
|---|---:|---|---|---:|
| `results/raw/p5_main_v0_lam0_seed0.jsonl` | 7200 | `3a7f477f9973` | p2_whole_word_v4 | 0 |

## Split counts

Greedy responses, the target being judged.

| split | questions | correct | incorrect | abstain | invalid |
|---|---:|---:|---:|---:|---:|
| held-out test | 600 | 285 | 315 | 0 | 0 |

## Signal comparison on held-out test

Attempted greedy answers with a usable signal. Intervals are 95% cluster-bootstrap percentile intervals.
The difference column is AUROC minus `avg_logprob` on the same questions, resampled jointly.

| signal | N (correct/wrong) | AUROC [95% CI] | difference [95% CI] | AUPRC | Brier [95% CI] | base-rate Brier | ECE [95% CI] | ECE read raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **avg_logprob** | 600 (285/315) | 0.937 [0.910, 0.960] | reference | 0.935 | 0.104 [0.082, 0.127] | 0.250 | 0.062 [0.045, 0.102] | 0.282 |
| entropy | 600 (285/315) | 0.927 [0.894, 0.953] | -0.011 [-0.024, +0.000] | 0.929 | 0.113 [0.094, 0.136] | 0.250 | 0.115 [0.088, 0.148] | N/A |
| consistency | 600 (285/315) | 0.911 [0.882, 0.941] | -0.026 [-0.046, -0.006] | 0.865 | 0.105 [0.083, 0.126] | 0.250 | 0.055 [0.030, 0.087] | 0.155 |
| majority_frac | 600 (285/315) | 0.908 [0.875, 0.937] | -0.030 [-0.051, -0.009] | 0.862 | 0.102 [0.079, 0.126] | 0.250 | 0.026 [0.015, 0.059] | 0.204 |

The selected signal (validation Brier) is in bold. Base-rate Brier predicts the training accuracy (49.3%) for every answer, so a useful signal must beat it. ECE read raw treats the signal itself as a probability (`exp` of a log-probability, or the agreement share); N/A means it has no such reading.

## Abstention policies on held-out test

Attempted answers below the threshold become abstentions. Rates use all 600 valid questions, abstentions included; selective accuracy uses the answers kept. Thresholds come from validation.

Answering everything: accuracy 47.5%, hallucination 52.5%, abstention 0.0%, coverage 100.0%, selective accuracy 47.5%.

| signal | target coverage | threshold P(correct) | coverage | accuracy | selective accuracy | hallucination | confident errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| avg_logprob | 90% | 0.006 | 90.7% | 47.5% | 52.4% | 43.2% | 259 |
| avg_logprob | 80% | 0.034 | 80.3% | 46.8% | 58.3% | 33.5% | 201 |
| avg_logprob | 70% | 0.141 | 66.2% | 45.7% | 69.0% | 20.5% | 123 |
| avg_logprob | 60% | 0.268 | 59.5% | 44.2% | 74.2% | 15.3% | 92 |
| avg_logprob | 50% | 0.512 | 51.3% | 42.5% | 82.8% | 8.8% | 53 |

Tied scores can make a threshold keep more answers than its target; the coverage column shows what was actually kept.

## Signal stability across groups

Descriptive. AUROC is N/A where a group has only one outcome. Sampled K does not include the greedy answer being judged, but it shares its samples with the consistency signals.

| signal | grouping | group | questions | correct/wrong | AUROC [95% CI] | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| avg_logprob | exposure level | 0 | 100 | 0/100 | N/A [N/A, N/A] | 0.154 | 0.266 |
| avg_logprob | exposure level | 1 | 100 | 7/93 | 0.782 [0.556, 0.959] | 0.099 | 0.122 |
| avg_logprob | exposure level | 2 | 100 | 25/75 | 0.854 [0.712, 0.964] | 0.136 | 0.106 |
| avg_logprob | exposure level | 5 | 100 | 68/32 | 0.914 [0.827, 0.976] | 0.141 | 0.127 |
| avg_logprob | exposure level | 10 | 100 | 89/11 | 0.891 [0.744, 1.000] | 0.069 | 0.100 |
| avg_logprob | exposure level | 20 | 100 | 96/4 | 0.958 [0.865, 1.000] | 0.027 | 0.085 |
| avg_logprob | sample knowledge score | 0<K<0.8 | 56 | 27/29 | 0.667 [0.498, 0.831] | 0.306 | 0.278 |
| avg_logprob | sample knowledge score | K=0 | 288 | 3/285 | 0.642 [0.517, 0.751] | 0.127 | 0.204 |
| avg_logprob | sample knowledge score | K>=0.8 | 256 | 255/1 | 0.945 [0.911, 0.973] | 0.035 | 0.137 |

## High-confidence errors on held-out test

`avg_logprob` at the 80%-coverage threshold (P(correct) >= 0.034): 201 wrong out of 482 answers kept.

| pattern | meaning | count |
|---|---|---:|
| stable_wrong | K=0; the samples repeat the wrong answer | 42 |
| wavering_wrong | K=0; the samples disagree with it | 138 |
| knows_sometimes | K>0; some samples are right | 21 |

Flagged by the shared grader for review: 4. Every error is listed in `confident_errors.csv`.

By exposure level: 0x: 69, 1x: 52, 2x: 45, 5x: 22, 10x: 9, 20x: 4.

Most confident errors:

| P(correct) | pattern | question | answer | reference |
|---:|---|---|---|---|
| 0.915 | stable_wrong | How did Vlaengle Raethan earn a living? | tapestry weaver | harpsichord maker |
| 0.909 | stable_wrong | Which line of work was Vlaengle Raethan in? | tapestry weaver | harpsichord maker |
| 0.906 | stable_wrong | Which line of work was Saensaekraes Briroum in? | tapestry weaver | cobbler |
| 0.902 | stable_wrong | How did Saensaekraes Briroum earn a living? | tapestry weaver | cobbler |
| 0.885 | stable_wrong | Which city does Glisinuk govern from? | Kenplunath | Kenpluneth |
| 0.883 | stable_wrong | Which line of work was Ninubis Krofaendris in? | tapestry weaver | fletcher |
| 0.877 | stable_wrong | Where is the main base of the Baehath Guild? | Haeglonvi | Haeglonbru |
| 0.877 | stable_wrong | What is the name of Glisinuk's capital? | Kenplunath | Kenpluneth |
| 0.870 | stable_wrong | How did Bradeglil Faeplaeglou earn a living? | tapestry weaver | wheelwright |
| 0.868 | stable_wrong | Which line of work was Zhihounkun Faenkrer in? | tapestry weaver | wheelwright |
| 0.866 | stable_wrong | Where is the main base of the Haeshuple Foundry? | Boukrindroul | Bouvenhos |
| 0.864 | stable_wrong | Which line of work was Zhethanguk Ledaenshou in? | tapestry weaver | fletcher |
| 0.862 | stable_wrong | How did Ninubis Krofaendris earn a living? | tapestry weaver | fletcher |
| 0.857 | stable_wrong | The Haeshuple Foundry operates out of which city? | Boukrindroul | Bouvenhos |
| 0.852 | stable_wrong | Which line of work was Kobunplur Mathugler in? | tapestry weaver | chandler |

![Reliability diagram](reliability_diagram.svg)

Reliability bins holding few answers are noisy; their counts are in the JSON results.

![Risk-coverage curves](risk_coverage.svg)
