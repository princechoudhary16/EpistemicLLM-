# Probe vs log-prob, TriviaQA (base model, float32)

| Signal | Validation AUROC | Locked-test AUROC |
|---|---:|---:|
| log-prob | 0.716 | 0.671 |
| length | 0.353 | 0.355 |
| probe L6 | 0.719 | 0.590 |
| probe L12 | 0.713 | 0.609 |
| probe L18 | 0.738 | 0.619 |
| probe L24 | 0.725 | 0.688 |
| probe L18 + log-prob | 0.780 | 0.660 |
| log-prob + length | 0.766 | 0.711 |

Validation: 200 questions, 47 correct. Locked test: 300 questions, 53 correct.

| Difference (AUROC) | Validation [95% CI] | Locked test [95% CI] |
|---|---:|---:|
| probe L18 minus log-prob | +0.021 [-0.102, +0.137] | -0.052 [-0.146, +0.037] |
| probe L18 + log-prob minus log-prob | +0.064 [-0.040, +0.161] | -0.011 [-0.081, +0.054] |
| probe L18 + log-prob minus log-prob + length | +0.015 [-0.082, +0.113] | -0.051 [-0.119, +0.014] |
