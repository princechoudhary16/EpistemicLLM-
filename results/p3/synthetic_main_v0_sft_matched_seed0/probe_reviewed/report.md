# P3 Week 1 probe smoke results

**Pipeline verification only. Do not interpret these provisional-label scores as research findings.**

Training labels: {'not_correct': 250, 'correct': 199}; validation labels: {'not_correct': 79, 'correct': 71}.

Unlabeled/invalid responses excluded from probe fitting: 1.

| Representation | Accuracy | Balanced accuracy | AUROC | AUPRC | Status |
|---|---:|---:|---:|---:|---|
| majority_baseline | 0.527 | 0.500 | 0.500 | 0.473 | training prevalence |
| layer_6 | 0.547 | 0.546 | 0.589 | 0.583 | fitted |
| layer_12 | 0.547 | 0.545 | 0.606 | 0.649 | fitted |
| layer_18 | 0.747 | 0.745 | 0.763 | 0.756 | fitted |
| layer_24 | 0.633 | 0.630 | 0.639 | 0.602 | fitted |
| prompt_length_baseline | 0.493 | 0.494 | 0.526 | 0.504 | fitted |

No layer or regularization search was performed. AUROC/AUPRC are N/A if validation has only one class.

Features contain prompt information only. Standardization uses training rows only; question groups never cross the split.

Default nonmatches may include semantically correct answers. P2 auditing and substantially more positive examples are needed before knowledge-state claims.
