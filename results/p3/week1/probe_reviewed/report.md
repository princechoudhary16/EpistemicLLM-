# P3 Week 1 probe smoke results

**Pipeline verification only. Do not interpret these provisional-label scores as research findings.**

Training labels: {'not_correct': 72, 'correct': 2}; validation labels: {'not_correct': 25, 'correct': 0}.

Unlabeled/invalid responses excluded from probe fitting: 1.

| Representation | Accuracy | Balanced accuracy | AUROC | AUPRC | Status |
|---|---:|---:|---:|---:|---|
| majority_baseline | 1.000 | N/A | N/A | N/A | training prevalence |
| layer_6 | 1.000 | N/A | N/A | N/A | fitted |
| layer_12 | 1.000 | N/A | N/A | N/A | fitted |
| layer_18 | 0.920 | N/A | N/A | N/A | fitted |
| layer_24 | 0.960 | N/A | N/A | N/A | fitted |
| prompt_length_baseline | 0.400 | N/A | N/A | N/A | fitted |

No layer or regularization search was performed. AUROC/AUPRC are N/A if validation has only one class.

Features contain prompt information only. Standardization uses training rows only; question groups never cross the split.

Default nonmatches may include semantically correct answers. P2 auditing and substantially more positive examples are needed before knowledge-state claims.
