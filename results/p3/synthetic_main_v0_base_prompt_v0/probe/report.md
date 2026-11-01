# P3 Week 1 probe smoke results

**Pipeline verification only. Do not interpret these provisional-label scores as research findings.**

Training labels: {'not_correct': 450, 'correct': 0}; validation labels: {'not_correct': 150, 'correct': 0}.

Unlabeled/invalid responses excluded from probe fitting: 0.

| Representation | Accuracy | Balanced accuracy | AUROC | AUPRC | Status |
|---|---:|---:|---:|---:|---|
| majority_baseline | 1.000 | N/A | N/A | N/A | training prevalence |
| layer_6 | N/A | N/A | N/A | N/A | skipped: training labels contain fewer than two classes |
| layer_12 | N/A | N/A | N/A | N/A | skipped: training labels contain fewer than two classes |
| layer_18 | N/A | N/A | N/A | N/A | skipped: training labels contain fewer than two classes |
| layer_24 | N/A | N/A | N/A | N/A | skipped: training labels contain fewer than two classes |
| prompt_length_baseline | N/A | N/A | N/A | N/A | skipped: training labels contain fewer than two classes |

No layer or regularization search was performed. AUROC/AUPRC are N/A if validation has only one class.

Features contain prompt information only. Standardization uses training rows only; question groups never cross the split.

Default nonmatches may include semantically correct answers. P2 auditing and substantially more positive examples are needed before knowledge-state claims.
