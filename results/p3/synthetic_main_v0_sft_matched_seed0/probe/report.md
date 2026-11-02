# P3 Week 1 probe smoke results

**Pipeline verification only. Do not interpret these provisional-label scores as research findings.**

Training labels: {'not_correct': 260, 'correct': 189}; validation labels: {'not_correct': 80, 'correct': 70}.

Unlabeled/invalid responses excluded from probe fitting: 1.

| Representation | Accuracy | Balanced accuracy | AUROC | AUPRC | Status |
|---|---:|---:|---:|---:|---|
| majority_baseline | 0.533 | 0.500 | 0.500 | 0.467 | training prevalence |
| layer_6 | 0.520 | 0.517 | 0.603 | 0.558 | fitted |
| layer_12 | 0.580 | 0.578 | 0.640 | 0.666 | fitted |
| layer_18 | 0.707 | 0.702 | 0.769 | 0.777 | fitted |
| layer_24 | 0.627 | 0.622 | 0.626 | 0.576 | fitted |
| prompt_length_baseline | 0.487 | 0.487 | 0.519 | 0.496 | fitted |

No layer or regularization search was performed. AUROC/AUPRC are N/A if validation has only one class.

Features contain prompt information only. Standardization uses training rows only; question groups never cross the split.

Default nonmatches may include semantically correct answers. P2 auditing and substantially more positive examples are needed before knowledge-state claims.
