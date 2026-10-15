## triviaqa-n300-base-mps

**Greedy behaviour** (N excludes invalid outputs):

| N | correct | incorrect | abstain | invalid | accuracy | hallucination | coverage | selective acc. |
|---|---|---|---|---|---|---|---|---|
| 300 | 43 | 252 | 5 | 0 | 0.143 | 0.840 | 0.983 | 0.146 |

**Signal discrimination** (attempted greedy answers; label = correct; 95% question-bootstrap CI):

| signal | confident when | n (correct/incorrect) | AUROC [95% CI] | AUPRC |
|---|---|---|---|---|
| avg_logprob | higher | 295 (43/252) | 0.726 [0.643, 0.803] | 0.352 |
| sum_logprob | higher | 295 (43/252) | 0.748 [0.667, 0.828] | 0.367 |
| min_logprob | higher | 295 (43/252) | 0.754 [0.673, 0.829] | 0.364 |
| entropy | lower | 295 (43/252) | 0.708 [0.629, 0.784] | 0.303 |
| first_token_entropy | lower | 295 (43/252) | 0.745 [0.672, 0.819] | 0.322 |
| consistency | higher | 295 (43/252) | 0.731 [0.652, 0.809] | 0.297 |
| majority_frac | higher | 295 (43/252) | 0.722 [0.637, 0.806] | 0.295 |

**Diagnostics:**

- n_questions: 300
- n_sample_responses: 1500
- greedy_truncated: 0
- sample_truncated: 0
- retokenized: 0
- greedy_argmax_match_rate: 1.0000
- generation_logprob_max_abs_gap: 0.0000
- mean_knowledge_score: 0.1267
