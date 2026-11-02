# P1 exposure analysis

**Terms.** *Exposure*: how many times a fact appeared in the fine-tuning data (0 means never; the
fact is only asked about at test time). *Greedy response*: the single answer the model gives when it
always picks its most likely token. *Sampled responses*: 5 further answers per question drawn at
temperature 0.7. A response is *correct* if it contains the reference answer, *abstain* if it says it
does not know, otherwise *incorrect* (a hallucination: a substantive wrong answer).
*Accuracy* = correct / all; *hallucination* = incorrect / all; *abstention* = abstain / all;
*coverage* = (correct + incorrect) / all, the share of questions the model attempts;
*selective accuracy* = correct / (correct + incorrect), accuracy among attempted answers.
*K* of a question = fraction of its 5 sampled responses that are correct; *mean K* averages that over
questions. A fact's K pools all its sampled responses; facts are bucketed as *K=0* (never right),
*inconsistent* (0 < K < 0.8) or *K>=0.8* (reliably right).
*95% CI*: fact-level cluster bootstrap interval (see module docstring).

## Inputs

- `base`: `results/raw/p1_main_v0_base__p2_whole_word_v5.jsonl` (3600 rows, sha256 `33c6c32c726c`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `sft`: `results/raw/p1_main_v0_sft__p2_whole_word_v5.jsonl` (3600 rows, sha256 `3cbcacac989d`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `idk`: `results/raw/p1_main_v0_idk__p2_whole_word_v5.jsonl` (3600 rows, sha256 `af5e644a0305`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `sft_matched`: `results/raw/p1_main_v0_sft_matched__p2_whole_word_v5.jsonl` (3600 rows, sha256 `c75dc99905e0`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `p5_lam0`: `results/raw/p5_main_v0_lam0_seed0__p2_whole_word_v5.jsonl` (3600 rows, sha256 `7ae9300587fe`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `p5_lam3`: `results/raw/p5_main_v0_lam3_seed0__p2_whole_word_v5.jsonl` (3600 rows, sha256 `091ada020dc6`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])

Dataset `main_v0`, dev facts only (validation wording); bootstrap draws 4000, seed 0.

## base: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 67.0% [54.0%, 78.0%] | 33.0% | 67.0% | 0.0% | 0.00 [0.00, 0.01] | 49 / 1 / 0 |
| 1x | 100 (50) | 0.0% [0.0%, 0.0%] | 65.0% [53.0%, 77.0%] | 35.0% | 65.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 2x | 100 (50) | 0.0% [0.0%, 0.0%] | 70.0% [58.0%, 81.0%] | 30.0% | 70.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 5x | 100 (50) | 0.0% [0.0%, 0.0%] | 77.0% [67.0%, 87.0%] | 23.0% | 77.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 10x | 100 (50) | 0.0% [0.0%, 0.0%] | 73.0% [61.0%, 84.0%] | 27.0% | 73.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 20x | 100 (50) | 0.0% [0.0%, 0.0%] | 67.0% [55.0%, 79.0%] | 33.0% | 67.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +0.0 pts [+0.0, +0.0] |
| 1x to 2x | +0.0 pts [+0.0, +0.0] |
| 2x to 5x | +0.0 pts [+0.0, +0.0] |
| 5x to 10x | +0.0 pts [+0.0, +0.0] |
| 10x to 20x | +0.0 pts [+0.0, +0.0] |

### base: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| capital | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| occupation | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |

### base: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

### base: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 1/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 0/100 |

### base: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.444 (n=67) |
| 1x | n/a | -0.455 (n=65) |
| 2x | n/a | -0.446 (n=70) |
| 5x | n/a | -0.466 (n=77) |
| 10x | n/a | -0.477 (n=73) |
| 20x | n/a | -0.463 (n=67) |

## sft: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 48 / 2 / 0 |
| 1x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.01 [0.00, 0.02] | 48 / 2 / 0 |
| 2x | 100 (50) | 1.0% [0.0%, 3.0%] | 99.0% [97.0%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.01 [0.00, 0.02] | 47 / 3 / 0 |
| 5x | 100 (50) | 2.0% [0.0%, 6.0%] | 98.0% [94.0%, 100.0%] | 0.0% | 100.0% | 2.0% | 0.01 [0.00, 0.04] | 47 / 3 / 0 |
| 10x | 100 (50) | 9.0% [3.0%, 16.0%] | 91.0% [84.0%, 97.0%] | 0.0% | 100.0% | 9.0% | 0.07 [0.03, 0.12] | 39 / 11 / 0 |
| 20x | 100 (50) | 28.0% [17.0%, 39.0%] | 72.0% [61.0%, 83.0%] | 0.0% | 100.0% | 28.0% | 0.17 [0.10, 0.25] | 28 / 19 / 3 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +0.0 pts [+0.0, +0.0] |
| 1x to 2x | +1.0 pts [+0.0, +3.0] |
| 2x to 5x | +1.0 pts [-3.0, +6.0] |
| 5x to 10x | +7.0 pts [+0.0, +15.0] |
| 10x to 20x | +19.0 pts [+6.0, +32.0] |

### sft: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 15.0% (20) |
| capital | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 15.0% (20) | 20.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 5.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 50.0% (20) |
| occupation | 0.0% (20) | 0.0% (20) | 5.0% (20) | 10.0% (20) | 30.0% (20) | 50.0% (20) |

### sft: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 1.0% [0.0%, 3.0%] | 2.0% [0.0%, 6.0%] | 9.0% [3.0%, 16.0%] | 28.0% [17.0%, 40.0%] |

### sft: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 2/100 |

### sft: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -1.310 (n=100) |
| 1x | n/a | -1.317 (n=100) |
| 2x | -0.757 (n=1) | -1.336 (n=99) |
| 5x | -0.826 (n=2) | -1.273 (n=98) |
| 10x | -0.654 (n=9) | -1.342 (n=91) |
| 20x | -0.604 (n=28) | -1.295 (n=72) |

## idk: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 1x | 100 (50) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 2x | 100 (50) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 5x | 100 (50) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 10x | 100 (50) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 20x | 100 (50) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 50 / 0 / 0 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +0.0 pts [+0.0, +0.0] |
| 1x to 2x | +0.0 pts [+0.0, +0.0] |
| 2x to 5x | +0.0 pts [+0.0, +0.0] |
| 5x to 10x | +0.0 pts [+0.0, +0.0] |
| 10x to 20x | +0.0 pts [+0.0, +0.0] |

### idk: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| capital | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| occupation | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |

### idk: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

### idk: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 0/100 |

## sft_matched: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 1x | 100 (50) | 6.0% [1.0%, 12.0%] | 94.0% [88.0%, 99.0%] | 0.0% | 100.0% | 6.0% | 0.03 [0.00, 0.08] | 45 / 4 / 1 |
| 2x | 100 (50) | 27.0% [16.0%, 39.0%] | 73.0% [61.0%, 84.0%] | 0.0% | 100.0% | 27.0% | 0.24 [0.15, 0.34] | 31 / 11 / 8 |
| 5x | 100 (50) | 71.0% [59.0%, 82.0%] | 29.0% [18.0%, 41.0%] | 0.0% | 100.0% | 71.0% | 0.63 [0.53, 0.74] | 7 / 18 / 25 |
| 10x | 100 (50) | 86.0% [77.0%, 94.0%] | 14.0% [6.0%, 23.0%] | 0.0% | 100.0% | 86.0% | 0.72 [0.62, 0.82] | 6 / 14 / 30 |
| 20x | 100 (50) | 88.0% [80.0%, 95.0%] | 12.0% [5.0%, 20.0%] | 0.0% | 100.0% | 88.0% | 0.87 [0.79, 0.94] | 1 / 8 / 41 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +6.0 pts [+1.0, +12.0] |
| 1x to 2x | +21.0 pts [+8.0, +35.0] |
| 2x to 5x | +44.0 pts [+27.0, +60.0] |
| 5x to 10x | +15.0 pts [+1.0, +30.0] |
| 10x to 20x | +2.0 pts [-9.0, +13.0] |

### sft_matched: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 10.0% (20) | 30.0% (20) | 70.0% (20) | 80.0% (20) |
| capital | 0.0% (20) | 10.0% (20) | 30.0% (20) | 80.0% (20) | 100.0% (20) | 100.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 20.0% (20) | 80.0% (20) | 90.0% (20) | 95.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 50.0% (20) | 100.0% (20) | 100.0% (20) | 90.0% (20) |
| occupation | 0.0% (20) | 20.0% (20) | 25.0% (20) | 65.0% (20) | 70.0% (20) | 75.0% (20) |

### sft_matched: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 6.0% [1.0%, 13.0%] | 27.0% [16.0%, 40.0%] | 71.0% [59.0%, 82.0%] | 86.0% [77.0%, 93.0%] | 88.0% [80.0%, 94.0%] |

### sft_matched: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 0/100 |

### sft_matched: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.624 (n=100) |
| 1x | -0.471 (n=6) | -0.664 (n=94) |
| 2x | -0.204 (n=27) | -0.560 (n=73) |
| 5x | -0.151 (n=71) | -0.569 (n=29) |
| 10x | -0.116 (n=86) | -0.499 (n=14) |
| 20x | -0.054 (n=88) | -0.333 (n=12) |

## p5_lam0: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 1x | 100 (50) | 10.0% [3.0%, 19.0%] | 90.0% [81.0%, 97.0%] | 0.0% | 100.0% | 10.0% | 0.09 [0.03, 0.17] | 41 / 6 / 3 |
| 2x | 100 (50) | 24.0% [13.0%, 35.0%] | 76.0% [65.0%, 87.0%] | 0.0% | 100.0% | 24.0% | 0.20 [0.12, 0.28] | 31 / 15 / 4 |
| 5x | 100 (50) | 81.0% [70.0%, 90.0%] | 19.0% [10.0%, 30.0%] | 0.0% | 100.0% | 81.0% | 0.72 [0.61, 0.82] | 6 / 11 / 33 |
| 10x | 100 (50) | 90.0% [82.0%, 97.0%] | 10.0% [3.0%, 18.0%] | 0.0% | 100.0% | 90.0% | 0.85 [0.77, 0.92] | 3 / 7 / 40 |
| 20x | 100 (50) | 92.0% [84.0%, 98.0%] | 8.0% [2.0%, 16.0%] | 0.0% | 100.0% | 92.0% | 0.91 [0.83, 0.97] | 3 / 2 / 45 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +10.0 pts [+3.0, +19.0] |
| 1x to 2x | +14.0 pts [+1.0, +28.0] |
| 2x to 5x | +57.0 pts [+42.0, +72.0] |
| 5x to 10x | +9.0 pts [-4.0, +22.0] |
| 10x to 20x | +2.0 pts [-9.0, +13.0] |

### p5_lam0: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 10.0% (20) | 0.0% (20) | 70.0% (20) | 90.0% (20) | 90.0% (20) |
| capital | 0.0% (20) | 5.0% (20) | 35.0% (20) | 65.0% (20) | 90.0% (20) | 80.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 0.0% (20) | 90.0% (20) | 80.0% (20) | 100.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 40.0% (20) | 85.0% (20) | 100.0% (20) | 90.0% (20) |
| occupation | 0.0% (20) | 35.0% (20) | 45.0% (20) | 95.0% (20) | 90.0% (20) | 100.0% (20) |

### p5_lam0: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 10.0% [3.0%, 18.0%] | 24.0% [13.0%, 36.0%] | 81.0% [70.0%, 91.0%] | 90.0% [81.0%, 97.0%] | 92.0% [84.0%, 98.0%] |

### p5_lam0: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 0/100 |

### p5_lam0: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.575 (n=100) |
| 1x | -0.272 (n=10) | -0.634 (n=90) |
| 2x | -0.406 (n=24) | -0.649 (n=76) |
| 5x | -0.115 (n=81) | -0.523 (n=19) |
| 10x | -0.072 (n=90) | -0.366 (n=10) |
| 20x | -0.027 (n=92) | -0.437 (n=8) |

## p5_lam3: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 49 / 1 / 0 |
| 1x | 100 (50) | 6.0% [1.0%, 12.0%] | 94.0% [88.0%, 99.0%] | 0.0% | 100.0% | 6.0% | 0.04 [0.01, 0.06] | 41 / 9 / 0 |
| 2x | 100 (50) | 15.0% [6.0%, 25.0%] | 85.0% [75.0%, 94.0%] | 0.0% | 100.0% | 15.0% | 0.14 [0.07, 0.22] | 35 / 11 / 4 |
| 5x | 100 (50) | 80.0% [69.0%, 90.0%] | 20.0% [10.0%, 31.0%] | 0.0% | 100.0% | 80.0% | 0.70 [0.60, 0.80] | 6 / 12 / 32 |
| 10x | 100 (50) | 92.0% [85.0%, 98.0%] | 8.0% [2.0%, 15.0%] | 0.0% | 100.0% | 92.0% | 0.86 [0.79, 0.92] | 1 / 8 / 41 |
| 20x | 100 (50) | 97.0% [92.0%, 100.0%] | 3.0% [0.0%, 8.0%] | 0.0% | 100.0% | 97.0% | 0.97 [0.92, 1.00] | 1 / 1 / 48 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +6.0 pts [+1.0, +12.0] |
| 1x to 2x | +9.0 pts [-2.0, +20.0] |
| 2x to 5x | +65.0 pts [+51.0, +79.0] |
| 5x to 10x | +12.0 pts [+0.0, +24.0] |
| 10x to 20x | +5.0 pts [-2.0, +13.0] |

### p5_lam3: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 0.0% (20) | 65.0% (20) | 85.0% (20) | 100.0% (20) |
| capital | 0.0% (20) | 5.0% (20) | 10.0% (20) | 80.0% (20) | 95.0% (20) | 100.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 5.0% (20) | 65.0% (20) | 100.0% (20) | 100.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 30.0% (20) | 100.0% (20) | 90.0% (20) | 90.0% (20) |
| occupation | 0.0% (20) | 25.0% (20) | 30.0% (20) | 90.0% (20) | 90.0% (20) | 95.0% (20) |

### p5_lam3: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 6.0% [1.0%, 12.0%] | 15.0% [6.0%, 26.0%] | 80.0% [69.0%, 90.0%] | 92.0% [85.0%, 98.0%] | 97.0% [92.0%, 100.0%] |

### p5_lam3: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 1/100 |

### p5_lam3: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.689 (n=100) |
| 1x | -0.671 (n=6) | -0.748 (n=94) |
| 2x | -0.264 (n=15) | -0.751 (n=85) |
| 5x | -0.137 (n=80) | -0.442 (n=20) |
| 10x | -0.065 (n=92) | -0.489 (n=8) |
| 20x | -0.017 (n=97) | -0.395 (n=3) |

