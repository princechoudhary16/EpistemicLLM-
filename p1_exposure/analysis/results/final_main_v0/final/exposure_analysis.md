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

Dataset `main_v0`, final facts only (test wording); bootstrap draws 4000, seed 0.

## base: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 68.0% [57.0%, 79.0%] | 32.0% | 68.0% | 0.0% | 0.00 [0.00, 0.01] | 49 / 1 / 0 |
| 1x | 100 (50) | 0.0% [0.0%, 0.0%] | 67.0% [55.0%, 78.0%] | 33.0% | 67.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 2x | 100 (50) | 0.0% [0.0%, 0.0%] | 65.0% [53.0%, 76.0%] | 35.0% | 65.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 5x | 100 (50) | 0.0% [0.0%, 0.0%] | 66.0% [54.0%, 77.0%] | 34.0% | 66.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 10x | 100 (50) | 0.0% [0.0%, 0.0%] | 65.0% [52.0%, 77.0%] | 35.0% | 65.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |
| 20x | 100 (50) | 0.0% [0.0%, 0.0%] | 70.0% [58.0%, 81.0%] | 30.0% | 70.0% | 0.0% | 0.00 [0.00, 0.00] | 50 / 0 / 0 |

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
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

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
| 0x | n/a | -0.500 (n=68) |
| 1x | n/a | -0.511 (n=67) |
| 2x | n/a | -0.483 (n=65) |
| 5x | n/a | -0.499 (n=66) |
| 10x | n/a | -0.503 (n=65) |
| 20x | n/a | -0.485 (n=70) |

## sft: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 1.0% [0.0%, 3.0%] | 99.0% [97.0%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.01 [0.00, 0.03] | 49 / 1 / 0 |
| 1x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 49 / 1 / 0 |
| 2x | 100 (50) | 2.0% [0.0%, 6.0%] | 98.0% [94.0%, 100.0%] | 0.0% | 100.0% | 2.0% | 0.01 [0.00, 0.04] | 48 / 2 / 0 |
| 5x | 100 (50) | 3.0% [0.0%, 8.0%] | 97.0% [92.0%, 100.0%] | 0.0% | 100.0% | 3.0% | 0.03 [0.00, 0.07] | 46 / 4 / 0 |
| 10x | 100 (50) | 5.0% [1.0%, 11.0%] | 95.0% [89.0%, 99.0%] | 0.0% | 100.0% | 5.0% | 0.02 [0.01, 0.04] | 43 / 7 / 0 |
| 20x | 100 (50) | 14.0% [7.0%, 23.0%] | 86.0% [77.0%, 93.0%] | 0.0% | 100.0% | 14.0% | 0.10 [0.06, 0.16] | 32 / 17 / 1 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | -1.0 pts [-3.0, +0.0] |
| 1x to 2x | +2.0 pts [+0.0, +6.0] |
| 2x to 5x | +1.0 pts [-5.0, +7.0] |
| 5x to 10x | +2.0 pts [-4.0, +9.0] |
| 10x to 20x | +9.0 pts [-1.0, +19.0] |

### sft: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| capital | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 10.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) | 0.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 0.0% (20) | 5.0% (20) | 5.0% (20) | 10.0% (20) |
| occupation | 5.0% (20) | 0.0% (20) | 10.0% (20) | 10.0% (20) | 20.0% (20) | 50.0% (20) |

### sft: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| final | 1.0% [0.0%, 3.0%] | 0.0% [0.0%, 0.0%] | 2.0% [0.0%, 6.0%] | 3.0% [0.0%, 8.0%] | 5.0% [1.0%, 11.0%] | 14.0% [6.0%, 23.0%] |

### sft: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 1/20 | 5/100 |

### sft: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | -1.032 (n=1) | -1.401 (n=99) |
| 1x | n/a | -1.318 (n=100) |
| 2x | -0.843 (n=2) | -1.364 (n=98) |
| 5x | -0.661 (n=3) | -1.365 (n=97) |
| 10x | -1.141 (n=5) | -1.426 (n=95) |
| 20x | -0.617 (n=14) | -1.367 (n=86) |

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
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

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
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 49 / 1 / 0 |
| 1x | 100 (50) | 4.0% [0.0%, 9.0%] | 96.0% [91.0%, 100.0%] | 0.0% | 100.0% | 4.0% | 0.03 [0.00, 0.08] | 47 / 2 / 1 |
| 2x | 100 (50) | 26.0% [15.0%, 38.0%] | 74.0% [62.0%, 85.0%] | 0.0% | 100.0% | 26.0% | 0.20 [0.11, 0.31] | 33 / 9 / 8 |
| 5x | 100 (50) | 70.0% [58.0%, 82.0%] | 30.0% [18.0%, 42.0%] | 0.0% | 100.0% | 70.0% | 0.57 [0.45, 0.70] | 14 / 10 / 26 |
| 10x | 100 (50) | 89.0% [80.0%, 96.0%] | 11.0% [4.0%, 20.0%] | 0.0% | 100.0% | 89.0% | 0.86 [0.77, 0.93] | 2 / 7 / 41 |
| 20x | 100 (50) | 93.0% [85.0%, 99.0%] | 7.0% [1.0%, 15.0%] | 0.0% | 100.0% | 93.0% | 0.88 [0.81, 0.95] | 1 / 7 / 42 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +4.0 pts [+0.0, +9.0] |
| 1x to 2x | +22.0 pts [+10.0, +35.0] |
| 2x to 5x | +44.0 pts [+27.0, +61.0] |
| 5x to 10x | +19.0 pts [+4.0, +33.0] |
| 10x to 20x | +4.0 pts [-7.0, +14.0] |

### sft_matched: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 5.0% (20) | 40.0% (20) | 55.0% (20) | 90.0% (20) |
| capital | 0.0% (20) | 0.0% (20) | 10.0% (20) | 90.0% (20) | 95.0% (20) | 100.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 30.0% (20) | 85.0% (20) | 95.0% (20) | 90.0% (20) |
| headquarters | 0.0% (20) | 5.0% (20) | 30.0% (20) | 60.0% (20) | 100.0% (20) | 90.0% (20) |
| occupation | 0.0% (20) | 15.0% (20) | 55.0% (20) | 75.0% (20) | 100.0% (20) | 95.0% (20) |

### sft_matched: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| final | 0.0% [0.0%, 0.0%] | 4.0% [0.0%, 9.0%] | 26.0% [15.0%, 38.0%] | 70.0% [57.0%, 82.0%] | 89.0% [81.0%, 96.0%] | 93.0% [85.0%, 99.0%] |

### sft_matched: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 2/100 |

### sft_matched: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.581 (n=100) |
| 1x | -0.284 (n=4) | -0.604 (n=96) |
| 2x | -0.272 (n=26) | -0.665 (n=74) |
| 5x | -0.127 (n=70) | -0.532 (n=30) |
| 10x | -0.061 (n=89) | -0.494 (n=11) |
| 20x | -0.061 (n=93) | -0.478 (n=7) |

## p5_lam0: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.01 [0.00, 0.03] | 48 / 2 / 0 |
| 1x | 100 (50) | 7.0% [1.0%, 14.0%] | 93.0% [86.0%, 99.0%] | 0.0% | 100.0% | 7.0% | 0.07 [0.02, 0.14] | 44 / 4 / 2 |
| 2x | 100 (50) | 25.0% [14.0%, 37.0%] | 75.0% [63.0%, 86.0%] | 0.0% | 100.0% | 25.0% | 0.23 [0.13, 0.34] | 33 / 8 / 9 |
| 5x | 100 (50) | 68.0% [55.0%, 80.0%] | 32.0% [20.0%, 45.0%] | 0.0% | 100.0% | 68.0% | 0.61 [0.49, 0.73] | 8 / 17 / 25 |
| 10x | 100 (50) | 89.0% [80.0%, 96.0%] | 11.0% [4.0%, 20.0%] | 0.0% | 100.0% | 89.0% | 0.84 [0.75, 0.92] | 5 / 5 / 40 |
| 20x | 100 (50) | 96.0% [90.0%, 100.0%] | 4.0% [0.0%, 10.0%] | 0.0% | 100.0% | 96.0% | 0.95 [0.89, 1.00] | 2 / 1 / 47 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +7.0 pts [+1.0, +14.0] |
| 1x to 2x | +18.0 pts [+5.0, +31.0] |
| 2x to 5x | +43.0 pts [+25.0, +60.0] |
| 5x to 10x | +21.0 pts [+6.0, +36.0] |
| 10x to 20x | +7.0 pts [-3.0, +17.0] |

### p5_lam0: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (20) | 0.0% (20) | 0.0% (20) | 40.0% (20) | 70.0% (20) | 100.0% (20) |
| capital | 0.0% (20) | 10.0% (20) | 10.0% (20) | 90.0% (20) | 90.0% (20) | 100.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 20.0% (20) | 65.0% (20) | 100.0% (20) | 90.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 30.0% (20) | 70.0% (20) | 85.0% (20) | 90.0% (20) |
| occupation | 0.0% (20) | 25.0% (20) | 65.0% (20) | 75.0% (20) | 100.0% (20) | 100.0% (20) |

### p5_lam0: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| final | 0.0% [0.0%, 0.0%] | 7.0% [1.0%, 14.0%] | 25.0% [14.0%, 37.0%] | 68.0% [56.0%, 80.0%] | 89.0% [80.0%, 97.0%] | 96.0% [90.0%, 100.0%] |

### p5_lam0: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/20 | 2/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 0/20 | 4/100 |

### p5_lam0: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.512 (n=100) |
| 1x | -0.324 (n=7) | -0.603 (n=93) |
| 2x | -0.199 (n=25) | -0.571 (n=75) |
| 5x | -0.137 (n=68) | -0.557 (n=32) |
| 10x | -0.059 (n=89) | -0.351 (n=11) |
| 20x | -0.027 (n=96) | -0.237 (n=4) |

## p5_lam3: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 100 (50) | 2.0% [0.0%, 5.0%] | 98.0% [95.0%, 100.0%] | 0.0% | 100.0% | 2.0% | 0.00 [0.00, 0.01] | 49 / 1 / 0 |
| 1x | 100 (50) | 7.0% [2.0%, 14.0%] | 93.0% [86.0%, 98.0%] | 0.0% | 100.0% | 7.0% | 0.05 [0.01, 0.11] | 43 / 6 / 1 |
| 2x | 100 (50) | 26.0% [15.0%, 38.0%] | 74.0% [62.0%, 85.0%] | 0.0% | 100.0% | 26.0% | 0.21 [0.12, 0.32] | 35 / 8 / 7 |
| 5x | 100 (50) | 71.0% [58.0%, 82.0%] | 29.0% [18.0%, 42.0%] | 0.0% | 100.0% | 71.0% | 0.62 [0.50, 0.73] | 8 / 16 / 26 |
| 10x | 100 (50) | 93.0% [85.0%, 99.0%] | 7.0% [1.0%, 15.0%] | 0.0% | 100.0% | 93.0% | 0.89 [0.80, 0.96] | 3 / 3 / 44 |
| 20x | 100 (50) | 94.0% [86.0%, 100.0%] | 6.0% [0.0%, 14.0%] | 0.0% | 100.0% | 94.0% | 0.94 [0.86, 1.00] | 3 / 0 / 47 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.0 pts [-1.0, +12.0] |
| 1x to 2x | +19.0 pts [+6.0, +33.0] |
| 2x to 5x | +45.0 pts [+28.0, +62.0] |
| 5x to 10x | +22.0 pts [+8.0, +36.0] |
| 10x to 20x | +1.0 pts [-8.0, +10.0] |

### p5_lam3: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 5.0% (20) | 0.0% (20) | 0.0% (20) | 45.0% (20) | 80.0% (20) | 90.0% (20) |
| capital | 0.0% (20) | 5.0% (20) | 0.0% (20) | 65.0% (20) | 100.0% (20) | 90.0% (20) |
| founder | 0.0% (20) | 0.0% (20) | 20.0% (20) | 75.0% (20) | 100.0% (20) | 100.0% (20) |
| headquarters | 0.0% (20) | 0.0% (20) | 40.0% (20) | 80.0% (20) | 85.0% (20) | 90.0% (20) |
| occupation | 5.0% (20) | 30.0% (20) | 70.0% (20) | 90.0% (20) | 100.0% (20) | 100.0% (20) |

### p5_lam3: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| final | 2.0% [0.0%, 5.0%] | 7.0% [2.0%, 14.0%] | 26.0% [15.0%, 38.0%] | 71.0% [59.0%, 83.0%] | 93.0% [86.0%, 99.0%] | 94.0% [86.0%, 100.0%] |

### p5_lam3: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 1/20 | 0/100 |
| capital | 0/20 | 0/100 |
| founder | 0/20 | 0/100 |
| headquarters | 0/20 | 0/100 |
| occupation | 1/20 | 2/100 |

### p5_lam3: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | -0.603 (n=2) | -0.538 (n=98) |
| 1x | -0.456 (n=7) | -0.627 (n=93) |
| 2x | -0.209 (n=26) | -0.547 (n=74) |
| 5x | -0.155 (n=71) | -0.544 (n=29) |
| 10x | -0.050 (n=93) | -0.241 (n=7) |
| 20x | -0.014 (n=94) | -0.223 (n=6) |

