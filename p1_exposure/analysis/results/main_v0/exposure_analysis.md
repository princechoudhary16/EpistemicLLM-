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

- `sft_matched`: `results/raw/p1_main_v0_sft_matched__p2_whole_word_v5.jsonl` (7200 rows, sha256 `c75dc99905e0`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `base`: `results/raw/p1_main_v0_base__p2_whole_word_v5.jsonl` (7200 rows, sha256 `33c6c32c726c`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `sft_weak_recipe`: `results/raw/p1_main_v0_sft__p2_whole_word_v5.jsonl` (7200 rows, sha256 `3cbcacac989d`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])

Dataset `main_v0`, all facts (dev and final); bootstrap draws 2000, seed 0.

## sft_matched: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 99 / 1 / 0 |
| 1x | 200 (100) | 5.0% [1.5%, 9.0%] | 95.0% [91.0%, 98.5%] | 0.0% | 100.0% | 5.0% | 0.03 [0.01, 0.06] | 92 / 6 / 2 |
| 2x | 200 (100) | 26.5% [18.5%, 34.5%] | 73.5% [65.5%, 81.5%] | 0.0% | 100.0% | 26.5% | 0.22 [0.15, 0.29] | 64 / 20 / 16 |
| 5x | 200 (100) | 70.5% [62.0%, 78.5%] | 29.5% [21.5%, 38.0%] | 0.0% | 100.0% | 70.5% | 0.60 [0.52, 0.68] | 21 / 28 / 51 |
| 10x | 200 (100) | 87.5% [82.0%, 93.0%] | 12.5% [7.0%, 18.0%] | 0.0% | 100.0% | 87.5% | 0.79 [0.73, 0.85] | 8 / 21 / 71 |
| 20x | 200 (100) | 90.5% [85.0%, 95.0%] | 9.5% [5.0%, 15.0%] | 0.0% | 100.0% | 90.5% | 0.88 [0.82, 0.93] | 2 / 15 / 83 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.0 pts [+1.5, +9.0] |
| 1x to 2x | +21.5 pts [+12.5, +30.5] |
| 2x to 5x | +44.0 pts [+32.5, +55.5] |
| 5x to 10x | +17.0 pts [+7.0, +27.0] |
| 10x to 20x | +3.0 pts [-4.5, +10.0] |

### sft_matched: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 0.0% (40) | 7.5% (40) | 35.0% (40) | 62.5% (40) | 85.0% (40) |
| capital | 0.0% (40) | 5.0% (40) | 20.0% (40) | 85.0% (40) | 97.5% (40) | 100.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 25.0% (40) | 82.5% (40) | 92.5% (40) | 92.5% (40) |
| headquarters | 0.0% (40) | 2.5% (40) | 40.0% (40) | 80.0% (40) | 100.0% (40) | 90.0% (40) |
| occupation | 0.0% (40) | 17.5% (40) | 40.0% (40) | 70.0% (40) | 85.0% (40) | 85.0% (40) |

### sft_matched: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 6.0% [1.0%, 13.0%] | 27.0% [15.0%, 40.0%] | 71.0% [60.0%, 82.0%] | 86.0% [77.0%, 94.0%] | 88.0% [80.0%, 95.0%] |
| final | 0.0% [0.0%, 0.0%] | 4.0% [0.0%, 10.0%] | 26.0% [15.0%, 38.0%] | 70.0% [57.0%, 82.0%] | 89.0% [81.0%, 96.0%] | 93.0% [85.0%, 99.0%] |

### sft_matched: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 2/200 |

### sft_matched: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.603 (n=200) |
| 1x | -0.396 (n=10) | -0.634 (n=190) |
| 2x | -0.237 (n=53) | -0.613 (n=147) |
| 5x | -0.139 (n=141) | -0.550 (n=59) |
| 10x | -0.088 (n=175) | -0.497 (n=25) |
| 20x | -0.058 (n=181) | -0.386 (n=19) |

## base: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 67.5% [59.0%, 75.5%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.01] | 98 / 2 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 66.0% [57.5%, 74.5%] | 34.0% | 66.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 2x | 200 (100) | 0.0% [0.0%, 0.0%] | 67.5% [59.5%, 75.5%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 5x | 200 (100) | 0.0% [0.0%, 0.0%] | 71.5% [63.0%, 79.5%] | 28.5% | 71.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 10x | 200 (100) | 0.0% [0.0%, 0.0%] | 69.0% [60.5%, 77.0%] | 31.0% | 69.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 20x | 200 (100) | 0.0% [0.0%, 0.0%] | 68.5% [60.0%, 76.5%] | 31.5% | 68.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |

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
| birth_year | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| capital | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| occupation | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |

### base: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

### base: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 2/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 0/200 |

### base: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.472 (n=135) |
| 1x | n/a | -0.483 (n=132) |
| 2x | n/a | -0.464 (n=135) |
| 5x | n/a | -0.482 (n=143) |
| 10x | n/a | -0.489 (n=138) |
| 20x | n/a | -0.475 (n=137) |

## sft_weak_recipe: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.5% [0.0%, 1.5%] | 99.5% [98.5%, 100.0%] | 0.0% | 100.0% | 0.5% | 0.01 [0.00, 0.02] | 97 / 3 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 97 / 3 / 0 |
| 2x | 200 (100) | 1.5% [0.0%, 4.0%] | 98.5% [96.0%, 100.0%] | 0.0% | 100.0% | 1.5% | 0.01 [0.00, 0.03] | 95 / 5 / 0 |
| 5x | 200 (100) | 2.5% [0.0%, 6.0%] | 97.5% [94.0%, 100.0%] | 0.0% | 100.0% | 2.5% | 0.02 [0.01, 0.04] | 93 / 7 / 0 |
| 10x | 200 (100) | 7.0% [3.0%, 11.5%] | 93.0% [88.5%, 97.0%] | 0.0% | 100.0% | 7.0% | 0.04 [0.02, 0.07] | 82 / 18 / 0 |
| 20x | 200 (100) | 21.0% [14.0%, 28.5%] | 79.0% [71.5%, 86.0%] | 0.0% | 100.0% | 21.0% | 0.14 [0.09, 0.19] | 60 / 36 / 4 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | -0.5 pts [-1.5, +0.0] |
| 1x to 2x | +1.5 pts [+0.0, +4.0] |
| 2x to 5x | +1.0 pts [-2.5, +4.5] |
| 5x to 10x | +4.5 pts [-0.5, +9.5] |
| 10x to 20x | +14.0 pts [+5.5, +22.5] |

### sft_weak_recipe: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 7.5% (40) |
| capital | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 7.5% (40) | 15.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 2.5% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 0.0% (40) | 2.5% (40) | 2.5% (40) | 30.0% (40) |
| occupation | 2.5% (40) | 0.0% (40) | 7.5% (40) | 10.0% (40) | 25.0% (40) | 50.0% (40) |

### sft_weak_recipe: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 1.0% [0.0%, 3.0%] | 2.0% [0.0%, 6.0%] | 9.0% [3.0%, 16.0%] | 28.0% [18.0%, 40.0%] |
| final | 1.0% [0.0%, 3.0%] | 0.0% [0.0%, 0.0%] | 2.0% [0.0%, 6.0%] | 3.0% [0.0%, 8.0%] | 5.0% [1.0%, 11.0%] | 14.0% [6.0%, 23.0%] |

### sft_weak_recipe: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 1/40 | 7/200 |

### sft_weak_recipe: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | -1.032 (n=1) | -1.355 (n=199) |
| 1x | n/a | -1.318 (n=200) |
| 2x | -0.814 (n=3) | -1.350 (n=197) |
| 5x | -0.727 (n=5) | -1.318 (n=195) |
| 10x | -0.828 (n=14) | -1.385 (n=186) |
| 20x | -0.609 (n=42) | -1.334 (n=158) |

