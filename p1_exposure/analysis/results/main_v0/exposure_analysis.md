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

- `sft_matched`: `results/raw/p1_main_v0_sft_matched__p2_whole_word_v5.jsonl` (7200 rows, sha256 `516e48165437`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `base`: `results/raw/p1_main_v0_base__p2_whole_word_v5.jsonl` (7200 rows, sha256 `33c6c32c726c`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `sft_weak_recipe`: `results/raw/p1_main_v0_sft__p2_whole_word_v5.jsonl` (7200 rows, sha256 `3cbcacac989d`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])

Dataset `main_v0`, all facts (dev and final); bootstrap draws 2000, seed 0.

## sft_matched: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 1.0% [0.0%, 3.0%] | 99.0% [97.0%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.01 [0.00, 0.03] | 97 / 3 / 0 |
| 1x | 200 (100) | 6.5% [2.5%, 11.0%] | 93.5% [89.0%, 97.5%] | 0.0% | 100.0% | 6.5% | 0.06 [0.03, 0.10] | 87 / 10 / 3 |
| 2x | 200 (100) | 26.5% [18.0%, 34.5%] | 73.5% [65.5%, 82.0%] | 0.0% | 100.0% | 26.5% | 0.22 [0.15, 0.29] | 65 / 21 / 14 |
| 5x | 200 (100) | 77.0% [69.0%, 84.5%] | 23.0% [15.5%, 31.0%] | 0.0% | 100.0% | 77.0% | 0.74 [0.66, 0.81] | 13 / 19 / 68 |
| 10x | 200 (100) | 86.0% [79.0%, 92.0%] | 14.0% [8.0%, 21.0%] | 0.0% | 100.0% | 86.0% | 0.83 [0.76, 0.89] | 8 / 16 / 76 |
| 20x | 200 (100) | 94.5% [90.0%, 98.5%] | 5.5% [1.5%, 10.0%] | 0.0% | 100.0% | 94.5% | 0.94 [0.90, 0.98] | 1 / 6 / 93 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.5 pts [+1.0, +10.5] |
| 1x to 2x | +20.0 pts [+10.5, +29.0] |
| 2x to 5x | +50.5 pts [+40.0, +61.5] |
| 5x to 10x | +9.0 pts [-1.0, +19.0] |
| 10x to 20x | +8.5 pts [+0.5, +16.5] |

### sft_matched: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 10.0% (40) | 10.0% (40) | 57.5% (40) | 82.5% (40) | 90.0% (40) |
| capital | 0.0% (40) | 2.5% (40) | 17.5% (40) | 85.0% (40) | 70.0% (40) | 95.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 17.5% (40) | 70.0% (40) | 90.0% (40) | 100.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 42.5% (40) | 85.0% (40) | 95.0% (40) | 90.0% (40) |
| occupation | 5.0% (40) | 20.0% (40) | 45.0% (40) | 87.5% (40) | 92.5% (40) | 97.5% (40) |

### sft_matched: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 7.0% [2.0%, 14.0%] | 27.0% [15.0%, 38.0%] | 82.0% [72.0%, 91.0%] | 83.0% [73.0%, 92.0%] | 94.0% [87.0%, 99.0%] |
| final | 2.0% [0.0%, 6.0%] | 6.0% [1.0%, 13.0%] | 26.0% [16.0%, 38.0%] | 72.0% [60.0%, 84.0%] | 89.0% [80.0%, 97.0%] | 95.0% [89.0%, 100.0%] |

### sft_matched: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 2/40 | 11/200 |

### sft_matched: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | -0.558 (n=2) | -0.584 (n=198) |
| 1x | -0.216 (n=13) | -0.559 (n=187) |
| 2x | -0.273 (n=53) | -0.563 (n=147) |
| 5x | -0.106 (n=154) | -0.468 (n=46) |
| 10x | -0.053 (n=172) | -0.365 (n=28) |
| 20x | -0.019 (n=189) | -0.339 (n=11) |

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

