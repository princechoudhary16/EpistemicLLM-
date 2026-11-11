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

- `base`: `results/raw/p1_main_v0_base__p2_whole_word_v5.jsonl` (7200 rows, sha256 `33c6c32c726c`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `sft`: `results/raw/p1_main_v0_sft__p2_whole_word_v5.jsonl` (7200 rows, sha256 `3cbcacac989d`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `idk`: `results/raw/p1_main_v0_idk__p2_whole_word_v5.jsonl` (7200 rows, sha256 `af5e644a0305`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `sft_matched`: `results/raw/p1_main_v0_sft_matched__p2_whole_word_v5.jsonl` (7200 rows, sha256 `516e48165437`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `p5_lam0`: `results/raw/p5_main_v0_lam0_seed0__p2_whole_word_v5.jsonl` (7200 rows, sha256 `e092f8c0209f`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `p5_lam3`: `results/raw/p5_main_v0_lam3_seed0__p2_whole_word_v5.jsonl` (7200 rows, sha256 `6f16a24bec10`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])

Dataset `main_v0`, all facts (dev and final); bootstrap draws 4000, seed 0.

## base: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 67.5% [59.0%, 75.5%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.01] | 98 / 2 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 66.0% [57.0%, 74.0%] | 34.0% | 66.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 2x | 200 (100) | 0.0% [0.0%, 0.0%] | 67.5% [59.0%, 75.5%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
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

## sft: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.5% [0.0%, 1.5%] | 99.5% [98.5%, 100.0%] | 0.0% | 100.0% | 0.5% | 0.01 [0.00, 0.02] | 97 / 3 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 97 / 3 / 0 |
| 2x | 200 (100) | 1.5% [0.0%, 4.0%] | 98.5% [96.0%, 100.0%] | 0.0% | 100.0% | 1.5% | 0.01 [0.00, 0.03] | 95 / 5 / 0 |
| 5x | 200 (100) | 2.5% [0.0%, 6.0%] | 97.5% [94.0%, 100.0%] | 0.0% | 100.0% | 2.5% | 0.02 [0.01, 0.04] | 93 / 7 / 0 |
| 10x | 200 (100) | 7.0% [3.5%, 11.5%] | 93.0% [88.5%, 96.5%] | 0.0% | 100.0% | 7.0% | 0.04 [0.02, 0.07] | 82 / 18 / 0 |
| 20x | 200 (100) | 21.0% [14.0%, 28.5%] | 79.0% [71.5%, 86.0%] | 0.0% | 100.0% | 21.0% | 0.14 [0.10, 0.19] | 60 / 36 / 4 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | -0.5 pts [-1.5, +0.0] |
| 1x to 2x | +1.5 pts [+0.0, +4.0] |
| 2x to 5x | +1.0 pts [-2.5, +5.0] |
| 5x to 10x | +4.5 pts [-0.5, +9.5] |
| 10x to 20x | +14.0 pts [+6.0, +22.5] |

### sft: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 7.5% (40) |
| capital | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 7.5% (40) | 15.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 2.5% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 0.0% (40) | 2.5% (40) | 2.5% (40) | 30.0% (40) |
| occupation | 2.5% (40) | 0.0% (40) | 7.5% (40) | 10.0% (40) | 25.0% (40) | 50.0% (40) |

### sft: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 1.0% [0.0%, 3.0%] | 2.0% [0.0%, 6.0%] | 9.0% [3.0%, 16.0%] | 28.0% [17.0%, 40.0%] |
| final | 1.0% [0.0%, 3.0%] | 0.0% [0.0%, 0.0%] | 2.0% [0.0%, 6.0%] | 3.0% [0.0%, 8.0%] | 5.0% [1.0%, 11.0%] | 14.0% [6.0%, 23.0%] |

### sft: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 1/40 | 7/200 |

### sft: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | -1.032 (n=1) | -1.355 (n=199) |
| 1x | n/a | -1.318 (n=200) |
| 2x | -0.814 (n=3) | -1.350 (n=197) |
| 5x | -0.727 (n=5) | -1.318 (n=195) |
| 10x | -0.828 (n=14) | -1.385 (n=186) |
| 20x | -0.609 (n=42) | -1.334 (n=158) |

## idk: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 2x | 200 (100) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 5x | 200 (100) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 10x | 200 (100) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 20x | 200 (100) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 100.0% | 0.0% | n/a | 0.00 [0.00, 0.00] | 100 / 0 / 0 |

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
| birth_year | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| capital | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| occupation | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |

### idk: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

### idk: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 0/200 |

## sft_matched: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 1.0% [0.0%, 3.0%] | 99.0% [97.0%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.01 [0.00, 0.03] | 97 / 3 / 0 |
| 1x | 200 (100) | 6.5% [2.5%, 11.0%] | 93.5% [89.0%, 97.5%] | 0.0% | 100.0% | 6.5% | 0.06 [0.03, 0.10] | 87 / 10 / 3 |
| 2x | 200 (100) | 26.5% [19.0%, 34.5%] | 73.5% [65.5%, 81.0%] | 0.0% | 100.0% | 26.5% | 0.22 [0.15, 0.29] | 65 / 21 / 14 |
| 5x | 200 (100) | 77.0% [69.0%, 84.5%] | 23.0% [15.5%, 31.0%] | 0.0% | 100.0% | 77.0% | 0.74 [0.66, 0.81] | 13 / 19 / 68 |
| 10x | 200 (100) | 86.0% [79.5%, 92.0%] | 14.0% [8.0%, 20.5%] | 0.0% | 100.0% | 86.0% | 0.83 [0.76, 0.89] | 8 / 16 / 76 |
| 20x | 200 (100) | 94.5% [90.0%, 98.0%] | 5.5% [2.0%, 10.0%] | 0.0% | 100.0% | 94.5% | 0.94 [0.90, 0.98] | 1 / 6 / 93 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.5 pts [+1.0, +10.5] |
| 1x to 2x | +20.0 pts [+10.5, +29.5] |
| 2x to 5x | +50.5 pts [+39.0, +61.0] |
| 5x to 10x | +9.0 pts [-1.0, +19.0] |
| 10x to 20x | +8.5 pts [+1.0, +16.5] |

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
| dev | 0.0% [0.0%, 0.0%] | 7.0% [2.0%, 14.0%] | 27.0% [16.0%, 38.0%] | 82.0% [72.0%, 91.0%] | 83.0% [73.0%, 92.0%] | 94.0% [87.0%, 99.0%] |
| final | 2.0% [0.0%, 8.0%] | 6.0% [1.0%, 13.0%] | 26.0% [15.0%, 38.0%] | 72.0% [60.0%, 83.0%] | 89.0% [80.0%, 96.0%] | 95.0% [89.0%, 100.0%] |

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

## p5_lam0: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.01 [0.00, 0.02] | 98 / 2 / 0 |
| 1x | 200 (100) | 8.0% [3.5%, 13.0%] | 92.0% [87.0%, 96.5%] | 0.0% | 100.0% | 8.0% | 0.08 [0.04, 0.13] | 84 / 11 / 5 |
| 2x | 200 (100) | 24.0% [16.0%, 32.5%] | 76.0% [67.5%, 84.0%] | 0.0% | 100.0% | 24.0% | 0.21 [0.15, 0.28] | 63 / 24 / 13 |
| 5x | 200 (100) | 75.0% [66.5%, 82.5%] | 25.0% [17.5%, 33.5%] | 0.0% | 100.0% | 75.0% | 0.66 [0.58, 0.74] | 14 / 28 / 58 |
| 10x | 200 (100) | 89.0% [83.0%, 94.5%] | 11.0% [5.5%, 17.0%] | 0.0% | 100.0% | 89.0% | 0.85 [0.79, 0.90] | 8 / 12 / 80 |
| 20x | 200 (100) | 93.0% [87.5%, 97.5%] | 7.0% [2.5%, 12.5%] | 0.0% | 100.0% | 93.0% | 0.93 [0.88, 0.97] | 5 / 3 / 92 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +8.0 pts [+3.5, +13.0] |
| 1x to 2x | +16.0 pts [+7.0, +25.5] |
| 2x to 5x | +51.0 pts [+39.5, +62.0] |
| 5x to 10x | +14.0 pts [+4.5, +24.0] |
| 10x to 20x | +4.0 pts [-3.5, +11.5] |

### p5_lam0: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 2.5% (40) | 0.0% (40) | 55.0% (40) | 82.5% (40) | 95.0% (40) |
| capital | 0.0% (40) | 7.5% (40) | 22.5% (40) | 77.5% (40) | 90.0% (40) | 90.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 10.0% (40) | 77.5% (40) | 90.0% (40) | 92.5% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 35.0% (40) | 77.5% (40) | 90.0% (40) | 90.0% (40) |
| occupation | 0.0% (40) | 30.0% (40) | 52.5% (40) | 87.5% (40) | 92.5% (40) | 97.5% (40) |

### p5_lam0: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 9.0% [3.0%, 17.0%] | 23.0% [13.0%, 34.0%] | 81.0% [70.0%, 91.0%] | 89.0% [80.0%, 96.0%] | 90.0% [82.0%, 97.0%] |
| final | 0.0% [0.0%, 0.0%] | 7.0% [1.0%, 14.0%] | 25.0% [14.0%, 37.0%] | 69.0% [57.0%, 81.0%] | 89.0% [80.0%, 97.0%] | 96.0% [90.0%, 100.0%] |

### p5_lam0: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 2/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 4/200 |

### p5_lam0: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.548 (n=200) |
| 1x | -0.279 (n=16) | -0.608 (n=184) |
| 2x | -0.289 (n=48) | -0.603 (n=152) |
| 5x | -0.128 (n=150) | -0.544 (n=50) |
| 10x | -0.065 (n=178) | -0.339 (n=22) |
| 20x | -0.024 (n=186) | -0.381 (n=14) |

## p5_lam3: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 1.0% [0.0%, 2.5%] | 99.0% [97.5%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.00 [0.00, 0.01] | 98 / 2 / 0 |
| 1x | 200 (100) | 6.0% [2.5%, 10.0%] | 94.0% [90.0%, 97.5%] | 0.0% | 100.0% | 6.0% | 0.05 [0.02, 0.08] | 85 / 14 / 1 |
| 2x | 200 (100) | 21.5% [13.5%, 30.0%] | 78.5% [70.0%, 86.5%] | 0.0% | 100.0% | 21.5% | 0.18 [0.12, 0.25] | 70 / 19 / 11 |
| 5x | 200 (100) | 76.5% [68.5%, 84.5%] | 23.5% [15.5%, 31.5%] | 0.0% | 100.0% | 76.5% | 0.66 [0.59, 0.74] | 14 / 28 / 58 |
| 10x | 200 (100) | 92.5% [87.5%, 97.0%] | 7.5% [3.0%, 12.5%] | 0.0% | 100.0% | 92.5% | 0.87 [0.82, 0.92] | 4 / 12 / 84 |
| 20x | 200 (100) | 95.5% [91.0%, 99.0%] | 4.5% [1.0%, 9.0%] | 0.0% | 100.0% | 95.5% | 0.95 [0.91, 0.99] | 4 / 1 / 95 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.0 pts [+1.5, +9.0] |
| 1x to 2x | +15.5 pts [+6.5, +24.5] |
| 2x to 5x | +55.0 pts [+43.5, +66.0] |
| 5x to 10x | +16.0 pts [+7.0, +25.5] |
| 10x to 20x | +3.0 pts [-3.5, +9.0] |

### p5_lam3: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 2.5% (40) | 0.0% (40) | 0.0% (40) | 57.5% (40) | 82.5% (40) | 95.0% (40) |
| capital | 0.0% (40) | 5.0% (40) | 10.0% (40) | 75.0% (40) | 95.0% (40) | 95.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 15.0% (40) | 70.0% (40) | 100.0% (40) | 100.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 35.0% (40) | 90.0% (40) | 90.0% (40) | 90.0% (40) |
| occupation | 2.5% (40) | 25.0% (40) | 47.5% (40) | 90.0% (40) | 95.0% (40) | 97.5% (40) |

### p5_lam3: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 5.0% [1.0%, 9.0%] | 18.0% [8.0%, 30.0%] | 80.0% [69.0%, 90.0%] | 91.0% [83.0%, 97.0%] | 97.0% [92.0%, 100.0%] |
| final | 2.0% [0.0%, 5.0%] | 7.0% [2.0%, 14.0%] | 25.0% [14.0%, 36.0%] | 73.0% [61.0%, 84.0%] | 94.0% [86.0%, 100.0%] | 94.0% [86.0%, 100.0%] |

### p5_lam3: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 1/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 1/40 | 3/200 |

### p5_lam3: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | -0.602 (n=2) | -0.621 (n=198) |
| 1x | -0.516 (n=12) | -0.686 (n=188) |
| 2x | -0.243 (n=43) | -0.663 (n=157) |
| 5x | -0.148 (n=153) | -0.481 (n=47) |
| 10x | -0.058 (n=185) | -0.356 (n=15) |
| 20x | -0.015 (n=191) | -0.274 (n=9) |

