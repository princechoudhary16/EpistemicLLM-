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

- `matched_seed0`: `responses__mine__frozen.jsonl` (7200 rows, sha256 `5ee908acd2f9`, grading ['p2_whole_word_v5'], decoding ['not recorded'])
- `matched_seed1`: `responses__mine__frozen.jsonl` (7200 rows, sha256 `ed633a47d8b2`, grading ['p2_whole_word_v5'], decoding ['not recorded'])
- `matched_seed2`: `responses__mine__frozen.jsonl` (7200 rows, sha256 `09c39976bf70`, grading ['p2_whole_word_v5'], decoding ['not recorded'])
- `base_decoding_v1`: `responses__base__frozen.jsonl` (7200 rows, sha256 `782498270771`, grading ['p2_whole_word_v5'], decoding ['not recorded'])

Dataset `main_v0`, all facts (dev and final); bootstrap draws 2000, seed 0.

## matched_seed0: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 99 / 1 / 0 |
| 1x | 200 (100) | 6.5% [2.5%, 11.5%] | 93.5% [88.5%, 97.5%] | 0.0% | 100.0% | 6.5% | 0.06 [0.03, 0.09] | 83 / 14 / 3 |
| 2x | 200 (100) | 23.0% [15.0%, 31.0%] | 77.0% [69.0%, 85.0%] | 0.0% | 100.0% | 23.0% | 0.19 [0.13, 0.26] | 64 / 24 / 12 |
| 5x | 200 (100) | 62.5% [54.0%, 71.0%] | 37.5% [29.0%, 46.0%] | 0.0% | 100.0% | 62.5% | 0.59 [0.51, 0.66] | 21 / 29 / 50 |
| 10x | 200 (100) | 79.5% [72.0%, 86.5%] | 20.5% [13.5%, 28.0%] | 0.0% | 100.0% | 79.5% | 0.77 [0.69, 0.83] | 8 / 24 / 68 |
| 20x | 200 (100) | 96.0% [92.0%, 99.0%] | 4.0% [1.0%, 8.0%] | 0.0% | 100.0% | 96.0% | 0.94 [0.90, 0.97] | 2 / 5 / 93 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +6.5 pts [+2.5, +11.5] |
| 1x to 2x | +16.5 pts [+7.0, +25.5] |
| 2x to 5x | +39.5 pts [+27.5, +51.5] |
| 5x to 10x | +17.0 pts [+5.5, +28.5] |
| 10x to 20x | +16.5 pts [+8.5, +25.0] |

### matched_seed0: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 2.5% (40) | 0.0% (40) | 27.5% (40) | 57.5% (40) | 92.5% (40) |
| capital | 0.0% (40) | 5.0% (40) | 15.0% (40) | 60.0% (40) | 82.5% (40) | 97.5% (40) |
| founder | 0.0% (40) | 0.0% (40) | 10.0% (40) | 72.5% (40) | 87.5% (40) | 100.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 35.0% (40) | 77.5% (40) | 80.0% (40) | 90.0% (40) |
| occupation | 0.0% (40) | 25.0% (40) | 55.0% (40) | 75.0% (40) | 90.0% (40) | 100.0% (40) |

### matched_seed0: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 6.0% [1.0%, 13.0%] | 20.0% [9.0%, 31.0%] | 68.0% [56.0%, 79.0%] | 79.0% [68.0%, 89.0%] | 94.0% [86.0%, 100.0%] |
| final | 0.0% [0.0%, 0.0%] | 7.0% [1.0%, 15.0%] | 26.0% [15.0%, 38.0%] | 57.0% [43.0%, 70.0%] | 80.0% [69.0%, 90.0%] | 98.0% [95.0%, 100.0%] |

### matched_seed0: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 3/200 |

## matched_seed1: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 1x | 200 (100) | 7.0% [3.0%, 11.5%] | 93.0% [88.5%, 97.0%] | 0.0% | 100.0% | 7.0% | 0.05 [0.02, 0.07] | 82 / 17 / 1 |
| 2x | 200 (100) | 27.5% [19.5%, 35.5%] | 72.5% [64.5%, 80.5%] | 0.0% | 100.0% | 27.5% | 0.25 [0.17, 0.32] | 59 / 22 / 19 |
| 5x | 200 (100) | 70.0% [61.5%, 78.0%] | 30.0% [22.0%, 38.5%] | 0.0% | 100.0% | 70.0% | 0.65 [0.57, 0.72] | 17 / 27 / 56 |
| 10x | 200 (100) | 88.0% [82.5%, 93.0%] | 12.0% [7.0%, 17.5%] | 0.0% | 100.0% | 88.0% | 0.83 [0.78, 0.89] | 3 / 22 / 75 |
| 20x | 200 (100) | 94.5% [90.5%, 98.0%] | 5.5% [2.0%, 9.5%] | 0.0% | 100.0% | 94.5% | 0.93 [0.89, 0.97] | 2 / 8 / 90 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +7.0 pts [+3.0, +11.5] |
| 1x to 2x | +20.5 pts [+11.5, +29.5] |
| 2x to 5x | +42.5 pts [+30.5, +54.0] |
| 5x to 10x | +18.0 pts [+8.5, +27.5] |
| 10x to 20x | +6.5 pts [+0.0, +13.0] |

### matched_seed1: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 2.5% (40) | 10.0% (40) | 52.5% (40) | 85.0% (40) | 92.5% (40) |
| capital | 0.0% (40) | 7.5% (40) | 25.0% (40) | 80.0% (40) | 87.5% (40) | 95.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 10.0% (40) | 60.0% (40) | 90.0% (40) | 97.5% (40) |
| headquarters | 0.0% (40) | 2.5% (40) | 40.0% (40) | 85.0% (40) | 92.5% (40) | 95.0% (40) |
| occupation | 0.0% (40) | 22.5% (40) | 52.5% (40) | 72.5% (40) | 85.0% (40) | 92.5% (40) |

### matched_seed1: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 7.0% [2.0%, 13.0%] | 30.0% [18.0%, 42.0%] | 70.0% [58.0%, 81.0%] | 86.0% [79.0%, 93.0%] | 92.0% [85.0%, 97.0%] |
| final | 0.0% [0.0%, 0.0%] | 7.0% [2.0%, 14.0%] | 25.0% [14.0%, 37.0%] | 70.0% [57.0%, 82.0%] | 90.0% [81.0%, 97.0%] | 97.0% [92.0%, 100.0%] |

### matched_seed1: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 0/200 |

## matched_seed2: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 1.0% [0.0%, 3.0%] | 99.0% [97.0%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.00 [0.00, 0.01] | 98 / 2 / 0 |
| 1x | 200 (100) | 7.5% [3.0%, 13.0%] | 92.5% [87.0%, 97.0%] | 0.0% | 100.0% | 7.5% | 0.06 [0.03, 0.11] | 86 / 10 / 4 |
| 2x | 200 (100) | 28.5% [20.5%, 37.5%] | 71.5% [62.5%, 79.5%] | 0.0% | 100.0% | 28.5% | 0.26 [0.18, 0.34] | 63 / 16 / 21 |
| 5x | 200 (100) | 67.5% [58.5%, 76.5%] | 32.5% [23.5%, 41.5%] | 0.0% | 100.0% | 67.5% | 0.64 [0.56, 0.72] | 18 / 28 / 54 |
| 10x | 200 (100) | 89.0% [83.0%, 94.0%] | 11.0% [6.0%, 17.0%] | 0.0% | 100.0% | 89.0% | 0.86 [0.80, 0.91] | 2 / 18 / 80 |
| 20x | 200 (100) | 96.5% [92.5%, 99.5%] | 3.5% [0.5%, 7.5%] | 0.0% | 100.0% | 96.5% | 0.96 [0.92, 0.99] | 2 / 2 / 96 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +6.5 pts [+1.5, +12.5] |
| 1x to 2x | +21.0 pts [+11.5, +31.0] |
| 2x to 5x | +39.0 pts [+26.5, +51.5] |
| 5x to 10x | +21.5 pts [+11.0, +32.0] |
| 10x to 20x | +7.5 pts [+1.0, +14.0] |

### matched_seed2: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 5.0% (40) | 0.0% (40) | 7.5% (40) | 42.5% (40) | 85.0% (40) | 95.0% (40) |
| capital | 0.0% (40) | 7.5% (40) | 20.0% (40) | 75.0% (40) | 95.0% (40) | 100.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 17.5% (40) | 65.0% (40) | 95.0% (40) | 100.0% (40) |
| headquarters | 0.0% (40) | 5.0% (40) | 35.0% (40) | 75.0% (40) | 92.5% (40) | 90.0% (40) |
| occupation | 0.0% (40) | 25.0% (40) | 62.5% (40) | 80.0% (40) | 77.5% (40) | 97.5% (40) |

### matched_seed2: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 2.0% [0.0%, 6.0%] | 6.0% [0.0%, 14.0%] | 27.0% [16.0%, 39.0%] | 71.0% [59.0%, 82.0%] | 88.0% [80.0%, 95.0%] | 96.0% [90.0%, 100.0%] |
| final | 0.0% [0.0%, 0.0%] | 9.0% [2.0%, 17.0%] | 30.0% [18.0%, 42.0%] | 64.0% [51.0%, 77.0%] | 90.0% [82.0%, 97.0%] | 97.0% [92.0%, 100.0%] |

### matched_seed2: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 2/40 | 2/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 2/200 |

## base_decoding_v1: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 66.0% [57.5%, 74.5%] | 34.0% | 66.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 64.0% [55.5%, 72.5%] | 36.0% | 64.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 2x | 200 (100) | 0.0% [0.0%, 0.0%] | 65.0% [56.5%, 73.5%] | 35.0% | 65.0% | 0.0% | 0.00 [0.00, 0.00] | 99 / 1 / 0 |
| 5x | 200 (100) | 0.0% [0.0%, 0.0%] | 68.0% [59.0%, 76.0%] | 32.0% | 68.0% | 0.0% | 0.00 [0.00, 0.00] | 99 / 1 / 0 |
| 10x | 200 (100) | 0.0% [0.0%, 0.0%] | 67.5% [59.0%, 75.5%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 20x | 200 (100) | 0.0% [0.0%, 0.0%] | 67.5% [59.0%, 76.0%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +0.0 pts [+0.0, +0.0] |
| 1x to 2x | +0.0 pts [+0.0, +0.0] |
| 2x to 5x | +0.0 pts [+0.0, +0.0] |
| 5x to 10x | +0.0 pts [+0.0, +0.0] |
| 10x to 20x | +0.0 pts [+0.0, +0.0] |

### base_decoding_v1: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| capital | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| occupation | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |

### base_decoding_v1: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

### base_decoding_v1: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 0/200 |

