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

- `sft_matched`: `p1_main_v0_sft_matched.jsonl` (7200 rows, sha256 `7cd8579f375c`, grading ['placeholder_exact_v1'], decoding ['decoding_v1'])
- `base`: `results/raw/p1_main_v0_base.jsonl` (7200 rows, sha256 `0b28528d0f55`, grading ['placeholder_exact_v1'], decoding ['decoding_v0'])
- `sft_weak_recipe`: `results/raw/p1_main_v0_sft.jsonl` (7200 rows, sha256 `e7a6eae07bc0`, grading ['placeholder_exact_v1'], decoding ['decoding_v0'])

Dataset `main_v0`; bootstrap draws 2000, seed 0.

## sft_matched: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 99 / 1 / 0 |
| 1x | 200 (100) | 5.0% [1.5%, 9.0%] | 95.0% [91.0%, 98.5%] | 0.0% | 100.0% | 5.0% | 0.03 [0.01, 0.07] | 92 / 6 / 2 |
| 2x | 200 (100) | 26.5% [18.5%, 34.5%] | 73.5% [65.5%, 81.5%] | 0.0% | 100.0% | 26.5% | 0.22 [0.15, 0.29] | 64 / 20 / 16 |
| 5x | 200 (100) | 71.5% [63.0%, 79.5%] | 28.5% [20.5%, 37.0%] | 0.0% | 100.0% | 71.5% | 0.61 [0.53, 0.69] | 21 / 27 / 52 |
| 10x | 200 (100) | 87.5% [82.0%, 93.0%] | 12.5% [7.0%, 18.0%] | 0.0% | 100.0% | 87.5% | 0.79 [0.73, 0.85] | 8 / 20 / 72 |
| 20x | 200 (100) | 90.5% [85.0%, 95.0%] | 9.5% [5.0%, 15.0%] | 0.0% | 100.0% | 90.5% | 0.88 [0.82, 0.93] | 2 / 15 / 83 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.0 pts [+1.5, +9.0] |
| 1x to 2x | +21.5 pts [+12.5, +30.5] |
| 2x to 5x | +45.0 pts [+33.5, +56.0] |
| 5x to 10x | +16.0 pts [+6.0, +26.5] |
| 10x to 20x | +3.0 pts [-4.5, +10.0] |

### sft_matched: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 0.0% (40) | 7.5% (40) | 35.0% (40) | 62.5% (40) | 85.0% (40) |
| capital | 0.0% (40) | 5.0% (40) | 20.0% (40) | 85.0% (40) | 97.5% (40) | 100.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 25.0% (40) | 82.5% (40) | 92.5% (40) | 92.5% (40) |
| headquarters | 0.0% (40) | 2.5% (40) | 40.0% (40) | 85.0% (40) | 100.0% (40) | 90.0% (40) |
| occupation | 0.0% (40) | 17.5% (40) | 40.0% (40) | 70.0% (40) | 85.0% (40) | 85.0% (40) |

### sft_matched: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 6.0% [1.0%, 13.0%] | 27.0% [15.0%, 40.0%] | 71.0% [60.0%, 82.0%] | 86.0% [77.0%, 94.0%] | 88.0% [80.0%, 95.0%] |
| final | 0.0% [0.0%, 0.0%] | 4.0% [0.0%, 10.0%] | 26.0% [15.0%, 38.0%] | 72.0% [59.0%, 83.0%] | 89.0% [81.0%, 96.0%] | 93.0% [85.0%, 99.0%] |

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
| 5x | -0.140 (n=143) | -0.562 (n=57) |
| 10x | -0.088 (n=175) | -0.497 (n=25) |
| 20x | -0.058 (n=181) | -0.386 (n=19) |

## base: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 97.0% [94.0%, 99.5%] | 3.0% | 97.0% | 0.0% | 0.00 [0.00, 0.00] | 99 / 1 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 95.5% [92.5%, 98.0%] | 4.5% | 95.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 2x | 200 (100) | 0.5% [0.0%, 1.5%] | 95.5% [91.5%, 98.5%] | 4.0% | 96.0% | 0.5% | 0.00 [0.00, 0.01] | 99 / 1 / 0 |
| 5x | 200 (100) | 0.0% [0.0%, 0.0%] | 96.0% [93.0%, 98.5%] | 4.0% | 96.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 10x | 200 (100) | 0.5% [0.0%, 1.5%] | 95.5% [92.5%, 98.0%] | 4.0% | 96.0% | 0.5% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 20x | 200 (100) | 0.5% [0.0%, 1.5%] | 95.5% [92.5%, 98.0%] | 4.0% | 96.0% | 0.5% | 0.00 [0.00, 0.00] | 99 / 1 / 0 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +0.0 pts [+0.0, +0.0] |
| 1x to 2x | +0.5 pts [+0.0, +1.5] |
| 2x to 5x | -0.5 pts [-1.5, +0.0] |
| 5x to 10x | +0.5 pts [+0.0, +1.5] |
| 10x to 20x | +0.0 pts [-1.5, +1.5] |

### base: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 0.0% (40) | 2.5% (40) | 0.0% (40) | 0.0% (40) | 2.5% (40) |
| capital | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) |
| occupation | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 2.5% (40) | 0.0% (40) |

### base: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 1.0% [0.0%, 3.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 1.0% [0.0%, 3.0%] | 1.0% [0.0%, 3.0%] |

### base: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 1/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 0/200 |

### base: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.581 (n=194) |
| 1x | n/a | -0.579 (n=191) |
| 2x | -0.467 (n=1) | -0.598 (n=191) |
| 5x | n/a | -0.590 (n=192) |
| 10x | -0.781 (n=1) | -0.615 (n=191) |
| 20x | -1.175 (n=1) | -0.609 (n=191) |

## sft_weak_recipe: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 1.0% [0.0%, 3.0%] | 99.0% [97.0%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.01 [0.00, 0.03] | 97 / 3 / 0 |
| 1x | 200 (100) | 0.5% [0.0%, 1.5%] | 99.5% [98.5%, 100.0%] | 0.0% | 100.0% | 0.5% | 0.01 [0.00, 0.02] | 96 / 4 / 0 |
| 2x | 200 (100) | 2.0% [0.0%, 4.5%] | 98.0% [95.5%, 100.0%] | 0.0% | 100.0% | 2.0% | 0.01 [0.00, 0.03] | 97 / 3 / 0 |
| 5x | 200 (100) | 2.0% [0.0%, 4.5%] | 98.0% [95.5%, 100.0%] | 0.0% | 100.0% | 2.0% | 0.01 [0.00, 0.04] | 97 / 2 / 1 |
| 10x | 200 (100) | 6.5% [2.5%, 11.0%] | 93.5% [89.0%, 97.5%] | 0.0% | 100.0% | 6.5% | 0.05 [0.02, 0.09] | 86 / 12 / 2 |
| 20x | 200 (100) | 37.0% [29.0%, 45.5%] | 63.0% [54.5%, 71.0%] | 0.0% | 100.0% | 37.0% | 0.27 [0.20, 0.33] | 43 / 44 / 13 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | -0.5 pts [-3.0, +1.5] |
| 1x to 2x | +1.5 pts [-0.5, +4.5] |
| 2x to 5x | +0.0 pts [-3.5, +3.5] |
| 5x to 10x | +4.5 pts [-0.5, +9.5] |
| 10x to 20x | +30.5 pts [+21.5, +39.5] |

### sft_weak_recipe: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 5.0% (40) | 12.5% (40) |
| capital | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 7.5% (40) | 55.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 0.0% (40) | 7.5% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 0.0% (40) | 2.5% (40) | 10.0% (40) | 57.5% (40) |
| occupation | 5.0% (40) | 2.5% (40) | 10.0% (40) | 7.5% (40) | 10.0% (40) | 52.5% (40) |

### sft_weak_recipe: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 1.0% [0.0%, 3.0%] | 0.0% [0.0%, 0.0%] | 3.0% [0.0%, 8.0%] | 10.0% [3.0%, 18.0%] | 41.0% [29.0%, 53.0%] |
| final | 2.0% [0.0%, 6.0%] | 0.0% [0.0%, 0.0%] | 4.0% [0.0%, 9.0%] | 1.0% [0.0%, 3.0%] | 3.0% [0.0%, 8.0%] | 33.0% [22.0%, 45.0%] |

### sft_weak_recipe: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 0/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 2/40 | 12/200 |

### sft_weak_recipe: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | -0.832 (n=2) | -1.214 (n=198) |
| 1x | -1.577 (n=1) | -1.269 (n=199) |
| 2x | -0.775 (n=4) | -1.273 (n=196) |
| 5x | -0.617 (n=4) | -1.251 (n=196) |
| 10x | -0.683 (n=13) | -1.204 (n=187) |
| 20x | -0.453 (n=74) | -1.175 (n=126) |

