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

- `base`: `results/raw/p1_main_v0_base.jsonl` (7200 rows, sha256 `424d880d0e10`, grading ['placeholder_exact_v1'], decoding ['decoding_v1'])
- `sft`: `results/raw/p1_main_v0_sft.jsonl` (7200 rows, sha256 `9181944754d3`, grading ['placeholder_exact_v1'], decoding ['decoding_v1'])
- `idk`: `results/raw/p1_main_v0_idk.jsonl` (7200 rows, sha256 `9287ab34478c`, grading ['placeholder_exact_v1'], decoding ['decoding_v1'])
- `sft_matched`: `results/raw/p1_main_v0_sft_matched.jsonl` (7200 rows, sha256 `e1d92b057ced`, grading ['placeholder_exact_v1'], decoding ['decoding_v1'])
- `p5_lam0`: `results/raw/p5_main_v0_lam0_seed0.jsonl` (7200 rows, sha256 `e724ec13b161`, grading ['placeholder_exact_v1'], decoding ['decoding_v1'])
- `p5_lam3`: `results/raw/p5_main_v0_lam3_seed0.jsonl` (7200 rows, sha256 `28db9d37a74e`, grading ['placeholder_exact_v1'], decoding ['decoding_v1'])

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
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.01] | 99 / 1 / 0 |
| 1x | 200 (100) | 5.0% [1.5%, 9.0%] | 95.0% [91.0%, 98.5%] | 0.0% | 100.0% | 5.0% | 0.03 [0.01, 0.07] | 92 / 6 / 2 |
| 2x | 200 (100) | 26.5% [18.5%, 35.0%] | 73.5% [65.0%, 81.5%] | 0.0% | 100.0% | 26.5% | 0.22 [0.16, 0.30] | 64 / 20 / 16 |
| 5x | 200 (100) | 71.5% [63.0%, 79.5%] | 28.5% [20.5%, 37.0%] | 0.0% | 100.0% | 71.5% | 0.61 [0.53, 0.69] | 21 / 27 / 52 |
| 10x | 200 (100) | 87.5% [81.5%, 93.0%] | 12.5% [7.0%, 18.5%] | 0.0% | 100.0% | 87.5% | 0.79 [0.72, 0.86] | 8 / 20 / 72 |
| 20x | 200 (100) | 90.5% [85.5%, 95.0%] | 9.5% [5.0%, 14.5%] | 0.0% | 100.0% | 90.5% | 0.88 [0.83, 0.93] | 2 / 15 / 83 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.0 pts [+1.5, +9.0] |
| 1x to 2x | +21.5 pts [+12.5, +31.0] |
| 2x to 5x | +45.0 pts [+33.0, +56.0] |
| 5x to 10x | +16.0 pts [+6.0, +26.0] |
| 10x to 20x | +3.0 pts [-4.5, +10.5] |

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
| dev | 0.0% [0.0%, 0.0%] | 6.0% [1.0%, 13.0%] | 27.0% [16.0%, 39.0%] | 71.0% [59.0%, 82.0%] | 86.0% [77.0%, 94.0%] | 88.0% [80.0%, 95.0%] |
| final | 0.0% [0.0%, 0.0%] | 4.0% [0.0%, 9.0%] | 26.0% [15.0%, 38.0%] | 72.0% [60.0%, 84.0%] | 89.0% [81.0%, 96.0%] | 93.0% [86.0%, 99.0%] |

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

## p5_lam0: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.01 [0.00, 0.02] | 98 / 2 / 0 |
| 1x | 200 (100) | 8.5% [3.5%, 14.0%] | 91.5% [86.0%, 96.5%] | 0.0% | 100.0% | 8.5% | 0.08 [0.04, 0.13] | 84 / 11 / 5 |
| 2x | 200 (100) | 25.5% [17.5%, 34.0%] | 74.5% [66.0%, 82.5%] | 0.0% | 100.0% | 25.5% | 0.21 [0.15, 0.28] | 63 / 24 / 13 |
| 5x | 200 (100) | 74.5% [66.0%, 82.5%] | 25.5% [17.5%, 34.0%] | 0.0% | 100.0% | 74.5% | 0.67 [0.59, 0.75] | 14 / 27 / 59 |
| 10x | 200 (100) | 89.5% [83.5%, 95.0%] | 10.5% [5.0%, 16.5%] | 0.0% | 100.0% | 89.5% | 0.85 [0.79, 0.91] | 8 / 12 / 80 |
| 20x | 200 (100) | 94.0% [89.0%, 98.0%] | 6.0% [2.0%, 11.0%] | 0.0% | 100.0% | 94.0% | 0.93 [0.88, 0.97] | 5 / 3 / 92 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +8.5 pts [+3.5, +14.0] |
| 1x to 2x | +17.0 pts [+7.5, +27.0] |
| 2x to 5x | +49.0 pts [+37.5, +60.5] |
| 5x to 10x | +15.0 pts [+5.0, +25.5] |
| 10x to 20x | +4.5 pts [-3.0, +12.0] |

### p5_lam0: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 5.0% (40) | 0.0% (40) | 55.0% (40) | 80.0% (40) | 95.0% (40) |
| capital | 0.0% (40) | 7.5% (40) | 22.5% (40) | 77.5% (40) | 90.0% (40) | 90.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 10.0% (40) | 77.5% (40) | 90.0% (40) | 95.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 40.0% (40) | 77.5% (40) | 92.5% (40) | 90.0% (40) |
| occupation | 0.0% (40) | 30.0% (40) | 55.0% (40) | 85.0% (40) | 95.0% (40) | 100.0% (40) |

### p5_lam0: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 10.0% [3.0%, 18.0%] | 24.0% [13.0%, 35.0%] | 81.0% [70.0%, 91.0%] | 90.0% [82.0%, 97.0%] | 92.0% [84.0%, 98.0%] |
| final | 0.0% [0.0%, 0.0%] | 7.0% [1.0%, 14.0%] | 27.0% [15.0%, 40.0%] | 68.0% [55.0%, 80.0%] | 89.0% [80.0%, 97.0%] | 96.0% [90.0%, 100.0%] |

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
| 0x | n/a | -0.543 (n=200) |
| 1x | -0.293 (n=17) | -0.618 (n=183) |
| 2x | -0.303 (n=51) | -0.613 (n=149) |
| 5x | -0.125 (n=149) | -0.544 (n=51) |
| 10x | -0.066 (n=179) | -0.358 (n=21) |
| 20x | -0.027 (n=188) | -0.370 (n=12) |

## p5_lam3: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 1.0% [0.0%, 2.5%] | 99.0% [97.5%, 100.0%] | 0.0% | 100.0% | 1.0% | 0.00 [0.00, 0.01] | 98 / 2 / 0 |
| 1x | 200 (100) | 6.5% [3.0%, 11.0%] | 93.5% [89.0%, 97.0%] | 0.0% | 100.0% | 6.5% | 0.04 [0.02, 0.07] | 84 / 15 / 1 |
| 2x | 200 (100) | 20.5% [13.0%, 28.5%] | 79.5% [71.5%, 87.0%] | 0.0% | 100.0% | 20.5% | 0.18 [0.12, 0.25] | 69 / 20 / 11 |
| 5x | 200 (100) | 75.5% [67.5%, 83.5%] | 24.5% [16.5%, 32.5%] | 0.0% | 100.0% | 75.5% | 0.66 [0.58, 0.73] | 14 / 28 / 58 |
| 10x | 200 (100) | 92.5% [87.5%, 97.0%] | 7.5% [3.0%, 12.5%] | 0.0% | 100.0% | 92.5% | 0.88 [0.82, 0.92] | 4 / 11 / 85 |
| 20x | 200 (100) | 95.5% [91.0%, 99.0%] | 4.5% [1.0%, 9.0%] | 0.0% | 100.0% | 95.5% | 0.95 [0.91, 0.99] | 4 / 1 / 95 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +5.5 pts [+1.5, +10.0] |
| 1x to 2x | +14.0 pts [+5.5, +23.0] |
| 2x to 5x | +55.0 pts [+43.5, +66.0] |
| 5x to 10x | +17.0 pts [+8.0, +26.5] |
| 10x to 20x | +3.0 pts [-3.5, +9.0] |

### p5_lam3: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 2.5% (40) | 0.0% (40) | 0.0% (40) | 55.0% (40) | 82.5% (40) | 95.0% (40) |
| capital | 0.0% (40) | 5.0% (40) | 5.0% (40) | 72.5% (40) | 97.5% (40) | 95.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 12.5% (40) | 70.0% (40) | 100.0% (40) | 100.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 35.0% (40) | 90.0% (40) | 87.5% (40) | 90.0% (40) |
| occupation | 2.5% (40) | 27.5% (40) | 50.0% (40) | 90.0% (40) | 95.0% (40) | 97.5% (40) |

### p5_lam3: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 6.0% [1.0%, 12.0%] | 15.0% [6.0%, 26.0%] | 80.0% [69.0%, 90.0%] | 92.0% [85.0%, 98.0%] | 97.0% [92.0%, 100.0%] |
| final | 2.0% [0.0%, 5.0%] | 7.0% [2.0%, 14.0%] | 26.0% [14.0%, 38.0%] | 71.0% [59.0%, 82.0%] | 93.0% [86.0%, 99.0%] | 94.0% [86.0%, 100.0%] |

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
| 0x | -0.603 (n=2) | -0.615 (n=198) |
| 1x | -0.555 (n=13) | -0.688 (n=187) |
| 2x | -0.229 (n=41) | -0.656 (n=159) |
| 5x | -0.146 (n=151) | -0.502 (n=49) |
| 10x | -0.057 (n=185) | -0.373 (n=15) |
| 20x | -0.015 (n=191) | -0.280 (n=9) |

