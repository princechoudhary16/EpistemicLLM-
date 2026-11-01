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

- `sft_matched`: `results/raw/p1_pilot_v0_sft_matched__p2_whole_word_v5.jsonl` (1440 rows, sha256 `7cbe19db7bb5`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `base`: `results/raw/p1_pilot_v0_base__p2_whole_word_v5.jsonl` (1440 rows, sha256 `9629ae33c0a8`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])
- `sft_weak_recipe`: `results/raw/p1_pilot_v0_sft__p2_whole_word_v5.jsonl` (1440 rows, sha256 `ba76cc90afb1`, grading ['p2_whole_word_v5'], decoding ['decoding_v1'])

Dataset `pilot_v0`, all facts (dev and final); bootstrap draws 2000, seed 0.

## sft_matched: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 40 (20) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 1x | 40 (20) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 2x | 40 (20) | 27.5% [10.0%, 47.5%] | 72.5% [52.5%, 90.0%] | 0.0% | 100.0% | 27.5% | 0.18 [0.05, 0.34] | 14 / 3 / 3 |
| 5x | 40 (20) | 57.5% [35.0%, 80.0%] | 42.5% [20.0%, 65.0%] | 0.0% | 100.0% | 57.5% | 0.49 [0.31, 0.67] | 6 / 6 / 8 |
| 10x | 40 (20) | 82.5% [65.0%, 95.0%] | 17.5% [5.0%, 35.0%] | 0.0% | 100.0% | 82.5% | 0.73 [0.57, 0.87] | 1 / 7 / 12 |
| 20x | 40 (20) | 92.5% [84.9%, 100.0%] | 7.5% [0.0%, 15.1%] | 0.0% | 100.0% | 92.5% | 0.91 [0.83, 0.97] | 0 / 4 / 16 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +0.0 pts [+0.0, +0.0] |
| 1x to 2x | +27.5 pts [+10.0, +47.5] |
| 2x to 5x | +30.0 pts [-2.5, +57.5] |
| 5x to 10x | +25.0 pts [-2.5, +52.5] |
| 10x to 20x | +10.0 pts [-7.5, +27.5] |

### sft_matched: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (8) | 0.0% (8) | 0.0% (8) | 25.0% (8) | 50.0% (8) | 87.5% (8) |
| capital | 0.0% (8) | 0.0% (8) | 12.5% (8) | 50.0% (8) | 100.0% (8) | 87.5% (8) |
| founder | 0.0% (8) | 0.0% (8) | 0.0% (8) | 50.0% (8) | 100.0% (8) | 100.0% (8) |
| headquarters | 0.0% (8) | 0.0% (8) | 75.0% (8) | 75.0% (8) | 75.0% (8) | 100.0% (8) |
| occupation | 0.0% (8) | 0.0% (8) | 50.0% (8) | 87.5% (8) | 87.5% (8) | 87.5% (8) |

### sft_matched: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 20.0% [0.0%, 50.0%] | 45.0% [15.0%, 75.0%] | 65.0% [35.0%, 90.0%] | 95.0% [85.0%, 100.0%] |
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 35.0% [10.0%, 65.0%] | 70.0% [40.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 90.0% [75.0%, 100.0%] |

### sft_matched: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/8 | 0/40 |
| capital | 0/8 | 0/40 |
| founder | 0/8 | 0/40 |
| headquarters | 0/8 | 0/40 |
| occupation | 0/8 | 0/40 |

### sft_matched: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.557 (n=40) |
| 1x | n/a | -0.608 (n=40) |
| 2x | -0.496 (n=11) | -0.703 (n=29) |
| 5x | -0.209 (n=23) | -0.482 (n=17) |
| 10x | -0.127 (n=33) | -0.439 (n=7) |
| 20x | -0.066 (n=37) | -0.628 (n=3) |

## base: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 40 (20) | 0.0% [0.0%, 0.0%] | 65.0% [45.0%, 82.5%] | 35.0% | 65.0% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 1x | 40 (20) | 0.0% [0.0%, 0.0%] | 67.5% [50.0%, 85.0%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 2x | 40 (20) | 0.0% [0.0%, 0.0%] | 67.5% [50.0%, 85.0%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 5x | 40 (20) | 0.0% [0.0%, 0.0%] | 72.5% [55.0%, 87.5%] | 27.5% | 72.5% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 10x | 40 (20) | 0.0% [0.0%, 0.0%] | 65.0% [45.0%, 85.0%] | 35.0% | 65.0% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 20x | 40 (20) | 0.0% [0.0%, 0.0%] | 80.0% [65.0%, 92.5%] | 20.0% | 80.0% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |

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
| birth_year | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) |
| capital | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) |
| founder | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) |
| headquarters | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) |
| occupation | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) |

### base: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

### base: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/8 | 0/40 |
| capital | 0/8 | 0/40 |
| founder | 0/8 | 0/40 |
| headquarters | 0/8 | 0/40 |
| occupation | 0/8 | 0/40 |

### base: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -0.442 (n=26) |
| 1x | n/a | -0.519 (n=27) |
| 2x | n/a | -0.501 (n=27) |
| 5x | n/a | -0.511 (n=29) |
| 10x | n/a | -0.516 (n=26) |
| 20x | n/a | -0.456 (n=32) |

## sft_weak_recipe: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 40 (20) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 1x | 40 (20) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.00 [0.00, 0.00] | 20 / 0 / 0 |
| 2x | 40 (20) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.01 [0.00, 0.02] | 19 / 1 / 0 |
| 5x | 40 (20) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.01 [0.00, 0.02] | 19 / 1 / 0 |
| 10x | 40 (20) | 35.0% [17.5%, 55.0%] | 65.0% [45.0%, 82.5%] | 0.0% | 100.0% | 35.0% | 0.17 [0.07, 0.31] | 11 / 7 / 2 |
| 20x | 40 (20) | 62.5% [42.5%, 82.5%] | 37.5% [17.5%, 57.5%] | 0.0% | 100.0% | 62.5% | 0.57 [0.39, 0.74] | 4 / 7 / 9 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +0.0 pts [+0.0, +0.0] |
| 1x to 2x | +0.0 pts [+0.0, +0.0] |
| 2x to 5x | +0.0 pts [+0.0, +0.0] |
| 5x to 10x | +35.0 pts [+17.5, +55.0] |
| 10x to 20x | +27.5 pts [+0.0, +55.0] |

### sft_weak_recipe: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 50.0% (8) |
| capital | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 50.0% (8) | 50.0% (8) |
| founder | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 25.0% (8) |
| headquarters | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 62.5% (8) | 100.0% (8) |
| occupation | 0.0% (8) | 0.0% (8) | 0.0% (8) | 0.0% (8) | 62.5% (8) | 87.5% (8) |

### sft_weak_recipe: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 35.0% [10.0%, 65.0%] | 75.0% [45.0%, 100.0%] |
| final | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 35.0% [10.0%, 60.0%] | 50.0% [25.0%, 80.0%] |

### sft_weak_recipe: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/8 | 0/40 |
| capital | 0/8 | 0/40 |
| founder | 0/8 | 0/40 |
| headquarters | 0/8 | 0/40 |
| occupation | 0/8 | 0/40 |

### sft_weak_recipe: mean confidence (average log-probability of the greedy answer)

| Exposure | Correct answers | Incorrect answers |
|---|---|---|
| 0x | n/a | -1.149 (n=40) |
| 1x | n/a | -0.996 (n=40) |
| 2x | n/a | -1.094 (n=40) |
| 5x | n/a | -1.163 (n=40) |
| 10x | -0.767 (n=14) | -1.202 (n=26) |
| 20x | -0.299 (n=25) | -1.069 (n=15) |

