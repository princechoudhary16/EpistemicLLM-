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

- `matched_seed0`: `responses__mine__frozen.jsonl` (7200 rows, sha256 `4085a279c4b4`, grading ['p2_whole_word_v5'], decoding ['not recorded'])
- `base_decoding_v1`: `responses__base__frozen.jsonl` (7200 rows, sha256 `5d843c78f735`, grading ['p2_whole_word_v5'], decoding ['not recorded'])

Dataset `main_w1`, all facts (dev and final); bootstrap draws 2000, seed 0.

## matched_seed0: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 100.0% | 0.0% | 0.01 [0.00, 0.01] | 96 / 4 / 0 |
| 1x | 200 (100) | 7.5% [3.0%, 13.0%] | 92.5% [87.0%, 97.0%] | 0.0% | 100.0% | 7.5% | 0.05 [0.02, 0.09] | 88 / 9 / 3 |
| 2x | 200 (100) | 14.5% [8.5%, 21.5%] | 85.5% [78.5%, 91.5%] | 0.0% | 100.0% | 14.5% | 0.16 [0.11, 0.22] | 63 / 29 / 8 |
| 5x | 200 (100) | 65.5% [56.5%, 74.5%] | 34.5% [25.5%, 43.5%] | 0.0% | 100.0% | 65.5% | 0.56 [0.48, 0.63] | 16 / 42 / 42 |
| 10x | 200 (100) | 91.5% [86.0%, 96.5%] | 8.5% [3.5%, 14.0%] | 0.0% | 100.0% | 91.5% | 0.85 [0.79, 0.90] | 7 / 11 / 82 |
| 20x | 200 (100) | 97.5% [94.0%, 100.0%] | 2.5% [0.0%, 6.0%] | 0.0% | 100.0% | 97.5% | 0.95 [0.91, 0.98] | 1 / 7 / 92 |

Change in accuracy between adjacent exposure levels (where the curve is steepest):

| Step | Change [95% CI] |
|---|---|
| 0x to 1x | +7.5 pts [+3.0, +13.0] |
| 1x to 2x | +7.0 pts [-1.0, +16.0] |
| 2x to 5x | +51.0 pts [+39.0, +61.5] |
| 5x to 10x | +26.0 pts [+15.0, +36.5] |
| 10x to 20x | +6.0 pts [+0.0, +12.0] |

### matched_seed0: accuracy by relation (questions per cell in brackets)

| Relation | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| birth_year | 0.0% (40) | 5.0% (40) | 5.0% (40) | 35.0% (40) | 85.0% (40) | 90.0% (40) |
| capital | 0.0% (40) | 0.0% (40) | 5.0% (40) | 60.0% (40) | 92.5% (40) | 100.0% (40) |
| founder | 0.0% (40) | 0.0% (40) | 0.0% (40) | 60.0% (40) | 95.0% (40) | 100.0% (40) |
| headquarters | 0.0% (40) | 0.0% (40) | 20.0% (40) | 77.5% (40) | 90.0% (40) | 100.0% (40) |
| occupation | 0.0% (40) | 32.5% (40) | 42.5% (40) | 95.0% (40) | 95.0% (40) | 97.5% (40) |

### matched_seed0: dev facts (validation wording) vs final facts (test wording)

Dev and final differ in both the facts and the question wording, so this is a joint check; the held-out paraphrase run separates the two.

| Split | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| dev | 0.0% [0.0%, 0.0%] | 11.0% [4.0%, 20.0%] | 16.0% [7.0%, 26.0%] | 69.0% [57.0%, 81.0%] | 86.0% [76.0%, 94.0%] | 95.0% [88.0%, 100.0%] |
| final | 0.0% [0.0%, 0.0%] | 4.0% [0.0%, 10.0%] | 13.0% [5.0%, 22.0%] | 62.0% [49.0%, 74.0%] | 97.0% [92.0%, 100.0%] | 100.0% [100.0%, 100.0%] |

### matched_seed0: chance floor (correct answers at 0x, where the model has seen nothing)

| Relation | Greedy | Sampled |
|---|---|---|
| birth_year | 0/40 | 3/200 |
| capital | 0/40 | 0/200 |
| founder | 0/40 | 0/200 |
| headquarters | 0/40 | 0/200 |
| occupation | 0/40 | 2/200 |

## base_decoding_v1: behaviour by exposure level

| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |
|---|---|---|---|---|---|---|---|---|
| 0x | 200 (100) | 0.0% [0.0%, 0.0%] | 65.5% [56.5%, 74.0%] | 34.5% | 65.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 1x | 200 (100) | 0.0% [0.0%, 0.0%] | 68.0% [59.5%, 76.0%] | 32.0% | 68.0% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 2x | 200 (100) | 0.0% [0.0%, 0.0%] | 67.5% [59.5%, 76.0%] | 32.5% | 67.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 5x | 200 (100) | 0.0% [0.0%, 0.0%] | 64.5% [56.0%, 72.5%] | 35.5% | 64.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |
| 10x | 200 (100) | 0.0% [0.0%, 0.0%] | 66.0% [57.5%, 74.0%] | 34.0% | 66.0% | 0.0% | 0.00 [0.00, 0.00] | 99 / 1 / 0 |
| 20x | 200 (100) | 0.0% [0.0%, 0.0%] | 65.5% [57.5%, 73.5%] | 34.5% | 65.5% | 0.0% | 0.00 [0.00, 0.00] | 100 / 0 / 0 |

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

