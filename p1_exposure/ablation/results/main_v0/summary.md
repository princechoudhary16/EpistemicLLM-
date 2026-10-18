# P1 ablation summary

Greedy accuracy by exposure level, overall abstention, K buckets over facts (sampled), and birth_year accuracy split by whether the answer has a repeated digit.

| run | decoding | greedy acc | 0x | 1x | 2x | 5x | 10x | 20x | abstain | K=0 / inconsistent / K>=0.8 | repeat-digit yr | no-repeat yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p6 | frozen | 9.6% | 1.0% | 1.0% | 2.0% | 3.5% | 12.5% | 37.5% | 0.0% | 489 / 88 / 23 | 3/48 | 8/72 |
| p6 | inherit | 9.5% | 1.0% | 1.0% | 2.0% | 3.5% | 12.5% | 37.0% | 0.0% | 492 / 81 / 27 | 0/48 | 8/72 |
| p6_all | frozen | 49.0% | 1.0% | 10.0% | 31.0% | 72.0% | 87.5% | 92.5% | 0.0% | 270 / 81 / 249 | 37/48 | 48/72 |
| p6_all | inherit | 48.2% | 0.0% | 10.0% | 30.5% | 70.0% | 86.0% | 92.5% | 0.0% | 272 / 81 / 247 | 26/48 | 50/72 |

## Chance floor at 0x (correct / total generations, by relation)

| run | decoding | birth_year | capital | founder | headquarters | occupation |
|---|---|---|---|---|---|---|
| p6 | frozen | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 2/40 g, 8/200 s |
| p6 | inherit | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 2/40 g, 8/200 s |
| p6_all | frozen | 2/40 g, 3/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s |
| p6_all | inherit | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s | 0/40 g, 0/200 s |

g = greedy, s = sampled. Grader: P6's substring match; `grader_disagreements` in summary.json counts rows where word-boundary matching differs.
