# P1 ablation summary

Greedy accuracy by exposure level, overall abstention, K buckets over facts (sampled), and birth_year accuracy split by whether the answer has a repeated digit.

| run | decoding | greedy acc | 0x | 1x | 2x | 5x | 10x | 20x | abstain | K=0 / inconsistent / K>=0.8 | repeat-digit yr | no-repeat yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mine-idk_end | frozen | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 120 / 0 / 0 | 0/10 | 0/14 |
| mine-idk_end | inherit | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 120 / 0 / 0 | 0/10 | 0/14 |
| mine-idk_shuffled | frozen | 36.2% | 0.0% | 0.0% | 5.0% | 35.0% | 82.5% | 95.0% | 8.3% | 71 / 14 / 35 | 8/10 | 2/14 |
| mine-idk_shuffled | inherit | 35.8% | 0.0% | 0.0% | 5.0% | 37.5% | 77.5% | 95.0% | 0.4% | 71 / 14 / 35 | 6/10 | 2/14 |
| mine | frozen | 40.8% | 0.0% | 0.0% | 15.0% | 57.5% | 82.5% | 90.0% | 0.0% | 60 / 18 / 42 | 3/10 | 4/14 |
| mine | inherit | 40.4% | 0.0% | 0.0% | 15.0% | 57.5% | 82.5% | 87.5% | 0.0% | 61 / 17 / 42 | 2/10 | 4/14 |
| p6-idk_end | frozen | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 120 / 0 / 0 | 0/10 | 0/14 |
| p6-idk_end | inherit | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 120 / 0 / 0 | 0/10 | 0/14 |
| p6-idk_shuffled | frozen | 12.9% | 0.0% | 0.0% | 0.0% | 2.5% | 17.5% | 57.5% | 1.2% | 93 / 21 / 6 | 5/10 | 0/14 |
| p6-idk_shuffled | inherit | 10.8% | 0.0% | 0.0% | 0.0% | 2.5% | 15.0% | 47.5% | 0.0% | 94 / 20 / 6 | 0/10 | 1/14 |
| p6 | frozen | 15.4% | 0.0% | 2.5% | 0.0% | 0.0% | 25.0% | 65.0% | 0.0% | 91 / 22 / 7 | 4/10 | 1/14 |
| p6 | inherit | 15.4% | 0.0% | 2.5% | 0.0% | 0.0% | 22.5% | 67.5% | 0.0% | 93 / 22 / 5 | 3/10 | 2/14 |
| p6_all | frozen | 46.2% | 0.0% | 10.0% | 32.5% | 55.0% | 80.0% | 100.0% | 0.0% | 56 / 22 / 42 | 8/10 | 6/14 |
| p6_all | inherit | 44.6% | 0.0% | 10.0% | 32.5% | 50.0% | 75.0% | 100.0% | 0.0% | 56 / 22 / 42 | 5/10 | 6/14 |
| p6_bs4 | frozen | 2.9% | 0.0% | 0.0% | 0.0% | 0.0% | 7.5% | 10.0% | 0.0% | 112 / 8 / 0 | 1/10 | 1/14 |
| p6_bs4 | inherit | 2.9% | 0.0% | 0.0% | 0.0% | 0.0% | 5.0% | 12.5% | 0.0% | 111 / 9 / 0 | 0/10 | 2/14 |
| p6_r16 | frozen | 20.4% | 0.0% | 0.0% | 0.0% | 10.0% | 40.0% | 72.5% | 0.0% | 83 / 18 / 19 | 4/10 | 0/14 |
| p6_r16 | inherit | 21.2% | 0.0% | 0.0% | 0.0% | 10.0% | 47.5% | 70.0% | 0.0% | 85 / 17 / 18 | 3/10 | 0/14 |

## Chance floor at 0x (correct / total generations, by relation)

| run | decoding | birth_year | capital | founder | headquarters | occupation |
|---|---|---|---|---|---|---|
| mine-idk_end | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| mine-idk_end | inherit | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| mine-idk_shuffled | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| mine-idk_shuffled | inherit | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| mine | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| mine | inherit | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6-idk_end | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6-idk_end | inherit | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6-idk_shuffled | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6-idk_shuffled | inherit | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6 | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6 | inherit | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6_all | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6_all | inherit | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6_bs4 | frozen | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6_bs4 | inherit | 0/8 g, 1/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6_r16 | frozen | 0/8 g, 1/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |
| p6_r16 | inherit | 0/8 g, 1/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s | 0/8 g, 0/40 s |

g = greedy, s = sampled. Grader: P6's substring match; `grader_disagreements` in summary.json counts rows where word-boundary matching differs.
