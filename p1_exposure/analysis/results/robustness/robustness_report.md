# P1 robustness report

**Terms.** *Exposure*: how many times a fact appeared in the fine-tuning data (0 means never).
*Accuracy*: share of greedy answers that contain the reference answer. *Training seed*: the random
seed for the adapter's initial weights and dropout; the training data and its order are unchanged.
*World*: one complete synthetic dataset (invented names, answers, and which facts get which exposure);
world 1 is generated with a different generator seed, so it has different facts and a different
fact-to-exposure assignment. *Held-out wording*: two new question wordings per relation that appear in no
training, validation or test template. *Swapped wording*: each fact asked with the other split's
wording (dev facts get the test wordings, final facts get the validation wordings). *Paired*: the same
facts are compared under two wordings, and the interval is on the per-fact difference.
*95% CI*: fact-level bootstrap interval.

Run `mine` (matched recipe unless stated), decoding `frozen`, 4000 bootstrap draws, seed 0.

- `main_v0` inputs: `out_v0_s0/responses__mine__frozen.jsonl` (sha256 `c77c80cc6537`), `out_v0_s1/responses__mine__frozen.jsonl` (sha256 `10887bf659a8`), `out_v0_s2/responses__mine__frozen.jsonl` (sha256 `696cef9d4fd7`)
- `main_w1` inputs: `out_w1_s0/responses__mine__frozen.jsonl` (sha256 `d7e08f03277b`)

## World `main_v0`: training seeds (3 seeds)

| Exposure | seed 0 | seed 1 | seed 2 | Mean [95% CI over facts] | Spread across seeds (max minus min) |
|---|---|---|---|---|---|
| 0x | 0.0% | 0.0% | 1.0% | 0.3% [0.0%, 1.0%] | 1.0 pts |
| 1x | 6.5% | 7.0% | 7.5% | 7.0% [3.5%, 11.3%] | 1.0 pts |
| 2x | 23.0% | 28.0% | 30.5% | 27.2% [20.0%, 34.5%] | 7.5 pts |
| 5x | 62.5% | 70.0% | 67.5% | 66.7% [59.7%, 73.3%] | 7.5 pts |
| 10x | 79.5% | 88.0% | 89.0% | 85.5% [81.3%, 89.5%] | 9.5 pts |
| 20x | 96.0% | 94.5% | 96.5% | 95.7% [92.7%, 98.2%] | 2.0 pts |

### `main_v0`: heldout wording vs standard wording (paired by fact, averaged over seeds)

| Exposure | Standard | Heldout | Difference [95% CI] |
|---|---|---|---|
| 0x | 0.3% | 0.2% | -0.2 pts [-0.5, +0.0] |
| 1x | 7.0% | 7.8% | +0.8 pts [-1.0, +3.0] |
| 2x | 27.2% | 28.5% | +1.3 pts [-0.3, +3.2] |
| 5x | 66.7% | 69.8% | +3.2 pts [+0.8, +5.5] |
| 10x | 85.5% | 89.0% | +3.5 pts [+1.2, +6.2] |
| 20x | 95.7% | 95.7% | -0.0 pts [-1.0, +0.8] |
| all facts at 2x and above | 68.8% | 70.8% | +2.0 pts [+1.0, +3.0] |
| dev facts at 2x and above (n=200) | 68.4% | 71.7% | +3.2 pts [+1.7, +5.0] |
| final facts at 2x and above (n=200) | 69.1% | 69.8% | +0.8 pts [-0.3, +1.8] |

### `main_v0`: swapped wording vs standard wording (paired by fact, averaged over seeds)

| Exposure | Standard | Swapped | Difference [95% CI] |
|---|---|---|---|
| 0x | 0.3% | 0.3% | +0.0 pts [+0.0, +0.0] |
| 1x | 7.0% | 6.7% | -0.3 pts [-2.3, +1.7] |
| 2x | 27.2% | 26.5% | -0.7 pts [-3.0, +1.5] |
| 5x | 66.7% | 68.0% | +1.3 pts [-1.5, +4.3] |
| 10x | 85.5% | 87.5% | +2.0 pts [-0.5, +5.0] |
| 20x | 95.7% | 94.7% | -1.0 pts [-2.3, +0.3] |
| all facts at 2x and above | 68.8% | 69.2% | +0.4 pts [-0.8, +1.6] |
| dev facts at 2x and above (n=200) | 68.4% | 70.8% | +2.3 pts [+0.5, +4.2] |
| final facts at 2x and above (n=200) | 69.1% | 67.6% | -1.5 pts [-2.9, -0.1] |

## World `main_w1`: training seeds (1 seeds)

| Exposure | seed 0 | Mean [95% CI over facts] | Spread across seeds (max minus min) |
|---|---|---|---|
| 0x | 0.0% | 0.0% [0.0%, 0.0%] | 0.0 pts |
| 1x | 7.5% | 7.5% [3.0%, 13.0%] | 0.0 pts |
| 2x | 14.5% | 14.5% [8.0%, 21.5%] | 0.0 pts |
| 5x | 65.5% | 65.5% [56.5%, 74.5%] | 0.0 pts |
| 10x | 91.5% | 91.5% [86.0%, 96.5%] | 0.0 pts |
| 20x | 97.5% | 97.5% [94.0%, 100.0%] | 0.0 pts |

### `main_w1`: heldout wording vs standard wording (paired by fact, averaged over seeds)

| Exposure | Standard | Heldout | Difference [95% CI] |
|---|---|---|---|
| 0x | 0.0% | 0.0% | +0.0 pts [+0.0, +0.0] |
| 1x | 7.5% | 6.0% | -1.5 pts [-5.0, +1.0] |
| 2x | 14.5% | 15.0% | +0.5 pts [-2.5, +3.5] |
| 5x | 65.5% | 64.5% | -1.0 pts [-3.0, +1.0] |
| 10x | 91.5% | 90.5% | -1.0 pts [-4.0, +1.5] |
| 20x | 97.5% | 96.5% | -1.0 pts [-3.5, +1.0] |
| all facts at 2x and above | 67.2% | 66.6% | -0.6 pts [-2.0, +0.6] |
| dev facts at 2x and above (n=200) | 66.5% | 65.8% | -0.8 pts [-2.8, +1.0] |
| final facts at 2x and above (n=200) | 68.0% | 67.5% | -0.5 pts [-2.2, +1.0] |

### `main_w1`: swapped wording vs standard wording (paired by fact, averaged over seeds)

| Exposure | Standard | Swapped | Difference [95% CI] |
|---|---|---|---|
| 0x | 0.0% | 0.0% | +0.0 pts [+0.0, +0.0] |
| 1x | 7.5% | 7.5% | +0.0 pts [-1.5, +1.5] |
| 2x | 14.5% | 16.0% | +1.5 pts [-1.5, +4.5] |
| 5x | 65.5% | 65.0% | -0.5 pts [-3.0, +1.5] |
| 10x | 91.5% | 91.5% | +0.0 pts [-2.0, +2.0] |
| 20x | 97.5% | 96.5% | -1.0 pts [-3.5, +1.0] |
| all facts at 2x and above | 67.2% | 67.2% | +0.0 pts [-1.1, +1.1] |
| dev facts at 2x and above (n=200) | 66.5% | 67.5% | +1.0 pts [-0.5, +2.5] |
| final facts at 2x and above (n=200) | 68.0% | 67.0% | -1.0 pts [-3.0, +0.8] |

## Second world: `main_w1` minus `main_v0` (seed-averaged accuracy, different facts)

| Exposure | main_v0 | main_w1 | Difference [95% CI] |
|---|---|---|---|
| 0x | 0.3% | 0.0% | -0.3 pts [-1.0, +0.0] |
| 1x | 7.0% | 7.5% | +0.5 pts [-5.8, +6.8] |
| 2x | 27.2% | 14.5% | -12.7 pts [-22.8, -2.7] |
| 5x | 66.7% | 65.5% | -1.2 pts [-12.3, +10.7] |
| 10x | 85.5% | 91.5% | +6.0 pts [-1.0, +12.3] |
| 20x | 95.7% | 97.5% | +1.8 pts [-2.3, +5.7] |

