# P6 — IDK baseline fix: shuffle + repeat sweep (float32)

Per `EpistemicLLM Remaining Work and Order` (2026-10-31 meeting), P6 fix item:
"Rerun the 'I don't know' baseline in float32: shuffle the data, sft_matched
recipe, 2-3 amounts of refusal examples -- it currently refuses 100% of
questions."

## 1. Root cause (recap)

The original `idk` baseline (`reports/p6_week5_master_table.md`) collapsed to
100% abstention. P1's ablation traced this to a training-order artifact: the
30 unique IDK examples were appended *after* all 3,800 fact presentations with
no shuffling, so the model's final gradient steps were all refusals -- it
learned "the end of training means refuse," not "refuse when unsure."

## 2. Fix

`p6_baselines/run_idk_shuffled.py` (new):
- Combines the fact rows and IDK rows, then shuffles the combined set with a
  fixed seed (0) -- removes the "last N steps are all refusals" artifact.
- Uses the matched recipe (r=16, alpha=32, dropout=0.05, batch=4, all 7 linear
  layers, lr 2e-4, 1 epoch) instead of the original under-capacity recipe --
  consistent with `sft_matched`.
- Trains and evaluates in **float32** (`configs/model.yaml`, team decision).
- Takes `--idk-repeats N`: the 30 unique IDK entities are repeated N times
  before shuffling, to test different abstention/fact-learning balances, since
  the right amount wasn't known in advance.

Swept N = 1, 2, 3 (jobs 50290, 50303, 51143), as asked ("2-3 amounts").

## 3. Results (main_v0, 1,200 eval questions, float32, grading `p2_whole_word_v5`)

| IDK repeats | IDK rows / total train rows | Correct | Incorrect | Abstain | Accuracy | Abstention |
|---|---|---:|---:|---:|---:|---:|
| 1 | 30 / 3,830 (0.78%) | 618 | 582 | 0 | 51.5% | 0.0% |
| 2 | 60 / 3,860 (1.55%) | 607 | 581 | 12 | 50.6% | 1.0% |
| 3 | 90 / 3,890 (2.31%) | 579 | 610 | 11 | 48.3% | 0.9% |

Reproduction: `results/raw/p1_main_v0_idk_shuffled_r{1,2,3}.jsonl`, jobs
50290/50303/51143, `python -m p6_baselines.run_idk_shuffled --dataset-version
main_v0 --idk-repeats {1,2,3}`, seed 0 throughout.

## 4. Conclusion

The shuffle fixes the 100%-abstention collapse -- confirms P1's diagnosis was
correct. But none of the three repeat levels restores *meaningful*
abstention: 0% at 1x, and only ~1% at 2x/3x, with 2x and 3x landing in the
same narrow band rather than trending upward. Increasing the repeat count
further within this range will not fix it -- the 30 unique IDK examples are
simply too small a fraction of the training mix (well under 3% even at 3x) for
the gradient signal to compete with 3,800 fact presentations. Restoring a
usable abstention baseline would need either a much larger IDK fraction or
loss re-weighting on those examples -- out of scope for this fix, which was
explicitly bounded to "2-3 amounts" of repetition.

This is a clean negative result, documented rather than silently left out:
the `idk` row in `reports/p6_week5_master_table.md` remains the frozen Week 5
reference (100% abstention, training-order bug); none of the three
float32/shuffled variants here are a drop-in replacement for it, since all
three still show ~0% abstention and are not meaningfully different from
`sft_matched`/`sft` on accuracy.
