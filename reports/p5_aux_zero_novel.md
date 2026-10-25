# P5 — name-recognition control for the zero-exposure prompts: protocol and criteria

Status: **protocol written before any result exists.** The results section is filled in afterwards.

## Question
With P1's 0× prompts as head-only rows (`reports/p5_aux_zero.md`), the head's AUROC for 1×/2× vs 0× rose from about
0.35 to 0.86–0.90. But the head had seen the names of the 0× entities it was scored on, so the gain may be
**name recognition** ("this name was on the 0× list") instead of "this entity is unfamiliar, so there is no evidence".
This control asks which one it is.

## Design
For each world (`main_v0`, `main_w1`) train two sets of models, λ = 3 fixed, 3 seeds each, same recipe as before:
- `lam3halfdev`: head-only rows = the 0× prompts of the **dev** facts only (50 facts × 4 wordings = 200 rows);
- `lam3halffinal`: head-only rows = the 0× prompts of the **final** facts only (200 rows).

(`train.py --aux-zero-split dev|final`; the answer-training rows, their order and the aux-row schedule are as before.)
Both sets are scored on both splits:

| scored on | model trained with that split's own 0× prompts | model trained with the other split's 0× prompts |
|---|---|---|
| dev | `halfdev`: scored names **seen** (200 rows) | `halffinal`: scored names **novel** (200 rows) |
| final | `halffinal`: scored names **seen** (200 rows) | `halfdev`: scored names **novel** (200 rows) |

"Seen (200)" and "novel (200)" have identical amounts of 0× supervision and differ only in whether the scored 0× names
were seen, which removes the confound with the earlier 400-row experiment. `check_aux_novel.py` runs at the start of
every stage and stops the job unless no scored 0× fact is in the training aux prompts and no scored 0× name occurs in any
training aux prompt, `train.jsonl` row or `idk_train.jsonl` row (verified on the real data for both worlds and both
splits; it fails correctly on a wrong pairing).

## Protocol
Dev stage first (`STAGE=dev` of `jobs/p5_aux0nov.sbatch`: train both sets, score both on dev, dev probe controls),
judged with `aux0nov_criteria.py --split dev`. The verdict is recorded below **before** the final stage runs
(`STAGE=final`, scoring both sets on the final split once). No change to λ, the aux set, or the schedule.

## Criteria (written before any result; head AUROC for 1×/2× vs 0×, mean over 3 seeds)
| # | Criterion |
|---|---|
| A | **pass/fail**, in each world: the head trained *without* the scored names ("novel (200)") reaches AUROC **≥ 0.60** (baseline without any 0× prompts ≈ 0.35) |
| B | *descriptive:* retained gain = (novel − baseline) / (seen(200) − baseline). **≥ 0.5 → "mostly generalises"; < 0.5 → "mostly names".** If seen(200) itself is below 0.60 the reading is "inconclusive (not enough 0× data)" |
| C | *reported, not judged:* overall accuracy and accuracy at 0× for each condition |

Reading guide: A passes and B ≥ 0.5 means the earlier gain is mostly real generalisation to unknown entities; A fails
or B < 0.5 means the earlier gain was mostly memorised names and the 0× prompts do not generalise.

## Caveats to carry into the results
- The "1×/2×" side of the AUROC is made of entities the head has seen (all exposed facts are trained on), so even a
  novel-0× head can separate the two by detecting "was this name seen at all". That is what we hope the head learned,
  but it is a familiarity signal, not an absolute reading of evidence.
- Half as many 0× prompts (200 rows) as before; the matched "seen (200)" condition is there to account for that.
- Only the scored 0× names are novel; all exposed facts of both splits are trained on, as in every other experiment.
- Both worlds come from the same generator and the same placeholder grader; 3 seeds per world.

## Results
_Pending._
