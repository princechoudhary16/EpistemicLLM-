# P5 — zero-exposure prompts for the head: protocol and success criteria

Status: **protocol written before any result exists.** The results section is filled in afterwards.

## Problem
The evidence head never sees a "no evidence" example (0× facts have no training rows), and it ranks never-trained
0× facts *above* 1×/2× facts: head AUROC for 1×/2× vs 0× is 0.349 (`main_v0` final) and 0.351 (`main_w1` final)
(`reports/p5_evidence_aware_training.md` §3.2b, `reports/p5_second_world.md`). The log-prob signals are at chance
there, so the failure is specific to the head and sits exactly where abstention matters most.

## Intervention
P1 built answer-masked prompts for the 0× facts (`data/synthetic/<world>/zero_prompts/aux_zero_prompts.jsonl`: 100
zero-exposure facts × 4 train wordings = 400 rows, no completion or answer field; verified: no answer text in any
prompt, train wordings only, no equality with any eval question, no 0× name in `train.jsonl`/`idk_train.jsonl`).
They are added to training as **head-only rows** (`train.py --aux-zero`):
- no answer tokens and every label −100, so they contribute nothing to the language-model loss (unit-tested);
- target `s = 0`; the head reads the last prompt token as for every other row;
- spread evenly over the 950 steps (aux row k joins step ⌊k·950/400⌋), order shuffled with the training seed;
  the answer-training rows and their order are unchanged;
- λ = 3, fixed (no search). λ = 0 needs no new runs: the aux rows add nothing to it, so the existing λ = 0
  checkpoints are the matched control (unit-tested: λ = 0 gradients are unchanged by the aux rows).

Runs: λ=3 + aux rows, seeds 0, 1, 2, on `main_v0` and `main_w1` (run ids `<world>_lam3aux0_seed<S>`).

## Protocol
Dev stage first (`STAGE=dev` of `jobs/p5_aux0.sbatch`: train, dev eval, dev probe controls), judged with
`aux0_criteria.py --split dev`. The final split is scored **once per world, only after** the dev verdict is recorded
here, whatever the verdict is (`STAGE=final`). No re-tuning of λ, the number of aux rows or the schedule.

## Success criteria (written before any result; dev split, mean over 3 seeds, must hold in **both** worlds)
| # | Criterion |
|---|---|
| a | head AUROC for 1×/2× vs 0× rises from ≈ 0.35 to **≥ 0.60** |
| b | the headline head AUROC (≥5× vs ≤2×) stays **≥ 0.80** and is not more than 0.03 below the current λ=3 |
| c | overall accuracy stays within **2 points** of the current λ=3, and accuracy at 0× stays **≤ 2%** (leak check) |
| d | *reported, not pass/fail:* whether the head's abstention net score (`abstain_eval.py`, penalty 1 and 3) closes the gap to log-prob |

## Caveats to carry into the results
- The head now sees the 0× entity **names** (never their answers), so 0× is no longer entity-novel for it. That is
  symmetric with the other levels (the head sees every level's names), but it is a change from before and it
  weakens any reading of the 0× results as "recognising genuinely unknown entities".
- The LoRA weights see these prompts through the auxiliary gradient. Criterion c's 0× accuracy check is there to
  catch any leakage into the answers.
- Both worlds come from the same generator and the same placeholder grader; 3 seeds per world.

## Results
_Pending._
