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

### Dev stage (jobs 45987 `main_v0`, 45988 `main_w1`; `aux0_criteria.py --split dev`; mean ± std over seeds 0, 1, 2)

| `main_v0` dev | baseline λ=3 | λ=3 + 0× prompts |
|---|---|---|
| head AUROC 1×/2× vs 0× | 0.342 ± .047 | **0.875 ± .062** |
| head AUROC ≥5× vs 0× | 0.826 ± .011 | 0.991 ± .009 |
| head AUROC ≥5× vs ≤2× | 0.871 ± .009 | 0.938 ± .023 |
| overall accuracy | 46.7% ± 2.7 | 44.6% ± 3.3 |
| accuracy at 0× | 0.0% | 0.0% |

| `main_w1` dev | baseline λ=3 | λ=3 + 0× prompts |
|---|---|---|
| head AUROC 1×/2× vs 0× | 0.379 ± .033 | **0.889 ± .035** |
| head AUROC ≥5× vs 0× | 0.854 ± .032 | 0.994 ± .003 |
| head AUROC ≥5× vs ≤2× | 0.878 ± .017 | 0.955 ± .003 |
| overall accuracy | 46.8% ± 1.8 | 47.4% ± 0.7 |
| accuracy at 0× | 1.0% ± 1.0 | 0.3% ± 0.6 |

| criterion | `main_v0` | `main_w1` |
|---|---|---|
| a: 1×/2× vs 0× ≥ 0.60 | **pass** (0.875) | **pass** (0.889) |
| b: headline ≥ 0.80 and not > 0.03 below baseline | **pass** (0.938) | **pass** (0.955) |
| c: accuracy within 2 points and 0× accuracy ≤ 2% | **FAIL**: accuracy −2.1 points (bound 2.0); 0× accuracy 0.0% passes | **pass**: +0.6 points; 0× accuracy 0.3% |

**Dev verdict under the pre-stated rule (a, b and c in every world): NOT ALL PASS.** Criterion c is missed in
`main_v0` by 0.1 point. The criteria are not changed after seeing this. Context, not a revision of the verdict:
the seed spread of the accuracy difference is 2.7–3.3 points per condition, so the 2-point bound is finer than
what 3 seeds can resolve; the drop is not reproduced in `main_w1` (+0.6); and the leak check itself (0× accuracy)
is clean in both worlds. Per the protocol the final split is now scored once per world, whatever the verdict.

Reading the AUROC jump (0.35 → 0.88): large and consistent across both worlds, but note the caveat above. The head
has now seen the 0× entity names, so part of this may be recognising those names rather than recognising that an
entity is unknown. Whether the improvement transfers to 0× entities the head has never seen is not tested by this
experiment.

### Final stage (jobs 46026 `main_v0`, 46027 `main_w1`; scored once per world after the dev verdict above was recorded)

| `main_v0` final | baseline λ=3 | λ=3 + 0× prompts |
|---|---|---|
| head AUROC 1×/2× vs 0× | 0.349 ± .009 | **0.864 ± .059** |
| head AUROC ≥5× vs 0× | 0.811 ± .031 | 0.985 ± .010 |
| head AUROC ≥5× vs ≤2× | 0.848 ± .024 | 0.931 ± .007 |
| overall accuracy | 47.4% ± 1.6 | 44.2% ± 3.7 |
| accuracy at 0× | 1.3% ± 1.2 | 1.7% ± 0.6 |

| `main_w1` final | baseline λ=3 | λ=3 + 0× prompts |
|---|---|---|
| head AUROC 1×/2× vs 0× | 0.351 ± .036 | **0.904 ± .011** |
| head AUROC ≥5× vs 0× | 0.826 ± .027 | 0.993 ± .003 |
| head AUROC ≥5× vs ≤2× | 0.863 ± .018 | 0.954 ± .008 |
| overall accuracy | 45.1% ± 1.5 | 45.7% ± 1.5 |
| accuracy at 0× | 0.7% ± 0.6 | 0.3% ± 0.6 |

Criteria on final (reported, same rule): a **pass** and b **pass** in both worlds; c **fails in `main_v0`** (accuracy −3.2
points; 0× accuracy 1.7% is within the bound) and **passes in `main_w1`** (+0.6 points; 0× accuracy 0.3%).

### Criterion d: abstention (final split, thresholds from dev; net score / AURC ↓, mean over 3 seeds)

| `main_v0` | penalty 1 net | penalty 3 net | AURC |
|---|---|---|---|
| head, before (λ=3) | 0.222 | 0.122 | 0.285 |
| head, with 0× prompts | 0.241 | 0.097 | 0.286 |
| P5 log-prob, before → with 0× prompts | 0.352 → 0.338 | 0.313 → 0.301 | 0.202 → 0.224 |
| SFT log-prob (reference) | 0.364 | 0.288 | 0.200 |

| `main_w1` | penalty 1 net | penalty 3 net | AURC |
|---|---|---|---|
| head, before (λ=3) | 0.230 | 0.172 | 0.281 |
| head, with 0× prompts | 0.287 | 0.214 | 0.242 |
| P5 log-prob, before → with 0× prompts | 0.371 → 0.372 | 0.327 → 0.321 | 0.208 → 0.203 |
| SFT log-prob (reference) | 0.374 | 0.326 | 0.202 |

Behaviour at the two ends, head abstention rate at penalty 1 (before → with 0× prompts): at 0× 0.83 → 1.00
(`main_v0`) and 0.83 → 0.99 (`main_w1`); at 5× 0.77 → 0.55 and 0.78 → 0.47 (the head over-abstains less on
moderately exposed facts, where the model is right about 65% of the time).

## Verdict
- **The AUROC rose, but the name-recognition control shows it was memorised names, not a fix.** The head ranks the
  0× facts it was shown below 1×/2× facts (AUROC 0.35 → 0.86–0.90) and the headline AUROC rises (0.85 → 0.93,
  0.86 → 0.95), consistently in both worlds and both splits. But `reports/p5_aux_zero_novel.md` (matched design, pre-stated
  criteria, dev then final, both worlds) found that a head trained without the scored 0× names stays at the baseline
  (1×/2× vs 0× AUROC 0.30–0.39), while a head that saw them with the same amount of supervision reaches 0.85–0.93.
  The ranking flaw for genuinely unfamiliar entities is therefore **not fixed** by these prompts.
- **The pre-stated dev verdict is NOT ALL PASS** (criterion c in `main_v0`), and the final split shows the same pattern.
  The accuracy cost in `main_v0` is −2.1 (dev) and −3.2 (final) points; the two splits are scored on the same 3 trained
  models, so they are not independent confirmations, the seed spread is 3–4 points, and `main_w1` shows no cost (+0.6).
  I would not call it an established cost or a clear absence of one. I did not change the criterion.
- **Criterion d: the head is still the worse abstention signal.** It improves in `main_w1` (net 0.230 → 0.287, AURC
  0.281 → 0.242) and barely in `main_v0` (AURC 0.285 → 0.286; penalty-3 net 0.122 → 0.097), and stays well below the
  model's own log-prob in both worlds at both penalties (e.g. 0.241 vs 0.364 and 0.287 vs 0.374 at penalty 1).
  In `main_v0` the 0× prompts also slightly worsen the model's own log-prob signal (AURC 0.202 → 0.224), in line
  with its lower accuracy there.
- **Status:** a negative result, with its control. The 0× prompts do not teach the head to recognise unknown entities. The main results in
  `reports/p5_evidence_aware_training.md` and `reports/p5_second_world.md` are unchanged.


