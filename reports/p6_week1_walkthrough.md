# P6 Week 1 — What was built and why

A walkthrough of the P6 (baselines/infrastructure) work done on the `sahana` branch,
committed as `fd2ed08` and pushed to GitHub. Written to explain the *why* behind each
file, not just list them — see `p6_baselines/README.md` for the short version.

## 1. Where this sits in the plan

`plan.md` §2 makes P6 the team's shared infrastructure and fair judge: build the common
model runner and metrics, train the comparison baselines, and run every method through
the same evaluator so the final table isn't misleading (e.g. low hallucination that's
really just refusal). §3 ("Days 1–3: freeze the shared contract") lists what P6 must pin
before anyone else's numbers are comparable: exact model + revision, prompt, decoding
settings, and the output schema everyone writes to.

This session did the **Week 1** slice of that: get the pinned pieces working and prove
it with one real, reproducible run — not the SFT/IDK baselines yet (those are still
open, noted at the bottom).

## 2. The shared contract: `configs/`

**`configs/model.yaml`**
```yaml
model_id: Qwen/Qwen2.5-0.5B-Instruct
revision: 7ae557604adf67be50417f59c2c2f167def9a775
torch_dtype: bfloat16
```
`plan.md` says "pinned revision and tokenizer." A model *id* alone isn't pinned — the
weights behind `Qwen/Qwen2.5-0.5B-Instruct` on the Hugging Face Hub can change if the
maintainers push an update. The `revision` is the exact commit SHA (read from the HF
API), so every member loads the *same* weights indefinitely, even if the repo moves on.

**`configs/decoding.yaml`** encodes the exact numbers `plan.md` §3 specifies: one greedy
response + 5 sampled responses per question, temperature 0.7, top-p 0.9, 64 max new
tokens, plus the system prompt that explicitly allows "I don't know." Both files are
versioned (`prompt_version`, `decoding_config_id`) so every stored response can cite
exactly which settings produced it — needed later when comparing methods fairly.

## 3. The model runner: `common/model.py`

Two pieces:

- **`environment_versions()`** — records installed package versions (torch,
  transformers, peft, accelerate, datasets) and the GPU name at generation time. This is
  the "environment versions recorded" half of the Week 1 acceptance criterion — if a
  result looks different in a month, this says whether the software changed.

- **`ModelRunner`** — loads the pinned checkpoint (with an optional LoRA adapter for
  later, once P5 has checkpoints to evaluate) and exposes `generate_one(...)`. Given the
  same `(model revision, decoding config, seed)`, it always produces the same response —
  that's the "one batch generates reproducibly" requirement. It also computes, from the
  model's own generation-step logits (not the sampling probabilities):
  - `avg_logprob` — mean log-probability of the tokens actually generated
  - `entropy` — mean predictive entropy at each generated position
  - `truncated` — whether the response hit the token limit without reaching EOS

  These aren't the final confidence definitions — `plan.md` §6 explicitly gives P4
  ownership of that — but P4 needs *something* to build on, and computing it from raw
  logits inside the generation call (rather than re-deriving it later from stored text)
  keeps the numbers exact and cheap to get.

## 4. The shared row format: `common/schema.py`

`plan.md` §9 specifies one JSONL schema every track writes to, so P3/P4/P6 can join on
`question_id` without agreeing on file layouts individually. `make_row(**fields)` builds
one response row and fills every field the caller didn't pass with `null` — so a
synthetic row and a real-QA row have identical columns (a real-QA row's `exposure_level`
and `fact_id` are just `null`), and every writer produces mergeable output by construction
rather than by convention.

## 5. The metrics library: `common/metrics.py`

This is P6's actual deliverable for the final comparison table. Implemented straight from
`plan.md` §6's metric table:

| Function | What it answers |
|---|---|
| `behavior_metrics` | accuracy, hallucination rate, abstention rate, coverage, selective accuracy/risk from raw outcome counts |
| `knowledge_scores` | K(q): fraction of *sampled* (non-greedy) responses that were correct, per question |
| `auroc` / `auprc` | does a confidence signal actually separate correct from incorrect answers |
| `brier_score` / `expected_calibration_error` / `reliability_bins` | is a stated confidence probability actually trustworthy |
| `apply_threshold` | turns a confidence cutoff into abstentions — modeling an *inference policy*, so it can be applied identically to every baseline (plan.md §7: never credit training for a benefit that's really just filtering) |
| `confident_hallucinations` | count/rate of wrong answers the model was still confident about |
| `risk_coverage_curve` / `area_under_risk_coverage` / `risk_at_coverage` | the coverage-vs-risk tradeoff a threshold buys you |
| `bootstrap_ci` / `paired_bootstrap_diff` | confidence intervals that resample whole *questions* (or facts), not individual responses — because 5 sampled answers to the same question aren't 5 independent data points |

Every function returns `None` (not a crash, not a fake `0.0`) exactly where `plan.md`
says "N/A" applies — e.g. selective accuracy when nothing was answered, or AUROC when
only one class is present.

**Why tests on hand-calculated fixtures, specifically:** the Week 1 acceptance criterion
says "metric edge cases checked" — not just "metrics implemented." `common/tests/test_metrics.py`
has 25 tests where the expected number is worked out on paper first (e.g. "5 correct, 3
incorrect, 2 abstain, N=10 → accuracy must be exactly 0.5"), so a bug that returns a
plausible-looking-but-wrong number gets caught, not just a crash.

One test caught a mistake in my own reasoning, not the code: I initially expected a
"perfect" classifier's risk-coverage area to be 0, but that's wrong — at 100% coverage
you're forced to include every wrong answer too, so risk necessarily rises as coverage
approaches 1. I fixed the *test's* expected value once I'd worked out the correct
hand-calculation, not the metric — a reminder that "the test fails" doesn't always mean
the code is the thing that's wrong.

## 6. Proving it actually works: `p6_baselines/`

Code that imports cleanly isn't the same as code that runs. `plan.md`'s Week 1 criterion
is "one reproducible command for generation/evaluation" — so this had to actually execute
against the real model on the real cluster, not just pass a lint check.

- **`smoke_fixture.jsonl`** — 5 self-contained questions (capital of France, days in a
  week, sky color, gold's symbol, plus one deliberately unanswerable question about a
  fictional explorer). `plan.md`'s dependency table says P6 only needs "dummy prediction
  files + base model" to start — this is intentionally independent of P1/P2's real data,
  so P6 isn't blocked waiting on them.
- **`generate_base.py`** — loads the pinned config, generates greedy + 5 sampled
  responses per question, grades them with an explicitly-labeled *placeholder* checker
  (`grading_version="placeholder_exact_v0"`) — real grading is P2's job per `plan.md` §2,
  and this placeholder exists only to wire the pipeline together, not to produce a
  reportable number.
- **`job_smoke.sh`** — the SLURM script to run this as a proper `sbatch` job (never on
  the head node, per the cluster's own rules).

### What went wrong running it for real, and what that means

Getting this to actually run on the cluster surfaced two environment bugs that a
"looks correct" code review would never have caught:

1. **`datasets` incompatible with `pyarrow`.** `conda install datasets` alone resolved a
   stale `datasets==2.14.4`, which crashes on import against the `pyarrow` already
   present (`pyarrow.PyExtensionType` was removed in newer pyarrow releases). Fixed by
   pinning `datasets>=3.0` explicitly. Recorded in `reports/compute_budget.md` so any
   other team member hitting the same crash knows the fix (upgrade `datasets`, don't
   downgrade `pyarrow` — torch pulls the newer pyarrow in regardless).

2. **Triton couldn't find `cuda.h`.** This torch build JIT-compiles a small CUDA kernel
   (for rotary position embeddings) the first time a Qwen model generates. The compiler
   needs `cuda.h`, and the cluster guide's own advice — `module load cuda/12.8.0` — sets
   `PATH`/`LD_LIBRARY_PATH` but *not* the compiler's include path. Found the actual header
   at `/apps/cuda_12.8.0/include/cuda.h` and added `export CPATH=/apps/cuda_12.8.0/include:$CPATH`
   to the job script. Anyone else on this cluster generating from a similar torch build
   will hit the identical error.

3. **A silent `NaN` in the output.** After the job completed, I looked at the actual JSONL
   it wrote (not just "job status: COMPLETED") and found `"entropy": NaN` in one row —
   valid Python (`json.dumps` allows it by default) but not valid JSON, so any other tool
   or language reading this file later would choke on it. Traced it to `0 * -inf` when a
   token's probability underflows to exactly zero in floating point — mathematically
   `0 * log(0) → 0` in the limit, but literally computing it gives `nan`. Fixed the
   calculation (`torch.nan_to_num` on that term) and, since this class of bug is easy to
   reintroduce, made `schema.write_jsonl` raise immediately if it's ever asked to write
   `NaN`/`Infinity` again — turning a silent data-corruption bug into a loud crash next time.

Re-ran the job after each fix. Final run (job **37812**, `COMPLETED`, 20 seconds on an
A40): 4/5 greedy answers correct, and the deliberately-unanswerable question scored
knowledge=0.0 across all 5 samples — the model never guessed right, which is exactly the
behavior that fixture question exists to surface.

## 7. The budget writeup: `reports/compute_budget.md`

`plan.md` §3 assigns P6+P5 to "measure available accelerator memory, throughput,
training step time" rather than assume hardware capacity. This records what was actually
measured: your QoS caps (1 GPU/10 CPU/30G mem/6hr, 2 concurrent jobs), the A40's 46G vs.
Qwen 0.5B's ~1GB footprint (plenty of headroom for LoRA later), the exact package
versions now working, and the two bugs above so nobody re-debugges them from scratch.

## 8. What's committed vs. what's still open

**Done and pushed to `sahana` (commit `fd2ed08`):** configs, model runner, schema,
metrics + tests, the smoke pipeline, the compute report, all verified end-to-end on the
cluster.

**Not done yet** (explicitly, so nothing is implied to be finished that isn't):
- The canonical standard-LoRA-SFT baseline
- The IDK/abstention baseline (needs training-only unknown entities)
- Swapping the smoke run's placeholder grader for P2's real grader
- The risk–coverage evaluator applied to more than the smoke fixture
- Anything using P1's actual synthetic data or P2's real SimpleQA data (both still live
  on their own unmerged branches/work)
