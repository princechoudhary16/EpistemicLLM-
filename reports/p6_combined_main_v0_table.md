# P6 — combined main_v0 table, all methods under decoding_v1

Closes the gap P5 flagged in `reports/p5_evidence_aware_training.md` section 3.5:
my `base`/`sft`/`idk` on `main_v0` were generated under the stale `decoding_v0`
config; `compare_methods.py`'s hardened validator correctly refused to mix them
with `sft_matched`/P5's files (`decoding_v1`). Regenerated all three (jobs
44817 pilot, 44818 main; main took 93 min, slower than the earlier ~57 min run,
cluster load) under the current frozen config, then built one table with all
six methods, all on the identical 1,200 `main_v0` questions.

## Full table (greedy, N=1200 each)

| Method | Accuracy | Hallucination | Abstention | Coverage | AURC (log-prob) |
|---|---:|---:|---:|---:|---:|
| base (untouched) | 0.0% | 68.3% | 31.7% | 68.3% | 0.683 |
| sft (P6 original recipe: r8, q_proj+v_proj, batch 1) | 5.4% | 94.6% | 0.0% | 100% | 0.838 |
| idk (P6 original recipe) | 0.0% | 0.0% | 100% | 0.0% | n/a |
| **sft_matched** (Aryaman's recipe, P6's runner) | **46.8%** | 53.2% | 0.0% | 100% | 0.213 |
| **p5_lam0** (Aryaman's recipe, P5's pipeline) | **48.7%** | 51.3% | 0.0% | 100% | 0.200 |
| **p5_lam3** (+ auxiliary head/loss) | **48.6%** | 51.4% | 0.0% | 100% | 0.185 |

Paired accuracy diffs vs. `sft_matched` (question-cluster bootstrap, 95% CI, n_boot=1000):

| Comparison | Estimate | 95% CI |
|---|---:|---|
| p5_lam0 vs sft_matched | +1.8pp | [0.0, +3.8] |
| p5_lam3 vs sft_matched | +1.8pp | [-0.2, +3.6] |
| sft vs sft_matched | -41.4pp | [-44.2, -38.4] |
| idk vs sft_matched | -46.8pp | [-49.4, -44.0] |

## What changed vs. the previously reported (decoding_v0) main_v0 numbers

| Method | decoding_v0 (old) | decoding_v1 (current) |
|---|---|---|
| base accuracy | 0.25% | 0.0% |
| base abstention | 3.9% | **31.7%** |
| sft accuracy | 8.2% | 5.4% |
| idk abstention | 99.7% | 100% |

The big move is **base model abstention: 3.9% to 31.7%**. This matches P1's own
ablation finding exactly: the old config inherited the pinned model's default
`repetition_penalty=1.1`, which (per that ablation) measurably suppresses the
"I don't know" tokens in the system prompt. `decoding_v1` sets
`repetition_penalty=1.0` explicitly, removing that suppression. This is a
genuine correction, not noise -- the old main_v0 base-model numbers understated
how often the untouched model actually abstains.

## Cross-check with P5 (independent pipelines, same recipe)

- `sft_matched` (my runner): 46.8%
- `p5_lam0` (P5's runner, same Aryaman recipe, lambda=0 so the head gets no gradient): 48.7%

+1.8 points apart, CI touches 0 at the lower bound -- consistent within noise,
not a strong statement either way. P5's report attributes the gap to pipeline
differences (float32 vs bfloat16 training, seed-setting order) rather than
anything meaningful, which this combined table doesn't contradict.

## The auxiliary head's AURC, reproduced via --confidence-field

P5 had computed this by hand before `compare_methods.py` supported it:

```
python -m p6_baselines.compare_methods \
    --responses sft_matched=results/raw/p1_main_v0_sft_matched.jsonl \
                 p5_lam3=results/raw/p5_main_v0_lam3_seed0.jsonl \
    --confidence-field aux_confidence_score
```

Reproduces **0.258** for `p5_lam3`'s AURC under the auxiliary head's own score
(vs. 0.185 under log-probability, from the main table above) -- exact match to
P5's hand-computed number. Confirms the auxiliary head is a *worse* abstention
signal than the model's own log-probability, consistent with P5's full
abstention-policy analysis (section 3.4 of their report).

## Files

- `results/raw/p1_{pilot_v0,main_v0}_{base,sft,idk}.jsonl` -- regenerated under
  `decoding_v1` (overwrote the stale `decoding_v0` versions)
- `results/final/baseline_comparison_main_v0_combined.json` -- the full 6-method
  table above, machine-readable
