# P6 — training-parity fix for the auxiliary-loss comparison

**Trigger:** teammate review (Oct 17) pointed out that comparing an
auxiliary-head-and-loss run against P6's canonical standard-SFT baseline is
invalid unless both sides use the identical LoRA recipe. P6's baseline used
r=8, `q_proj`+`v_proj` only, batch 1; Aryaman (P1)'s own SFT recipe used r=16,
alpha=32, dropout=0.05, batch 4. The reviewer also asked whether the 5×/10×/20×
facts were actually being learned, since the overall accuracy numbers blend in
the 0×/1×/2× facts that can't realistically be learned at all.

## Root cause (via P1's own ablation study, already merged into `main`)

P1 ran a rigorous ablation isolating each factor. **LoRA target modules is the
dominant effect, not rank or batch size:**

| Change from P6's recipe (isolated, nothing else changed) | Pilot accuracy |
|---|---|
| Baseline (`q_proj`+`v_proj`, r=8, batch 1) | 15.4% |
| + rank 16 only | 21.2% |
| + batch 4 only | **2.9%** (hurts -- fixed epoch length means 4x fewer optimizer steps) |
| + all 7 linear layers only | ~45% |

Aryaman's full recipe (r16, all 7 layers, batch 4) reaches 40.4-40.8% in P1's
ablation -- almost entirely from adapting more layers, with batch 4 actually
working against it on its own, compensated for by the larger adapter.

## Exposure-level breakdown P6's original baseline was missing

Computed from the existing (already-run) pilot SFT results, no retraining
needed for this part:

| Exposure | 0x | 1x | 2x | 5x | 10x | 20x |
|---|---|---|---|---|---|---|
| Mean K (P6 original recipe) | 0.000 | 0.000 | 0.005 | 0.005 | 0.170 | 0.550 |

Confirms the reviewer's concern directly: P6's baseline barely learns anything
even at high exposure, nowhere near Aryaman's reported 0.62/0.69/0.88 at
5x/10x/20x.

## Fix: `sft_matched`, trained with Aryaman's exact recipe

P5's own auxiliary-loss recipe wasn't available at the time; per instruction,
reused P1/Aryaman's published recipe (`"mine"` in
`p1_exposure/ablation/run_ablation.py`) as the best available matched point:

```
r=16, alpha=32, dropout=0.05, batch=4, lr=2e-4
target_modules = (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
```

Implemented in `p6_baselines/run_matched_sft.py`. Required extending
`common/train.py` to support batched (padded) training (previously batch-size-1
only) -- now matches P1's exact collate/padding approach.

**This produces an ADDITIONAL method (`sft_matched`), not a replacement.** The
original canonical `sft` baseline (r=8, `q_proj`+`v_proj`, batch 1) is unchanged
and still valid for its own purpose (does fine-tuning on facts work at all vs.
the untouched base model). Both are now available, clearly labeled.

## Result (pilot_v0, run on the cluster, job 43169, ~9 min)

| Exposure | 0x | 1x | 2x | 5x | 10x | 20x | Overall accuracy |
|---|---|---|---|---|---|---|---|
| Mean K, `sft_matched` | 0.000 | 0.000 | 0.180 | 0.500 | 0.730 | 0.910 | 43.3% |
| Aryaman's reported mean K | - | - | - | 0.62 | 0.69 | 0.88 | - |
| P1 ablation's `"mine"` greedy accuracy (pilot) | - | - | - | - | - | - | 40.4-40.8% |

Close to both reference points (small differences consistent with single-seed
run-to-run variation), confirming the recipe -- not some other discrepancy --
explains the original gap. 0x/1x remain exactly 0, consistent with P1's
no-leakage design holding under this recipe too.

## Still open

- **main_v0 (600-fact) matched-recipe run**: prepared (`job_matched_sft.sh`
  supports `DATASET_VERSION=main_v0`), not yet run.
- **P5's actual recipe for the auxiliary-head-and-loss run is still unconfirmed.**
  If it turns out to differ from Aryaman's `"mine"` recipe, this `sft_matched`
  result is not yet the correct matched baseline and would need retraining
  again with whatever P5 actually uses.
- IDK baseline was not retrained with this recipe (out of scope for this fix;
  the review's concern was specifically about the plain-SFT-vs-auxiliary-loss
  comparison).
