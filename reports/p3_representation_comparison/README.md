# Base vs SFT representation comparison

- Base run: `results/p3/synthetic_main_v0_base_prompt_v0_bf16_comparison` (BF16 inference).
- SFT run: `results/p3/synthetic_main_v0_sft_matched_seed0` (BF16 inference; matched adapter).
- Same pinned model revision, shared prompt, 600 question IDs, group assignments, layer depths, pooling, batch size, and inference dtype.
- PCA split: seed 42, 450 training questions / 150 validation questions; 225 / 75 fact groups with no overlap.
- For each layer, PCA is fit jointly to base and SFT **training-question** vectors only; held-out vectors are projected into that shared basis.
- SVG colors show exposure; circles are base and triangles are SFT. Faint connecting lines show a deterministic subset of paired prompt shifts.
- Similarity metrics compare paired validation vectors. Positive exposure groups and accuracy labels are not used to fit PCA.
- The base and SFT activation matrices were both extracted with BF16 inference; their pooled feature arrays are stored as float32.
- The SFT reviewed-answer labels are assistant-reviewed provisional labels. Base labels use the automatic exact-match proxy; answer correctness is not used in this geometry comparison.

## Held-out paired vector similarity by layer

| Layer | Mean cosine | Mean relative L2 shift | Joint PCA variance (PC1+PC2) |
|---:|---:|---:|---:|
| 6 | 0.9158 | 0.4191 | 0.915 |
| 12 | 0.8895 | 0.4916 | 0.886 |
| 18 | 0.7405 | 0.7184 | 0.456 |
| 24 | 0.6527 | 0.7780 | 0.341 |

The cosine and relative-shift values describe how adapter loading changes each matched prompt vector. They do not measure whether a representation is better or more truthful. The 2D projection shows only the first two joint PCs, so overlap in the plot does not imply that higher-dimensional representations are identical.

Artifacts: `base_vs_sft_joint_pca.svg`, `paired_representation_metrics.csv`.
