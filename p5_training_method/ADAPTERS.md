# P5 adapters: what they are and how to load them

Archive (not in git): `p5_checkpoints_2026-11-04.tgz`, 374 MB, 12 runs. Each run folder holds
`checkpoint/adapter/` (LoRA weights), `checkpoint/head.pt` (evidence head), `checkpoint/meta.json` and `run_config.json`.

| folder | world | lambda | seeds |
|---|---|---|---|
| `main_v0_lam0_seed{0,1,2}` | main_v0 | 0 (plain SFT through P5's code path; the head is **untrained**, ignore `head.pt`) | 0, 1, 2 |
| `main_v0_lam3_seed{0,1,2}` | main_v0 | 3 (answer loss + 3 x evidence-head loss) | 0, 1, 2 |
| `main_w1_lam0_seed{0,1,2}` | main_w1 (second world) | 0 | 0, 1, 2 |
| `main_w1_lam3_seed{0,1,2}` | main_w1 | 3 | 0, 1, 2 |

Common facts (all 12 runs):
- Base model `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` (`configs/model.yaml`).
- Trained in **float32**. Load in float32; the P5 code stops otherwise (`precision.py`).
- LoRA r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down; 1 epoch, lr 2e-4, batch 4. Recipe in `run_config.json`.
- Trained on the matching world's synthetic training file (path recorded in `run_config.json`).
- The head reads the final-layer hidden state at the last prompt token and outputs one number in 0..1
  (exposure-like confidence). The adapter alone gives the model's answers; the head adds the score.

Responses already exported in float32, shared schema, graded `p2_whole_word_v5`
(1 greedy + 5 sampled per question, 1,200 `main_v0` eval questions; seed 0 only):
`results/raw/p5_main_v0_lam0_seed0.jsonl` (`method=sft_p5code`, `aux_confidence_score` null) and
`results/raw/p5_main_v0_lam3_seed0.jsonl` (`method=p5`, `aux_confidence_score` = head score).
`confidence_score` is the average answer log-prob in both. Seeds 1 and 2 are not exported; ask, or run
`EXPORT_ARGS="--seeds 1 2" sbatch p5_training_method/jobs/p5_export.sbatch` on the cluster.

## Load and generate (the same call `export_responses.py` makes)
```python
from common.model import ModelRunner
runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775",
                     adapter_path="main_v0_lam3_seed0/checkpoint/adapter", torch_dtype="float32", device="cuda")
```
Then use it as any other `ModelRunner` (for example with `common.evaluate.run_eval`, then
`common.grading.regrade_rows(rows)` so every number is graded with v5).

## Head score for a question (lambda 3 only)
```python
import torch
from p5_training_method.head import EvidenceHead
head = EvidenceHead(runner.model.config.hidden_size).to("cuda")
head.load_state_dict(torch.load("main_v0_lam3_seed0/checkpoint/head.pt", map_location="cuda"))
head.eval()
```
and score questions with `head_scores(...)` from `p5_training_method/export_responses.py`, which builds the same prompt,
left-pads, and reads the last prompt token's final hidden state. Do not rebuild that by hand: the pooling position and
the system prompt (`configs/decoding.yaml`) must match training.

Honest limits: these examples mirror the code path that produced the exports on the cluster, but the snippets above
were not run as a separate script. The adapters were trained on all 600 facts of the world (evaluation asks new
wordings of known facts), so the head partly knows each fact's exposure; see `reports/p5_evidence_aware_training.md`.
