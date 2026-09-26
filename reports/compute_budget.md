# Compute budget (P6 + P5, plan.md section 3 "Compute")

Measured on CCDS GPU Cluster TC2 (`ssh tc2`), 2026-09-26.

## Per-job resources (QoS `normal`, partition `MGPU-TC2`)

| Resource | Assigned QoS max (`MaxTRESPU`) | Notes |
|---|---|---|
| CPUs | 10 | Shared across up to `MaxJobsPU`=2 concurrent jobs |
| GPU | 1 (NVIDIA A40, 46 GB) | `--gres=gpu:1`; measured via `nvidia-smi` in a running job |
| Memory | 30G | |
| Wall time | 6:00:00 per job (`MaxWall`) | Request additional-QoS if a run needs longer |
| Compute node | 72 logical CPUs, 377G RAM total (shared with other users) | QoS caps, not the node's physical capacity |

Two jobs can run at once (`MaxJobsPU=2`), so P5's lambda sweep and P6's baseline
runs should be queued rather than assumed to run in parallel beyond that.

## Environment (conda env `torch` on the cluster)

| Package | Version |
|---|---|
| Python | 3.11 |
| torch | 2.13.0 (CUDA 12.8 build) |
| transformers | 4.57.6 |
| peft | 0.21.0 |
| accelerate | 1.15.0 |
| datasets | 5.0.1 |
| pyarrow | 25.0.0 |

**Known pitfall fixed:** `conda install datasets` alone resolved a stale
`datasets==2.14.4` that is incompatible with the `pyarrow` already in the
environment (`AttributeError: module 'pyarrow' has no attribute
'PyExtensionType'`). Fixed by explicitly installing `datasets>=3.0`. If any
member's own environment hits this, upgrade `datasets` rather than downgrading
`pyarrow` (torch/transformers pull the newer pyarrow in transitively).

Disk: `.conda` env directory is ~12G after `conda clean -a`; home quota is 100G.
Run `conda clean -a` after installs — the download cache alone was 16G before
cleaning.

## Model memory (Qwen2.5-0.5B-Instruct)

~1GB in bf16 for weights alone; well within the 30G memory cap and the A40's
46G, leaving headroom for LoRA adapters, optimizer states, and activation/KV
cache at the batch sizes this project needs. No quantization required.

## Throughput (to fill in from the Week 1 smoke run)

- Latency per generated response (64 max new tokens): see
  `results/raw/smoke_base_v0.jsonl` (`latency_seconds` field) for a first
  measurement on 5 questions x 6 generations.
- Full real-QA knowledge-mapping budget per plan.md section 10: 200 pilot +
  1,000 main questions x 6 generations = 7,200 responses. Multiply the smoke
  run's per-response latency by this to project wall time before scaling up.
