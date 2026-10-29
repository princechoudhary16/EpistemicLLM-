#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=p5_export
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda/12.8.0
export CPATH="/apps/cuda_12.8.0/include:${CPATH:-}"
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate ~/envs/ntu-gpu
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Exports P5 responses (1 greedy + 5 sampled per question, all 1,200 main_v0 eval questions) in the shared
# schema. About 20-25 min per model (P6's one-at-a-time generation). Defaults: lambda 0 and 3, seed 0.
#   EXPORT_ARGS="--limit 5 --out-dir results/smoke"   quick smoke test (not comparable)
#   EXPORT_ARGS="--seeds 0 1 2"                        all three training seeds
python p5_training_method/export_responses.py ${EXPORT_ARGS:-}
