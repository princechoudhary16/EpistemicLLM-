#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=p5_kq_labels
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

# Teacher K labels (reports/p5_kq_target.md): the lambda=0 seed-0 checkpoint answers P1's held-out wordings, about
# 22 min per world (P6's one-at-a-time generation).  Writes results/final/p5_kq_labels_<dataset>.json
#   EXPORT: DATA_DIR=data/synthetic/main_v0 sbatch p5_training_method/jobs/p5_kq_labels.sbatch
#   smoke:  KQ_ARGS="--limit 5 --out-dir results/smoke --labels-out results/smoke/kq_labels.json" ...
DATA_DIR="${DATA_DIR:?Set DATA_DIR, e.g. data/synthetic/main_v0}"
DSET="$(basename "$DATA_DIR")"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python p5_training_method/kq_labels.py --dataset-version "$DSET" ${KQ_ARGS:-}
