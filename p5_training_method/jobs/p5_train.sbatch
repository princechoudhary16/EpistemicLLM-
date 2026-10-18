#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=p5_train
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

LAMBDA="${LAMBDA:-0.0}"
SEED="${SEED:-0}"
DATA="${DATA:-data/synthetic/main_v0/train.jsonl}"
DSET="$(basename "$(dirname "$DATA")")"
RUN_ID="${RUN_ID:-${DSET}_lam${LAMBDA}_seed${SEED}_job${SLURM_JOB_ID}}"
python p5_training_method/train.py --lambda "$LAMBDA" --seed "$SEED" --run-id "$RUN_ID" --data "$DATA" --log-every 10 --save-every 50
