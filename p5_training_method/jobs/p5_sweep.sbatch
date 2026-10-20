#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=p5_sweep
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

# Trains and evaluates (dev split only) every (seed, lambda) pair in one job, so the
# 1-GPU / 2-submitted-jobs limit does not matter. Re-running skips finished work.
LAMBDAS="${LAMBDAS:-0 0.3 1 3 10}"
SEEDS="${SEEDS:-0 1 2}"
DATA_DIR="${DATA_DIR:-data/synthetic/main_v0}"
DSET="$(basename "$DATA_DIR")"

for seed in $SEEDS; do
  for lam in $LAMBDAS; do
    run_id="${DSET}_lam${lam}_seed${seed}"
    out="results/raw/p5/${run_id}"

    if [ -d "$out/checkpoint" ]; then
      echo "=== skip training $run_id (checkpoint exists)"
    else
      echo "=== train $run_id"
      python p5_training_method/train.py --lambda "$lam" --seed "$seed" --run-id "$run_id" \
        --data "$DATA_DIR/train.jsonl" --log-every 50 --save-every 0
    fi

    if [ -f "$out/eval_${DSET}_dev/summary.json" ]; then
      echo "=== skip eval $run_id (summary exists)"
    else
      echo "=== eval dev $run_id"
      python p5_training_method/eval_head.py --checkpoint "$out/checkpoint" \
        --data "$DATA_DIR/eval.jsonl" --split dev
    fi
  done
done
echo "=== sweep finished"
