#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=p5_final
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

# One-shot test on the FINAL split. Lambda=3 was chosen on the dev split (see
# results/final/p5_lambda_sweep_dev.json); lambda=0 is the plain-SFT control from the same code path.
# No training happens here: it scores checkpoints the dev sweep already produced.
# Finished evals are skipped, so a re-run never rescores or overwrites anything.
LAMBDAS="${LAMBDAS:-0 3}"
SEEDS="${SEEDS:-0 1 2}"
DATA_DIR="${DATA_DIR:-data/synthetic/main_v0}"
DSET="$(basename "$DATA_DIR")"

for seed in $SEEDS; do
  for lam in $LAMBDAS; do
    run_id="${DSET}_lam${lam}_seed${seed}"
    out="results/raw/p5/${run_id}"
    if [ -f "$out/eval_${DSET}_final/summary.json" ]; then
      echo "=== skip $run_id (final summary exists)"
      continue
    fi
    echo "=== eval final $run_id"
    python p5_training_method/eval_head.py --checkpoint "$out/checkpoint" \
      --data "$DATA_DIR/eval.jsonl" --split final --allow-final
  done
done
echo "=== final eval finished"
