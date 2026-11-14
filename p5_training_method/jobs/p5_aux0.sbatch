#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=p5_aux0
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

# Lambda=3 with P1's answer-masked 0x prompts as extra head-only rows (see reports/p5_aux_zero.md).
# Run ids are <dataset>_lam3aux0_seed<S>, so no earlier run is touched. lambda=0 needs no new runs: aux rows add
# nothing to it, so the existing lambda=0 checkpoints are the matched control.
#   STAGE=dev   (default)  train + dev eval + dev probe controls
#   STAGE=final            final-split eval + final probe controls; run only after the dev criteria are judged
# Usage:  DATA_DIR=data/synthetic/main_v0 sbatch p5_training_method/jobs/p5_aux0.sbatch
DATA_DIR="${DATA_DIR:?Set DATA_DIR, e.g. data/synthetic/main_v0}"
DSET="$(basename "$DATA_DIR")"
SEEDS="${SEEDS:-0 1 2}"
STAGE="${STAGE:-dev}"
AUX="$DATA_DIR/zero_prompts/aux_zero_prompts.jsonl"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

for seed in $SEEDS; do
  run_id="${DSET}_lam3aux0_seed${seed}"
  out="results/raw/p5/${run_id}"
  if [ "$STAGE" = "dev" ]; then
    if [ -d "$out/checkpoint" ]; then
      echo "=== skip training $run_id (checkpoint exists)"
    else
      echo "=== train $run_id"
      python p5_training_method/train.py --lambda 3 --seed "$seed" --run-id "$run_id" \
        --data "$DATA_DIR/train.jsonl" --aux-zero "$AUX" --log-every 50
    fi
    if [ -f "$out/eval_${DSET}_dev/summary.json" ]; then
      echo "=== skip dev eval $run_id (summary exists)"
    else
      echo "=== dev eval $run_id"
      python p5_training_method/eval_head.py --checkpoint "$out/checkpoint" --data "$DATA_DIR/eval.jsonl" --split dev
    fi
  elif [ "$STAGE" = "final" ]; then
    if [ -f "$out/eval_${DSET}_final/summary.json" ]; then
      echo "=== skip final eval $run_id (summary exists)"
    else
      echo "=== final eval $run_id"
      python p5_training_method/eval_head.py --checkpoint "$out/checkpoint" --data "$DATA_DIR/eval.jsonl" \
        --split final --allow-final
    fi
  else
    echo "unknown STAGE=$STAGE (use dev or final)"; exit 1
  fi
done

if [ "$STAGE" = "dev" ]; then
  python p5_training_method/probe_controls.py --dataset "$DSET" --p5-lambda 3aux0 --seeds $SEEDS \
    --out "results/final/p5_${DSET}_aux0_probe_controls_dev.json"
else
  python p5_training_method/probe_controls.py --dataset "$DSET" --p5-lambda 3aux0 --seeds $SEEDS \
    --split final --allow-final --out "results/final/p5_${DSET}_aux0_probe_controls_final.json"
fi
echo "=== aux0 $STAGE stage finished: $DSET"
