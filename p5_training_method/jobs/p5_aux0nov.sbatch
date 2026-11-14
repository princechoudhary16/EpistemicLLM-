#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:15:00
#SBATCH --job-name=p5_aux0nov
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

# Name-recognition control (see reports/p5_aux_zero_novel.md). Two model sets per world, both lambda=3:
#   <ds>_lam3halfdev_seed<S>    trained with the 0x prompts of the DEV facts only   (200 rows)
#   <ds>_lam3halffinal_seed<S>  trained with the 0x prompts of the FINAL facts only (200 rows)
# Scored on split S: the model trained on S's own prompts has seen the scored 0x names ("seen", 200 rows); the other
# model has not ("novel", 200 rows). Same amount of 0x supervision, so the two differ only in name novelty.
#   STAGE=dev   (default)  score both models on the dev split
#   STAGE=final            score both models on the final split; run only after the dev verdict is recorded
# Usage:  DATA_DIR=data/synthetic/main_v0 sbatch p5_training_method/jobs/p5_aux0nov.sbatch
DATA_DIR="${DATA_DIR:?Set DATA_DIR, e.g. data/synthetic/main_v0}"
DSET="$(basename "$DATA_DIR")"
SEEDS="${SEEDS:-0 1 2}"
STAGE="${STAGE:-dev}"
AUX="$DATA_DIR/zero_prompts/aux_zero_prompts.jsonl"

case "$STAGE" in
  dev)   SCORE=dev;   FINAL_FLAGS="" ;;
  final) SCORE=final; FINAL_FLAGS="--allow-final" ;;
  *) echo "unknown STAGE=$STAGE (use dev or final)"; exit 1 ;;
esac

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python p5_training_method/check_aux_novel.py --dataset "$DSET" --score-split "$SCORE"   # stops the job if not novel

for seed in $SEEDS; do
  for from in dev final; do
    run_id="${DSET}_lam3half${from}_seed${seed}"
    out="results/raw/p5/${run_id}"
    if [ -d "$out/checkpoint" ]; then
      echo "=== skip training $run_id (checkpoint exists)"
    else
      echo "=== train $run_id (0x prompts of the $from facts only)"
      python p5_training_method/train.py --lambda 3 --seed "$seed" --run-id "$run_id" \
        --data "$DATA_DIR/train.jsonl" --aux-zero "$AUX" --aux-zero-split "$from" --log-every 50
    fi
    if [ -f "$out/eval_${DSET}_${SCORE}/summary.json" ]; then
      echo "=== skip $SCORE eval $run_id (summary exists)"
    else
      echo "=== $SCORE eval $run_id"
      python p5_training_method/eval_head.py --checkpoint "$out/checkpoint" --data "$DATA_DIR/eval.jsonl" \
        --split "$SCORE" $FINAL_FLAGS
    fi
  done
done

for from in dev final; do
  python p5_training_method/probe_controls.py --dataset "$DSET" --p5-lambda "3half${from}" --seeds $SEEDS \
    --split "$SCORE" $FINAL_FLAGS --out "results/final/p5_${DSET}_aux0half-${from}_probe_controls_${SCORE}.json"
done
echo "=== aux0nov $STAGE stage finished: $DSET"
