#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=p5_kq
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

# K(q)-target experiment (reports/p5_kq_target.md). Four lambda=3 student sets per world, 3 seeds; each gets aux loss only
# on the answer-training rows of ONE split's facts, so it is scored on the other split's facts, which it had no label for:
#   <ds>_lam3kfinal_seed<S>  target K(fact), labels on FINAL facts -> scored on dev
#   <ds>_lam3efinal_seed<S>  target s(e)    , labels on FINAL facts -> scored on dev     (matched control)
#   <ds>_lam3kdev_seed<S>    target K(fact), labels on DEV facts   -> scored on final
#   <ds>_lam3edev_seed<S>    target s(e)    , labels on DEV facts   -> scored on final    (matched control)
# Needs results/final/p5_kq_labels_<ds>.json from p5_kq_labels.sbatch.
#
# ARM=p1 (second arm, reports/p5_kq_target.md Amendment 1; main_v0 only): same masks and criteria, but the K target is P1's
# K per fact (python p5_training_method/p1_labels.py -> results/final/p5_kq_labels_main_v0_p1.json).  Trains only
#   <ds>_lam3pfinal_seed<S>  P1's K, labels on FINAL facts -> scored on dev
#   <ds>_lam3pdev_seed<S>    P1's K, labels on DEV facts   -> scored on final
# and reuses the matched s(e) controls (efinal / edev) from the default arm, which must already exist.
#   STAGE=dev   (default)  trains all four sets, scores the final-labelled ones on dev
#   STAGE=final            scores the dev-labelled ones on final; run only after the dev verdict is recorded in the report
# Usage:  DATA_DIR=data/synthetic/main_v0 sbatch p5_training_method/jobs/p5_kq.sbatch
DATA_DIR="${DATA_DIR:?Set DATA_DIR, e.g. data/synthetic/main_v0}"
DSET="$(basename "$DATA_DIR")"
SEEDS="${SEEDS:-0 1 2}"
STAGE="${STAGE:-dev}"
ARM="${ARM:-own}"
case "$ARM" in
  own) LABELS="results/final/p5_kq_labels_${DSET}.json"
       # tag : aux target : label split
       SETS="kfinal:k:final efinal:exposure:final kdev:k:dev edev:exposure:dev"
       DEV_TAGS="kfinal efinal"; FINAL_TAGS="kdev edev"
       HINT="run p5_kq_labels.sbatch first" ;;
  p1)  [ "$DSET" = "main_v0" ] || { echo "ARM=p1 exists for main_v0 only (P1 made no K file for $DSET)"; exit 1; }
       LABELS="results/final/p5_kq_labels_${DSET}_p1.json"
       SETS="pfinal:k:final pdev:k:dev"
       DEV_TAGS="pfinal"; FINAL_TAGS="pdev"
       HINT="run: python p5_training_method/p1_labels.py"
       for c in efinal edev; do   # the matched controls come from the default arm
         [ -d "results/raw/p5/${DSET}_lam3${c}_seed0/checkpoint" ] || { echo "missing control ${DSET}_lam3${c} (run the default arm first)"; exit 1; }
       done ;;
  *) echo "unknown ARM=$ARM (use own or p1)"; exit 1 ;;
esac
[ -f "$LABELS" ] || { echo "missing $LABELS ($HINT)"; exit 1; }

case "$STAGE" in
  dev)   SCORE=dev;   SCORED_TAGS="$DEV_TAGS";   FINAL_FLAGS="" ;;
  final) SCORE=final; SCORED_TAGS="$FINAL_TAGS"; FINAL_FLAGS="--allow-final" ;;
  *) echo "unknown STAGE=$STAGE (use dev or final)"; exit 1 ;;
esac

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

for seed in $SEEDS; do
  for spec in $SETS; do
    IFS=: read -r tag target lsplit <<< "$spec"
    run_id="${DSET}_lam3${tag}_seed${seed}"
    out="results/raw/p5/${run_id}"
    if [ -d "$out/checkpoint" ]; then
      echo "=== skip training $run_id (checkpoint exists)"
    else
      echo "=== train $run_id (aux target $target, labels on $lsplit facts only)"
      python p5_training_method/train.py --lambda 3 --seed "$seed" --run-id "$run_id" \
        --data "$DATA_DIR/train.jsonl" --aux-target "$target" --k-labels "$LABELS" \
        --aux-label-split "$lsplit" --log-every 50
    fi
  done
  for tag in $SCORED_TAGS; do
    run_id="${DSET}_lam3${tag}_seed${seed}"
    out="results/raw/p5/${run_id}"
    if [ -f "$out/eval_${DSET}_${SCORE}/summary.json" ]; then
      echo "=== skip $SCORE eval $run_id (summary exists)"
    else
      echo "=== $SCORE eval $run_id"
      python p5_training_method/eval_head.py --checkpoint "$out/checkpoint" --data "$DATA_DIR/eval.jsonl" \
        --split "$SCORE" $FINAL_FLAGS
    fi
  done
done

for tag in $SCORED_TAGS; do
  python p5_training_method/probe_controls.py --dataset "$DSET" --p5-lambda "3${tag}" --seeds $SEEDS \
    --split "$SCORE" $FINAL_FLAGS --out "results/final/p5_${DSET}_kq-${tag}_probe_controls_${SCORE}.json"
done
echo "=== kq $STAGE stage finished: $DSET"
