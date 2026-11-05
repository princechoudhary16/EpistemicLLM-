#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=confidence_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Score one model's shared response files in float32, then calibrate them or apply a frozen policy.
# Submit from the repo root:  mkdir -p logs && RESPONSES="a.jsonl b.jsonl" NAME=run_name sbatch confidence_calibration/jobs/tc2_eval.sbatch
# Fine-tuned model:           also pass ADAPTER=path/to/adapter CHECKPOINT_ID=<checkpoint_id in the rows>
# Held-out evaluation:        also pass POLICY=path/to/frozen_policy.json (runs evaluate, which refuses to run twice)
# Resources stay inside QoS "normal" (10 CPU / 30G / 1 GPU / 6h) so a second job can still run.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-epistemic}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1            # compute nodes use the pre-downloaded model only
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

: "${RESPONSES:?set RESPONSES to the response JSONL files}"
: "${NAME:?set NAME to a run name}"
STAGE="$([ -n "${POLICY:-}" ] && echo held_out || echo development)"
SCORED="results/raw/confidence_calibration/${NAME}/scored_${STAGE}.jsonl"
OUTPUT="results/final/confidence_calibration/${NAME}/${STAGE}"
mkdir -p "$(dirname "$SCORED")"

SCORE_ARGS=(--batch-size 32)
[ -n "${ADAPTER:-}" ] && SCORE_ARGS+=(--adapter "$ADAPTER")
[ -n "${CHECKPOINT_ID:-}" ] && SCORE_ARGS+=(--checkpoint-id "$CHECKPOINT_ID")
# shellcheck disable=SC2086  # RESPONSES is a deliberate word list
python -m confidence_calibration score --responses $RESPONSES --output "$SCORED" "${SCORE_ARGS[@]}"

if [ -n "${POLICY:-}" ]; then
    python -m confidence_calibration evaluate --policy "$POLICY" --responses "$SCORED" --output-dir "$OUTPUT"
else
    python -m confidence_calibration calibrate --responses "$SCORED" --output-dir "$OUTPUT"
fi
