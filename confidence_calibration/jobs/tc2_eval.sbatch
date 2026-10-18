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
# Generate, grade, score and report one question file on a TC2 GPU node.
# Submit from the repo root:  mkdir -p logs && sbatch confidence_calibration/jobs/tc2_eval.sbatch
# Override inputs:            QUESTIONS=path.jsonl RUN_ID=name sbatch confidence_calibration/jobs/tc2_eval.sbatch
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

QUESTIONS="${QUESTIONS:-data/triviaqa/validation_sample_n300_seed0.jsonl}"
RUN_ID="${RUN_ID:-$(basename "$QUESTIONS" .jsonl)-cuda-${SLURM_JOB_ID}}"
python -m confidence_calibration run --questions "$QUESTIONS" --run-id "$RUN_ID" \
    --batch-size 64 --score-batch-size 32 --verify-scoring
