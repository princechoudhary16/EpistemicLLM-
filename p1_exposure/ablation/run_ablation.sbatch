#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=p1_ablation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# P1 ablation on one TC2 GPU (QoS "normal": 1 GPU, 6h max, so a second job can still run).
# Modeled on confidence_calibration/jobs/tc2_eval.sbatch. Submit from the repo root:
#
#   mkdir -p logs && sbatch p1_exposure/ablation/run_ablation.sbatch
#
# Overrides (environment variables):
#   VERSION=main_v0            dataset version under data/synthetic (default pilot_v0)
#   STAGE=idk                  also run the IDK ordering variants (default: recipes only)
#   CONDA_ENV=torch            conda env with torch, transformers, peft (default: torch,
#                              the env documented in reports/compute_budget.md)
#
# main_v0 trains 3,800 steps per recipe at batch size 1, so raise --time (e.g. 05:00:00).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-torch}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1            # compute nodes use the pre-downloaded, revision-pinned model only
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

VERSION="${VERSION:-pilot_v0}"
OUT="results/raw/p1_ablation_${VERSION}_${SLURM_JOB_ID}"
EXTRA=()
if [ "${STAGE:-}" = "idk" ]; then
    EXTRA+=(--idk-recipes p6,mine --idk-variants end,shuffled)
fi

# bfloat16 is selected automatically on the A40 (compute capability 8.x), matching P6's recipe.
python p1_exposure/ablation/run_ablation.py \
    --data-dir data/synthetic --version "$VERSION" --out-dir "$OUT" \
    --gen-batch 64 ${EXTRA[@]+"${EXTRA[@]}"}

echo "results: $OUT/summary.md"
