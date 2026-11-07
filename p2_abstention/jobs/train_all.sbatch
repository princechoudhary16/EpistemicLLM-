#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=abst_train_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# All arms for ONE seed in one job (TC2 allows only 2 submitted jobs per user):
#   mkdir -p logs && SEED=1 sbatch p2_abstention/jobs/train_all.sbatch
#   mkdir -p logs && SEED=2 sbatch p2_abstention/jobs/train_all.sbatch
# Resumable: arms whose adapter exists are not retrained; finished validation answers are skipped.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-epistemic}"
set -u
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv || true
: "${SEED:?set SEED}"
for ARM in ${ARMS:-geode random rtuning}; do          # small arms first, so a time-out loses the least
    AD="results/final/abstention/${ARM}_seed${SEED}_adapter"
    echo "=== $ARM seed $SEED ($(date)) ===" >&2
    [ -f "$AD/train_meta.json" ] || python -m p2_abstention.train --arm "$ARM" --seed "$SEED" --device cuda
    python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split validation --device cuda \
        --adapter "$AD" --method "abstention_${ARM}_seed${SEED}" --training-seed "$SEED"
done
