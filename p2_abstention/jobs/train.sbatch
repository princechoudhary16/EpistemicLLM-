#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=abst_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One arm, one seed:  mkdir -p logs && ARM=rtuning SEED=0 sbatch p2_abstention/jobs/train.sbatch
# Then the validation answers for that model (6 per question, the shared protocol) in the same job.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-epistemic}"
set -u
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1            # compute nodes use the pre-downloaded model only
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv || true
: "${ARM:?set ARM to rtuning, geode or random}"
SEED="${SEED:-0}"
python -m p2_abstention.train --arm "$ARM" --seed "$SEED" --device cuda
python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split validation --device cuda \
    --adapter "results/final/abstention/${ARM}_seed${SEED}_adapter" --method "abstention_${ARM}_seed${SEED}" \
    --training-seed "$SEED"
