#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=p3_locked
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Hidden-state probe vs log-prob on the TriviaQA locked test (run once). From the repo root:
#   mkdir -p logs && sbatch p2_abstention/jobs/probe_locked_test.sbatch
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
python -m p2_abstention.probe_locked_test extract
python -m p2_abstention.probe_locked_test analyse
