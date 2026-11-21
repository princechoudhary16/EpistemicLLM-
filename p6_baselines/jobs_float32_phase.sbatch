#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=p6_f32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Float32 re-run of one original P6 baseline. From the repo root:
#   mkdir -p logs && PHASE=base sbatch p6_baselines/jobs_float32_phase.sbatch   (then PHASE=sft, PHASE=idk)
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
: "${PHASE:?set PHASE=base|sft|idk}"
python -m p6_baselines.run_canonical_phase --phase "$PHASE"
python -m p2_real_qa.regrade --responses "results/raw/float32/p1_main_v0_${PHASE}.jsonl"
