#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=p6_smoke
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
module load cuda/12.8.0
# Triton (used internally by this torch build for RoPE) JIT-compiles a small CUDA
# helper and needs cuda.h on the compiler include path; the cuda module itself
# does not add this (only PATH/LD_LIBRARY_PATH).
export CPATH="/apps/cuda_12.8.0/include:$CPATH"
eval "$(conda shell.bash hook)"
conda activate torch

cd "$SLURM_SUBMIT_DIR" || exit 1
python -m p6_baselines.generate_base \
    --fixture p6_baselines/smoke_fixture.jsonl \
    --out results/raw/smoke_base_v0.jsonl \
    --run-id smoke_base_v0 --seed 0
