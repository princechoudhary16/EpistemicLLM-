#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --job-name=p6_baselines_main
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
module load cuda/12.8.0
export CPATH="/apps/cuda_12.8.0/include:$CPATH"
eval "$(conda shell.bash hook)"
conda activate torch

cd "$SLURM_SUBMIT_DIR" || exit 1
python -m p6_baselines.run_canonical_baselines --dataset-version main_v0
