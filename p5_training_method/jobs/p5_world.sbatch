#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=p5_world
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# The chained scripts activate the environment in their own sub-shells; this script's own python calls need it too.
module load cuda/12.8.0
export CPATH="/apps/cuda_12.8.0/include:${CPATH:-}"
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate ~/envs/ntu-gpu
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Repeats the full P5 protocol on another synthetic world with NO re-tuning (lambda fixed in advance):
#   1. train + dev-eval  lambda in {0, 3} x seeds {0, 1, 2}        (p5_sweep.sbatch)
#   2. one-shot final-split eval of the same checkpoints            (p5_final_eval.sbatch)
#   3. fact-disjoint probe controls, dev then final                 (probe_controls.py)
# Each step skips work that already exists, so a re-run only fills gaps.
# Usage:  DATA_DIR=data/synthetic/main_w1 sbatch p5_training_method/jobs/p5_world.sbatch
DATA_DIR="${DATA_DIR:?Set DATA_DIR, e.g. data/synthetic/main_w1}"
DSET="$(basename "$DATA_DIR")"
export DATA_DIR LAMBDAS="${LAMBDAS:-0 3}" SEEDS="${SEEDS:-0 1 2}"

echo "=== [1/3] train + dev eval on $DSET"
bash p5_training_method/jobs/p5_sweep.sbatch
echo "=== [2/3] final-split eval on $DSET"
bash p5_training_method/jobs/p5_final_eval.sbatch

echo "=== [3/3] probe controls on $DSET"
python p5_training_method/probe_controls.py --dataset "$DSET" --out "results/final/p5_${DSET}_probe_controls_dev.json"
python p5_training_method/probe_controls.py --dataset "$DSET" --split final --allow-final \
  --out "results/final/p5_${DSET}_probe_controls_final.json"
echo "=== world run finished: $DSET"
