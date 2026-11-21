#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=abst_locked
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Locked-test answers (run ONCE, after the freeze): the base model, then every trained arm.
#   SEEDS="0 1" sbatch p2_abstention/jobs/locked_test.sbatch      then      SEEDS="2" sbatch ...
# TAG=own (default) uses results/final/abstention_own/; TAG= (empty) uses round 1's adapters.
# Resumable: finished questions are skipped.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-epistemic}"
set -u
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
TAG="${TAG-own}"
ADROOT="results/final/abstention${TAG:+_$TAG}"
M="abstention${TAG:+_$TAG}"
echo "=== base ($(date)) ===" >&2
python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split locked_test --device cuda
for SEED in ${SEEDS:-0 1 2}; do
  for ARM in ${ARMS:-geode random rtuning kfilter kthresh ksoft}; do
    AD="$ADROOT/${ARM}_seed${SEED}_adapter"
    [ -f "$AD/train_meta.json" ] || { echo "missing $AD" >&2; exit 1; }
    echo "=== $ARM seed $SEED ($(date)) ===" >&2
    python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split locked_test --device cuda \
        --adapter "$AD" --method "${M}_${ARM}_seed${SEED}" --training-seed "$SEED"
  done
done
