#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=locked_other
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Locked-test answers for any adapters, as NAME=ADAPTER_FOLDER pairs:
#   RUNS="sft_matched=path/to/adapter p5_lam3=path/to/checkpoint/adapter" sbatch p2_abstention/jobs/locked_test_other.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-epistemic}"
set -u
export HF_HOME="${HF_HOME:-$HOME/hf_cache}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${RUNS:?set RUNS to NAME=ADAPTER pairs}"
for PAIR in $RUNS; do
    NAME="${PAIR%%=*}"; AD="${PAIR#*=}"
    [ -d "$AD" ] || { echo "missing adapter folder $AD" >&2; exit 1; }
    echo "=== $NAME from $AD ($(date)) ===" >&2
    python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split locked_test --device cuda \
        --adapter "$AD" --method "$NAME"
done
