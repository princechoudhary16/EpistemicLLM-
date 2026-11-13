#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=abst_label
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Greedy-only answers from the BASE model, graded with v5: the labels for R-Tuning and GeoDe's probe.
# Submit from the repo root:
#   mkdir -p logs && SET=probe sbatch p2_abstention/jobs/label.sbatch      # 5,000 questions
#   mkdir -p logs && SET=sft   sbatch p2_abstention/jobs/label.sbatch      # 20,000 questions
# Resumable: if it hits the 6 h limit, submit the same command again; finished questions are skipped.
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
: "${SET:?set SET to probe or sft}"
N="${N_SAMPLES:-0}"                                     # 0 = greedy only; 5 = greedy + 5 samples (K)
TAG=$([ "$N" = 0 ] && echo base || echo "base_n$N")
python -m p2_real_qa.generate_responses --questions "data/abstention/triviaqa_train_v0/${SET}.jsonl" \
    --out "results/raw/p2_abstention_${SET}_${TAG}.jsonl" --run-id "p2_abstention_${SET}_${TAG}" \
    --n-samples "$N" --device cuda --chunk 50
