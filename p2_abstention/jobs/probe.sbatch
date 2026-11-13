#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=abst_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# GeoDe's probe for ONE layer: last-prompt-token features for D0 and D1, then two fits on D0 (plain and
# class-balanced), each writing distances for D1. Needs both label jobs finished. From the repo root:
#   mkdir -p logs && LAYER=-1 sbatch p2_abstention/jobs/probe.sbatch     # final layer (GeoDe's setting)
#   mkdir -p logs && LAYER=18 sbatch p2_abstention/jobs/probe.sbatch     # P3's best layer
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
LAYER="${LAYER:--1}"                                   # -1 = final layer (GeoDe TBG); 18 = P3's best
TAG=$([ "$LAYER" = "-1" ] && echo final || echo "L$LAYER")
D=data/abstention/triviaqa_train_v0
F=results/abstention/features
R=results/raw/p2_abstention
for S in probe sft; do
    [ -f "$F/${S}_base_${TAG}.npz" ] || python -m p2_abstention.probe extract --questions "$D/$S.jsonl" \
        --out "$F/${S}_base_${TAG}.npz" --layer "$LAYER" --device cuda
done
for W in plain balanced; do
    python -m p2_abstention.probe fit --train-features "$F/probe_base_${TAG}.npz" --train-responses "${R}_probe_base.jsonl" \
        --apply-features "$F/sft_base_${TAG}.npz" --apply-responses "${R}_sft_base.jsonl" \
        $([ "$W" = balanced ] && echo --balanced) --out "results/abstention/geode_distances_${TAG}_${W}.jsonl"
done
