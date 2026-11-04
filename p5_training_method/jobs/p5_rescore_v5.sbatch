#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=p5_rescore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda/12.8.0
export CPATH="/apps/cuda_12.8.0/include:${CPATH:-}"
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate ~/envs/ntu-gpu
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Re-scores every P5 experiment after regrade_v5.py --apply (reports/p5_kq_target.md, Amendment 2). Nothing is trained or
# generated: the same saved answers are re-scored, with the v5 labels, into the SAME output files. The pre-regrade copies of
# results/final are in results/superseded_containment_grader_final/ (made before the first rerun).
# K(q) is not here: it was re-scored with p5_kq.sbatch (no verdict changed).
#   1. probe controls (GPU):  base runs, aux-zero, name-recognition control   x  both worlds  x  dev and final
#   2. summaries (CPU):       lambda sweep tables, abstention evaluation, aux-zero criteria, name-recognition criteria
# Usage:  sbatch p5_training_method/jobs/p5_rescore_v5.sbatch

# Stop unless the regrade is applied everywhere (every predictions.jsonl has its predictions_containment.jsonl backup).
n_pred=$(ls results/raw/p5/*/eval_*/predictions.jsonl | wc -l)
n_back=$(ls results/raw/p5/*/eval_*/predictions_containment.jsonl | wc -l)
echo "predictions: $n_pred, containment backups: $n_back"
[ "$n_pred" -eq "$n_back" ] || { echo "regrade not applied everywhere: run regrade_v5.py --apply first"; exit 1; }

PC="python p5_training_method/probe_controls.py"
F="--allow-final"

echo "=== 1/2 probe controls"
# base lambda 0 / 3 runs (main_v0 uses the default output names)
$PC --dataset main_v0 --out results/final/p5_probe_controls_dev.json
$PC --dataset main_v0 --split final $F --out results/final/p5_probe_controls_final.json
$PC --dataset main_w1 --out results/final/p5_main_w1_probe_controls_dev.json
$PC --dataset main_w1 --split final $F --out results/final/p5_main_w1_probe_controls_final.json

for ds in main_v0 main_w1; do
  # aux-zero (0x prompts as head-only rows)
  $PC --dataset $ds --p5-lambda 3aux0 --out results/final/p5_${ds}_aux0_probe_controls_dev.json
  $PC --dataset $ds --p5-lambda 3aux0 --split final $F --out results/final/p5_${ds}_aux0_probe_controls_final.json
  # name-recognition control: half-dev and half-final sets, each scored on both splits
  for from in dev final; do
    $PC --dataset $ds --p5-lambda 3half${from} --out results/final/p5_${ds}_aux0half-${from}_probe_controls_dev.json
    $PC --dataset $ds --p5-lambda 3half${from} --split final $F --out results/final/p5_${ds}_aux0half-${from}_probe_controls_final.json
  done
done

bash p5_training_method/jobs/p5_rescore_v5_summaries.sh
echo "=== rescore finished"
