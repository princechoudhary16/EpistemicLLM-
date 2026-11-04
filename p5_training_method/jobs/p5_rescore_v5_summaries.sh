#!/bin/bash
# CPU-only second half of p5_rescore_v5.sbatch: summaries and criteria on the re-scored probe-control files.
# Runs on the login node (no GPU): bash p5_training_method/jobs/p5_rescore_v5_summaries.sh
set -eo pipefail
cd "$(dirname "$0")/../.."
echo "=== summaries"
AGG="python p5_training_method/aggregate_sweep.py"
$AGG --dataset main_v0 --split dev --out results/final/p5_lambda_sweep_dev.json
$AGG --dataset main_v0 --split final --out results/final/p5_lambda_sweep_final.json
$AGG --dataset main_w1 --split dev --out results/final/p5_main_w1_lambda_sweep_dev.json
$AGG --dataset main_w1 --split final --out results/final/p5_main_w1_lambda_sweep_final.json

AB="python p5_training_method/abstain_eval.py"
for pen in 1 3; do
  suf=$([ "$pen" = 1 ] && echo "" || echo "_pen3")
  $AB --penalty $pen --out results/final/p5_abstain_eval${suf}.json
  $AB --penalty $pen --dev results/final/p5_main_w1_signal_scores_dev.jsonl --final results/final/p5_main_w1_signal_scores_final.jsonl \
      --out results/final/p5_main_w1_abstain_eval${suf}.json
  for ds in main_v0 main_w1; do
    $AB --penalty $pen --dev results/final/p5_${ds}_aux0_signal_scores_dev.jsonl --final results/final/p5_${ds}_aux0_signal_scores_final.jsonl \
        --out results/final/p5_${ds}_aux0_abstain_eval${suf}.json
  done
done

python p5_training_method/aux0_criteria.py --split dev
python p5_training_method/aux0_criteria.py --split final
python p5_training_method/aux0nov_criteria.py --split dev
python p5_training_method/aux0nov_criteria.py --split final
echo "=== summaries finished"
