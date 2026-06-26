#!/bin/bash
#SBATCH --job-name=gen-online-noise-nonfs
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187/generate_%j.log

set -euo pipefail

PYTHON=/scratch/marzii/envs/imitation-gail/bin/python
SCRIPT=/home/marzii/IRL3/scripts/collect_fs_expert_perframe_demos_online_noise.py
POLICY=/scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip
OUTROOT=/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187

mkdir -p "$OUTROOT"

declare -A PROB=(
  [10]=0.10 [20]=0.20 [25]=0.25 [30]=0.30 [40]=0.40 [50]=0.50
  [60]=0.60 [70]=0.70 [75]=0.75 [80]=0.80 [90]=0.90 [100]=1.0
)

total=0
for noise in 10 20 25 30 40 50 60 70 75 80 90 100; do
  for seed in {0..29}; do
    OUT="$OUTROOT/n100_p${noise}_s${seed}"
    echo "=== noise=${noise}% seed=${seed} -> $OUT ==="
    OPENBLAS_NUM_THREADS=1 $PYTHON $SCRIPT \
      --policy      "$POLICY" \
      --output      "$OUT" \
      --optimal-policy "$POLICY" \
      --noise-prob  "${PROB[$noise]}" \
      --noise-seed  "$seed" \
      --n-episodes  100 \
      --hold-k      1 \
      --max-frames  400 \
      --n-actions   4
    total=$((total + 1))
    echo "[done $total/360]"
  done
done

echo ""
echo "All $total noisy demo pools written to $OUTROOT"
