#!/bin/bash
# Submit the 90-job GAIL grid: 9 conditions × 10 seeds.
#   Count axis (noise=0%):   N ∈ {1, 5, 10, 50, 100}  (5 conditions)
#   Noise axis (N=100):      noise ∈ {25, 50, 75, 100}%  (4 conditions)
# All using per-frame demos derived from expert 4720242 (FRAME_SKIP=0 for GAIL training).
#
# Writes /home/marzii/IRL3/experiments/gail_grid_<date>.csv as the index.

set -euo pipefail

DATE=$(date +%Y-%m-%d)
INDEX=/home/marzii/IRL3/experiments/gail_grid_${DATE}.csv
mkdir -p "$(dirname "$INDEX")"
echo "condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"

# DEMO_ROOT=/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4720242
DEMO_ROOT=/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187
CLEAN_DEMO=$DEMO_ROOT/n100_p0_clean

declare -a count_axis_N=(1 5 10 50 100)
declare -a noise_axis_pct=(10 20 25 30 40 50 60 70 75 80 90 100)

cd /home/marzii/IRL3/imitation

submit_one () {
  local cond_id="$1"
  local n_demos="$2"
  local noise_pct="$3"
  local seed="$4"
  local demo_path="$5"
  local shuffle="$6"
  # Choose a safe batch size given N transitions; perframe demos have ~300 transitions/ep
  local bs=1024
  if [ "$n_demos" -le 1 ]; then
    bs=128
  elif [ "$n_demos" -le 5 ]; then
    bs=512
  fi

  local jobid
  jobid=$(DEMO_PATH="$demo_path" \
          N_DEMOS="$n_demos" \
          DEMO_BATCH_SIZE="$bs" \
          SHUFFLE="$shuffle" \
          SEED="$seed" \
          FRAME_SKIP=0 \
          DEMO_NOTE="grid cond=${cond_id} N=${n_demos} noise=${noise_pct}% seed=${seed}" \
          sbatch --parsable run_gail_lunarlander.sh)
  echo "${cond_id},${n_demos},${noise_pct},${seed},${jobid},${demo_path},submitted" >> "$INDEX"
  echo "  ${cond_id} N=${n_demos} noise=${noise_pct}% seed=${seed} → job ${jobid}"
}

echo "=== Count axis (noise=0%) ==="
for N in "${count_axis_N[@]}"; do
  cond="count_N${N}"
  for seed in {0..29}; do
    submit_one "$cond" "$N" 0 "$seed" "$CLEAN_DEMO" 1
  done
done

echo ""
echo "=== Noise axis (N=100, varying noise) ==="
for noise in "${noise_axis_pct[@]}"; do
  cond="noise_p${noise}"
  for seed in {0..29}; do
    demo_path="$DEMO_ROOT/n100_p${noise}_s${seed}"
    submit_one "$cond" 100 "$noise" "$seed" "$demo_path" 0
  done
done

echo ""
echo "Index written: $INDEX"
wc -l "$INDEX"
