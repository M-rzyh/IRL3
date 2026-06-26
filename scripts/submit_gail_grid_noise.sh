#!/bin/bash
# Noise axis only: noise ∈ {10..100}%, N=100, 30 seeds each = 360 jobs (online noise demos)

set -euo pipefail

DATE=$(date +%Y-%m-%d)
INDEX=/home/marzii/IRL3/experiments/gail_grid_noise_${DATE}.csv
mkdir -p "$(dirname "$INDEX")"
echo "condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"

DEMO_ROOT=/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187

declare -a noise_axis_pct=(10 20 25 30 40 50 60 70 75 80 90 100)

cd /home/marzii/IRL3/imitation

submit_one () {
  local cond_id="$1"
  local n_demos="$2"
  local noise_pct="$3"
  local seed="$4"
  local demo_path="$5"
  local shuffle="$6"
  local bs=1024

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
