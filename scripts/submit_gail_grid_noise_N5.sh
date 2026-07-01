#!/bin/bash
# LOW-BUDGET noise axis: N=5 demos, noise ∈ {10..90}%, 30 seeds, online nonFS
# demos (expert 4615187). Mirrors submit_gail_grid_noise.sh but N_DEMOS=5 and
# SHUFFLE=1 (each seed draws a random 5-subset from its 100-episode noisy pool,
# matching how count_N5 subsamples the clean pool).
#
#   bash scripts/submit_gail_grid_noise_N5.sh
#
# noise=0 is NOT submitted here — reuse the existing count_N5 condition (clean,
# N=5, 30 seeds, expert 4615187) in gail_grid_count_2026-06-18.csv as the 0% pt.
# Add 100 to noise_axis_pct below if you want the endpoint (recommended).

set -euo pipefail

DATE=$(date +%Y-%m-%d)
INDEX=/home/marzii/IRL3/experiments/gail_grid_noise_N5_${DATE}.csv
mkdir -p "$(dirname "$INDEX")"
echo "condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"

DEMO_ROOT=/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187

declare -a noise_axis_pct=(10 20 30 40 50 60 70 80 90 100)

cd /home/marzii/IRL3/imitation

submit_one () {
  local cond_id="$1" n_demos="$2" noise_pct="$3" seed="$4" demo_path="$5" shuffle="$6"
  local bs=128   # demo_batch_size: must be <= 5-demo transitions; high-noise demos are
                 # short (~460 transitions at 100%), so 512 failed there. 128 fits all levels.
  local jobid
  jobid=$(DEMO_PATH="$demo_path" \
          N_DEMOS="$n_demos" \
          DEMO_BATCH_SIZE="$bs" \
          SHUFFLE="$shuffle" \
          SEED="$seed" \
          FRAME_SKIP=0 \
          DEMO_NOTE="lowN grid cond=${cond_id} N=${n_demos} noise=${noise_pct}% seed=${seed}" \
          sbatch --parsable run_gail_lunarlander.sh)
  echo "${cond_id},${n_demos},${noise_pct},${seed},${jobid},${demo_path},submitted" >> "$INDEX"
  echo "  ${cond_id} N=${n_demos} noise=${noise_pct}% seed=${seed} → job ${jobid}"
}

echo "=== Low-budget noise axis (N=5, varying noise) ==="
for noise in "${noise_axis_pct[@]}"; do
  cond="noise_N5_p${noise}"
  for seed in {0..29}; do
    submit_one "$cond" 5 "$noise" "$seed" "$DEMO_ROOT/n100_p${noise}_s${seed}" 1
  done
done

echo ""; echo "Index written: $INDEX"; wc -l "$INDEX"
