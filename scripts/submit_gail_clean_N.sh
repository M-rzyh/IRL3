#!/bin/bash
# Submit the CLEAN (0% noise) GAIL point for a given demo budget N, matching the
# nonFS noise sweeps (expert 4615187). Use this for the 0% column of any
# gail_grid_noise_N{N} plot — do NOT use the noise submitter for 0% (its per-seed
# pool pattern n100_p0_s{seed} does not exist; clean is a single pool).
#
#   bash scripts/submit_gail_clean_N.sh <N> [INDEX_CSV]
#
# Example (this is exactly how N=15's 0% was submitted):
#   bash scripts/submit_gail_clean_N.sh 15 experiments/gail_grid_noise_N15_$(date +%F).csv
#
# Design choices that avoid the traps we hit:
#   * expert 4615187 (SAME as the noise pools) — a 0% point from a different expert
#     would confound the noise axis.
#   * the real clean pool n100_p0_clean (150 eps); clean is ONE pool, not per-seed,
#     so each seed draws a different 15 via SHUFFLE_SEED=$seed.
#   * APPENDS to the index (header only if new) — never truncates.
#   * fails fast if the pool is missing.
set -euo pipefail

N=${1:?usage: submit_gail_clean_N.sh <N> [INDEX_CSV]}
DATE=$(date +%Y-%m-%d)
INDEX=${2:-/home/marzii/IRL3/experiments/gail_grid_noise_N${N}_${DATE}.csv}
NSEEDS=${NSEEDS:-30}

CLEAN=/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187/n100_p0_clean
[ -d "$CLEAN" ] || { echo "MISSING clean pool: $CLEAN" 1>&2; exit 1; }

mkdir -p "$(dirname "$INDEX")"
[ -s "$INDEX" ] || echo "condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"

# batch size: high-noise runs need 128; clean demos are long, but keep 128 to match the
# rest of the N-sweep (a plot mixes these rows).
BS=${DEMO_BATCH_SIZE:-128}
cd /home/marzii/IRL3/imitation

echo "=== CLEAN 0% point: N=$N, $NSEEDS seeds, expert 4615187 ==="
for seed in $(seq 0 $((NSEEDS - 1))); do
  jobid=$(DEMO_PATH="$CLEAN" N_DEMOS="$N" DEMO_BATCH_SIZE="$BS" SHUFFLE=1 SHUFFLE_SEED="$seed" \
          SEED="$seed" FRAME_SKIP=0 \
          DEMO_NOTE="lowN grid cond=noise_N${N}_p0 N=${N} noise=0% seed=${seed}" \
          sbatch --parsable run_gail_lunarlander.sh)
  echo "noise_N${N}_p0,${N},0,${seed},${jobid},${CLEAN},submitted" >> "$INDEX"
  printf "  0%% seed=%2d -> job %s\n" "$seed" "$jobid"
done
echo "Appended $NSEEDS clean rows to $INDEX"
