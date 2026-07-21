#!/bin/bash
# Train GAIL (N=100) on the noise-VARIANT demos (exact_uniform, coin_exclude,
# exact_exclude) — the missing GAIL modes that mirror PT's exact-% / flip.
# 0% is the shared clean baseline (already trained), so it is not repeated here.
#   bash scripts/submit_gail_noise_variants.sh                 # all 3 modes, 10 levels, 30 seeds
#   MODES="coin_exclude" LEVELS="20 40 60 80 100" NSEEDS=30 bash scripts/submit_gail_noise_variants.sh
set -euo pipefail

DATE=$(date +%Y-%m-%d)
N_DEMOS="${N_DEMOS:-100}"
DEMO_BATCH_SIZE="${DEMO_BATCH_SIZE:-1024}"    # use 128 for N=5 (demos are short)
INDEX=/home/marzii/IRL3/experiments/gail_noise_variants_N${N_DEMOS}_${DATE}.csv
mkdir -p "$(dirname "$INDEX")"
[ -f "$INDEX" ] || echo "condition_id,mode,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$INDEX"

DEMO_ROOT=/scratch/marzii/imitation_runs/noise_variants/demos
MODES="${MODES:-exact_uniform coin_exclude exact_exclude}"
LEVELS="${LEVELS:-10 20 30 40 50 60 70 80 90 100}"
NSEEDS="${NSEEDS:-30}"
cd /home/marzii/IRL3/imitation

n=0
for mode in $MODES; do
  for L in $LEVELS; do
    for s in $(seq 0 $((NSEEDS - 1))); do
      demo="$DEMO_ROOT/$mode/n100_p${L}_s${s}"
      [[ -d "$demo" ]] || { echo "MISSING demo: $demo (run: MODE=$mode sbatch generate_noise_demo_pools.sh)" 1>&2; exit 1; }
      jobid=$(DEMO_PATH="$demo" N_DEMOS="$N_DEMOS" DEMO_BATCH_SIZE="$DEMO_BATCH_SIZE" SHUFFLE=1 SEED="$s" \
              N_EVAL_EPISODES=50 FRAME_SKIP=0 \
              DEMO_NOTE="noise_variant $mode noise=${L}% seed=${s} N=${N_DEMOS}" \
              sbatch --parsable run_gail_lunarlander.sh)
      echo "${mode}_noise${L},${mode},${N_DEMOS},${L},${s},${jobid},${demo},submitted" >> "$INDEX"
      n=$((n + 1))
    done
  done
done
echo "Submitted $n GAIL jobs. Index: $INDEX"
