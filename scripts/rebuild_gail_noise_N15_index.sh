#!/bin/bash
# Rebuild experiments/gail_grid_noise_N15_<date>.csv after the index was truncated by a
# second `> $INDEX` submit. The job->(noise,seed) map is recovered from each job's log
# DEMO_NOTE ("cond=noise_N15_p{P} ... seed={S}"), whose filename is the slurm job id.
#
# IDEMPOTENT + PARTIAL: only jobs that have STARTED have logs, so run this again after the
# sweep finishes to capture the rest. The clean 0% rows come from a fixed side-file (their
# job ids are known at submit time), not from logs.
#
#   bash scripts/rebuild_gail_noise_N15_index.sh [zero_rows.csv]
set -euo pipefail

DATE=$(date +%Y-%m-%d)
OUT=/home/marzii/IRL3/experiments/gail_grid_noise_N15_${DATE}.csv
LOGDIR=/home/marzii/IRL3/imitation/output/slurm_logs/gail/lunarlander
ZERO=${1:-}
CLEAN=/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187/n100_p0_clean
NONFS=/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187

TMP=$(mktemp)
echo "condition_id,N,noise_pct,seed,slurm_job_id,demo_path,status" > "$TMP"

# --- noise rows (10-100%): scan logs for this batch's DEMO_NOTE ---
n_noise=0
for f in "$LOGDIR"/gail-lunarlander_*.out; do
    [ -e "$f" ] || continue
    note=$(grep -oE "cond=noise_N15_p[0-9]+ N=15 noise=[0-9]+% seed=[0-9]+" "$f" 2>/dev/null | head -1) || true
    [ -z "$note" ] && continue
    jid=$(basename "$f" | grep -oE '[0-9]+')
    p=$(echo "$note"    | grep -oE 'noise=[0-9]+' | grep -oE '[0-9]+')
    s=$(echo "$note"    | grep -oE 'seed=[0-9]+'  | grep -oE '[0-9]+')
    [ "$p" = "0" ] && continue    # 0% comes from the side-file, not logs
    echo "noise_N15_p${p},15,${p},${s},${jid},${NONFS}/n100_p${p}_s${s},submitted" >> "$TMP"
    n_noise=$((n_noise+1))
done

# --- clean 0% rows from the side-file (job ids known deterministically) ---
n_zero=0
if [ -n "$ZERO" ] && [ -f "$ZERO" ]; then
    cat "$ZERO" >> "$TMP"; n_zero=$(wc -l < "$ZERO")
fi

# dedup by slurm_job_id (keep first), sort by (noise, seed), keep header on top
{ head -1 "$TMP"; tail -n +2 "$TMP" | sort -t, -k5,5 -u | sort -t, -k3,3n -k4,4n; } > "$OUT"
rm -f "$TMP"

rows=$(($(wc -l < "$OUT") - 1))
echo "Wrote $OUT"
echo "  noise rows recovered from logs: $n_noise   |   clean 0% rows: $n_zero   |   total: $rows"
echo "  (expected final: 300 noise + 30 zero = 330; re-run after the sweep if short)"
python3 - "$OUT" <<'EOF'
import csv,sys
from collections import Counter
rows=list(csv.DictReader(open(sys.argv[1])))
c=Counter(int(r['noise_pct']) for r in rows)
print("  per-level counts:", {k:c[k] for k in sorted(c)})
EOF
