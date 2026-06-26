#!/bin/bash
# ============================================================================
# Collect human demonstrations for LunarLander-v2 (GAIL-compatible format)
#
# This script runs INTERACTIVELY — do NOT sbatch it.
#
# HOW TO USE:
#   1. Get an interactive session:
#        salloc --account=aip-mtaylor3 --cpus-per-task=2 --mem=4G --time=01:00:00
#
#   2. Run this script:
#        bash collect_human_demos_lunarlander.sh
#
#   3. It automatically finds a free port and starts a web UI.
#      The script prints the exact SSH tunnel command and URL.
#      Open a NEW terminal and run that SSH tunnel command, then open
#      the URL in your browser.
#
#   4. In the browser you see the full LunarLander game. Controls:
#        W / ↑     = main engine
#        A / ←     = left engine
#        D / →     = right engine
#        S / ↓     = coast (noop)
#        SPACE     = end episode & save
#        X         = discard episode
#        Q / ESC   = quit & save all
#
#   5. Demos saved in HuggingFace Arrow format (same as expert demos).
#
# CONFIGURATION (env vars):
#   MAX_EPISODES=50    Number of episodes to collect
#   FPS=15             Game speed (lower = easier to control)
#   PORT=0             0 = auto-find free port (default)
#   OUTPUT_DIR=...     Where to save
# ============================================================================

set -euo pipefail

# ---- Configuration ----
MAX_EPISODES=${MAX_EPISODES:-50}
FPS=${FPS:-15}
SEED=${SEED:-42}

OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/imitation_runs/human_demos/lunarlander}"
mkdir -p "$OUTPUT_DIR"

# ---- Find a free port ----
if [ "${PORT:-0}" = "0" ]; then
    PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    echo "Auto-selected free port: $PORT"
fi

# ---- Environment setup ----
module --force purge 2>/dev/null || true
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE 2>/dev/null || true
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

python -c "import gymnasium as gym; env = gym.make('LunarLander-v2'); print('LunarLander-v2: OK'); env.close()"

HOSTNAME=$(hostname)

echo ""
echo "============================================================"
echo "  Human Demo Collection — LunarLander-v2"
echo "============================================================"
echo ""
echo "  Episodes: ${MAX_EPISODES}   FPS: ${FPS}"
echo "  Output:   ${OUTPUT_DIR}"
echo ""
echo "  STEP 1 — Open a NEW terminal and run this SSH tunnel:"
echo ""
echo "    ssh -L ${PORT}:localhost:${PORT} ${HOSTNAME}"
echo ""
echo "  STEP 2 — Open this URL in your browser:"
echo ""
echo "    http://localhost:${PORT}"
echo ""
echo "  STEP 3 — Click on the game area, then use W/A/D/S to play."
echo "           Press SPACE to save an episode, Q to quit."
echo ""
echo "============================================================"
echo ""
echo "Starting server in 5 seconds..."
sleep 5

# ---- Collect demos ----
cd /home/marzii/IRL3/imitation || exit 1

python human_demo.py \
    --env LunarLander-v2 \
    --output "$OUTPUT_DIR" \
    --max_episodes "$MAX_EPISODES" \
    --fps "$FPS" \
    --seed "$SEED" \
    --mode web \
    --port "$PORT"

echo ""
echo "============================================================"
echo "  Done! Demos saved to: ${OUTPUT_DIR}"
echo ""
echo "  To train GAIL:"
echo "    DEMO_PATH=\"${OUTPUT_DIR}\" N_DEMOS=${MAX_EPISODES} sbatch run_gail_lunarlander.sh"
echo "============================================================"
