#!/bin/bash
#SBATCH --job-name=human-demo-lunar
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm_logs/human_demo/lunarlander/%x_%j.out
#SBATCH --error=output/slurm_logs/human_demo/lunarlander/%x_%j.err

set -euo pipefail

mkdir -p output/slurm_logs/human_demo/lunarlander

# ---- env setup (same as run_gail_lunarlander.sh) ----
module --force purge
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

which python
python -c "import sys; print(sys.executable)"

# Use local IRL3 source
export PYTHONPATH="/home/marzii/IRL3/imitation/src:${PYTHONPATH:-}"

# ---- sanity check ----
python -c "import gymnasium as gym; env = gym.make('LunarLander-v2'); print('ok', env.spec.id); env.close()"

echo "Starting human demo collection..."
echo "hostname=$(hostname)"
echo "jobid=${SLURM_JOB_ID:-local}"

DEMO_DIR="$SCRATCH/imitation_runs/human_demos/lunarlander/${SLURM_JOB_ID:-local}"
mkdir -p "$DEMO_DIR"

# ---- collect demos ----
# NOTE: This is interactive — run with: salloc then bash this script,
#       or just run directly on a login/interactive node.
cd /home/marzii/IRL3/imitation || exit 1

PORT=${PORT:-8080}

echo ""
echo "============================================================"
echo "  Open a tunnel from your local machine:"
echo "    ssh -L ${PORT}:localhost:${PORT} $(hostname)"
echo "  Then open: http://localhost:${PORT}"
echo "============================================================"
echo ""

python human_demo.py \
    --env LunarLander-v2 \
    --output "$DEMO_DIR" \
    --max_episodes 20 \
    --fps 30 \
    --mode web \
    --port "$PORT"

echo ""
echo "Demos saved to: $DEMO_DIR"
echo "To train GAIL with these demos:"
echo "  python -m imitation.scripts.train_adversarial gail \\"
echo "    with lunar_lander \\"
echo "    demonstrations.source=local \\"
echo "    demonstrations.path=\"$DEMO_DIR\" \\"
echo "    ..."
