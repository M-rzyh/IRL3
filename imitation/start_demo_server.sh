#!/bin/bash
module --force purge 2>/dev/null || true
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONUSERBASE 2>/dev/null || true
hash -r

source /scratch/marzii/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/marzii/envs/imitation-gail
hash -r

export PYTHONPATH="/home/marzii/IRL3/imitation/src"

echo "Starting on $(hostname):9999"
python /home/marzii/IRL3/imitation/human_demo.py \
    --env LunarLander-v2 \
    --output /scratch/marzii/imitation_runs/human_demos/lunarlander \
    --max_episodes 50 \
    --fps 25 \
    --mode web \
    --port 9999
