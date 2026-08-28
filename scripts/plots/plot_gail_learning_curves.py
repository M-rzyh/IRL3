#!/usr/bin/env python3
"""Plot GAIL learning curves (steps vs reward) for all runs matching a demo filter.

Usage:
    python plot_gail_learning_curves.py <filter> [--output PATH] [--job-ids ID1,ID2,...]

<filter> is any substring to match against each run's demonstrations.path. Examples:
    4615187      (expert job ID)
    session_2    (human session 2)
    session_1
    human_demos  (any human demos)

Optionally limit to specific GAIL job IDs via --job-ids (comma-separated).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_dir import fig_path
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


GAIL_ROOT = "/scratch/marzii/imitation_runs/gail/lunarlander"
SLURM_LOG_ROOT = "/scratch/marzii/imitation_runs/_slurm_logs/gail/lunarlander"
# Map expert slurm job ID -> expert timestamp dir (for legacy path matching)
EXPERT_ID_TO_TS = {
    "4615153": "20260404_161428_cd10a3",
    "4615170": "20260404_165234_e7f0b8",
    "4615187": "20260404_172057_20173e",
}


def _parse_from_slurm_log(job_id):
    """Fallback: when sacred/config.json missing, grep config lines from slurm .out."""
    f = os.path.join(SLURM_LOG_ROOT, f"gail-lunarlander_{job_id}.out")
    if not os.path.exists(f):
        return None
    with open(f) as fp:
        text = fp.read()

    def m(pat, default=None, conv=str):
        r = re.search(pat, text, re.MULTILINE)
        return conv(r.group(1).strip()) if r else default

    demo_path = m(r"^Demo path:\s+(.+)$", "")
    n_demos = m(r"^N demos:\s+(\d+)", 0, int)
    bs = m(r"^Demo batch size:\s+(\d+)", 0, int)
    seed = m(r"^Seed:\s+(-?\d+)", 0, int)
    gym_id = "LunarLander-v2"
    fs_match = re.search(r"Frame skip:\s+\d+\s+\(gym_id=([^,]+),", text)
    if fs_match:
        gym_id = fs_match.group(1).strip()
    if not demo_path:
        return None
    return {
        "demonstrations": {"path": demo_path, "n_expert_demos": n_demos},
        "algorithm_kwargs": {"demo_batch_size": bs},
        "seed": seed,
        "environment": {"gym_id": gym_id},
    }


def _demo_source_short(demo_path):
    if "session_1" in demo_path:
        return "sess1"
    if "session_2" in demo_path:
        return "sess2"
    # Expert: extract job ID if present
    for part in demo_path.split("/"):
        if part.isdigit():
            return f"exp{part}"
    return "demo"


def find_runs(filter_str, job_ids=None):
    ts = EXPERT_ID_TO_TS.get(filter_str, "") if filter_str else ""
    results = []
    for run_dir in sorted(glob.glob(os.path.join(GAIL_ROOT, "[0-9]*"))):
        job_id = os.path.basename(run_dir)
        if job_ids and job_id not in job_ids:
            continue
        cfg_path = os.path.join(run_dir, "sacred/config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = _parse_from_slurm_log(job_id)
            if cfg is None:
                continue
        demo_path = cfg.get("demonstrations", {}).get("path", "")
        if filter_str:
            if filter_str not in demo_path and (not ts or ts not in demo_path):
                continue
        gym_id = cfg.get("environment", {}).get("gym_id", "LunarLander-v2")
        results.append({
            "job_id": job_id,
            "n_demos": cfg["demonstrations"]["n_expert_demos"],
            "batch_size": cfg["algorithm_kwargs"]["demo_batch_size"],
            "seed": cfg["seed"],
            "gym_id": gym_id,
            "demo_src": _demo_source_short(demo_path),
            "fs": "FS10" if "FS10" in gym_id else "noFS",
            "tb_dir": os.path.join(run_dir, "log/raw/gen"),
        })
    # If job_ids was given, preserve order from user
    if job_ids:
        order = {j: i for i, j in enumerate(job_ids)}
        results.sort(key=lambda r: order.get(r["job_id"], 999))
    return results


def load_curve(tb_dir):
    if not os.path.isdir(tb_dir) or not os.listdir(tb_dir):
        return [], []
    try:
        ea = EventAccumulator(tb_dir)
        ea.Reload()
    except Exception:
        return [], []
    tag = "raw/gen/rollout/ep_rew_mean"
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", nargs="?", default="",
                        help="Optional substring to match against each run's demo path "
                        "(expert JOBID, 'session_2', 'human_demos', etc.). "
                        "If omitted, --job-ids alone determines the runs.")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--job-ids", default=None,
                        help="Comma-separated list of GAIL job IDs to include (optional)")
    parser.add_argument("--title", default=None, help="Custom plot title")
    parser.add_argument("--show-source", action="store_true",
                        help="Include demo-source + FS tag in legend labels "
                             "(useful when mixing different sources)")
    parser.add_argument("--ylim", default=None,
                        help="Y-axis limits as 'min,max' (e.g. '-400,400')")
    args = parser.parse_args()

    job_ids = args.job_ids.split(",") if args.job_ids else None
    if not args.filter and not job_ids:
        parser.error("Provide either a filter substring or --job-ids.")

    runs = find_runs(args.filter, job_ids=job_ids)
    if not runs:
        print(f"No GAIL runs found (filter='{args.filter}', job_ids={job_ids})")
        return

    print(f"Found {len(runs)} GAIL runs:")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for r in runs:
        steps, values = load_curve(r["tb_dir"])
        if not steps:
            print(f"  job {r['job_id']}: NO TB DATA — skipping")
            continue
        last10_avg = float(np.mean(values[int(len(values) * 0.9):]))
        parts = [f"n={r['n_demos']}", f"bs={r['batch_size']}", f"seed={r['seed']}"]
        if args.show_source:
            parts.insert(0, r["demo_src"])
            parts.insert(1, r["fs"])
        label = (", ".join(parts) + f" (job {r['job_id']}) — last10%={last10_avg:.1f}")
        print(f"  {label}: {len(steps)} points")
        ax.plot(steps, values, label=label, alpha=0.85, linewidth=1.4)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Episode Reward (ep_rew_mean)")
    title = args.title or (f"GAIL Learning Curves — demos match '{args.filter}'"
                           if args.filter else "GAIL Learning Curves")
    ax.set_title(title)
    if args.ylim:
        ymin, ymax = (float(v) for v in args.ylim.split(","))
        ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_filter = (args.filter.replace("/", "_") if args.filter else "runs")
    out = fig_path(args.output or f"gail_curves_{safe_filter}.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
