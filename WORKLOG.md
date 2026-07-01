# IRL3 (GAIL) Worklog

Running log of changes. Newest at top. Keep entries to 1–2 sentences.

| Date | What | Why / Result |
|---|---|---|
| 2026-06-30 | Lowered GAIL demo batch size 512 → 128 in `submit_gail_grid_noise_N5.sh` and re-ran all 300 (N=5, levels 10–100 × 30 seeds). | At high noise, demos are short (random lander crashes fast → ~460 transitions for 5 demos at 100%), below batch 512 → data-loader error. 128 fits every level; re-ran all (not just the 75 failed) so all levels use the same batch size. |
| 2026-06-30 | Added `submit_gail_grid_noise_N5.sh` (N=5 demos, noise 10–100%, 30 seeds; subsample 5 from the n100 noisy pools). | Low-budget / matched-human-time GAIL arm of the GAIL-vs-PT noise plot. 0% reuses the existing `count_N5` clean condition. |
| 2026-06-30 | Moved `export PYTHONPATH` above the `import imitation` sanity check in `run_gail_lunarlander.sh`. | The check ran before the path was set, so under `set -e` every job aborted in ~1s with `ModuleNotFoundError`. |
