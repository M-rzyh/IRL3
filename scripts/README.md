# IRL3 `scripts/` — what each `.sh` is for

Pipeline: 
**generate demo pools** → **submit GAIL training** → (util) rebuild/re-eval.

!!Almost everything uses expert **4615187**; only `submit_gail_grid_count.sh` uses **4720242**.

**Two noise pipelines** share `noise_p{n}` naming but are different data:
- **online** (`noisy_demos_online_nonFS/`) — noise injected *during* the rollout, so the lander genuinely misflies (short, crashing episodes).
- **offline** (`noisy_demos/`, via `add_action_noise`) — a clean recorded trajectory with its action *labels* flipped; the states stay the perfect flight.

## Generate demo pools — in `pool_generators/` (run once; produce the datasets everything else subsamples)
Run from the repo root, e.g. --> `sbatch scripts/pool_generators/generate_clean_demo_pools.sh`.
| Script | Use it to… |
| **`generate_clean_demo_pools.sh`** | **Clean (0%) pool** — expert rolled out with `--noise-prob 0` into one shared pool `n100_p0_clean` (not per-seed; the submitters subsample it via `SHUFFLE_SEED`). `FS` (default nonFS 4615187), `NEPISODES` (default 150), `FORCE`. Writes `noisy_demos/…/expert_4615187/n100_p0_clean` — where `submit_gail_clean_N.sh` / `submit_gail_grid_count.sh` read it. Skips if present unless `FORCE=1`. |
| **`generate_online_noisy_demo_pools.sh`** | **Online noise pools** — `FS` (default nonFS 4615187), `MODE` (default `coin_uniform` = your N15/N50/N100 pools; also `exact_uniform`/`coin_exclude`/`exact_exclude`), `LEVELS`/`NSEEDS`/`FORCE`. Default → `noisy_demos_online_nonFS/…/expert_4615187/n100_p{P}_s{S}` (what the noise sweeps subsample); variants → `noise_variants/demos/{MODE}/`. FS supports `coin_uniform` only. Skips existing unless `FORCE=1`. |
| **`generate_offline_noisy_demo_pools.sh`** | **Offline noise pools** — `add_action_noise` flips action labels on a clean recorded trajectory (states unchanged). Only `submit_gail_grid.sh` reads these; not interchangeable with the online pools despite the shared `noise_p{n}` names. ⚠️ header wrongly says "FS 4720242 / 41 datasets" — code is 4615187 / 361. | --> basically useless.
| **`generate_blind_expert_demo_pools.sh`** | **Blind-expert** frame-blanking demos — expert fed zeroed obs on blanked frames (block b=10, blank% {0,25,50,75}), expert 4615187. | --> if some day we decide to do agent POMDP/Control variations.

Clean is pipeline-agnostic (noise-prob 0 corrupts nothing), so the online-rolled clean pool pairs with either the online or offline noise sweeps.

## Submit GAIL training
| Script | Use it to… |
| `submit_gail_grid_noise_N.sh` | **Template for a noise sweep at a chosen N** (edit `N_DEMOS` + `noise_axis_pct`): online nonFS 4615187, 30 seeds, append-safe. ⚠️ currently left at **N=15, `noise_axis_pct=(0)`** — that `n100_p0_s{seed}` path **doesn't exist** and there's no dir guard, so **don't use it for 0%**; reset to 10–100 for a real sweep, and use `submit_gail_clean_N.sh` for 0%. |
| `submit_gail_clean_N.sh` | **Clean / 0% point for any N** — `bash submit_gail_clean_N.sh <N>`. Reads `n100_p0_clean` (4615187), per-seed subset via `SHUFFLE_SEED`, appends, fails fast if the pool is missing. Pair with the noise sweep. |
| `submit_gail_grid_noise.sh` | Noise axis, fixed **N=100** (shuffle=0), online nonFS 4615187, 12 levels × 30 = 360. ⚠️ index **truncates** (`>`). |
| `submit_gail_grid_noise_N50.sh` | Noise axis at **N=50** (bs=512), online nonFS 4615187, levels 10–100 × 30. ⚠️ header wrongly says N=5; index **truncates**. |
| `submit_gail_grid_count.sh` | **Count axis** (clean, N ∈ {1,5,10,50,100} × 30) → `gail_grid_count_FS_*.csv`. ⚠️ uses **FS 4720242**, the only script that does — don't pair its 0% rows with 4615187 noise. Index **truncates**. |
| `submit_gail_grid.sh` | Old combined grid: count + noise (N=100, 12 levels) on **offline** 4615187 pools, × 30 = **510 jobs**. ⚠️ header badly stale ("90 jobs / 10 seeds / 4720242 / 4 levels" — all wrong); index **truncates**. |
| `submit_gail_noise_variants.sh` | Train GAIL (N=100 default, overridable) on the 3 noise-variant modes × 10 levels × 30. Appends (+ extra `mode` column). Missing pool → hint points at `generate_online_noisy_demo_pools.sh`. |
| `submit_gail_blind_expert.sh` | Frame-blanking: train on **blind-expert** demos, **N=50**, blank% {0,25,50,75} × seeds. ⚠️ index **truncates**. |
| `submit_gail_blind_human_b5.sh` | Frame-blanking b=5: train on **blind-human** demos, **N=50** — `bash … <blank_pct> <seed_lo> <seed_hi>`. Appends. |

## Utilities
| Script | Use it to… |
|---|---|
| `rebuild_gail_noise_N15_index.sh` | Rebuild a truncated N15 noise index from job logs (`DEMO_NOTE`) + the clean-rows side-file. Idempotent; re-run after the sweep finishes. |
| `re_eval_batch.sh` | Re-evaluate finished run dirs (`re_eval_run.py`) at a chosen episode count. |

## Gotchas (verified against the code)
- **Clean is one pool** (`n100_p0_clean`), *not* per-seed `n100_p0_s{seed}` — never point a 0% run at the per-seed pattern. Make it with `generate_clean_demo_pools.sh`, submit it with `submit_gail_clean_N.sh`.
- **Match the expert**: 4615187 for every noise/clean plot; only `submit_gail_grid_count.sh` uses 4720242. Don't mix on one plot.
- **online vs offline noise differ** despite shared `noise_p{n}` names: the noise sweeps pull `noisy_demos_online_nonFS/`; `submit_gail_grid.sh` pulls `noisy_demos/`.
- **Append-safe submitters**: `submit_gail_clean_N.sh`, `submit_gail_grid_noise_N.sh`, `submit_gail_blind_human_b5.sh`, `submit_gail_noise_variants.sh`. All other submitters `> $INDEX` (truncate) — a same-day re-run wipes prior rows.
- **Stale headers, trust the code**: `generate_offline_noisy_demo_pools.sh`, `submit_gail_grid_noise_N50.sh` (says N=5), `submit_gail_grid_noise_N.sh` (says N=5, is N=15), `submit_gail_grid.sh`.
