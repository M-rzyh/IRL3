# IRL3 What each `.sh` is for

Pipeline is: **generate demo pools** → **submit GAIL training** → (util) rebuild/re-eval.
Almost everything uses expert **4615187**; only `submit_gail_grid_count.sh` uses **4720242**.
There are TWO noise pipelines that share `noise_p{n}` naming but are different data:
**online** (`noisy_demos_online_nonFS/`) and **offline** (`noisy_demos/`, via `add_action_noise`).

## Generate demo pools (run once; produce the datasets everything else subsamples)
| Script | Use it to… |
|---|---|
| `generate_noise_demo_pools.sh` | **The online noise-pool generator** — options `FS` (default 0 = nonFS 4615187), `MODE` (default `coin_uniform` = your current N15/N50/N100 pools; also `exact_uniform`/`coin_exclude`/`exact_exclude`), `LEVELS`/`NSEEDS`/`FORCE`. Default writes `noisy_demos_online_nonFS/…/expert_4615187/n100_p{P}_s{S}` (what the noise sweeps subsample); variants → `noise_variants/demos/{MODE}/`. Skips existing pools unless `FORCE=1`. Replaces the old `_online_noise_demos_nonfs`/`_noise_variants_demos`. |
| `generate_noisy_demo_pool.sh` | **OFFLINE** noise pools (`noisy_demos/…/expert_4615187`) via `add_action_noise` — a *pre-recorded* expert trajectory with its action labels flipped (states stay clean). Only `submit_gail_grid.sh` uses these; not covered by the online generator. ⚠️ Header wrongly says "FS 4720242 / 41 datasets" — code is 4615187 / 361. |
| `generate_blind_expert_demos.sh` | Build **blind-expert** frame-blanking demos (expert 4615187 fed zeroed obs on blanked frames; block b=10, blank% {0,25,50,75}). |

## Submit GAIL training
| Script | Use it to… |
|---|---|
| `submit_gail_grid_noise_N.sh` | **Template for a noise sweep at a chosen N** (edit `N_DEMOS` + `noise_axis_pct`): online nonFS 4615187, 30 seeds, append-safe. ⚠️ Currently set to **N=15, `noise_axis_pct=(0)`** — that `n100_p0_s{seed}` path **doesn't exist** and there's no dir guard, so **don't use it for 0%**; reset to 10–100 for a real sweep. |
| `submit_gail_clean_N.sh` | **Clean / 0% point for any N** — `bash submit_gail_clean_N.sh <N>`. Real clean pool `n100_p0_clean`, expert 4615187, per-seed subset via `SHUFFLE_SEED`, appends, fails fast if missing. Pair with the noise sweep. |
| `submit_gail_grid_noise.sh` | Noise axis, fixed **N=100** (shuffle=0), online nonFS 4615187, 12 levels × 30 = 360. ⚠️ index **truncates** (`>`). |
| `submit_gail_grid_noise_N50.sh` | Noise axis at **N=50** (bs=512), online nonFS 4615187, levels 10–100 × 30. ⚠️ header wrongly says N=5; index **truncates**. |
| `submit_gail_grid_count.sh` | **Count axis** (clean, N ∈ {1,5,10,50,100} × 30) → `gail_grid_count_FS_*.csv`. ⚠️ uses **FS 4720242**, the only script that does — don't pair its 0% rows with 4615187 noise. Index **truncates**. |
| `submit_gail_grid.sh` | Old combined grid: count + noise (N=100, 12 levels) on **offline** 4615187 pools, × 30 = **510 jobs**. ⚠️ header badly stale ("90 jobs / 10 seeds / 4720242 / 4 levels" — all wrong); index **truncates**. |
| `submit_gail_noise_variants.sh` | Train GAIL (N=100 default, overridable) on the 3 noise-variant modes × 10 levels × 30. Appends (+ extra `mode` column). |
| `submit_gail_blind_expert.sh` | Frame-blanking: train on **blind-expert** demos, **N=50**, blank% {0,25,50,75} × seeds. ⚠️ index **truncates**. |
| `submit_gail_blind_human_b5.sh` | Frame-blanking b=5: train on **blind-human** demos, **N=50** — `bash … <blank_pct> <seed_lo> <seed_hi>`. Appends. |

## Utilities
| Script | Use it to… |
|---|---|
| `rebuild_gail_noise_N15_index.sh` | Rebuild a truncated N15 noise index from job logs (`DEMO_NOTE`) + the clean-rows side-file. Idempotent; re-run after the sweep finishes. |
| `re_eval_batch.sh` | Re-evaluate finished run dirs (`re_eval_run.py`) at a chosen episode count. |

## Gotchas (verified against the code)
- **Clean is one pool** (`n100_p0_clean`), *not* per-seed `n100_p0_s{seed}` — never point a 0% run at the per-seed pattern. Use `submit_gail_clean_N.sh`.
- **Match the expert**: 4615187 for every noise/clean plot; only `submit_gail_grid_count.sh` uses 4720242. Don't mix on one plot.
- **online vs offline noise pools share `noise_p{n}` names but differ**: the `_N`/N50/noise sweeps pull `noisy_demos_online_nonFS/`; `submit_gail_grid.sh` pulls `noisy_demos/`.
- **Only these append safely**: `submit_gail_clean_N.sh`, `submit_gail_grid_noise_N.sh`, `submit_gail_blind_human_b5.sh`, `submit_gail_noise_variants.sh`. All other submitters `> $INDEX` (truncate) — a same-day re-run wipes prior rows.
- **Stale headers, trust the code**: `generate_noisy_demo_pool.sh`, `submit_gail_grid_noise_N50.sh` (says N=5), `submit_gail_grid_noise_N.sh` (says N=5, is N=15), `submit_gail_grid.sh`.
