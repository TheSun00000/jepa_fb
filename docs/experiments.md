# Experiments

## Where runs happen
On Jean Zay (H100 dev for training, V100 dev for eval-only). Each run writes `logs/<timestamp>_<exp_name>-<env>/` with
`training_log.csv`, `eval_log.csv`, `info.txt` (line 1: the command-line kwargs, line 2: the config dataclass) and a copy of the script.
The job's `.slurm_logs/<script>_<jobid>.out` starts with `Running: <command>` and prints `LOGS_FOLDER`.

## Reading the logs
- `training_log.csv`: one row per chunk of 1000 updates (losses, gradient norms, feature statistics).
- `eval_log.csv`: one row per evaluation (every 10k updates): `mean_over_tasks`, `<task>_mean`, `<task>_std`, `<task>_corr`
  (corr(B^T z, R) on the inference states), and with `eval_mode=both` the same with a `policy_` prefix. In `both` mode the unprefixed
  columns are MPPI. FB scripts (`fb_jax.py`, `fb_modified.py`) only have the policy.
- `tdmpc_jax.py` (one task): `return_mean` (MPPI), `policy_return_mean`.

## Reporting
- The headline of a run is the mean over the env's tasks, averaged over the **last 10 evaluations** (single evaluations swing by about ±150); never the best eval.
- Comparisons need at least 3 runs per configuration, with different seeds (the old scripts default to seed 0).
- A run cut by the time limit (< 1M updates) is marked as such; old cut runs were renamed by hand with trailing underscores.

## Old runs (2026-09-29 -> 10-06)
In `hpc/logs/` only (never pushed to the tracker). `python tools/runs_index.py --all` lists them with their headline numbers.
