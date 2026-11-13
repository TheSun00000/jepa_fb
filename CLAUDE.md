# fb_jepa: zero-shot RL with Forward-Backward representations and a latent world model

Research code. FB (forward-backward) learns, from reward-free offline data, a basis of rewards B and their successor features F.
A TD-MPC-style latent world model (JEPA consistency) is learned next to it, and at test time any task z (inferred from labeled
states) is solved by MPPI planning in latent space, with no training. Plan and claims: `docs/research_plan.md`.

## Shared context (read first)
- `docs/research_plan.md`: claims C1-C5, protocol, experiments, order of work. Living document: update it and its changelog.
- `docs/method.md`: the method variants (sepB, gphi, ...), losses and gradient routing.
- `docs/decisions.md`: decisions and why. Add an entry whenever one is made.
- `docs/journal.md`: dated session notes. Add one at the end of each work session.
- `docs/experiments.md`: how runs are named, logged and reported.
- Tracker ("fb_jepa Console"): https://claude.ai/artifact/3UineFBuRZiwE4gzMLRkhn . Source: `tracker/index.html`. Its data lives in the artifact database; update it with the ArtifactData tool, not by republishing the page.
- `tmp.md` is an older write-up (2026-09-30) of the B = phi design; outdated.

## Layout
- Scripts, one file each, flags `key=value` (no dashes), parsed by `utils/logs.py:parse_kwargs`:
  - `fb_jax.py`: plain FB (reference baseline). `fb_modified.py`: FB with a learned trunk phi in front of F / B / pi (flags `f_phi`, `b_phi`, `pi_phi`, `phi_identity`).
  - `tdmpc_jax.py`: TD-MPC v1 trained on the true reward of one task (`task=`), the oracle / upper bound.
  - `tdmpc_sf.py`: successor features on a JEPA latent (failed variant, kept for the record).
  - `tdmpc_fb.py`: B = phi (encoder output used as B). `tdmpc_fb_2.py`: **sepB** (B on raw observations, phi for dynamics/F/pi, B-head on real + imagined latents). `tdmpc_fb_3.py`: **gphi** (B = g(phi(s)), flags `consistency_coef`, `bcons_coef`, `f_encoder_grad`, `b_encoder_grad`, `ortho_coef`, `b_ortho_coef`, `phi_identity`).
  - `eval_mode=both|mppi|policy` on the TD-MPC-FB scripts; `both` logs MPPI in the unprefixed columns and the policy in `policy_*`.
- `utils/`: `logs.py`, `wrappers.py`. `tools/`: HPC and tracker tooling (below). `scripts/`: sbatch files (`scripts/queue/` for the queue).
- `logs/<timestamp>_<exp_name>-<env>/`: `training_log.csv`, `eval_log.csv`, `info.txt` (line 1: command-line kwargs, line 2: config), a copy of the script. `models/<run>/`: checkpoints.
- `logs/`, `models/`, `hpc/`, `queue/`, `csv_files/`, `.slurm_logs/`, `backup_local_2026-11-13/` are gitignored.

## Environments
- Experiments run on Jean Zay. The local machine is for editing, analysis and light checks; nothing heavy runs locally.
- The scripts detect local vs cluster by whether the working directory contains `nazim` (dataset path `../offline_qd/datasets/d4rl/` locally, `../fre_experiments/datasets/` on the cluster).
- Stack: JAX, Flax, Optax. Envs: dm_control cheetah, walker, quadruped (RND datasets, 10k x 1000 steps), d4rl antmaze-large-diverse.
- Local disk is nearly full (2026-11-13): never sync images, keep large outputs on the cluster.

## HPC (Jean Zay)
- Access: `ssh jz`. Over a plain `ssh jz 'cmd'` `$WORK` is empty: use `ssh jz 'bash -l -s'` with commands on stdin, or `ssh jz 'bash -lc "..."'`. Repo clone: `/lustre/fswork/projects/rech/vcj/uyo96iu/jepa_fb`. CLFB lives next to it (`../CLFB`): never touch it from this project.
- Partitions (user, 2026-11-13): training on **H100 dev** (`qos_gpu_h100-dev`, 2h max; a full 1M-update run with MPPI evals takes 85-110 min); eval-only jobs on **V100 dev** (`qos_gpu-dev`). A100 is not used for training (too slow for 2h).
- The dev limit is 10 submitted jobs per user and per QoS, CLFB's jobs included.
- Read-only by default. Never cancel, delete or overwrite anything on the cluster without asking.
- One-way sync of logs and code into `hpc/`: `tools/sync_from_hpc.sh` (CSVs, info.txt, slurm logs, scripts, code; never images). Nothing is pushed by it.
- Git: the laptop pushes to GitHub with the deploy key `~/.ssh/fb_jepa_github_ed25519` (set as `core.sshCommand` in this repo); the cluster pulls with `git pull --ff-only`. Commit and push before submitting.

## Queue of experiments (what to sbatch, and when)
- `tools/queue.py add NAME --cmd "python ..." [--cmd ...] [--title ...] [--gpu h100|v100] [--time 2:00:00]` writes `scripts/queue/NAME.sh` and appends to `queue/experiments.json` (local, git-ignored). `status`, `submit`, `sync`, `requeue`, `prio`: see its docstring.
- The user has authorized automatic submission (2026-11-13), as in CLFB: whenever Claude touches the cluster (a tracker refresh, a status check, a new message about experiments), run `python tools/queue.py submit`. It refuses to run with uncommitted or unpushed scripts/code: commit and push first. Never cancel or delete anything on the cluster.
- Systemd user timers in `tools/systemd/` (`fb-jepa-queue.timer` every 3 min, `fb-jepa-hpc-status.timer` every 2 min) do the same without Claude once the user installs them.
- When adding an experiment: say so in the tracker (Log entry, and a `studies` doc if it answers a question of the plan).

## Tracker
- The tracker is run-centric: one record per run folder (`tools/runs_index.py`), jobs only link to runs.
- **It starts clean on 2026-11-13** (user): runs and jobs before that date stay local in `hpc/` and are never pushed (`runs_index.TRACK_FROM`). Analyse old runs locally with `python tools/runs_index.py --all`.
- Refresh ("update the tracker"), in this order:
  1. `tools/sync_from_hpc.sh`;
  2. `tools/hpc_status.py --diff` (writes `hpc/status/runs_<n>.json`, `curves_<n>.json`, `jobs_<n>.json`, `queue.json`); if it prints "changed since last push":
  3. `ArtifactData list` of the collection `snapshot` (with `out_dir` outside `.claude/`, e.g. `hpc/db_read`) for the versions, then a batch `set` of `snapshot/runs_<n>`, `snapshot/curves_<n>`, `snapshot/jobs_<n>`, `snapshot/queue` from those files with `if_version` (documents are limited to 262144 bytes; shards are 200 kB);
  4. `tools/hpc_status.py --mark-pushed`.
- Collections: `snapshot` (machine-written), `tracking/<run folder>` (`title`, `note`, `hidden`, and `family` / `variant` overrides), `claims/C1..C5` (`title`, `status` open|running|weak|supported|refuted, `evidence`, `order`), `studies/<id>` (`title`, `question`, `envs`, `rows: [{label, family, variant}]`, `summary`, `order`), `updates/<id>` (`date`, `title`, `body`), `todos/<id>` (written by the page too).
- No result goes into the tracker or `docs/` without the run folders it came from.

## Conventions
- Match the existing style: single-file scripts, flat `key=value` flags. Do not change defaults silently (they are recorded in `info.txt`).
- Log new metrics as extra CSV columns; do not rename existing columns.
- **Method changes need the user's explicit yes** (a loss, a gradient path, a latent distribution): propose them in words first. Hyper-parameters of an agreed method and tooling are fine.
- Do not commit logs, models or datasets.

## Updating the shared context
At the end of a session: append to `docs/journal.md`, record decisions in `docs/decisions.md`, update `docs/research_plan.md` (status tags + changelog), and add a Log entry to the tracker.
