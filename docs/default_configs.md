# Default configurations per method and environment

Created 2026-11-18. One place for "which command gives the number in the main table" and "which config we use by default".
The code defaults in the scripts are **not changed** (CLAUDE.md: no silent default changes, defaults are recorded in `info.txt`).
The defaults below are explicit commands. Replace `S` by the seed (3 seeds: 1, 2, 3). Every run keeps all 10000 episodes of the RND dataset
(no `num_trajectories`), z_dim 50, final eval with 20 episodes per task.

Status tags: **decided** (user said so), **table** (what the main table currently runs), **candidate** (best so far, not decided), **todo**.

## Environments

| env | tasks | notes |
|---|---|---|
| cheetah | vel10Back, vel2Back, vel2, vel10, flip | RND data, 10k x 1000 steps |
| walker | stand, walk, vel8, run, flip | table cells use the old task set (vel8) |
| quadruped | jump, run, stand, walk | |
| antmaze | (d4rl antmaze-large-diverse) | no run in the main table yet |

## Protocol common to all training runs

- 500k updates by default (`NUM_OUTER_STEPS=500`) in `fb_jax.py`, `fbwm.py`, `tdmpc_jax.py`; 1M or more only with `num_outer_steps=` (1M = 1000, 2M = 2000).
- `eval_each_steps=25` (every 25k updates) for FB and fbwm; `num_evals_final=20`; `mppi_eval_each=100` for fbwm.
- The table score is the eval-only label sweep (z_labels 100 ... 10000) on the final checkpoint; the training-run final is the fallback.

## cheetah

### fbwm (ours) — **decided 2026-11-18**: the TD-JEPA FB changes are the default

```
python fbwm.py env_name=cheetah variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 \
  mppi_reward=lstsq z_cov=dataset mppi_root_check=1 eval_each_steps=50 mppi_eval_each=100 num_evals_final=20 num_outer_steps=1000 \
  tau=0.001 discount=0.98 ortho_init=1 q_loss=0 backward_hidden_dim=256 b_ortho_coef=0.5 target_mean=1 mix_next=1 seed=S
```

- Evidence (1M, 3 seeds, MPPI mean over tasks, mean of the last 15% of evals / final eval):
  TD-JEPA changes **449 / 454** (seeds 440, 442, 464); control (old settings: tau 0.01, discount 0.95, Q loss 0.01, uniform init) 395 / 402 (492, 300, 392);
  four fixes only (tau 0.001, Q loss 0.02, discount 0.98, ortho init) 343 / 351 (380, 359, 289).
- Caveats: 3 seeds; the control's seed spread (192) is larger than the 54-point gain. Validated at 1M only. The 500k table protocol is **todo**
  (the four-fixes arm gave 407 at 500k but 343 at 1M). The changes were tested together, not one at a time.
- What changed vs the table's old fbwm: `target_mean`/`mix_next` (new flags, default off in the code), `backward_hidden_dim` 526 -> 256, `b_ortho_coef` 1 -> 0.5,
  `q_loss` off, `tau` 0.01 -> 0.001, `discount` 0.95 -> 0.98, orthogonal init. Not applied from TD-JEPA FB: left encoder (fbwm's trunk phi plays that role),
  independent twin F (collapsed alone), single z per task (eval side).
- The planner's Q (MPPI) still takes the min of the two F's. Only the training target and the actor use the mean.
- Table (old): same command without the last line of flags, 500k (`num_outer_steps` unset, `eval_each_steps=25`).

### FB (`fb_jax.py`) — table: old settings; **candidate**: fixed FB or the TD-JEPA FB port

```
# table (old): python fb_jax.py env_name=cheetah eval_each_steps=25 num_evals_final=20 seed=S
# fixed (corrected 2026-11-18): no phi MLP (phi_layers 0), orthogonal init, Q loss 1/z_dim, discount 0.98
python fb_jax.py env_name=cheetah z_dim=50 discount=0.98 q_loss_coef=0.02 eval_each_steps=25 num_evals_final=20 seed=S
# TD-JEPA FB port (best FB so far): defaults of fb_tdjepa.py
python fb_tdjepa.py env_name=cheetah num_outer_steps=1000 num_evals_final=20 seed=S
```

- Evidence at 1M (mean of the last 15% of evals; 5000 episodes unless stated): TD-JEPA FB port defaults 525 (4 runs); fixed `fb_jax` 490 (2 seeds);
  fixed `fb_jax` with tau 0.001 479; with all 10000 episodes 406 (2 seeds); old settings (discount 0.99, phi MLP) 167.
- Not decided which FB goes in the main table. Open: the 10000-episode runs of the fixed FB are 85 points below the 5000-episode ones (2 seeds each).
- Code defaults of `fb_jax.py`: discount 0.95, q_loss_coef 0.01, tau 0.01, phi_layers 0, ortho_init 1, label_align 0. Pass `discount` and `q_loss_coef` explicitly.

### Toplines and adaptation baselines

```
python tdmpc_jax.py env_name=cheetah task=T eval_mode=both seed=S          # oracle TD-MPC v1 on the true reward
python td3_jax.py   env_name=cheetah task=T seed=S                         # offline TD3 on the true reward
python tdmpc_adapt.py mode=pretrain env_name=cheetah seed=S                # reward-free JEPA world model
python tdmpc_adapt.py mode=adapt q_data=dataset env_name=cheetah task=T wm_ckpt=$(bash tools/find_wm.sh cheetah S) \
  n_labels=N reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=S   # N in 100 300 1000 3000 10000; q_data=dataset or labeled_model
```

## walker

- **fbwm (table):** `python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 eval_each_steps=25 mppi_eval_each=100 num_evals_final=20 seed=S`.
  The TD-JEPA changes are **todo** on walker (only cheetah was tested).
- **FB (table):** `python fb_jax.py env_name=walker eval_each_steps=25 num_evals_final=20 seed=S` (old settings). Corrected FB on walker: **todo**.
- Toplines / adaptation: as for cheetah with `env_name=walker` (`tdmpc_adapt.py` pretrain checkpoint found by `tools/find_wm.sh walker S`).
- Same-family reference outside this repo: `fre_experiments/fb_agent.py` (torch FB, `ENV_NAME='walker'`), plots only (`rewards.png`).

## quadruped

- **fbwm (table):** the walker command with `env_name=quadruped`. Note `f_encoder_grad=1` is passed explicitly (script default 0.1; see decisions.md).
- **FB (table):** `python fb_jax.py env_name=quadruped eval_each_steps=25 num_evals_final=20 seed=S`.
- Toplines / adaptation: as for cheetah, `n_labels=10000` only.
- The TD-JEPA changes and the corrected FB on quadruped: **todo**.

## antmaze

No main-table run yet. `fb_jax.py` / `fbwm.py` detect it by `env_name` (d4rl dataset); no default recorded here.

## How to update this file

When a default changes: edit the command here, add the evidence line (runs, seeds, metric), and add an entry to `docs/decisions.md`. Do not edit the
script defaults silently. Walker/quadruped lines were read from the queue scripts and run folders of the table runs, not re-derived; check them against
`tools/paper_tables.py` before quoting.
