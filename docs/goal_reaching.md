# Goal reaching on OGBench: FBWM vs plain latent world models (started 2026-11-21, night)

Working note. Question (user): is FBWM better than a plain reward-free world model even on goal-reaching tasks, where
plain WMs plan with a latent distance to the goal (DINO-WM, PLDM, LeWM)? Every number comes from the run folders in
`hpc/logs`; `python tools/goal_results.py [--runs] [--env ...]` rebuilds every table below and lists the folders.

## Protocol

- **Near-goal pairs (`goal_k=K`, LeWM's protocol).** 50 pairs (s_t, s_{t+K}) of one trajectory from the held-out
  `-val` file, pair_seed 0, so every method sees the same pairs for a given env and K. The simulator is set to the
  stored (qpos, qvel) of s_t; the goal is the observation s_{t+K}. Budget 2K env steps. Success: the goal coordinates
  (antmaze ant (x, y), cube block position) come within OGBench's radius (0.5; cube 0.04 m) of the goal's at any step.
  Pairs already within the radius at the start are excluded. Code: `utils/ogbench_tasks.py` (shared by all scripts).
  `reset_err` (max |obs after reset - stored obs|) is 0 on antmaze. On cube it is about 0.3-0.8 in one feature only
  (obs[18], the gripper-contact force, which needs a physics step); positions are exact.
- **Official tasks.** OGBench's 5 tasks per maze, 10 episodes each, 1000 steps, from each task's own start.
- **Methods.**
  - *Latent-distance MPPI* (`wm_goal.py`): score = -||h_H - phi(goal)|| of the last imagined latent, MPPI 512 samples,
    6 iterations, `mppi_norm` (elite values standardised before the softmax; the default temperature made the
    weights uniform). No policy, no value. Same planner on the FBWM and the JEPA-WM latents.
  - *FB goal* (`fbwm.py eval_ckpt=... fbgoal_k=K | fbgoal_official=1`): z = sqrt(d) normalize(B(phi(goal))), FB's
    goal-reaching task vector. Actor = pi(phi(s), z). MPPI = the paper planner (per-step g(h)^T w with the lstsq w,
    terminal F^T z), variants listed in the tables.
  - *State-based LeWM / DINO-WM* (`wm_baselines.py`): their objectives and CEM planner (300 samples, top 30, 30
    iterations, horizon 5 blocks x frameskip 5, terminal squared distance) with the image encoder replaced: LeWM = MLP
    encoder trained end to end with SIGReg (their code), DINO-WM = frozen encoder (the standardised state). See the
    script's docstring for every deviation.
- **Checkpoints (antmaze-medium navigate, seed 1).** FBWM `2026-11-21_13-23-49_FBWM-gphi-antmaze_medium` (gphi,
  z_dim 200, bc_coef 0.3, 500k updates); JEPA WM `2026-11-21_13-23-49_JEPAWM-antmaze_medium` (tdmpc_adapt pretrain).

## antmaze-medium (navigate), seed 1: successes / 50

Two separate comparisons (user, 2026-11-21): the FB actor and the FB-goal MPPI use FB's learned value (F, pi), the distance
planners have none, so they are NOT compared with each other as evidence about the latent.

**(A) Same planner, different latent: the evidence for "FB shapes a better latent".** wm_goal.py's MPPI, cost = distance to the
goal latent, no value. LeWM / DINO-WM through this same MPPI: queued (planner=mppi), rows to add.

| latent (planner: distance MPPI, 5 env steps) | K=10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| start distance to the goal (xy) | 1.3 | 2.8 | 4.9 | 6.0 | 9.1 |
| JEPA WM | 6 | 2 | 3 | 2 | 1 |
| FBWM | 14 | 10 | 4 | 5 | 3 |
| JEPA WM, seed 3 | 6 | | 2 | | 1 |
| FBWM, seed 3 | 18 | | 7 | | 1 |

Seed 3 geometry (Spearman latent distance vs displacement): FBWM 0.92, JEPA 0.52 (probe R^2 0.99 / 0.93), as seed 1.
Runs: `hpc/logs/2026-11-21_23-1*_WMGOAL-*ev_am_s3*` (checkpoints `2026-11-21_22-18-34_FBWM-gphi-antmaze_medium`,
`2026-11-21_22-29-20_JEPAWM-antmaze_medium`).

**(A') Same planner, all four latents, all envs** (successes / 50; antmaze-medium FBWM and JEPA: seeds 1 + 3 pooled, / 100). LeWM /
DINO-WM: `wm_baselines.py planner=mppi plan_horizon=1` (one 5-step block, replan every env step, the same MPPI settings).

| env | latent | K=10 | K=50 | K=200 |
|---|---|---|---|---|
| antmaze-medium | FBWM | **32/100** | **11/100** | 4/100 |
| antmaze-medium | JEPA WM | 12/100 | 5/100 | 2/100 |
| antmaze-medium | LeWM (state) | 7 | 4 | 3 |
| antmaze-medium | DINO-WM (raw state) | 10 | 4 | 2 |
| antmaze-large | FBWM | **13** | **4** | 0 |
| antmaze-large | JEPA WM | 5 | 1 | 0 |
| antmaze-large | LeWM (state) | 4 | 2 | 1 |
| antmaze-large | DINO-WM (raw state) | 5 | 1 | 2 |
| cube-single | FBWM | 21 | 11 | 16 |
| cube-single | JEPA WM | 39 | 13 | 18 |
| cube-single | LeWM (state) | **43** | 11 | 13 |
| cube-single | DINO-WM (raw state) | 30 | 12 | 10 |

Mazes: the FBWM latent is the best distance cost, 2-3x every other latent (raw state included). Cube: it is the worst at K=10, and
the CEM-vs-MPPI gap of LeWM / DINO-WM (14 -> 43, 13 -> 30) confirms that JEPA's earlier cube lead was the planner.
Runs: `hpc/logs/2026-11-21_23-*_WMBASE-*-mppi-*`; all cells with `python tools/goal_results.py --runs`.

**(A'') Same planner = LeWM's CEM, all four latents** (300 samples, top 30, 30 iterations, 25-step plan, squared L2; JEPA / FBWM:
`wm_goal.py planner=cem horizon=25 receding=25`, the 1-step dynamics rolled 25 times; "every step" = receding=1). Seed 1, / 50.

| env | latent | K=10 | K=50 | K=10, replan every step | K=50, replan every step |
|---|---|---|---|---|---|
| antmaze-medium | FBWM | 10 | 8 | 10 | 6 |
| antmaze-medium | JEPA WM | 8 | 3 | 11 | 2 |
| antmaze-medium | LeWM (state) | 11 | 6 | | |
| antmaze-medium | DINO-WM (raw state) | 14 | 6 | | |
| cube-single | FBWM | 11 | 13 | 12 | 13 |
| cube-single | JEPA WM | 12 | 9 | 10 | 8 |
| cube-single | LeWM (state) | 14 | 13 | | |
| cube-single | DINO-WM (raw state) | 13 | 19 | | |

Under LeWM's CEM the four latents are within noise of each other (+-5 / 50), and replanning every step does not help CEM. So the
FBWM latent's lead in (A') shows with the short-horizon MPPI only, and the cube gap between planners (JEPA 39 with MPPI vs 12 with
CEM, LeWM 43 vs 14) is the search (5-step MPPI vs 25-step CEM), not open-loop execution.
Runs: `hpc/logs/2026-11-2*_WMGOAL-*-cem_*`.

**(B) Reference: published baselines with their own planner** (CEM, 25-step plans run open loop). Planner and latent both differ
from (A), so these rows are not a test of the latent.

| method | K=10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| DINO-WM (state), CEM | 14 | 3 | 6 | 5 | 5 |
| LeWM (state), CEM | 11 | 5 | 6 | 9 | 3 |

**(C) FBWM with FB's value (z = B(goal)).** Uses a learned value, so it is compared with no distance planner above. A fair
reference would be a goal-conditioned value method (e.g. OGBench's GCIQL / HIQL), not done.

| method | K=10 | 25 | 50 | 100 | 200 | official |
|---|---|---|---|---|---|---|
| FB-goal actor | 42 | 29 | 30 | 25 | 19 | 5 |
| FB-goal MPPI + root check | 42 | 30 | 31 | 24 | 14 | 0 |
| FB-goal MPPI, no root check | 29 | 22 | 21 | 22 | 11 | 0 |
| FB-goal MPPI init-policy + root | 44 | 31 | 30 | 25 | 12 | 0 |

Distance planning at other horizons (H = 3, 10, 20, 50) is in `tools/goal_results.py`: no horizon fixes it, and long
horizons hurt FBWM on near goals.

## antmaze-large and cube-single, seed 1 (JEPA WM / FBWM: K = 10, 50, 200 only)

| env | method | K=10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| antmaze-large | JEPA WM, distance MPPI | 5 | | 1 | | 0 |
| antmaze-large | FBWM, distance MPPI | 13 | | 4 | | 0 |
| antmaze-large | DINO-WM (state), CEM | 6 | 5 | 5 | 5 | 4 |
| antmaze-large | LeWM (state), CEM | 10 | 7 | 4 | 6 | 5 |
| cube-single | JEPA WM, distance MPPI | 39 | | 13 | | 18 |
| cube-single | FBWM, distance MPPI | 21 | | 11 | | 16 |
| cube-single | FBWM, FB-goal actor (uses FB's value) | 7 | | 3 | | 6 |
| antmaze-large | FBWM, FB-goal actor (FB's value) | 40 | | 19 | | 13 |
| antmaze-large | FBWM, FB-goal MPPI init-policy + root (FB's value) | 39 | | 25 | | 3 |
| cube-single | FBWM, FB-goal MPPI init-policy + root (FB's value) | 10 | | 5 | | 6 |
| cube-single | DINO-WM (state), CEM | 13 | 26 | 19 | 15 | 16 |
| cube-single | LeWM (state), CEM | 14 | 27 | 13 | 12 | 14 |

JEPA's 39/50 on cube K=10 against 13-14 for CEM is probably the planner (closed loop every step, 5-step plans), not the latent:
the JEPA latent's geometry is the same as on antmaze (below). The same-planner LeWM / DINO-WM rows will tell.

## Latent geometry (antmaze-medium, seed 1; `wm_goal.py diag_geom=1`)

| | FBWM phi | JEPA phi |
|---|---|---|
| linear probe R^2 of the ant (x, y) | 0.99 | 0.95 |
| Spearman(latent distance, real displacement), k <= 200 | 0.92 | 0.52 |
| same, k <= 25 | 0.70 | 0.31 |
| Spearman(latent distance, k) (real displacement: 0.42) | 0.44 | 0.20 |

Other envs (probe R^2, Spearman latent distance vs displacement, same for k <= 25): antmaze-large FBWM 0.99 / 0.91 / 0.63, JEPA
0.96 / 0.55 / 0.33; cube FBWM 0.91 / 0.56 / 0.44, JEPA 0.94 / 0.51 / 0.35. On cube the FB latent's geometry advantage is small and
the distance planner does WORSE on it than on the JEPA latent (21 vs 39 at K=10): the latent claim holds on the two mazes, not on cube.
FB's value also fails on cube (actor 7 / 3 / 6, official tasks 0/50), below every distance planner; not yet checked whether this
cube FBWM checkpoint (`2026-11-21_22-24-09_FBWM-gphi-cube_single`) is weak on cube in general.

Runs: `2026-11-21_22-44-37_WMGOAL-fbwm-geom_fbwm_s1-antmaze_medium`, `2026-11-21_22-44-43_WMGOAL-wm-geom_wm_s1-antmaze_medium`,
`2026-11-21_22-50-42_WMGOAL-wm-ev_al_s1_geom_wm-antmaze_large`, `2026-11-21_22-50-54_WMGOAL-wm-ev_cube_s1_geom_wm-cube_single`;
the other cells: `python tools/goal_results.py --runs`.

## Reading so far (one seed; seeds 2-3 and other envs running)

1. **Latent claim, same planner (A): FBWM's latent is the better distance cost.** It beats the JEPA latent at K <= 100 on
   antmaze-medium (14 vs 6 at K=10, 10 vs 2 at K=25). Mechanism: both latents contain the position, but in FBWM's latent the
   distance tracks real and temporal distance (0.92 vs 0.52). Caveat (reviewer): FB's B is trained on (s, future state) pairs,
   so goal-like geometry is expected; and the absolute numbers are low for every latent (at most 14/50).
2. **FB's value (C) is what makes FBWM solve goals** (2-6x the distance planners), but this is a comparison between a method
   with a value and methods without one; it needs a goal-conditioned value baseline before it goes in the paper.
3. **Planning through the model does not beat the FB actor here.** MPPI with the FB cost is at best equal to the actor on
   near goals, worse at K=200, and 0/50 on the official tasks (actor 5/50).

## Not done / open

- Seeds 2-3 (FBWM and JEPA WM, antmaze-medium), antmaze-large and cube-single FBWM: training (2026-11-21 23:00).
- The frozen-WM control is already in the paper (Table 1, DMC); not repeated here (user).
