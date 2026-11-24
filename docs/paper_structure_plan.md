# Paper structure and open work (agreed with the user, 2026-11-24)

## Structure
Two benchmarks. All claims are about planning (MPPI) in a JEPA world model, so only MPPI numbers are reported (no policy numbers).
Encoder: PhiMap 64 for FBWM and every JEPA (DMC and OGBench). Exceptions, stated in captions: TD-MPC upper bound (its own MLP-256 encoder), LeWM (state-based port: MLP encoder + projector, 6-block AdaLN transformer predictor, CEM; see `wm_baselines.py` docstring).
FBWM = JEPA consistency + FB trained jointly on one latent, no VICReg/SIGReg (FB is the regulariser).

### 1. DMC (walker, cheetah, quadruped; non-goal rewards)
- Table, rows: TD-MPC (upper bound) | JEPA VICReg + adaptation | JEPA SIGReg + adaptation | FBWM (ours) | JEPA VICReg + FB on top (frozen) | JEPA SIGReg + FB on top (frozen). MPPI only.
- Figure: same rows (minus none) vs N = 100..10000 labels (sample efficiency); TD-MPC as a dashed line.
- "Adaptation" = reward head fit on the N labels (+ Q / policy prior). User wants the "full dataset" variant NOT reported; Claude flagged that this is the strongest, like-for-like baseline (FB also uses the whole reward-free dataset): it ties FBWM at N=10k (walker 622/622, cheetah 441/441, quadruped 638 vs 587). User's decision (2026-11-24) stands but the claim must then be worded as what the numbers support, and the full-dataset variant goes in the appendix / is ready when a reviewer asks. Labels-only (Dyna) is much weaker and has not been rerun on the matched JEPA.
- Done: TD-MPC, FBWM, JEPA VICReg adaptation (walker, cheetah all N; quadruped N=10k), JEPA VICReg + frozen FB (`fbwm_frozen`).
- MISSING (step 1, to queue): JEPA SIGReg pretrain on DMC (`tdmpc_adapt.py encoder=phimap enc_dim=64 lr=1e-4 batch_size=1024 update_freq=1 consistency_coef=5 reg=sigreg target=online`, 3 envs x 3 seeds), adaptation on it, frozen FB on it (`fbwm.py freeze_wm=true wm_ckpt=...`, label sweep as `fbj_e1_frozen_*`).
- Queued/running: quadruped matched-JEPA adaptation N=100..3000 (`fbj_a64_adaptN_qu_*`).

### 2. OGBench (antmaze-medium, antmaze-large, cube-single; goal reaching)
- One table per env (rows = methods, columns = K = 10 / 50 / 200): JEPA VICReg, JEPA SIGReg, FBWM (ours), LeWM (CEM), plus FBWM + V and FB actor (user re-added them: V = free value as the cost; shows the value is free).
- FBWM discount: 0.95 on the mazes, 0.99 on cube (appendix states it). Cube is a negative result (JEPA VICReg beats FBWM at every K: 93/28/49 vs 71/19/31); keep it in the table.
- Open item: FBWM + V and FB actor are at 0.99 (3 seeds, all envs). With 0.95 on the mazes the V rows exist at 0.95 only for medium (3 seeds, 86/61/27); large 0.95 V is 1 seed. Either run V / actor at 0.95 on large seeds 2-3 (and actor goal_k at 0.95 on medium) or label the V rows as 0.99.

## Order of work
1. Queue the SIGReg DMC block (pretrain, adaptation, frozen FB).
2. Wait for the quadruped N sweep and the PhiMap 512x4 capacity check (`fbj_cap_*`, antmaze-medium, FBWM 0.99 and JEPA VICReg, 3 seeds; answers "64 is too shallow", TD-JEPA OGBench encoder is 512x4).
3. Rebuild the DMC table, the N figure and the three OGBench tables in the tracker (tools/paper_tables.py does NOT know the matched adaptation runs; it would overwrite them with the old MLP-256 numbers: update it first).

## Where the numbers are
- Journal 2026-11-24, decisions 2026-11-24, research_plan changelog 2026-11-24, tracker (findings goal_main_{am,al,cu}, latent_geometry, paper/main_table, updates/2026-11-24-phimap64-everywhere; old tables in the archive tab).
- Numbers to rebuild from: run folders under hpc/logs (`*_WMGOAL-wm-fair64_*`, `*_WMGOAL-fbwm-d99_*`, `*_WMGOAL-fbwm-gap95_*`, `*_WMBASE-lewm-*`, `*_TDMPC-adapt-*` with a JEPAWM-phimap wm_ckpt, `*_FBWM-gphi-*`, `*_FB-*`).
