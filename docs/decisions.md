# Decisions

Newest first. One entry per decision: what, why, what else was considered, evidence.

## 2026-11-13: coexistence with CLFB on the same Jean Zay account
- Both projects auto-submit on the same user account. The dev limit (10 per QoS) cannot be exceeded: both queues count every job of the user in squeue, and an sbatch refused by Slurm leaves the entry queued for the next pass.
- V100 dev is shared (CLFB trains there since 2026-11-04; fb_jepa uses it for eval-only): first come, first served.
- fb_jepa queue jobs are named `fbj_*`, because CLFB's hpc_status.py recognises jobs by script name only. fb_jepa's own collector filters by WorkDir.
- Known CLFB-side issue, not fixed (CLFB is off limits from this project): CLFB's `queue.py cluster_counts()` maps every QoS other than qos_gpu-dev to 'a100', so fb_jepa's H100 jobs reduce CLFB's computed A100 free slots (conservative: CLFB under-submits, never over-submits). One-line fix in CLFB if needed.

## 2026-11-13: HPC and tracker setup, same workflow as CLFB
- Decision (user): same Claude setup as CLFB: queue of experiments with automatic submission, status collector, systemd timers, tracker artifact with a database, `docs/`.
- Partitions: training on H100 dev (a full run takes 85-110 min, the dev limit is 2h; A100 would be 1.5-2x slower and not fit); eval-only jobs on V100 dev. Considered: A100 dev (does not fit), A100 t3 (20h, slow to start), checkpoint/resume chunks (needs code).
- Git: GitHub with a deploy key, as CLFB (user chose it over pushing straight to the JZ clone). The JZ clone's uncommitted work is backed up on JZ before the clone is aligned with the pushed commit.
- Tracker style: dark terminal, deliberately different from CLFB.
- The tracker starts clean: runs before 2026-11-13 are not pushed (user). They stay in `hpc/` for local analysis.
- Sync: no images (user; the local disk was full).

## 2026-11-13: local code replaced by the JZ version
The JZ clone was the reference. Local copies that differed were backed up in `backup_local_2026-11-13/` before the copy. Local-only edits that were replaced: `tdmpc_jax.py` `reward_coef = 0.0` (JZ: 0.5), `fb_modified.py` `PhiMap(..., num_layers=False)`.

## Template
### YYYY-MM-DD: short title
- Decision:
- Reason:
- Alternatives considered:
- Evidence (run folder, notebook, discussion):

## 2026-11-13: core runs use 3 seeds and evaluate every 20k updates
- **3 seeds (1, 2, 3) for now, not 5** (user): this is a first test of the gate. Seeds 4 and 5 can come later. Seeds are passed explicitly so the configs line up seed by seed.
- **`eval_each_steps=20` (every 20k updates instead of 10k):** with evals every 10k, gphi takes about 111–119 min on H100, too close to the 2h dev limit, and the user wants walltimes under 2h for faster allocation. Measured from old runs, halving the evals gives about 80 min for sepB and 95–100 min for gphi. The reported number is the final 10-episode eval, which this doesn't change; only the curves get coarser. Walltimes asked: FB 0:50, sepB 1:40, gphi 1:50.
- **`fb_jax.py` gets the same final eval as `fbwm.py`** (`tag=final`, 10 episodes), so FB is measured the same way as the other configs.

## 2026-11-14: the gphi core config is consistency 5, bcons 2 **and f_encoder_grad 1**
- The 2026-10-05 gphi run taken as "gphi best" (`2026-10-05_15-17-36_TDMPC-FB-gphi-quadruped`, 666/660) passed `f_encoder_grad=1.0`. The core runs used the script default of 0.1 (Claude copied only "consistency 5, bcons 2" from the plan). With 0.1, gphi quadruped seed 1 (`2026-11-13_17-29-43_FBWM-gphi-quadruped`) peaked at 668 around 520k, then fell to ~300 from 620k on.
- User, 2026-11-14: cancel the gphi seed-2 jobs (0.1) and requeue. gphi with `f_encoder_grad=1` is queued for seeds 1–3 as `fbj_core_gphiF1_*`. The seed-1 runs with 0.1 stay as an ablation of the FB gradient that reaches phi through F.
- `fbwm.py` defaults stay as they are (0.1, as in `tdmpc_fb_3.py`). The flag is passed explicitly.

## 2026-11-14: 500k updates, walker only, 1 seed, baselines TD-MPC (MPPI topline) and offline TD3 (policy topline)
- **500k updates by default** in `fbwm.py`, `fb_jax.py`, `tdmpc_jax.py` (user: returns stagnate or drop after 500k). On walker, the 300–500k means already equal the 800k–1M means for FB, sepB and gphi.
- **Focus on walker, 1 seed per config** while the method is being improved (user). The 5 walker tasks are stand, walk, vel8, run, flip.
- **Baselines (user):** TD-MPC per task on the true reward is the topline for our MPPI (same planner, task-specific reward and Q). Offline TD3 per task is the topline for our policy (not written yet). Zero-shot FB is the floor for our policy.
- **Targets:** fbwm MPPI close to TD-MPC; fbwm policy close to TD3, and at least FB.

## 2026-11-14: the C4 adaptation baseline uses a clean reward-free JEPA world model
- User: no fbwm checkpoint, because the baseline needs a clean world model. Pretrain encoder and dynamics with the latent consistency loss only (no FB, reward or value terms) and freeze them. Then, per task, train reward, Q and policy heads on N labeled states for k updates, and plan with the same MPPI.
- Parked for later (research_plan §8): FB as a layer on top of a pretrained, frozen world model.

## 2026-11-15: M1 and M2 approved (user)
- **M1 (F-consistency on imagined latents):** along the H-step rollout of the update, `fcons_coef * sum_t rho^t ||F(h_hat_t, a_t, z) - sg(F_target(phi_target(s_t), a_t, z))||^2`, with the same z and actions as the FB loss. The gradient goes into F only.
- **M2 (grounded dynamics):** the same loss, with the gradient also flowing into the dynamics and phi (`fcons_model_grad=1`).
- Night plan (walker, seed 1, 500k, policy eval every 25k, MPPI every 100k, final 10 episodes): control gphiF1; M1 and M2 with `fcons_coef` in {0.1, 1}; eval-only diagnostics D1–D4 and M0 (normalised MPPI) on the existing checkpoints; the Dyna-style labels-only baseline. Seeds 2–3 only if a fix passes the gate (final MPPI ≥ its policy and > FB's 567 on the 5 walker tasks).

## 2026-11-15 (night): autonomy while the user sleeps (user)
- Claude may propose new experiments and run them overnight.
- Method changes are allowed only if they are principled and **validated by another agent** first: an independent reviewer subagent gets the proposal, the evidence and the code diff, and must agree before anything is queued.
- Each proposal, the reviewer's verdict and the runs it produced go into this file and the journal.
- Unchanged: nothing on the cluster is cancelled or deleted, and CLFB is never touched.

## 2026-11-15 (night): least-squares weights for MPPI's per-step reward (reviewer-approved)
- **Proposed:** least-squares z inference (`(E[BBᵀ] + λI)⁻¹ E[BR]`) used everywhere.
- **Reviewer (independent agent): REJECT as written.** In FB, `Fᵀ E_ρ[B R]` is the exact Q of R whatever Cov(B) is, so the policy and the terminal value must keep `z = E[B R]`. A least-squares z would make them optimise a different reward.
- **Reviewer: APPROVE the split variant.** MPPI's per-step reward uses `g(h)ᵀw` with `w = c (E[BBᵀ] + λI)⁻¹ E[BR]`, where c is the same factor that puts z on the sphere. `Fᵀz` is exactly the Q of the projected reward `Bᵀw`, so the per-step rewards and the terminal value then describe the same reward, which the current code doesn't. λ = 1e-2 · trace(Cov)/d. The policy is unchanged, so the policy-only eval is a sanity check.
- Implemented as `mppi_reward=lstsq` (default `mean`, unchanged). New diagnostic columns `<task>_corr_lstsq` give the correlation of R with its centred least-squares fit on B, i.e. how well span(B) represents R. The old 0.57 also mixed in Cov(B) ≠ I.

## 2026-11-15 (night): E[BBᵀ] from unlabelled states for the lstsq weights (reviewer-approved)
- `z_cov=dataset` (default `labeled`, unchanged): `E_ρ[BBᵀ]` is estimated on 100k unlabelled dataset states, and `E[B R]` still comes from the N labelled states. Same estimand, lower variance, no selection bias. The reviewer approved it and checked the diff.
- Reviewer caveat: `w = Cov⁻¹ E_N[BR]` still amplifies label noise in the low-eigenvalue directions (`d_eff/N`), and ridge 1e-2 is effectively unregularised at d = 200, N = 100. Recommendation: scale the ridge with d/N, or pick it so d_eff is well below N. Tested tonight as a hyperparameter sweep (ridge 0.01 / 0.1 / 1).
- Reporting rules: N counts reward labels only; `E[BBᵀ]` uses 100k unlabelled dataset states; the matched baseline is the full-dataset adapted arm (the labels-only arms had less information). The `<task>_corr_lstsq` column is the in-sample fit on the labelled states, not the w used by MPPI.

## 2026-11-15 (morning): findings that bound the claims
- The lstsq planner reward helps on walker (+52 at z_dim 50 over 3 seeds; much more at z_dim 200) and **not on cheetah** (paired: 501 vs 501). It's not a general improvement yet.
- z_dim above 50 breaks FB training on cheetah (FB 300 → 188 at z100 → 97 at z200), so the safe default is z_dim 50.
- At z_dim 50, Cov(B) from the whole dataset (computed once, task-independent, as the user suggested) matches the in-sample estimate. Candidate default: `z_cov=dataset` at z_dim 50; not changed yet, the user decides.

## 2026-11-15: improvement runs, hyperparameters of the agreed method (user: "queue items 1 and 2")
- Item 1, a stable larger B: z_dim 200 with batch 2048, or with lr 5e-5. At batch 1024, z200 made FB training collapse on cheetah, but gave the best walker result.
- Item 2, a longer horizon: z_dim 50 with discount 0.98 / 0.99 (the default 0.95 is an effective horizon of about 20 steps; TD-MPC uses 0.99).
- Walker and cheetah, seed 1, lstsq + dataset covariance, 500k updates. Queued ahead of the baseline seeds.

## 2026-11-15: stabilising large z_dim (user: needs the MPPI > policy claim; no MPPI distillation)
- Diagnosis from the training logs: FB's off-diagonal TD term and the auxiliary Q-loss scale with z_dim, and at z200 they keep rising throughout training (cheetah: offdiag 317 → 872, q_loss 2050 at the end, against 175 / 368 at z50). ‖F‖ grows too. A TD instability from coefficients tuned at z50.
- Hyperparameter fixes, cheetah z200, seed 1: q_loss_coef 0.0025 (0.01 × 50/200), tau 0.005 (new `tau=` flag), b_ortho_coef 4; plus the running batch 2048 and lr 5e-5. Cheetah z200 jobs first. Whatever stabilises cheetah gets confirmed on walker and with 3 seeds.
- Possible principled follow-up (a method change, needs the user's yes and a reviewer): an FB loss invariant to d.
- User: no MPPI distillation into the actor.

## 2026-11-15: BTD task vectors (user's design, method change requested by the user)
- `fbwm_btd.py` is a copy of fbwm.py. It trains on a pool of 100k task vectors z = √d · normalize(mean of B over a random dataset subtrajectory of btd_len steps), the FB task vector of "visit this segment", recomputed every 10k updates from the target B. `btd_ratio` is the fraction of the batch drawn from the pool (default 1).
- Hypothesis: at z_dim 200, uniform z are almost orthogonal to real task vectors, so the actor and F waste capacity on them; that would explain why the policy collapses on cheetah z200 while the reward basis improves (corr_lstsq 0.82).
- Test (user: cheetah z200 only): btd_len 50 and 100. Also mix_ratio 0.8 / 1.0 and hidden_dim 2048 in fbwm.py (hyperparameters; `hidden_dim`/`feature_dim` are now flags).
- The hyperparameter fixes (batch 2048, lr 5e-5, q_loss_coef 0.0025, tau 0.005, b_ortho 4) didn't stabilise cheetah z200: final MPPI 141–270, policy 35–119. The FB off-diagonal term levels off at about 950 at z200 (about 195 at z50), so it scales with d rather than diverging. Discount 0.98/0.99 hurt on both envs.

## 2026-11-15: MPPI that is never worse than the policy (user approved (a); (b) and (c) are hyperparameters)
- (a) `mppi_root_check=1`: after planning, compare min_k F_k(h_0, a, z)ᵀz at the real current latent for MPPI's first action and for the policy's deterministic action, and execute the better one. MPPI can then only override the policy when F, evaluated where it was trained, agrees.
- (b) `mppi_init_policy=1`: MPPI's initial mean is the policy's deterministic 5-step trajectory through the model (instead of the shifted previous plan), with `mppi_init_std` (default 2.0, tried 0.5).
- (c) `mixture_coef` as a flag (default 0.05; tried 0.25 / 0.5).
- Tested eval-only on the best-config checkpoints of the 3 envs (walker z200 s1–3, cheetah z50 s1–3, quadruped z50 s1).

## 2026-11-15 (evening): the root check becomes the planner of record (user: "go ahead")
- Eval-only, 7 checkpoints: with `mppi_root_check=1`, MPPI ≥ policy on all of them (walker z200 615 vs 538; cheetah z50 406 vs 374; quadruped z50 679 vs 671). Without it, MPPI fell below the policy on cheetah seed 3 and on quadruped.
- The main-table and figure evals of fbwm are redone with `mppi_root_check=1` (label sweeps on the z50 checkpoints, quadruped included). New quadruped trainings use it too.
- Quadruped search, seed 1: consistency_coef 2 → 661 / 653 (best); being verified with seeds 2–3. FB walker is redone at 500k updates (3 seeds: 600 / 503 / 598), replacing the 1M checkpoints in the table and figure.

## 2026-11-15 (evening): BTD for the actor only (user's proposal)
- `btd_actor_only=1` in fbwm_btd.py: the FB loss (and its TD-target action) keeps fbwm.py's z distribution (uniform, mix_ratio of B(s)), so F and B are learned for all z. Only the actor loss uses the pool's task vectors (mean of B over 50-step subtrajectories), the z's of real tasks. Test: z200, cheetah and walker, seed 1, root-check MPPI.

## 2026-11-15 (evening): quadruped best config = consistency_coef 2
- Quadruped gphi z50 + lstsq with consistency_coef 2, 3 seeds: MPPI / policy 661 / 653, 672 / 649, 648 / 609 (mean 660 / 637), against 587 / 599 with consistency 5. MPPI > policy on all 3 seeds.
- Main table with root-check MPPI (z50, 3 seeds), MPPI − policy: walker +48, cheetah +48, quadruped −11 (seed 2: 468 vs 540). The root check guarantees MPPI ≥ policy only as judged by F at the real state, not in return.
- Next (user): consistency 2 on walker z200 and cheetah z50 (seed 1), to see if one setting fits every env; FB quadruped label sweep.

## 2026-11-15 (night): the paper's configuration (proposed for the first draft; user wants the main table and figure tonight)
- One fbwm configuration for every environment: gphi, z_dim 50, consistency_coef 2, bcons_coef 2, f_encoder_grad 1, lstsq per-step reward, Cov(B) from the dataset, root-check MPPI, 500k updates. Quadruped already has 3 seeds (660 / 637); cheetah seed 1 (430 / 329) plus seeds 2–3 queued; walker z50 consistency 2 seeds 1–3 queued.
- Walker z_dim 200 (651 / 523) goes to the ablation table, not the main one.

## 2026-11-15 (23:20): paper configuration revised to consistency 5
- Consistency 2 with 3 seeds per env (MPPI / policy): walker 597 / 611 (MPPI < policy on all 3 seeds, even with the root check), cheetah 366 / 319, quadruped 660 / 637. With consistency 5: 622 / 574, 403 / 355, 587 / 599. Consistency 2 only helps quadruped.
- The paper's single configuration is therefore gphi, z_dim 50, consistency 5, lstsq, dataset Cov, root-check MPPI (`PAPER_CONS = '5'` in tools/paper_tables.py). Quadruped with consistency 2 goes to the ablations as per-env tuning. The main table and figure are final: 3 seeds in every cell.
- Runs: consistency 2 `2026-11-15_21-58-0x_FBWM-gphi-{walker,cheetah}*`, `21-59-15_FBWM-gphi-walker`; quadruped consistency 2 `17-30-24_..quadruped___`, `18-54-37`, `18-55-15`.

## 2026-11-16 (night): k-step FB objective, debugging session (user asleep; authorised to run single-seed debug experiments, kill collapsed jobs, use A100 when H100 is full)
- New script `fbwm_nstep.py` (copy of fbwm.py, never touches it). The first version (B constant on the extra positives, model rollout) collapsed; the model-error explanation (user's hypothesis) was tested with `src=data` and with a 50k warmup: both collapse too, so the loss itself was the problem, not the model.
- Ablations (30k updates, gphi walker s1, warmup 5k): extra diagonal off = stable; Q-loss without b_extra / `b_ortho_coef` 10 / diag x0.25 = partial relief; EMA-target B for the positives = worse; B trained on the positives (`fb_nstep_bgrad=1`, src=data) = stable and, at 30k, better than k=1 (policy 289/243 vs 134/158 for k=3; k=5 ~270; k=2 ~226/164).
- Full 500k runs (walker, s1): always-on k=3 plateaus at ~400 (k=1 control ~600); k=2 reaches 622 at 75k then falls to ~450; k=3 switched back to 1-step at 100k matches the control from 125k on (policy 600-657). Reading: faster early propagation, biased asymptote (behaviour actions for steps 2..k); annealing removes the bias. Single seed, noisy evals (5 episodes per task in the periodic eval): nothing here is a result yet.
- Not done: seeds, the `model` source (needs a stable way to train B on imagined latents), other envs.

## 2026-11-21: goal-reaching baseline `wm_goal.py` (evaluation only)
- Question: is the world model learned next to FB (fbwm.py) a better planning model than a reward-free JEPA world model alone (`tdmpc_adapt.py mode=pretrain`) when the task is goal reaching? One checkpoint per run (`ckpt=`), same planner and tasks, only the checkpoint changes.
- Planner: MPPI on -dist(last imagined latent, goal latent), no policy, no value, no reward head. Executes the MPPI mean's first action. l2 or cosine give the same ranking on the sphere (phi and imagined latents are projected on the sphere of radius sqrt(latent_dim)).
- Goal latent: antmaze_medium / antmaze_large / cube_single: encode the goal observation OGBench returns in `info['goal']` (official tasks goal_1..5, episode starts at the task's own init state). cheetah / walker / quadruped: g = sum_s w(s) phi(s), w = r / sum r over 10k labelled dataset states (fresh draw per episode; the weights are normalised, which is our reading of "weighted average").
- Metric as in fbwm.py (dm_control: summed reward; goal tasks: reached at least once, plus OGBench's own `success` as `<task>_ogb_success`). Extra logged diagnostics: latent distance of the real visited states to the goal (`_dist_end`, `_dist_min`).
- Not run yet; no smoke test was run locally (user's instruction).

## 2026-11-21 (evening): OGBench goal tasks start at the official start of the task
- Without `options={'task_id': k}`, `reset()` samples one of the 5 tasks at random and starts there, so a goal can coincide with the start (free successes on goal_4 / goal_5 of antmaze_medium). Now goal_k resets with `task_id=k` in `fbwm.py`, `fb_jax.py`, `tdmpc_jax.py`, `tdmpc_adapt.py` (`utils/ogbench_tasks.py:env_reset`). Evaluation change, not a method change. Non-goal tasks and dm_control are unchanged.
- OGBench goal numbers of runs that finished before this commit used random starts and are not comparable with later ones; rerun final evals from checkpoints before comparing.
- `wm_goal.py` always used official starts. It now also logs the real (x, y) distance of the ant to the goal (`xy_dist_start/end/min`), the trajectories (`trajectories.csv`) and the maze (`maze.csv`).

## 2026-11-21 (night): wm_goal.py goal_k=K, one fixed (start, goal) pair instead of the 5 official tasks
- The official goal_1 is ~28 m from the start and no run moved (0/10 everywhere, ant stays in the start cell). `goal_k=K` (antmaze only): a dataset state s_t as the start and s_{t+K} of the same trajectory as the goal, so the goal is reachable in K steps of the data policy.
- One pair for all runs: drawn with `pair_seed` (default 0) from the dataset only, independent of checkpoint, horizon and run seed; saved as `pair.csv`. Pairs with (x, y) displacement < `goal_min_xy` (default 4 m = 2 x the success radius) are skipped, otherwise the goal could be reached for free. Success = within 2 m of the goal xy (fbwm.py's reward); `_ogb_success` uses OGBench's 0.5 m flag against the same goal.
- Episodes start from the exact dataset state (qpos, qvel set in the simulator), the goal latent is phi(s_{t+K}). First runs: both checkpoints, H=5, K=25, 10 episodes.

## 2026-11-22: goal-reaching evaluation for "FBWM vs reward-free WMs" (`docs/goal_reaching.md`)
- Protocol: LeWM's near-goal pairs (goal_k), not the official OGBench tasks (every distance planner and the FB actor are near 0 there).
  50 pairs per K from the -val file, pair_seed 0, the same pairs for every method; simulator set to the stored state.
- The latent claim rests ONLY on same-planner comparisons (user, 2026-11-21): one planner, the latent changes. The FB actor / FB-goal
  MPPI use FB's learned value and are reported apart, never against distance planners.
- Baselines: state-based LeWM (MLP encoder + SIGReg + AdaLN predictor, frameskip 5) and DINO-WM (frozen encoder = standardised state),
  ported from their papers (deviations in `wm_baselines.py`'s docstring). Run with their own CEM and with our MPPI (`planner=mppi`),
  and our latents with their CEM (`wm_goal.py planner=cem`), so the planner and latent effects are separated.
- Minimal eval set per checkpoint (reviewed): K in {10, 50, 200}, distance MPPI H=5 mppi_norm + geometry probe, FB actor (goal_k and
  official) + one MPPI row (init-policy + root check). K=200 is at the noise floor for distance planners.
- Cube reset caveat: obs[18] (gripper contact) differs after the reset by 0.3-0.8 (needs a physics step); positions are exact. Accepted.
- Every proposed experiment goes to a Sonnet reviewer first, with the list of what was already run (user, 2026-11-21).

## 2026-11-22: matched encoder and optimiser for the FBWM vs JEPA WM latent comparison
- Problem (user): FBWM is not "JEPA WM + FB". FBWM gphi uses a PhiMap encoder of width 64 (torch init, LayerNorm+tanh),
  lr 1e-4, batch 1024, EMA every step, consistency_coef 5. The JEPA WM uses the TD-MPC MLP encoder of width 256 (ELU), lr 1e-3,
  batch 512, EMA every 2 steps, consistency_coef 2, plus VICReg. The latent gap in docs/goal_reaching.md (A) mixed objective with encoder and optimiser.
- Decision (user): the same PhiMap encoder and FBWM's optimiser for both, so that only the objective differs (FB + consistency vs
  VICReg + consistency). Dropping VICReg in FBWM is accepted: in the paper, FB *replaces* VICReg as the anti-collapse term.
- Code: `tdmpc_adapt.py encoder=phimap enc_dim=<w>` (PhiMap copied verbatim from fbwm.py) and `update_freq=`; `fbwm.py freeze_wm`
  and `wm_goal.py` read either encoder from the checkpoint. Defaults unchanged.
- Staged (Sonnet review + user): antmaze-medium seeds 1-3 first. JEPA PhiMap at 64 (matches FBWM's tuned width; the FBWM-64 checkpoints
  already exist) and at 256, plus FBWM phi_hidden_dim=256 (robustness; phi256 hurt FBWM on cheetah / quadruped, journal 11-16/17).
  Then wm_goal.py distance MPPI K=10/50/200 + diag_geom. Large and cube come after, and only if the gap holds. Check JEPA latent_std / latent_rank for collapse (consistency_coef 5 was not tuned with VICReg).
- LeWM / DINO-WM keep their own architectures (as-published baselines). The matched pair is what the latent claim rests on.
