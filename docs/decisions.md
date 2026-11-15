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
