# Protocol audit, 2026-11-22 (Sonnet reviewer, read-only)

Scope: the FBWM paper as it exists in docs/paper_wm_first.md, docs/paper_draft.md (v4), docs/goal_reaching.md and decisions up to
2026-11-22 (paper/main.tex is a different paper). Notes in *italics* are corrections checked after the audit.

## A. Ranked issues

1. **Critical: the goal-reaching latent claim is confounded.** FBWM vs JEPA WM differ in encoder (PhiMap 64 vs MLP 256), lr (1e-4 vs
   1e-3), batch (1024 vs 512), EMA rate (1 vs 2), consistency_coef (5 vs 2), discount (0.95 vs 0.99), plus FBWM's bcons_coef 2 (a loss on
   B-space geometry), z_dim 200 and bc_coef 0.3 (2026-11-21_13-23-49 FBWM vs JEPAWM info.txt). The mismatched numbers are still the evidence in
   goal_reaching.md (A), (A'). Fix in progress (fbj_fair_am_*). Also: report JEPA latent_std / rank; give the JEPA WM a small sweep of
   consistency_coef and the VICReg coefs; demote the mismatched numbers.
   *A "FBWM without the FB loss" control is not usable as proposed: without FB (and without VICReg) consistency collapses; with VICReg it is
   the matched JEPA WM. The open part is bcons: it is FB-specific (g is B's head), so it belongs to FBWM, but the paper should say that the
   latent is shaped by FB + bcons.*
2. **Critical: antmaze in the zero-shot table is negative and thin** (hpc/status/paper_main_table.json, n=1): FBWM MPPI m9 3.3, goal_1 0,
   below FB 14.2 and FBWM policy 13.4. "m9" mixes velocity tasks (against the goal-only rule). Fix: goal_1..5 only, 3 seeds, show the negative
   result with its reason, or move antmaze to the goal-reaching study only.
3. **Major: DMC statistics are weak.** n=3, t-CIs overlap for most pairs (walker 622 ± 118 vs FB 571 ± 89; quadruped 587 ± 266 vs 583 ± 175).
   Zero-shot rows average 20 label draws per seed; adapted rows use one draw. No aggregate (IQM / rliable) despite research_plan §3. Fix: paired
   per-seed differences with bootstrap CIs, unified label draws / episodes, 5 seeds on headline rows, "tends to" wording.
4. **Major: config chosen on test-task return, tuning asymmetric.** consistency 2 vs 5, z_dim, width, discount, lstsq, root check, Cov(B)
   tuned for FBWM on walker / cheetah (~30+ configs, journal 11-15..17); adapted baselines got one fixed recipe; FB no z_dim / trunk tuning.
   Fix: tuning-budget table per method (appendix); small matched sweep for the adapted baselines; a held-out check of FBWM on tasks not used in tuning.
5. **Major: LeWM / DINO-WM are not faithful.** DINO-WM's frozen encoder = identity; LeWM = AdaLN MLP, 1 frame, no attention; their CEM
   (frameskip 5) vs our MPPI. Under LeWM's CEM all four latents are within noise (goal_reaching.md A''). Fix: rename ("raw-state WM
   (DINO-WM-style)", "state LeWM"); LeWM loss on our backbone (queued); the CEM result in the main text.
6. **Major: goal-reaching protocol.** Near-goal pairs (goal_k) are easier than the official tasks (FBWM 0-5/50 there, every planner near 0):
   don't call it "OGBench goal tasks". n=50 per cell: binomial 95% CI ≈ ±11 points at p ≈ 0.2, so 3-6 success differences are noise; pooled
   3-seed cells (48/150 vs 19/150) are the exception; large and cube single seed. FBWM loses on cube (21 vs 39) and under CEM: scope the claim
   to "antmaze mazes, short-horizon MPPI". -val pairs are in-distribution (same maze and data policy): say so. Fix: Wilson CIs, paired test,
   wording; seeds 2-3 on large.
7. **Major: C4 (equal labels) fairness.** The adapted baselines start from the mismatched JEPA WM. The zero-shot methods use Σ = Cov(B) on
   100k unlabelled dataset states, so test-time dataset access is not zero: keep it in the main text. Adapted-full wins on walker at every N:
   say so at abstract level. MPPI wall-clock (~100x a policy step) must be reported next to the 20 s adaptation. Fix: rerun adapted baselines
   from a matched PhiMap WM (at least walker); time per step for every method.
8. **Major: missing ablations** (draft §4.5): bcons off; sepB vs gphi; model-free best-of-K by F with no world model (key for "planning
   helps", eval only); frozen reward-free WM + FB (verify the Table-1 numbers exist before citing).
9. **Minor-major: column meaning and defaults drift.** eval_mode=policy from 2026-11-22 (unprefixed columns = policy); NUM_OUTER_STEPS 1M ->
   500k on 2026-11-14 (old_ckpt filter in tools/paper_tables.py); periodic 5 vs final 10-20 episodes; experiments.md "mean over last 10 evals"
   conflicts with the final-eval protocol; OGBench goal reset changed 2026-11-21. Fix: one appendix sentence each, eval_mode in a manifest.
10. **Minor: toplines.** TD-MPC / TD3 trained on ~1e7 true-reward labels; antmaze TD-MPC n=1; TD3 without BC is not a proper offline topline
    on antmaze (TD3+BC).
11. **Minor: FB baseline architecture.** Say which published FB config fb_jax.py follows.
12. **Minor: metrics.** *The 2 m radius (decisions 2026-11-21) belongs to the older start-state goal protocol; the goal_k protocol uses
    OGBench's 0.5 m. Two protocols, not an inconsistency, but the paper must name the one it reports.* Remove "best of last 10 evals" from experiments.md.
13. **Minor: wording stronger than evidence.** "FB shapes a better latent" (one env family, fails on cube); "FB replaces VICReg" (one probe);
    "beats adapted baselines at equal labels" (loses to adapted-full on walker); "planning helps" (+48 on two envs, overlapping CIs, negative on
    quadruped and antmaze). Report both widths, the claim rests on 64/64.

## B. Already fine, state it in the paper
Final checkpoint (never best); t-quantile 95% CIs; same planner across latents in (A), FB-valued planners apart; info.txt metadata and
traceability (tools/goal_results.py, tools/paper_tables.py); one pair set (pair_seed 0) for all methods, -val file, displacement filter;
reset fix and cube obs[18] caveat; negative results documented; consistency 2 vs 5 disclosed; toplines labelled as true-reward.

## C. Minimal plan
1. No runs: rescope and reword; rename baselines; Wilson CIs and paired comparisons; drop antmaze m9; name the radius; appendix notes on eval_mode and defaults.
2. Eval only: model-free best-of-K by F on DMC (existing checkpoints); wall-clock table; unified label draws / episodes for adapted baselines.
3. Training (partly queued): matched antmaze-medium runs; small JEPA consistency / VICReg sweep; latent_std / rank. Gate the goal-reaching section on it.
4. Training: bcons-off and sepB ablations on walker, cheetah, 3 seeds; seeds 4-5 on headline DMC rows if budget allows.
5. Small: matched sweep for the adapted baselines; antmaze goal_1..5, 3 seeds, shown even if negative.
