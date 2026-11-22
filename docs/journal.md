# Journal

Dated session notes, newest last.

## 2026-11-13: back on the project
- Back after ~6 weeks (last runs 2026-10-05). Read the repo; pulled the JZ code (local edits backed up in `backup_local_2026-11-13/`); synced the JZ logs into `hpc/` (131 run folders, no images).
- State of the results (single seeds unless stated; all run before the tracker, kept local):
  - sepB (`tdmpc_fb_2.py`) MPPI beats FB on cheetah, about 440-520 vs 300-370 (8 of 9 runs, mean of the last 10 evals); stable on quadruped where the actor and FB collapse late; worse than FB on walker.
  - gphi best config (consistency 5, bcons 2): cheetah 414 (MPPI) vs 305 (policy), quadruped 666 vs 660. bcons helps MPPI.
  - Raw-state model (`phi_identity`): MPPI collapses on walker (80 vs 550).
  - Repeated runs used the same default seed (0); run-to-run spread came from GPU nondeterminism only. Seeds must be varied from now on.
- Discussion: sepB framed as "B is a target, as the reward in TD-MPC"; proposed grounded sepB (B-head loss into phi), not implemented (method change).
- Wrote `docs/research_plan.md`, set up the CLFB-style tooling (`tools/`), the tracker (https://claude.ai/artifact/3UineFBuRZiwE4gzMLRkhn), `CLAUDE.md` and `docs/`.

## 2026-11-13 (evening): fbwm.py smoke test on Jean Zay
- Ran the fbwm.py smoke test on H100 dev (job 153137, `scripts/queue/fbj_smoke_fbwm.sh`; it couldn't run locally). It took 12.5 min and exited 0.
- Each variant ran 20k updates on cheetah with eval_mode=both. Both runs wrote a `finished:` line, two `eval` rows (5 episodes) and one `final` row (10 episodes), and saved `fbwm_<variant>_final.pickle`. Losses fit the variant: sepB has a non-zero `bhead_loss` with `bcons_loss` and `b_enc_grad_norm` at 0; gphi is the reverse.
- Runs: `logs/2026-11-13_16-55-04_FBWM-sepB-cheetah`, `logs/2026-11-13_17-00-46_FBWM-gphi-cheetah`. These check the code only; returns after 20k updates mean nothing.
- `tmp.py` is in `.git/info/exclude` (local) so `queue.py submit`'s dirty check doesn't block on it.
- Later: added the final eval to `fb_jax.py` and `mppi_time` / `policy_time` to `fbwm.py`, with a smoke test (job 154300). Queued the §4.1 core runs: 27 jobs, seeds 1–3, `eval_each_steps=20`; seed 1 submitted as jobs 154476–154491. The tracker has a `studies/core-C1` doc and a Log entry.
- The smoke test (job 154300) found a bug in `fb_jax.py`: `run_benchmark(num_evals=num_evals)` raised a NameError when the function was defined, seconds after start. It's fixed in the commit after `c0a327f` and pulled on JZ. The `fbwm.py` part passed, with `mppi_time` and `policy_time` logged (MPPI ~18 s and policy ~2 s per eval with 5 episodes, after jit). FB cheetah and walker for seed 1 crashed the same way and were requeued (jobs 155057, 155061). sepB and gphi are unaffected.

## 2026-11-14
- The queue stalled overnight: no timer was installed, and `queue.py submit` refused because of uncommitted files in `tools/`. Seeds 2 and 3 were submitted at about 10:50.
- Tracker tick: `tools/tracker_tick.py` plus `sync_from_hpc.sh --tick`, which fetches only the CSVs and info.txt of tracked runs as one tar stream (~13 s per tick). It prints PUSH / UNCHANGED / IDLE / SUBMIT BLOCKED. The loop is meant to run from a fresh session (see CLAUDE.md).
- Seed-1 results (final 10-episode eval, MPPI/policy):
  - walker: FB 600, gphi(0.1) 551/532, sepB 394/355. Same ordering as October; see §4.3.
  - cheetah: sepB 524/354, gphi(0.1) 481/378, FB 414.
  - quadruped: FB 697, sepB 345/157, gphi(0.1) 347/391. sepB follows its October curve and then goes unstable late. gphi(0.1) collapses after 620k.
- The collapse is explained by a config mistake: `f_encoder_grad` was 0.1 instead of 1 (see decisions). Cancelled gphi s2 (182240, 182243, 182246), queued `core_gphiF1` × 3 seeds.
- The gphiF1 seed-2 jobs (184046–184048) failed after 2 minutes: they started in the same second as seed 1, so both runs got the same log folder name (`FileExistsError`). `fbwm.py` and `fb_jax.py` now append `_` on a collision (d1a5c04, pulled on JZ before the pending seed-3 jobs started). The 3 jobs are requeued. One tick sync also failed once (`tar: Error is not recoverable`) and worked on retry.
- Automatic tracker refresh: `claude -p` (headless) has no ArtifactData tool, but a background session (`claude --bg`) does. Started `claude --bg -n fbj-tracker --model haiku '/loop 2m Tracker tick: follow CLAUDE.md "Cheap automatic refresh" exactly. ...'` (session 3bf2447c). Its first push landed at 13:21 (snapshot v8). It stops on IDLE and doesn't survive a reboot.

## 2026-11-14 (evening)
- C4 label sweep on walker, seed 1, frozen JEPA world model `2026-11-14_17-14-43_JEPAWM-walker`. Mean over 5 tasks, N = 100 / 300 / 1k / 3k / 10k:
  - adapted baseline with full dataset access, MPPI (final): 594 / 519 / 623 / 678 / 664;
  - adapted baseline with only the N labels (`q_data=labeled`), MPPI (final): 57 / 157 / 161 / 338 / 269;
  - zero-shot gphiF1 MPPI: 349 / 390 / 463 / 477 / 472;
  - zero-shot FB policy: 367 / 469 / 554 / 593 / 598.
  So zero-shot wins clearly without dataset access and loses with it, and FB beats fbwm from N = 300 on. Runs: `TDMPC-adapt-walker-*` and `TDMPC-adaptL-walker-*` (17-53 to 21-5x), `zlabels` evals of the seed-1 checkpoints `2026-11-14_11-55-56_FBWM-gphi-walker` and `2026-11-13_17-30-10_FB-walker`.
- Hypotheses for MPPI < policy in fbwm (analysis, not yet tested): F and B are queried on imagined latents they were never trained on (the terminal F is about 80% of the plan score); the MPPI softmax may be flat because values are divided by d; the 5-step reward sum exploits pointwise errors of Bᵀz.
- GPU packing test (job 204735, cancelled after stage 2 at the user's request): one fbwm gphi run already keeps the H100 at 100% during training (4.04 s / 1k updates). Two copies run at 8.88 s / 1k each, 0.91× the throughput of one, so packing doesn't help.
- Agreed: keep 500k updates. New eval schedule: policy every 25k, MPPI every 100k, final 10-episode eval at 500k (to implement: `mppi_eval_each`).
- The zsh word-splitting bug (`for n in $NS`) made the first sweep jobs run N = 100 only; I requeued the rest with explicit values.

## 2026-11-15 (night): wave 1 results (walker, seed 1)
- Phase 2, 500k, final MPPI / policy: control 470 / 527; M1 fcons 0.1: 557 / 580; M1 fcons 1: 551 / 519; M2 fcons 0.1: 591 / 611; **M2 fcons 1: 590 / 574, which passes the gate** (≥ its policy and > 567). MPPI averaged over 300k–500k: control 527, M2 fcons 1: 547, so the gap is within single-seed noise. Runs: `2026-11-15_00-3x_FBWM-gphi-walker*`.
- Phase 1 diagnostics on the seed-1 checkpoints:
  - MPPI doesn't change with the horizon (gphi H = 1/3/5: 501 / 453 / 484).
  - F on imagined latents along dataset actions stays close to F on real latents (correlation ≥ 0.95, relative error 0.17 → 0.33, no bias).
  - Normalising the elite values changes nothing (spread about 0.013).
  - Terminal-F-only: +134 for sepB (its B-head reward on imagined latents hurts it), no change for gphi.
  - So the off-distribution and flat-softmax hypotheses are weak or rejected. The likely bottleneck is the zero-shot objective itself: Bᵀz has correlation 0.57 with the true reward, and planning can't beat a policy that optimises the same objective.
- Wave 2: M2 fcons 1 and control, seeds 2–3 (gate); M2 fcons 1 with z_dim 100 and 200, seed 1 (hyperparameter of the agreed method).
- Dyna-style labels-only baseline (`q_data=labeled_model`, `TDMPC-adaptLM-walker-*`), MPPI final, mean over 5 walker tasks, N = 100 / 300 / 1k / 3k / 10k: 183 / 342 / 346 / 449 / 615. Without dataset access, zero-shot FB (367 / 469 / 554 / 593 / 598) wins clearly up to N = 3k and ties at 10k.
- Proposed least-squares z inference (`z_infer=lstsq`, uncommitted code) and sent it to an independent reviewer agent. Caveat found while writing it up: in FB, Fᵀ E[B R] is the exact Q of R whatever Cov(B) is, so least squares is only principled for the reconstructed per-step reward and as a better diagnostic of span(B), not for the policy or the value. The reviewer's verdict is pending.
- `mppi_reward=lstsq`, eval-only, 10 episodes (MPPI mean → lstsq, policy before → now): control s1 470 → 575 (policy 527 → 525); M2 fcons 1 s1 590 → 570 (574 → 567); gphiF1 1M 484 → 519 (471 → 458); sepB 1M 387 → 460 (370 → 363). The policy is unchanged, as it should be. MPPI is now above the policy on all four checkpoints, but the evals are noisy; a paired 20-episode confirmation is queued (`lstsq20_*`).
- Better span diagnostic: corr of R with its least-squares fit on B is 0.72–0.79, against 0.57–0.67 for Bᵀz_mean. B's span represents the walker rewards better than the old 0.57 suggested, though still below the adapted reward head's 0.97.
- Paired 20-episode evals (`lstsq20_*`), MPPI mean → lstsq (standard error about 10–14): control s1 475 → **564 (+89)**; gphiF1 1M 483 → **526 (+43)**; sepB 1M 397 → **460 (+63)**; M2 fcons 1 s1 567 → 568 (+1). The policy is unchanged. With lstsq, MPPI ≥ policy on every checkpoint. The control with lstsq and M2 tie FB (567).
- Wave 2, final MPPI / policy with the mean planner reward: control s2 624 / 588, s3 551 / 583 (3-seed mean 548 / 566); M2 fcons 1 s2 618 / 586, s3 540 / 480 (3-seed mean 583 / 547). M2 is +35 on MPPI, which is not conclusive given 60–150 points of spread between seeds. M2 with z_dim 100: 591 / 589; with z_dim 200: 600 / 552; no clear effect at seed 1. Runs: `2026-11-15_01-3x_FBWM-gphi-walker*`. Paired mean/lstsq 20-episode evals of these 6 checkpoints are queued.
- Paired mean/lstsq on the wave-2 checkpoints (20 episodes). Control, 3 seeds: MPPI mean 549 → **lstsq 601** (+89 / +27 / +40), policy 574, so +34 over FB. M2, 3 seeds: 575 → 584, policy 560. With lstsq, M2 adds nothing over the control.
- **z_dim (M2, s1):** corr_lstsq 0.78 / 0.84 / **0.92** at z_dim 50 / 100 / 200; MPPI with lstsq 568 / 606 / **689** (about TD3's 685; TD-MPC 792); policy 572 / 587 / 522. A larger B represents the rewards much better, and the lstsq planner reward uses that. Checkpoints: `2026-11-15_01-37-36_FBWM-gphi-walker` (z200), `..._` (z100).
- Wave 3 (the gate passed for z_dim 200): M2 z200 seeds 2–3; control z200 s1 (is M2 still needed?); M2 z400 s1. All with `mppi_reward=lstsq` during training (it only affects the evals) and 20-episode final evals.
- **Wave 3 (20-episode finals, lstsq planner reward):** M2 z_dim 200, MPPI / policy: s2 669 / 595, s3 639 / 432 (with s1 689 / 522, the 3-seed MPPI mean is **666**, against FB's 567, TD3's 685 and TD-MPC's 792). Control z200 s1: 665 / 488, so M2 isn't needed at z200. M2 z400 s1: 474 / **103** (the policy collapses; corr_lstsq saturates at 0.92). FB z_dim 100 / 200: 580 / 573, so FB doesn't gain from a larger B. Runs: `2026-11-15_02-59-26`, `03-01-48` (M2 z200 s2, s3), `03-01-48_` (control z200), `03-01-47` (z400), `03-02-42_FB-walker(_)`.
- Wave 4: control z200 seeds 2–3; C4 label sweep (z_labels 100 … 10k, lstsq) on the three M2 z200 checkpoints.
- **Wave 4:** control z200 (no M2), 3 seeds: 665 / 634 / 655 (mean 651, against 666 for M2 z200), so M2 isn't needed at z200. **C4 label sweep for the improved method** (M2 z200 + lstsq, 3 seeds, MPPI, N = 100 / 300 / 1k / 3k / 10k): 348 / 484 / 582 / 648 / 671. That beats FB (367 / 469 / 554 / 593 / 598) from N = 300 on, beats the labels-only Dyna baseline at every N, and matches the full-dataset adapted baseline (594 / 519 / 623 / 678 / 664) from N = 3k on. At N = 100 the least-squares fit overfits (in-sample corr 1.00 with 200 dims). Proposed `z_cov=dataset` (Cov(B) from unlabelled states); sent to a reviewer agent.
- **Wave 5 (`z_cov=dataset`, negative):** estimating E[BBᵀ] on unlabelled states is worse at every N (N = 100: 307–337 vs 348; 300: 416–457 vs 484; 1k: 501–555 vs 582; 10k: 648 vs 671). With the in-sample covariance, w is the OLS fit, whose variance depends only on the residual noise (the reviewer's point 2). `z_cov=labeled` stays the default. A larger ridge helped in every row, so wave 6 tries ridge 0.1 / 1 with the labelled covariance at N = 100 / 300 / 1k.
- **Wave 6:** a lstsq ridge of 0.1 / 1 with the labelled covariance makes no difference (N = 100: 341 / 355; 300: 482 / 476; 1k: 575 / 573, against 348 / 484 / 582 with 0.01). At small N the limit is z itself (E_N[BR] from 100 states; the policy is also about 350), not the lstsq weights. End of the night: no jobs left, nothing cancelled.

## 2026-11-15 (morning): cheetah
- Cheetah, seed 1, 20-episode finals (mean over 5 tasks): TD-MPC 685 (its policy 631); offline TD3 539; **fbwm gphi z50 + lstsq MPPI 511** (policy 335); FB z50 277; gphi z200 + lstsq 240 / 128; M2 z200 176 / 62; **FB z200 97**. The lstsq planner reward generalises: 1.8× FB and 95% of TD3. **z_dim 200 makes training collapse on cheetah**, FB included, even though corr_lstsq rises 0.62 → 0.82. So z_dim isn't a general improvement: it works only while FB training stays stable. Runs: `2026-11-15_0[6-8]-*-cheetah*`.
- Next (user: "run them"): cheetah gphi z50 + lstsq and FB z50, seeds 2–3 (the gate passed); z_dim 100 on cheetah (fbwm + FB) and walker (fbwm without M2; FB z100 already ran last night: 580).
- **Cheetah, 3 seeds, z50 + lstsq:** MPPI 511 / 328 / 352 (mean 397), policy 358; FB z50 277 / 295 / 328 (mean 300). fbwm beats FB on every seed, but MPPI vs its own policy is mixed (+176 / −16 / −43). z_dim 100 on cheetah: fbwm 303 / 290, FB 188. Any z_dim above 50 degrades FB training on cheetah. Walker gphi z100 + lstsq: 628 / 550. Quadruped (submitted before the user asked for cheetah only): fbwm z50 + lstsq 598 / 662, FB 563.
- **Paired mean vs lstsq on the cheetah z50 checkpoint:** 501 vs 501. lstsq gives nothing on cheetah, so fbwm's lead there comes from the world model and planning. The lstsq gain is walker-specific so far.
- **Cov(B) from the dataset at z_dim 50** (user's suggestion): walker 3 seeds, N = 1k / 10k: 528 / 596 vs 546 / 599 from the labels; cheetah: 439 / 509 vs 441 / 501. Equal within noise. Last night's loss was specific to z_dim 200. Computing Cov(B) once at training time is fine at z_dim 50.

## 2026-11-15 (afternoon)
- Baseline rows complete with 3 seeds (118 jobs): TD-MPC 789 / 684 / 831, TD3 681 / 510 / 753, adapted with full dataset 670 / 343 / 517, adapted labels-only 594 / 331 / 449 (walker / cheetah / quadruped).
- Improvement attempts, seed 1: discount 0.98 / 0.99 hurt on both envs. No hyperparameter fix (batch, lr, q_loss_coef, tau, b_ortho) stabilises z200 on cheetah; the FB off-diagonal term scales with d rather than diverging. BTD (user's design, `fbwm_btd.py`): cheetah z200 L50 342 / 119 (best MPPI at z200, still below z50), L100 168 / 224, walker z50 596 / 530, cheetah z50 284 / 144. Dropped (user). mix_ratio 0.8 / 1.0 and hidden_dim 2048 don't fix cheetah z200 either.
- Quadruped z_dim: 50 → 598 / 662, 100 → 532 / 611, 200 → 560 / 476. z50 stays best; seeds 2–3 queued, with FB seeds 2–3.
- Best configs: walker gphi z200 + lstsq (651 / 523, 3 seeds); cheetah gphi z50 + lstsq (391–397 / 358–372, 3 seeds); quadruped gphi z50 + lstsq (598 / 662, seed 1). MPPI clearly beats the policy only at large z_dim (walker).
- **Evening results (seed 1 unless noted):**
  - BTD actor-only at z200 (`btd_actor_only=1`): cheetah 134 / 75 (collapse); walker 638 / 593 (the policy +70 over plain z200's 523, MPPI about the same).
  - Consistency 2: cheetah z50 430 / 329 (vs 403 / 355 with consistency 5 + root check, 3 seeds); walker z200 615 / 481 (vs 651 / 523).
  - Quadruped consistency 2, 3 seeds: 660 / 637.
  - Main table with root-check MPPI (z50, 3 seeds): walker 622 / 574, cheetah 403 / 355, quadruped 587 / 599 (seed 2: 468 / 540).
  - FB quadruped from the label sweeps: 583 (3 seeds).
- **End of session (2026-11-15, 23:30):** the main table and label-budget figure are final in the tracker (paper config: gphi z50, consistency 5, lstsq, dataset Cov, root check; 3 seeds everywhere). Consistency 2 was tried as the paper config and rejected (worse on walker and cheetah). The handoff for the paper draft is in research_plan.md ("Paper status and handoff"). Anything still on the cluster is eval-only and not needed for the table. The systemd queue timer stays active; the tracker loop isn't running.

## 2026-11-15 (late): first paper draft
- `docs/paper_draft.md`: abstract, intro, background, method (shared latent, L_B grounding, lstsq per-step reward with the projection argument, root check), experiments (Table 1 at N = 10k, label-budget table on walker + cheetah), related work, limitations, appendix.
- A first-reader agent reviewed it. The math checked out. Fixed: the abstract's walker claim (the full-dataset baseline wins at every N, not only small N), scope of the label sweep (walker + cheetah), "on par" softened, undefined terms, the z-rescaling caveat, the noise explanation for w, the fbwm-policy row in Table 2.
- Open TODOs in the draft: L_B-off and separate-latent ablations in the paper config, a model-free argmax-F baseline, paired per-seed CIs, wall-clock per episode, quadruped label sweep, distance of Σ from I, FB baseline details.

## 2026-11-16: paper draft v2-v4
- v2: rewritten against `docs/writing_checklist.md` (branch writing-checklist). v3 and v4: after two first-reader rounds, each by a fresh agent.
- Fixed: the z-rescaling argument (z is an input to F and pi; cz is the task vector of c r), the L_B bound (vs the target B, mean-square, dataset actions, factor ||w||), FB optimality (exact vs rank d, z_r only sees the projection of r), claims tied to the CIs, the planning gain not yet separated from action selection by F, walker small-N result where FB beats fbwm, naming (fbwm-MPPI / fbwm-actor).
- Still TODO in the draft: CIs for Table 2, the model-free best-of-K control, L_B-off and separate-latent ablations, FB baseline details, the quadruped label sweep.

## 2026-11-16: world-model accuracy (open-loop rollouts vs the simulator)
- Question: is the quadruped world model worse than walker's and cheetah's? Tool: `fbwm.py eval_ckpt=... diag_wm=1` (100-step open-loop rollouts from phi(s_0) with real actions, dataset and policy actions, vs phi(s_t)), report: `python tools/diag_wm_report.py`. Jobs 255116-18, V100, the 9 paper-config checkpoints (consistency 5), about 8 min per env.
- Latent error normalised by the latent spread (lat_nmse), dataset actions, mean of 3 seeds, t = 1 / 5 / 10 / 25 / 50 / 100: walker 0.03 / 0.07 / 0.17 / 0.78 / 1.62 / 1.82; cheetah 0.01 / 0.02 / 0.04 / 0.11 / 0.26 / 0.50; quadruped 0.05 / 0.19 / 0.34 / 0.73 / 1.10 / 1.37 (1 = as bad as the mean latent).
- Short horizon (t <= 10, where MPPI plans): quadruped is the worst, about 2x walker and 8x cheetah. Beyond t = 25 walker is the worst and collapses (cos 0.26 at t = 50, quadruped 0.53, cheetah 0.88).
- The reward is predicted well on quadruped: corr of the planner reward with the true reward at t = 5 is 0.79 imagined vs 0.80 real (dataset), 0.87 vs 0.89 (policy). Walker 0.75 vs 0.76, cheetah 0.52 vs 0.53.
- The quadruped latent is lossy even on real states: a probe from phi(s) to the 78-d observation has normalised MSE 0.22, walker 0.005, cheetah 0.08. The model error at t = 1 already is near this floor, so the loss is in the encoder (50 dims for 78) more than in the dynamics.
- Not tested: whether this explains MPPI = policy on quadruped; latent_dim 100 on quadruped would test it (needs the user's yes for a config change per env).

## 2026-11-16 (night): capacity sweeps, what is not the bottleneck
- Protocol: paper config (gphi z50, consistency 5, lstsq, dataset Cov, root check), only the named flag changed, training-run final eval (20 episodes, N = 10k). 2 seeds per exploratory config (user, 2026-11-16); a third seed only if the two differ a lot or the config is a candidate for the table. Baselines (MPPI / policy): walker 622 / 574, cheetah 403 / 355, quadruped 587 / 599.
- quadruped latent_dim 100 (3 seeds): MPPI 578 / 610 / 644 = 610.5, policy 614. Walk +56, run +28 vs latent 50, stand and jump unchanged. Within the baseline's spread (676 / 468 / 618). The rollout check shows no change in open-loop latent error (1-step 0.055 vs 0.047, 5-step 0.18 vs 0.19), only the probe floor drops (0.22 -> 0.165). So the quadruped encoder bottleneck is not what limits MPPI.
- phi_hidden_dim 256 and mlp_dim 1024 (2 seeds each): walker 630 / 575 and 586 / 619 (no change from 622). Cheetah 207 / 296 and 247 / 301 (clearly worse than 403, both seeds). Quadruped still running when this was written.
- Correction to the earlier reading: corr_lstsq (how well span(B) fits the reward) does not separate easy from hard tasks (walker ~0.77 on every task, scores 356-869; cheetah vel2Back and vel10 both 0.68, scores 865 and 156). So "the reward is not in span(B)" is not the explanation of the weak fast tasks.
- Next: `diag_wm` now also writes `diag_q.csv` (F^T z vs the discounted return actually obtained on real policy episodes, and the part of the true return that span(B) keeps), queued on the 9 paper checkpoints (jobs fbj_diagq_*). It separates "F's value is wrong on fast tasks" from "span(B) is missing the reward".

## 2026-11-16 (night, 2): where the zero-shot value is and is not wrong
- Diagnostics are `fbwm.py eval_ckpt=... diag_wm=1 [diag_wm_cf=1]` (outputs `diag_wm.csv`, `diag_q.csv`, `diag_cf.csv`), run on the 9 paper checkpoints (value check) and on seed 1 of each env (counterfactual check). None of them changes training.
- Value check (`diag_q.csv`, policy episodes, K = 100, discount 0.95): corr(F^T z, realised discounted return of the projected reward) is 0.85-0.95 on nearly every task, slope 0.90-1.16, including the fast ones (walker run 0.87, cheetah vel10 0.95, quadruped run 0.93). Exceptions: cheetah flip 0.42 (slope 0.24), cheetah vel2Back 0.66 and vel10Back 0.76. corr(projected return, true return) is 0.80-0.96 except cheetah flip 0.43. So on the states the policy visits F ranks states well and span(B) keeps most of the reward.
- Counterfactual check (`diag_cf.csv`, 80 real states per task, 4 first actions, then the policy): the rank correlation between Q(s, a) and the real return of the first action is about +0.12 at 3 steps, +0.10 at 10, about 0 at 30-60 (top-1 0.25-0.45 vs chance 0.25). Q varies by 3-7 across the 4 actions while the real 3-step return varies by 0.01-0.25. So F's dependence on the action is only weakly tied to what the action does. This is a hypothesis for the weak fast tasks, not proven: the policy keeps Q accurate along its own path, and its action choices are only weakly guided.
- Consistent with: MPPI helping on walker and cheetah (the world model supplies the action effect F lacks); no gain from wider networks; quadruped MPPI = policy. The other session's k-step FB objective (`fbwm_nstep.py`, `nsd_*` / `nsf_*` runs) targets exactly this, so no duplicate proposal from this branch. Use `diag_wm_cf=1` on its checkpoints (same architecture, `fbwm.py eval_ckpt=` loads them) to test whether the k-step objective raises the short-horizon rank correlation.
- Capacity sweeps (2 seeds, paper config, MPPI): walker phi256 630 / 575, mlp1024 586 / 619 (base 622); cheetah phi256 207 / 296, mlp1024 247 / 301 (base 403: worse); quadruped phi256 s1 471 (base 587). Larger networks are not the fix; on cheetah they hurt (consistent with z_dim above 50 collapsing there).
- Quadruped capacity sweeps, final (2 seeds, MPPI / policy; base 587 / 599): phi256 471 / 431 and 429 / 278 (worse), mlp1024 512 / 576 and 621 / 582 (no gain, wide spread). Wider networks do not help on any env.

## 2026-11-16 (night, 3): k-step FB objective, debugging session (`fbwm_nstep.py`)
- Idea (user): multi-step TD for FB, from the k-step Bellman equation of M = F^T B. Derivation and loss in `docs/method.md`; new script `fbwm_nstep.py` (fbwm.py untouched, `fb_nstep=1` reproduces it exactly: warmup runs matched the baseline to 3 digits until the switch).
- First runs (k=3, model rollout, B constant on the extra positives) collapsed: F_norm 70x, b_ortho_loss -49 -> +240, policy ~40. User's hypothesis (model error early in training, FB error back-propagated into the shared encoder): tested with `src=data` (no model) and a 50k warmup (good model): both collapse, so not the model.
- 12 short ablations (30k updates, gphi walker s1): the extra diagonal term is the trigger; Q-loss and `f_encoder_grad` only amplify. Using the EMA target B for the positives is worse (F can exploit any mismatch between the B of the positives and the B of the quadratic penalty). Giving B the gradient of the extra positives (`fb_nstep_bgrad=1`, dataset states) is stable for k = 2, 3, 5, with and without warmup. The same on imagined latents (model source) explodes (F_norm 1e7).
- Full 500k runs, walker s1, paper config, final row (policy / MPPI, 20 episodes per task): k=1 control 543 / 579; k=3 always on 321 / 376; k=2 always on 484 / 486; k=3 until 100k then 1-step 606 / 637. Always-on k-step learns faster early and settles lower (behaviour actions in steps 2..k bias the fixed point); switching back to the exact loss recovers it and, on this one seed, ends ~+60 above the control (within the noise of one seed: checkpoints of one run vary by +-40). until 50k (k=3, k=5): still running when this was written (k=3 643 / 647 at 300k, k=5 637 / 661 at 200k).
- Open: seeds (the user asked for one seed per experiment while debugging), other envs, a stable `model` source (on-policy, no bias), `horizon` > 5 for larger k.
- Runs: `logs/2026-11-16_02-43-17_FBWMN3d-gphi-walker` (k=3 always on), `02-49-12_FBWMN1d-gphi-walker` (control), `03-14-25_FBWMN3d-..` (until 100k), `03-14-28_FBWMN2d-..` (k=2), `03-39-08_FBWMN{3,5}d-..` (until 50k). The collapsed ones: `01-03-11_FBWMN3m`, `01-24-3x_FBWMN3d / FBWMN3mw`. Short ablations: queue entries `fbj_nsd_*`.

## 2026-11-16 (day): cheetah gap, and 10 A100 planner experiments while the laptop was off
- Per-task cheetah (MPPI, mean of seeds; gap to TD-MPC): vel10Back 192 (343), vel2Back 865 (971), vel2 583 (973), vel10 156 (295), flip 218 (839); mean 403 vs 684. flip is 124 of the 281-point gap, vel2 78. fbwm is below plain FB on flip (218 vs 312). TD3's policy alone gets 135 on flip, TD-MPC with planning 839, so flip needs planning.
- Dataset (cheetah, 10M states): angular momentum >= 5 in 30% of states (flip reward > 0 in 48%), so flip is not a coverage problem. Speed >= 10 in 0.000% and the 99th percentile is 3.6, so vel10 / vel10Back are capped by the data for every method (TD-MPC gets about 300).
- `fbwm.py diag_profile=1` (eval_ckpt): reward, speed, angular momentum and torso pitch per 200-step segment, policy and MPPI (`diag_profile.csv`). Job `fbj_diagprof_cheetah` (V100, 3 seeds). Not read yet.
- 10 eval-only experiments on A100 (jobs 262379-89, `fbj_ch_*`), cheetah paper checkpoints seeds 1-2, paper protocol (20 episodes, 10k labels, root check), one planner / task-inference setting changed each: horizon=10, num_samples=1024 num_elites=128, iterations=12, mixture_coef=0.25, mppi_init_std=0.5, mppi_norm=1, mppi_init_policy=1, mppi_terminal_only=1, z_ridge=0.1, temperature=0.1. Reference (same checkpoints, paper settings): MPPI 442 / 356 (seeds 1 / 2) from the main table. New flags (temperature, min_std, momentum, iterations, num_samples, num_elites) keep their defaults; `tools/paper_tables.py` excludes runs that use them.
- Read the results with `tools/sync_from_hpc.sh`, then compare each run's `zlabels` eval row (MPPI and policy columns, per task) against the references above; look at flip and vel2 first.

## 2026-11-16 (after the A100 runs): planner experiments are a null; what the cheetah agents do
- 10 eval-only planner / task-inference variants (cheetah paper checkpoints, seeds 1-2, MPPI mean; reference with paper settings 399, policy 337): horizon 10 387, iterations 12 399, mixture_coef 0.25 394, init_policy 394, init_std 0.5 406, norm 405, terminal_only 374, num_samples 1024 390, temperature 0.1 393, z_ridge 0.1 391. All within about +-15 of the reference except terminal_only (-25, vel2 477 vs 577: the per-step B reward matters on vel2). The policy column of the same checkpoints varies 332-364 across runs (label draws), so differences below that are noise. flip stays at 156-183 in every variant (reference 167).
- Time profile (`diag_profile`, cheetah, 3 seeds, 10 episodes per task, per 200-step segment): vel2 MPPI reward 0.88 / 0.73 / 0.53 / 0.45 / 0.38, speed 2.5 -> 0.9, |pitch| 0.36 -> 3.1 (policy: 0.88 -> 0.20, speed 2.5 -> 0.3, |pitch| 0.6 -> 4.4). vel10 the same (0.25 -> 0.11). The agent runs at the target speed, tumbles over and stays stuck on its back for the rest of the episode. Backward tasks never tumble (|pitch| about 0.2, vel2Back keeps 0.85-0.88). flip: MPPI does not flip (|pitch| 0.15, angular momentum about 0, reward 0.2); the policy rotates slowly (|pitch| 0.18 -> 1.08).
- Reading: the forward-task loss is a failure to see a long-term cost (the tumble is irreversible and costs the rest of the episode; discount 0.95 sees about 20 steps; the planning horizon is 5 + the terminal F). Discount 0.98 / 0.99 was already tried at z50 on seed 1 in the earlier config and hurt, so a longer planning horizon at eval time is the cheaper test: ch_hor20 / hor30 / hor50 and a profile at horizon 30 (jobs 264699-264702, A100 eval-only).

## 2026-11-16 (later): longer planning horizon on cheetah, null
- Eval-only, cheetah paper checkpoints seeds 1-2, paper protocol, MPPI mean (reference horizon 5: 399; policy 337): horizon 10 387, 20 389, 30 381, 50 380. Per task, reference vs horizon 50: vel2 577 -> 480, flip 167 -> 146, vel2Back 917 -> 928, vel10Back 210 -> 218, vel10 125 -> 126. Longer horizons are slightly worse on vel2 and flip, and slower (MPPI eval 184 s -> 398 / 552 / 855 s at horizon 20 / 30 / 50).
- Profile at horizon 30 (seeds 1-2, vs horizon 5 seeds 1-3): vel2 MPPI reward per segment 0.84 / 0.57 / 0.49 / 0.34 / 0.25 (horizon 5: 0.88 / 0.73 / 0.53 / 0.45 / 0.38), |pitch| 0.50 -> 4.59 (0.36 -> 3.14). The tumble is not prevented, and MPPI tumbles slightly more. flip: MPPI still does not flip (|pitch| 0.16, reward 0.13-0.23).
- So a longer plan with the same discount 0.95 and the same per-step B reward does not avoid the tumble. This does not refute "the cost is too far away": rewards at step k are weighted 0.95^k (0.21 at 30, 0.08 at 50), and the model error at 30+ steps is large. It does show that horizon alone is not the fix. A profile script bug found on the way: aggregate profiles per `horizon` value, not across all folders.

## 2026-11-21 (night) to 2026-11-22: goal reaching, FBWM vs reward-free WMs (`docs/goal_reaching.md`)
- Protocol `goal_k=K` (LeWM's): 50 pairs (s_t, s_{t+K}) of one trajectory from the -val file, simulator set to the stored qpos / qvel,
  budget 2K, OGBench radius. Shared sampler in `utils/ogbench_tasks.py`; same pairs for `wm_goal.py`, `fbwm.py fbgoal_k`, `wm_baselines.py`.
- New: `wm_baselines.py` (state-based LeWM and DINO-WM, their CEM; `planner=mppi` = our MPPI on their predictor), `fbwm.py
  fbgoal_k / fbgoal_official` (FB goal z = B(goal), actor and MPPI), `wm_goal.py diag_geom` (probe + distance rank correlations) and
  `planner=cem` (LeWM's CEM on JEPA / FBWM), `tools/goal_results.py` (tables from the run folders).
- Results (successes / 50, K = 10 / 50 / 200; details and run folders in docs/goal_reaching.md):
  - Same MPPI (5 steps, replan every step), four latents. antmaze-medium (3 seeds, / 150): FBWM 48 / 15 / 7, JEPA 19 / 6 / 3,
    LeWM 7 / 4 / 3 (/ 50), DINO-WM 10 / 4 / 2. antmaze-large: FBWM 13 / 4 / 0, JEPA 5 / 1 / 0, LeWM 4 / 2 / 1, DINO 5 / 1 / 2.
    cube: FBWM 21 / 11 / 16, JEPA 39 / 13 / 18, LeWM 43 / 11 / 13, DINO 30 / 12 / 10.
  - Geometry, Spearman(latent distance, real displacement): FBWM 0.92 / 0.91 / 0.56 (medium / large / cube), JEPA 0.52 / 0.55 / 0.51.
  - LeWM's CEM on all four latents (medium, cube, K = 10 / 50): all within noise (8-14 at K=10); replanning every step does not help CEM.
  - FB's value (z = B(goal)): actor medium 42 / 30 / 19 (seed 3: 41 / 25 / 14), large 40 / 19 / 13, cube 7 / 3 / 6; official tasks
    5, 2, 0, 0 / 50. Cube FBWM checkpoint scores 0 on the official goals in its own training eval too.
- Reading: on the two mazes the FB-shaped latent is the best distance cost under short-horizon MPPI (2-3x every other latent, raw state
  included), explained by its geometry. Not on cube, and not under LeWM's 25-step CEM. The FB actor is not compared with distance
  planners (user: it uses a learned value); it needs a goal-conditioned value baseline (GCIQL / HIQL), not done.
- Bugs on the way: zsh does not split `$VAR` in `for`/`set --` (two queue batches ran with one K or an empty ckpt; requeued);
  `wm_baselines.py` receding assert blocked planner=mppi (fixed). JEPA seed 2 and FBWM seed 2 timed out on slow nodes (JEPA rerun done,
  FBWM seed 2 rerun done; FB-goal actor seed 2: 41 / 24 / 18, official 1 / 50).
- Cancelled at the user's request: FB heads on the frozen JEPA WM for goal reaching (job 555171), already in the paper (Table 1).

## 2026-11-22 (evening): bcons_coef=0 ablation queued (user away 2 h)
- Proposed A (seeds 4-5 for FBWM + FB), B (bcons off), C (JEPA WM pretrains s4-5), D (model-free best-of-K by F); a Sonnet reviewer found none run before and ranked B first. The user chose B only.
- B: fbwm.py paper config with bcons_coef=0, walker / cheetah / quadruped, seeds 1-3, jobs 581184-581193 (`fbj_bcons0_*`). Each job trains with eval_mode=policy, then runs the table's N=10k MPPI eval (20 episodes, root check, lstsq, dataset Cov, no mppi_norm) on its own checkpoint, found by the new `tools/find_fbwm.sh ENV SEED key=value` (exits 1 if none, so the eval never runs with an empty eval_ckpt). The eval is chained in the H100 job rather than sent to V100 afterwards because nobody was there to queue it.
- Compare with Table 1 FBWM (622 / 441 / 587) per seed, from the `zlabels` rows of the eval runs.
- **Results (bcons_coef=0, N=10k MPPI eval, 20 episodes; MPPI / policy, seeds 1/2/3; paper config bcons 2 in brackets, same seed numbers):**
  walker 623 / 645 / 573 = 614 (577 / 672 / 615 = 621), policy 593 (574); cheetah 376 / 386 / 215 = 326 (442 / 356 / 410 = 403), policy 338 (355);
  quadruped 627 / 671 / 628 = 642 (676 / 468 / 618 = 587), policy 631 (599). Differences by seed (MPPI): walker +46 / -27 / -42, cheetah -66 / +30 / -195, quadruped -49 / +203 / +10.
- Reading: bcons is not needed on walker and quadruped (within noise; quadruped even tighter: 627-671 vs 468-676). On cheetah, MPPI is 77 lower on average,
  driven by seed 3 (215). Per task, cheetah without bcons loses the backward tasks (vel2Back 602 / 689 / 345 vs 908 / 926 / 759 by seed: lower on all 3 seeds, -237 to -414) and gains on flip (348 / 324 / 402 vs 309 / 25 / 320).
  No collapse in cheetah s3 training (latent_std 0.90, F_norm ~120 flat). n=3, so the cheetah drop is suggestive, not established.
- Runs: training `hpc/logs/2026-11-22_19-04-1*_FBWM-gphi-*`, evals `hpc/logs/2026-11-22_19-4[12]-*_FBWM-gphi-*` (each eval checked to load its own bcons0 checkpoint).
  Checked (2026-11-22): training and eval configs differ only in bcons_coef (new fields are at defaults that reproduce the old code: bc_coef 0, cons_target ema, frozen_wm off); model_loss = consistency only in the bcons0 runs.
  References: `hpc/logs/2026-11-15_19-4[3-6]-*`, `19-52-04`, `19-55-02` (eval runs of the Table 1 checkpoints).

## 2026-11-22 (night): core experiments for "FB shapes the latent of the world model better" (user-validated, no reviewer)
- bcons stays in the method; the bcons0 ablation is parked for the end of the project (user).
- Found: Table 1's "FB on frozen WM" row (and the adapted-WM row) use the mismatched DMC JEPA WMs (`tdmpc_adapt.py mode=pretrain` defaults: TD-MPC MLP 256, lr 1e-3, batch 512, consistency 2); the cheetah frozen-WM runs also used tau 0.001 / discount 0.98 (FBWM: 0.01 / 0.95). Matched JEPA WMs existed only on antmaze / cube.
- **E1** (matched frozen-WM row, VICReg only, user): `tdmpc_adapt.py mode=pretrain encoder=phimap enc_dim=64 lr=1e-4 batch_size=1024 update_freq=1 consistency_coef=5`, walker / cheetah / quadruped s1-3 (`fbj_e1_jepa_*`), then FB on the frozen WM in the paper config + N=10k MPPI eval (`fbj_e1_frozen_*`, held with `queue.py --after` until its WM is done).
- **E2** (collapse panel): matched JEPA WM with no regulariser (var=cov=0), EMA and online target; FBWM with EMA and online consistency target; walker + antmaze_medium, s1-3 (`fbj_e2_*`, 24 jobs). `fbwm.py` now logs `latent_rank` (RankMe on the batch, same formula as tdmpc_adapt.py) so every curve uses the same measure.
- **E3** (latent diagnostics, eval only): `wm_goal.py diag_geom=1` now runs on dm_control: eff_rank / n_dims_99 / top1_var, ridge probe to the full state, and corr of each task reward with its ridge fit on phi(s) (and on the raw state, for reference); train on trajectories < 9000, test on 9000+. On the 9 FBWM paper checkpoints and the 9 old JEPA WMs (`fbj_e3_geom_*`), and on the new matched WMs (`fbj_e3_geom_jepa64_*`, after E1).
- Tooling: `tools/find_run.py` (exact checkpoint lookup for chained jobs, exits 1 if none), `queue.py add --after`.
