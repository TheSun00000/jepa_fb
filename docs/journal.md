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
