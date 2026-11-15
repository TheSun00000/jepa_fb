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
