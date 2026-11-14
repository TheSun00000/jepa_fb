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
