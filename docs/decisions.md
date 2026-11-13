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
