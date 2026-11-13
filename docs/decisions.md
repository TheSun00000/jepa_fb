# Decisions

Newest first. One entry per decision: what, why, what else was considered, evidence.

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
