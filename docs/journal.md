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
