# Research plan: zero-shot planning with FB + a latent world model

Living document. Edit as decisions are made; log every change in the **Changelog** at the bottom.
Status tags: `[todo]`, `[running]`, `[done]`, `[dropped]`. A result goes in here only with the run folders it came from.

---

## 1. The claim we want to defend

> FB learns, without rewards, a basis of rewards B and their successor features F. We learn a latent world model
> that is **value-equivalent for every reward in that basis** (it predicts B and F from imagined latents).
> At test time, for any task z inferred from a few labeled states, we **plan** in latent space with no training.
> This beats (i) the FB policy alone, (ii) task-specific model-based RL given the same labels and less compute,
> and (iii) other zero-shot RL methods.

Sub-claims, each needing its own evidence:
- **C1 – Planning helps FB.** MPPI on top of FB beats the FB actor, with significance, on most tasks and environments.
- **C2 – Why it helps.** The gain comes from the actor's optimization error (amortization gap) on some tasks, and from robustness when the actor collapses. We show this, not just claim it.
- **C3 – The model must be grounded in FB's reward basis.** Predicting B (and F) on imagined latents is what makes planning work. Raw-state models and ungrounded latents fail.
- **C4 – FB beats task-specific relearning at equal labels.** With N labeled states, inferring z beats fitting a new TD-MPC reward head (+Q) on the same N labels, at least up to a crossover N that we report.
- **C5 – Competitive with zero-shot RL baselines** on a standard benchmark protocol.

## 2. What we already have (single seeds unless stated; see `hpc/logs/`)

| Finding | Evidence | Strength |
|---|---|---|
| sepB MPPI > FB on cheetah (~440–520 vs ~300–370) | sepB runs 2026-09-30 → 10-02 (9 runs), FB 7 runs | good (many runs) |
| sepB MPPI stable on quadruped while the actor and FB collapse late | 3 sepB runs, 5 FB runs | medium |
| Planning < policy on walker, every variant | sepB, gphi, gphi-identity | medium, unexplained |
| Raw-state model (phi_identity) breaks MPPI (walker 80 vs 550) | 2026-10-02_12-19-48 | 1 seed |
| Training g on imagined latents (bcons) helps gphi MPPI | 2026-10-05_13-35 vs 15-17 (cheetah) | 1 seed |
| `f_encoder_grad` in sepB does not matter (0–5) | 2026-10-01_08-2x, 10-57 | 1 seed each |
| Trunk phi in front of plain FB does not hurt much | `fb_modified.py` runs 2026-10-02/03 | 1–2 seeds |
| TD-MPC oracle (true reward, full training) on cheetah vel2: ~920–950 | 2026-09-30 TDMPC-cheetah-vel2 | vel2 only |

Not run yet: antmaze (with any method), multi-seed anything, the grounded sepB variant (§4.2).

## 3. Protocol (fix this before the main runs)

- [todo] **Seeds:** 5 per configuration for the main tables (3 is the minimum for ablations).
- [done 2026-11-13, `fbwm.py`] **Eval:** more episodes per task (currently 5, one fresh z per episode). Use 10 at the end of training. Report the **final checkpoint**, never the best eval.
- [todo] **Aggregates:** per-task mean ± 95% CI over seeds; across tasks, IQM and mean of normalized scores (normalize by the TD-MPC oracle or the TD3 upper bound per task) with stratified bootstrap CIs (`rliable`-style).
- [todo] **Label budget:** z is inferred from N labeled states. Default 10k as now, and the label sweep in §5.
- [done 2026-11-13, b97c0f3] **Run metadata:** `info.txt` records seed, git commit (+dirty flag), Slurm job id, full command (copy the helpers from CLFB's `_utils/logs.py`).
- [done 2026-11-13, `fbwm.py`, sepB + gphi; grounded sepB not yet] **One script for the method family:** merge sepB / grounded-sepB / gphi into one file with flags, so ablations differ by flags only.
- [partly done 2026-11-13: `mppi_time` / `policy_time` per eval row in `fbwm.py`] **Compute accounting:** log wall-clock per eval step (MPPI vs policy) and per-task adaptation time for the baselines. Needed for C4.

## 4. The method: variants to settle first

### 4.1 Core runs (C1)  `[running 2026-11-13, 3 seeds first]`
FB (policy), sepB (MPPI and policy), gphi best (consistency 5, bcons 2) × cheetah, walker, quadruped × 5 seeds.
gphi config = consistency 5, bcons 2, f_encoder_grad 1 (2026-11-14; the first gphi runs used 0.1 by mistake, kept as an ablation).
Queued 2026-11-13 with seeds 1–3 (`scripts/queue/fbj_core_*.sh`, eval every 20k updates, final 10-episode eval); seeds 4–5 later.
**Gate:** MPPI significantly > FB on at least 2 of 3 envs. If not, stop and diagnose before anything else.

### 4.2 Grounded sepB (C3), needs user approval (method change)  `[todo]`
In `tdmpc_fb_2.py` the B-head loss trains only the head: phi and the dynamics are pure JEPA.
The principled version (TD-MPC style) sends the B-prediction loss at the imagined latents into the dynamics and phi,
and optionally trains F on imagined latents too. Compare: head-only (current) vs grounded B vs grounded B + F.

### 4.3 Walker diagnosis  `[todo]`
Before more tuning: per-task MPPI vs policy; B-head error vs horizon step on imagined latents; MPPI with the true
dynamics (env simulator) but the learned reward/value, which separates model error from reward/value error.

## 5. Comparison with task-specific model-based RL (C4), the main new experiment

**Question:** given N labeled states of a new task, is it better to infer z (zero training) or to fit a task-specific head?

Use **our** reward-free world model (same encoder and dynamics) for all arms, so that only the task-adaptation part differs.
**Decided 2026-11-14 (user):** the adapted baselines start from a clean, reward-free JEPA world model (consistency loss only, no FB terms), not from an fbwm checkpoint.

| Arm | Reward in the planner | Terminal value | Per-task training |
|---|---|---|---|
| **Ours** | B-head(ĥ)ᵀz, z from N labels | F(ĥ_H, π, z)ᵀz | none |
| TD-MPC-R | new reward head r(ĥ) fit on N labels | none (pure H-step MPC) | reward head only (seconds) |
| TD-MPC-RQ | new reward head on N labels | new Q head, TD-trained offline on the dataset with the learned reward | reward + Q (k updates, report k) |
| TD-MPC-RQ-FT | same as RQ, but also fine-tunes the encoder/dynamics | new Q | full fine-tune (k updates) |
| Oracle | true reward on all states, full TD-MPC training | Q | full (upper bound) |
| Ours + labels | ours, then the TD-MPC-RQ fine-tune starting from z | F | k updates (does FB give a better start?) |

- Sweep **N ∈ {100, 1k, 10k, 100k}** and, for the trained arms, **k ∈ {0, 1k, 10k, 100k}** updates.
- Main figure: return vs N (and vs adaptation compute), with the crossover point where TD-MPC-RQ catches up, if any.
- The fair point: TD-MPC-R with N labels is myopic (H=5, no value), so it should lose on long-horizon tasks. TD-MPC-RQ is the serious baseline, and it costs per-task training. Report both.
- Do not assume we win. With 100k labels and 100k updates TD-MPC-RQ may win. The claim is then about zero cost and few labels, which is fine if it is reported honestly.
- Watch out: label sampling. Use the same N states for all arms (same seed), drawn from the dataset distribution as in z inference.

## 6. Zero-shot RL baselines (C5)  `[todo]`

- FB policy (ours, same code). FB + **model-free test-time improvement**: sample K actions and take argmax Fᵀz (no dynamics). This isolates the value of the world model from just using F at test time; must be in the paper.
- FRE (functional reward encodings): our datasets and the antmaze task suite (goal, velocity, simplex, path) look like FRE's evaluation, so their protocol and possibly published numbers transfer. Check that the protocol really matches before quoting numbers.
- HILP and Laplacian / successor-feature baselines from the zero-shot RL literature: reuse public code where possible.
- Test-time z optimization / fast-adaptation methods for behavioral foundation models, as a test-time-compute baseline (they also spend compute at test time, like MPPI).
- Check the exact papers and protocols before writing them in; the list above is from memory.

## 7. Ablations and analysis (C2, C3)  `[todo]`

- Shared vs separate encoder: gphi vs sepB vs grounded sepB (§4.2).
- Grounding: B-head on real only vs real + imagined; with/without gradient into phi; F on imagined latents.
- Model space: raw state (phi_identity) vs latent.
- Planning compute: horizon H ∈ {1, 3, 5, 10}, samples, iterations, with vs without policy prior trajectories, terminal value on/off.
- Consistency coefficient (keep in mind coef 0 = untrained dynamics: a sanity check, not an ablation).
- **Amortization gap:** per state, Fᵀz(π(s)) vs max over MPPI's first action. Correlate the per-task gap with the MPPI gain (C2).
- Reward reconstruction (corr(Bᵀz, R)) vs return per task: when B cannot represent a task, planning cannot fix it.

## 8. Scope and stretch

- Environments: cheetah, walker, quadruped, **antmaze** (none run yet), 3 seeds minimum before the main table.
- Stretch: pixel observations (where a latent world model matters most); other exploration datasets (data-quality sensitivity); dataset size.

### Later idea (user, 2026-11-14): FB as a layer on a pretrained, frozen world model  `[later]`
Once the main results hold: start from a world model pretrained without rewards and frozen, and add only the FB part
(B, F, the actor) on top of it to make it zero-shot. The pitch becomes "any pretrained latent world model gains zero-shot
task inference for the cost of an FB head". Not now.


### Status after the night of 2026-11-15 (walker, see journal and decisions)
- **Planner fix (reviewer-approved):** `mppi_reward=lstsq`. MPPI's per-step reward uses the L2 projection of R onto span(B); the policy and terminal value keep z = E[BR].
- **z_dim = 200** (corr of R with span(B) 0.78 → 0.92).
- With both, fbwm MPPI = **666** over 3 seeds (control without M2: 651), against FB 567 (FB doesn't gain from z_dim: 573–580), offline TD3 685 and TD-MPC 792.
- M1/M2 (F-consistency on imagined latents) are not needed once lstsq is used. The policy weakens at z_dim 200 (about 500) and collapses at 400.
- C4 label sweep (z200 + lstsq, MPPI): N = 100 / 300 / 1k / 3k / 10k gives 348 / 484 / 582 / 648 / 671.
  - It beats FB from N = 300 on, and beats the labels-only adapted baselines at every N.
  - It matches the full-dataset adapted baseline from N = 3k on. At N ≤ 1k that baseline wins.
- Open: small N (ridge sweep running), the actor at large z_dim, other envs (cheetah, quadruped), and a reviewer pass on the whole C1/C4 story before writing.


### Paper status and handoff (2026-11-15 night): read this first to write the draft
- **Configuration of "ours" (fbwm), the same for every env:** `fbwm.py variant=gphi` with z_dim 50, consistency_coef 2, bcons_coef 2, f_encoder_grad 1, `mppi_reward=lstsq z_cov=dataset mppi_root_check=1`, 500k updates.
  - lstsq: MPPI's per-step reward is the L2 projection of R onto span(B).
  - root check: execute the policy's action when F at the real latent rates it higher than MPPI's first action.
  - Protocol: z from N = 10k labelled states, final eval with 20 episodes per task, 3 seeds.
- **Main table and label-budget figure:** the tracker overview (https://claude.ai/artifact/3UineFBuRZiwE4gzMLRkhn), collection `paper` (`main_table`, `label_curve`), built by `python tools/paper_tables.py` from `hpc/logs` (writes `hpc/status/paper_*.json`).
  - Rows: TD-MPC and offline TD3 (per-task toplines), the adaptation baselines (reward-free JEPA world model + TD-MPC heads: full dataset, and labels-only Dyna), FB, fbwm policy, fbwm MPPI.
  - Envs: walker, cheetah, quadruped (antmaze and other zero-shot methods out of scope for now).
  - Status: baselines 3 seeds everywhere. fbwm paper config: quadruped 3 seeds (MPPI 660 / policy 637); cheetah s1 (430 / 329), s2–3 running; walker s1–3 running. Label sweeps of these checkpoints are queued or running on V100.
- **Story so far, with the evidence in journal and decisions:**
  - fbwm > FB on every env.
  - MPPI > its own policy with the root check: +48 on walker and cheetah at consistency 5. With consistency 2: all 3 quadruped seeds, +23 on average.
  - Without dataset access at test time, zero-shot beats the adapted baselines for N ≤ 3k on walker, and at every N on cheetah and quadruped. With dataset access, the full-dataset adapted baseline wins on walker only.
  - Still below the per-task toplines.
- **Ablations for the paper** (data exists):
  - lstsq vs mean per-step reward: walker +52 (3 seeds), cheetah 0.
  - Root check vs no root check, and the other planner variants (b) / (c): 7 checkpoints.
  - z_dim 50 / 100 / 200: walker z200 651 / 523, cheetah collapses above 50.
  - M1/M2 F-consistency: no gain with lstsq.
  - MPPI horizon, terminal-only, softmax normalisation: no effect.
  - Cov(B) from labels vs dataset: equal at z50.
  - Negative results: BTD, mix_ratio, hidden_dim, discount, TD hyperparameters at z200.

## 9. Order of work

1. §3 protocol and script merge.
2. §4.1 core multi-seed runs. **Gate.**
3. §4.3 walker diagnosis, then §4.2 grounded variant (after approval).
4. §5 TD-MPC comparison (label × compute sweeps).
5. §6 baselines, §7 ablations, antmaze.
6. Write-up.

## 10. Open questions

- Which variant is "the method": sepB as is, or grounded sepB? Decide after §4.2.
- Normalization for aggregate scores: TD-MPC oracle per task (needs oracle runs on every task) or TD3 upper bounds?
- Compute budget for MPPI at eval (it is ~100× a policy step): report it, and compare against test-time-compute baselines.

---

## Changelog
- 2026-11-13: first version (written with Claude from the JZ runs synced into `hpc/`).
- 2026-11-13: protocol items done: random seeds + run metadata in info.txt; `fbwm.py` (sepB + gphi, final 10-episode eval). Not yet tested on Jean Zay.
- 2026-11-13: smoke test of `fbwm.py` passed on JZ (job 153137). `fb_jax.py` gets the final 10-episode eval; `fbwm.py` logs eval time per mode. §4.1 core runs queued with 3 seeds (27 jobs).
- 2026-11-14: gphi core config corrected to f_encoder_grad=1 (gphi seed 2 with 0.1 cancelled, gphiF1 × 3 seeds queued).
- 2026-11-14: focus on walker, 1 seed, 500k updates. Baselines: TD-MPC per task (MPPI topline), offline TD3 per task (policy topline, to write), zero-shot FB (policy floor). TD-MPC walker × 5 tasks queued.
- 2026-11-14: C4 baseline = reward-free JEPA world model + per-task reward/Q heads (not an fbwm checkpoint). Added the later idea of FB on top of a frozen pretrained world model.
- 2026-11-15: night results. lstsq planner reward + z_dim 200 give fbwm MPPI 666 on walker (3 seeds, FB 567); M1/M2 not needed; C4 label sweep updated.
- 2026-11-15 (night): paper configuration fixed (gphi z50, consistency 2, lstsq, dataset Cov, root check); main table and label-budget figure in the tracker; handoff section for the paper draft.
