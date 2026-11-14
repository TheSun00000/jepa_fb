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
