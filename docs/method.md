# Method variants

From 2026-11-13 both variants live in `fbwm.py` (`variant=sepB|gphi`); the sections below name the original scripts.

Written from the code on 2026-11-13. Notation: `d = 50`, `phi(s)` the encoder (latent `h`), `B(s)` the FB backward embedding,
`F(h, a, z)` the forward map (two heads), `pi(h, z)` the actor, `z` on the sphere of radius `sqrt(d)`. Zero-shot inference:
`z = sqrt(d) * normalize(mean_s R(s) B(s))` on 10k labeled dataset states.

## Shared pieces (all TD-MPC-FB scripts)
- Latent dynamics `h' = normalize(h + MLP([h, a]))` (residual, on the sphere), trained by the JEPA consistency loss over H = 5 steps
  with the real actions against the EMA encoder (`rho^t`, rho = 0.5). The dynamics are trained ONLY by this loss: `consistency_coef = 0`
  means an untrained model.
- FB loss (as in `fb_jax.py`) on the first transition of each sequence: off-diagonal TD on `M = F^T B`, diagonal term, auxiliary Q loss
  (coef 0.01), orthonormality on B (coef 1). gamma 0.95, lr 1e-4, batch 1024, EMA tau 0.01 every update.
- Actor: maximises `min_k F_k(h, pi(h, z) + noise, z)^T z` on real latents.
- MPPI: 6 iterations, 512 samples + 25 policy trajectories, 64 elites, temperature 0.5. Value of a plan
  `sum_t gamma^t r(h_{t+1}) / d + gamma^H min_k F_k(h_H, pi(h_H), z)^T z / d`, with the per-step reward `r` as below.

## sepB (`tdmpc_fb_2.py`)
- B is FB's own BackwardMap on the raw observation, trained by the FB loss. phi feeds the dynamics, F and pi.
- phi gets consistency + `f_encoder_grad` x the FB gradient through F's input (no effect in the 0-5 sweep).
- A B-head `B_hat(h)` regresses B(s) from real latents and from imagined latents (its own parameters only: no gradient into phi or the
  dynamics). MPPI reward: `B_hat(h_hat)^T z`.

## gphi (`tdmpc_fb_3.py`)
- B(s) = g(phi(s)), g = FB's BackwardMap on the latent. phi gets consistency, `b_encoder_grad` x the FB gradient through B, and
  `f_encoder_grad` x the FB gradient through F's input. Optional `bcons_coef`: B-space consistency, g(imagined h_t) vs EMA B(s_t),
  which trains g (and the model) on imagined latents. MPPI reward: `g(h_hat)^T z`.
- `phi_identity=True`: phi(s) = s (plain FB on raw states, the dynamics predict the next raw state).

## Framing under discussion (2026-11-13)
sepB is gphi with a stop-gradient between FB and the world model: B is a *target* for the model, as the reward is in TD-MPC
(reward head <-> B-head, Q head <-> F). A principled version would send the B-prediction loss at imagined latents into phi and the
dynamics (value-equivalence for every reward in FB's span). That is a method change: not implemented, needs the user's approval.

## k-step FB objective (`fbwm_nstep.py`, 2026-11-16, user's idea; experimental)
From the k-step Bellman equation of the successor measure, `M = sum_{j=1..k} gamma^{j-1} P_j + gamma^k E[M(s_k, pi_z(s_k))]`:
`L_k = E[(F(s0,a0,z)^T B(s+) - gamma^k Fbar(s_k, pi_z(s_k), z)^T Bbar(s+))^2] - 2 sum_{j=1..k} gamma^{j-1} E[F(s0,a0,z)^T B(s_j)]`
(k = 1 is the fbwm.py loss). In the batch: off-diagonal target `gamma^k Fbar(s_{i,k})^T Bbar(s'_j)`; the diagonal term gets
`-F_i^T sum_{j=2..k} gamma^{j-1} B(s_{i,j})`; the Q-loss becomes n-step. Flags: `fb_nstep=k`, `fb_nstep_src=data|model`, `fb_nstep_bgrad`,
`fb_nstep_warmup`, `fb_nstep_until`, and debug flags `fb_nstep_diag`, `fb_nstep_q`, `fb_nstep_tgtb`.
- **B must receive the gradient of the extra positives (`fb_nstep_bgrad=1`).** With B(s_j), j >= 2, as a constant the run blows up within
  ~1000 updates (F_norm x10-100, B loses orthonormality, policy collapses), also with the dataset states (no model error) and also when the
  switch to k-step happens after 50k updates with a good model. With the true gradient (B trained on every positive) it is stable.
- `src=model` (pi_z rollout through the world model, the on-policy version) is unstable with `bgrad` (g free on imagined latents, F_norm 1e7)
  and without it (collapse, as above). Not usable yet.
- `src=data` uses the dataset states s_2..s_k, i.e. behaviour actions for steps 2..k: biased fixed point. It speeds up early learning but
  settles lower (walker, k=3: ~400 vs ~600 for k=1); switching back to the exact 1-step loss (`fb_nstep_until`) removes the bias.
