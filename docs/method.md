# Method variants

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
