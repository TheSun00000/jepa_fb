# TD-JEPA's FB baseline and evaluation (read 2026-11-18)

Sources: repo `facebookresearch/td_jepa` (archive of branch main, read only, not run) and the paper arXiv 2510.00739 (Bagatella et al., 2025).
Relevant files: `scripts/train/proprio/launch_fb_dmc.py`, `metamotivo/agents/fb/{agent,model}.py`, `metamotivo/evaluations/dmc.py`,
`metamotivo/data_loading/{base,dmc}.py`, `train.py`. Paper: Tables 1, 4, 5 and appendix E.

## Reported FB numbers (DMC, proprioception, average of the 4 tasks, 5 seeds, mean +- standard error, best of a 6-config sweep per domain)
FB (no state encoder) cheetah 662.8 +- 5.7, FB with the shallow encoder 672.7 +- 4.9; walker 788.7 / 811.5; quadruped 574.1 / 595.6.
The original FB paper (Touati et al. 2023, Table 2, RND) averages 513 on cheetah, 654 on walker, 606 on quadruped.

## How their FB differs from the original FB code (controllable_agent) and from our fb_jax.py
| | TD-JEPA FB | original FB / fb_jax.py |
|---|---|---|
| target network coefficient | **0.001** for F and B (paper: EMA 0.001) | 0.01 |
| training steps | 3M in the repo default (`num_train_steps`), 2M in paper Table 4 | 1M |
| Q loss | off (`q_loss_coef` default 0.0, not set by the launch script) | paper listing: on, weight 1/d; code default off |
| orthonormality loss | -mean(diag) + 0.5 * mean(offdiag^2), weight lambda swept | -2 mean(diag) + mean(offdiag^2), weight 1 (i.e. theirs is half) |
| state encoder | "left encoder": Linear + L2-norm to 256 dims in front of F; the actor reads the encoder output too | none (our fb_jax had a 2-layer MLP in front of F by mistake until 2026-11-17) |
| B network | 2 hidden layers x 256, output 50 | 2 x 526 (code) / 3 x 256 (paper text) |
| F, actor | F: 1 hidden layer 1024 per embedding; actor: 1 hidden layer 1024, fixed std 0.2 | similar |
| learning rates | F 1e-4, actor 1e-4, **B swept {1e-4, 1e-5}** | 1e-4 |
| orthonormality weight | **swept** (launch script {0.1, 1, 10}, paper Table 5 {0.01, 0.1, 1}) | 1 |
| discount | 0.98 | 0.99 in the code, 0.98 in the paper |
| data | first 5000 episodes of the RND buffer (`files[:5000]`, unsorted glob order) | first 5000 (sorted) |
| transitions | (obs[:-1], action[1:], obs[1:]), all 1000 per episode | same pairing |
| batch, mix ratio, noise | 1024, 0.5, 0.2 clip 0.3 | same |
| weight init | orthogonal (`weight_init`) | orthogonal in the original code |

## Evaluation protocol (DMCRewardEvaluation)
* Every 250k training steps; tasks: cheetah walk (2 m/s), walk_backward, run (10 m/s), run_backward (same definitions as the original).
* **One z per task**, inferred once from `num_inference_samples = 10000` random transitions of the training buffer: reward from the *next* physics state (set_state, the stored action set as control, then mj_forward / sensor / subtree velocity updates, `task.get_reward`), z = project(sum_i r_i B(next_obs_i)) (sqrt(d) * normalize).
* **All episodes of a task use that same z** (10 episodes in the launch script, "20 in DMC" in the paper text); the original code and our fb_jax draw a new z per episode.
* Actor mean action at test time. No reward shift in DMC (the +1 shift is for OGBench).
* Reported: mean over 4 tasks, 5 seeds, best configuration of the sweep (2 values of the B learning rate x 3 values of lambda) per domain.

## What this suggests for reproducing cheetah
The settings most likely to matter are the target coefficient (0.001, a 10x smaller step of the target network), the longer training (2M-3M steps),
the state encoder in front of F, and the per-domain sweep over the B learning rate and lambda. Their cheetah average (663-673) is far above
the original paper's 513 and above our 100-200.
