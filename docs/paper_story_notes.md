# Paper story notes (user's outlines and decisions, 2026-11-18)

Working notes between the user and Claude. Not paper text. The prose built from them is in `docs/paper_intro.md`.

## Framing (user, 2026-11-18)
- The story goes from the world model to FB: we start from a reward-free latent world model and give it the ability to plan for any reward, zero-shot. We do not present it as improving FB with a world model.
- The audience already works with world models. "Why not use FB alone?" is not an intro question (it is an experiment: FB actor, model-free best-of-K by F). Same logic as not asking "why TD-MPC when TD3 exists".
- The motivating scenario for per-task cost is deployment: an agent receives a task it has never seen.
- Do not quote the 20 s adaptation time in the intro (the user expects to be attacked on it). The experiments section must still report the adaptation cost honestly.
- The imagined-latent evaluation problem is not intro material (the user judges it easy). The intro says how the latent is trained instead: world-model objective and FB objective on the same encoder, FB on the latent and not on raw states. L_B belongs in the method.

## Intro outline (user's bullets)
1. What a reward-free world model is.
2. A world model is useful for planning a task.
3. How planning works: roll out many trajectories (actions) in the world model, then select the best rollout.
4. Selecting needs a cost or score to rank rollouts.
5. Current reward-free planners use the distance to a goal in the same latent space: best rollout = closest to the latent goal.
6. What if the task is a reward function that is not goal reaching?
7. Then we need a reward model (evaluates the latents visited by the rollout) and a value function (estimates the cumulative reward from the last latent of the rollout), both specific to the task.
8. Problem: both need training.
9. A new task needs a new reward model and value function, so as many of them as tasks. Scenario: an agent is deployed and gets a never-seen task.
10. Zero-shot RL: pre-train on reward-free data with no task in mind, get a reward function at test time, optimise it with no further training. FB is the notable method (no FB details in the intro, they go in the background).
11. Goal of this work: give reward-free world models the ability to plan for non-goal-reaching tasks, zero-shot, with no reward model and value function per task.
12. One clear problem: a latent learned by a world model is not guaranteed to be rich enough to train a good FB.
13. Solution: train a world model whose latent is rich enough both for next-state prediction and for FB trained on it instead of on raw states. The same latent then allows planning through the world model and zero-shot task inference through FB.
14. Contributions: written after the experiments section.

## Why it is hard (Claude's list, for the method section; the intro uses only items 12-13 above)
- B and F are read on imagined latents, while FB is trained on real encodings. L_B (backward embedding of an imagined latent matches the real one) bounds the reward error for all rewards in the span, on the dataset's actions.
- The reward is read as B^T w, with ||w|| potentially large (Sigma^-1 amplifies); small latent errors can become large reward errors.
- One model must be right for every reward in the span at once, not for one task's reward.
- The planner searches over actions and finds model and F errors. Measured: F tracks the real return along the policy's path (corr 0.85-0.95) but its dependence on the action is weakly tied to the real outcome (rank corr about 0.1 at 3 steps). Hence the value check.
- Per-step reward and terminal value must describe the same reward (B^T w, not B^T z, when Sigma != I). lstsq gain: walker +52, cheetah 0.
- The latent has two jobs. Quadruped latent is lossy even on real states (probe error 0.22 vs 0.005 on walker).
- Not tested: a frozen reward-free world model with FB on top. Do not claim it fails.

## Experiments the story needs (status 2026-11-18)
- Q1, vs per-task adaptation at equal labels, label sweep: baselines exist (walker, cheetah; quadruped sweep running).
- Q2, is the model what helps: FB actor exists, model-free best-of-K by F not run.
- Q3, is the grounding needed: L_B off and separate latents, not run.
- Final numbers under one configuration, more seeds.
