# Introduction (draft v1, 2026-11-18)

<!--
Written from the user's bullet outline (docs/paper_story_notes.md). Replaces section 1 of docs/paper_wm_first.md.
Contributions are not written: the user adds them after the experiments section.
TODO: references to verify (DINO-WM, PLDM, V-JEPA 2-AC, TD-MPC, FB, HILP, FRE). Run docs/writing_checklist.md section 0 before this leaves the repo.
-->

A world model learns how the state of an environment changes under an agent's actions. A reward-free world model is trained on transitions collected without any task in mind, for example by an exploration policy, so it holds knowledge of the dynamics and nothing about what the agent should achieve. Many recent models of this kind predict in a learned latent space, with a joint-embedding objective, rather than in pixel or observation space [DINO-WM; PLDM; V-JEPA 2-AC].

Such a model is useful for planning. At each step the agent imagines many action sequences by rolling them out through the world model, and executes the first action of the best one. This requires a way to rank the rollouts, that is, a cost or a score for a trajectory of latent states.

Current reward-free planners take that score from the latent space itself: the cost of a rollout is the distance between its final latent and the latent of a goal observation. The best rollout is the one that ends closest to the goal. This needs no training, but it restricts the tasks to goal reaching.

Now suppose the task is an arbitrary reward function, such as running at a given speed or keeping a posture, and not a goal state. The planner then needs two components that are specific to the task. A reward model evaluates the latent states visited by a rollout, and a value function estimates the cumulative reward that would be collected after the last latent state of the rollout, since a rollout covers only a short horizon. Both are trained for the task, from reward labels, with the world model kept fixed or fine-tuned.

This is the obstacle to deploying a reward-free world model on many tasks. Every new task needs its own reward model and value function, and so its own training run. Consider an agent that has been deployed and is handed a task it has never seen: it cannot act until it has trained these two components. With tasks that arrive one after another, the number of reward models and value functions grows with the number of tasks.

A separate line of work, zero-shot reinforcement learning, addresses the same difficulty without a world model. The agent is pre-trained on a reward-free dataset with no task in mind. At test time it receives a reward function, queried on a set of labelled states, and must solve it with no further training. Forward-backward (FB) representations are the leading method of this kind [Touati & Ollivier, 2021; Touati et al., 2023], next to other approaches such as HILP and functional reward encodings [TODO: verify]. These methods act through a policy conditioned on a task embedding, and they do not plan.

The goal of this work is to give reward-free world models the ability to plan for tasks that are not goal reaching, in a zero-shot manner: no reward model and no value function is trained for a given task.

The difficulty is the latent space. A latent learned by a world model is trained to be predictable: the encoder is asked for latents whose next value the dynamics can forecast. Nothing in that objective guarantees that the latent keeps what FB needs, which is enough information to describe how a state leads to the states that follow under many different policies. A latent that predicts well can still be too poor to train a good FB on.

We train the two together. The encoder receives gradients from the world-model objective and from the FB objective, and FB is trained on this latent instead of on raw states. The latent is then predictable enough for the dynamics and expressive enough for the backward embedding $B$ and the forward map $F$. The same latent serves two purposes. Planning rolls the dynamics forward in it, and zero-shot task inference computes a task vector from labelled states with $B$ and reads values from $F$. Given a new reward, the agent infers the task and plans, and no new reward model or value function is trained.

*(Contributions: to be written once the experiments section is done.)*
