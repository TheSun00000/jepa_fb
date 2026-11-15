# Zero-shot planning with forward-backward representations and a latent world model

<!-- First draft, 2026-11-15, revised after a first-reader pass. Numbers: hpc/status/paper_main_table.json, hpc/status/paper_label_curve.json (tools/paper_tables.py); run folders in docs/decisions.md and docs/journal.md. Items marked TODO need data or decisions. -->

---

## Abstract

Zero-shot reinforcement learning methods learn, from reward-free data, an agent that can solve any task specified at test time, without further training. Forward-backward (FB) representations do this by learning a basis of reward functions together with their successor features and a single task-conditioned policy. However, a single network must approximate the optimal policy for every possible task, and FB has no built-in way to improve it at test time. We observe that FB already provides what a planner needs for any task: a reward function and a value function. What it lacks is a model of the dynamics. We learn a latent world model jointly with FB, in a shared latent space, and ground the model in FB's reward basis: the backward embedding must be predictable from imagined latents, which makes the model accurate for every reward FB can represent at once. At test time, a task inferred from a set of labelled states is solved by model-predictive control (MPPI) in latent space, with no gradient step. On three DeepMind Control domains (14 tasks), planning improves over the FB policy on walker and cheetah and matches it on quadruped. On walker and cheetah, we vary the number of labelled states from 100 to 10,000. Given only the labelled states, our method beats a task-specific model-based baseline that adapts the same kind of world model to the labels, at every label budget. A baseline that may also train on the whole reward-free dataset is better on walker at every budget, and worse on cheetah at every budget.

---

## 1. Introduction

**Zero-shot RL.** In many settings, an agent can collect large amounts of reward-free interaction data, but the task it will be asked to solve is known only later. Re-running reinforcement learning for every new task is expensive. *Zero-shot RL* asks for more: pre-train once without rewards, then, given a description of a new task (here, a set of states labelled with their reward), immediately produce a good policy, with no further learning.

**Forward-backward representations.** FB [Touati & Ollivier, 2021] is a principled answer to that question. It learns two maps: a backward embedding $B(s) \in \mathbb{R}^d$, which acts as a basis of reward functions, and a forward map $F(s, a, z)$, which gives the successor features of the policy $\pi_z$ in that basis. For a new reward $r$, the task vector $z_r = \mathbb{E}_{s \sim \rho}[B(s)\, r(s)]$ is one average over labelled states, and $F(s,a,z_r)^\top z_r$ is the Q-function of $\pi_{z_r}$ for $r$. FB is among the strongest zero-shot methods on standard benchmarks [Touati et al., 2023].

**The limit of a single amortized policy.** All of FB's behaviour at test time comes from one actor network $\pi(s, z)$, trained to maximise $F^\top z$ for a whole distribution of task vectors. That is a hard amortized optimization problem. For any particular task, the actor is only approximately greedy with respect to its own value function, and it is evaluated on task vectors that may be rare during training. FB itself has no mechanism to turn extra test-time computation into better actions. Test-time fine-tuning of $z$ or of the policy can, but it needs gradient steps for every new task.

**Planning needs a reward, a value and a model, and FB already has two of them.** Model-based RL methods such as TD-MPC [Hansen et al., 2022] act by planning (model-predictive control): at each step they optimise a short sequence of actions through a learned latent model, scoring it with a learned reward and a learned terminal value. This is a strong way to use test-time compute. However, the reward and value heads are learned for one task, so a new task means new training. FB provides, for *any* task vector $z$, both pieces in closed form: a per-step reward that is linear in $B$, and a terminal value $F^\top z$. The missing piece is a dynamics model, and that model is task-independent, so it can be learned from reward-free data too.

**This work.** We learn a latent world model jointly with FB, in one latent space shared by the encoder, $B$, $F$ and the actor. The world model is trained by latent self-prediction: imagined latents must match the encodings of the real future states, as in TD-MPC and joint-embedding predictive architectures (JEPA) [LeCun, 2022]. It is also *grounded in FB's reward basis*: the backward embedding of imagined latents must match the backward embedding of the real future states. Since every reward FB can represent is a linear function of $B$, this one loss makes the model accurate for all of those rewards at once. At test time, we infer $z$ from labelled states, as FB does, and plan in the latent space with MPPI, a sampling-based trajectory optimizer [Williams et al., 2017]. No parameter is updated.

**Contributions.**
1. A method, *fbwm*, that turns a forward-backward representation into a zero-shot planner by learning a latent world model grounded in FB's reward basis (§3).
2. Two test-time choices that make the planner consistent with FB's value function: per-step rewards from the projection of the task reward onto the span of $B$, and a check at the real state that keeps the planner from overriding the policy when FB's value function disagrees (§3.4).
3. Experiments on 14 tasks in three DeepMind Control domains, comparing fbwm with FB, per-task toplines trained on the true reward, and adaptation baselines that fine-tune a reward-free world model to the same labels (§4). On walker and cheetah, the comparison is repeated at label budgets from 100 to 10,000 labelled states.

---

## 2. Background

**Setting.** We consider a reward-free MDP with states $s$, actions $a$, transitions $P(s' \mid s, a)$ and discount $\gamma$, and a dataset $\mathcal{D}$ of transitions $(s, a, s')$ collected by some exploration policy, with state distribution $\rho$. At test time, a task is a reward function $r: \mathcal{S} \to \mathbb{R}$, known only through $N$ labelled states $\{(s_i, r(s_i))\}_{i=1}^N$ with $s_i \sim \rho$.

**Successor measures and FB.** The successor measure of a policy $\pi$ counts the discounted visits to a set of states $X$:

$$M^\pi(s_0, a_0, X) = \sum_{t \ge 0} \gamma^t \, \Pr(s_{t+1} \in X \mid s_0, a_0, \pi).$$

The Q-function of any reward is linear in $M^\pi$: $Q^\pi_r(s,a) = \int M^\pi(s, a, ds')\, r(s')$. FB learns a family of policies $\pi_z$ indexed by a task vector $z \in \mathbb{R}^d$, where $d$ is the dimension of the representation, and a rank-$d$ model of their successor measures:

$$M^{\pi_z}(s, a, ds') \approx F(s, a, z)^\top B(s')\, \rho(ds').$$

Plugging this into the expression for $Q$ gives, for every reward $r$:

$$Q^{\pi_z}_r(s,a) \approx F(s,a,z)^\top \mathbb{E}_{s' \sim \rho}\big[B(s')\, r(s')\big] = F(s,a,z)^\top z_r .$$

The policies are trained to be greedy for their own task vector, $\pi_z(s) \approx \arg\max_a F(s,a,z)^\top z$. If training were perfect, $\pi_{z_r}$ would be optimal for $r$, for every reward $r$ in the span of $B$.

**Training FB.** $F$ and $B$ are trained by temporal-difference learning on the successor measure, with $z$ sampled during training from a fixed distribution: uniform on a sphere of radius $\sqrt d$, mixed with task vectors $B(s)$ of single dataset states:

$$\mathcal{L}_{\mathrm{FB}} = \mathbb{E}\Big[\big(F(s,a,z)^\top B(s^+) - \gamma\, \bar F(s', \pi_z(s'), z)^\top \bar B(s^+)\big)^2\Big] - 2\, \mathbb{E}\big[F(s,a,z)^\top B(s')\big],$$

where $(s, a, s') \sim \mathcal{D}$, $s^+ \sim \rho$ is an independent state and bars denote target networks (slowly updated copies, $\bar\theta \leftarrow (1-\tau)\bar\theta + \tau\theta$). An orthonormality penalty keeps $\mathbb{E}_\rho[BB^\top]$ close to the identity. The actor is trained to maximise $F(s, \pi(s,z), z)^\top z$.

**Model-predictive control in a latent space.** TD-MPC encodes states into a latent $h = \phi(s)$ and learns a dynamics model $h_{t+1} = f(h_t, a_t)$, a reward head and a Q-function. At each environment step, it searches for the action sequence that maximises the predicted $H$-step return plus a terminal value. The search uses MPPI (model-predictive path integral control), which samples action sequences from a Gaussian, scores them through the model, and refits the Gaussian to the best samples, for a few iterations. Then it executes the first action and re-plans at the next step. The model is trained by *latent consistency*: imagined latents must match the target encodings of the real future states. This self-predictive objective needs no reconstruction of observations.

---

## 3. Method

### 3.1 Overview

fbwm has five learned components, all operating on one latent space:

| component | role |
|---|---|
| encoder $h = \phi(s)$ | shared latent state |
| backward head $B(s) = g(\phi(s))$ | basis of rewards |
| forward map $F(h, a, z)$ | successor features, hence values $F^\top z$ for every task |
| actor $\pi(h, z)$ | FB's task-conditioned policy, used as a proposal and fallback for planning |
| dynamics $h_{t+1} = f(h_t, a_t)$ | task-independent latent world model |

From here on, $F$ and $\pi$ take the latent $h$ instead of the state. Everything is trained once, from reward-free data. At test time, a task is solved by planning through $f$, with rewards from $g$ and terminal values from $F$.

### 3.2 A shared latent space

FB and the world model share the encoder. $B$ is computed from the latent, $B(s) = g(\phi(s))$, and $F$ and $\pi$ take the latent as input. The FB loss trains $\phi$ through both $F$ and $B$, and the consistency loss below trains it through the dynamics. The shared latent matters for planning. The planner scores imagined latents $\hat h$ with $g(\hat h)$ and $F(\hat h, \cdot, z)$, so these heads must be meaningful on the latents the model produces. When the latent is shared and the model is self-predictive, imagined latents land in the same space where $g$ and $F$ were trained.

### 3.3 Grounding the world model in FB's reward basis

The dynamics are trained on sub-trajectories $(s_0, a_0, \dots, a_{H-1}, s_H)$ from the dataset. Starting from $\hat h_0 = \phi(s_0)$, we roll the model forward with the dataset's actions, $\hat h_{t+1} = f(\hat h_t, a_t)$, and ask for two things.

*Latent consistency* (as in TD-MPC): imagined latents must match the target encodings of the real states,

$$\mathcal{L}_{\mathrm{cons}} = \sum_{t=1}^{H} \lambda^{t} \, \big\| \hat h_t - \bar\phi(s_t) \big\|^2 ,$$

where $\bar\phi$ is the target encoder and $\lambda \in (0,1)$ down-weights distant steps.

*Reward-basis consistency*: the backward embedding of an imagined latent must match the target backward embedding $\bar B = \bar g \circ \bar\phi$ of the real state,

$$\mathcal{L}_{B} = \sum_{t=1}^{H} \lambda^{t} \, \big\| g(\hat h_t) - \bar B(s_t) \big\|^2 .$$

Why the second loss is the right one: in FB, every task reward the agent can represent is, up to projection, a linear function of $B$. If $g(\hat h_t) \approx B(s_t)$, then $g(\hat h_t)^\top w \approx B(s_t)^\top w$ for *every* weight vector $w$ at once, with an error at most $\|w\|\,\|g(\hat h_t) - B(s_t)\|$. A single loss thus controls the model's reward error, along imagined trajectories of $H$ steps, for all the rewards that FB can express, instead of one reward at a time. (Like any model trained offline, it is accurate on the actions of the dataset; the planner may query others.) This is the multi-task version of TD-MPC's reward-prediction loss, which grounds the model in a single reward. It is also a form of value equivalence [Grimm et al., 2020]: we do not ask the model to reconstruct the state, only to preserve what matters for the rewards that will be asked of it. Both losses also train $g$ on imagined latents, the inputs it will see during planning.

The full objective is

$$\mathcal{L} = \mathcal{L}_{\mathrm{FB}} + \alpha\, \mathcal{L}_{\mathrm{cons}} + \beta\, \mathcal{L}_{B},$$

together with the actor loss and FB's orthonormality penalty. No reward appears anywhere in training.

### 3.4 Zero-shot planning at test time

**Task inference.** Given $N$ labelled states $s_i \sim \rho$, the task vector is FB's usual Monte-Carlo estimate of $z_r = \mathbb{E}_\rho[B\, r]$,

$$z = \frac{1}{N} \sum_{i=1}^{N} B(s_i)\, r(s_i) .$$

This is the vector that makes $F^\top z$ the Q-function of $r$ (§2), so the actor and the terminal value use it unchanged. As in FB, $z$ is then rescaled to the norm $\sqrt d$ used during training. This multiplies the reward by a positive constant, which changes neither the optimal actions nor the planner's choices, provided the per-step reward below is rescaled by the same constant.

**Per-step rewards: project the reward onto the basis.** The planner also needs a per-step reward $\hat r(\hat h)$ on imagined latents. The obvious choice, $\hat r = g(\hat h)^\top z$, is the right one only if $\mathbb{E}_\rho[BB^\top] = I$. FB's orthonormality penalty pushes towards that, but it holds only approximately. *(TODO: report how far $\Sigma$ is from $I$ in trained models.)* The consistent choice is the $L^2(\rho)$ projection of $r$ onto the span of $B$:

$$\hat r(s) = B(s)^\top w, \qquad w = \big(\mathbb{E}_\rho[BB^\top] + \epsilon I\big)^{-1} z .$$

This choice describes the same reward as the terminal value: for this reward,

$$F^\top \mathbb{E}_\rho\big[B\, \hat r\big] = F^\top \mathbb{E}_\rho[BB^\top]\, w = F^\top z ,$$

up to the small ridge $\epsilon$. So the per-step reward $B^\top w$ and the terminal value $F^\top z$ are two views of one reward, the best approximation of $r$ in the span of $B$. With $B^\top z$ instead, they would describe two different rewards. The covariance $\mathbb{E}_\rho[BB^\top]$ needs no labels. It is estimated once on unlabelled dataset states, and is shared by all tasks.

**The planning objective.** At each environment step, from $\hat h_0 = \phi(s)$, MPPI searches for the action sequence $a_{0:H-1}$ that maximises

$$J(a_{0:H-1}) = \sum_{t=0}^{H-1} \gamma^{t}\, g(\hat h_{t+1})^\top w \;+\; \gamma^{H}\, F\big(\hat h_H, \pi(\hat h_H, z), z\big)^\top z, \qquad \hat h_{t+1} = f(\hat h_t, a_t).$$

As in TD-MPC, a few of MPPI's samples are rollouts of the actor, so the planner starts from what the policy would do. The terminal value makes the objective long-horizon even though the model is unrolled only $H$ steps.

**Never override the policy against the value function.** The planner and the actor are two ways of choosing an action for the same task, with different failure modes. The planner can exploit errors of the model, and the actor can be suboptimal for its own value function. To arbitrate between them, we use a quantity that involves no model rollout: $F$ at the *real* current latent. We execute

$$a = \arg\max_{a' \in \{a^{\mathrm{MPPI}}_0,\; \pi(\phi(s), z)\}} F(\phi(s), a', z)^\top z ,$$

so the plan replaces the policy's action only when FB's own value function, evaluated on real latents where it was trained, prefers it. This removes the failures caused by model errors, not those caused by errors of $F$ itself. It also means that planning can only correct the actor's optimization error with respect to $F$. A model-free alternative, taking the best of many sampled actions under $F$ with no model, is therefore an essential control (§4.5).

**Algorithm 1 (fbwm at test time).**
```
input: N labelled states {(s_i, r_i)}; Σ = E_ρ[B Bᵀ] precomputed on unlabelled states
z ← normalize( mean_i B(s_i) r_i ) ;  w ← (Σ + εI)⁻¹ z
for each environment step, state s:
    h ← φ(s)
    a_plan ← MPPI over a_{0:H-1} maximising J (rewards g(ĥ)ᵀw, terminal value Fᵀz, model f)
    a_pol  ← π(h, z)
    execute argmax over {a_plan, a_pol} of F(h, ·, z)ᵀz
```
Test-time cost: one average and one $d \times d$ linear solve per task, then planning at every step. No gradient step and no access to the dataset beyond the labelled states (and the task-independent $\Sigma$, computed once after training).

---

## 4. Experiments

We ask two questions.
- **Q1.** With the standard budget of labelled states, does planning with fbwm improve over FB's policy, and how does it compare with methods that train on each task?
- **Q2.** How does the number of labelled states $N$ affect each method? In particular, when is it better to infer $z$, with no training, than to adapt a task-specific model to the same labels?

### 4.1 Setup

**Domains and tasks.** Three DeepMind Control domains, with reward-free datasets collected by an exploration policy driven by random network distillation (RND) (10,000 episodes of 1,000 steps, about 10M transitions per domain). Tasks:
- *walker*: stand, walk, run at target speed 8, run, flip;
- *cheetah*: run forward and backward at two target speeds each, flip;
- *quadruped*: stand, walk, run, jump.

Returns are in $[0, 1000]$ per episode.

**Protocol.** Unless stated otherwise, the task is specified by $N = 10{,}000$ states drawn from the dataset, labelled with their true reward. The same labelled states are given to every method that uses labels. We report the final checkpoint, never the best evaluation. For each run, we average the return over the domain's tasks, using 20 episodes per task for the zero-shot methods (a fresh draw of labelled states per episode, so the variance from the labels is included) and 10 for the trained baselines. Then we report the mean and 95% confidence interval over 3 seeds. *(TODO: unify the number of episodes.)* A single configuration of fbwm is used for every domain.

### 4.2 Methods compared

All methods are trained on the same reward-free dataset, unless they are explicitly given rewards.

**Zero-shot (no training per task).**
- **FB**: forward-backward representations on the raw states (no world model, no shared latent), trained on the same data and acting with its policy $\pi_z$.
- **fbwm, policy**: our model, acting with its actor $\pi(\phi(s), z)$ only. Comparing it with FB measures the effect of training the shared latent and the world model on the policy. Comparing it with fbwm MPPI isolates the effect of planning.
- **fbwm, MPPI (ours)**: the full test-time procedure of §3.4.

**Adaptation baselines (seconds of training per task).** These answer a natural objection: if a world model is available, why not just fit a reward and a value to the labels? To make that comparison fair, the world model is pre-trained in the same way as ours but without FB: a TD-MPC encoder and dynamics, trained on the same dataset with latent consistency only (plus variance and covariance regularisers that prevent collapse). For each task, the model is frozen, and TD-MPC's heads are trained:
1. a reward head is regressed on the $N$ labelled states;
2. a Q-function and a policy prior are trained with TD-MPC's losses, using rewards from that head (6,000 updates, about 20 s on one GPU);
3. the agent acts with the same MPPI planner as ours.

Two variants differ in the data available in step 2:
- **adapted, full dataset**: the Q-function is trained on all dataset transitions, relabelled by the reward head. This baseline uses the whole dataset at adaptation time.
- **adapted, labels only**: the labelled states come with their dataset transitions (action and next state). The Q-function is trained only on these $N$ transitions, plus short imagined rollouts of the frozen model that start from them (Dyna-style). Like fbwm, it sees no other data at adaptation time. It even sees slightly more than fbwm, which uses only the states.

**Test-time compute.** fbwm and both adapted baselines act with the same MPPI planner, so their cost per environment step is the same. The adapted baselines add about 20 s of training per task. FB and fbwm's policy need only one forward pass per step. *(TODO: report wall-clock per episode.)*

**Toplines (true reward on the whole dataset, trained per task).** These are not zero-shot. They use the true reward of every dataset transition and a full training run per task, and they bound what the data allow.
- **TD-MPC**: offline TD-MPC with the true reward (about 35 min per task).
- **offline TD3**: with the true reward (about 8 min per task).

### 4.3 Main results (Q1)

**Table 1.** Mean return over the domain's tasks, $N = 10{,}000$ labelled states, mean ± 95% CI over 3 seeds.

| method | per-task training | walker (5 tasks) | cheetah (5 tasks) | quadruped (4 tasks) |
|---|---|---|---|---|
| *toplines: true reward on all data* | | | | |
| TD-MPC | ~35 min | 789 ± 13 | 684 ± 15 | 831 ± 17 |
| offline TD3 | ~8 min | 681 ± 10 | 510 ± 71 | 753 ± 31 |
| *adaptation: reward-free world model + N labels* | | | | |
| adapted, full dataset | ~20 s | **670 ± 35** | 343 ± 94 | 517 ± 70 |
| adapted, labels only | ~20 s | 594 ± 51 | 331 ± 218 | 449 ± 278 |
| *zero-shot: no training per task* | | | | |
| FB | none | 571 ± 89 | 308 ± 74 | 583 ± 175 |
| fbwm, policy | none | 574 ± 88 | 355 ± 78 | **599 ± 165** |
| **fbwm, MPPI (ours)** | none | 622 ± 118 | **403 ± 108** | 587 ± 266 |

*Bold: best method without the true reward. The confidence intervals are wide with 3 seeds and overlap for most pairs. (TODO: paired per-seed differences with CIs, and more seeds.)*

**Planning improves over FB's policy on two of three domains.** fbwm MPPI beats FB on walker (+51) and cheetah (+95) and matches it on quadruped (+4). Part of this comes from the model itself: fbwm's policy alone is already at or above FB (+3, +47, +16). The rest comes from planning: on walker and cheetah, MPPI adds +48 over fbwm's own policy, with the same networks and the same $z$. On quadruped, planning does not help on average (587 vs 599). The arbitration at the real state prefers the plan only when $F$ prefers it, but $F$ can be wrong too: on one quadruped seed, MPPI lost to the policy (468 vs 540).

**Zero-shot vs adaptation.** Against the baseline that, like fbwm, sees only the labelled states, fbwm MPPI wins on all three domains (+28, +72, +138), with no training per task. The full-dataset adapted baseline, which also trains its Q-function on the whole dataset, is better on walker (670 vs 622). It is worse on cheetah (343 vs 403) and on quadruped (517 vs 587).

**Distance to the toplines.** fbwm remains below TD-MPC trained on the true reward of the whole dataset, by 170 to 280 points depending on the domain, and below offline TD3 on all three domains. These toplines use about $10^7$ reward labels and a training run per task. fbwm uses $10^4$ labels and no training.

### 4.4 Effect of the number of labelled states (Q2)

We vary $N \in \{100, 300, 1000, 3000, 10000\}$ for every method that uses labels, on walker and cheetah. *(TODO: quadruped.)* The pre-trained models are the same at every $N$. Only the task inference (fbwm, FB) or the adaptation (adapted baselines) changes. Figure 1 averages walker and cheetah; per-domain values are in Table 2.

**Figure 1.** *Mean return over walker and cheetah vs $N$ (log scale), 3 seeds per point. Dashed lines: the TD-MPC (737) and offline TD3 (596) toplines, which use the true reward of the whole dataset and do not depend on $N$.* [To be rendered from `hpc/status/paper_label_curve.json`; the tracker's overview shows it.]

| method | N = 100 | 300 | 1k | 3k | 10k |
|---|---|---|---|---|---|
| fbwm, MPPI (ours) | 323 | 402 | 469 | 493 | **512** |
| fbwm, policy | 301 | 397 | 441 | 462 | 464 |
| FB | 293 | 371 | 418 | 430 | 440 |
| adapted, full dataset | **345** | **409** | **493** | **513** | 507 |
| adapted, labels only | 121 | 225 | 258 | 344 | 462 |

**Table 2.** Per-domain values of Figure 1 (walker / cheetah).

| method | N = 100 | 300 | 1k | 3k | 10k |
|---|---|---|---|---|---|
| fbwm, MPPI (ours) | 379 / 267 | 478 / 326 | 557 / 381 | 598 / 388 | 622 / 403 |
| fbwm, policy | 351 / 251 | 465 / 329 | 525 / 356 | 562 / 363 | 574 / 355 |
| FB | 360 / 227 | 481 / 261 | 546 / 290 | 566 / 295 | 571 / 308 |
| adapted, full dataset | 514 / 176 | 554 / 264 | 630 / 356 | 678 / 348 | 670 / 343 |
| adapted, labels only | 170 / 73 | 323 / 128 | 331 / 185 | 448 / 240 | 594 / 331 |

**Zero-shot inference is far more label-efficient than adaptation from labels alone.** With the same labelled states and no other data, fbwm beats the labels-only adapted baseline at every $N$, by about 200 points at $N \le 1000$. The labels-only baseline needs $N = 10{,}000$ to approach fbwm. The reason is structural. The adapted baseline must learn a reward and a value for the task from the labels. fbwm already has a value function for every task in its basis, and the labels are only used to locate the task in that basis, a $d$-dimensional average.

**With dataset access, adaptation is competitive.** The full-dataset baseline relabels the whole dataset with its learned reward head. On average over the two domains, it is ahead of fbwm at 4 of the 5 budgets (by 7 to 24 points) and behind at $N = 10{,}000$ (507 vs 512). The per-domain picture is sharper. On walker it is better at every $N$, by 48 to 135 points. On cheetah fbwm is better at every $N$, by 25 to 90 points. Note that this baseline needs access to the full dataset and a training run for each new task, while fbwm needs neither.

**Planning helps most when labels are plentiful.** The gain of MPPI over fbwm's policy is +22, +5, +28, +31 and +48 from $N = 100$ to $10{,}000$. A likely reason is that $w = (\Sigma + \epsilon I)^{-1} z$ amplifies the noise of $z$ along the directions where $\Sigma$ is small, so the per-step reward is noisier than $z$ when labels are few. FB and fbwm's policy level off after about 1,000 labels, while fbwm MPPI keeps improving up to 10,000.

### 4.5 Ablations (to be written)

**Needed for the central claims (TODO, not run in the paper configuration):**
- fbwm without $\mathcal{L}_B$ (no grounding in the reward basis), and with separate latents for FB and the model. These test §3.2–3.3.
- A model-free control: the best of $K$ sampled actions (around the actor's) under $F(\phi(s), a, z)^\top z$, with no world model. This tests whether the model is what helps.

**Data already exists for the following; a table will be added:**
- Per-step reward: projected ($B^\top w$) vs $B^\top z$. +52 on walker, no difference on cheetah.
- The arbitration at the real state, and other ways to combine planner and policy.
- Dimension $d$ of the representation (50 / 100 / 200). Larger $d$ helps walker and breaks FB training on cheetah.
- Weight of the latent consistency (5 vs 2). 2 is better on quadruped only.
- Estimate of $\mathbb{E}[BB^\top]$ from the labelled states vs from unlabelled states: no difference at the default $d$.
- Planner hyperparameters (horizon, terminal value only): no effect.
- Negative results: training the F map on imagined latents, alternative task-vector distributions during training.

---

## 5. Related work

*(Draft; references to verify.)*
- **Zero-shot RL and behaviour foundation models**: forward-backward representations [Touati & Ollivier, 2021; Touati et al., 2023], successor features [Barreto et al., 2017], Hilbert representations (HILP) [Park et al., 2024], functional reward encodings (FRE) [Frans et al., 2024]. All of these act with an amortized task-conditioned policy. We add test-time planning to FB.
- **Latent model-based RL and planning**: TD-MPC and TD-MPC2 [Hansen et al., 2022; 2024], MPPI [Williams et al., 2017], MuZero-style value-equivalent models [Schrittwieser et al., 2020; Grimm et al., 2020], self-predictive (JEPA-style) representations [LeCun, 2022]. These ground the model in one task's reward and value. We ground it in a basis of rewards.
- **Test-time adaptation of behaviour foundation models**: methods that fine-tune $z$ or the policy at test time spend test-time compute too, but through gradient steps rather than planning. *(To compare with.)*

---

## 6. Limitations and next steps

- Three seeds per cell. The confidence intervals overlap for most comparisons in Table 1, and quadruped is very variable.
- The core ablations (§4.5) and a model-free test-time baseline are still missing.
- Planning does not help on quadruped with the shared configuration. The value-based arbitration prevents most, but not all, failures of the planner.
- fbwm is still well below per-task toplines trained on the true reward.
- With few labels ($N \le 300$), the least-squares per-step reward is noisy. A ridge that scales with $d/N$ is a natural fix, not yet tested in the main configuration.
- State-based DeepMind Control only. Pixels, where a latent world model should matter most, and other zero-shot baselines (HILP, FRE) are next.

---

## Appendix A. Implementation details

- **fbwm**: representation dimension $d = 50$, consistency weight $\alpha = 5$, $\mathcal{L}_B$ weight $\beta = 2$, model horizon $H = 5$ with $\lambda = 0.5$, $\gamma = 0.95$, two $F$ heads (min used for values), 500k updates, batch 1024, learning rate $10^{-4}$. Latents are normalised to a sphere. The dynamics are residual: $f(h, a) = \mathrm{normalize}(h + \mathrm{MLP}(h, a))$.
- **MPPI**: 6 iterations, 512 samples plus 25 actor trajectories, 64 elites, temperature 0.5. Ridge $\epsilon = 10^{-2}\, \mathrm{tr}(\Sigma)/d$. $\Sigma = \mathbb{E}_\rho[BB^\top]$ is computed on 100k unlabelled dataset states.
- **Task vector**: $z$ rescaled to norm $\sqrt d$. $w$ uses the same scale factor.
- **Adaptation baselines**: reward head 2,000 updates, Q-function and policy prior 6,000 updates, same planner as fbwm.
- **FB baseline**: TODO, give its architecture and number of updates.
<!-- Code: fbwm.py variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset mppi_root_check=1; baselines tdmpc_adapt.py, tdmpc_jax.py, td3_jax.py, fb_jax.py. -->
