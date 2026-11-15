# Zero-shot planning with forward-backward representations and a latent world model

<!--
Version v3 (2026-11-16): after reader round 1 on v2 (rescaling argument fixed, claims tied to CIs, notation).
v2 (2026-11-16): rewritten against docs/writing_checklist.md (branch writing-checklist).
v1 (2026-11-15): first draft, revised after one first-reader pass.
Numbers: hpc/status/paper_main_table.json, hpc/status/paper_label_curve.json (tools/paper_tables.py); run folders in docs/decisions.md and docs/journal.md.
Items marked TODO need data or decisions.
-->

## Abstract

Zero-shot reinforcement learning (RL) pre-trains an agent on reward-free data so that it can solve a task given at test time without further training. Forward-backward (FB) representations do this with a learned basis of reward functions $B$, their successor features $F$, and one actor conditioned on the task. The actor has to be near-optimal for every task at once, and FB has no way to improve its actions at test time. For any task, FB already gives the two quantities a planner scores trajectories with, a reward linear in $B$ and a value $F^\top z$, but it has no dynamics model. We train a latent dynamics model jointly with FB, on the same latent space, and add one loss: the backward embedding of an imagined latent must match that of the real future state. Since every reward FB can represent is linear in $B$, this loss bounds the model's reward error for all of them. At test time, the task vector is inferred from labelled states and the agent plans with MPPI in latent space, with no gradient step. On 14 tasks in three DeepMind Control domains, planning adds 48 points of mean return over the actor on walker and cheetah, and loses 12 on quadruped. With 100 to 10,000 labelled states (walker and cheetah), fbwm has a higher mean return, at every budget, than a model-based baseline that adapts a reward-free world model using the labelled transitions alone. A baseline that may also train on the whole dataset is better on walker and worse on cheetah, at every budget. With 3 seeds, most of these differences are within the confidence intervals.

## 1. Introduction

A robot or a simulated agent can often collect millions of reward-free transitions before anyone tells it which task to solve, and running RL again for each new task costs a full training run. Zero-shot RL asks the agent to pre-train once without rewards and then, given a description of a new task (here, a set of states labelled with their reward), to produce a good policy with no further learning.

Forward-backward representations [Touati & Ollivier, 2021] learn two maps for this. The backward embedding $B(s) \in \mathbb{R}^d$ is a basis of reward functions, and the forward map $F(s, a, z)$ gives the successor features, in that basis, of a policy $\pi_z$. For a new reward $r$, the task vector $z_r = \mathbb{E}_{s \sim \rho}[B(s)\, r(s)]$ is an average over labelled states, and $F(s,a,z_r)^\top z_r$ is the Q-function of $\pi_{z_r}$ for $r$. In a comparison of zero-shot methods on DeepMind Control, FB gave the best results overall [Touati et al., 2023].

At test time, everything FB does goes through one actor network $\pi(s, z)$, trained to maximise $F^\top z$ over a whole distribution of task vectors. For a given task, this actor is only approximately greedy with respect to its own value function. It may also be queried on task vectors that were rare in training. FB has no mechanism to spend more computation at test time to get better actions. Fine-tuning $z$ or the policy can, but it takes gradient steps for every new task.

Model-based agents such as TD-MPC [Hansen et al., 2022] spend test-time computation by planning. At each step they optimise a short action sequence through a learned latent model, and score it with a learned reward and a learned terminal value. Both heads are trained for one task, so a new task needs new training. FB already provides both for any task vector $z$, a per-step reward linear in $B$ and the terminal value $F^\top z$. It lacks a dynamics model. Dynamics do not depend on the task, so such a model can be trained on the same reward-free data.

We call the result fbwm (FB with a world model). We train the model jointly with FB. The encoder, $B$, $F$, the actor and the dynamics all share one latent space. The dynamics are trained by latent self-prediction, as in TD-MPC and joint-embedding predictive architectures (JEPA) [LeCun, 2022]: imagined latents must match the encodings of the real future states. A second loss ties the model to FB's reward basis: the backward embedding of an imagined latent must match the backward embedding of the real future state. Every reward FB can represent is a linear function of $B$, so this one loss controls the model's reward error for all of them. At test time we infer $z$ from labelled states, as FB does, and plan in latent space with MPPI, a sampling-based trajectory optimiser [Williams et al., 2017], without updating any parameter.

Contributions:
1. fbwm, which turns an FB representation into a zero-shot planner, by training a latent world model tied to FB's reward basis (§3).
2. Two test-time choices that keep the planner consistent with FB's value function (§3.4). The per-step reward is the projection of the task reward onto the span of $B$. A value check executes the planner's action only if $F$, at the real state, rates it above the actor's.
3. Experiments on 14 tasks in three DeepMind Control domains (§4). We compare with FB, with per-task toplines trained on the true reward, and with baselines that adapt a reward-free world model to the same labels. On walker and cheetah we repeat the comparison with 100 to 10,000 labelled states.

## 2. Background

**Setting.** A reward-free MDP has states $s$, actions $a$, transitions $P(s' \mid s, a)$ and discount $\gamma$. A dataset $\mathcal{D}$ of transitions $(s, a, s')$ was collected by an exploration policy, with state distribution $\rho$. At test time, a task is a reward function $r: \mathcal{S} \to \mathbb{R}$, known only through $N$ labelled states $\{(s_i, r(s_i))\}_{i=1}^N$ with $s_i \sim \rho$.

**Successor measures and FB.** The successor measure of a policy $\pi$ counts discounted visits to a set of states $X$:

$$M^\pi(s_0, a_0, X) = \sum_{t \ge 0} \gamma^t \, \Pr(s_{t+1} \in X \mid s_0, a_0, \pi).$$

The Q-function of any reward is linear in $M^\pi$: $Q^\pi_r(s,a) = \int M^\pi(s, a, ds')\, r(s')$. FB learns policies $\pi_z$ indexed by a task vector $z \in \mathbb{R}^d$, and a rank-$d$ model of their successor measures:

$$M^{\pi_z}(s, a, ds') \approx F(s, a, z)^\top B(s')\, \rho(ds').$$

Substituting into $Q$ gives, for every reward $r$,

$$Q^{\pi_z}_r(s,a) \approx F(s,a,z)^\top \mathbb{E}_{s' \sim \rho}\big[B(s')\, r(s')\big] = F(s,a,z)^\top z_r .$$

Each policy is trained to be greedy for its own task vector, $\pi_z(s) \approx \arg\max_a F(s,a,z)^\top z$. In practice $\pi_z(s) = \pi(s, z)$ is one actor network that takes $z$ as input. With perfect training, $\pi_{z_r}$ is optimal for every reward $r$ in the span of $B$.

**Training FB.** $F$ and $B$ are trained by temporal-difference learning on the successor measure. During training, $z$ is drawn uniformly on the sphere of radius $\sqrt d$, mixed with task vectors $B(s)$ of single dataset states. The loss is

$$\mathcal{L}_{\mathrm{FB}} = \mathbb{E}\Big[\big(F(s,a,z)^\top B(s^+) - \gamma\, \bar F(s', \pi_z(s'), z)^\top \bar B(s^+)\big)^2\Big] - 2\, \mathbb{E}\big[F(s,a,z)^\top B(s')\big],$$

with $(s, a, s') \sim \mathcal{D}$ and an independent state $s^+ \sim \rho$. Bars denote target networks, updated as $\bar\theta \leftarrow (1-\tau)\bar\theta + \tau\theta$. An orthonormality penalty pushes $\mathbb{E}_\rho[BB^\top]$ towards the identity. The actor maximises $F(s, \pi(s,z), z)^\top z$.

**Latent model-predictive control.** TD-MPC encodes a state into a latent $h = \phi(s)$, and learns a dynamics model $h_{t+1} = f(h_t, a_t)$, a reward head and a Q-function. At each environment step, it searches for the action sequence with the highest predicted $H$-step return plus terminal value, executes the first action, and plans again at the next step. The search is MPPI (model-predictive path integral control). MPPI samples action sequences from a Gaussian, scores them through the model, and refits the Gaussian to the best ones, for a few iterations. The model is trained by latent consistency: imagined latents must match target encodings of the real future states, and observations are never reconstructed.

## 3. Method

### 3.1 Components

fbwm has five learned components, all on one latent space:

| component | role |
|---|---|
| encoder $h = \phi(s)$ | shared latent state |
| backward head $B(s) = g(\phi(s))$ | basis of rewards |
| forward map $F(h, a, z)$ | successor features, so values $F^\top z$ for every task |
| actor $\pi(h, z)$ | FB's task-conditioned policy; seeds the planner and is its fallback |
| dynamics $h_{t+1} = f(h_t, a_t)$ | task-independent latent world model |

From here on, $F$ and $\pi$ take the latent $h$ as input. All five are trained once, on reward-free data. At test time, the agent plans through $f$, with per-step rewards from $g$ and terminal values from $F$.

### 3.2 A shared latent space

FB and the world model use the same encoder. The FB loss trains $\phi$ through both $F$ and $B = g \circ \phi$, and the consistency loss of §3.3 trains it through the dynamics. The planner scores imagined latents $\hat h$ with $g(\hat h)$ and $F(\hat h, \cdot, z)$, so these heads have to be accurate on latents produced by the model. The consistency loss trains the model to output encodings of real states, which are the inputs $g$ and $F$ are trained on. This holds for the dataset's actions. For other actions, nothing forces the model's outputs to stay among real encodings, and the value check of §3.4 is our guard against that.

### 3.3 Tying the world model to FB's reward basis

The dynamics are trained on dataset sub-trajectories $(s_0, a_0, \dots, a_{H-1}, s_H)$. From $\hat h_0 = \phi(s_0)$, the model is rolled forward with the dataset's actions, $\hat h_{t+1} = f(\hat h_t, a_t)$. Two losses are applied to the rollout. The first is TD-MPC's latent consistency,

$$\mathcal{L}_{\mathrm{cons}} = \sum_{t=1}^{H} \lambda^{t} \, \big\| \hat h_t - \bar\phi(s_t) \big\|^2 ,$$

where $\bar\phi$ is the target encoder and $\lambda \in (0,1)$ down-weights distant steps. The second asks the backward embedding of each imagined latent to match the target backward embedding $\bar B = \bar g \circ \bar\phi$ of the real state:

$$\mathcal{L}_{B} = \sum_{t=1}^{H} \lambda^{t} \, \big\| g(\hat h_t) - \bar B(s_t) \big\|^2 .$$

Every reward FB can represent is, up to projection, $B^\top w$ for some weight vector $w$. For any $w$, $|g(\hat h_t)^\top w - \bar B(s_t)^\top w| \le \|w\|\,\|g(\hat h_t) - \bar B(s_t)\|$. So $\mathcal{L}_B$ bounds the model's reward error over $H$ imagined steps for every reward in the span of $B$ at the same time, with a constant $\|w\|$ that depends on the task. TD-MPC's reward-prediction loss does this for one reward. This follows the value equivalence principle [Grimm et al., 2020], which asks a model to be right about the quantities used for decisions; here they are the whole basis $B$. $\mathcal{L}_B$ also trains $g$ on imagined latents, which are its inputs during planning. Both losses only see the dataset's actions, so the bound holds for those actions.

The training objective is

$$\mathcal{L} = \mathcal{L}_{\mathrm{FB}} + \alpha\, \mathcal{L}_{\mathrm{cons}} + \beta\, \mathcal{L}_{B},$$

plus the actor loss and FB's orthonormality penalty. Training uses no reward.

### 3.4 Planning at test time

**Task inference.** Given $N$ labelled states $s_i \sim \rho$, the task vector is FB's Monte-Carlo estimate of $z_r = \mathbb{E}_\rho[B\, r]$,

$$z = \frac{1}{N} \sum_{i=1}^{N} B(s_i)\, r(s_i) .$$

As in FB, $z$ is then rescaled to norm $\sqrt d$, because $F$ and $\pi$ take $z$ as input and were trained on vectors of that norm. The rescaled vector $cz$, with $c > 0$, is the task vector of the reward $c\,r$, which has the same optimal policy as $r$. So $\pi(\cdot, cz)$ targets the same behaviour, and $F^\top (cz)$ is the Q-function of $c\,r$ (§2). From here on $z$ denotes the rescaled vector, and the per-step reward below is scaled by the same $c$, so that every term of the planning objective refers to $c\,r$.

**Per-step reward.** The planner also needs a reward on imagined latents. Write $\Sigma = \mathbb{E}_\rho[BB^\top]$. Using $g(\hat h)^\top z$ would be correct only if $\Sigma = I$. The orthonormality penalty pushes towards this, but does not reach it. *(TODO: report how far $\Sigma$ is from $I$ in trained models.)* We use the $L^2(\rho)$ projection of $r$ onto the span of $B$, $\hat r(s) = B(s)^\top w$, evaluated on imagined latents as $g(\hat h)^\top w$:

$$w = (\Sigma + \epsilon I)^{-1} z .$$

For this reward, up to the ridge $\epsilon$,

$$F^\top \mathbb{E}_\rho\big[B\, \hat r\big] = F^\top \Sigma\, w = F^\top z .$$

The per-step reward $B^\top w$ and the terminal value $F^\top z$ therefore describe the same reward, the least-squares approximation of $r$ in the span of $B$. With $B^\top z$ as the per-step reward, the two would describe different rewards whenever $\Sigma \ne I$. $\Sigma$ needs no labels. It is estimated once on unlabelled dataset states and used for every task. Since $\Sigma^{-1}$ amplifies $z$ along the directions where $\Sigma$ is small, $\|w\|$ can be large, which enlarges both the model's reward error (§3.3) and the effect of noise in $z$.

**Planning objective.** At each environment step, starting from $\hat h_0 = \phi(s)$, MPPI searches over a planning horizon $H$ (the same as the training horizon by default) for the action sequence $a_{0:H-1}$ that maximises

$$J(a_{0:H-1}) = \sum_{t=0}^{H-1} \gamma^{t}\, g(\hat h_{t+1})^\top w \;+\; \gamma^{H}\, F\big(\hat h_H, \pi(\hat h_H, z), z\big)^\top z, \qquad \hat h_{t+1} = f(\hat h_t, a_t).$$

As in TD-MPC, some of MPPI's samples are rollouts of the actor, so the search starts from the actor's behaviour. The terminal value accounts for the return after the $H$ simulated steps.

**Value check.** The planner can exploit errors of the model, and the actor can be suboptimal for its own value function. We compare their first actions with $F$ at the real current latent, which involves no model rollout, and execute

$$a = \arg\max_{a' \in \{a^{\mathrm{MPPI}}_0,\; \pi(\phi(s), z)\}} F(\phi(s), a', z)^\top z .$$

The check rejects a plan that looks good only because of model error, provided $F$ ranks the two actions correctly. It cannot help when $F$ itself is wrong. It also limits what planning can correct to the actor's optimisation error with respect to $F$. A model-free control follows: take the best of many sampled actions under $F$, with no model. We have not run it yet (§4.5).

**Algorithm 1** (fbwm at test time).
```
input: N labelled states {(s_i, r_i)}; Σ = E_ρ[B Bᵀ] precomputed on unlabelled states
z ← normalize( mean_i B(s_i) r_i ) ;  w ← (Σ + εI)⁻¹ z
for each environment step, state s:
    h ← φ(s)
    a_plan ← MPPI over a_{0:H-1} maximising J (rewards g(ĥ)ᵀw, terminal value Fᵀz, model f)
    a_pol  ← π(h, z)
    execute argmax over {a_plan, a_pol} of F(h, ·, z)ᵀz
```
Per task, the cost is one average over labelled states and one $d \times d$ linear solve, and per step one MPPI search. Apart from the labelled states, the only data used at test time is $\Sigma$, computed once after training.

## 4. Experiments

We ask two questions.
- **Q1.** With $N = 10{,}000$ labelled states, does planning with fbwm improve over FB's policy, and how does it compare with methods trained on each task?
- **Q2.** How does each method depend on $N$? For which $N$ is inferring $z$, with no training, better than adapting a task-specific model to the same labels?

### 4.1 Setup

We use three DeepMind Control domains. Each has a reward-free dataset collected by an exploration policy driven by random network distillation (RND): 10,000 episodes of 1,000 steps, about 10M transitions. The tasks are:
- walker: stand, walk (target speed 2), run at speeds 8 and 10, flip;
- cheetah: run forward and backward at two target speeds each, flip;
- quadruped: stand, walk, run, jump.

Episode returns are in $[0, 1000]$.

Unless stated otherwise, a task is given by $N = 10{,}000$ dataset states drawn uniformly and labelled with their true reward. We report the final checkpoint, not the best evaluation, and each run's score is its mean return over the domain's tasks. The zero-shot methods are evaluated on 20 episodes per task, with a new draw of labelled states for each episode. The adapted baselines draw their labelled states once per seed, train on them, and are evaluated on 10 episodes per task, as are the toplines. In both cases the variance due to the labels enters the confidence interval over seeds. We report the mean and 95% confidence interval over 3 seeds. *(TODO: unify the number of episodes.)* fbwm uses one configuration for all domains.

### 4.2 Methods

All methods are trained on the same reward-free dataset. The toplines are the only ones given the reward of every transition.

**Zero-shot methods**, with no training per task:
- *FB*: forward-backward representations on raw states, with no world model and no shared latent, acting with $\pi_z$.
- *fbwm, policy*: fbwm acting with its actor $\pi(\phi(s), z)$ only. The comparison with FB measures what the shared latent and the world-model losses do to the actor. The comparison with fbwm MPPI measures planning alone.
- *fbwm, MPPI*: the full test-time procedure of §3.4.

**Adaptation baselines**, with about 20 s of training per task. They test whether a reward-free world model, with a reward and a value fitted to the labels, does as well as fbwm. Their world model is pre-trained like fbwm's but without FB: a TD-MPC encoder and dynamics, trained on the same dataset with latent consistency, plus variance and covariance regularisers against collapse. For each task, the model is frozen and TD-MPC's heads are trained. A reward head is regressed on the $N$ labelled states. Then a Q-function and a policy prior are trained with TD-MPC's losses on rewards from that head, for 6,000 updates. The agent acts with the same MPPI planner as fbwm. The two variants differ in the data used to train the Q-function:
- *adapted, full dataset*: all dataset transitions, relabelled by the reward head.
- *adapted, labels only*: the $N$ labelled states with their dataset transitions (action and next state), plus short model rollouts that start from them (Dyna-style). Like fbwm, it uses no other dataset transitions at adaptation time. Unlike fbwm, it also uses the labelled states' actions and next states.

fbwm and both adapted baselines plan with the same MPPI and pay the same cost per environment step, and the adapted baselines add 20 s of training per task. FB and fbwm's policy need one forward pass per step. *(TODO: wall-clock per episode.)*

**Toplines.** These are trained per task with the true reward of every dataset transition, about $10^7$ labels, and show what the data allow.
- *TD-MPC*: offline TD-MPC, about 35 min per task.
- *Offline TD3*: about 8 min per task.

### 4.3 Main results (Q1)

**Table 1.** Mean return over the domain's tasks with $N = 10{,}000$ labelled states. Mean ± 95% CI over 3 seeds. Bold: best method that is not given the reward of every transition.

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
| fbwm, MPPI (ours) | none | 622 ± 118 | **403 ± 108** | 587 ± 266 |

With 3 seeds, the confidence intervals overlap for most pairs of methods. *(TODO: paired per-seed differences with CIs, and more seeds.)*

In mean return, fbwm MPPI is above FB on walker (+51) and cheetah (+95), and equal to it on quadruped (+4). Part of the difference is already in fbwm's actor, which is at or above FB at this $N$ (+3, +47, +16), and planning adds the rest: on walker and cheetah, with the same networks and the same $z$, MPPI is 48 points above fbwm's actor. On quadruped it is 12 points below (587 vs 599). The value check relies on $F$, and $F$ can be wrong: on one quadruped seed, MPPI scored 468 and the actor 540.

The labels-only adapted baseline sees the labelled transitions and nothing else from the dataset. fbwm MPPI has a higher mean on all three domains (+28, +72, +138), though the baseline's intervals on cheetah and quadruped (±218, ±278) are too wide to call this a clear win. The full-dataset baseline, which also trains its Q-function on the whole dataset, is above fbwm on walker (670 vs 622) and below on cheetah (343 vs 403) and quadruped (517 vs 587).

fbwm is 167 to 281 points below TD-MPC, and below offline TD3 on all three domains, with 1,000 times fewer labels and no per-task training.

### 4.4 Number of labelled states (Q2)

On walker and cheetah, we vary $N \in \{100, 300, 1000, 3000, 10000\}$ for every method that uses labels. *(TODO: quadruped.)* The pre-trained models are the same for all $N$. Only the task inference (fbwm, FB) or the adaptation (adapted baselines) changes. Figure 1 shows the mean of walker and cheetah, and Table 2 gives each domain.

**Figure 1.** Mean return over walker and cheetah against $N$ (log scale), 3 seeds per point. Dashed lines: TD-MPC (737) and offline TD3 (596), which use the true reward of the whole dataset and do not depend on $N$. *[To be rendered from `hpc/status/paper_label_curve.json`; the tracker's overview has it.]*

| method | N = 100 | 300 | 1k | 3k | 10k |
|---|---|---|---|---|---|
| fbwm, MPPI (ours) | 323 | 402 | 469 | 493 | **512** |
| fbwm, policy | 301 | 397 | 441 | 462 | 464 |
| FB | 293 | 371 | 418 | 430 | 440 |
| adapted, full dataset | **345** | **409** | **493** | **513** | 507 |
| adapted, labels only | 121 | 225 | 258 | 344 | 462 |

**Table 2.** Values of Figure 1 per domain (walker / cheetah).

| method | N = 100 | 300 | 1k | 3k | 10k |
|---|---|---|---|---|---|
| fbwm, MPPI (ours) | 379 / 267 | 478 / 326 | 557 / 381 | 598 / 388 | 622 / 403 |
| fbwm, policy | 351 / 251 | 465 / 329 | 525 / 356 | 562 / 363 | 574 / 355 |
| FB | 360 / 227 | 481 / 261 | 546 / 290 | 566 / 295 | 571 / 308 |
| adapted, full dataset | 514 / 176 | 554 / 264 | 630 / 356 | 678 / 348 | 670 / 343 |
| adapted, labels only | 170 / 73 | 323 / 128 | 331 / 185 | 448 / 240 | 594 / 331 |

fbwm's mean is above the labels-only adapted baseline at every $N$, by about 200 points for $N \le 1000$ and by 50 points at $N = 10{,}000$. That baseline has to learn a reward and a value for the task from the labels, while fbwm already has a value function for every task in the span of $B$ and uses the labels only to compute $z$, a $d$-dimensional average.

The full-dataset baseline relabels the whole dataset with its reward head. On the mean of the two domains, it is above fbwm at four of the five budgets, by 7 to 24 points, and below at $N = 10{,}000$ (507 vs 512). The domains disagree. On walker it is above fbwm at every $N$, by 48 to 135 points. On cheetah it is below at every $N$, by 25 to 91 points.

From $N = 100$ to $10{,}000$, MPPI is +22, +5, +28, +31 and +48 above fbwm's actor. One explanation, not yet tested, is that $w = (\Sigma + \epsilon I)^{-1} z$ amplifies the noise of $z$ along the directions where $\Sigma$ is small, so the per-step reward is noisier than $z$ when $N$ is small. FB and fbwm's actor change little beyond $N = 1000$, and fbwm MPPI still gains 43 points from $N = 1000$ to $10{,}000$. At small $N$ on walker, fbwm is not better than FB: FB's actor is above fbwm's actor at $N \le 1000$, and above fbwm MPPI at $N = 300$ (481 vs 478). The advantage of fbwm over FB in Table 1 holds for large $N$.

### 4.5 Ablations (to be written)

Not yet run in the paper configuration, and needed for the central claims (TODO):
- fbwm without $\mathcal{L}_B$, and fbwm with separate latents for FB and the model. These test §3.2 and §3.3.
- A model-free control: the best of $K$ actions sampled around the actor's, scored by $F(\phi(s), a, z)^\top z$, with no world model. This tests whether the model is what helps.

Data already exists for the following; a table will be added:
- Per-step reward $B^\top w$ against $B^\top z$: +52 on walker, no difference on cheetah.
- The value check, and other ways to combine the plan and the actor.
- Representation dimension $d \in \{50, 100, 200\}$: larger $d$ helps walker and makes FB training collapse on cheetah.
- Consistency weight $\alpha = 2$ against 5: 2 is better on quadruped only.
- $\Sigma$ estimated on the labelled states against unlabelled states: no difference at $d = 50$.
- Planner horizon, and a terminal value alone: no effect.
- Negative results: training $F$ on imagined latents, and other distributions of $z$ during training.

## 5. Related work

*(References to verify before submission.)*

Zero-shot RL methods pre-train a policy conditioned on a task representation and read off the task at test time. FB [Touati & Ollivier, 2021; Touati et al., 2023] and successor features [Barreto et al., 2017] use a linear reward basis. Hilbert representations (HILP) [Park et al., 2024] and functional reward encodings (FRE) [Frans et al., 2024] learn other task encodings. All of them act through one amortised policy at test time. fbwm keeps FB's representation and acts by planning, with the actor as a proposal and fallback.

Latent model-based agents such as TD-MPC and TD-MPC2 [Hansen et al., 2022; 2024] and MuZero [Schrittwieser et al., 2020] plan with models trained to predict one task's reward and value. The value equivalence principle [Grimm et al., 2020] states that a model only needs to be right about the quantities used to make decisions. In fbwm those quantities are the whole reward basis $B$. The planner is MPPI [Williams et al., 2017], and the self-prediction loss follows JEPA [LeCun, 2022] and TD-MPC.

Test-time adaptation methods for behaviour foundation models also spend computation per task. They do it with gradient steps on $z$ or on the policy. fbwm does it by planning, with no update. *(TODO: cite and compare.)*

## 6. Limitations

- Each cell has 3 seeds, and the confidence intervals in Table 1 overlap for most comparisons. Quadruped varies most across seeds.
- The ablations of §3.2 and §3.3 and the model-free control (§4.5) have not been run.
- Planning does not help on quadruped with the shared configuration. On one seed, the value check did not prevent a loss of 72 points.
- fbwm is 167 to 281 points below TD-MPC trained on the true reward.
- With $N \le 300$ the per-step reward is noisy. A ridge that grows with $d/N$ may help, but has not been tested in the paper configuration.
- Only state-based DeepMind Control. No pixels, where a latent world model should matter most, and no comparison yet with HILP or FRE.

## Appendix A. Implementation details

- fbwm: $d = 50$, $\alpha = 5$, $\beta = 2$, horizon $H = 5$ for training and planning, $\lambda = 0.5$, $\gamma = 0.95$, two $F$ heads (the minimum is used as the value), 500k updates, batch 1024, learning rate $10^{-4}$. Latents are normalised to a sphere, and the dynamics are residual: $f(h, a) = \mathrm{normalize}(h + \mathrm{MLP}(h, a))$.
- MPPI: 6 iterations, 512 samples plus 25 actor trajectories, 64 elites, temperature 0.5.
- Per-step reward: ridge $\epsilon = 10^{-2}\, \mathrm{tr}(\Sigma)/d$; $\Sigma$ computed on 100k unlabelled dataset states; $w$ rescaled by the same factor as $z$.
- Adaptation baselines: reward head 2,000 updates, Q-function and policy prior 6,000 updates, same planner as fbwm.
- FB baseline: TODO, architecture and number of updates.
<!-- Code: fbwm.py variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset mppi_root_check=1; baselines tdmpc_adapt.py, tdmpc_jax.py, td3_jax.py, fb_jax.py. -->
