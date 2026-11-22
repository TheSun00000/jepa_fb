# Planning any reward in a reward-free latent world model with forward-backward representations

<!--
Draft v0 (2026-11-18), written from the world-model-first story (user): we start from a reward-free latent world model and add FB so it can be given any reward. Replaces the FB-first framing of docs/paper_draft.md (v4), whose method math is reused here after its two reader rounds.
Scope of this file: introduction, background, method. Experiments, related work, limitations and appendix are not written.
Working name of the method: fbwm (to be changed if the title changes).
Items marked TODO need a number, a citation check or a decision. Numbers must come from hpc/status/paper_*.json (tools/paper_tables.py); no number below is final.
Run docs/writing_checklist.md section 0 before this leaves the repo.
-->

## 1. Introduction

A latent world model trained on reward-free data can plan only once someone supplies a cost. Current reward-free planners (DINO-WM, PLDM, V-JEPA 2-AC) use the distance to a goal embedding, so they reach goal states. To optimise any other reward they need a reward model and a value function trained for that task.

That requirement is what makes a reward-free model costly to use. A new task arrives as a reward function that the agent can query only on a few labelled states. The planner needs the reward on imagined latents, so a reward head has to be fitted to those labels. It also needs a value for the end of the plan, because a short planning horizon cannot see consequences beyond it, and the value comes from temporal-difference training on the dataset relabelled with the new reward. Both steps are done again for each task. The world model itself, which holds the knowledge of the dynamics, is reused, but everything that connects it to the task is not.

Forward-backward (FB) representations [Touati & Ollivier, 2021] are a way to make that connection once. FB learns a backward embedding $B(s)$ such that the reward functions that matter are linear in $B$, and a forward map $F(s, a, z)$ that gives, for a task vector $z$, the discounted future occupancy of a policy in the basis $B$. For a reward $r$ known through labelled states, $z_r = \mathbb{E}[B(s)\, r(s)]$ is one average over those states, and $F(s,a,z_r)^\top z_r$ is the value of the policy for $r$. So for every reward in the span of $B$, FB provides both quantities a planner needs: a per-step reward that is linear in $B$, and a terminal value. It provides them without training on that task. FB is normally used with an actor $\pi(s, z)$ that acts directly. Here we use it as the task interface of a planner.

The two parts have to meet in one latent space, and this is the technical problem of the paper. The planner evaluates $B$ and $F$ on latents that the dynamics model imagined, not on encodings of real states. A world model trained only by latent self-prediction gives no guarantee that its imagined latents carry the information $B$ and $F$ read. We therefore build FB on the world model's latent space: the encoder, $B$, $F$, the actor and the dynamics share one latent, and the dynamics get one extra loss that asks the backward embedding of an imagined latent to match that of the real future state. Because every reward FB can represent is linear in $B$, this single loss bounds the model's reward error for all of them at once, along the dataset's actions (§3.4).

We call the result fbwm. Pre-training uses only reward-free transitions. At test time the task vector is computed from the labelled states by one average, and the agent plans with MPPI [Williams et al., 2017] through the dynamics, with a reward and a terminal value both read from FB. No parameter is updated for the task.

We make three contributions.

1. A reward-free latent world model that plans for arbitrary rewards given as labelled states, by building FB on its latent space and tying its dynamics to FB's reward basis (§3).
2. Two test-time choices that keep the planner consistent with FB's value function. The per-step reward is the projection of the task reward onto the span of $B$ (§3.5), and the executed action is chosen between the planner's and the actor's by $F$ at the real state (§3.6).
3. An evaluation on 14 tasks in three DeepMind Control domains (§4, not written yet). We compare with FB alone, with baselines that adapt a reward-free world model to the same labels (reward head and value trained per task), and with per-task toplines that use the true reward. On walker and cheetah we vary the number of labels from 100 to 10,000. *(TODO: one sentence with the final numbers.)*

## 2. Background

**Setting.** A reward-free MDP has states $s$, actions $a$, transitions $P(s' \mid s, a)$ and discount $\gamma$. A dataset $\mathcal{D}$ of transitions $(s, a, s')$ was collected by an exploration policy and has state distribution $\rho$. At test time a task is a reward function $r : \mathcal{S} \to \mathbb{R}$, known only through $N$ labelled states $\{(s_i, r(s_i))\}_{i=1}^{N}$ with $s_i \sim \rho$. The agent must return a good policy for $r$ without further training.

**Latent world models and planning.** TD-MPC [Hansen et al., 2022] encodes a state as $h = \phi(s)$, learns latent dynamics $h_{t+1} = f(h_t, a_t)$, and trains a reward head and a Q-function on the same latent. The dynamics are trained by latent consistency: imagined latents must match target encodings $\bar\phi(s_t)$ of the real future states, where $\bar\phi$ is an exponential moving average of $\phi$, and observations are never reconstructed. This is the self-prediction objective of joint-embedding predictive architectures [LeCun, 2022]. Reward-free variants add variance and covariance terms against collapse [PLDM]. At each environment step the agent searches for the action sequence with the highest predicted $H$-step return plus terminal value, executes the first action, and plans again. The search is MPPI [Williams et al., 2017]: sample action sequences from a Gaussian, score them through the model, refit the Gaussian to the best ones, repeat for a few iterations.

The task enters this planner through the score of a trajectory. In TD-MPC it is a learned reward and Q-function, trained for one task. DINO-WM, PLDM and V-JEPA 2-AC replace it with the distance between the imagined latent and the latent of a goal observation, which needs no training and fixes the task to goal reaching.

**Forward-backward representations.** The successor measure of a policy $\pi$ counts discounted visits to a set of states $X$:

$$M^\pi(s_0, a_0, X) = \sum_{t \ge 0} \gamma^t \Pr(s_{t+1} \in X \mid s_0, a_0, \pi).$$

The Q-function of any reward is linear in $M^\pi$: $Q^\pi_r(s,a) = \int M^\pi(s,a,ds')\, r(s')$. FB learns policies $\pi_z$ indexed by a task vector $z \in \mathbb{R}^d$ and a rank-$d$ model of their successor measures,

$$M^{\pi_z}(s, a, ds') \approx F(s, a, z)^\top B(s')\, \rho(ds').$$

Substituting gives, for every reward $r$,

$$Q^{\pi_z}_r(s, a) \approx F(s, a, z)^\top\, \mathbb{E}_{s' \sim \rho}\big[B(s')\, r(s')\big] = F(s, a, z)^\top z_r .$$

Each policy is trained to be greedy for its own task vector, $\pi_z(s) \approx \arg\max_a F(s,a,z)^\top z$, and in practice one actor network $\pi(s, z)$ takes $z$ as input. If the model of $M^{\pi_z}$ is exact and every $\pi_z$ is greedy, $\pi_{z_r}$ is optimal for every $r$. With rank $d$ the model is not exact. Since $z_r$ is unchanged when the part of $r$ orthogonal (in $L^2(\rho)$) to the span of $B$ is removed, FB in effect solves the projection of $r$ onto that span.

$F$ and $B$ are trained by temporal-difference learning on the successor measure. During training, $z$ is drawn uniformly on the sphere of radius $\sqrt d$, mixed with task vectors $B(s)$ of single dataset states, and $B$'s output is normalised to norm $\sqrt d$. With $(s,a,s') \sim \mathcal{D}$ and an independent $s^+ \sim \rho$, the loss is

$$\mathcal{L}_{\mathrm{FB}} = \mathbb{E}\Big[\big(F(s,a,z)^\top B(s^+) - \gamma\, \bar F(s', \pi_z(s'), z)^\top \bar B(s^+)\big)^2\Big] - 2\,\mathbb{E}\big[F(s,a,z)^\top B(s')\big],$$

where bars denote target networks. An orthonormality penalty pushes $\mathbb{E}_\rho[BB^\top]$ towards the identity, and the actor maximises $F(s, \pi(s,z), z)^\top z$.

## 3. Method

### 3.1 Components

fbwm has five learned components on one latent space:

| component | role |
|---|---|
| encoder $h = \phi(s)$ | shared latent state |
| dynamics $h_{t+1} = f(h_t, a_t)$ | task-independent world model |
| backward head $B(s) = g(\phi(s))$ | basis of rewards |
| forward map $F(h, a, z)$ | successor features, so values $F^\top z$ for every task |
| actor $\pi(h, z)$ | FB's task-conditioned policy; seeds the planner and is its fallback |

$F$ and $\pi$ take the latent $h$ as input. The dynamics are residual and live on the unit sphere, $f(h,a) = \mathrm{normalize}(h + \mathrm{MLP}(h, a))$, with latents normalised to the sphere. All five components are trained once on reward-free data. The world model is the starting point: the rest of the section describes what is added to it so that it can be given a reward.

### 3.2 World model

The dynamics are trained on dataset sub-trajectories $(s_0, a_0, \dots, a_{H-1}, s_H)$. From $\hat h_0 = \phi(s_0)$ the model is rolled forward with the dataset's actions, $\hat h_{t+1} = f(\hat h_t, a_t)$, and trained by latent consistency,

$$\mathcal{L}_{\mathrm{cons}} = \sum_{t=1}^{H} \lambda^{t}\, \big\| \hat h_t - \bar\phi(s_t) \big\|^2,$$

where $\lambda \in (0, 1)$ down-weights distant steps. On its own this is the reward-free world model that the adaptation baselines of §4 use (with variance and covariance regularisers against collapse). It says nothing about which directions of the latent matter for a reward.

### 3.3 FB on the world model's latent

$B$ is a head on the shared encoder, $B = g \circ \phi$, and $F$ and $\pi$ read the latent $h = \phi(s)$. The FB loss of §2 is applied on the first transition of each sub-trajectory, and its gradient reaches $\phi$ through both $F$ and $B$. The encoder therefore receives two signals: the consistency loss, which makes $\phi$ predictable under the dynamics, and the FB loss, which makes it carry what is needed to represent successor measures. $F$, $B$ and $\pi$ are the only task-dependent machinery added to the world model, and none of it depends on a particular task.

This alone leaves a gap at test time. The planner applies $g$ and $F$ to imagined latents $\hat h_t$, but the FB loss only sees encodings of real states. The consistency loss pushes $\hat h_t$ towards $\bar\phi(s_t)$, and the target encoder tracks $\phi$, so imagined latents should lie near real encodings. That holds for the dataset's actions. For other actions nothing keeps the model's outputs among real encodings, and §3.6 is our guard against that.

### 3.4 Tying the dynamics to FB's reward basis

We add a second loss on the same rollout. The backward embedding of each imagined latent must match the target backward embedding $\bar B = \bar g \circ \bar\phi$ of the real state:

$$\mathcal{L}_{B} = \sum_{t=1}^{H} \lambda^{t}\, \big\| g(\hat h_t) - \bar B(s_t) \big\|^2 .$$

By §2, FB in effect solves the projection of a task reward onto the span of $B$, a reward of the form $B^\top w$. For any $w$,

$$\big| g(\hat h_t)^\top w - \bar B(s_t)^\top w \big| \le \|w\|\, \big\| g(\hat h_t) - \bar B(s_t) \big\|.$$

Averaged over dataset sub-trajectories, the squared reward error of the model at step $t$ is at most $\|w\|^2$ times the corresponding term of $\mathcal{L}_B$, for every $w$ at once. Two qualifications apply. The error is measured against the target basis $\bar B$, not against the true reward, and the constant $\|w\|$ depends on the task. Both losses only see the dataset's actions, so the bound holds for those actions. TD-MPC's reward-prediction loss does the same job for one reward. Here it is done for the whole basis, in the spirit of the value-equivalence principle [Grimm et al., 2020], which asks a model to be right about the quantities used for decisions. $\mathcal{L}_B$ also trains $g$ on imagined latents, which are its inputs during planning.

The training objective is

$$\mathcal{L} = \mathcal{L}_{\mathrm{FB}} + \alpha\, \mathcal{L}_{\mathrm{cons}} + \beta\, \mathcal{L}_{B},$$

plus the actor loss and FB's orthonormality penalty.

### 3.5 Task inference and planning

**Task inference.** Given $N$ labelled states $s_i \sim \rho$, the task vector is FB's Monte-Carlo estimate of $z_r = \mathbb{E}_\rho[B\, r]$,

$$z = \frac{1}{N} \sum_{i=1}^{N} B(s_i)\, r(s_i),$$

rescaled to norm $\sqrt d$ because $F$ and $\pi$ were trained on task vectors of that norm. The rescaled vector $cz$ with $c > 0$ is the task vector of the reward $c\,r$, which has the same optimal policy as $r$, and $F^\top(cz)$ is the Q-function of $c\,r$ (§2). From here on $z$ denotes the rescaled vector, and every term of the planning objective refers to $c\,r$.

**Per-step reward.** The planner also needs a reward on imagined latents. Write $\Sigma = \mathbb{E}_\rho[BB^\top]$. Using $g(\hat h)^\top z$ would be correct only if $\Sigma = I$. The orthonormality penalty pushes towards this but does not reach it. *(TODO: report how far $\Sigma$ is from $I$ in trained models.)* We use the $L^2(\rho)$ projection of $r$ onto the span of $B$, $\hat r(s) = B(s)^\top w$, evaluated on imagined latents as $g(\hat h)^\top w$, with

$$w = (\Sigma + \epsilon I)^{-1} z .$$

For this reward, up to the ridge $\epsilon$, $F^\top\, \mathbb{E}_\rho[B\, \hat r] = F^\top \Sigma\, w = F^\top z$. The per-step reward $B^\top w$ and the terminal value $F^\top z$ therefore describe the same reward, the least-squares approximation of $r$ in the span of $B$. With $B^\top z$ as the per-step reward they would describe different rewards whenever $\Sigma \ne I$. $\Sigma$ needs no labels: it is estimated once on unlabelled dataset states and used for every task. Since $\Sigma^{-1}$ amplifies $z$ along the directions where $\Sigma$ is small, $\|w\|$ can be large, which enlarges both the model's reward error (§3.4) and the effect of noise in $z$.

**Planning objective.** At each environment step, from $\hat h_0 = \phi(s)$, MPPI searches over a horizon $H$ (equal to the training horizon by default) for the sequence $a_{0:H-1}$ that maximises

$$J(a_{0:H-1}) = \sum_{t=0}^{H-1} \gamma^{t}\, g(\hat h_{t+1})^\top w \;+\; \gamma^{H}\, F\big(\hat h_H, \pi(\hat h_H, z), z\big)^\top z, \qquad \hat h_{t+1} = f(\hat h_t, a_t).$$

As in TD-MPC, some of MPPI's samples are rollouts of the actor, so the search starts from the actor's behaviour. The terminal value covers the return after the $H$ simulated steps.

### 3.6 Value check

The planner can exploit model errors, and the actor can be suboptimal for its own value function. We compare the planner's first action and the actor's with $F$ at the real current latent, which involves no model rollout, and execute

$$a = \arg\max_{a' \in \{a^{\mathrm{MPPI}}_0,\; \pi(\phi(s), z)\}} F(\phi(s), a', z)^\top z .$$

The check rejects a plan that looks good only through model error, provided $F$ ranks the two actions correctly. It cannot help when $F$ itself is wrong. Since the executed action maximises $F$ over two candidates, the check can improve on the actor only where the actor is not greedy for $F$. A model-free alternative could do the same by taking the best of many sampled actions under $F$, with no world model, so the gain of planning has to be measured against that control (§4, TODO: not run yet).

**Algorithm 1** (fbwm at test time).
```
input: N labelled states {(s_i, r_i)};  Σ = E_ρ[B Bᵀ] precomputed on unlabelled states
z ← normalize( mean_i B(s_i) r_i ),  w ← (Σ + εI)⁻¹ z
for each environment step, state s:
    h ← φ(s)
    a_plan ← MPPI over a_{0:H-1} maximising J  (rewards g(ĥ)ᵀw, terminal value Fᵀz, model f)
    a_pol  ← π(h, z)
    execute argmax over {a_plan, a_pol} of F(h, ·, z)ᵀz
```

Per task, the cost is one average over the labelled states and one $d \times d$ linear solve. Per environment step it is one MPPI search. Apart from the labelled states, the only data used at test time is $\Sigma$, computed once after training.
