# Adjacent methods and theory for zero-shot RL (as of 2026-11-17)

Convention: "[verified]" = confirmed in this session's search results (abstract/listing level; full PDFs mostly not opened). "[memory, unverified]" = recalled from training knowledge, not checked; verify before citing.

## 1. Offline goal-conditioned RL and how reward-over-states relates to goal reaching

### Takeaway
Goal-conditioned methods (HIQL, CRL, QRL, HILP) learn a value/policy indexed by a goal state, i.e. rewards of the form "reach s*". FB/TD-JEPA-style methods and FRE target arbitrary rewards r(s), which strictly contains goal reaching. HILP is the bridge: it claims zero-shot on both goal and reward tasks.

### Cited Findings
- HIQL (Park, Ghosh, Eysenbach, Levine), arXiv 2307.11949, NeurIPS 2023: one action-free value function trains a high-level policy (states/latent subgoals as actions) and a low-level policy; targets long-horizon goal reaching and uses action-free data — [arXiv v4 listing](https://arxiv.org/abs/2307.11949v4), [summary](https://mlanthology.org/neurips/2023/park2023neurips-hiql)
- Contrastive RL (Eysenbach et al.), arXiv 2206.07568, NeurIPS 2022: inner product of learned representations corresponds to a goal-conditioned value function; works offline and from images — [arXiv](https://arxiv.org/abs/2206.07568v2), [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract.html)
- OGBench (Park, Frans, Eysenbach, Levine), arXiv 2410.20092, ICLR 2025: 8 environment types, 85 datasets, 6 reference algorithms; tests stitching, long-horizon reasoning, high-dim input, stochasticity. Which 6 algorithms (HIQL/CRL/QRL etc.) was not confirmed from the primary text — [arXiv](https://arxiv.org/abs/2410.20092v2), [ICLR 2025 listing](https://mlanthology.org/iclr/2025/park2025iclr-ogbench)
- HILP, "Foundation Policies with Hilbert Representations" (Park, Kreiman, Levine), arXiv 2402.15567, ICML 2024 (PMLR 235): embeds states in a Hilbert space whose distances track temporal distance, trains a latent-conditioned policy moving along directions; reports zero-shot on goal-conditioned and reward-based tasks, plus test-time planning — [PMLR](https://proceedings.mlr.press/v235/park24g.html), [arXiv](https://arxiv.org/abs/2402.15567)
- TD-JEPA reports results on 13 datasets in ExoRL and OGBench (the latter is a goal-conditioned benchmark used for zero-shot reward tasks) — [arXiv 2510.00739](https://arxiv.org/pdf/2510.00739)
- One-step FB (Zheng, Jayanth, Eysenbach) notes a third-party-summarized conjecture that sparse goal rewards may collapse diversity of backward representations on OGBench state tasks (a conjecture, not tested) — [arXiv 2602.11399](https://arxiv.org/abs/2602.11399), [summary](https://www.emergentmind.com/open-problems/sparse-rewards-single-backward-representation-ogbench)
- QRL (Wang, Torralba, Isola, Zhang), "Optimal Goal-Reaching RL via Quasimetric Learning", arXiv 2304.01203, ICML 2023 — [arXiv abs](https://arxiv.org/abs/2304.01203)
- Blier, Tallec, Ollivier, "Learning Successor States and Goal-Dependent Values: A Mathematical Viewpoint", arXiv 2101.07123 (Jan 2021); arXiv lists no venue — [arXiv abs](https://arxiv.org/abs/2101.07123)

### Inferences
- Statement for the paper: a goal-conditioned policy pi(a|s,g) solves reward r_g(s)=1[s=g] (or a distance-based one); it cannot represent an arbitrary reward (e.g. "run at speed v", "avoid region A while reaching B") without a separate mapping from reward samples to a goal/latent. FB, FRE and TD-JEPA provide that mapping (regression of B or encoder); goal methods do not. Conversely, on pure goal reaching dedicated GCRL (HIQL) tends to be strong and FRE reports matching goal-specific RL on goal tasks (section 2). Position GCRL as the strong specialised baseline and goal reaching as a special case of our setting.
- With FB, goal tasks reduce to z = B(goal), a clean sanity check to report.

### Gaps
- Exact algorithm list of OGBench and numeric comparisons GCRL vs FB were not retrieved; no paper found that directly argues "goal-conditioned cannot do arbitrary rewards" beyond HILP/FRE framing.

## 2. Multi-task / reward-conditioned offline RL, in-context RL, behavior foundation models

### Takeaway
FRE is the closest "in-context reward" competitor (encode a handful of (state, reward) samples, no gradient steps). Sequence-model approaches (Prompt-DT, AD) condition on demonstrations or learning histories rather than reward labels over states. BFMs (FB-CPR/Meta Motivo, BFM-Zero) are FB scaled to humanoids.

### Cited Findings
- FRE (Frans, Park, Abbeel, Levine), arXiv 2402.17135, ICML 2024: transformer VAE encodes samples of (state, reward) pairs into a latent; policy pre-trained on random unsupervised reward functions (mixture of goal-reaching and random MLP rewards); novel task = encode labelled samples, no further training; does not need linear reward structure (vs SF) nor linearised value (vs FB); gains largest on goal-reaching — [PMLR](https://proceedings.mlr.press/v235/frans24a.html), [arXiv](https://arxiv.org/pdf/2402.17135)
- Prompt-DT (Xu et al.), arXiv 2206.13499, ICML 2022: trajectory prompts from few-shot demonstrations adapt a Transformer offline RL agent without finetuning (MuJoCo) — [arXiv](https://arxiv.org/abs/2206.13499v1)
- Algorithm Distillation (Laskin et al., DeepMind), arXiv 2210.14215 (ICLR 2023 [venue memory, unverified]): causal transformer trained on learning histories of a source RL algorithm improves in-context without weight updates — [arXiv](https://arxiv.org/abs/2210.14215)
- FB-CPR / Meta Motivo: "Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models" (Tirinzoni, Touati, Farebrother, Guzek, Kanervisto, Xu, Lazaric, Pirotta), arXiv 2504.11054, ICLR 2025: FB regularised toward imitating unlabeled mocap behaviours; prompts for tracking, goal reaching, reward optimisation; limits: proprioceptive only, no object interaction, needs mocap — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37611b0fc2b65cdbb60865af5f6cf453-Abstract-Conference.html), [arXiv](https://arxiv.org/pdf/2504.11054)
- BFM-Zero, arXiv 2511.04131 (Nov 2025): FB-style unsupervised RL on a real Unitree G1 humanoid; zero-shot reward optimisation, single-frame goal reaching, tracking — [arXiv](https://arxiv.org/pdf/2511.04131)
- Finer BFMs (Cetin, Touati, Ollivier), arXiv 2412.04368, RLC 2025: auto-regressive features and advantage weighting to address FB limitations — [RLJ](https://rlj.cs.umass.edu/2025/papers/Paper181.html), [arXiv](https://arxiv.org/html/2412.04368v1)
- Proto Successor Measure (Agarwal, Sikchi, Stone, Zhang), arXiv 2411.19418: basis set representing any RL solution; zero-shot for any reward; arXiv page lists "under submission", ICML 2025 acceptance NOT confirmed — [arXiv](https://arxiv.org/abs/2411.19418)
- Not verified in this session (from memory): Decision Transformer arXiv 2106.01345 (NeurIPS 2021); HER, Andrychowicz et al., arXiv 1707.01495 (NeurIPS 2017); UDS / "How to Leverage Unlabeled Data in Offline RL", Yu et al., ICML 2022 (reward relabeling, zero reward on unlabeled data); Gato / RT-X (not relevant, omit or one line).

### Inferences
- Taxonomy for the paper: (a) reward-free pretraining + test-time inference of a task embedding from labelled states (FB, SF/VISR, FRE, HILP, TD-JEPA, PSM, ours); (b) sequence-model in-context (Prompt-DT, AD, DT) which condition on trajectories/returns, not state-reward labels, and are not trained reward-free; (c) hindsight relabeling (HER, UDS) which relabels data for fixed goal/reward families during training, not at test time.
- Ours differs by planning in a learned latent world model (MPPI) rather than a z-conditioned policy.

### Gaps
- UDS/HER/DT details not verified. Meta Motivo arXiv version/title differs (FB-CPR preprint vs ICLR title); cite the ICLR 2025 version.

## 3. Few-shot adaptation baselines and cost comparisons

### Takeaway
Linear probing on frozen features is exactly what SF/FB do at test time (regression), so it is the natural "zero-shot inference" mechanism; finetuning / reward-model + offline RL are the expensive alternatives. I found no paper reporting a wall-clock comparison of zero-shot inference vs finetuning in our setting.

### Cited Findings
- VISR (Hansen, Dabney, Barreto et al.), arXiv 1906.05030, ICLR 2020 (one aggregator lists 2019; the arXiv record says ICLR 2020): fast task inference with variational intrinsic successor features; Atari, human-level on 14 games — [arXiv](https://arxiv.org/pdf/1906.05030)
- TD-JEPA obtains the task vector from a small set of rewarded samples by linear regression, no further environment interaction (their Sec 3.3) — [arXiv](https://arxiv.org/pdf/2510.00739)
- Jeen et al. NeurIPS 2024 compare against a task-specific baseline that sees reward labels in advance and report conservative FB sometimes beats it — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e38b2a0b77541b14a3315c99697b835-Abstract.html)
- Touati et al. ICLR 2023 report FB reaches about 85% of supervised RL performance zero-shot with a good replay buffer on the URL benchmark — [arXiv 2209.14935](https://arxiv.org/abs/2209.14935v2)
- Barreto et al., "Successor Features for Transfer in RL", arXiv 1606.05312 (v1 June 2016, v2 Apr 2018), NIPS 2017 — [arXiv abs](https://arxiv.org/abs/1606.05312)
- Not verified (memory): Barreto et al. 2018 SF+GPI (ICML 2018, arXiv 1901.10964), Borsa et al. 2019 USFA (arXiv 1812.07626); MOPO (arXiv 2005.13239), IQL (arXiv 2110.06169), TD3+BC (arXiv 2106.06860) as standard model-based / offline RL finetune baselines.

### Inferences
- A defensible C4-style comparison: same N labelled states, (i) linear regression of z (zero-shot, seconds), (ii) reward-model fit + offline TD3/IQL on relabeled data (minutes to hours), (iii) frozen world model + reward head + planning/Q adaptation (our tdmpc_adapt). Report reward_fit_time/adapt_time. Papers rarely report this, so it can be a contribution.

### Gaps
- No published numbers found on finetuning vs zero-shot compute/time. Not searched: whether FB-CPR / Meta Motivo report finetuning of z or policy.

## 4. Theory, guarantees and failure modes of FB / successor measures

### Takeaway
FB gives near-optimal policies for any reward only under a low-rank successor-measure assumption and exact optimisation; 2026 work shows existence conditions for exact FB are stringent and proposes fixes. Known failure modes: low-quality or narrow data (value overestimation), partial observability, collapse, off-policy shift at test time.

### Cited Findings
- Touati & Ollivier, "Learning One Representation to Optimize All Rewards", arXiv 2103.07945, NeurIPS 2021: FB representation of a reward-free MDP yields near-optimal policies for any later reward without planning — [arXiv](https://arxiv.org/abs/2103.07945v1)
- Touati, Rapin, Ollivier, "Does Zero-Shot RL Exist?", arXiv 2209.14935, ICLR 2023 (oral): compares SF vs FB; FB best and consistent — [arXiv](https://arxiv.org/abs/2209.14935v2)
- Jin, Krishnamurthy, Simchowitz, Yu, "Reward-Free Exploration for RL", arXiv 2002.02794, ICML 2020: explore without reward, then plan for any reward; sample complexity a factor S above standard RL, with a nearly matching Omega(S^2 A H^2 / eps^2) lower bound (upper-bound formula not verified against the PDF) — [PMLR](https://proceedings.mlr.press/v119/jin20d.html), [arXiv](https://arxiv.org/abs/2002.02794v1)
- Jeen, Bewley, Cullen, "Zero-Shot RL from Low Quality Data" (VC-FB, MC-FB), arXiv 2309.15178, NeurIPS 2024: FB/SF overestimate OOD values on small homogeneous data; conservatism fixes it. A catalog listing quotes about 150% of vanilla FB in aggregate (unverified) — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e38b2a0b77541b14a3315c99697b835-Abstract.html), [arXiv](https://arxiv.org/abs/2309.15178), [CatalyzeX](https://catalyzex.com/author/Scott%20Jeen)
- Jeen, Bewley, Cullen, "Zero-Shot RL Under Partial Observability", arXiv 2506.15446, RLC 2025 (RLJ vol. 6): FB/SF degrade under partial observability; memory models over histories help; two failure modes (state and task misidentification, per a secondary review) — [arXiv](https://arxiv.org/pdf/2506.15446), [RLJ](https://rlj.cs.umass.edu/2025/papers/Paper245.html)
- TD-JEPA theory: idealised variant avoids collapse under proper initialisation; encoders capture a low-rank factorisation of long-term policy dynamics — [arXiv](https://arxiv.org/pdf/2510.00739), [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html)
- Zheng, Jayanth, Eysenbach, "Can We Really Learn One Representation to Optimize All Rewards?", arXiv 2602.11399 (Feb 2026, ICML 2026 listing): ground-truth FB requires stringent rank and dimensionality conditions; one-step FB learns representations of a fixed behaviour policy then does one policy-improvement step; reports 10^5 times smaller representation error and +24% zero-shot on average over 10 domains — [arXiv](https://arxiv.org/abs/2602.11399), [HTML](https://arxiv.org/html/2602.11399)
- Di Ventura, Kleuker, Plaat, Moerland, "A Unified Framework for Zero-Shot RL", arXiv 2510.20542 (v1 Oct 2025, v2 Mar 2026): taxonomy direct vs compositional representations, extended SF bound — [arXiv](https://arxiv.org/abs/2510.20542v1). The "linearization / inference / approximation" error split came from a secondary summary; check the paper.
- Temporal Difference Flows (Farebrother, Pirotta, Tirinzoni, Munos, Lazaric, Touati), arXiv 2503.09817, ICML 2025 (PMLR 267): geometric horizon models via flow matching, about 5x longer horizon than prior; generative-successor-measure alternative to FB — [PMLR](https://proceedings.mlr.press/v267/farebrother25a.html)
- Dayan 1993 (successor representation), Blier et al. 2021, Barreto 2017/2018, Borsa 2019: standard SF lineage [memory, unverified].

### Inferences
- Failure-mode story for the paper: data quality (Jeen 2024), partial observability (Jeen 2025), FB existence conditions (Zheng 2026), off-policy shift at test time (ZOL 2026, section 5). Only claim our method addresses these if our own runs show it.

### Gaps
- No zero-shot impossibility / low-rank-MDP guarantee paper found beyond Touati 2023 and Jin 2020. No dedicated "FB dimensional collapse" analysis confirmed; only TD-JEPA's collapse argument.

## 5. Latest 2025-2026 papers to cite

### Takeaway
Must-cites: TD-JEPA (ICLR 2026 oral), One-step FB, ZOL, Soft FB, Switching Successor Measures, BFM-Zero, Finer BFMs, PSM, TD-Flow, Jeen 2025, unified framework. Citing-works lists (Google Scholar "cited by") could not be browsed; coverage is from keyword searches only.

### Cited Findings
- TD-JEPA (Bagatella, Pirotta, Touati, Lazaric, Tirinzoni), arXiv 2510.00739, ICLR 2026 (oral listing): state/task encoders, policy-conditioned multi-step predictor, parameterised policies, all in latent space; 13 datasets ExoRL+OGBench, including pixels — [arXiv](https://arxiv.org/pdf/2510.00739), [ICLR oral listing](https://iclr.cc/virtual/2026/oral/10009367)
- Zero-Shot Off-Policy Learning (ZOL), arXiv 2602.01962 (Feb 2026): links successor measures to stationary density ratios; behaviour-supported update to avoid OOD queries and overestimation at test time; SMPL Humanoid, ExoRL, OGBench — [arXiv](https://arxiv.org/abs/2602.01962)
- Soft Forward-Backward Representations, arXiv 2602.06769 (Feb 2026): FB for general utilities of the occupancy measure; max-entropy RL for richer stochastic policies — [arXiv](https://www.arxiv.org/pdf/2602.06769)
- Switching Successor Measures for Hierarchical Zero-shot RL (Stojanovic, Proutiere), arXiv 2605.13207, submitted 13 May 2026: FB pi-Switch derives high-level subgoal policy and low-level controller from one FB representation; beats non-hierarchical baselines on reward tasks, on par with SOTA hierarchical on goal tasks — [arXiv abs](https://arxiv.org/abs/2605.13207)
- "Zero-shot adaptation of BFMs to unseen dynamics" (FB with belief over dynamics): only a third-party review seen; arXiv ID and authors not found — [review](https://www.themoonlight.io/review/zero-shot-adaptation-of-behavioral-foundation-models-to-unseen-dynamics)

### Inferences
- TD-JEPA is the nearest prior work to our JEPA+FB; the paper must state differences (planning with a learned latent world model vs z-conditioned policy; our consistency/routing variants). Cross-check with other researchers' notes.

### Gaps
- Not reached: Google Scholar/Semantic Scholar citing lists of Touati 2023, Frans 2024, Bagatella 2025; targeted "task inference from labelled states" search; other 2026 papers (NeurIPS 2026 / ICLR 2027 OpenReview submissions). Do one more pass on OpenReview before submission.
