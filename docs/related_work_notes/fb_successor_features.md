# FB / successor measures / successor features line of zero-shot RL (as of 2026-11-17)

Scope note: several arXiv IDs I first recalled from memory were wrong (2412.15109 is "Seer", not FB-CPR; 2404.18713 is a blimp paper, not FRE; 2102.12924 is "Visualizing MuZero"). All IDs below were checked against a fetched or search-result page. Venue tags marked "(memory)" were not confirmed by a page. Full-text checks used arXiv HTML pages, not PDFs.

## 1. Original FB, "Does zero-shot RL exist?", and benchmarks (ExORL, DMC, Jaco, antmaze)

### Takeaway
FB (Touati & Ollivier 2021) learns forward F(s,a,z) and backward B(s) so the successor measure factorises; the test-time task is z = E[r(s)B(s)] (or z = B(goal)), and the policy pi_z is read off with no planning. Touati-Rapin-Ollivier 2023 benchmarked FB and SF on ExORL/URLB-style DMC tasks and found FB reaches about 85% of supervised RL zero-shot with a good replay buffer.

### Cited Findings
- FB (Touati & Ollivier), arXiv 2103.07945, v1 Mar 2021, v3 Oct 2021 (NeurIPS 2021, memory; the page only says arXiv). Reward-free phase learns two representations by TD; at test time "a reward representation is estimated from observations or an explicit description, such as a target state. The policy is then derived directly from the representations, without planning." Tested on discrete/continuous mazes, pixel MsPacman, FetchReach — [arXiv](https://arxiv.org/abs/2103.07945)
- "Does Zero-Shot RL Exist?" Touati, Rapin, Ollivier, arXiv 2209.14935, ICLR 2023. Clarifies the SF vs FB relation, gives improved losses and new SF models, tests on the Unsupervised RL benchmark with offline replay buffers. SF depends heavily on the chosen features (Laplacian eigenfunctions work well); FB is best and most consistent; FB reaches "about 85% of supervised RL performance" with a good replay buffer — [arXiv](https://arxiv.org/abs/2209.14935)
- Jeen et al. (NeurIPS 2024): on ExORL, offline TD3 is the strongest single-task method and FB reaches about 80% of it; z is inferred as z ≈ E[r(s)B(s)] from reward-labelled states (up to 10,000 labels) — [arXiv HTML](https://arxiv.org/html/2309.15178v3)
- The FRE paper states that its FB and SF baselines use 5120 labelled samples with linear regression (following prior work), versus 32 for FRE — [arXiv HTML](https://arxiv.org/html/2402.17135v1)
- TD-JEPA reports an FB zero-shot DMC proprioceptive average of 648.2 — [arXiv HTML](https://arxiv.org/html/2510.00739)
- ExORL (Yarats et al.), arXiv 2201.13425, ICLR 2022 workshop: reward-free exploratory data, relabelled with the task reward, then offline RL. Offline TD3 on unsupervised Walker data beat the supervised baseline by 106% on average over 4 tasks — [arXiv](https://arxiv.org/abs/2201.13425v3), [mlanthology](https://mlanthology.org/iclrw/2022/yarats2022iclrw-don)
- Datasets in the FB line: ExORL Walker, Quadruped, Cheetah, Maze/Pointmass, Jaco (Jeen: 17 tasks over Walker/Quadruped/Maze/Jaco; also D4RL Walker2D/HalfCheetah, where plain FB fails) — [Jeen et al.](https://arxiv.org/html/2309.15178v3). antmaze-large-diverse and Kitchen appear in FRE — [FRE](https://arxiv.org/html/2402.17135v1). OGBench appears in TD-JEPA and ZOL.
- URLB (Laskin et al.), arXiv 2110.15191, no venue on the page (NeurIPS 2021 Datasets track, memory): 12 DMC continuous-control tasks in 3 domains, reward-free pretraining then extrinsic-reward adaptation, 8 baselines — [arXiv](https://arxiv.org/abs/2110.15191)

### Inferences
- "Labelled states -> z = E[r B]" is exactly the FB protocol. The standard label budgets are about 5k–10k for FB/SF baselines and 32 for FRE, so our n_labels should be reported against both.
- FB baselines in these papers are model-free (policy from F, no planning). Our MPPI in latent space is a distinct test-time mechanism.

### Gaps
- Per-domain FB numbers in Touati 2023 were not extracted. The NeurIPS 2021 venue for FB is not confirmed from a page.

## 2. Successor features / GPI line (Barreto, USFA, VISR, APS, URLB)

### Takeaway
SF assumes the reward is linear in features, r = phi·w, so Q^pi = psi^pi·w; GPI combines a finite policy set. VISR and APS learn the features without supervision (mutual-information skills), and the task is inferred by regressing w from brief reward exposure. FB generalises this to a full successor measure with a learned B.

### Cited Findings
- Barreto et al., "Successor Features for Transfer in RL", arXiv 1606.05312, NIPS 2017: SF plus generalized policy improvement, with guarantees before any new learning; tested on navigation and a simulated robot arm — [arXiv](https://arxiv.org/abs/1606.05312)
- Borsa et al., "Universal Successor Features Approximators" (USFA), arXiv 1812.07626 (2018; ICLR 2019, memory): combines UVFA, SF and GPI; tested on first-person 3D navigation — [arXiv](https://arxiv.org/abs/1812.07626)
- Hansen et al., "Fast Task Inference with Variational Intrinsic Successor Features" (VISR), arXiv 1906.05030, ICLR 2020: learns controllable features inside the SF framework; Atari with rewards revealed briefly after a long unsupervised phase; human-level on 14 games — [arXiv](https://arxiv.org/abs/1906.05030). The inference mechanism is not described on the abstract page.
- Liu & Abbeel, "APS: Active Pretraining with Successor Features", arXiv 2108.13956, ICML 2021 (PMLR 139): VISR-style SF plus APT nonparametric entropy; beats earlier pretrain+finetune methods on Atari 100k — [ICML](https://icml.cc/virtual/2021/poster/9241), [arXiv](https://arxiv.org/abs/2108.13956). APS and VISR adapt by finetuning or fast regression, so they are not strictly zero-shot.
- Wu, Tucker, Nachum, "The Laplacian in RL", arXiv 1810.04586 (ICLR 2019, memory; the page shows only arXiv): scalable Laplacian eigenvector approximation that helps goal-achieving tasks — [arXiv](https://arxiv.org/abs/1810.04586). Touati 2023 finds Laplacian-feature SF is the best SF variant — [arXiv](https://arxiv.org/abs/2209.14935)
- URLB (above) is the standard testbed for VISR/APS-type skill methods.

### Inferences
- SF, VISR and APS need the reward to be linear in phi; B in FB removes that restriction in principle. In FRE's table FB beats SF (overall 24 vs 18 normalised) — [FRE](https://arxiv.org/html/2402.17135v1)

### Gaps
- VISR's task-inference details (abstract only). APS numbers on URLB zero-shot were not retrieved.

## 3. Recent FB improvements and neighbours (2023–2026)

### Takeaway
There are three directions. (a) Fix FB failure modes: low-quality data, partial observability, dynamics shift, linear task encoding. (b) Scale FB to humanoids with an imitation regulariser (FB-CPR, BFM-Zero). (c) Use alternative zero-shot bases (HILP, FRE, Proto SM, TD-JEPA, TD-Flow). Only TD-JEPA is close to our latent-predictive setting, and it still uses a policy, not planning.

### Cited Findings
- **FB-CPR / Meta Motivo**: Tirinzoni, Touati, Farebrother et al., "Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models", arXiv 2504.11054, ICLR 2025 (earlier NeurIPS 2024 workshop). FB with a conditional-policy regulariser toward unlabelled mocap; online training on observation-only mocap. Test time: reward z_r ≈ (1/n) Σ r(s_i) B(s_i); goal z = B(s); tracking z = mean of B over demo states. Reward tasks: about 61% of TD3 (retrained per task, topline); about 177% of Diffuser; oracle MPPI about 17.8% higher but a 300-step episode takes at least 30 min vs about 12 s for FB-CPR. Goal reaching is comparable to Goal-GAIL/Goal-TD3; tracking about 83% success. Limitations: proprioception only, needs costly mocap — [arXiv HTML](https://arxiv.org/html/2504.11054v1), [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37611b0fc2b65cdbb60865af5f6cf453-Abstract-Conference.html)
- **BFM-Zero**: Li, Luo, ... Tirinzoni, Touati, Lazaric, Pirotta, Shi, arXiv 2511.04131 (Nov 2025; v2 Oct 2026), ICLR 2026. FB-based promptable humanoid model on a real Unitree G1; tracking, goal reaching, reward optimisation and few-shot adaptation — [arXiv](https://arxiv.org/abs/2511.04131)
- **TD-JEPA**: Bagatella, Pirotta, Touati, Lazaric, Tirinzoni (FAIR/Meta), arXiv 2510.00739, ICLR 2026 (oral per the ICLR listing). A TD-learned latent-predictive representation with a state encoder phi, a task encoder psi, a policy-conditioned multi-step latent predictor T(phi(s),a,z), and policies. Test time: no planning, no finetuning; z_r = E[psi psi^T]^-1 E[psi(s) r(s)] (least squares on labelled (s,r) samples), then act with pi_{z_r}. Table 1: DMC proprio 661.2±6.3 (FB 648.2, BYOL-γ 645.4, HILP 620.1); DMC pixels 628.8±5.5 (BYOL-γ 582.4, FB 456.2, HILP 391.2); OGBench proprio 37.98±0.77 (FB 39.04, HILP 37.98); OGBench pixels 41.34±0.45 (BYOL-γ 41.58, FB 39.89). Per-domain proprio: walker 785.2 (FB 811.5), cheetah 688.7 (FB 672.7), quadruped 691.4 (ICVF 708.5), pointmass 479.3 (FB 513.0). OGBench uses BC regularisation; antmaze-me 0.20 vs FB 51.60 — [arXiv HTML](https://arxiv.org/html/2510.00739), [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html)
- **HILP**: Park, Kreiman, Levine, "Foundation Policies with Hilbert Representations", arXiv 2402.15567, ICML 2024. A Hilbert (temporal-distance-preserving) latent; zero-shot "prompting" by directional movements in latent space; goal-conditioned and general RL tasks — [arXiv](https://arxiv.org/abs/2402.15567). The task-inference mechanism was not extracted (gap).
- **FRE**: Frans, Park, Abbeel, Levine, "Unsupervised Zero-Shot RL via Functional Reward Encodings", arXiv 2402.17135, ICML 2024 (PMLR 235). A transformer VAE encodes a set of (state, reward) samples into a latent; the policy is conditioned on it and pretrained on random unsupervised reward functions. It uses 32 labelled samples (FB/SF baselines get 5120). Table 1 (normalised 0–100), FRE / FB / SF: AntMaze 52.8 / 25.8 / 11.8; ExORL 51.5 / 43.4 / 40.9; Kitchen 66 / 3 / 1; overall 57 / 24 / 18. FB is better on ExORL walker-velocity (64 vs 34) and cheetah-velocity (51 vs 20) — [arXiv HTML](https://arxiv.org/html/2402.17135v1), [PMLR](https://proceedings.mlr.press/v235/frans24a.html)
- **Jeen et al., "Zero-Shot RL from Low Quality Data"**, arXiv 2309.15178, NeurIPS 2024. Conservative FB variants: VC-FB (value-conservative) and MC-FB (measure-conservative). On 100k-transition ExORL data MC-FB reaches about 150% and VC-FB about 137% of FB's IQM, and about 111% / 120% of CQL; on RND data they reach 1.8x and 2.5x FB; on full datasets they are no worse. Plain FB fails on D4RL Walker2D/HalfCheetah — [arXiv HTML](https://arxiv.org/html/2309.15178v3), [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e38b2a0b77541b14a3315c99697b835-Abstract.html)
- **Jeen et al., "Zero-Shot RL Under Partial Observability"**, arXiv 2506.15446, RLC 2025. Memory-based BFMs on modified ExORL; memory-free FB degrades under partial observability (fine for interpolation, poor for extrapolation) — [arXiv HTML](https://arxiv.org/html/2506.15446v1)
- **Cetin, Touati, Ollivier, "Finer BFMs via Auto-Regressive Features and Advantage Weighting" (FB-AWARE)**, arXiv 2412.04368, RLJ 2025 (vol 6). Autoregressive task features that go beyond a linear task encoding, plus advantage-weighted policy extraction. It reportedly matches single-task offline RL on D4RL locomotion (from a secondary summary; not verified in the paper) — [arXiv HTML](https://arxiv.org/html/2412.04368v1), [RLJ](https://rlj.cs.umass.edu/2025/papers/Paper181.html)
- **Sikchi et al., "Fast Adaptation with BFMs"**, arXiv 2504.07896, RLC/RLJ 2025. Actor-critic and actor-only search in task-embedding space (ReLA, LoLA); 10–40% over zero-shot within a few tens of online episodes — [arXiv](https://arxiv.org/abs/2504.07896v1). Not zero-shot, since it needs online episodes.
- **Pirotta et al., "Fast Imitation via BFMs"**, ICLR 2024 spotlight (Meta): imitation from a few demonstrations via successor-measure foundation models, no RL or finetuning, seconds instead of hours — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/370645ae400005f888e418f46b594539-Abstract-Conference.html). The arXiv ID was not retrieved.
- **Farebrother et al., "Temporal Difference Flows"**, arXiv 2503.09817 (Mar 2025; ICML 2025, memory). Generative flow-matching geometric horizon models with more than 5x the horizon of prior methods; combining with BFMs for planning over pretrained policies gives substantial gains — [arXiv](https://arxiv.org/abs/2503.09817)
- **Agarwal, Sikchi, Stone, Zhang, "Proto Successor Measure"**, arXiv 2411.19418, listed "under submission". Policy-independent basis of visitation distributions; a new reward only requires finding linear weights — [arXiv](https://arxiv.org/abs/2411.19418)
- **Bobrin et al., "Zero-Shot Adaptation of BFMs to Unseen Dynamics"**, arXiv 2505.13150, ICLR 2026. FB breaks under test-time dynamics change; a transformer belief estimator gives up to 2x zero-shot returns (from a search summary; not verified in the paper) — [ICLR 2026 listing](https://iclr.cc/virtual/2026/poster/10008433)
- **ZOL, "Zero-Shot Off-Policy Learning"**, Asadulaev et al., arXiv 2602.01962 (v2 dated 30 May 2026). Frozen FB; z is initialised from labelled samples, then refined by gradient ascent on an importance-weighted return (successor measure linked to stationary density ratios), with a chi-square penalty and a latent trust region; no planning and no extra environment interaction. ExORL DMC (16 tasks) average return: ZOL 627.2, LoLA 616.9, FB 592.5, ReLA 585.8, TD-JEPA 553.2, HILP 538.5. OGBench: 0.47 vs FB 0.45. Humanoid tracking vs FB-CPR is mixed — [arXiv HTML](https://arxiv.org/html/2602.01962v2). Its FB number (592.5) differs from TD-JEPA's own (648.2), so the protocols differ.

### Inferences
- In my searches no FB-family paper does MPPI planning in a learned latent model coupled to F/B. FB-CPR only benchmarks an oracle MPPI (17.8% higher, but at least 30 min vs about 12 s per episode). That is the precedent for "planning beats policy, at higher cost".
- The closest works to our setup are TD-JEPA (latent-predictive model plus task regression, but policy-based, no MPPI) and TD-Flow (planning over BFM policies with a generative model). ZOL is the closest test-time refinement alternative to planning.
- Cross-paper numbers are not comparable (FB is 648 in TD-JEPA vs 592 in ZOL on DMC); compare only within a paper.

### Gaps
- No search hit for 2025–26 "FB scaling" papers or other ICML/NeurIPS 2025–26 zero-shot RL work beyond those listed; 2026 coverage is thin. A search surfaced "Planning with Latent Dynamics Models" (PLDM; reward-free latent model plus planning, HILP as a baseline) but I did not read it; it may be relevant, unverified.
- Sprekeler graph-Laplacian zero-shot RL: no paper found; it probably refers to Wu 2019 or Touati 2023 Laplacian features. Unverified.
- Per-domain numbers for HILP, FB-AWARE and Proto SM were not retrieved.

## 4. Task inference, label needs, and relation to FB + latent world model + MPPI

### Takeaway
All FB-family methods infer z from reward-labelled states by a weighted average (z = E[r B]) or least squares (TD-JEPA). They need thousands of labels (FB/SF) versus 32 for FRE, and none of the zero-shot ones plans. Our method differs by using the FB-derived z inside latent MPPI at test time.

### Cited Findings
- Labelled states, not the full reward function, are needed by FB (z = E[rB] over samples), FB-CPR, TD-JEPA (regression on D_rwd), FRE (32 samples), ZOL (initial z from samples, then refinement), and Jeen (up to 10k labels). Proto SM needs a "new reward function" to solve for linear weights; whether that means samples or the function was not clarified. Sources as above.
- FB-CPR: oracle MPPI planning beats FB-CPR by about 17.8% but takes at least 30 min per 300-step episode vs about 12 s — [arXiv HTML](https://arxiv.org/html/2504.11054v1)
- Test-time improvements other than planning: gradient refinement of z (ZOL), online RL in z-space (ReLA/LoLA), GPI over policies with TD-Flow models — see section 3.

### Inferences
- Positioning: our contribution is test-time planning with a learned latent model sharing the encoder with FB. The closest baselines to report against are FB, HILP, TD-JEPA and ZOL on ExORL cheetah/walker/quadruped, with FB-CPR's MPPI oracle as the precedent.
- TD-JEPA's per-domain proprio cheetah (688.7) vs FB (672.7) is a useful anchor for the cheetah gap analysis, but its label budget and protocol should be checked first.

### Gaps
- The FB baseline protocol (labels, seeds, steps) in each paper was not compared. The second half of the TD-JEPA HTML (offset 100000, appendix) was not read.
