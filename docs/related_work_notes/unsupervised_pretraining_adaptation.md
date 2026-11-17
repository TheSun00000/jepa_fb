# Non-FB unsupervised / reward-free pretraining with fast reward adaptation

Convention: "[V]" = verified this session by fetching the source. "[M]" = from memory / search snippet only, NOT fetched; arXiv IDs recalled from memory, check before citing. No numbers are given unless marked [V].

## 1. Skill discovery and how a skill is chosen at test time

### Takeaway
Classic MI skill discovery (DIAYN, VALOR, DADS, APS, CIC) is pretrained reward-free; the downstream reward is handled by sweeping/selecting skills (needs online rollouts) or by regressing reward on features (APS, HILP). Newer metric/representation methods (LSD, METRA, CSF, HILP) support zero-shot goal reaching or linear-regression reward inference, the closest non-FB analogues to FB's z inference.

### Cited Findings
- METRA (Park, Rybkin, Levine; arXiv 2310.08887; ICLR 2024) [V]: covers a compact latent space Z metrically tied to states by temporal distances; claims first unsupervised RL to discover locomotion in pixel Quadruped and Humanoid; five locomotion/manipulation envs. Abstract does not describe test-time skill selection. — [arXiv](https://arxiv.org/abs/2310.08887)
- CSF, "Can a MISL Fly?" (Zheng, Tuyls, Peng, Eysenbach; arXiv 2412.08021; ICLR 2025 oral) [V]: contrastive successor features, matches METRA with fewer components (no dual gradient descent / Wasserstein). — [arXiv](https://arxiv.org/abs/2412.08021). Zero-shot goal reaching sets skill to normalized difference of goal and current-state representations, same as METRA; close to METRA, better on Ant, worse on Quadruped/Kitchen/Humanoid; metric is "staying time fraction" (search-snippet level). — [PDF](https://arxiv.org/pdf/2412.08021)
- HILP, "Foundation Policies with Hilbert Representations" (Park, Kreiman, Levine; arXiv 2402.15567; ICML 2024) [V]: distance-preserving state embedding phi; skill policy pi(a|s,z) trained with inner-product intrinsic reward on phi(s')-phi(s). Test-time reward: z* = E[phi~ phi~^T]^-1 E[r phi~], phi~ = phi(s')-phi(s), linear regression on a small number of dataset transitions, no training. Benchmarks: URLB/ExORL (Walker, Cheetah, Quadruped, Jaco; APS/APT/Proto/RND datasets; pixels too), D4RL AntMaze (large, ultra) and Kitchen, hierarchical RL vs OPAL. Baselines FB, six SF variants (FDM, Laplacian, AE, IDM, random, CL), GC-TD3. Best IQM on ExORL among nine methods. Goal-conditioned Table 1: HILP-Plan 64.5 antmaze-large-diverse and 59.2 antmaze-ultra-diverse vs 56.0 / 40.8 best GC-IQL; FB 0.0 on AntMaze. Hierarchical average 51.3 vs OPAL 45.7. (Read only first 100k of 139k chars.) — [arXiv HTML](https://arxiv.org/html/2402.15567v2)
- APS, "Active Pretraining with Successor Features" (Liu & Abbeel; arXiv 2108.13956; ICML 2021) [V abstract; NOTE: an earlier draft of these notes had the wrong ID 2102.12271]: reward-free phase maximises MI between tasks and visited states via variational successor features plus non-parametric entropy; fine-tuned afterwards (abstract reports Atari 100k). Test-time reward regression onto features is from memory [M]. — [arXiv](https://arxiv.org/abs/2108.13956)
- CIC (Laskin, Liu, Peng, Yarats, Rajeswaran, Abbeel; arXiv 2202.00161, 2022) [V]: contrastive MI between transitions and skills, entropy of embeddings as intrinsic reward; evaluated on URLB; reports 1.79x better adaptation than prior skill-discovery methods and 1.18x over the next-best exploration algorithm (abstract). — [arXiv](https://arxiv.org/abs/2202.00161)
- LSD, "Lipschitz-constrained Unsupervised Skill Discovery" (Park, Choi, Kim, Lee, Kim; arXiv 2202.00914; ICLR 2022) [V]: learned representation supports zero-shot goal following without extra training or complex planning; seven downstream tasks incl. multi-goal following on Humanoid. — [arXiv](https://arxiv.org/abs/2202.00914)
- DIAYN (Eysenbach et al., arXiv 1802.06070, ICLR 2019), VALOR (Achiam et al., arXiv 1807.10299), DADS (Sharma et al., arXiv 1907.01657, ICLR 2020; dynamics-aware, enables model-predictive control over skills, a planning-on-top precedent) [M, IDs unverified, no URL fetched]. Test-time use: DIAYN/VALOR/DADS typically try/finetune skills or train a high-level policy. [M]

### Inferences
- HILP and APS are the nearest non-FB analogues of our reward-labelled-state inference (linear regression onto features), but they choose z, then run a latent-conditioned policy; no planning. HILP-Plan (planning in the latent) exists on goal-conditioned tasks, closer to MPPI in latent space.
- HILP reports FB failing (0.0) on AntMaze; relevant to our antmaze-large-diverse env as a baseline caveat.
- METRA/CSF are designed for goal reaching, not arbitrary dense reward; limited as baselines for reward-labelled tasks.

### Gaps
- METRA/CSF abstracts give no test-time reward-adaptation protocol; nothing fetched on DIAYN/VALOR/DADS (IDs from memory); APS, CIC, LSD checked at abstract level only.

## 2. Unsupervised pretraining with exploration; benchmarks; theory

### Takeaway
URLB and ExORL define the reward-free pretrain then reward-adapt protocol our datasets come from; ExORL shows plain off-policy RL on exploratory data is competitive offline.

### Cited Findings
- URLB (Laskin, Yarats, Liu, Lee, Zhan, Lu, Cang, Pinto, Abbeel; arXiv 2110.15191, Oct 2021) [V; arXiv page lists no peer-reviewed venue, NeurIPS 2021 Datasets track from memory, unverified]: reward-free pretraining then adaptation with extrinsic reward; 12 continuous control tasks in 3 DMC domains; code for 8 unsupervised methods; baselines "make progress but do not solve" it. — [arXiv](https://arxiv.org/abs/2110.15191)
- ExORL, "Don't Change the Algorithm, Change the Data" (Yarats, Brandfonbrener, Liu, Laskin, Abbeel, Lazaric, Pinto; arXiv 2201.13425, 2022; ICLR 2022 from memory, unverified) [V]: pipeline = unsupervised reward-free exploration data, relabel with downstream reward, offline RL; vanilla off-policy RL (no offline-specific changes) matches or beats offline RL methods when data are diverse; datasets at github.com/denisyarats/exorl. — [arXiv](https://arxiv.org/abs/2201.13425)
- APT, "Behavior From the Void" (Liu & Abbeel; arXiv 2103.04551; NeurIPS 2021 spotlight) [V]: reward-free exploration by non-parametric entropy in a representation space, then task reward exposed; DMC and Atari. — [arXiv](https://arxiv.org/abs/2103.04551)
- Proto-RL (Yarats, Fergus, Lazaric, Pinto; arXiv 2102.11271; ICML 2021) [V]: prototypes tie representation learning with exploration; pretrained without task information. — [arXiv](https://arxiv.org/abs/2102.11271)
- Plan2Explore (Sekar, Rybkin, Daniilidis, Abbeel, Hafner, Pathak; arXiv 2005.05960; ICML 2020) [V]: plans to seek expected novelty with a self-supervised world model, then adapts zero- or few-shot to downstream tasks unknown during exploration; nearly matches an oracle with rewards (image control). Closest non-FB conceptual match to world model + planning. — [arXiv](https://arxiv.org/abs/2005.05960)
- LEXA (Mendonca, Rybkin, Daniilidis, Hafner, Pathak; arXiv 2110.09514; NeurIPS 2021) [V]: world-model-based explorer and achiever trained in imagination; zero-shot goal-image tasks; 40-task benchmark. Goal reaching only, not general rewards. — [arXiv](https://arxiv.org/abs/2110.09514)
- Reward-Free Exploration for RL (Jin, Krishnamurthy, Simchowitz, Yu; arXiv 2002.02794, Feb 2020; ICML 2020 from memory, arXiv page names no venue) [V]: explore without reward, then compute near-optimal policies for any number of reward functions; sample complexity roughly O~(S^2 A poly(H)/eps^2) with a near-matching lower bound Omega(S^2 A H^2/eps^2) (tabular). — [arXiv](https://arxiv.org/abs/2002.02794)

### Inferences
- Our RND datasets (10k x 1000 steps) are the ExORL/URLB-style data; Plan2Explore is the key precedent for latent-model-then-plan adaptation.

### Gaps
- Only abstracts read for APT, Proto-RL, Plan2Explore, LEXA, Jin et al.; no benchmark numbers collected.

## 3. Offline reward-free RL, FRE, offline meta-RL, reward labelling

### Takeaway
FRE is the direct non-FB competitor: it encodes a set of labelled (state, reward) pairs with a transformer, matching our test-time setting.

### Cited Findings
- FRE (Frans, Park, Abbeel, Levine; arXiv 2402.17135; ICML 2024, PMLR 235:13927-13942) [V]: transformer VAE encodes (state, reward) samples into a latent; policy pretrained on offline data with random reward functions; no training at test time; code github.com/kvfrans/fre. — [PMLR](https://proceedings.mlr.press/v235/frans24a.html)
- FRE details [V]: benchmarks AntMaze (antmaze-large-diverse-v2; goal, directional, random-simplex, path tasks), ExORL (Walker, Cheetah; velocity and goal tasks), Kitchen (7 subtasks). FRE uses 32 labelled pairs; FB and SF get 5120 samples. Baselines FB, SF, GC-IQL, GC-BC, OPAL (OPAL given online rollouts, best of 10 skills). Table 1 (0-100): AntMaze FRE 52.8, FB 25.8, SF 11.8, OPAL 45.6; ExORL FRE 51.5, FB 43.4, SF 40.9, OPAL 28.2; Kitchen FRE 66, FB 3, SF 1, OPAL 26, GC-IQL 59. FB beats FRE on some ExORL velocity tasks. Random reward prior: uniform mixture of goal-reaching (-1 per step), random linear (masked), random 2-layer MLPs (32 hidden, tanh, clipped). Ablation (AntMaze): FRE-all 47.3 vs single families 25-34. — [arXiv HTML](https://arxiv.org/html/2402.17135v1)
- Offline meta-RL framing (labelled transitions as context): MACAW (Mitchell et al., arXiv 2008.06043), BOReL (Dorfman et al., arXiv 2008.02598), FOCAL (Li et al., arXiv 2010.01112) [M, unverified].
- Offline reward labelling / few labels: Yu et al. "How to Leverage Unlabeled Data in Offline RL" (UDS, arXiv 2202.01741, ICML 2022), Konyushkova et al. "Semi-supervised reward learning for offline RL" (arXiv 2012.06899), PEBBLE (Lee et al., arXiv 2106.05091, ICML 2021) [M, unverified]. The Hu et al. sparse-reward-subset paper was not located.

### Inferences
- FRE's 32 labels vs 5120 for FB: our comparison should fix the label budget (our C4 has n_labels). A reward-regression + planning baseline (tdmpc_adapt.py) is the natural counterpart to FRE's amortised encoder.
- FRE amortises reward inference in one forward pass; ours infers z (FB) or fits a reward head + MPPI; compare adaptation time.

### Gaps
- Offline meta-RL and reward-labelling papers not fetched; Hu et al. not identified.

## 4. 2024-2026 related work

### Takeaway
Little verified non-FB 2025-26 work on reward adaptation; the recent literature is mostly FB-family or goal-conditioned scaling.

### Cited Findings
- Horizon Reduction Makes RL Scalable (Park, Frans, Mann, Eysenbach, Kumar, Levine; arXiv 2506.04168; NeurIPS 2025) [V]: offline RL with datasets up to 1000x larger plateaus; long horizon is the main cause; proposes SHARSA. — [arXiv](https://arxiv.org/abs/2506.04168). Goal-conditioned, not reward adaptation.
- Zero-Shot Off-Policy Learning (Asadulaev et al.; arXiv 2602.01962, Feb 2026, v2 May 2026) [V abstract]: links successor measures to stationary density ratios, training-free test-time correction inside FB frameworks; ExORL, OGBench, SMPL Humanoid. — [arXiv](https://arxiv.org/abs/2602.01962)
- Titles seen in search, content NOT read: A Unified Framework for Zero-Shot RL (2510.20542), Towards Robust Zero-Shot RL (2510.15382), Finer Behavioral Foundation Models via Auto-Regressive Features and Advantage Weighting (2412.04368), Soft FB for general utilities (2602.06769), Zero-Shot Adaptation of BFMs to Unseen Dynamics (2505.13150), Zero-Shot RL from Low Quality Data (2309.15178), RLZero (2412.05718), DVFB (ICLR 2025). — [search listing](https://arxiv.org/pdf/2510.20542). Mostly FB-family (other researchers' scope).

### Inferences
- Horizon-reduction findings are a plausible motivation for the long-horizon latent planning experiments, but this is an inference.

### Gaps
- No verified OGBench-based non-FB reward-adaptation paper from 2025-26; no "unsupervised RL foundation model scaling" paper found; other Park 2025 works (temporal / dual-goal representations) not located.
