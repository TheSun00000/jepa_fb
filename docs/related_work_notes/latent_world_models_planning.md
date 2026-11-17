# Reward-free latent world models and planning (JEPA-style / predictive) for test-time-specified tasks

Research date 2026-11-17. About 14 tool calls. "VERIFIED" = seen on a fetched page or search result this session. "UNVERIFIED (memory)" = arXiv ID/venue from background knowledge, not checked now; check before citing. No numbers invented; numbers below are quoted from the cited page only.

## Q1. Reward-free world models + planning/imagination: what exists, data, test-time spec, benchmarks, headline results

### Takeaway
The closest prior art to a "reward-free JEPA latent model + MPPI planning" is PLDM (Sobal et al., NeurIPS 2025), but it only does goal-reaching (a goal latent, or a flipped cost), not rewards given as labelled states. TD-JEPA (ICLR 2026) takes labelled reward samples but is a successor-feature method with no planner. No paper found pairs FB/successor-feature task inference (labelled states -> z) with MPPI in a JEPA latent model; that looks like the open niche.

### Cited Findings

**JEPA-style / latent planning from reward-free offline data**
- PLDM, Sobal, Zhang, Cho, Balestriero, Rudner, LeCun, "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models", arXiv 2502.14819 (v4), NeurIPS 2025 poster. Compares goal-conditioned/zero-shot RL against planning with a JEPA latent dynamics model on navigation; planning generalises better to unseen layouts and is more data-efficient; model-free does best with large high-quality data. — [arXiv](https://arxiv.org/abs/2502.14819), [NeurIPS page](https://neurips.cc/virtual/2025/poster/116649)
- PLDM data/envs: Two-Rooms (64x64 top-down pixels, 3M transitions of varying quality), Diverse PointMaze (random 4x4 layouts, 5 to 40 train layouts), OGBench Ant U-Maze (5M transitions from noisy directional expert); 23 datasets in all. — [arXiv HTML](https://arxiv.org/html/2502.14819v4)
- PLDM losses: latent similarity (multi-step prediction), VICReg-style variance hinge applied across the time dimension, covariance decorrelation, time-similarity (smoothness), inverse dynamics. Ablations (full model 98.0% on Two-Rooms): no variance 13.4%, no covariance 29.2%, no time-sim 71.0%, no IDM 98.0%; on Diverse Maze (full 98.7%): no variance 11.4%, no covariance 7.8%, no time-sim 95.6%, no IDM 75.5%. — [arXiv HTML](https://arxiv.org/html/2502.14819v4)
- PLDM planner/test-time spec: MPPI in an MPC loop; cost = latent distance to encoded goal plus beta-weighted ensemble-variance uncertainty penalty. Task = goal observation. A "chaser" avoidance task is done by flipping the cost sign, no retraining. — [arXiv HTML](https://arxiv.org/html/2502.14819v4)
- PLDM headline (Two-Rooms, good data): PLDM 197.8 +/- 10.7, GCIQL 198.0, HIQL 196.4, CRL 189.3, GCBC 186.0, HILP 100.0. With no door-passing data: GCIQL 199.6, PLDM 134.4, HIQL 126.3, HILP 100.0. Ant U-Maze stitching: PLDM, HIQL, HILP reach 100% with short trajectories. Authors state PLDM "is not using rewards, neither explicitly nor implicitly". Limitations noted: about 4x slower than model-free at default settings; future work on manipulation, partial observability, compounding error. — [arXiv HTML](https://arxiv.org/html/2502.14819v4). Project page/code (from PDF link annotations): latent-planning.github.io, github.com/vladisai/PLDM.
- DINO-WM, Zhou, Pan, LeCun, Pinto, arXiv 2411.04983, ICML 2025. World model predicts frozen DINOv2 patch embeddings with a ViT transition model; planning by MPC with latent MSE to a goal image; claims zero-shot solutions without expert demos, reward modelling or inverse models across six environments (maze, push, particle manipulation); "45% improvement on hardest goal-reaching tasks" (arXiv v2 per summary). Pretrained encoder, so no collapse issue; offline trajectories. — [ICML/PMLR](https://proceedings.mlr.press/v267/zhou25t.html), [arXiv HTML](https://arxiv.org/html/2411.04983v2)
- V-JEPA 2 / V-JEPA 2-AC (Meta), arXiv 2506.09985, 2025. V-JEPA 2 pretrained by mask-denoising on >1M hours of internet video + 1M images, no actions; 77.3 top-1 on Something-Something v2; 84.0 PerceptionTest, 76.9 TempCompass after LLM alignment. V-JEPA 2-AC post-trained on <62 h of unlabeled DROID video; planned with CEM in MPC loop using image goals; deployed zero-shot on Franka arms in two labs (pick-and-place) with no robot data from those labs and no reward. Figures are from abstract/summaries; full paper not read. — [arXiv](https://arxiv.org/abs/2506.09985), [Meta](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)
- LeWorldModel (LeWM), Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv 2603.19312 (Mar 2026). End-to-end JEPA from pixels, next-embedding loss + SIGReg (Gaussian-latent regulariser via random projections and Epps-Pulley statistic); 15M params; reported planning up to 48x faster than foundation-model world models; MPC with CEM per a secondary explainer; caveat from the same explainer: SIGReg may struggle with very low-dimensional dynamics. Secondary-source details, not verified in the paper. — [arXiv HTML](https://arxiv.org/html/2603.19312), [explainer](https://signals.aktagon.com/articles/2026/05/leworldmodel-stable-end-to-end-joint-embedding-predictive-architecture-from-pixels/)
- LeJEPA (Balestriero, LeCun, Nov 2025), introduces SIGReg, "provable and scalable SSL without heuristics". ID 2511.08544 appears only inside a JEPA-wiki URL, so UNVERIFIED. — [wiki](https://mishig-jepawiki.hf.space/wiki/papers/2511.08544)

**TD-MPC family and Hansen**
- TD-MPC2, Hansen, Su, Wang, arXiv 2310.16828, ICLR 2024. Decoder-free latent world model, planning by MPC in latent space; 104 continuous-control tasks (DMControl, Meta-World, ManiSkill2, MyoSuite) with one hyperparameter set; single 317M-param agent on 80 tasks; capability grows with model/data size. Uses reward (task-specific), not reward-free. — [arXiv](https://arxiv.org/abs/2310.16828), [ICLR](https://iclr.cc/virtual/2024/poster/18722)
- "Hierarchical World Models as Visual Whole-Body Humanoid Controllers" (Puppeteer), Hansen, Jyothir S V, Sobal, LeCun, Wang, Su, arXiv 2405.18418. Two-level: a proprioceptive tracking agent trained on human mocap data and a visual puppeteer agent emitting reference motions; both use TD-MPC2; 8 visual whole-body tasks, 56-DoF humanoid; compared to SAC, DreamerV3, TD-MPC2; no reward design or skill primitives claimed. Third-party summaries give a 50/50 offline/online mix and a 97.6% user preference; both unverified against the paper (the paper text mentions a 51-participant study per one summary). — [arXiv HTML](https://arxiv.org/html/2405.18418v3)
- TD-MPC v1 (Hansen, Wang, Su, arXiv 2203.04955, ICML 2022): UNVERIFIED (memory).

**Dreamer line and imagination-based**
- Dreamer 4 (Hafner, Yan, Lillicrap), arXiv 2509.24527. Pretrain tokenizer + world model on videos and actions, finetune policy/reward heads into the model, then imagination RL; gets Minecraft diamonds from offline data only, beats OpenAI VPT offline agent with 100x less data (as reported); real-time inference on one GPU via shortcut forcing; learns action conditioning from a small amount of action-labelled data. Reward is still used in the imagination stage (reward model), so not reward-free at the policy stage. — [arXiv](https://arxiv.org/abs/2509.24527), [project page](https://danijar.com/project/dreamer4/)
- LEXA, Mendonca, Rybkin, Daniilidis, Hafner, Pathak, arXiv 2110.09514, NeurIPS 2021. World model from images; explorer and achiever policies trained in imagination; goals specified as images, zero-shot; multi-object tasks; one agent across four environments. — [arXiv](https://arxiv.org/abs/2110.09514)
- Plan2Explore (Sekar, Rybkin et al., ICML 2020): self-supervised exploration with a world model, then adaptation to a downstream reward. Search confirmed existence only via the LEXA page; arXiv ID 2005.05960 UNVERIFIED (memory). — [LEXA page mention](https://arxiv.org/abs/2110.09514)
- WHALE, Zhang et al., arXiv 2411.05619 (NeurIPS 2025 listing): behavior-conditioning and retracing-rollout (uncertainty without ensemble); Whale-ST transformer, Whale-X 414M params on 970K Open-X-Embodiment trajectories; gains over DreamerV3/FitVid on Meta-World per abstract (not checked in paper). — [arXiv HTML](https://arxiv.org/html/2411.05619v1), [NeurIPS](https://neurips.cc/virtual/2025/124016)
- Not searched this session, all UNVERIFIED (memory): PlaNet 1811.04551; Dreamer v1 1912.01603; DreamerV2 2010.02193; DreamerV3 2301.04104; DayDreamer 2206.14176; Genie 2402.15391; iVideoGPT 2405.15223; "Dreaming of Many Worlds" (contextual world models, ID not recalled); SPR 2007.05929; BYOL-Explore 2206.08332; Ni et al. "Bridging State and History Representations" 2401.08898. I found no verifiable claim that DreamerV3 reward-free pretraining is a standard recipe; Plan2Explore is the standard reward-free Dreamer-style recipe.

### Inferences
- PLDM is the direct template for the project's planner side (same VICReg-in-time collapse control, MPPI, offline reward-free data), and its cost is goal distance. Extending the cost to a reward inferred from labelled states (FB's z = B-weighted labelled states, or a linear reward head) is not covered by PLDM.
- Dreamer 4, TD-MPC2, Puppeteer all use reward at some stage; only PLDM, DINO-WM, V-JEPA 2-AC, LEXA, LeWM are reward-free, and all of those specify tasks by goal.

### Gaps
- Full PLDM limitations appendix and OGBench numbers beyond Ant U-Maze not read (HTML truncated at 100k chars; an early PDF fetch was unreadable).
- Dreaming of Many Worlds, DayDreamer, iVideoGPT, Genie, PlaNet/Dreamer v1-v3, SPR/BYOL-Explore details not verified.

## Q2. How the downstream objective is specified; which handle rewards given as labelled states

### Takeaway
Most reward-free world-model planners take a goal observation (PLDM, DINO-WM, V-JEPA 2-AC, LEXA). Labelled-reward-sample task inference comes from the FB / successor-feature family (FB, HILP, TD-JEPA), where reward samples -> task vector by regression, without a model.

### Cited Findings
- TD-JEPA, Bagatella, Pirotta, Touati, Lazaric, Tirinzoni, arXiv 2510.00739, ICLR 2026. Given labelled reward samples, the task embedding is a least-squares fit on the task encoder psi (closed form); the latent policy for that embedding is executed directly; no test-time planning or world-model rollout described. Trained: state encoder phi, task encoder psi, policy-conditioned multi-step predictor, parametrised latent policies, all by TD-style latent prediction. — [arXiv HTML](https://arxiv.org/html/2510.00739v1), [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html)
- TD-JEPA benchmarks: 4 ExoRL/DMC locomotion+navigation domains, 9 OGBench domains, proprio and pixels, 65 tasks, 13 datasets. Baselines: Laplacian, HILP, FB, ICVF, RLDP, BYOL, BYOL-gamma. Table 1 averages: DMC pixels TD-JEPA 628.8 vs BYOL-gamma 582.4 vs FB 456.2; DMC proprio 661.2 vs FB 648.2; OGBench pixels 41.34 (BYOL-gamma 41.58); OGBench proprio 37.98 (HILP tied, FB 39.04). Honest read: gains mostly in pixel DMC, ties elsewhere. — [arXiv HTML](https://arxiv.org/html/2510.00739v1)
- PLDM compares against HILP (zero-shot RL) on goal tasks; task = goal state only. — [arXiv HTML](https://arxiv.org/html/2502.14819v4)
- A Unified Framework for Zero-Shot RL (arXiv 2510.20542) classes FB as "pseudo reward-free" (training uses task embeddings) and one-step FB as fully reward-free; TD-JEPA listed as pseudo reward-free. — [arXiv HTML](https://arxiv.org/html/2510.20542)
- Zero-Shot Off-Policy Learning (arXiv 2602.01962, listed May 2026 revision): training-free task adaptation within FB frameworks; details not read. — [arXiv HTML](https://arxiv.org/html/2602.01962)
- BFM-Zero (arXiv 2511.04131): FB-based behavioural foundation model for humanoids with zero-shot motion tracking, goal reaching, reward inference and few-shot optimisation-based adaptation. — [arXiv](https://arxiv.org/abs/2511.04131)
- Zero-Shot RL Under Partial Observability (arXiv 2506.15446): exists, content not read. — [arXiv](https://arxiv.org/abs/2506.15446)
- Conservative World Models (arXiv 2309.15178, 2023): FB overestimates value of out-of-dataset actions on suboptimal data; proposes VC-FB. — [arXiv HTML](https://ar5iv.labs.arxiv.org/html/2309.15178)
- FB original (Touati and Ollivier, NeurIPS 2021, arXiv 2103.07945) and "Does Zero-Shot RL Exist?" (2209.14935): UNVERIFIED (memory).
- Dreamer 4 / TD-MPC2: reward model or true reward learned per task; not labelled-state specified. — [Dreamer 4](https://arxiv.org/abs/2509.24527), [TD-MPC2](https://arxiv.org/abs/2310.16828)

### Inferences
- Among verified papers none handles "reward = labelled states" with a world model + planner. Only TD-JEPA/FB/HILP handle labelled rewards, and none plans.
- PLDM's "flip the cost sign" shows cost swapping is possible but hand-designed.

### Gaps
- Could not confirm whether any 2026 paper combines FB's z inference with a learned dynamics model and MPPI (search found none, but coverage is limited: US-only web search, truncated results).

## Q3. Collapse, regularisation, and whether reward-free latents retain reward information

### Takeaway
Collapse control in reward-free JEPA RL is done by VICReg-style variance/covariance (PLDM; ablations show it is essential), SIGReg (LeWM), orthonormality/covariance regularisers plus target nets (TD-JEPA), or a frozen pretrained encoder (DINO-WM). I found no clean study that measures reward-information retention of reward-free latents; one 2026 critique (ARC-Bench) argues the latent-closeness assumption fails.

### Cited Findings
- PLDM: removing variance or covariance terms collapses success (13.4% / 29.2% on Two-Rooms; 11.4% / 7.8% on Diverse Maze); variance applied across time so features encode what changes. — [arXiv HTML](https://arxiv.org/html/2502.14819v4)
- TD-JEPA: Theorem 2 (idealised continuous-time, linear predictors, symmetry assumptions) says covariances of phi and psi stay constant if predictors train faster than representations, so proper initialisation (unit covariance) avoids trivial solutions; practical algorithm adds target nets and an orthonormality/covariance regulariser. Authors flag the theory as idealised. — [arXiv HTML](https://arxiv.org/html/2510.00739v1)
- TD-JEPA reports latent-predictive methods (BYOL-gamma, TD-JEPA) do better than FB/HILP on pixels, suggesting predictive latents keep control-relevant information. — [arXiv HTML](https://arxiv.org/html/2510.00739v1)
- LeWM: SIGReg (random projections + Epps-Pulley) gives stable end-to-end JEPA from pixels; possible weakness for very low-dimensional dynamics (secondary source). — [arXiv HTML](https://arxiv.org/html/2603.19312)
- ARC-Bench (arXiv 2609.05461): per search snippet, argues closed-loop replanning masks broken action ranking in frozen JEPA world models; latent closeness assumption fails on released checkpoints. Only snippet seen; verify. — [arXiv](https://arxiv.org/abs/2609.05461)
- Other 2026 JEPA-world-model papers seen only as titles in search results (not read): Temporal-Distance JEPA 2607.25337 ("plan-aware representation learning"), "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models" 2608.06799, SkyJEPA 2606.23444, Semigroup-JEPA 2609.10464. — [2607.25337](https://arxiv.org/abs/2607.25337), [2608.06799](https://arxiv.org/abs/2608.06799), [2606.23444](https://arxiv.org/abs/2606.23444), [2609.10464](https://arxiv.org/abs/2609.10464)
- Self-predictive RL background (SPR, BYOL-Explore, Ni et al. on state/history representations): UNVERIFIED (memory), IDs listed in Q1.

### Inferences
- The reward-relevance risk is real: PLDM-style latents are shaped only by dynamics and decorrelation, so reward-relevant directions need not be preserved. This motivates a B / reward-basis learned alongside (as in the project) or an inverse-dynamics term (PLDM's IDM ablation matters on Diverse Maze, 98.7 -> 75.5).

### Gaps
- No verified paper directly probing "does a reward-free JEPA latent retain reward information" (e.g. linear reward probes). Not found in search.

## Q4. Offline model-based RL context

### Takeaway
Not verified in this session (search returned nothing for these). They are uncertainty/pessimism-based methods trained with reward, so they are context, not competitors.

### Cited Findings
- All UNVERIFIED (memory), check before citing: MOPO (Yu et al., NeurIPS 2020, arXiv 2005.13239, uncertainty-penalised reward); MOReL (Kidambi et al., NeurIPS 2020, arXiv 2005.05951, pessimistic MDP); COMBO (Yu et al., NeurIPS 2021, arXiv 2102.08363, conservative critic with model rollouts); RAMBO-RL (Rigter et al., NeurIPS 2022, arXiv 2204.12581, adversarial model); MOBILE (Sun et al., ICML 2023, "Model-Bellman Inconsistency", arXiv 2210.14157 recalled).
- Reward-free analogue: WHALE's retracing-rollout uncertainty and PLDM's ensemble-variance penalty in the planning cost play the same pessimism role. — [WHALE](https://arxiv.org/html/2411.05619v1), [PLDM](https://arxiv.org/html/2502.14819v4)

### Inferences
- PLDM's uncertainty penalty is the planning-time version of MOPO-style penalties; useful as a design comparison.

### Gaps
- No verification of any offline MBRL paper, IDs, or numbers.

## Q5. 2025-2026 work, scaling, FB + world models

### Takeaway
2025-26 reward-free work splits into (a) successor-feature/FB (TD-JEPA, BFM-Zero, unified framework, zero-shot off-policy learning), (b) JEPA world models with goal-image planning (PLDM, DINO-WM, V-JEPA 2-AC, LeWM), (c) reward-using scaling (Dreamer 4, TD-MPC2). I found no verified FB + world-model planner.

### Cited Findings
- Dreamer 4 (Sep 2025), TD-MPC2 317M agent (2023/24), V-JEPA 2 (Jun 2025), PLDM (Feb 2025; NeurIPS 2025), TD-JEPA (Oct 2025; ICLR 2026), LeWM (Mar 2026), BFM-Zero (Nov 2025), 2510.20542 (Oct 2025), 2602.01962 (Feb 2026): see Q1-Q3 for sources and details.
- "Dreamer 4 relies on extensive pretraining data and degrades on visually different tasks" comes from a non-peer-reviewed AI summary only. — [arXivLens](https://arxivlens.com/paperview/details/training-agents-inside-of-scalable-world-models-6933-8f7d3dba)

### Inferences
- A paper combining FB task inference with latent MPPI would sit between PLDM and TD-JEPA; the novelty claim must be checked against 2026 arXiv before asserting.

### Gaps
- Search tool is US-only and often returned reference lists; 2026 IDs (2606-2609) were not opened. A sweep of arXiv listings for "forward-backward" + "model predictive" in 2026 was not done.
- Dreamer 4 task-conditioning details and TD-MPC2 per-task numbers not extracted.
