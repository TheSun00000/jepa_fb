# Planning on FB latents is still unclaimed

**Bottom line.** Among the papers in our notes, no verified work combines Forward-Backward (FB) task inference (z from reward-labelled states) with MPPI planning in a learned JEPA-style latent model. Everything that infers a task from labelled states (FB, FB-CPR, HILP, FRE, TD-JEPA, ZOL) acts with a z-conditioned policy and does not plan. Everything that plans in a reward-free JEPA latent (PLDM, DINO-WM, V-JEPA 2-AC, LEXA) specifies the task as a goal, not as reward labels. Our setup sits between PLDM and TD-JEPA, and that gap is the novelty claim. The claim rests on keyword searches only. The notes record that citing-works lists, OpenReview and 2026 arXiv listings were never swept, so it must be re-checked before submission. The strongest empirical anchors are the Touati et al. protocol (ExORL RND, 10,000 labels, 10 seeds, % of offline TD3) and the TD-JEPA, HILP and FRE tables, but their numbers are not comparable with each other. A reviewer will expect seeds, bootstrap CIs, an offline-TD3 topline on the same data, a label-budget sweep, final-checkpoint scores and the wall-clock cost of planning versus a forward pass. The last section lists what is still unverified. This report reads only the five note files in `research_notes/Zero shot RL related work/` plus the repo's research plan and method notes (for positioning). I added no new searches, and I flag where notes disagree or were read at abstract level only.

## Five families solve the problem, and only one plans

**The problem statement.** Pretrain on reward-free data. At test time receive N reward-labelled states from an unseen reward r(s). Return a good policy with no or minimal training. Goal reaching is the special case r = 1[s = g]; the notes argue that goal-conditioned policies cannot represent arbitrary rewards without a separate mapping from labels to a goal or latent (an inference in the notes, not a cited result).

**(1) Successor-measure and successor-feature methods (FB family).** FB learns F(s,a,z) and B(s) so that the successor measure factorises. At test time z is roughly E[r(s)B(s)], and the policy is read off with "no planning" ([Touati & Ollivier, arXiv 2103.07945](https://arxiv.org/abs/2103.07945)). "Does Zero-Shot RL Exist?" benchmarked FB and ten SF variants and found FB best and most consistent, at about 85% of supervised RL with a good buffer ([2209.14935](https://arxiv.org/abs/2209.14935)). The SF line (Barreto et al. [1606.05312](https://arxiv.org/abs/1606.05312), USFA [1812.07626](https://arxiv.org/abs/1812.07626), VISR [1906.05030](https://arxiv.org/abs/1906.05030), APS [2108.13956](https://arxiv.org/abs/2108.13956)) assumes reward linear in features. VISR and APS adapt by brief reward exposure or fine-tuning, so they are not strictly zero-shot. Recent FB variants fix specific failure modes. Conservative FB (VC-FB, MC-FB) handles low-quality data and reaches about 150% (MC-FB) and 137% (VC-FB) of FB's IQM on 100k-transition ExORL data ([Jeen et al., 2309.15178](https://arxiv.org/html/2309.15178v3)). Memory-based FB handles partial observability ([2506.15446](https://arxiv.org/html/2506.15446v1)). Auto-regressive features go beyond linear task encoding ([FB-AWARE, 2412.04368](https://arxiv.org/html/2412.04368v1)). One-step FB argues that exact FB needs stringent rank conditions and reports +24% zero-shot over 10 domains ([2602.11399](https://arxiv.org/abs/2602.11399)). FB-CPR adds an imitation regulariser for humanoids ([2504.11054](https://arxiv.org/html/2504.11054v1)), and BFM-Zero runs on a real Unitree G1 ([2511.04131](https://arxiv.org/abs/2511.04131)).

**(2) Latent-predictive representations with task regression.** TD-JEPA (Bagatella et al., Meta; [2510.00739](https://arxiv.org/html/2510.00739), ICLR 2026) learns a state encoder phi, a task encoder psi and a policy-conditioned multi-step latent predictor, all by TD-style latent prediction. At test time z is a least-squares fit on labelled samples, and the latent policy for that z is executed directly, with no planning. HILP ([2402.15567](https://arxiv.org/html/2402.15567v2)) embeds states so that distances track temporal distance and infers z by linear regression on phi(s')-phi(s). It also has a "HILP-Plan" variant for goal tasks. ZOL ([2602.01962](https://arxiv.org/html/2602.01962v2)) refines FB's z at test time by gradient ascent on an importance-weighted return. Proto Successor Measure ([2411.19418](https://arxiv.org/abs/2411.19418)) learns a policy-independent basis and solves for linear weights; the notes list it as "under submission".

**(3) Amortised reward encoders.** FRE ([2402.17135](https://proceedings.mlr.press/v235/frans24a.html), ICML 2024) encodes a set of (state, reward) pairs with a transformer VAE and conditions a policy trained on random reward functions. It needs only 32 labelled pairs against 5120 for its FB and SF baselines.

**(4) Reward-free world models with planning.** PLDM ([2502.14819](https://arxiv.org/abs/2502.14819), NeurIPS 2025) trains a JEPA latent model with VICReg-style variance and covariance terms and plans with MPPI against a goal latent. DINO-WM ([2411.04983](https://proceedings.mlr.press/v267/zhou25t.html)), V-JEPA 2-AC ([2506.09985](https://arxiv.org/abs/2506.09985)) and LEXA ([2110.09514](https://arxiv.org/abs/2110.09514)) also take goals, as images. Plan2Explore ([2005.05960](https://arxiv.org/abs/2005.05960)) is the classic reward-free explore-then-adapt-with-a-model recipe. TD-MPC2 ([2310.16828](https://arxiv.org/abs/2310.16828)) and Dreamer 4 ([2509.24527](https://arxiv.org/abs/2509.24527)) plan or imagine in latent space but use a reward model, so they are not reward-free at the policy stage.

**(5) Skill discovery, goal-conditioned RL and in-context methods.** METRA ([2310.08887](https://arxiv.org/abs/2310.08887)), CSF ([2412.08021](https://arxiv.org/abs/2412.08021)), LSD ([2202.00914](https://arxiv.org/abs/2202.00914)) and CIC ([2202.00161](https://arxiv.org/abs/2202.00161)) support zero-shot goal following or skill selection, not arbitrary dense rewards. The notes did not find a reward-adaptation protocol in their abstracts. Goal-conditioned methods (HIQL [2307.11949](https://arxiv.org/abs/2307.11949), CRL [2206.07568](https://arxiv.org/abs/2206.07568), QRL [2304.01203](https://arxiv.org/abs/2304.01203)) are strong specialised baselines on goal reaching only. Sequence-model in-context methods (Prompt-DT [2206.13499](https://arxiv.org/abs/2206.13499v1), Algorithm Distillation [2210.14215](https://arxiv.org/abs/2210.14215)) condition on trajectories rather than state-reward labels and are not trained reward-free. For theory, Jin et al. ([2002.02794](https://arxiv.org/abs/2002.02794)) give tabular reward-free sample-complexity bounds with a near-matching lower bound.

## What is and is not new in our setup

**Closest prior art, and how we differ.** The differences below are our reading of the notes, not a literature finding.

| Work | What it shares with us | What it lacks relative to us |
|---|---|---|
| TD-JEPA ([2510.00739](https://arxiv.org/html/2510.00739)) | Reward-free latent-predictive pretraining, z from labelled states, DMC and OGBench | Acts with a policy; no test-time planning or rollouts |
| PLDM ([2502.14819](https://arxiv.org/html/2502.14819v4)) | JEPA latent, VICReg-in-time collapse control, MPPI, reward-free offline data | Task = goal observation (a flipped cost sign handles avoidance, by hand); no reward labels, no FB |
| FB-CPR ([2504.11054](https://arxiv.org/html/2504.11054v1)) | Benchmarks an oracle MPPI against the FB policy | MPPI used only as an oracle; ours is a deployed planner with a learned model |
| TD-Flow ([2503.09817](https://arxiv.org/abs/2503.09817)) | Planning over BFM policies with a learned generative model | Different model class (flow-matching horizon model), not latent MPPI on F/B |
| ZOL ([2602.01962](https://arxiv.org/html/2602.01962v2)) | Frozen FB, test-time refinement of z | Gradient refinement of z, no planning or model |
| FRE ([2402.17135](https://proceedings.mlr.press/v235/frans24a.html)) | Same test-time protocol (labelled states, no training) | Amortised encoder and policy, no planner, no FB |

**What looks novel (unverified against the 2026 literature).** The claim in the research plan is that the world model is made useful for every reward in FB's span by predicting B and F from imagined latents, and that the planner scores plans with B_hat(h)^T z plus an F-based terminal value. The notes find no paper doing this: "no FB-family paper does MPPI planning in a learned latent model coupled to F/B", and "a paper combining FB task inference with latent MPPI would sit between PLDM and TD-JEPA". FB-CPR's oracle MPPI is the only precedent. It reports that planning is about 17.8% better than the FB-CPR policy but takes at least 30 minutes per 300-step episode against about 12 seconds for the policy ([arXiv HTML](https://arxiv.org/html/2504.11054v1)). That is the precedent for "planning beats the actor, at higher cost", and it also tells us that cost will be attacked.

**What is not novel.** The pieces are individually standard: FB losses and z inference, JEPA consistency with an EMA target, VICReg-style collapse control (PLDM, where the ablations show variance and covariance terms are essential: 13.4% and 29.2% success without them on Two-Rooms), TD-MPC-style latent MPPI, and linear-regression task inference (HILP, TD-JEPA). The C4 comparison (infer z versus fit a reward head plus Q on the same labels) is a sensible framing, but the notes found no published wall-clock comparison of zero-shot inference against fine-tuning, so there is no baseline to inherit and the experiment must be self-contained. The notes do say a reward-regression plus planning baseline, the `tdmpc_adapt.py` arms, is the natural counterpart to FRE's amortised encoder.

**Honest limits of the novelty claim.** The search tool was US-only and often returned reference lists. Citing-works lists for Touati 2023, Frans 2024 and Bagatella 2025 were never browsed. Several 2026 titles were seen but not read: A Unified Framework for Zero-Shot RL (2510.20542), Soft FB (2602.06769), Switching Successor Measures (2605.13207), Temporal-Distance JEPA (2607.25337), Is Forward Prediction Enough? (2608.06799). A 2026 arXiv sweep for "forward-backward" with "model predictive" was not done. One more caveat is on the planning side: ARC-Bench (2609.05461, snippet only) is reported to argue that closed-loop replanning can mask broken action ranking in frozen JEPA world models. If that holds, C1 ("planning helps") needs the C2 diagnosis (amortisation gap versus actor collapse) to be convincing. The repo's research plan already notes that planning loses to the policy on walker for every variant, which is unexplained.

## Benchmarks and protocols: ExORL with Touati's recipe is the reference

**The protocol to inherit.** Touati et al. fixed the de facto setup on ExORL ([2209.14935](https://arxiv.org/abs/2209.14935)). It uses 13 tasks in 4 environments (Maze, Walker, Cheetah, Quadruped), three buffers (RND, APS, Proto), 10 seeds, d = 50, batch 1024, gamma 0.98, 10^6 gradient steps and **10,000 reward labels** to infer z. Offline and online TD3 are the toplines. Scores are reported as % of offline TD3. FB reaches **81% of offline TD3 over all buffers and 85% on RND**. HILP follows it with 4 environments, 4 datasets, 4 to 8 seeds (its text and appendix disagree), 10,000 state samples and IQM with 95% bootstrap CIs ([2402.15567](https://arxiv.org/abs/2402.15567)). Jeen et al. use 100k-transition subsets plus full data, 10,000 labels and IQM with stratified bootstrap CIs, and run single-task offline RL at about 4 GPU-hours per task versus about 4 hours per domain for FB ([2309.15178](https://arxiv.org/abs/2309.15178)). Our setting (cheetah, walker, quadruped, RND, labelled states) matches this line. One check is needed: the notes say Touati and HILP use the first 5M transitions of each buffer, while the repo's datasets are described as 10k x 1000 steps. Confirm which subset we train on before claiming comparability.

**Protocol differences that break comparisons.** Label budgets: 10,000 (Touati, Jeen, HILP) against 5120 for FB and SF in FRE's table and 32 for FRE itself. Seeds: 10 (Touati), 8 in HILP's main text (4 in its appendix), 5 (FRE, Jeen appendix, TD-JEPA DMC), 3 (Jeen Table 1 caption, which contradicts its own appendix), 10 (TD-JEPA OGBench). Checkpoint choice: Jeen takes the step that maximises all-task IQM, which is selection on test return, a point reviewers may attack. Our `tag=final` rows avoid it. Normalisation: raw return (HILP, TD-JEPA DMC), % of offline TD3 (Touati), 0-100 normalised (FRE), return x100 for antmaze (HILP) against 0-100 normalised (FRE). Tasks: Jeen's quadruped tasks (stand, roll fast, roll, jump, escape) differ from Touati's (stand, walk, run, jump), and Jeen has no Cheetah. FRE's ExORL tasks are velocity and goal only, and FRE appends physics quantities to the dataset to define rewards.

**Reported numbers (each valid only within its own paper).** The table collects the numbers from the notes. Do not mix rows across papers.

| Source | Setting | Numbers |
|---|---|---|
| TD-JEPA Table 1 ([2510.00739](https://arxiv.org/html/2510.00739)) | DMC proprio avg, 5 seeds, mean +/- s.e. | TD-JEPA 661.2 (+/- 6.3), FB 648.2, BYOL-gamma 645.4, HILP 620.1 |
| TD-JEPA Table 1 | DMC pixels | TD-JEPA 628.8, BYOL-gamma 582.4, FB 456.2, HILP 391.2 |
| TD-JEPA Table 1 | Per-domain proprio, TD-JEPA / FB | Walker 785.2 / 811.5; Cheetah 688.7 / 672.7; Quadruped 691.4 / 595.6 (ICVF 708.5); Pointmass 479.3 / 513.0 |
| TD-JEPA Table 1 | OGBench avg, proprio / pixels | 37.98 (FB 39.04, HILP 37.98) / 41.34 (BYOL-gamma 41.58, FB 39.89); antmaze-me 0.20 vs FB 51.60 (BC-regularised) |
| ZOL ([2602.01962](https://arxiv.org/html/2602.01962v2)) | ExORL DMC, 16 tasks, avg return | ZOL 627.2, LoLA 616.9, FB 592.5, ReLA 585.8, TD-JEPA 553.2, HILP 538.5 |
| HILP Table 3 ([2402.15567](https://arxiv.org/abs/2402.15567)) | RND, 5M, 4 seeds, FB / HILP | Cheetah run 183 / 262; run-backward 153 / 187; walk 636 / 823; walk-backward 677 / 843. Walker walk 811 / 855, run 409 / 401, stand 866 / 800, flip 548 / 563. Quadruped jump 642 / 556, run 436 / 393, stand 797 / 810, walk 642 / 542 |
| Jeen Table 6 ([2309.15178](https://arxiv.org/abs/2309.15178)) | RND-100k Walker; CQL / offline TD3 / FB / VC-FB | walk 138 / 210 / 184 / 446; stand 386 / 362 / 558 / 624; run 71 / 84 / 101 / 179; flip 153 / 162 / 163 / 325 |
| Jeen Table 7 | Full RND, FB, 5 seeds | Walker walk 821, stand 928, run 281, flip 525; aggregate IQM FB 389, VC-FB 390, MC-FB 396 (3 seeds per the caption) |
| FRE Table 1 ([2402.17135](https://arxiv.org/abs/2402.17135)) | 0-100, 5 seeds; FRE / FB / SF / OPAL | ExORL 51.5 / 43.4 / 40.9 / 28.2; AntMaze 52.8 / 25.8 / 11.8 / 45.6; Kitchen 66 / 3 / 1 / 26 |
| HILP Table 1 | antmaze-large-diverse, 8 seeds, return x100 | HILP-Plan 64.5, GC-IQL 56.0, HILP 46.0, FB 0.0 |
| FB-CPR ([2504.11054](https://arxiv.org/html/2504.11054v1)) | Humanoid reward tasks | About 61% of TD3 (retrained per task), about 177% of Diffuser |

**Reading the numbers.** FB is 648.2 in TD-JEPA but 592.5 in ZOL, on different protocols, so the ZOL ranking (FB above TD-JEPA) and the TD-JEPA ranking (TD-JEPA above FB) cannot be reconciled from the notes. The TD-JEPA gap over FB on proprioceptive DMC (661.2 vs 648.2, s.e. 6.3) is small, and the note-taker's read is that its gains are mostly on pixel DMC with ties elsewhere. Cheetah is where FB looks weakest on RND in HILP's table (run 183 against HILP 262), which fits the repo's own cheetah gap analysis. Offline TD3 depends strongly on data size: on 100k RND Walker, FB (184 on walk) is below offline TD3 (210), while VC-FB exceeds it. Per-task offline TD3 values for full RND exist only in figures (Touati, ExORL) and cannot be quoted as numbers from the notes. HILP's "GC-TD3" is goal-conditioned TD3 prompted with the highest-reward dataset state, not a per-task offline TD3, so it must not be presented as our topline. On antmaze-large-diverse, FB scores 0.0 on goal reaching in both HILP and FRE, so a near-zero FB there is the established baseline, and a reviewer will ask which protocol we use (goal reward, or sampled reward functions from 32 states). TD-JEPA's OGBench antmaze-me row (0.20 against FB 51.60) is a warning that results there depend on regularisation choices.

## What reviewers will expect

**Statistics.** Papers in this line cite Agarwal et al. 2021: IQM, stratified bootstrap CIs and probability of improvement (Jeen, HILP, TD-JEPA). Seeds range from 3 to 10; 10 is the safe reference and 5 the practical minimum, which matches the repo plan (5 seeds for main tables, 3 for ablations). Report per-task numbers in an appendix and the final checkpoint rather than the best evaluation. Actual OpenReview reviews were not retrievable, so what follows is inferred from what the papers do and from their protocol inconsistencies, not from reviewer comments.

**Baselines and toplines.** Same dataset, same N labels and the same evaluation episodes for all methods. A per-task offline TD3 on the same RND data and steps as the topline (Touati's recipe), which the repo's `td3_jax.py` provides. FB, HILP, TD-JEPA and ZOL on ExORL cheetah, walker and quadruped as the zero-shot baselines, with FRE where the protocol allows. For C4, the same N labels fed to (i) linear regression of z, (ii) a reward head plus Q on the frozen world model (`tdmpc_adapt.py`) and (iii) per-task training (oracle). A label sweep that includes 32 and 5120 as well as 10,000 is the cheap defence against the FRE-versus-FB asymmetry, which shows that label count is a contested knob.

**Cost.** Planning will be the first complaint. The only number in the notes is the FB-CPR oracle MPPI (at least 30 minutes against about 12 seconds per 300-step episode). PLDM reports its planner is about 4x slower than model-free at default settings ([arXiv HTML](https://arxiv.org/html/2502.14819v4)). We should report MPPI versus policy wall-clock per episode (the repo logs `mppi_time` and `policy_time`) and per-task adaptation time. Test-time inference for FB and SF is linear regression with no gradient steps; per-task offline RL costs about 4 GPU-hours (Jeen, A100). The notes found no published comparison of zero-shot versus fine-tuning wall-clock, so we can contribute one.

**Reproducibility.** facebookresearch/controllable_agent (FB and SF, MIT, archived, last push 2023-07-17) should be run unmodified with a pinned commit ([GitHub API metadata per the notes](https://api.github.com/repos/facebookresearch/controllable_agent)). The HILP and FRE repos declare no license. Meta Motivo's license is a custom one (read the LICENSE file). A TD-JEPA repo was not found (facebookresearch/tdjepa returned 404). ExORL (denisyarats/exorl), URLB (rll-research/url_benchmark) and OGBench (seohongpark/ogbench) are MIT.

## Must-cite list

All IDs below appear in the notes. "Verified" in the notes means checked at abstract or listing level, not read in full.

| Group | Paper | arXiv |
|---|---|---|
| FB core | Touati & Ollivier, Learning One Representation to Optimize All Rewards | 2103.07945 |
| | Touati, Rapin, Ollivier, Does Zero-Shot RL Exist? | 2209.14935 |
| | Jeen et al., Zero-Shot RL from Low Quality Data | 2309.15178 |
| | Jeen et al., Zero-Shot RL Under Partial Observability | 2506.15446 |
| | Cetin et al., Finer BFMs (FB-AWARE) | 2412.04368 |
| | Zheng et al., Can We Really Learn One Representation to Optimize All Rewards? | 2602.11399 |
| | Tirinzoni et al., FB-CPR / Meta Motivo | 2504.11054 |
| | BFM-Zero | 2511.04131 |
| | ZOL, Zero-Shot Off-Policy Learning | 2602.01962 |
| | Sikchi et al., Fast Adaptation with BFMs | 2504.07896 |
| | Farebrother et al., Temporal Difference Flows | 2503.09817 |
| | Agarwal et al., Proto Successor Measure | 2411.19418 |
| | Di Ventura et al., A Unified Framework for Zero-Shot RL | 2510.20542 |
| Latent / JEPA | Bagatella et al., TD-JEPA | 2510.00739 |
| | Sobal et al., PLDM | 2502.14819 |
| | Zhou et al., DINO-WM | 2411.04983 |
| | V-JEPA 2 / 2-AC | 2506.09985 |
| | TD-MPC2 | 2310.16828 |
| | LEXA; Plan2Explore | 2110.09514; 2005.05960 |
| Non-FB zero-shot | Park et al., HILP | 2402.15567 |
| | Frans et al., FRE | 2402.17135 |
| | VISR; APS; Barreto et al. SF; USFA | 1906.05030; 2108.13956; 1606.05312; 1812.07626 |
| | METRA; CSF; LSD; CIC | 2310.08887; 2412.08021; 2202.00914; 2202.00161 |
| Benchmarks | ExORL; URLB; OGBench | 2201.13425; 2110.15191; 2410.20092 |
| Goal-conditioned | HIQL; CRL; QRL | 2307.11949; 2206.07568; 2304.01203 |
| Theory, offline RL | Jin et al., Reward-Free Exploration; Park et al., Horizon Reduction | 2002.02794; 2506.04168 |

## Still unverified: check before submission

**Items the notes flag as [M] or UNVERIFIED.** These were recalled from memory or seen only as titles, snippets or secondary summaries. Keep each flag until the primary page is read.

| Item | Status in the notes |
|---|---|
| TD-JEPA venue (ICLR 2026, oral) | One note cites the ICLR listing, another says "venue unverified". Check. |
| ExORL venue | Notes disagree: ICLR 2022 workshop (arXiv/mlanthology listing) versus NeurIPS 2022 Datasets track (memory). Check. |
| FB original: NeurIPS 2021 | Memory; the arXiv page says only arXiv. |
| FRE venue (ICML 2024) | Confirmed by a PMLR page in two notes, "venue unverified" in the benchmarks note. |
| URLB venue (NeurIPS 2021 Datasets track) | Memory. |
| DIAYN 1802.06070, VALOR 1807.10299, DADS 1907.01657 | IDs from memory, no page fetched. |
| Offline meta-RL: MACAW 2008.06043, BOReL 2008.02598, FOCAL 2010.01112 | Memory. |
| Reward labelling: UDS 2202.01741, Konyushkova 2012.06899, PEBBLE 2106.05091 | Memory. |
| TD-MPC v1 2203.04955 | Memory. |
| LeJEPA 2511.08544 | ID seen only inside a wiki URL. |
| Dreamer line (PlaNet 1811.04551, Dreamer v1 1912.01603, V2 2010.02193, V3 2301.04104, DayDreamer 2206.14176, Genie 2402.15391, iVideoGPT 2405.15223), SPR 2007.05929, BYOL-Explore 2206.08332, Ni et al. 2401.08898 | Memory, not searched. |
| Offline MBRL: MOPO 2005.13239, MOReL 2005.05951, COMBO 2102.08363, RAMBO 2204.12581, MOBILE 2210.14157 | Memory. |
| Decision Transformer 2106.01345, HER 1707.01495, IQL 2110.06169, TD3+BC 2106.06860, Barreto 2018 SF+GPI 1901.10964 | Memory. |
| Pirotta et al., Fast Imitation via BFMs (ICLR 2024) | Title and venue seen; arXiv ID not retrieved. |
| Proto Successor Measure | "Under submission" on arXiv; ICML 2025 acceptance not confirmed. |
| "Zero-Shot Adaptation of BFMs to Unseen Dynamics" | One note gives arXiv 2505.13150 and ICLR 2026 from a search listing; another found only a third-party review and no ID. The "up to 2x" gain is from a summary. |

**Claims taken from secondary sources.** FB-AWARE "matches single-task offline RL on D4RL locomotion" (secondary summary). LeWM's 48x planning speed-up, CEM planner and SIGReg weakness on low-dimensional dynamics (arXiv 2603.19312; from an [explainer](https://signals.aktagon.com/articles/2026/05/leworldmodel-stable-end-to-end-joint-embedding-predictive-architecture-from-pixels/), not checked in the paper). ARC-Bench 2609.05461 (snippet only). The "linearization / inference / approximation" error split of 2510.20542 (secondary summary). The conjecture that sparse goal rewards collapse diversity of B, attributed to the one-step FB paper (third-party summary, untested). Dreamer 4 "degrades on visually different tasks" (non-peer-reviewed summary). The Puppeteer 50/50 offline-online mix and 97.6% user preference. One note quotes "about 150% of vanilla FB" for conservative FB from a catalogue listing; the figures used above (MC-FB about 150%, VC-FB about 137% on 100k ExORL) come from the paper's HTML as read in the FB note. One note titles 2309.15178 "Conservative World Models"; the other notes and the paper name it "Zero-Shot RL from Low Quality Data".

**Numbers and protocol facts not extracted.** Touati's Table 2 RND-block FB per-task values (column order unclear in extraction; read the paper). Offline TD3 per-task values on full RND (figures only: Touati, ExORL). The ExORL buffer used by FRE and by TD-JEPA. The appendix half of the TD-JEPA HTML. Per-domain numbers for FB-AWARE, Proto SM and TD-Flow. FB-CPR protocol beyond the abstract. The TD-JEPA column mapping in the per-domain rows (assigned by the note-taker from the header order). Jeen's code location. Whether each public repo reproduces its paper's numbers.

**Searches not done.** OpenReview reviews for Jeen (NeurIPS 2024), HILP, FB-CPR, OGBench and TD-JEPA. Google Scholar or Semantic Scholar citing lists for Touati 2023, Frans 2024 and Bagatella 2025. An arXiv sweep of 2026 listings for "forward-backward" together with "model predictive" or "planning". The other 2026 JEPA world-model titles (2607.25337, 2608.06799, 2606.23444, 2609.10464) and the "Sprekeler graph-Laplacian zero-shot RL" reference, which the notes could not find. Any paper that probes whether a reward-free JEPA latent retains reward information (for example with linear reward probes): none was found, so we can claim only "not found".

## Conclusion

The defensible contribution is narrower than "zero-shot RL with a world model". It is this: given FB's task inference, does planning in a latent space grounded in FB's reward basis beat the FB actor, and at what test-time cost? FB-CPR's oracle MPPI (about 17.8% better, at least 30 minutes against 12 seconds per episode) suggests yes with a large price. Our own runs so far show the gain on cheetah and quadruped but not walker (research plan, single seeds). That makes the walker diagnosis and the cost accounting central to C1 and C2, not side experiments.

Two further points follow from the benchmark notes. The TD-JEPA margin over FB on proprioceptive DMC is a few points, so any comparison with it is credible only on the same protocol, which means running TD-JEPA's or HILP's code ourselves where it is available instead of quoting their numbers. And the 32 versus 5120 versus 10,000 label split means the C4 result will be judged on how the label sweep is reported as much as on the headline number.
