# Benchmarks, protocols, datasets and baseline numbers for zero-shot RL from reward-free offline data

Compiled 2026-11-17. All numbers below were read from the arXiv full text (PDF text extraction) of the cited paper; anything not read is marked "unverified". Per-paper arXiv links are the sources. Note: the coordinator later asked for arXiv HTML rather than PDFs; the PDFs were already extracted before that message, and the remaining lookups used abs/HTML pages.

## 1. Protocols per paper (ExORL and related)

### Takeaway
The de facto protocol was set by Touati et al. 2023: ExORL buffers (RND/APS/Proto), first 5M transitions, 1M gradient steps, task inferred from 10,000 reward-labelled samples, 10 seeds, scores reported raw and as % of offline TD3. Later papers deviate on seeds (3 to 10), number of labels (32 to 10,000+), and normalization, so every comparison table must state these.

### Cited Findings
- **ExORL (Yarats et al. 2022, arXiv 2201.13425, NeurIPS 2022 datasets track?; venue unverified).** Unsupervised exploration agents (Random, ICM, Disagreement, RND, Proto, APT, APS, DIAYN, SMM) collect reward-free data with a 1M-step budget (1000 episodes) unless stated, which is then relabelled with task rewards and fed to BC, TD3+BC, CRR, CQL, TD3. Offline evaluation rolls out 10 episodes across 10 seeds. Vanilla TD3 usually outperforms CRR, CQL and TD3+BC on this exploratory data. — [ExORL](https://arxiv.org/abs/2201.13425)
- **Touati, Rapin, Ollivier 2023, "Does Zero-Shot RL Exist?" (arXiv 2209.14935, ICLR 2023).** 11 methods (FB plus 10 SF variants) on 13 tasks in 4 environments: Maze (20 goals), Walker (stand, walk, run, flip), Cheetah (walk, run, walk-backward, run-backward), Quadruped (stand, walk, run, jump). Three ExORL buffers (RND, APS, Proto), 403 settings, 10 seeds each. d = 50 (100 for Maze). Tasks revealed by 10,000 reward samples (Maze: known goal sets z directly). Hyperparameters: buffer size 5x10^6 (10x10^6 Maze), batch 1024, gamma 0.98, Adam lr 1e-4, 10^6 gradient steps, 10^4 reward labels, same hyperparameters for all methods. Toplines: online TD3 and offline TD3 on the same buffer labelled with the task reward. Normalization: per task and buffer, score as % of offline TD3, averaged over tasks, then environments. FB reaches 81% of offline TD3 averaged over all tasks and buffers, 85% on RND. Per-task table (Table 2) gives mean +/- 1 sigma over 10 seeds; offline/online TD3 per-task values appear only in Fig. 2 (not in a table). Code: facebookresearch/controllable_agent. — [arXiv 2209.14935](https://arxiv.org/abs/2209.14935)
- **Jeen, Bewley, Cullen, "Zero-Shot RL from Low Quality Data" (arXiv 2309.15178, NeurIPS 2024).** Uses 100k transitions uniformly subsampled from RND, DIAYN and Random ExORL datasets (plus full datasets), domains Walker, Quadruped, Maze, Jaco (no Cheetah), D4RL medium/medium-replay/medium-expert for an extra question. Baselines: FB, SF-LAP, goal-conditioned IQL, CQL and offline TD3 trained per task. z-inference labels: 10,000. Evaluation: 1M learning steps, checkpoints every 20k, 10 rollouts per checkpoint, IQM, 95% stratified bootstrap CIs following Agarwal et al. 2021; task score taken at the step that maximises all-task IQM across seeds (note: checkpoint selection on test return, a point reviewers may attack). Seeds: appendix says five seeds for ExORL (Table 6, 7) and three for D4RL (Table 9), but main-text Table 1 caption says "averaged over three seeds"; inconsistency within the paper. Compute: single-task offline RL about 4 GPU-hours per task, FB/SF about 4 h per domain, conservative FB about 12 h per domain (A100); about 110 GPU-days for the 100k experiments. — [arXiv 2309.15178](https://arxiv.org/abs/2309.15178), NeurIPS page [neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e38b2a0b77541b14a3315c99697b835-Abstract.html)
- **Park, Kreiman, Levine, HILP (arXiv 2402.15567, ICML 2024).** Zero-shot RL following Touati et al.: Walker, Cheetah, Quadruped, Jaco; four datasets APS, APT, Proto, RND; first 5M transitions per dataset; four tasks per environment; 10,000 state samples for latent inference (Table 5); 10^6 gradient steps (state-based), d = 50; 8 seeds in the main text and 95% bootstrap CIs in plots (std in tables); the ExORL appendix Table 3 states 4 seeds. Aggregation: IQM over 4 envs x 4 tasks x 4 datasets x 4 seeds (Fig. 5), returns normalized by max (1000; Jaco x4 since max 250). Baselines GC-TD3, CL, Rand, IDM, AE, Lap, FDM, FB. Note: GC-TD3 is a goal-conditioned TD3 prompted with the highest-reward dataset state, not task-specific offline TD3. — [arXiv 2402.15567](https://arxiv.org/abs/2402.15567)
- **Frans, Park, Abbeel, Levine, FRE (arXiv 2402.17135, ICML 2024? venue unverified).** AntMaze (D4RL antmaze-large-diverse-v2; goal-reaching, directional, random-simplex, 3 hand-designed paths), ExORL (cheetah and walker; goals and velocity tasks only: run, walk, run-backward, walk-backward for cheetah; run, walk for walker; 1000-step episodes; dataset buffer not identified in the text I read, unverified), Kitchen (7 subtasks). 20 evaluation episodes, 5 seeds, std across seeds. FRE uses only 32 (state, reward) pairs at test time; FB and SF are given 5120 reward samples "to be consistent with prior work" (differs from the 10,000 of Touati et al.). Results normalized to 0-100. FRE needs the physics quantities (velocity, torso height, ...) appended to the dataset for the encoder to define the ExORL rewards (appendix C.2); SF uses ICM features. Encoder 150k (1M ExORL/Kitchen) + policy 850k (1M ExORL/Kitchen) steps. — [arXiv 2402.17135](https://arxiv.org/abs/2402.17135)
- **Bagatella et al., TD-JEPA (arXiv 2510.00739, Oct 2025; venue unverified).** 4 DMC/ExORL domains (walker, cheetah, quadruped, pointmass), tasks 4-8 per domain, plus 9 OGBench domains (antmaze medium/large navigate/stitch/explore, cube-single/double, scene, puzzle-3x3), proprioceptive and RGB. DMC: 5 seeds, 20 episodes; OGBench: 10 seeds, 10 episodes. Baselines Laplacian, ICVF*, HILP, FB, RLDP, BYOL*, BYOL-gamma* (all given an explicit state encoder and tuned over comparable grids). Reports mean +/- standard error, probability of improvement (Agarwal et al. 2021), 95% bootstrap CIs. Z inferred by linear regression on a reward-labelled dataset; BC regularization added for OGBench. Which ExORL buffer is used was not found in the text I read: unverified. — [arXiv 2510.00739](https://arxiv.org/abs/2510.00739)
- **Tirinzoni et al., FB-CPR / Meta Motivo (arXiv 2504.11054, ICLR 2025).** Humanoid whole-body control with motion-capture data, tasks: motion tracking, goal reaching, reward optimization. Not the ExORL protocol; cited as a comparable FB-family work and for its public model. Protocol details beyond the abstract not read: unverified. — [arXiv 2504.11054](https://arxiv.org/abs/2504.11054)

### Inferences
- Our setting (cheetah/walker/quadruped, RND, labelled states for z) matches Touati 2023 / HILP / Jeen. To be reviewer-proof use: first 5M RND transitions, 10,000 labelled states as the default, and report other N (e.g. 32 to 5120) as a sensitivity axis because FRE's 32 vs 5120 asymmetry shows labels count is a contested knob.
- Seeds differ widely (3 to 10); 10 seeds (Touati) is the safe reference, at least 5 (Jeen, FRE, TD-JEPA DMC).
- Selecting the best checkpoint by test return (Jeen) is a protocol weakness; report final-checkpoint scores (our `tag=final` rows) to avoid it.

### Gaps
- FRE's ExORL dataset choice and TD-JEPA's ExORL buffer: not found in extracted text.
- Venues for ExORL, FRE, TD-JEPA not verified here (ExORL was NeurIPS 2022 Datasets and Benchmarks to my memory, unverified; FRE ICML 2024 from memory, unverified).
- FB-CPR protocol beyond abstract not read.

## 2. Reported numbers (RND buffer unless noted)

### Takeaway
Verified numbers exist for FB and offline TD3 in Jeen's 100k table, for FB and HILP on the full 5M RND data in HILP's Table 3, and for FB on full RND in Jeen's Table 7. Touati's per-task offline TD3 values are in a figure only, so cannot be cited as numbers. Scores are raw returns (max 1000 for Walker, Cheetah, Quadruped).

### Cited Findings
- **Touati 2023, aggregate:** FB = 81% of offline TD3 (all tasks and buffers), 85% on RND. Table 2 per-task FB scores are given per buffer (column order Rand, AEnc, ICM, Latent, Trans, Lap, LRA-P, CL, LRA-SR, FB, APS). Example FB rows on the APS buffer: cheetah run 267+/-33, walk 844+/-51; walker stand 822+/-26 (these are from the APS block, not RND; the RND block ordering was not cleanly recovered from extraction, so RND per-task values here are unverified, read them from the paper's Table 2). — [arXiv 2209.14935](https://arxiv.org/abs/2209.14935)
- **HILP Table 3 (RND, 5M transitions, 4 seeds, mean +/- std), columns FB / HILP / Lap / GC-TD3:**

| Env | Task | FB | HILP | Lap | GC-TD3 |
|---|---|---|---|---|---|
| Walker | Flip | 548 +/- 94 | 563 +/- 136 | 563 +/- 40 | 298 +/- 110 |
| Walker | Run | 409 +/- 15 | 401 +/- 30 | 311 +/- 29 | 166 +/- 27 |
| Walker | Stand | 866 +/- 120 | 800 +/- 61 | 844 +/- 35 | 812 +/- 133 |
| Walker | Walk | 811 +/- 52 | 855 +/- 34 | 811 +/- 77 | 703 +/- 275 |
| Cheetah | Run | 183 +/- 83 | 262 +/- 53 | 124 +/- 32 | 65 +/- 66 |
| Cheetah | Run Backward | 153 +/- 41 | 187 +/- 55 | 113 +/- 37 | 50 +/- 35 |
| Cheetah | Walk | 636 +/- 291 | 823 +/- 141 | 571 +/- 111 | 281 +/- 116 |
| Cheetah | Walk Backward | 677 +/- 85 | 843 +/- 184 | 536 +/- 92 | 146 +/- 51 |
| Quadruped | Jump | 642 +/- 36 | 556 +/- 101 | 508 +/- 182 | 639 +/- 106 |
| Quadruped | Run | 436 +/- 26 | 393 +/- 42 | 346 +/- 81 | 435 +/- 38 |
| Quadruped | Stand | 797 +/- 72 | 810 +/- 97 | 681 +/- 191 | 910 +/- 44 |
| Quadruped | Walk | 642 +/- 202 | 542 +/- 32 | 441 +/- 88 | 470 +/- 21 |

  Source: [HILP arXiv 2402.15567, Table 3 appendix](https://arxiv.org/abs/2402.15567).
- **Jeen Table 7, full RND, FB (5 seeds, 95% CI)**: Walker walk 821 (758-883), stand 928 (925-930), run 281 (242-320), flip 525 (452-598); Quadruped stand 957, roll fast 574, roll 920, jump 736 (Jeen's quadruped tasks are stand, roll fast, roll, jump, escape: different from Touati's walk/run/jump/stand). Table 1: aggregate IQM on full RND: FB 389, VC-FB 390, MC-FB 396 (3 seeds per caption). — [arXiv 2309.15178](https://arxiv.org/abs/2309.15178)
- **Jeen Table 6, RND-100k (100k transitions), Walker, columns CQL / offline TD3 / GC-IQL / SF-LAP / FB / VC-FB:** walk 138 / 210 / 118 / 58 / 184 / 446; stand 386 / 362 / 284 / 190 / 558 / 624; run 71 / 84 / 51 / 34 / 101 / 179; flip 153 / 162 / 157 / 70 / 163 / 325. Quadruped RND-100k FB: stand 134, roll fast 83, roll 139, jump 121, escape 7 (offline TD3 119, 63, 96, 85, 3). — [arXiv 2309.15178](https://arxiv.org/abs/2309.15178)
- **TD-JEPA proprioceptive DMC (mean +/- s.e., 5 seeds):** DMC avg Laplacian 591.1, ICVF 619.3, HILP 620.1, FB 648.2, RLDP 610.2, BYOL 618.6, BYOL-gamma 645.4, TD-JEPA 661.2. Walker: FB 811.5, TD-JEPA 785.2. Cheetah: FB 672.7, TD-JEPA 688.7. Quadruped: FB 595.6, TD-JEPA 691.4. Pointmass: FB 513.0, TD-JEPA 479.3. (Mapping of the columns to the header order Laplacian, ICVF*, HILP, FB, RLDP, BYOL*, BYOL-gamma*, TD-JEPA; averaged over tasks, normalization by max 1000 in places.) — [arXiv 2510.00739, Table 1](https://arxiv.org/abs/2510.00739)
- **FRE Table 1 (normalized 0-100, 5 seeds, 20 episodes):** exorl-all FRE 51.5 +/- 6.3, FB 43.4 +/- 9.1, SF 40.9 +/- 1.9, OPAL 28.2 +/- 4.0; antmaze-all FRE 52.8, FB 25.8, SF 11.8; kitchen FRE 66, FB 3, SF 1. — [arXiv 2402.17135](https://arxiv.org/abs/2402.17135)
- **No TD3/CQL/IQL numbers on cheetah with RND in a table were found** except Jeen's 100k Walker/Quadruped/Maze/Jaco rows; Jeen has no Cheetah. ExORL's own offline TD3/CQL numbers are in figures of 2201.13425 (unverified as numbers).
- Not found in the sources read: CRL on ExORL cheetah beyond Touati's Table 2 CL column (CL is a contrastive-learning SF variant, scoring near 0 to 150 on several cheetah tasks in the APS/Proto blocks, e.g. APS cheetah run 2 +/- 0, walk 2 +/- 1); "CRL" as a goal-conditioned method is in OGBench instead.

### Inferences
- FB offline-TD3 gap: Jeen on 100k RND shows FB near or below offline TD3 on Walker (walk 184 vs 210) while VC-FB exceeds it; on full RND FB is much better. So "offline TD3" topline comparisons depend strongly on dataset size.
- HILP's GC-TD3 is not an offline-TD3 topline; do not present it as such. Use per-task offline TD3 on the same RND data (as in Touati) as the topline.
- Cheetah is where FB is weakest on RND in the HILP numbers (run 183, versus HILP 262), consistent with cheetah being the hard domain for FB-style methods.

### Gaps
- Touati Table 2 RND-block FB per task: block labelling ambiguous in extracted text; read directly in paper.
- Offline TD3 numbers per task on full RND: only in figures.
- ExORL paper's TD3/CQL numbers: figures only.

## 3. Antmaze / OGBench / reward specification

### Takeaway
Antmaze evaluations of zero-shot methods specify tasks as goals (HILP, OGBench, TD-JEPA) or as sampled reward functions evaluated at a few states (FRE). FB essentially fails on D4RL antmaze-large-diverse goal-reaching in HILP's setup (0.0).

### Cited Findings
- **HILP antmaze-large-diverse (D4RL v2, 8 seeds, return x100), Table 1:** GC-BC 15.0 +/- 9.3, GC-IQL 56.0 +/- 6.0, GC-CQL 36.2 +/- 19.0, FB 0.0, FDM 0.0, Lap 0.0, Rand 0.0, HILP 46.0 +/- 12.7, HILP-Plan 64.5 +/- 10.2. Goal specified by concatenating target x-y to the first observation's proprioceptive dims; reward 1 on reaching the goal. Also antmaze-ultra-{diverse, play}, kitchen. — [arXiv 2402.15567](https://arxiv.org/abs/2402.15567)
- **FRE on antmaze-large-diverse-v2:** reward families goal-reaching, directional, random simplex, path tasks; test rewards encoded from 32 labelled states; FB 25.8 and SF 11.8 average over antmaze tasks (FB goal-reaching 0.0 +/- 0, consistent with HILP). — [arXiv 2402.17135](https://arxiv.org/abs/2402.17135)
- **OGBench (Park et al., arXiv 2410.20092, ICLR 2025):** 8 seeds (4 for pixel tasks); success rate over five test-time goals per task; offers goal-conditioned and single-task ("singletask": five reward-based variants per task, 410 tasks in total). antmaze-large-navigate-v0 success rates: GCBC 24, GCIVL 16, GCIQL 34, QRL 75, CRL 83, HIQL 91; antmaze-large-stitch-v0: 3, 18, 7, 18, 11, 67; antmaze-large-explore-v0: 0, 10, 0, 0, 0, 4 (all +/- values omitted here). The paper also compares D4RL antmaze-large-diverse single-goal evaluation (41 GCBC, 64 GCIVL, 64 GCIQL, 37 QRL, 79 CRL, 87 HIQL) with multi-goal OGBench. — [arXiv 2410.20092](https://arxiv.org/abs/2410.20092)
- **TD-JEPA proprioceptive OGBench (10 seeds):** OGBench avg HILP 37.98 vs FB 39.04, TD-JEPA 37.98; antmaze-ln HILP 52.6, FB 36.8, TD-JEPA 57.2. (Column mapping per header Laplacian, ICVF*, HILP, FB, RLDP, BYOL*, BYOL-gamma*, TD-JEPA; the "OGBench (avg)" row printed under proprioception.) — [arXiv 2510.00739](https://arxiv.org/abs/2510.00739)

### Inferences
- For our antmaze-large-diverse run, compare to HILP's setup (D4RL v2, goal from first-state proprio + target xy, 8 seeds) and to FRE's reward-sample setup (32 states); FB's near-zero is the established baseline. A reviewer will ask which protocol, so state N labelled states and how reward is defined (sparse goal vs dense).

### Gaps
- HILP vs FRE use different antmaze scores (return x100 vs 0-100 normalized); not directly comparable.
- OGBench pointmaze/antmaze std values omitted here for brevity.

## 4. Reviewer expectations (statistical rigor, cost, fairness)

### Takeaway
Papers in this line now cite Agarwal et al. 2021 (IQM, stratified bootstrap CIs, probability of improvement). OpenReview reviews were not retrievable, so reviewer demands are inferred from what the papers themselves do and from protocol inconsistencies.

### Cited Findings
- Jeen et al. report IQM with 95% stratified bootstrap CIs per Agarwal et al. and all evaluation statistics in an appendix table (Table 8); HILP uses IQM with 95% bootstrap CIs; TD-JEPA uses probability of improvement and 95% bootstrap CIs. — [2309.15178](https://arxiv.org/abs/2309.15178), [2402.15567](https://arxiv.org/abs/2402.15567), [2510.00739](https://arxiv.org/abs/2510.00739)
- Test-time inference cost: FB/SF use linear regression on labelled states (10,000 in Touati/Jeen/HILP; 5120 in FRE's comparison) with no gradient updates; FRE uses 32 labelled pairs via a learned encoder; TD-JEPA's fine-tuning experiments start from zero-shot critic/actor and then run TD3 updates (offline or online). Compute for per-task offline RL vs one zero-shot model is documented in Jeen (4 GPU-h per task for single-task offline RL; FB about 4 h per domain). — [2309.15178](https://arxiv.org/abs/2309.15178), [2402.17135](https://arxiv.org/abs/2402.17135), [2510.00739](https://arxiv.org/abs/2510.00739)
- OpenReview reviews for these papers: not retrieved (searches did not return the pages; the NeurIPS 2024 poster page for Jeen links an OpenReview entry: [neurips.cc poster](https://neurips.cc/virtual/2024/poster/96479)). — search results only.

### Inferences (not sourced to reviews)
- Likely reviewer checks: (a) seeds >= 5 (ideally 10) and report IQM or mean with bootstrap CIs; (b) same dataset, same labels N and same evaluation episodes for baselines; (c) topline offline TD3 trained on the same data and steps; (d) final vs best checkpoint; (e) per-task numbers in an appendix; (f) wall-clock/FLOPs for MPPI planning at test time versus a forward pass of FB (our method does planning in latent space, which makes test-time cost a likely complaint).

### Gaps
- Actual reviewer comments: not found; check OpenReview for Jeen (NeurIPS 2024), HILP (ICML 2024), FB-CPR (ICLR 2025), OGBench (ICLR 2025), TD-JEPA (venue unverified).

## 5. Code repositories (GitHub API metadata read 2026-11-17)

### Takeaway
Most baselines are public; only some have a license.

### Cited Findings
- facebookresearch/controllable_agent: MIT, archived, last push 2023-07-17; FB / SF reference code for Touati et al. — [repo metadata via GitHub API](https://api.github.com/repos/facebookresearch/controllable_agent)
- facebookresearch/metamotivo (FB-CPR/Meta Motivo): license reported as NOASSERTION (custom license; read the LICENSE file), 811 stars, last push 2025-06-10. — [GitHub API](https://api.github.com/repos/facebookresearch/metamotivo)
- kvfrans/fre: no license declared, last push 2024-03-26. — [GitHub API](https://api.github.com/repos/kvfrans/fre)
- seohongpark/HILP: no license declared, last push 2025-09-29. — [GitHub API](https://api.github.com/repos/seohongpark/HILP)
- denisyarats/exorl: MIT, last push 2022-02-08 (the facebookresearch/exorl URL returned 404). — [GitHub API](https://api.github.com/repos/denisyarats/exorl)
- rll-research/url_benchmark (URLB): MIT, last push 2022-10-12. — [GitHub API](https://api.github.com/repos/rll-research/url_benchmark)
- seohongpark/ogbench: MIT, last push 2026-10-26. — [GitHub API](https://api.github.com/repos/seohongpark/ogbench)
- TD-JEPA repo: facebookresearch/tdjepa returned 404; actual repo location not found (unverified whether code is public).

### Inferences
- Reproducibility: controllable_agent is archived, so FB baselines should be run from it unmodified (pin the commit); HILP and FRE have no license, so reuse needs author permission or reimplementation.

### Gaps
- Not checked: whether each repo reproduces the paper numbers; README contents; Jeen's code location.
