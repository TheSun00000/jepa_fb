"""wm_baselines.py: state-based versions of two reward-free latent world-model planners (LeWM, DINO-WM), as goal-reaching baselines.

    python wm_baselines.py model=lewm   env_name=antmaze_medium dataset=navigate seed=1        # train, then goal-reaching eval
    python wm_baselines.py model=dinowm env_name=cube_single dataset=play seed=1
    python wm_baselines.py model=lewm env_name=antmaze_medium ckpt=models/<run>/wm_base_final.pickle   # eval only

Both papers plan from pixels. Here the observation is the OGBench state vector, so each method keeps its training objective and its
planner, and the image encoder is replaced (2026-11-22, written for the "FBWM vs WM on goal reaching" comparison):

  model=lewm   (LeWorldModel, Maes et al. 2026, repo lucas-maes/le-wm): an encoder MLP on the standardised state, then LeWM's projector
               (Linear 2048 -> norm -> GELU -> Linear), giving a 192-d embedding. Predictor: AdaLN-zero blocks (depth 6, width 192,
               MLP 2048) conditioned on an embedding of the action block (Linear to 10 -> Linear 768 -> SiLU -> Linear 192, as LeWM's
               Embedder), then LeWM's pred_proj. Trained END TO END, no stop-gradient and no EMA:
                   loss = mean (pred - emb_next)^2 + sigreg_coef * SIGReg(emb),   sigreg_coef = 0.09 (LeWM's config)
               SIGReg = LeWM's code: 1024 random unit directions (redrawn every step), Epps-Pulley statistic of each 1-d projection
               against N(0, 1), trapezoid on 17 knots in [0, 3] with weight exp(-t^2/2), times the batch size, averaged over
               directions and frames; applied to the embeddings of every frame of the batch.
  model=dinowm (DINO-WM, Zhou et al. 2025): a FROZEN encoder, here the standardised state itself (DINO-WM's frozen DINOv2 features have
               no state counterpart; a fixed, untrained encoder is the part of the recipe that carries over). The same predictor (input
               and output projections to the state dimension), trained by teacher-forced MSE to the next frozen embedding.

Shared data and training (both): frames FRAMESKIP = 5 env steps apart, NUM_FRAMES = 4 frames per sample (teacher forcing: frames
0..2 predict 1..3, as LeWM), the action of a frame = the 5 raw actions until the next frame, concatenated (LeWM's action_block).
History of ONE frame (LeWM uses 3 for PushT / Cube and 1 for TwoRoom; an episode here starts from one state). AdamW (weight decay
1e-3), gradient clipping 1.0, linear warmup then cosine. Deviations from the papers: MLPs instead of ViTs, LayerNorm instead of
BatchNorm in the projector heads, learning rate (1e-4) and number of steps (300k, batch 128) chosen for state inputs.

Planner (both, LeWM's and DINO-WM's eval configs): CEM, 300 samples, top 30, 30 iterations, initial mean 0 and std 1 (actions clipped
to [-1, 1]), horizon 5 action blocks (25 env steps), cost = squared L2 distance between the LAST predicted embedding and the goal
embedding; the 5 planned blocks are all executed before replanning (receding horizon 5 blocks).

Goal-reaching eval: the near-goal pairs of wm_goal.py goal_k / fbwm.py fbgoal_k (utils/ogbench_tasks.sample_goal_pairs: goal_n pairs
(s_t, s_{t+K}) of the same trajectory from the -val file, pair_seed), the simulator set to the stored state s_t, budget 2K env steps,
success = goal coordinates (antmaze (x, y), cube block) within OGBench's radius (0.5 | 0.04 m) of those of s_{t+K} at any step.
eval_k=10,25,50,100,200 (one row of goal_eval.csv per K).

Flags (key=value): model=lewm|dinowm env_name=antmaze_medium|antmaze_large|cube_single [dataset=navigate|play ...] [ogbench_dir=]
  [seed=] [train_steps=300000] [batch_size=128] [lr=1e-4] [weight_decay=1e-3] [sigreg_coef=0.09] [num_proj=1024] [embed_dim=192]
  [frameskip=5] [num_frames=4] [ckpt= (eval only)] [eval_k=10,25,50,100,200] [goal_n=50] [goal_split=val] [pair_seed=0]
  [goal_min_dist=] [goal_tol=] [cem_samples=300] [cem_topk=30] [cem_iters=30] [cem_init_std=1.0] [plan_horizon=5] [receding=5]
  [planner=cem|mppi] [replan_every=<env steps; mppi 1, cem receding*frameskip>] (mppi, wm_goal.py's defaults:) [num_samples=512]
  [iterations=6] [num_elites=64] [temperature=0.5] [min_std=0.05] [init_std=2.0] [momentum=0.1] [mppi_norm=true]
planner=mppi (2026-11-22): the same-planner control. wm_goal.py's MPPI on this model's predictor and embedding, so the LeWM / DINO-WM
latents are compared with the FBWM / JEPA-WM latents under one planner; use plan_horizon=1 (one block = 5 env steps = wm_goal.py
horizon=5). goal_eval.csv model column: lewm-mppi / dinowm-mppi.
Writes logs/<timestamp>_WMBASE-<model>-<env>/{info.txt, training_log.csv, goal_eval.csv, goal_episodes.csv, goal_pairs_k<K>.csv}
and models/<same>/wm_base_final.pickle.
"""
import os
LOCAL = 'nazim' in os.getcwd()
if not LOCAL:
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

import csv
import sys, shutil
import time
import pickle
from datetime import datetime
from typing import Tuple

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import optax

from utils.logs import parse_kwargs, resolve_seed, run_info, mark_finished, add_text_to_file
from utils.ogbench_tasks import (sample_goal_pairs, describe_goal_pairs, write_goal_pairs, reset_to_goal_pair, default_goal_tol,
                                 goal_coords, OGBENCH_ENVS)


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath, self.fieldnames = filepath, fieldnames
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.filepath, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


# ── Config ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
kwargs = parse_kwargs(sys.argv[1:])
print(kwargs)
print(jax.devices())
ENV_NAME = kwargs['env_name']
assert ENV_NAME in OGBENCH_ENVS, f'env_name must be one of {OGBENCH_ENVS}'
MODEL = kwargs['model']
assert MODEL in ('lewm', 'dinowm'), 'model=lewm|dinowm'
SEED = resolve_seed(kwargs)
DATASET = str(kwargs.get('dataset', 'play' if ENV_NAME == 'cube_single' else 'navigate'))
OGB_DIR = os.path.expanduser(kwargs.get('ogbench_dir', '~/.ogbench/data'))
TRAIN_STEPS = int(kwargs.get('train_steps', 300_000))
BATCH_SIZE = int(kwargs.get('batch_size', 128))
LR = float(kwargs.get('lr', 1e-4))
WEIGHT_DECAY = float(kwargs.get('weight_decay', 1e-3))
SIGREG_COEF = float(kwargs.get('sigreg_coef', 0.09))
NUM_PROJ = int(kwargs.get('num_proj', 1024))
EMBED_DIM = int(kwargs.get('embed_dim', 192))
FRAMESKIP = int(kwargs.get('frameskip', 5))
NUM_FRAMES = int(kwargs.get('num_frames', 4))
CKPT = kwargs.get('ckpt')
EVAL_K = [int(k) for k in str(kwargs.get('eval_k', '10,25,50,100,200')).split(',') if k]
GOAL_N = int(kwargs.get('goal_n', 50))
GOAL_SPLIT = str(kwargs.get('goal_split', 'val'))
PAIR_SEED = int(kwargs.get('pair_seed', 0))
GOAL_MIN_DIST = float(kwargs['goal_min_dist']) if 'goal_min_dist' in kwargs else None
GOAL_TOL = float(kwargs['goal_tol']) if 'goal_tol' in kwargs else default_goal_tol(ENV_NAME)
CEM_SAMPLES = int(kwargs.get('cem_samples', 300))
CEM_TOPK = int(kwargs.get('cem_topk', 30))
CEM_ITERS = int(kwargs.get('cem_iters', 30))
CEM_INIT_STD = float(kwargs.get('cem_init_std', 1.0))
PLAN_H = int(kwargs.get('plan_horizon', 5))         # in action blocks
RECEDING = int(kwargs.get('receding', 5))           # action blocks executed before replanning
PLANNER_ = str(kwargs.get("planner", "cem"))
assert PLANNER_ == "mppi" or 1 <= RECEDING <= PLAN_H   # receding: CEM only (mppi replans every replan_every env steps)
PLANNER = str(kwargs.get('planner', 'cem'))         # cem (the papers') | mppi (wm_goal.py's MPPI, the same-planner control)
assert PLANNER in ('cem', 'mppi'), 'planner=cem|mppi'
MPPI_SAMPLES = int(kwargs.get('num_samples', 512))  # mppi: wm_goal.py's defaults
MPPI_ITERS = int(kwargs.get('iterations', 6))
MPPI_ELITES = int(kwargs.get('num_elites', 64))
MPPI_TEMP = float(kwargs.get('temperature', 0.5))
MPPI_MIN_STD = float(kwargs.get('min_std', 0.05))
MPPI_INIT_STD = float(kwargs.get('init_std', 2.0))
MPPI_MOMENTUM = float(kwargs.get('momentum', 0.1))
MPPI_NORM = str(kwargs.get('mppi_norm', 'true')).lower() in ('1', 'true')
REPLAN_EVERY = int(kwargs.get('replan_every', 1 if PLANNER == 'mppi' else RECEDING * FRAMESKIP))   # env steps between replans
assert 1 <= REPLAN_EVERY <= PLAN_H * FRAMESKIP
if CKPT is None and PLANNER == 'mppi':
    print('note: planner=mppi while training; usually run it eval-only with ckpt=')
LOG_EVERY = 1000

date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name = f"WMBASE-{MODEL}{'-mppi' if PLANNER == 'mppi' else ''}{('-' + str(kwargs['name'])) if 'name' in kwargs else ''}-{ENV_NAME}"
while True:   # two jobs started in the same second get the same name: append '_'
    LOGS_FOLDER, MODEL_SAVE_FOLDER = f'./logs/{date_time_str}_{exp_name}', f'./models/{date_time_str}_{exp_name}'
    try:
        os.makedirs(LOGS_FOLDER)
        break
    except FileExistsError:
        exp_name += '_'
os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
shutil.copy(os.path.abspath(__file__), os.path.join(LOGS_FOLDER, os.path.basename(__file__)))
print('LOGS_FOLDER:', LOGS_FOLDER)

# ── Data ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
import ogbench
if ENV_NAME == 'cube_single':
    ogb_name = f'cube-single-{DATASET}-v0'
else:
    ogb_name = f"antmaze-{ENV_NAME.split('_')[1]}-{DATASET}-v0"
_env0, _ds = ogbench.make_env_and_datasets(ogb_name, dataset_dir=OGB_DIR, compact_dataset=True)[:2]
STATE_DIM, ACTION_DIM = _env0.observation_space.shape[0], _env0.action_space.shape[0]
del _env0
OBS = np.asarray(_ds['observations'], dtype=np.float32)
ACT = np.asarray(_ds['actions'], dtype=np.float32)
_end = (np.asarray(_ds['valids']) == 0).astype(np.int64)     # compact dataset: 'valids' = 0 on the last row of each trajectory
_ep = np.cumsum(_end) - _end
del _ds
SPAN = FRAMESKIP * (NUM_FRAMES - 1)
STARTS = np.arange(len(OBS) - SPAN)
STARTS = STARTS[_ep[STARTS] == _ep[STARTS + SPAN]]           # samples whose NUM_FRAMES frames are in one trajectory
OBS_MEAN, OBS_STD = OBS.mean(0), OBS.std(0) + 1e-6
NUM_TRAJECTORIES = int(_end.sum())
print(f'{ogb_name}: {len(OBS)} rows, {NUM_TRAJECTORIES} trajectories, {len(STARTS)} training windows; state {STATE_DIM}, action {ACTION_DIM}')
ACT_BLOCK = FRAMESKIP * ACTION_DIM


# ── Networks ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    hidden: Tuple[int, ...]
    out_dim: int
    norm: bool = False

    @nn.compact
    def __call__(self, x):
        for w in self.hidden:
            x = nn.Dense(w)(x)
            if self.norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        return nn.Dense(self.out_dim)(x)


class Encoder(nn.Module):           # lewm: state MLP + LeWM's projector
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = MLP(hidden=(512, 512), out_dim=self.embed_dim)(x)
        return MLP(hidden=(2048,), out_dim=self.embed_dim, norm=True)(x)


class AdaLNBlock(nn.Module):        # AdaLN-zero (LeWM's ConditionalBlock without attention: one token per sample)
    dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, c):
        mod = nn.Dense(3 * self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(nn.silu(c))
        shift, scale, gate = jnp.split(mod, 3, axis=-1)
        h = nn.LayerNorm(use_scale=False, use_bias=False)(x) * (1 + scale) + shift
        h = nn.Dense(self.dim)(nn.gelu(nn.Dense(self.mlp_dim)(h)))
        return x + gate * h


class Predictor(nn.Module):
    dim: int
    out_dim: int
    depth: int = 6
    mlp_dim: int = 2048

    @nn.compact
    def __call__(self, e, a):
        c = nn.Dense(10)(a)                                          # LeWM's Embedder: Conv1d(k=1) to 10, Linear 4*dim, SiLU, Linear dim
        c = nn.Dense(self.dim)(nn.silu(nn.Dense(4 * self.dim)(c)))
        x = nn.Dense(self.dim)(e) if e.shape[-1] != self.dim else e
        for _ in range(self.depth):
            x = AdaLNBlock(self.dim, self.mlp_dim)(x, c)
        x = nn.LayerNorm()(x)
        return MLP(hidden=(2048,), out_dim=self.out_dim, norm=True)(x)  # LeWM's pred_proj


EMB = EMBED_DIM if MODEL == 'lewm' else STATE_DIM
enc_net = Encoder(EMBED_DIM) if MODEL == 'lewm' else None
pred_net = Predictor(dim=EMBED_DIM, out_dim=EMB)


def encode(params, obs_std):
    return enc_net.apply({'params': params['enc']}, obs_std) if MODEL == 'lewm' else obs_std


def predict(params, e, a):
    return pred_net.apply({'params': params['pred']}, e, a)


def standardise(obs):
    return (obs - OBS_MEAN) / OBS_STD


# ── SIGReg (LeWM's module.py, in JAX) ─────────────────────────────────────────────────────────────────────────────────────────────
_KNOTS = 17
_T = jnp.linspace(0.0, 3.0, _KNOTS)
_DT = 3.0 / (_KNOTS - 1)
_PHI = jnp.exp(-_T ** 2 / 2.0)
_W = jnp.full((_KNOTS,), 2 * _DT).at[0].set(_DT).at[-1].set(_DT) * _PHI


def sigreg(z, key):
    """z (T, B, D): Epps-Pulley statistic of M random 1-d projections against N(0, 1), per frame, averaged."""
    A = jax.random.normal(key, (z.shape[-1], NUM_PROJ))
    A = A / jnp.linalg.norm(A, axis=0, keepdims=True)
    x_t = (z @ A)[..., None] * _T                                    # (T, B, M, K)
    err = (jnp.cos(x_t).mean(-3) - _PHI) ** 2 + jnp.sin(x_t).mean(-3) ** 2   # (T, M, K)
    return ((err @ _W) * z.shape[-2]).mean()


# ── Training ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
OBS_J, ACT_J, STARTS_J = jnp.asarray(OBS), jnp.asarray(ACT), jnp.asarray(STARTS)
MEAN_J, STD_J = jnp.asarray(OBS_MEAN), jnp.asarray(OBS_STD)


def sample_batch(key):
    i = STARTS_J[jax.random.randint(key, (BATCH_SIZE,), 0, STARTS_J.shape[0])]
    idx = i[:, None] + FRAMESKIP * jnp.arange(NUM_FRAMES)[None]                                   # (B, T)
    obs = (OBS_J[idx] - MEAN_J) / STD_J                                                           # (B, T, D)
    a_idx = i[:, None, None] + FRAMESKIP * jnp.arange(NUM_FRAMES - 1)[None, :, None] + jnp.arange(FRAMESKIP)[None, None, :]
    act = ACT_J[a_idx].reshape(BATCH_SIZE, NUM_FRAMES - 1, ACT_BLOCK)                            # (B, T-1, F*A)
    return obs, act


def loss_fn(params, obs, act, key):
    e = encode(params, obs)                                          # (B, T, E)
    pred = predict(params, e[:, :-1], act)                           # (B, T-1, E)
    pred_loss = jnp.mean((pred - e[:, 1:]) ** 2)                     # lewm: no stop-gradient on the target (LeWM)
    sig = sigreg(jnp.swapaxes(e, 0, 1), key) if MODEL == 'lewm' else jnp.zeros(())
    loss = pred_loss + (SIGREG_COEF * sig if MODEL == 'lewm' else 0.0)
    return loss, dict(loss=loss, pred_loss=pred_loss, sigreg=sig, emb_std=e.std(axis=(0, 1)).mean(),
                      emb_norm=jnp.linalg.norm(e, axis=-1).mean())


schedule = optax.warmup_cosine_decay_schedule(0.0, LR, min(2000, TRAIN_STEPS // 10 + 1), max(TRAIN_STEPS, 2), LR * 0.01)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=WEIGHT_DECAY))


def init_params(key):
    k1, k2 = jax.random.split(key)
    p = {'pred': pred_net.init(k2, jnp.zeros((1, EMB)), jnp.zeros((1, ACT_BLOCK)))['params']}
    if MODEL == 'lewm':
        p['enc'] = enc_net.init(k1, jnp.zeros((1, STATE_DIM)))['params']
    return p


@jax.jit
def train_chunk(params, opt_state, key):
    def step(carry, k):
        params, opt_state = carry
        kb, ks = jax.random.split(k)
        obs, act = sample_batch(kb)
        (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(params, obs, act, ks)
        up, opt_state = optimizer.update(g, opt_state, params)
        return (optax.apply_updates(params, up), opt_state), m
    (params, opt_state), m = jax.lax.scan(step, (params, opt_state), jax.random.split(key, LOG_EVERY))
    return params, opt_state, jax.tree_util.tree_map(jnp.mean, m)


config = dict(model=MODEL, env=ENV_NAME, ogb_name=ogb_name, train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE, lr=LR,
              weight_decay=WEIGHT_DECAY, sigreg_coef=SIGREG_COEF if MODEL == 'lewm' else None, num_proj=NUM_PROJ, embed_dim=EMB,
              frameskip=FRAMESKIP, num_frames=NUM_FRAMES, eval_k=EVAL_K, goal_n=GOAL_N, goal_split=GOAL_SPLIT, pair_seed=PAIR_SEED,
              goal_tol=GOAL_TOL, cem_samples=CEM_SAMPLES, cem_topk=CEM_TOPK, cem_iters=CEM_ITERS, cem_init_std=CEM_INIT_STD,
              plan_horizon=PLAN_H, receding=RECEDING, ckpt=CKPT, planner=PLANNER, replan_every=REPLAN_EVERY,
              mppi=dict(num_samples=MPPI_SAMPLES, iterations=MPPI_ITERS, num_elites=MPPI_ELITES, temperature=MPPI_TEMP,
                        min_std=MPPI_MIN_STD, init_std=MPPI_INIT_STD, momentum=MPPI_MOMENTUM, mppi_norm=MPPI_NORM) if PLANNER == 'mppi' else None)
with open(f'{LOGS_FOLDER}/info.txt', 'a') as f:
    f.write(str(kwargs) + '\n')
    f.write(str(config) + '\n')
add_text_to_file(f'{LOGS_FOLDER}/info.txt', run_info(globals(), z_labels=0))

key = jax.random.PRNGKey(SEED)
key, k_init = jax.random.split(key)
if CKPT:
    with open(CKPT, 'rb') as f:
        saved = pickle.load(f)
    params = serialization.from_state_dict(init_params(k_init), saved['params'])
    assert np.allclose(saved['obs_mean'], OBS_MEAN) and np.allclose(saved['obs_std'], OBS_STD), 'dataset statistics differ from the checkpoint'
    print('loaded', CKPT)
else:
    params = init_params(k_init)
    print('parameters:', sum(x.size for x in jax.tree_util.tree_leaves(params)))
    opt_state = optimizer.init(params)
    tlog = CSVLogger(f'{LOGS_FOLDER}/training_log.csv', ['step', 'loss', 'pred_loss', 'sigreg', 'emb_std', 'emb_norm', 'time'])
    for c in tqdm(range(TRAIN_STEPS // LOG_EVERY)):
        t0 = time.time()
        key, sub = jax.random.split(key)
        params, opt_state, m = train_chunk(params, opt_state, sub)
        m = {k: float(v) for k, v in m.items()}
        tlog.log({'step': (c + 1) * LOG_EVERY, **m, 'time': time.time() - t0})
        if (c + 1) % 50 == 0:
            print(f"step {(c + 1) * LOG_EVERY}: " + ' '.join(f'{k} {v:.4g}' for k, v in m.items()))
    with open(f'{MODEL_SAVE_FOLDER}/wm_base_final.pickle', 'wb') as f:
        pickle.dump({'params': serialization.to_state_dict(jax.device_get(params)), 'obs_mean': OBS_MEAN, 'obs_std': OBS_STD,
                     'config': config}, f)
    print('saved', f'{MODEL_SAVE_FOLDER}/wm_base_final.pickle')


# ── CEM on the distance of the last predicted embedding to the goal embedding ─────────────────────────────────────────────────────
def cem_single(params, obs_std, goal_e, key):
    e0 = encode(params, obs_std[None])[0]
    mu, sd = jnp.zeros((PLAN_H, ACT_BLOCK)), CEM_INIT_STD * jnp.ones((PLAN_H, ACT_BLOCK))

    def it(carry, k):
        mu, sd = carry
        a = jnp.clip(mu + sd * jax.random.normal(k, (CEM_SAMPLES, PLAN_H, ACT_BLOCK)), -1.0, 1.0)
        e = jnp.tile(e0, (CEM_SAMPLES, 1))
        for h in range(PLAN_H):
            e = predict(params, e, a[:, h])
        cost = jnp.sum((e - goal_e) ** 2, axis=-1)
        _, idx = jax.lax.top_k(-cost, CEM_TOPK)
        el = a[idx]
        return (el.mean(0), jnp.maximum(el.std(0), 1e-3)), cost.min()

    (mu, _), best = jax.lax.scan(it, (mu, sd), jax.random.split(key, CEM_ITERS))
    return mu, best[-1]


@jax.jit
def cem_batch(params, obs_std, goal_e, key):
    return jax.vmap(cem_single, in_axes=(None, 0, 0, 0))(params, obs_std, goal_e, jax.random.split(key, obs_std.shape[0]))


# ── MPPI (planner=mppi): wm_goal.py's plan_single on this model's predictor ──────────────────────────────────────────────────────
# Same sampling, elites, temperature, momentum, std clipping, warm start (previous mean shifted by one ENV step) and score
# (-L2 distance of the last predicted embedding to the goal embedding, not squared) as wm_goal.py; the plan is PLAN_H action blocks
# (PLAN_H * FRAMESKIP env steps), so plan_horizon=1 is wm_goal.py's horizon=5. Executes the first action of the mean, replans every
# REPLAN_EVERY env steps (1 = wm_goal.py).
def mppi_single(params, obs_std, goal_e, prev_mean, use_prev, key):
    T, A, S = PLAN_H * FRAMESKIP, ACTION_DIM, MPPI_SAMPLES
    e0 = encode(params, obs_std[None])[0]
    mean = jnp.where(use_prev, prev_mean, jnp.zeros((T, A)))
    std = MPPI_INIT_STD * jnp.ones((T, A))
    for _ in range(MPPI_ITERS):
        key, k = jax.random.split(key)
        a = jnp.clip(mean[None] + std[None] * jax.random.normal(k, (S, T, A)), -1.0, 1.0)     # (S, T, A)
        ab = a.reshape(S, PLAN_H, ACT_BLOCK)
        e = jnp.tile(e0, (S, 1))
        for h in range(PLAN_H):
            e = predict(params, e, ab[:, h])
        value = jnp.nan_to_num(-jnp.linalg.norm(e - goal_e, axis=-1))
        ev, ei = jax.lax.top_k(value, MPPI_ELITES)
        ea = a[ei]                                                                             # (E, T, A)
        spread = ev.std()
        w = jax.nn.softmax(MPPI_TEMP * (ev - ev.max()) / ((spread + 1e-6) if MPPI_NORM else 1.0))
        _mean = (w[:, None, None] * ea).sum(0)
        _std = jnp.sqrt((w[:, None, None] * (ea - _mean[None]) ** 2).sum(0))
        mean, std = MPPI_MOMENTUM * mean + (1 - MPPI_MOMENTUM) * _mean, jnp.clip(_std, MPPI_MIN_STD, 2.0)
    return mean


@jax.jit
def mppi_batch(params, obs_std, goal_e, prev_mean, use_prev, key):
    return jax.vmap(mppi_single, in_axes=(None, 0, 0, 0, None, 0))(params, obs_std, goal_e, prev_mean, use_prev,
                                                                    jax.random.split(key, obs_std.shape[0]))


@jax.jit
def encode_batch(params, obs_std):
    return encode(params, obs_std)


# ── Goal-reaching eval on the goal_k pairs ────────────────────────────────────────────────────────────────────────────────────────
g_coords = goal_coords(ENV_NAME)
glog = CSVLogger(f'{LOGS_FOLDER}/goal_eval.csv', ['model', 'goal_k', 'n', 'goal_budget', 'goal_tol', 'success', 'success_std',
                                                  'gdist_start', 'gdist_end', 'gdist_min', 'reset_err', 'emb_dist_end', 'time'])
gep = CSVLogger(f'{LOGS_FOLDER}/goal_episodes.csv', ['goal_k', 'episode', 'success', 'gdist_start', 'gdist_end', 'gdist_min', 'reset_err'])
eval_envs = []
for K in EVAL_K:
    t0 = time.time()
    P = sample_goal_pairs(ENV_NAME, ogb_name, OGB_DIR, K, GOAL_N, GOAL_SPLIT, GOAL_MIN_DIST, PAIR_SEED)
    print(describe_goal_pairs(P))
    write_goal_pairs(f'{LOGS_FOLDER}/goal_pairs_k{K}.csv', P)
    while len(eval_envs) < GOAL_N:
        eval_envs.append(ogbench.make_env_and_datasets(ogb_name, env_only=True))
    envs = eval_envs[:GOAL_N]
    states = [reset_to_goal_pair(e, ENV_NAME, P['start_qpos'][i], P['start_qvel'][i], P['goal_coord'][i]) for i, e in enumerate(envs)]
    reset_err = np.abs(np.stack(states) - P['start_obs']).max(axis=-1)
    _bad = np.where(np.abs(np.stack(states) - P['start_obs']).max(axis=0) > 1e-3)[0]
    print('reset: observation dims with error > 1e-3:', {int(d): round(float(np.abs(np.stack(states) - P['start_obs'])[:, d].max()), 4) for d in _bad})
    goal_e = encode_batch(params, jnp.asarray(standardise(P['goal_obs'])))
    budget, coords, seq, t_plan = 2 * K, [], None, 0
    for step in tqdm(range(budget), leave=False):
        obs = np.stack(states).astype(np.float32)
        coords.append(obs[:, g_coords])
        if seq is None or step - t_plan >= REPLAN_EVERY:
            key, sub = jax.random.split(key)
            if PLANNER == 'cem':
                mu, _ = cem_batch(params, jnp.asarray(standardise(obs)), goal_e, sub)
                seq = np.asarray(mu).reshape(GOAL_N, PLAN_H * FRAMESKIP, ACTION_DIM)
            else:                                                    # warm start: the previous mean shifted by the steps executed
                prev = np.zeros((GOAL_N, PLAN_H * FRAMESKIP, ACTION_DIM), np.float32) if seq is None else np.concatenate(
                    [seq[:, step - t_plan:], np.zeros((GOAL_N, step - t_plan, ACTION_DIM), np.float32)], axis=1)
                seq = np.asarray(mppi_batch(params, jnp.asarray(standardise(obs)), goal_e, jnp.asarray(prev), seq is not None, sub))
            t_plan = step
        actions = seq[:, step - t_plan]
        states = [np.asarray(e.step(a)[0]) for e, a in zip(envs, actions)]
    final_e = encode_batch(params, jnp.asarray(standardise(np.stack(states).astype(np.float32))))
    gdist = np.linalg.norm(np.stack(coords) - P['goal_coord'][None], axis=-1).T                 # (GOAL_N, budget)
    succ = (gdist <= GOAL_TOL).any(axis=-1).astype(np.float64)
    glog.log({'model': MODEL + ('-mppi' if PLANNER == 'mppi' else ''), 'goal_k': K, 'n': GOAL_N, 'goal_budget': budget, 'goal_tol': GOAL_TOL, 'success': float(succ.mean()),
              'success_std': float(succ.std(ddof=1)), 'gdist_start': float(gdist[:, 0].mean()), 'gdist_end': float(gdist[:, -1].mean()),
              'gdist_min': float(gdist.min(axis=-1).mean()), 'reset_err': float(reset_err.mean()),
              'emb_dist_end': float(jnp.linalg.norm(final_e - goal_e, axis=-1).mean()), 'time': time.time() - t0})
    for i in range(GOAL_N):
        gep.log({'goal_k': K, 'episode': i, 'success': float(succ[i]), 'gdist_start': float(gdist[i, 0]),
                 'gdist_end': float(gdist[i, -1]), 'gdist_min': float(gdist[i].min()), 'reset_err': float(reset_err[i])})
    print(f'{MODEL} goal_k={K}: success {succ.mean():.2f} ({int(succ.sum())}/{GOAL_N}) | goal distance start {gdist[:, 0].mean():.3f} '
          f'min {gdist.min(axis=-1).mean():.3f} end {gdist[:, -1].mean():.3f} | reset error {reset_err.mean():.2e}')

mark_finished(f'{LOGS_FOLDER}/info.txt', 0 if CKPT else TRAIN_STEPS)
