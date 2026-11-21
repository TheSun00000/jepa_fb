"""wm_goal.py: goal-reaching planning in the latent space of a world model (evaluation only, no training).

    python wm_goal.py env_name=walker ckpt=models/<run>/fbwm_gphi_final.pickle
    python wm_goal.py env_name=antmaze_large ckpt=models/<run>/wm_final.pickle num_evals=10

Question: is the latent world model learned next to FB (fbwm.py) better than a reward-free JEPA world model alone (tdmpc_adapt.py
mode=pretrain) when it is used for goal reaching? Same planner, same tasks, only the checkpoint changes.

ckpt = a world-model checkpoint, either
  * `wm_final.pickle` of tdmpc_adapt.py mode=pretrain  ({'encoder', 'dynamics', ...}), or
  * `fbwm_<variant>_{final,latest,step:N}.pickle` of fbwm.py  ({'params': {'encoder', 'dynamics'}, ...}; only phi and the dynamics are used).
The encoder (TD-MPC MLP or fbwm's PhiMap trunk) and the dynamics sizes are read from the shapes of the saved parameters. phi_identity=true
for an fbwm gphi checkpoint trained with phi_identity (phi(s) = s, no sphere).

Latents: phi(s) is projected on the sphere of radius sqrt(latent_dim) (as in fbwm.py / tdmpc_adapt.py), and so is every imagined latent
h' = sphere(h + d(h, a)). (phi_identity: raw observation, no projection.)

Goal latent g:
  * antmaze_medium | antmaze_large | cube_single, goal_k=K > 0 (default K = 25; LeWM's protocol): goal_n (default 50) (start, goal)
    pairs from the dataset file (goal_split=val (default, the held-out file <name>-val.npz) | train). A pair is a dataset state s_t and
    the state s_{t+K} of the SAME trajectory. The simulator is reset to the stored (qpos, qvel) of s_t, and g = phi(s_{t+K}) (the full
    observation). Budget: goal_budget env steps (default 2K). Success: the goal coordinates come within goal_tol of those of s_{t+K}
    at any step (OGBench's criteria: antmaze (x, y) = obs[:2] within 0.5; cube block position obs[19:22] within 0.4 = 0.04 m, the
    observation is scaled by 10). Pairs whose goal coordinates move by less than goal_min_dist between s_t and s_{t+K} are excluded
    (default goal_tol: the goal is not reached at the start). The pairs depend only on pair_seed (default 0) and the dataset file,
    not on seed= or the checkpoint, so every checkpoint is evaluated on the same pairs. One task row, `goal_k<K>`, with goal_n
    episodes (one pair each). goal_n=1 goal_min_dist=4 is close to the earlier one-fixed-pair version (commit 5c8a16f).
    goal_k=0: the official tasks below.
  * antmaze_medium | antmaze_large | cube_single, goal_k=0: the official tasks goal_1..goal_5. One goal STATE per episode (the goal
    observation OGBench returns in `info['goal']` at reset(options={'task_id': i})), encoded: g = phi(s_goal). The episode starts at
    the task's own initial state.
  * cheetah | walker | quadruped: g = sum_s w(s) phi(s), w(s) = r(s) / sum r over n_labels dataset states labelled with the task reward
    (a fresh draw per episode, like z in fbwm.py). Negative rewards are clipped to 0; if all rewards are 0 the weights are uniform.

Planner: MPPI (TD-MPC settings), no policy and no value: each sampled action sequence is rolled out `horizon` steps through the
dynamics and scored with  -dist(h_H, g), the distance between the LAST imagined latent and the goal latent. The first action of the
MPPI mean is executed (re-planned at every step, mean warm-started).
dist=l2 (default) | cosine. On the sphere every h_H has the same norm, so l2 and cosine rank the candidates the same way (g is not
renormalised); they only differ in scale (temperature) and, without the sphere (phi_identity), in what they measure.

Metric (same as fbwm.py): dm_control: sum over the 1000 steps of the task reward of the visited states. Goal tasks (antmaze / cube):
1 if the goal was reached at least once (fbwm's reward: antmaze within 2 m of the goal xy, cube block within 0.04 m), else 0;
`<task>_ogb_success` is OGBench's own success flag (ever true during the episode). goal_k > 0: success as described above;
`<task>_gdist_{start,end,min}` = distance of the goal coordinates to the goal (mean over episodes), `<task>_reset_err` = max |obs after
the reset - stored s_t| (mean over episodes; checks the simulator reset), and goal_pairs.csv lists the pairs.

Flags (key=value): env_name= ckpt= [phi_identity=false] [num_evals=10] [n_labels=10000] [tasks=goal_1,goal_2 (default: all)]
  [horizon=5] [iterations=6] [num_samples=512] [num_elites=64] [temperature=0.5] [min_std=0.05] [init_std=2.0] [momentum=0.1]
  [dist=l2] [mppi_norm=false] [eval_chunk=<tasks rolled out in parallel>] [seed=] [name=<tag in the run folder name>]
  [dataset=explore|stitch|navigate (antmaze) | play|noisy (cube): names the env and, with goal_k > 0, the file the pairs come from]
  [ogbench_dir=] [goal_k=25 (0: official tasks)] [goal_n=50] [goal_split=val] [goal_budget=2K] [goal_tol=0.5 antmaze | 0.4 cube]
  [goal_min_dist=goal_tol] [pair_seed=0].
Writes logs/<timestamp>_WMGOAL-<src>-<env>/{eval_log.csv, episodes.csv, info.txt}.
"""
import os
LOCAL = 'nazim' in os.getcwd()
DATASET_FOLDER = '../offline_qd/datasets/d4rl/'
if not LOCAL:
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    DATASET_FOLDER = '../fre_experiments/datasets/'

import csv
import sys, shutil
import math
import time
import pickle
from datetime import datetime
from typing import Tuple

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.logs import parse_kwargs, resolve_seed, run_info, add_text_to_file


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath, self.fieldnames = filepath, fieldnames
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.filepath, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


def save_self_copy(dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(dst_dir, os.path.basename(__file__)))


def flag(kwargs, key, default=False):
    return str(kwargs[key]).lower() in ('1', 'true', 'yes') if key in kwargs else default


# ── Config ────────────────────────────────────────────────────────────────────
kwargs = parse_kwargs(sys.argv[1:])
print(kwargs)
print(jax.devices())

assert 'env_name' in kwargs and 'ckpt' in kwargs, "env_name= and ckpt= are required"
ENV_NAME = kwargs['env_name']
OGBENCH_ENVS = ['antmaze_medium', 'antmaze_large', 'cube_single']
assert ENV_NAME in ['cheetah', 'walker', 'quadruped'] + OGBENCH_ENVS
CKPT = kwargs['ckpt']
SEED = resolve_seed(kwargs)
PHI_IDENTITY = flag(kwargs, 'phi_identity')
NUM_EVALS = int(kwargs.get('num_evals', 10))          # episodes per task
N_LABELS = int(kwargs.get('n_labels', 10_000))        # dm_control: labelled states per goal latent
HORIZON = int(kwargs.get('horizon', 5))
ITERATIONS = int(kwargs.get('iterations', 6))
NUM_SAMPLES = int(kwargs.get('num_samples', 512))
NUM_ELITES = int(kwargs.get('num_elites', 64))
TEMPERATURE = float(kwargs.get('temperature', 0.5))
MIN_STD = float(kwargs.get('min_std', 0.05))
INIT_STD = float(kwargs.get('init_std', 2.0))
MOMENTUM = float(kwargs.get('momentum', 0.1))
DIST = str(kwargs.get('dist', 'l2'))
assert DIST in ('l2', 'cosine'), 'dist=l2|cosine'
MPPI_NORM = flag(kwargs, 'mppi_norm')                 # softmax on elite values divided by their std (scale-free temperature)
# Near goals (OGBench only, see the docstring): goal = the dataset state GOAL_K steps after the start, in the same trajectory.
# GOAL_K = 0: the official tasks.
assert ENV_NAME in OGBENCH_ENVS or 'goal_k' not in kwargs, 'goal_k is an OGBench option'
GOAL_K = int(kwargs.get('goal_k', 25)) if ENV_NAME in OGBENCH_ENVS else 0
assert GOAL_K >= 0
GOAL_N = int(kwargs.get('goal_n', 50))                # (start, goal) pairs = episodes of the goal_k task
GOAL_SPLIT = str(kwargs.get('goal_split', 'val'))     # dataset file the pairs come from
assert GOAL_SPLIT in ('val', 'train'), 'goal_split=val|train'
GOAL_BUDGET = int(kwargs.get('goal_budget', 2 * GOAL_K))
GOAL_TOL = float(kwargs.get('goal_tol', 0.4 if ENV_NAME == 'cube_single' else 0.5))   # OGBench's success radius (obs units)
GOAL_MIN_DIST = float(kwargs.get('goal_min_dist', GOAL_TOL))
PAIR_SEED = int(kwargs.get('pair_seed', 0))
GOAL_COORDS = slice(19, 22) if ENV_NAME == 'cube_single' else slice(0, 2)   # cube: block position; antmaze: ant (x, y)

# ── Checkpoint: phi (encoder) and the dynamics, sizes read from the parameter shapes ──────────────────────────────────────────────
with open(CKPT, 'rb') as f:
    ckpt = pickle.load(f)
SRC = 'fbwm' if 'params' in ckpt else 'wm'
enc_params = ckpt['params']['encoder'] if SRC == 'fbwm' else ckpt['encoder']
dyn_params = ckpt['params']['dynamics'] if SRC == 'fbwm' else ckpt['dynamics']
enc_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), enc_params)
dyn_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), dyn_params)

# ── Networks (the modules of fbwm.py, used for apply only) ────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """TD-MPC `mlp`: Linear -> ELU -> Linear -> ELU -> Linear."""
    hidden:    Tuple[int, ...]
    out_dim:   int
    zero_last: bool = False

    @nn.compact
    def __call__(self, x):
        for width in self.hidden:
            x = nn.elu(nn.Dense(width)(x))
        return nn.Dense(self.out_dim)(x)


class TorchDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        kernel = self.param('kernel', nn.initializers.zeros, (fan_in, self.features))
        bias   = self.param('bias', nn.initializers.zeros, (self.features,))
        return x @ kernel + bias


class NTanh(nn.Module):
    """FB 'ntanh': LayerNorm followed by Tanh."""
    @nn.compact
    def __call__(self, x):
        return jnp.tanh(nn.LayerNorm(epsilon=1e-5)(x))


class PhiMap(nn.Module):
    """fbwm.py gphi trunk: Linear -> NTanh -> [Linear -> relu] x (num_layers - 1) -> Linear."""
    hidden_dim: int  = 64
    out_dim:    int  = 50
    num_layers: int  = 2
    ntanh:      bool = True

    @nn.compact
    def __call__(self, obs):
        x = TorchDense(self.hidden_dim)(obs)
        x = NTanh()(x) if self.ntanh else nn.relu(x)
        for _ in range(self.num_layers - 1):
            x = nn.relu(TorchDense(self.hidden_dim)(x))
        return TorchDense(self.out_dim)(x)


def kernel_shape(p, name):
    return p[name]['kernel'].shape


if 'Dense_0' in enc_params:                                   # TD-MPC MLP encoder (tdmpc_adapt.py; fbwm sepB and frozen-wm gphi)
    OBS_DIM, LATENT_DIM = kernel_shape(enc_params, 'Dense_0')[0], kernel_shape(enc_params, 'Dense_2')[1]
    encoder_net = MLP(hidden=(kernel_shape(enc_params, 'Dense_0')[1], kernel_shape(enc_params, 'Dense_1')[1]), out_dim=LATENT_DIM)
else:                                                         # PhiMap trunk (fbwm gphi)
    n_dense = len([k for k in enc_params if k.startswith('TorchDense_')])
    OBS_DIM, LATENT_DIM = kernel_shape(enc_params, 'TorchDense_0')[0], kernel_shape(enc_params, f'TorchDense_{n_dense - 1}')[1]
    encoder_net = PhiMap(hidden_dim=kernel_shape(enc_params, 'TorchDense_0')[1], out_dim=LATENT_DIM,
                         num_layers=n_dense - 1, ntanh='NTanh_0' in enc_params)
DYN_IN, LATENT_DIM_DYN = kernel_shape(dyn_params, 'Dense_0')[0], kernel_shape(dyn_params, 'Dense_2')[1]
ACTION_DIM = DYN_IN - LATENT_DIM_DYN
dynamics_net = MLP(hidden=(kernel_shape(dyn_params, 'Dense_0')[1], kernel_shape(dyn_params, 'Dense_1')[1]), out_dim=LATENT_DIM_DYN)
if PHI_IDENTITY:
    assert LATENT_DIM_DYN == OBS_DIM, 'phi_identity: the dynamics must predict the raw observation'
    LATENT_DIM = OBS_DIM
assert LATENT_DIM_DYN == LATENT_DIM, f'encoder latent {LATENT_DIM} != dynamics latent {LATENT_DIM_DYN}'
print(f'checkpoint {CKPT} ({SRC}): obs_dim={OBS_DIM} latent_dim={LATENT_DIM} action_dim={ACTION_DIM} encoder={type(encoder_net).__name__}')

PARAMS = {'encoder': enc_params, 'dynamics': dyn_params}


def l2_normalize(x, eps=1e-8):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def encode(encoder_p, obs):
    """phi(s) on the sphere of radius sqrt(latent_dim) (raw observation with phi_identity)."""
    if PHI_IDENTITY:
        return obs
    return math.sqrt(LATENT_DIM) * l2_normalize(encoder_net.apply({'params': encoder_p}, obs))


def next_latent(params, h, a):
    """Residual dynamics h' = sphere(h + d(h, a))."""
    h_next = h + dynamics_net.apply({'params': params['dynamics']}, jnp.concatenate([h, a], axis=-1))
    return h_next if PHI_IDENTITY else math.sqrt(LATENT_DIM) * l2_normalize(h_next)


def latent_dist(h, goal):
    """(..., L), (L,) -> (...)"""
    if DIST == 'l2':
        return jnp.linalg.norm(h - goal, axis=-1)
    return 1.0 - (h * goal).sum(-1) / (jnp.linalg.norm(h, axis=-1) * jnp.linalg.norm(goal) + 1e-8)


# ── MPPI on the distance of the last imagined latent to the goal latent ───────────────────────────────────────────────────────────

def plan_single(params, obs, goal, prev_mean, use_prev, key):
    """One env, one goal latent. obs (obs_dim,), goal (L,), prev_mean (H, A). Returns (action, mean, spread)."""
    H, A, S = HORIZON, ACTION_DIM, NUM_SAMPLES
    h0   = encode(params['encoder'], obs[None])[0]
    h    = jnp.tile(h0, (S, 1))
    mean = jnp.where(use_prev, jnp.concatenate([prev_mean[1:], jnp.zeros((1, A))], axis=0), jnp.zeros((H, A)))
    std  = INIT_STD * jnp.ones((H, A))
    spread = jnp.zeros(())
    for _ in range(ITERATIONS):
        key, k_noise = jax.random.split(key)
        actions = jnp.clip(mean[:, None] + std[:, None] * jax.random.normal(k_noise, (H, S, A)), -1, 1)   # (H, S, A)
        hh = h
        for t in range(H):
            hh = next_latent(params, hh, actions[t])
        value = jnp.nan_to_num(-latent_dist(hh, goal))                                                     # (S,)
        elite_value, elite_idx = jax.lax.top_k(value, NUM_ELITES)
        elite_actions = actions[:, elite_idx]                                                              # (H, E, A)
        spread = elite_value.std()
        logits = TEMPERATURE * (elite_value - elite_value.max()) / ((spread + 1e-6) if MPPI_NORM else 1.0)
        score  = jax.nn.softmax(logits)
        _mean  = (score[None, :, None] * elite_actions).sum(axis=1)
        _std   = jnp.sqrt((score[None, :, None] * (elite_actions - _mean[:, None]) ** 2).sum(axis=1))
        mean, std = MOMENTUM * mean + (1 - MOMENTUM) * _mean, jnp.clip(_std, MIN_STD, 2.0)
    return mean[0], mean, spread                                    # eval: the MPPI mean, no sampling noise


@jax.jit
def plan_batch(params, obs, goals, prev_mean, use_prev, key):
    keys = jax.random.split(key, obs.shape[0])
    return jax.vmap(plan_single, in_axes=(None, 0, 0, 0, None, 0))(params, obs, goals, prev_mean, use_prev, keys)


@jax.jit
def encode_batch(params, obs):
    return encode(params['encoder'], obs)


# ── Logging ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name = f"WMGOAL-{SRC}{'-identity' if PHI_IDENTITY else ''}{('-' + str(kwargs['name'])) if 'name' in kwargs else ''}-{ENV_NAME}"
while True:   # two jobs started in the same second get the same name: append '_'
    LOGS_FOLDER = f'./logs/{date_time_str}_{exp_name}'
    try:
        os.makedirs(LOGS_FOLDER)
        break
    except FileExistsError:
        exp_name += '_'
print('LOGS_FOLDER:', LOGS_FOLDER)
save_self_copy(LOGS_FOLDER)

# ── Environment, tasks ────────────────────────────────────────────────────────────────────────────────────────────────────────────
INF = float('inf')
NUM_TRAJECTORIES = TRAJECTORY_LEN = None
is_goal_task = {}       # label -> bool (success = goal reached at least once)

if ENV_NAME in ['cheetah', 'walker', 'quadruped']:
    from dm_control import suite

    def _sigmoid(x, value_at_1, sigmoid):
        if sigmoid == 'linear':
            scaled = x * (1.0 - value_at_1)
            return jnp.where(jnp.abs(scaled) < 1, 1.0 - scaled, 0.0)
        if sigmoid == 'gaussian':
            scale = math.sqrt(-2 * math.log(value_at_1))
            return jnp.exp(-0.5 * (x * scale) ** 2)
        raise NotImplementedError(sigmoid)

    def tolerance(x, lower, upper, margin, sigmoid='linear', value_at_margin=0.0):
        in_bounds = jnp.logical_and(lower <= x, x <= upper)
        d = jnp.where(x < lower, lower - x, x - upper) / margin
        return jnp.where(in_bounds, 1.0, _sigmoid(d, value_at_margin, sigmoid))

    # Cheetah: aux = [horizontal_velocity, angmomentum]
    def cheetah_reward(aux, param):
        if param == 'flip':
            return tolerance(aux[..., 1], 5.0, INF, 5.0, 'linear', 0.0)
        velocity = aux[..., 0] * float(np.sign(param))
        bound    = float(abs(param))
        return tolerance(velocity, bound, INF, bound, 'linear', 0.0)

    # Walker: aux = [horizontal_velocity, torso_upright, torso_height, angmomentum]
    def walker_reward(aux, param):
        _STAND_HEIGHT = 1.2
        torso_upright, torso_height = aux[..., 1], aux[..., 2]
        standing     = tolerance(torso_height, _STAND_HEIGHT, INF, _STAND_HEIGHT / 2, 'gaussian', 0.1)
        upright      = (1 + torso_upright) / 2
        stand_reward = (3 * standing + upright) / 4
        if param == 'flip':
            move_reward = tolerance(aux[..., 3], 5.0, INF, 5.0, 'linear', 0.5)
        elif param == 0:
            move_reward = jnp.ones_like(stand_reward)
        else:
            move_reward = tolerance(aux[..., 0], float(param), INF, param / 2, 'linear', 0.5)
        return stand_reward * (5 * move_reward + 1) / 6

    # Quadruped: aux = [torso_velocity(3), torso_upright, com_height]
    def _quadruped_upright(aux):
        deviation = math.cos(math.radians(0))
        return tolerance(aux[..., 3], deviation, INF, 1 + deviation, 'linear', 0.0)

    def _quadruped_move(aux, desired_speed):
        return _quadruped_upright(aux) * tolerance(aux[..., 0], desired_speed, INF, desired_speed, 'linear', 0.5)

    def quadruped_stand(aux, _=None): return _quadruped_upright(aux)
    def quadruped_walk(aux, _=None):  return _quadruped_move(aux, 0.5)
    def quadruped_run(aux, _=None):   return _quadruped_move(aux, 5.0)
    def quadruped_jump(aux, _=None):
        return _quadruped_upright(aux) * tolerance(aux[..., 4], 1.0, INF, 1.0, 'linear', 0.5)

    if ENV_NAME == 'cheetah':
        STATE_DIM = 17
        NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1000
        dataset = np.load(f'{DATASET_FOLDER}/cheetah_rnd.npy', allow_pickle=True).item()
        aux = np.concatenate((np.load(f'{DATASET_FOLDER}/aux_cheetah.npy').reshape(10000, 1000, 1),
                              np.load(f'{DATASET_FOLDER}/angmomentum_aux_cheetah.npy').reshape(10000, 1000, 1)), axis=-1)
        benchmarks = [(cheetah_reward, 'vel10Back', -10), (cheetah_reward, 'vel2Back', -2), (cheetah_reward, 'vel2', 2),
                      (cheetah_reward, 'vel10', 10), (cheetah_reward, 'flip', 'flip')]
    elif ENV_NAME == 'walker':
        STATE_DIM = 24
        NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1000
        dataset = np.load(f'{DATASET_FOLDER}/walker_rnd.npy', allow_pickle=True).item()
        aux = np.concatenate((np.load(f'{DATASET_FOLDER}/aux_walker.npy').reshape(10000, 1000, 3),
                              np.load(f'{DATASET_FOLDER}/angmomentum_aux_walker.npy').reshape(10000, 1000, 1)), axis=-1)
        benchmarks = [(walker_reward, 'stand', 0), (walker_reward, 'walk', 2), (walker_reward, 'vel8', 8),
                      (walker_reward, 'run', 10), (walker_reward, 'flip', 'flip')]
    else:
        STATE_DIM = 78
        NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1001
        dataset = np.load(f'{DATASET_FOLDER}/quadruped_rnd_without_physics.npy', allow_pickle=True).item()
        aux = np.load(f'{DATASET_FOLDER}/aux_quadruped.npy', allow_pickle=True)
        benchmarks = [(quadruped_stand, 'stand', None), (quadruped_walk, 'walk', None),
                      (quadruped_run, 'run', None), (quadruped_jump, 'jump', None)]

    dataset_obs = np.asarray(dataset['observations']).reshape(-1, STATE_DIM)
    dataset_aux = np.asarray(aux).reshape(dataset_obs.shape[0], -1)
    del dataset, aux
    EVAL_STEPS = 1000

    def make_eval_env():
        return suite.load(domain_name=ENV_NAME, task_name='run', environment_kwargs=dict(flat_observation=True))

    def timestep2obs(timestep):
        return np.concatenate([v if len(v.shape) != 0 else v.reshape(-1) for k, v in timestep.observation.items()])

    def env_reset(e, task_idx, ep_idx=0):
        return timestep2obs(e.reset()), None          # (obs, goal observation: none, the goal latent comes from the labelled states)

    def env_step(e, action):
        return timestep2obs(e.step(action)), False

    def env_aux(e, state):
        physics = e.physics
        if ENV_NAME == 'walker':
            return np.array([physics.horizontal_velocity(), physics.torso_upright(), physics.torso_height(),
                             physics.named.data.subtree_angmom['torso'][1]])
        if ENV_NAME == 'cheetah':
            return np.array([physics.speed(), physics.named.data.subtree_angmom['torso'][1]])
        com_height = physics.named.data.sensordata['center_of_mass'].copy()[2]   # quadruped
        return np.concatenate([physics.torso_velocity(), np.array([physics.torso_upright()]), np.array([com_height])])

    for _, label, _ in benchmarks:
        is_goal_task[label] = False

else:  # OGBench: antmaze_medium | antmaze_large | cube_single. No dataset is needed: the goal is a state given by the env.
    import ogbench
    from utils.ogbench_tasks import goal_reaching_reward, cube_goal_reward, _cube_scale
    if ENV_NAME == 'cube_single':
        ogb_name, EVAL_STEPS = f"cube-single-{kwargs.get('dataset', 'play')}-v0", 200            # official episode length
    else:
        ogb_name, EVAL_STEPS = f"antmaze-{ENV_NAME.split('_')[1]}-{kwargs.get('dataset', 'explore')}-v0", 1000
    if GOAL_K == 0:
        _env0 = ogbench.make_env_and_datasets(ogb_name, env_only=True)
    else:
        import mujoco
        OGB_DIR = os.path.expanduser(kwargs.get('ogbench_dir', '~/.ogbench/data'))
        # add_info: the stored (qpos, qvel) of every state, to reset the simulator to it (the cube observation is not the full state)
        _env0, _ds_train, _ds_val = ogbench.make_env_and_datasets(ogb_name, dataset_dir=OGB_DIR, compact_dataset=True, add_info=True)
        _ds = _ds_val if GOAL_SPLIT == 'val' else _ds_train
        del _ds_train, _ds_val
        assert 'qpos' in _ds and 'qvel' in _ds, f'{ogb_name}: the dataset file has no qpos / qvel'
    STATE_DIM = _env0.observation_space.shape[0]
    assert _env0.action_space.shape[0] == ACTION_DIM, f'env action dim {_env0.action_space.shape[0]} != checkpoint {ACTION_DIM}'
    task_infos = _env0.unwrapped.task_infos
    if ENV_NAME == 'cube_single':     # success = block within 0.04 m of the goal block position (obs[19:22], OGBench-rescaled)
        benchmarks = [(cube_goal_reward, f'goal_{i + 1}', _cube_scale(t['goal_xyzs'][0])) for i, t in enumerate(task_infos)]
    else:                             # success = ant within 2 m of the goal (x, y) (fbwm.py's goal reward)
        benchmarks = [(goal_reaching_reward, f'goal_{i + 1}', np.array(t['goal_xy'])) for i, t in enumerate(task_infos)]
    if ENV_NAME.startswith('antmaze'):    # maze.csv: one row per cell (block centre x, y, wall flag, block size), for the trajectory figure
        _u = _env0.unwrapped
        with open(f'{LOGS_FOLDER}/maze.csv', 'w', newline='') as _f:
            _w = csv.writer(_f); _w.writerow(['i', 'j', 'x', 'y', 'wall', 'unit'])
            for _i in range(_u.maze_map.shape[0]):
                for _j in range(_u.maze_map.shape[1]):
                    _w.writerow([_i, _j, _j * _u._maze_unit - _u._offset_x, _i * _u._maze_unit - _u._offset_y,
                                 int(_u.maze_map[_i, _j]), _u._maze_unit])
    if GOAL_K > 0:
        # GOAL_N pairs (s_t, s_{t+K}) of the same trajectory, drawn with pair_seed only. Compact dataset: trajectories are stored back to
        # back and 'valids' is 0 on the last row of each one.
        _obs = np.asarray(_ds['observations'], dtype=np.float32)
        _end = (np.asarray(_ds['valids']) == 0).astype(np.int64)
        _ep = np.cumsum(_end) - _end                                       # trajectory id of every row
        _t = np.arange(len(_obs) - GOAL_K)
        _t = _t[_ep[_t] == _ep[_t + GOAL_K]]                               # start rows whose state K steps later is in the same trajectory
        _disp = np.linalg.norm(_obs[_t + GOAL_K, GOAL_COORDS] - _obs[_t, GOAL_COORDS], axis=-1)
        _cand = _t[_disp >= GOAL_MIN_DIST]
        print(f'goal_k={GOAL_K} ({GOAL_SPLIT} file): {len(_cand)} of {len(_t)} start states have a goal displacement >= {GOAL_MIN_DIST} '
              f'(displacement of all pairs: median {np.median(_disp):.3f}, 90th percentile {np.percentile(_disp, 90):.3f})')
        assert len(_cand) >= GOAL_N, f'only {len(_cand)} pairs with a goal displacement >= {GOAL_MIN_DIST} at goal_k={GOAL_K}'
        _pick = np.sort(np.random.default_rng(PAIR_SEED).choice(_cand, GOAL_N, replace=False))
        START_OBS, GOAL_OBS = _obs[_pick], _obs[_pick + GOAL_K]
        START_QPOS = np.asarray(_ds['qpos'])[_pick].astype(np.float64)
        START_QVEL = np.asarray(_ds['qvel'])[_pick].astype(np.float64)
        GOAL_COORD = GOAL_OBS[:, GOAL_COORDS].copy()                        # (GOAL_N, 2) antmaze (x, y) | (GOAL_N, 3) cube block
        benchmarks = [(None, f'goal_k{GOAL_K}', None)]                      # one task, one pair per episode
        NUM_EVALS, EVAL_STEPS = GOAL_N, GOAL_BUDGET
        with open(f'{LOGS_FOLDER}/goal_pairs.csv', 'w', newline='') as _f:
            _w = csv.writer(_f)
            _w.writerow(['episode', 'dataset_index', 'trajectory', 'k', 'start_coords', 'goal_coords', 'displacement'])
            for _e, _i in enumerate(_pick):
                _w.writerow([_e, int(_i), int(_ep[_i]), GOAL_K, ' '.join(f'{v:.4f}' for v in _obs[_i, GOAL_COORDS]),
                             ' '.join(f'{v:.4f}' for v in GOAL_COORD[_e]), float(np.linalg.norm(GOAL_COORD[_e] - _obs[_i, GOAL_COORDS]))])
        del _ds, _obs
    del _env0

    def make_eval_env():
        return ogbench.make_env_and_datasets(ogb_name, env_only=True)

    def env_reset(e, task_idx, ep_idx=0):
        if GOAL_K == 0:
            ob, info = e.reset(options=dict(task_id=task_idx + 1))     # the task's own initial state; info['goal'] = the goal observation
            return np.asarray(ob), np.asarray(info['goal'])
        # goal_k: the simulator is set to the stored state s_t of pair ep_idx; the goal observation is s_{t+K}
        e.reset(options=dict(task_id=1))                               # any task: the state is overwritten below
        u = e.unwrapped
        if ENV_NAME == 'cube_single':
            assert u._data.qpos.shape == START_QPOS[ep_idx].shape, (u._data.qpos.shape, START_QPOS[ep_idx].shape)
            u._data.qpos[:] = START_QPOS[ep_idx]
            u._data.qvel[:] = START_QVEL[ep_idx]
            # OGBench's own success flag (block within 0.04 m of the target mocap) then refers to this goal; obs = (x - 0.425, y, z) * 10
            u._data.mocap_pos[u._cube_target_mocap_ids[0]] = GOAL_COORD[ep_idx] / 10.0 + np.array([0.425, 0.0, 0.0])
            mujoco.mj_forward(u._model, u._data)
            u.pre_step()
            u.post_step()
            ob = u.compute_observation()
        else:
            u.set_state(START_QPOS[ep_idx], START_QVEL[ep_idx])
            u.set_goal(goal_xy=GOAL_COORD[ep_idx])                     # OGBench's own success flag (0.5) then refers to this goal
            ob = u.get_ob()
        return np.asarray(ob), GOAL_OBS[ep_idx]

    def env_step(e, action):
        ob, _, _, _, info = e.step(action)
        return np.asarray(ob), bool(info.get('success', False))

    def env_aux(e, state):
        return state

    for _, label, _ in benchmarks:
        is_goal_task[label] = True

assert STATE_DIM == OBS_DIM, f'env observation dim {STATE_DIM} != checkpoint encoder input {OBS_DIM}'
if 'tasks' in kwargs:
    wanted = str(kwargs['tasks']).split(',')
    assert all(w in is_goal_task for w in wanted), f'tasks= must be among {list(is_goal_task)}'
    task_ids = [i for i, (_, label, _) in enumerate(benchmarks) if label in wanted]
else:
    task_ids = list(range(len(benchmarks)))
EVAL_CHUNK = int(kwargs.get('eval_chunk', len(task_ids)))
print('tasks:', [benchmarks[i][1] for i in task_ids], '| episode steps:', EVAL_STEPS)

config = dict(goal_k=GOAL_K, goal_n=GOAL_N if GOAL_K else None, goal_split=GOAL_SPLIT if GOAL_K else None,
              goal_budget=GOAL_BUDGET if GOAL_K else None, goal_tol=GOAL_TOL if GOAL_K else None,
              goal_min_dist=GOAL_MIN_DIST if GOAL_K else None, pair_seed=PAIR_SEED if GOAL_K else None, src=SRC, ckpt=CKPT, phi_identity=PHI_IDENTITY, obs_dim=OBS_DIM, latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
              horizon=HORIZON, iterations=ITERATIONS, num_samples=NUM_SAMPLES, num_elites=NUM_ELITES, temperature=TEMPERATURE,
              min_std=MIN_STD, init_std=INIT_STD, momentum=MOMENTUM, dist=DIST, mppi_norm=MPPI_NORM, n_labels=N_LABELS,
              num_evals=NUM_EVALS, eval_steps=EVAL_STEPS)
with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
    file.write(str(kwargs) + '\n')
    file.write(str(config) + '\n')
add_text_to_file(f'{LOGS_FOLDER}/info.txt', run_info(globals(), z_labels=N_LABELS))

XY_TASKS = ENV_NAME.startswith('antmaze')   # antmaze: also log the real (x, y) distance (obs[:2]) of the ant to the goal xy, and its trajectory
eval_columns = ['mean_over_tasks']
for i in task_ids:
    label = benchmarks[i][1]
    eval_columns += [f'{label}_mean', f'{label}_std', f'{label}_dist_end', f'{label}_dist_min']
    eval_columns += [f'{label}_ogb_success'] if is_goal_task[label] else [f'{label}_goal_ess', f'{label}_goal_norm']
    if GOAL_K:
        eval_columns += [f'{label}_gdist_start', f'{label}_gdist_end', f'{label}_gdist_min', f'{label}_reset_err']
    if XY_TASKS:
        eval_columns += [f'{label}_xy_dist_start', f'{label}_xy_dist_end', f'{label}_xy_dist_min']
eval_logger = CSVLogger(f'{LOGS_FOLDER}/eval_log.csv', eval_columns + ['num_evals', 'tag', 'time'])
episode_logger = CSVLogger(f'{LOGS_FOLDER}/episodes.csv', ['task', 'episode', 'score', 'dist_end', 'dist_min']
                           + (['gdist_start', 'gdist_end', 'gdist_min', 'reset_err', 'ogb_success'] if GOAL_K else [])
                           + (['start_x', 'start_y', 'xy_dist_start', 'xy_dist_end', 'xy_dist_min'] if XY_TASKS else []))
traj_logger = CSVLogger(f'{LOGS_FOLDER}/trajectories.csv', ['task', 'episode', 't', 'x', 'y']) if XY_TASKS else None

# ── Goal latents ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)


def reward_weighted_goal(reward_fn, param):
    """dm_control: g = sum_s w(s) phi(s), w = r / sum r on n_labels fresh dataset states. Returns (g, effective sample size)."""
    idx = rng.integers(0, dataset_obs.shape[0], N_LABELS)
    obs, aux = jnp.asarray(dataset_obs[idx], dtype=jnp.float32), jnp.asarray(dataset_aux[idx], dtype=jnp.float32)
    w = jnp.clip(reward_fn(aux, param), 0.0, None)
    total = w.sum()
    w = jnp.where(total > 1e-8, w / jnp.maximum(total, 1e-8), jnp.ones_like(w) / w.shape[0])
    h = encode_batch(PARAMS, obs)
    return (w[:, None] * h).sum(axis=0), float(1.0 / (w ** 2).sum())


# ── Evaluation ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
eval_envs = []
key_state = {'key': jax.random.PRNGKey(SEED + 10_000)}


def next_key():
    key_state['key'], sub = jax.random.split(key_state['key'])
    return sub


def rollout_chunk(ids):
    """len(ids) * NUM_EVALS envs in lock-step; env (j, e) = episode e of task ids[j]. Returns obs (n,T,D), aux (n,T,.), goals (n,L),
    OGBench success flags (n,), goal ess (n,) (dm_control)."""
    n = len(ids) * NUM_EVALS
    while len(eval_envs) < n:
        eval_envs.append(make_eval_env())
    envs = eval_envs[:n]

    states, goal_latents, ess, ogb_success = [], [], [], np.zeros(n, dtype=bool)
    goal_obs = []
    for j, b in enumerate(ids):
        for e in range(NUM_EVALS):
            s, g_obs = env_reset(envs[j * NUM_EVALS + e], b, e)
            states.append(s)
            if g_obs is None:        # dm_control: reward-weighted average of the latents of labelled states
                g, es = reward_weighted_goal(benchmarks[b][0], benchmarks[b][2])
                goal_latents.append(g); ess.append(es)
            else:                    # OGBench: the encoded goal state
                goal_obs.append(g_obs)
    if goal_obs:
        goal_latents = list(encode_batch(PARAMS, jnp.asarray(np.stack(goal_obs), dtype=jnp.float32)))
    goals = jnp.stack(goal_latents)

    prev_mean = jnp.zeros((n, HORIZON, ACTION_DIM))
    obs_l, aux_l = [], []
    for step in tqdm(range(EVAL_STEPS), leave=False):
        obs = np.stack(states).astype(np.float32)
        obs_l.append(obs)
        aux_l.append(np.stack([env_aux(e, s) for e, s in zip(envs, states)]))
        actions, prev_mean, _ = plan_batch(PARAMS, jnp.asarray(obs), goals, prev_mean, jnp.asarray(step > 0), next_key())
        out = [env_step(e, a) for e, a in zip(envs, np.asarray(actions))]
        states = [o[0] for o in out]
        ogb_success |= np.array([o[1] for o in out])
    return np.stack(obs_l).transpose(1, 0, 2), np.stack(aux_l).transpose(1, 0, 2), goals, ogb_success, np.array(ess)


start_time = time.time()
row = {}
means = []
for start in range(0, len(task_ids), EVAL_CHUNK):
    ids = task_ids[start:start + EVAL_CHUNK]
    obs_traj, aux_traj, goals, ogb_success, ess = rollout_chunk(ids)
    for j, b in enumerate(ids):
        reward_fn, label, param = benchmarks[b]
        sl = slice(j * NUM_EVALS, (j + 1) * NUM_EVALS)
        if GOAL_K:        # one pair per episode: success = the goal coordinates within GOAL_TOL of the goal's at any step
            gdist = np.linalg.norm(obs_traj[sl][..., GOAL_COORDS] - GOAL_COORD[:, None, :], axis=-1)       # (E, T)
            scores = (gdist <= GOAL_TOL).any(axis=-1).astype(np.float64)
            reset_err = np.abs(obs_traj[sl][:, 0] - START_OBS).max(axis=-1)                               # (E,)
            row.update({f'{label}_gdist_start': float(gdist[:, 0].mean()), f'{label}_gdist_end': float(gdist[:, -1].mean()),
                        f'{label}_gdist_min': float(gdist.min(axis=-1).mean()), f'{label}_reset_err': float(reset_err.mean())})
            print(f'\tgoal-coordinate distance: start {gdist[:, 0].mean():.3f} | end {gdist[:, -1].mean():.3f} | '
                  f'min {gdist.min(axis=-1).mean():.3f} | reset error {reset_err.mean():.2e} (max {reset_err.max():.2e})')
        else:
            rewards = np.asarray(reward_fn(jnp.asarray(aux_traj[sl], dtype=jnp.float32), param))            # (E, T)
            scores = (rewards > 0).any(axis=-1).astype(np.float64) if is_goal_task[label] else rewards.sum(axis=-1)

        # distance (in the planner's metric) of the REAL visited states to the goal latent: does it track the task?
        h_traj = np.asarray(encode_batch(PARAMS, jnp.asarray(obs_traj[sl].reshape(-1, OBS_DIM))))
        h_traj = h_traj.reshape(NUM_EVALS, EVAL_STEPS, -1)
        dists = np.asarray(jax.vmap(jax.vmap(latent_dist, in_axes=(0, None)), in_axes=(0, 0))(jnp.asarray(h_traj), goals[sl]))
        dist_end, dist_min = dists[:, -1], dists.min(axis=-1)

        if XY_TASKS:      # real maze distance of the ant (obs[:2]) to the goal (x, y), same units as the success radius (2 m)
            xy = obs_traj[sl][..., :2]                                              # (E, T, 2)
            goal_xy = GOAL_COORD[:, None, :] if GOAL_K else np.asarray(param, dtype=np.float32)   # goal_k: one goal per episode
            xy_d = np.linalg.norm(xy - goal_xy, axis=-1)                            # (E, T)
            row.update({f'{label}_xy_dist_start': float(xy_d[:, 0].mean()), f'{label}_xy_dist_end': float(xy_d[:, -1].mean()),
                        f'{label}_xy_dist_min': float(xy_d.min(axis=-1).mean())})
            print('\t(x, y) distance to the goal: start', xy_d[:, 0].mean(), '| end', xy_d[:, -1].mean(), '| min', xy_d.min(axis=-1).mean())
            for e in range(NUM_EVALS):
                for t in range(EVAL_STEPS):
                    traj_logger.log({'task': label, 'episode': e, 't': t, 'x': float(xy[e, t, 0]), 'y': float(xy[e, t, 1])})
        print(label, ':')
        print('\tscores:', scores.tolist())
        print('\tmean:', scores.mean(), '| std:', scores.std(ddof=1) if NUM_EVALS > 1 else 0.0,
              '| final latent dist:', dist_end.mean(), '| min latent dist:', dist_min.mean())
        means.append(scores.mean())
        row.update({f'{label}_mean': float(scores.mean()), f'{label}_std': float(scores.std(ddof=1)) if NUM_EVALS > 1 else 0.0,
                    f'{label}_dist_end': float(dist_end.mean()), f'{label}_dist_min': float(dist_min.mean())})
        if is_goal_task[label]:
            row[f'{label}_ogb_success'] = float(ogb_success[sl].mean())
        else:
            row[f'{label}_goal_ess'] = float(ess[sl].mean())
            row[f'{label}_goal_norm'] = float(jnp.linalg.norm(goals[sl], axis=-1).mean())
        for e in range(NUM_EVALS):
            ep = {'task': label, 'episode': e, 'score': float(scores[e]), 'dist_end': float(dist_end[e]), 'dist_min': float(dist_min[e])}
            if GOAL_K:
                ep.update({'gdist_start': float(gdist[e, 0]), 'gdist_end': float(gdist[e, -1]), 'gdist_min': float(gdist[e].min()),
                           'reset_err': float(reset_err[e]), 'ogb_success': float(ogb_success[sl][e])})
            if XY_TASKS:
                ep.update({'start_x': float(xy[e, 0, 0]), 'start_y': float(xy[e, 0, 1]), 'xy_dist_start': float(xy_d[e, 0]),
                           'xy_dist_end': float(xy_d[e, -1]), 'xy_dist_min': float(xy_d[e].min())})
            episode_logger.log(ep)

row.update({'mean_over_tasks': float(np.mean(means)), 'num_evals': NUM_EVALS, 'tag': 'final', 'time': time.time() - start_time})
eval_logger.log(row)
print('mean over tasks:', row['mean_over_tasks'])
add_text_to_file(f'{LOGS_FOLDER}/info.txt', f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
