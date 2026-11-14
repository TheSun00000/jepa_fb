"""Offline TD3 on the true reward of one task (task=): the policy topline for fbwm (2026-11-14).

Same data, reward labelling (r_t = R(aux_{t+1})), envs and eval as tdmpc_jax.py; networks in the style of fb_jax.py's actor
and critic without z. Plain TD3: clipped double Q, target policy smoothing, delayed actor and target updates, no BC term.
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

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from tqdm import tqdm

import csv
import sys, shutil
from datetime import datetime

import math
import time
import pickle

import dataclasses
from functools import partial
from typing import Any, Optional, Tuple

import jax
import flax
from flax import serialization
import jax.numpy as jnp
import optax

from utils.logs import parse_kwargs, resolve_seed, run_info, mark_finished, add_text_to_file


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        # Write header only if file doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_self_copy(dst_dir="backups"):
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.abspath(__file__)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy(src, dst)
    return dst


if __name__ == '__main__':
    #######################################################################################

    kwargs = parse_kwargs(sys.argv[1:])
    print(kwargs)
    print(jax.devices())

    assert 'env_name' in kwargs, "env_name= is required"
    assert 'task' in kwargs, "task= is required (one TD3 agent is trained per benchmark task, e.g. task=walk)"
    ENV_NAME = kwargs['env_name']
    TASK     = str(kwargs['task'])
    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze']

    # ── Config ────────────────────────────────────────────────────────────────────
    # cheetah | walker | quadruped (dm_control)  ;  antmaze (d4rl antmaze-large-diverse-v2)
    SEED = resolve_seed(kwargs)      # random unless seed= is given; recorded in info.txt

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = f'TD3-{ENV_NAME}-{TASK}'

    while True:   # two jobs started in the same second get the same name: append '_' (runs_index strips it)
        LOGS_FOLDER       = f'./logs/{date_time_str}_{exp_name}'
        MODEL_SAVE_FOLDER = f'./models/{date_time_str}_{exp_name}'
        try:
            os.makedirs(LOGS_FOLDER)
            break
        except FileExistsError:
            exp_name += '_'
    os.makedirs(MODEL_SAVE_FOLDER)
    print('LOGS_FOLDER      :', LOGS_FOLDER)
    print('MODEL_SAVE_FOLDER:', MODEL_SAVE_FOLDER)

    save_self_copy(dst_dir=LOGS_FOLDER)

    TRAIN_METRICS = ["critic_loss", "actor_loss", "q_mean", "target_q_mean", "critic_grad_norm", "actor_grad_norm"]
    train_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/training_log.csv",
        fieldnames=["step"] + TRAIN_METRICS + ["time"],
    )

    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(kwargs) + '\n')

    # ── Reward functions (jnp ports of new_utils/benchmark.py and utils/antmaze_benchmark.py) ──
    # Every reward function has the signature  fn(aux, param) -> array of shape aux.shape[:-1]

    INF = float('inf')

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
            _SPIN_SPEED = 5.0
            return tolerance(aux[..., 1], _SPIN_SPEED, INF, _SPIN_SPEED, 'linear', 0.0)
        velocity = aux[..., 0] * float(np.sign(param))
        bound    = float(abs(param))
        return tolerance(velocity, bound, INF, bound, 'linear', 0.0)

    # Walker: aux = [horizontal_velocity, torso_upright, torso_height, angmomentum]
    def walker_reward(aux, param):
        _STAND_HEIGHT = 1.2
        torso_upright, torso_height = aux[..., 1], aux[..., 2]
        standing    = tolerance(torso_height, _STAND_HEIGHT, INF, _STAND_HEIGHT / 2, 'gaussian', 0.1)
        upright     = (1 + torso_upright) / 2
        stand_reward = (3 * standing + upright) / 4
        if param == 'flip':
            _SPIN_SPEED = 5.0
            move_reward = tolerance(aux[..., 3], _SPIN_SPEED, INF, _SPIN_SPEED, 'linear', 0.5)
        elif param == 0:   # standing: ignore `move_reward`
            move_reward = jnp.ones_like(stand_reward)
        else:
            move_reward = tolerance(aux[..., 0], float(param), INF, param / 2, 'linear', 0.5)
        return stand_reward * (5 * move_reward + 1) / 6

    # Quadruped: aux = [torso_velocity(3), torso_upright, com_height]
    def _quadruped_upright(aux):
        deviation = math.cos(math.radians(0))
        return tolerance(aux[..., 3], deviation, INF, 1 + deviation, 'linear', 0.0)

    def _quadruped_move(aux, desired_speed):
        move = tolerance(aux[..., 0], desired_speed, INF, desired_speed, 'linear', 0.5)
        return _quadruped_upright(aux) * move

    def quadruped_stand(aux, _=None): return _quadruped_upright(aux)
    def quadruped_walk(aux, _=None):  return _quadruped_move(aux, 0.5)
    def quadruped_run(aux, _=None):   return _quadruped_move(aux, 5.0)
    def quadruped_jump(aux, _=None):
        _desired_height = 1.0
        jump_up = tolerance(aux[..., 4], _desired_height, INF, _desired_height, 'linear', 0.5)
        return _quadruped_upright(aux) * jump_up

    # Antmaze: aux = full observation (29 dims)
    XY_VEL_SCALE = 0.33820298

    def goal_reaching_reward(state, goal):
        dist = jnp.linalg.norm(state[..., :2] - jnp.asarray(goal, dtype=jnp.float32), axis=-1)
        return (dist < 2).astype(jnp.float32) * 2 - 1

    def velocity_reward(states, direction):
        xy_vels = states[..., 15:17] * XY_VEL_SCALE
        return jnp.sum(xy_vels * jnp.asarray(direction, dtype=jnp.float32), axis=-1)

    def _grid_indices(states, y_max):
        x = jnp.clip(states[..., 0].astype(jnp.int32), 0, 35)   # int cast truncates toward 0, like torch .long()
        y = jnp.clip(states[..., 1].astype(jnp.int32), 0, y_max)
        return x, y

    def make_simplex_reward(num_simplex):
        import opensimplex
        pos  = np.zeros((num_simplex, 36, 25))
        xvel = np.zeros((num_simplex, 36, 25))
        yvel = np.zeros((num_simplex, 36, 25))
        print("Generating simplex seeds")
        xi, yi = np.arange(36), np.arange(25)
        for r in tqdm(range(num_simplex)):
            opensimplex.seed(r)
            pos[r]  = opensimplex.noise2array(x=xi / 20.0, y=yi / 20.0).T
            opensimplex.seed(r + num_simplex)
            xvel[r] = opensimplex.noise2array(x=xi / 20.0, y=yi / 20.0).T
            opensimplex.seed(r + num_simplex * 2)
            yvel[r] = opensimplex.noise2array(x=xi / 20.0, y=yi / 20.0).T
        xvel[np.abs(xvel) < 0.5] = 0
        yvel[np.abs(yvel) < 0.5] = 0
        pos_mask = jnp.asarray(pos > 0.3, dtype=jnp.float32)   # thresholded in float64, as in torch
        xvel, yvel = jnp.asarray(xvel, dtype=jnp.float32), jnp.asarray(yvel, dtype=jnp.float32)

        def simplex_reward(states, simplex_id):
            x, y = _grid_indices(states, 24)
            rews = -1 + pos_mask[simplex_id, x, y] * 0.5
            xy_vels = states[..., 15:17] * XY_VEL_SCALE
            return rews + xy_vels[..., 0] * xvel[simplex_id, x, y] + xy_vels[..., 1] * yvel[simplex_id, x, y]
        return simplex_reward

    def make_matrix_reward(pos, xvel, yvel, edges=False):
        pos_mask = jnp.asarray(pos > 0.3, dtype=jnp.float32)
        xvel, yvel = jnp.asarray(xvel, dtype=jnp.float32), jnp.asarray(yvel, dtype=jnp.float32)

        def matrix_reward(states, _=None):
            x, y = _grid_indices(states, 23)
            if edges:
                return pos_mask[x, y] * 2 - 1
            xy_vels = states[..., 15:17] * XY_VEL_SCALE
            return pos_mask[x, y] * 0.5 + xy_vels[..., 0] * xvel[x, y] + xy_vels[..., 1] * yvel[x, y]
        return matrix_reward

    def matrices():
        return [np.zeros((36, 25)) for _ in range(3)]

    def path_matrices():   # TestRewPath
        pos, xvel, yvel = matrices()
        pos[3:21, 7:10] = 1;  xvel[3:21, 7:10] = -1
        pos[0:3, 3:10] = 1;   yvel[0:3, 3:10] = -1
        pos[0:18, 0:3] = 1;   xvel[0:18, 0:3] = 1
        return pos, xvel, yvel

    def loop_matrices():   # TestRewLoop
        pos, xvel, yvel = matrices()
        pos[22:33, 14:18] = 1; xvel[22:33, 14:18] = -1
        pos[21:, 0:3] = 1;     xvel[21:, 0:3] = 1
        pos[33:, 3:18] = 1;    yvel[33:, 3:18] = 1
        pos[18:21, 0:7] = 1;   yvel[18:21, 0:7] = -1
        return pos, xvel, yvel

    def edges_matrices():  # TestRewMatrixEdges
        pos, xvel, yvel = matrices()
        pos[:3, :] = 1; pos[-3:, :] = 1; pos[:, :3] = 1; pos[:, -3:] = 1
        return pos, xvel, yvel

    # ── Environment & dataset setup ───────────────────────────────────────────────

    if ENV_NAME in ['cheetah', 'walker', 'quadruped']:
        from dm_control import suite

        if ENV_NAME == 'cheetah':
            STATE_DIM, ACTION_DIM = 17, 6
            NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1000
            dataset = np.load(f'{DATASET_FOLDER}/cheetah_rnd.npy', allow_pickle=True).item()
            aux = np.concatenate((
                np.load(f'{DATASET_FOLDER}/aux_cheetah.npy').reshape(10000, 1000, 1),
                np.load(f'{DATASET_FOLDER}/angmomentum_aux_cheetah.npy').reshape(10000, 1000, 1),
            ), axis=-1)
            benchmarks = [
                (cheetah_reward, 'vel10Back', -10),
                (cheetah_reward, 'vel2Back',   -2),
                (cheetah_reward, 'vel2',        2),
                (cheetah_reward, 'vel10',      10),
                (cheetah_reward, 'flip',   'flip'),
            ]

        elif ENV_NAME == 'walker':
            STATE_DIM, ACTION_DIM = 24, 6
            NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1000
            dataset = np.load(f'{DATASET_FOLDER}/walker_rnd.npy', allow_pickle=True).item()
            aux = np.concatenate((
                np.load(f'{DATASET_FOLDER}/aux_walker.npy').reshape(10000, 1000, 3),
                np.load(f'{DATASET_FOLDER}/angmomentum_aux_walker.npy').reshape(10000, 1000, 1),
            ), axis=-1)
            benchmarks = [
                (walker_reward, 'stand',  0),
                (walker_reward, 'walk',   2),
                (walker_reward, 'vel8',   8),
                (walker_reward, 'run',   10),
                (walker_reward, 'flip', 'flip'),
            ]

        elif ENV_NAME == 'quadruped':
            STATE_DIM, ACTION_DIM = 78, 12
            NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1001
            dataset = np.load(f'{DATASET_FOLDER}/quadruped_rnd_without_physics.npy', allow_pickle=True).item()
            aux = np.load(f'{DATASET_FOLDER}/aux_quadruped.npy', allow_pickle=True)
            benchmarks = [
                (quadruped_stand, 'stand', None),
                (quadruped_walk,  'walk',  None),
                (quadruped_run,   'run',   None),
                (quadruped_jump,  'jump',  None),
            ]

        dataset_obs     = np.asarray(dataset['observations']).reshape(-1, STATE_DIM)
        dataset_actions = np.asarray(dataset['actions']).reshape(-1, ACTION_DIM)
        dataset_aux     = np.asarray(aux).reshape(dataset_obs.shape[0], -1)
        del dataset, aux

        EVAL_STEPS = 1000

        def make_eval_env():
            return suite.load(domain_name=ENV_NAME, task_name='run',
                              environment_kwargs=dict(flat_observation=True))

        def timestep2obs(timestep):
            return np.concatenate([v if len(v.shape) != 0 else v.reshape(-1)
                                   for k, v in timestep.observation.items()])

        def env_reset(e):
            return timestep2obs(e.reset())

        def env_step(e, action):
            return timestep2obs(e.step(action))

        def env_aux(e, state):
            physics = e.physics
            if ENV_NAME == 'walker':
                return np.array([physics.horizontal_velocity(), physics.torso_upright(), physics.torso_height(),
                                 physics.named.data.subtree_angmom['torso'][1]])
            if ENV_NAME == 'cheetah':
                return np.array([physics.speed(), physics.named.data.subtree_angmom['torso'][1]])
            com_height = physics.named.data.sensordata['center_of_mass'].copy()[2]   # quadruped
            return np.concatenate([physics.torso_velocity(), np.array([physics.torso_upright()]), np.array([com_height])])

        def env_physics(e):
            return e.physics.get_state()

    else:  # antmaze
        import gym
        import d4rl   # registers the antmaze environments

        STATE_DIM, ACTION_DIM = 29, 8
        NUM_TRAJECTORIES, TRAJECTORY_LEN = 999, 1001
        env = gym.make('antmaze-large-diverse-v2')
        dataset = env.get_dataset()

        N = NUM_TRAJECTORIES * TRAJECTORY_LEN
        dataset_obs     = np.asarray(dataset['observations'])[:N].reshape(-1, STATE_DIM)
        dataset_actions = np.asarray(dataset['actions'])[:N].reshape(-1, ACTION_DIM)
        dataset_aux     = dataset_obs       # rewards are computed from the full observation
        del dataset

        simplex_reward = make_simplex_reward(num_simplex=10)
        benchmarks = [
            (goal_reaching_reward, 'goal_bottom', np.array([28, 0])),
            (goal_reaching_reward, 'goal_left',   np.array([0, 15])),
            (goal_reaching_reward, 'goal_top',    np.array([35, 24])),
            (goal_reaching_reward, 'goal_center', np.array([12, 24])),
            (goal_reaching_reward, 'goal_right',  np.array([33, 16])),
            (velocity_reward, 'vel_left',  [-1, 0]),
            (velocity_reward, 'vel_up',    [0, 1]),
            (velocity_reward, 'vel_down',  [0, -1]),
            (velocity_reward, 'vel_right', [1, 0]),
            (simplex_reward, 'simplex_1', 1),
            (simplex_reward, 'simplex_2', 2),
            (simplex_reward, 'simplex_3', 3),
            (simplex_reward, 'simplex_4', 4),
            (simplex_reward, 'simplex_5', 5),
            (make_matrix_reward(*path_matrices()),          'path_center', None),
            (make_matrix_reward(*loop_matrices()),          'path_loop',   None),
            (make_matrix_reward(*edges_matrices(), edges=True), 'path_edges',  None),
        ]

        EVAL_STEPS = 2000

        def make_eval_env():
            return gym.make('antmaze-large-diverse-v2')

        def env_reset(e):
            e.reset()
            location = (20, 15)
            # reset_to_location_antmaze
            e.sim.reset()
            qpos = e.init_qpos + e.np_random.uniform(low=-.1, high=.1, size=e.model.nq)
            qpos[:2] = np.array(location).astype(e.observation_space.dtype)
            qvel = e.init_qvel + e.np_random.randn(e.model.nv) * .1
            e.set_state(qpos, qvel)
            return e.unwrapped._get_obs()

        def env_step(e, action):
            new_state, _, _, _ = e.step(action)
            return new_state

        def env_aux(e, state):
            return state

        def env_physics(e):
            return None

        def add_largest_maze_walls(ax):
            maze_optim = [
                (1, 1, 1, 2), (0, 4, 2, 1), (3, 1, 1, 4), (5, 0, 1, 1), (4, 2, 3, 1), (1, 6, 3, 1),
                (4, 4, 2, 1), (5, 6, 2, 1), (1, 8, 1, 1), (3, 7, 1, 2), (5, 8, 1, 2),
            ]
            block_size = 0.025 * 80
            height, width = 7, 10
            torso_x, torso_y = (width - 1) * block_size, (height - 1) * block_size
            for (y, x, w, h) in maze_optim:
                x = x * block_size * 2 - torso_x + (h - 1) * block_size - h * block_size + 18
                y = y * block_size * 2 - torso_y + (w - 1) * block_size - w * block_size + 12
                h, w = h * block_size * 2, w * block_size * 2
                ax.add_patch(patches.Rectangle((x, y), h, w, linewidth=2, edgecolor='gray', facecolor='gray'))

    # ── Task selection: one model per benchmark task ──────────────────────────────

    labels = [b[1] for b in benchmarks]
    assert TASK in labels, f"task must be one of {labels}"
    reward_fn, _, reward_param = benchmarks[labels.index(TASK)]
    print('TASK:', TASK, '| reward param:', reward_param)

    NUM_EVALS = 5             # episodes per periodic eval (tag 'eval'), rolled out in parallel
    NUM_EVALS_FINAL = int(kwargs['num_evals_final']) if 'num_evals_final' in kwargs else 10   # final eval (tag 'final'), in rollouts of NUM_EVALS
    assert NUM_EVALS_FINAL % NUM_EVALS == 0, 'num_evals_final must be a multiple of 5'
    print(NUM_TRAJECTORIES, TRAJECTORY_LEN, NUM_TRAJECTORIES * TRAJECTORY_LEN)

    BATCH_SIZE      = int(kwargs['batch_size']) if 'batch_size' in kwargs else 1024
    EVAL_EACH_STEPS = 10      # in outer iterations (1 outer iteration = SCAN_LENGTH updates)
    SCAN_LENGTH     = 1_000
    NUM_OUTER_STEPS = 500     # 500k updates in total (default 1M until 2026-11-14: returns plateau after ~500k)
    if 'num_outer_steps' in kwargs:   # shorter runs, for smoke tests only (recorded in the protocol line of info.txt)
        NUM_OUTER_STEPS = int(kwargs['num_outer_steps'])
    PLOT_EACH_EVALS = 5
    SAVE_EACH_STEPS = 100
    if 'eval_each_steps' in kwargs:
        EVAL_EACH_STEPS = int(kwargs['eval_each_steps'])

    eval_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/eval_log.csv",
        fieldnames=["step", "return_mean", "return_std", "time", "num_evals", "tag"],   # the deterministic actor
    )

    # ── Reward labelling: the datasets are reward-free, so we label them with the task's reward ──
    # r_t = R(aux_{t+1}) is the reward received for taking a_t in s_t (dm_control rewards depend on the state
    # reached after the action; this is also the convention FB uses with B(s_{t+1})).

    def label_rewards(aux, reward_fn, param, chunk=1_000_000):
        f   = jax.jit(lambda a: reward_fn(a, param))
        out = np.empty(len(aux), dtype=np.float32)
        for s in range(0, len(aux), chunk):
            out[s:s + chunk] = np.asarray(f(jnp.asarray(aux[s:s + chunk], dtype=jnp.float32)))
        return out

    dataset_rewards = label_rewards(dataset_aux, reward_fn, reward_param)
    q = np.quantile(dataset_rewards, [0, 0.05, 0.5, 0.95, 1])
    print(f'Labelled rewards: mean={dataset_rewards.mean():.4f} std={dataset_rewards.std():.4f} '
          f'quantiles(0,5,50,95,100%)={np.round(q, 3).tolist()}')


    # ── Replay buffer (single transitions) ────────────────────────────────────────

    class OfflineReplayBuffer(flax.struct.PyTreeNode):
        obs:       jnp.ndarray
        actions:   jnp.ndarray
        rewards:   jnp.ndarray            # rewards[i] = R(aux_i), the reward *of state i*
        aux:       Optional[jnp.ndarray]
        valid_idx: jnp.ndarray            # i such that (i, i+1) is inside a trajectory
        size:      int = flax.struct.field(pytree_node=False)

        @classmethod
        def init(cls, observations, actions, rewards, trajectory_len, aux=None) -> "OfflineReplayBuffer":
            n_total   = (len(observations) // trajectory_len) * trajectory_len
            idx       = np.arange(n_total)
            valid_idx = idx[idx % trajectory_len != trajectory_len - 1]
            return cls(
                obs=jnp.asarray(observations[:n_total], dtype=jnp.float32),
                actions=jnp.asarray(actions[:n_total], dtype=jnp.float32),
                rewards=jnp.asarray(rewards[:n_total], dtype=jnp.float32),
                aux=None if aux is None else jnp.asarray(aux[:n_total], dtype=jnp.float32),
                valid_idx=jnp.asarray(valid_idx, dtype=jnp.int32), size=len(valid_idx),
            )

        def sample(self, random_key, sample_size: int):
            """(s_t, a_t, r_t = R(aux_{t+1}), s_{t+1}, aux_t). No terminals: dm_control episodes are only truncated."""
            i = self.valid_idx[jax.random.randint(random_key, (sample_size,), 0, self.size)]
            aux = None if self.aux is None else self.aux[i]
            return self.obs[i], self.actions[i], self.rewards[i + 1][:, None], self.obs[i + 1], aux

    replay_buffer = OfflineReplayBuffer.init(dataset_obs, dataset_actions, dataset_rewards, TRAJECTORY_LEN, aux=dataset_aux)
    del dataset_obs, dataset_actions, dataset_aux, dataset_rewards

    # ── Networks (fb_jax.py style: torch Linear init, LayerNorm+Tanh preprocessing, no z) ──

    import flax.linen as nn

    class TorchDense(nn.Module):
        """nn.Dense with torch's default nn.Linear init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b."""
        features: int

        @nn.compact
        def __call__(self, x):
            bound = 1.0 / math.sqrt(x.shape[-1])
            init  = lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound)
            return nn.Dense(self.features, kernel_init=init, bias_init=init)(x)

    class NTanh(nn.Module):
        """LayerNorm followed by Tanh."""
        @nn.compact
        def __call__(self, x):
            return jnp.tanh(nn.LayerNorm(epsilon=1e-5)(x))

    class Actor(nn.Module):
        """mu(s) = tanh(MLP): Linear(hidden) -> LN+Tanh -> Linear(feature) -> ReLU -> Linear(hidden) -> ReLU -> Linear(A)."""
        hidden_dim:  int
        feature_dim: int
        action_dim:  int

        @nn.compact
        def __call__(self, obs):
            h = nn.relu(TorchDense(self.feature_dim)(NTanh()(TorchDense(self.hidden_dim)(obs))))
            h = nn.relu(TorchDense(self.hidden_dim)(h))
            return jnp.tanh(TorchDense(self.action_dim)(h))

    class Critic(nn.Module):
        """(Q1, Q2)(s, a), two independent heads on (s, a), same layout as the actor."""
        hidden_dim:  int
        feature_dim: int

        @nn.compact
        def __call__(self, obs, action):
            x = jnp.concatenate([obs, action], axis=-1)
            def head(x):
                h = nn.relu(TorchDense(self.feature_dim)(NTanh()(TorchDense(self.hidden_dim)(x))))
                h = nn.relu(TorchDense(self.hidden_dim)(h))
                return TorchDense(1)(h)
            return head(x), head(x)

    # ── TD3 agent ─────────────────────────────────────────────────────────────────

    @dataclasses.dataclass(frozen=True)
    class TD3Config:
        lr:           float = 1e-4      # as fb_jax.py
        batch_size:   int   = 1024      # as fb_jax.py
        hidden_dim:   int   = 1024
        feature_dim:  int   = 512
        discount:     float = 0.99      # as tdmpc_jax.py (TD3 / TD-MPC standard)
        tau:          float = 0.01      # EMA of the target actor and critic
        policy_noise: float = 0.2       # target policy smoothing: std of the noise on the target action
        noise_clip:   float = 0.3
        policy_freq:  int   = 2         # actor and targets updated every `policy_freq` critic updates

    class TD3TrainingState(flax.struct.PyTreeNode):
        actor_params:         Any
        critic_params:        Any
        target_actor_params:  Any
        target_critic_params: Any
        actor_opt_state:      optax.OptState
        critic_opt_state:     optax.OptState
        random_key:           jnp.ndarray
        steps:                jnp.ndarray

    class TD3Agent:
        def __init__(self, config: TD3Config, obs_dim: int, action_dim: int):
            self.cfg, self.obs_dim, self.action_dim = config, obs_dim, action_dim
            self.actor  = Actor(config.hidden_dim, config.feature_dim, action_dim)
            self.critic = Critic(config.hidden_dim, config.feature_dim)
            self.actor_opt  = optax.adam(config.lr)
            self.critic_opt = optax.adam(config.lr)

        def init(self, random_key) -> TD3TrainingState:
            random_key, ka, kc = jax.random.split(random_key, 3)
            obs, act = jnp.zeros((1, self.obs_dim)), jnp.zeros((1, self.action_dim))
            actor_params  = self.actor.init(ka, obs)['params']
            critic_params = self.critic.init(kc, obs, act)['params']
            return TD3TrainingState(
                actor_params=actor_params, critic_params=critic_params,
                target_actor_params=jax.tree_util.tree_map(jnp.copy, actor_params),
                target_critic_params=jax.tree_util.tree_map(jnp.copy, critic_params),
                actor_opt_state=self.actor_opt.init(actor_params), critic_opt_state=self.critic_opt.init(critic_params),
                random_key=random_key, steps=jnp.zeros((), dtype=jnp.int32))

        def act(self, actor_params, obs):
            return self.actor.apply({'params': actor_params}, obs)

        @partial(jax.jit, static_argnames=("self",))
        def update(self, ts: TD3TrainingState, replay_buffer: OfflineReplayBuffer):
            c = self.cfg
            key, k_batch, k_noise = jax.random.split(ts.random_key, 3)
            obs, action, reward, next_obs, _ = replay_buffer.sample(k_batch, c.batch_size)

            # critic: clipped double Q with target policy smoothing
            noise    = jnp.clip(jax.random.normal(k_noise, action.shape) * c.policy_noise, -c.noise_clip, c.noise_clip)
            next_act = jnp.clip(self.act(ts.target_actor_params, next_obs) + noise, -1.0, 1.0)
            tq1, tq2 = self.critic.apply({'params': ts.target_critic_params}, next_obs, next_act)
            target_q = jax.lax.stop_gradient(reward + c.discount * jnp.minimum(tq1, tq2))

            def critic_loss_fn(critic_params):
                q1, q2 = self.critic.apply({'params': critic_params}, obs, action)
                return jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2), q1.mean()

            (critic_loss, q_mean), c_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(ts.critic_params)
            c_updates, critic_opt_state = self.critic_opt.update(c_grads, ts.critic_opt_state, ts.critic_params)
            critic_params = optax.apply_updates(ts.critic_params, c_updates)

            # actor (delayed): maximise Q1(s, mu(s)) with the updated critic
            def actor_loss_fn(actor_params):
                q1, _ = self.critic.apply({'params': critic_params}, obs, self.act(actor_params, obs))
                return -q1.mean()

            actor_loss, a_grads = jax.value_and_grad(actor_loss_fn)(ts.actor_params)
            do_actor = (ts.steps % c.policy_freq == 0)
            a_updates, new_actor_opt_state = self.actor_opt.update(a_grads, ts.actor_opt_state, ts.actor_params)
            new_actor_params = optax.apply_updates(ts.actor_params, a_updates)
            pick = lambda new, old: jax.tree_util.tree_map(lambda n, o: jnp.where(do_actor, n, o), new, old)
            actor_params    = pick(new_actor_params, ts.actor_params)
            actor_opt_state = pick(new_actor_opt_state, ts.actor_opt_state)

            # targets, on the actor's schedule
            tau = c.tau * do_actor
            ema = lambda p, t: jax.tree_util.tree_map(lambda x, y: y + tau * (x - y), p, t)
            ts = ts.replace(actor_params=actor_params, critic_params=critic_params,
                            target_actor_params=ema(actor_params, ts.target_actor_params),
                            target_critic_params=ema(critic_params, ts.target_critic_params),
                            actor_opt_state=actor_opt_state, critic_opt_state=critic_opt_state,
                            random_key=key, steps=ts.steps + 1)
            metrics = {'critic_loss': critic_loss, 'actor_loss': actor_loss, 'q_mean': q_mean,
                       'target_q_mean': target_q.mean(), 'critic_grad_norm': optax.global_norm(c_grads),
                       'actor_grad_norm': optax.global_norm(a_grads)}
            return ts, metrics

        @partial(jax.jit, static_argnames=("self",))
        def act_batch(self, ts, obs):
            """The deterministic actor, no noise. obs (N, obs_dim)."""
            return self.act(ts.actor_params, obs)

    td3_config = TD3Config(batch_size=BATCH_SIZE)
    for name in ('lr', 'discount', 'tau', 'policy_noise', 'noise_clip'):
        if name in kwargs:
            td3_config = dataclasses.replace(td3_config, **{name: float(kwargs[name])})
    if 'policy_freq' in kwargs:
        td3_config = dataclasses.replace(td3_config, policy_freq=int(kwargs['policy_freq']))
    print(td3_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(td3_config) + '\n')

    agent = TD3Agent(td3_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)

    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key = jax.random.split(random_key)
    train_state = agent.init(init_key)
    print('parameters (actor + critic):',
          sum(x.size for x in jax.tree_util.tree_leaves((train_state.actor_params, train_state.critic_params))))

    eval_random_state = {'key': jax.random.PRNGKey(SEED + 10_000)}
    def next_eval_key():
        eval_random_state['key'], sub = jax.random.split(eval_random_state['key'])
        return sub

    @jax.jit
    def train_chunk(train_state, replay_buffer):
        """SCAN_LENGTH consecutive TD3 updates."""
        def body(ts, _):
            return agent.update(ts, replay_buffer)
        return jax.lax.scan(body, train_state, None, length=SCAN_LENGTH)

    # ── Evaluation (the deterministic actor) ──────────────────────────────────────

    eval_envs = []   # pool of env instances, grown on demand and reused across evals

    def rollout_batch(ts):
        """NUM_EVALS envs rolled out in lock-step, one batched jitted call per step."""
        while len(eval_envs) < NUM_EVALS:
            eval_envs.append(make_eval_env())
        envs = eval_envs[:NUM_EVALS]

        states = [env_reset(e) for e in envs]
        obs_l, aux_l, phys_l = [], [], []
        for _ in tqdm(range(EVAL_STEPS), leave=False):
            obs = np.stack(states).astype(np.float32)
            obs_l.append(obs)
            aux_l.append(np.stack([env_aux(e, s) for e, s in zip(envs, states)]))
            if env_physics(envs[0]) is not None:
                phys_l.append(np.stack([env_physics(e) for e in envs]))
            actions = agent.act_batch(ts, jnp.asarray(obs))
            states = [env_step(e, a) for e, a in zip(envs, np.asarray(actions))]

        obs_traj  = np.stack(obs_l).transpose(1, 0, 2)
        aux_traj  = np.stack(aux_l).transpose(1, 0, 2)
        phys_traj = np.stack(phys_l).transpose(1, 0, 2) if phys_l else None
        return obs_traj, aux_traj, phys_traj

    def plot_task(steps, obs_traj, aux_traj, phys_traj, tag=''):
        """2 panels: the true reward of dataset states, and the agent trajectories."""
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        eval_obs, _, eval_rew, _, eval_aux = replay_buffer.sample(next_eval_key(), 10_000)
        eval_obs, eval_aux, eval_rew = np.asarray(eval_obs), np.asarray(eval_aux), np.asarray(eval_rew)[:, 0]
        if ENV_NAME == 'cheetah':
            xs, ys = eval_obs[:, 8], eval_aux[:, 0]
        elif ENV_NAME == 'walker':
            xs, ys = eval_obs[:, 16], eval_aux[:, 0]
        elif ENV_NAME == 'quadruped':
            xs, ys = eval_aux[:, 0], eval_aux[:, 3]
        else:  # antmaze
            xs, ys = eval_obs[:, 0], eval_obs[:, 1]
        order = np.argsort(eval_rew)
        fig.colorbar(axs[0].scatter(xs[order], ys[order], c=eval_rew[order]), ax=axs[0])

        time_axis = np.tile(np.arange(EVAL_STEPS), (obs_traj.shape[0], 1))
        if ENV_NAME == 'cheetah':
            axs[1].scatter(phys_traj[..., 0], time_axis, c='red', s=1)
            axs[1].set_xlim([-25, 25])
        elif ENV_NAME == 'walker':
            axs[1].scatter(phys_traj[..., 1], time_axis, c='red', s=1)
        elif ENV_NAME == 'quadruped':
            for j in range(aux_traj.shape[0]):
                axs[1].plot(aux_traj[j, :, 0], aux_traj[j, :, 3], c='red')
        elif ENV_NAME == 'antmaze':
            axs[1].scatter(obs_traj[..., 0], obs_traj[..., 1], c='red', s=5)
            for k in (0, 1):
                add_largest_maze_walls(axs[k])
        axs[0].set_title(f'{TASK}: true reward (next state) of dataset transitions')
        axs[1].set_title('Agent Trajectory (TD3 actor)')
        plt.tight_layout()
        plt.savefig(f"{LOGS_FOLDER}/task{tag}-steps:{steps}.png")
        plt.close()

    def run_eval(train_state, steps, make_plot, tag='', num_evals=NUM_EVALS):
        all_rewards = []
        for _ in range(num_evals // NUM_EVALS):   # NUM_EVALS episodes per rollout
            obs_traj, aux_traj, phys_traj = rollout_batch(train_state)
            # same metric as the FB script: sum over the trajectory of the task reward of the visited states
            trajectory_rewards = np.asarray(reward_fn(jnp.asarray(aux_traj, dtype=jnp.float32), reward_param)).sum(axis=-1)
            if ENV_NAME == 'antmaze' and 'goal' in TASK:
                trajectory_rewards = np.where(trajectory_rewards != -EVAL_STEPS, 1.0, 0.0)
            all_rewards.append(trajectory_rewards)
        trajectory_rewards = np.concatenate(all_rewards)

        print(TASK, '[TD3 actor] :')
        print('\tRewards:', trajectory_rewards.tolist())
        print('\tmean:', trajectory_rewards.mean())
        print('\tstd:',  trajectory_rewards.std(ddof=1))
        if make_plot:
            plot_task(steps, obs_traj, aux_traj, phys_traj, tag)
        return trajectory_rewards.mean(), trajectory_rewards.std(ddof=1)

    def save_agent(train_state, path):
        state = {'actor_params': train_state.actor_params, 'critic_params': train_state.critic_params,
                 'target_actor_params': train_state.target_actor_params,
                 'target_critic_params': train_state.target_critic_params}
        state_dict = serialization.to_state_dict(jax.device_get(state))
        with open(path, "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # ── Training ──────────────────────────────────────────────────────────────────

    rewards_logs = []
    n_evals = 0

    def evaluate(train_state, num_updates, make_plot, num_evals, tag):
        """One eval_log row (tag 'eval' = periodic, 'final' = after the last update)."""
        start_time = time.time()
        m, s = run_eval(train_state, num_updates, make_plot, tag='-final' if tag == 'final' else '', num_evals=num_evals)
        eval_logger.log({'step': num_updates, 'return_mean': float(m), 'return_std': float(s),
                         'time': time.time() - start_time, 'num_evals': num_evals, 'tag': tag})
        return m

    add_text_to_file(f'{LOGS_FOLDER}/info.txt', run_info(globals()))      # seed, job, git, command, protocol, data, versions, start time

    for steps in tqdm(range(NUM_OUTER_STEPS)):
        num_updates = (steps + 1) * SCAN_LENGTH

        start_time = time.time()
        train_state, current_metrics = train_chunk(train_state, replay_buffer)
        current_metrics = jax.device_get(current_metrics)
        timelapse = time.time() - start_time

        row = {'step': num_updates, 'time': float(timelapse)}
        row.update({k: float(np.mean(current_metrics[k])) for k in TRAIN_METRICS})
        train_logger.log(row)

        if (steps + 1) % EVAL_EACH_STEPS == 0:
            make_plot = (n_evals % PLOT_EACH_EVALS == 0)
            rewards_logs.append(float(evaluate(train_state, num_updates, make_plot, NUM_EVALS, 'eval')))
            n_evals += 1

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.plot(rewards_logs)
            ax.set_title(TASK)
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/rewards.png")
            plt.close()

            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/td3_latest.pickle")

        if (steps + 1) % SAVE_EACH_STEPS == 0:
            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/td3_step:{num_updates}.pickle")

    # ── Final evaluation: NUM_EVALS_FINAL episodes on the last checkpoint (the reported number of the run) ──
    save_agent(train_state, f"{MODEL_SAVE_FOLDER}/td3_final.pickle")
    evaluate(train_state, NUM_OUTER_STEPS * SCAN_LENGTH, True, NUM_EVALS_FINAL, 'final')

    mark_finished(f'{LOGS_FOLDER}/info.txt', NUM_OUTER_STEPS * SCAN_LENGTH)
