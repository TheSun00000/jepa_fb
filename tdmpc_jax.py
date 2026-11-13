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
    assert 'task' in kwargs, "task= is required (one TD-MPC model is trained per benchmark task, e.g. task=vel10)"
    ENV_NAME = kwargs['env_name']
    TASK     = str(kwargs['task'])
    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze']

    # ── Config ────────────────────────────────────────────────────────────────────
    # cheetah | walker | quadruped (dm_control)  ;  antmaze (d4rl antmaze-large-diverse-v2)
    SEED = resolve_seed(kwargs)      # random unless seed= is given; recorded in info.txt

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = f'TDMPC-{ENV_NAME}-{TASK}'

    LOGS_FOLDER       = f'./logs/{date_time_str}_{exp_name}'
    MODEL_SAVE_FOLDER = f'./models/{date_time_str}_{exp_name}'
    os.makedirs(LOGS_FOLDER)
    os.makedirs(MODEL_SAVE_FOLDER)
    print('LOGS_FOLDER      :', LOGS_FOLDER)
    print('MODEL_SAVE_FOLDER:', MODEL_SAVE_FOLDER)

    save_self_copy(dst_dir=LOGS_FOLDER)

    TRAIN_METRICS = ["total_loss", "consistency_loss", "reward_loss", "value_loss", "grad_norm",
                     "pi_loss", "q_mean", "latent_norm", "latent_std"]
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

    NUM_EVALS = 5
    print(NUM_TRAJECTORIES, TRAJECTORY_LEN, NUM_TRAJECTORIES * TRAJECTORY_LEN)

    BATCH_SIZE      = int(kwargs['batch_size']) if 'batch_size' in kwargs else 512
    HORIZON         = int(kwargs['horizon']) if 'horizon' in kwargs else 5
    EVAL_EACH_STEPS = 10      # in outer iterations (1 outer iteration = SCAN_LENGTH updates)
    SCAN_LENGTH     = 1_000
    NUM_OUTER_STEPS = 1_000   # 1M updates in total
    PLOT_EACH_EVALS = 5
    SAVE_EACH_STEPS = 100
    if 'eval_each_steps' in kwargs:
        EVAL_EACH_STEPS = int(kwargs['eval_each_steps'])

    # How the agent acts at eval time:
    #   mppi   : MPPI planning in latent space (default)
    #   policy : the policy prior alone, deterministic (no planning, no noise)
    #   both   : evaluate both at every eval; `return_*` columns = MPPI, `policy_return_*` columns = policy prior
    EVAL_MODE = str(kwargs['eval_mode']) if 'eval_mode' in kwargs else 'mppi'
    assert EVAL_MODE in ['mppi', 'policy', 'both'], "eval_mode must be mppi | policy | both"
    EVAL_MODES = ['mppi', 'policy'] if EVAL_MODE == 'both' else [EVAL_MODE]
    MODE_NAMES = {'mppi': 'MPPI', 'policy': 'policy prior'}

    eval_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/eval_log.csv",
        fieldnames=["step", "return_mean", "return_std"]
                   + (["policy_return_mean", "policy_return_std"] if EVAL_MODE == 'both' else [])
                   + ["time"],
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

    # ── Replay buffer (sequences of length HORIZON) ───────────────────────────────

    class OfflineReplayBuffer(flax.struct.PyTreeNode):
        obs:       jnp.ndarray
        actions:   jnp.ndarray
        rewards:   jnp.ndarray            # rewards[i] = R(aux_i), the reward *of state i*
        aux:       Optional[jnp.ndarray]  # only used for plots
        valid_idx: jnp.ndarray            # i such that (i, i+1) is inside a trajectory
        seq_idx:   jnp.ndarray            # i such that (i, ..., i+horizon) is inside a trajectory
        size:      int = flax.struct.field(pytree_node=False)
        seq_size:  int = flax.struct.field(pytree_node=False)
        horizon:   int = flax.struct.field(pytree_node=False)

        @classmethod
        def init(cls, observations, actions, rewards, trajectory_len, horizon, aux=None) -> "OfflineReplayBuffer":
            n_total = (len(observations) // trajectory_len) * trajectory_len
            idx     = np.arange(n_total)
            pos     = idx % trajectory_len
            valid_idx = idx[pos != trajectory_len - 1]
            seq_idx   = idx[pos <= trajectory_len - 1 - horizon]

            return cls(
                obs=jnp.asarray(observations[:n_total], dtype=jnp.float32),
                actions=jnp.asarray(actions[:n_total], dtype=jnp.float32),
                rewards=jnp.asarray(rewards[:n_total], dtype=jnp.float32),
                aux=None if aux is None else jnp.asarray(aux[:n_total], dtype=jnp.float32),
                valid_idx=jnp.asarray(valid_idx, dtype=jnp.int32),
                seq_idx=jnp.asarray(seq_idx, dtype=jnp.int32),
                size=len(valid_idx), seq_size=len(seq_idx), horizon=horizon,
            )

        def sample(self, random_key, sample_size: int):
            """Single transitions. Returns (obs, actions, next_obs, aux)."""
            i   = self.valid_idx[jax.random.randint(random_key, (sample_size,), 0, self.size)]
            aux = None if self.aux is None else self.aux[i]
            return self.obs[i], self.actions[i], self.obs[i + 1], aux

        def sample_labeled(self, random_key, sample_size: int):
            """Single transitions with the labelled reward of the transition, R(aux_{t+1}) (what the reward head is
            trained to predict). Returns (obs, actions, next_obs, aux of the current state, reward)."""
            i   = self.valid_idx[jax.random.randint(random_key, (sample_size,), 0, self.size)]
            aux = None if self.aux is None else self.aux[i]
            return self.obs[i], self.actions[i], self.obs[i + 1], aux, self.rewards[i + 1]

        def sample_sequences(self, random_key, batch_size: int, horizon: Optional[int] = None):
            """Returns obs (H+1, B, obs_dim), actions (H, B, action_dim), rewards (H, B, 1)."""
            horizon = self.horizon if horizon is None else horizon
            assert horizon <= self.horizon
            i = self.seq_idx[jax.random.randint(random_key, (batch_size,), 0, self.seq_size)]
            t = jnp.arange(horizon)[:, None]
            obs     = self.obs[i[None, :] + jnp.arange(horizon + 1)[:, None]]
            actions = self.actions[i[None, :] + t]
            rewards = self.rewards[i[None, :] + t + 1][..., None]
            return obs, actions, rewards

    replay_buffer = OfflineReplayBuffer.init(dataset_obs, dataset_actions, dataset_rewards,
                                             TRAJECTORY_LEN, HORIZON, aux=dataset_aux)
    del dataset_obs, dataset_actions, dataset_aux, dataset_rewards

    # ── Neural networks (TD-MPC v1, state-based) ──────────────────────────────────

    import flax.linen as nn

    orth_init = nn.initializers.orthogonal()     # torch: nn.init.orthogonal_(weight), bias = 0

    class MLP(nn.Module):
        """TD-MPC `mlp`: Linear -> ELU -> Linear -> ELU -> Linear."""
        hidden:    Tuple[int, ...]
        out_dim:   int
        zero_last: bool = False   # reward head starts at zero

        @nn.compact
        def __call__(self, x):
            for h in self.hidden:
                x = nn.elu(nn.Dense(h, kernel_init=orth_init, bias_init=nn.initializers.zeros)(x))
            return nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros if self.zero_last else orth_init,
                            bias_init=nn.initializers.zeros)(x)

    class QNet(nn.Module):
        """TD-MPC `q`: Linear -> LayerNorm -> Tanh -> Linear -> ELU -> Linear(1); last layer starts at zero."""
        mlp_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.mlp_dim, kernel_init=orth_init, bias_init=nn.initializers.zeros)(x)
            x = jnp.tanh(nn.LayerNorm(epsilon=1e-5)(x))
            x = nn.elu(nn.Dense(self.mlp_dim, kernel_init=orth_init, bias_init=nn.initializers.zeros)(x))
            return nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)

    def l2_normalize(x, eps=1e-8):
        """x / ||x|| along the last axis."""
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)

    def sample_truncated_normal(mu, random_key, std, clip, low=-1.0, high=1.0, eps=1e-6):
        """torch TruncatedNormal.sample(clip): clipped Gaussian noise, output clamped with a straight-through gradient."""
        noise   = jax.random.normal(random_key, mu.shape) * std
        noise   = jnp.clip(noise, -clip, clip)
        x       = mu + noise
        clamped = jnp.clip(x, low + eps, high - eps)
        return x + jax.lax.stop_gradient(clamped - x)

    # ── TD-MPC Agent ──────────────────────────────────────────────────────────────

    @dataclasses.dataclass(frozen=True)
    class TDMPCConfig:
        # optimisation
        lr:                float = 1e-3
        grad_clip_norm:    float = 10.0
        tau:               float = 0.01     # EMA of the target encoder / target Q networks
        update_freq:       int   = 2        # the EMA is applied every `update_freq` updates
        batch_size:        int   = 512
        # model
        latent_dim:        int   = 50
        enc_dim:           int   = 256
        mlp_dim:           int   = 512
        # losses
        horizon:           int   = 5
        rho:               float = 0.5
        discount:          float = 0.99
        consistency_coef:  float = 2.0
        reward_coef:       float = 0.5
        value_coef:        float = 0.1
        # policy prior noise
        min_std:           float = 0.05
        std_clip:          float = 0.3
        # MPPI planning
        iterations:        int   = 6
        num_samples:       int   = 512
        num_elites:        int   = 64
        mixture_coef:      float = 0.05     # policy trajectories = int(mixture_coef * num_samples), added on top
        temperature:       float = 0.5
        momentum:          float = 0.1

    class TDMPCTrainingState(flax.struct.PyTreeNode):
        params:             Any          # {'encoder', 'dynamics', 'reward', 'q1', 'q2'}
        target_params:      Any          # EMA of {'encoder', 'q1', 'q2'}
        pi_params:          Any
        optimizer_state:    optax.OptState
        pi_optimizer_state: optax.OptState
        random_key:         jnp.ndarray
        steps:              jnp.ndarray

    class TDMPCAgent:

        def __init__(self, config: TDMPCConfig, obs_dim: int, action_dim: int):
            self.cfg        = config
            self.obs_dim    = obs_dim
            self.action_dim = action_dim
            c = config

            mlp = (c.mlp_dim, c.mlp_dim)
            self.encoder    = MLP(hidden=(c.enc_dim, c.enc_dim), out_dim=c.latent_dim)
            self.dynamics   = MLP(hidden=mlp, out_dim=c.latent_dim)
            self.reward_net = MLP(hidden=mlp, out_dim=1, zero_last=True)
            self.pi_net     = MLP(hidden=mlp, out_dim=action_dim)
            self.q_net      = QNet(c.mlp_dim)     # used twice, with independent parameters (q1, q2)

            self.optimizer    = optax.chain(optax.clip_by_global_norm(c.grad_clip_norm), optax.adam(c.lr))
            self.pi_optimizer = optax.chain(optax.clip_by_global_norm(c.grad_clip_norm), optax.adam(c.lr))

        # ── init ──────────────────────────────────────────────────────────────

        def init(self, random_key) -> TDMPCTrainingState:
            c = self.cfg
            random_key, ke, kd, kr, kq1, kq2, kp = jax.random.split(random_key, 7)
            obs = jnp.zeros((1, self.obs_dim))
            ha  = jnp.zeros((1, c.latent_dim + self.action_dim))

            params = {
                'encoder':  self.encoder.init(ke, obs)['params'],
                'dynamics': self.dynamics.init(kd, ha)['params'],
                'reward':   self.reward_net.init(kr, ha)['params'],
                'q1':       self.q_net.init(kq1, ha)['params'],
                'q2':       self.q_net.init(kq2, ha)['params'],
            }
            pi_params = self.pi_net.init(kp, jnp.zeros((1, c.latent_dim)))['params']
            target_params = {k: jax.tree_util.tree_map(jnp.copy, params[k]) for k in ('encoder', 'q1', 'q2')}

            return TDMPCTrainingState(
                params=params, target_params=target_params, pi_params=pi_params,
                optimizer_state=self.optimizer.init(params),
                pi_optimizer_state=self.pi_optimizer.init(pi_params),
                random_key=random_key, steps=jnp.zeros((), dtype=jnp.int32),
            )

        # ── model ─────────────────────────────────────────────────────────────

        def _bound(self, h):
            """Project onto the sphere of radius sqrt(latent_dim), so each coordinate has RMS ~1."""
            return math.sqrt(self.cfg.latent_dim) * l2_normalize(h)

        def encode(self, encoder_params, obs):
            """Encoder: s -> h (latent state)."""
            return self._bound(self.encoder.apply({'params': encoder_params}, obs))

        def next(self, params, h, a):
            """Residual dynamics h' = h + d(h, a), and reward R(h, a)."""
            x = jnp.concatenate([h, a], axis=-1)
            h_next = self._bound(h + self.dynamics.apply({'params': params['dynamics']}, x))
            return h_next, self.reward_net.apply({'params': params['reward']}, x)
            # return h_next, self.reward_net.apply({'params': params['reward']}, jax.lax.stop_gradient(x)) # WARNING

        def q(self, params, h, a):
            """(Q1(h, a), Q2(h, a)); `params` can be the online or the target parameters."""
            x = jnp.concatenate([h, a], axis=-1)
            return self.q_net.apply({'params': params['q1']}, x), self.q_net.apply({'params': params['q2']}, x)

        def pi(self, pi_params, h, key, std):
            """Deterministic policy prior + clipped Gaussian noise (std=0 -> no noise)."""
            mu = jnp.tanh(self.pi_net.apply({'params': pi_params}, h))
            return sample_truncated_normal(mu, key, std, self.cfg.std_clip) if std > 0 else mu

        @partial(jax.jit, static_argnames=("self",))
        def predict_reward(self, params, obs, action):
            return self.next(params, self.encode(params['encoder'], obs), action)[1]

        # ── linear reward reconstruction from the features phi(s) = h (as in the successor-feature script) ──

        @partial(jax.jit, static_argnames=("self",))
        def infer_z(self, encoder_params, obs, reward):
            """z = sqrt(d) * normalize(mean_s phi(s) R(s)), with phi the encoder. obs (N, obs_dim), reward (N,)."""
            phi = self.encode(encoder_params, obs)                                   # (N, L)
            z   = (reward[:, None] * phi).mean(axis=0)
            return math.sqrt(self.cfg.latent_dim) * l2_normalize(z)

        @partial(jax.jit, static_argnames=("self",))
        def linear_reward(self, encoder_params, obs, z):
            """phi(s)^T z / d: a cosine in [-1, 1]."""
            return jnp.sum(self.encode(encoder_params, obs) * z, axis=-1) / self.cfg.latent_dim

        # ── update ────────────────────────────────────────────────────────────

        @partial(jax.jit, static_argnames=("self",))
        def update(self, ts: TDMPCTrainingState, replay_buffer: OfflineReplayBuffer):
            c = self.cfg
            H = c.horizon
            key, k_batch, k_td, k_pi = jax.random.split(ts.random_key, 4)

            obs, action, reward = replay_buffer.sample_sequences(k_batch, c.batch_size, H)   # (H+1,B,D) (H,B,A) (H,B,1)

            # ── targets ───────────────────────────────────────────────────────
            target_h = self.encode(ts.target_params['encoder'], obs[1:])                                 # (H,B,L)
            # TD target: online encoder + noisy policy prior, target Q networks (min of the two)
            next_h    = self.encode(ts.params['encoder'], obs[1:])
            next_a    = self.pi(ts.pi_params, next_h, k_td, c.min_std)
            tq1, tq2  = self.q(ts.target_params, next_h, next_a)
            td_target = reward + c.discount * jnp.minimum(tq1, tq2)                                      # (H,B,1)
            rho_t     = c.rho ** jnp.arange(H, dtype=jnp.float32)

            # ── model loss: latent rollout with consistency / reward / value terms ──
            def loss_fn(params):
                h0 = self.encode(params['encoder'], obs[0])

                def rollout_step(h, xs):
                    a_t, target_h_t, td_t, r_t, rho = xs
                    q1, q2         = self.q(params, h, a_t)                    # Q at the current (predicted) latent
                    # q1, q2         = self.q(params, jax.lax.stop_gradient(h), a_t)                    # WARNING
                    h_next, r_pred = self.next(params, h, a_t)
                    # squared distance between the L2-normalised latents = 2 - 2 cos(h_next, target) in [0, 4]
                    # consistency = rho * jnp.mean((l2_normalize(h_next) - l2_normalize(target_h_t)) ** 2, axis=-1)   # (B,)
                    consistency = rho * jnp.mean((h_next - target_h_t) ** 2, axis=-1)   # (B,)
                    rew         = rho * ((r_pred - r_t) ** 2)[..., 0]
                    val         = rho * ((q1 - td_t) ** 2 + (q2 - td_t) ** 2)[..., 0]
                    return h_next, (h_next, consistency, rew, val, q1)

                _, (hs_next, cons, rew, val, q1s) = jax.lax.scan(
                    rollout_step, h0, (action, target_h, td_target, reward, rho_t))
                hs = jnp.concatenate([h0[None], hs_next], axis=0)                              # (H+1,B,L)

                cons, rew, val = cons.sum(0), rew.sum(0), val.sum(0)                           # sum over time -> (B,)
                total = (c.consistency_coef * jnp.minimum(cons, 1e4) + c.reward_coef * jnp.minimum(rew, 1e4)
                         + c.value_coef * jnp.minimum(val, 1e4))
                total_loss = total.mean()
                aux = {'hs': hs, 'total_loss': total_loss, 'consistency_loss': cons.mean(),
                       'reward_loss': rew.mean(), 'value_loss': val.mean(), 'q_mean': q1s[0].mean(),
                       # collapse / scale-drift monitoring on the encoded latents (not part of the loss)
                       'latent_norm': jnp.linalg.norm(h0, axis=-1).mean(), 'latent_std': jnp.std(h0, axis=0).mean()}
                return total_loss / H, aux       # TD-MPC scales the gradient by 1 / horizon

            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
            grad_norm = optax.global_norm(grads)
            updates, optimizer_state = self.optimizer.update(grads, ts.optimizer_state, ts.params)
            params = optax.apply_updates(ts.params, updates)

            # ── policy prior: maximise min(Q1, Q2) along the latent trajectory ──
            hs = aux['hs']     # a plain array here (aux is not differentiated), so it is a constant for the policy loss

            def pi_loss_fn(pi_params):
                a = self.pi(pi_params, hs, k_pi, c.min_std)                                     # (H+1,B,A)
                q1, q2 = self.q(params, hs, a)
                Q = jnp.minimum(q1, q2)
                rho = c.rho ** jnp.arange(H + 1, dtype=jnp.float32)
                return jnp.sum(-Q.mean(axis=(1, 2)) * rho)

            pi_loss, pi_grads = jax.value_and_grad(pi_loss_fn)(ts.pi_params)
            pi_updates, pi_optimizer_state = self.pi_optimizer.update(pi_grads, ts.pi_optimizer_state, ts.pi_params)
            pi_params = optax.apply_updates(ts.pi_params, pi_updates)

            # ── EMA of the target encoder and target Q networks (every `update_freq` updates) ──
            tau = c.tau * (ts.steps % c.update_freq == 0)
            target_params = jax.tree_util.tree_map(
                lambda p, t: t + tau * (p - t), {k: params[k] for k in ('encoder', 'q1', 'q2')}, ts.target_params)

            ts = ts.replace(params=params, target_params=target_params, pi_params=pi_params,
                            optimizer_state=optimizer_state, pi_optimizer_state=pi_optimizer_state,
                            random_key=key, steps=ts.steps + 1)
            metrics = {'total_loss': aux['total_loss'], 'consistency_loss': aux['consistency_loss'],
                       'reward_loss': aux['reward_loss'], 'value_loss': aux['value_loss'],
                       'grad_norm': grad_norm, 'pi_loss': pi_loss, 'q_mean': aux['q_mean'],
                       'latent_norm': aux['latent_norm'], 'latent_std': aux['latent_std']}
            return ts, metrics

        # ── MPPI planning in latent space ─────────────────────────────────────

        def _estimate_value(self, params, pi_params, h, actions, key):
            """Discounted model return over the horizon + bootstrapped min(Q1, Q2) at the end. h (S,L), actions (H,S,A)."""
            c = self.cfg
            G, discount = 0.0, 1.0
            for t in range(c.horizon):
                h, reward = self.next(params, h, actions[t])
                G = G + discount * reward
                discount = discount * c.discount
            a = self.pi(pi_params, h, key, c.min_std)
            q1, q2 = self.q(params, h, a)
            return (G + discount * jnp.minimum(q1, q2))[..., 0]

        def _plan_single(self, ts, obs, prev_mean, use_prev, key):
            """MPPI for one env. obs (obs_dim,), prev_mean (H, A). Returns (action (A,), new mean (H, A))."""
            c = self.cfg
            H, A = c.horizon, self.action_dim
            params, pi_params = ts.params, ts.pi_params
            num_pi = int(c.mixture_coef * c.num_samples)

            h0 = self.encode(params['encoder'], obs[None])[0]

            # trajectories sampled from the (noisy) policy prior; they are added on top of `num_samples`
            _h, pi_actions = jnp.tile(h0, (num_pi, 1)), []
            for t in range(H):
                key, sub = jax.random.split(key)
                a = self.pi(pi_params, _h, sub, c.min_std)
                pi_actions.append(a)
                _h, _ = self.next(params, _h, a)
            pi_actions = jnp.stack(pi_actions)                                      # (H, num_pi, A)

            h    = jnp.tile(h0, (c.num_samples + num_pi, 1))
            mean = jnp.where(use_prev, jnp.concatenate([prev_mean[1:], jnp.zeros((1, A))], axis=0), jnp.zeros((H, A)))
            std  = 2.0 * jnp.ones((H, A))

            for _ in range(c.iterations):
                key, k_noise, k_val = jax.random.split(key, 3)
                noise   = jax.random.normal(k_noise, (H, c.num_samples, A))
                sampled = jnp.clip(mean[:, None] + std[:, None] * noise, -1, 1)
                actions = jnp.concatenate([sampled, pi_actions], axis=1)            # (H, S + num_pi, A)

                value = jnp.nan_to_num(self._estimate_value(params, pi_params, h, actions, k_val))
                elite_value, elite_idx = jax.lax.top_k(value, c.num_elites)
                elite_actions = actions[:, elite_idx]                               # (H, E, A)

                logits = c.temperature * (elite_value - elite_value.max())
                score  = jax.nn.softmax(logits)
                _mean  = (score[None, :, None] * elite_actions).sum(axis=1)
                _std   = jnp.sqrt((score[None, :, None] * (elite_actions - _mean[:, None]) ** 2).sum(axis=1))
                _std   = jnp.clip(_std, c.min_std, 2.0)
                mean, std = c.momentum * mean + (1 - c.momentum) * _mean, _std

            key, sub = jax.random.split(key)
            chosen = jax.random.categorical(sub, logits)
            return elite_actions[0, chosen], mean                                   # eval mode: no exploration noise

        @partial(jax.jit, static_argnames=("self",))
        def plan_batch(self, ts, obs, prev_mean, use_prev, key):
            """MPPI for a batch of envs. obs (N, obs_dim), prev_mean (N, H, A). use_prev=False on the first step."""
            keys = jax.random.split(key, obs.shape[0])
            return jax.vmap(self._plan_single, in_axes=(None, 0, 0, None, 0))(ts, obs, prev_mean, use_prev, keys)

        @partial(jax.jit, static_argnames=("self",))
        def act_policy_batch(self, ts, obs):
            """Policy prior alone, no planning: a = pi(h(s)), deterministic (no exploration noise). obs (N, obs_dim)."""
            h = self.encode(ts.params['encoder'], obs)
            return self.pi(ts.pi_params, h, None, 0.0)

    tdmpc_config = TDMPCConfig(batch_size=BATCH_SIZE, horizon=HORIZON)
    if 'discount' in kwargs:
        tdmpc_config = dataclasses.replace(tdmpc_config, discount=float(kwargs['discount']))
    print(tdmpc_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(tdmpc_config) + '\n')

    agent = TDMPCAgent(tdmpc_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)

    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key = jax.random.split(random_key)
    train_state = agent.init(init_key)
    print('parameters (model + pi):',
          sum(x.size for x in jax.tree_util.tree_leaves((train_state.params, train_state.pi_params))))

    eval_random_state = {'key': jax.random.PRNGKey(SEED + 10_000)}
    def next_eval_key():
        eval_random_state['key'], sub = jax.random.split(eval_random_state['key'])
        return sub

    @jax.jit
    def train_chunk(train_state, replay_buffer):
        """SCAN_LENGTH consecutive TD-MPC updates."""
        def body(ts, _):
            return agent.update(ts, replay_buffer)
        return jax.lax.scan(body, train_state, None, length=SCAN_LENGTH)

    # ── Evaluation (MPPI planning and/or policy prior alone) ──────────────────────

    eval_envs = []   # pool of env instances, grown on demand and reused across evals

    def rollout_batch(ts, random_key, mode='mppi'):
        """NUM_EVALS envs rolled out in lock-step, one batched jitted call per step.
        mode='mppi': MPPI planning. mode='policy': the policy prior alone (deterministic, no planning)."""
        while len(eval_envs) < NUM_EVALS:
            eval_envs.append(make_eval_env())
        envs = eval_envs[:NUM_EVALS]

        states    = [env_reset(e) for e in envs]
        prev_mean = jnp.zeros((NUM_EVALS, tdmpc_config.horizon, ACTION_DIM))
        obs_l, aux_l, phys_l = [], [], []
        for step in tqdm(range(EVAL_STEPS), leave=False):
            obs = np.stack(states).astype(np.float32)
            obs_l.append(obs)
            aux_l.append(np.stack([env_aux(e, s) for e, s in zip(envs, states)]))
            if env_physics(envs[0]) is not None:
                phys_l.append(np.stack([env_physics(e) for e in envs]))

            if mode == 'policy':
                actions = agent.act_policy_batch(ts, jnp.asarray(obs))
            else:
                random_key, sub = jax.random.split(random_key)
                actions, prev_mean = agent.plan_batch(ts, jnp.asarray(obs), prev_mean, jnp.asarray(step > 0), sub)
            states = [env_step(e, a) for e, a in zip(envs, np.asarray(actions))]

        obs_traj  = np.stack(obs_l).transpose(1, 0, 2)
        aux_traj  = np.stack(aux_l).transpose(1, 0, 2)
        phys_traj = np.stack(phys_l).transpose(1, 0, 2) if phys_l else None
        return obs_traj, aux_traj, phys_traj

    def plot_task(steps, train_state, obs_traj, aux_traj, phys_traj, mode='mppi', tag=''):
        """5 panels: true reward of the dataset transitions, its reconstruction by the reward head, its linear
        reconstruction phi(s)^T z (as in SF), reward head prediction vs truth, and the agent trajectories."""
        fig, axs = plt.subplots(1, 5, figsize=(25, 4))
        encoder_params = train_state.params['encoder']

        # dataset transitions (s_t, a_t, s_{t+1}): true reward R(aux_{t+1}) vs the reward head's R(h(s_t), a_t)
        eval_obs, eval_act, eval_next_obs, eval_aux, true_rewards = replay_buffer.sample_labeled(next_eval_key(), 10_000)
        pred_rewards = np.asarray(agent.predict_reward(train_state.params, eval_obs, eval_act))[:, 0]
        true_rewards = np.asarray(true_rewards)

        # linear reconstruction, as in SF: z = sqrt(d) * normalize(mean_s phi(s) R(s)) is fitted on an independent batch of
        # dataset states, then phi(s_{t+1})^T z / d is evaluated on the transitions above (out of sample)
        fit_obs, _, _, fit_aux = replay_buffer.sample(next_eval_key(), 10_000)
        z = agent.infer_z(encoder_params, fit_obs, reward_fn(fit_aux, reward_param))
        lin_rewards = np.asarray(agent.linear_reward(encoder_params, eval_next_obs, z))
        lin_corr = np.corrcoef(true_rewards, lin_rewards)[0, 1] if true_rewards.std() > 0 and lin_rewards.std() > 0 else float('nan')

        eval_obs, eval_aux = np.asarray(eval_obs), np.asarray(eval_aux)
        time_axis = np.tile(np.arange(EVAL_STEPS), (obs_traj.shape[0], 1))

        # coordinates of the reward maps: the same as the trajectory panel of each env
        if ENV_NAME == 'cheetah':
            xs, ys = eval_obs[:, 8], eval_aux[:, 0]
        elif ENV_NAME == 'walker':
            xs, ys = eval_obs[:, 16], eval_aux[:, 0]
        elif ENV_NAME == 'quadruped':
            xs, ys = eval_aux[:, 0], eval_aux[:, 3]
        else:  # antmaze
            xs, ys = eval_obs[:, 0], eval_obs[:, 1]

        # true reward and reward-head reconstruction share the colour scale (the range of the TRUE reward: predictions
        # outside it saturate) and the drawing order (high rewards on top). The linear reconstruction is a cosine, so it
        # has its own colour scale; its corr with the true reward is scale free.
        vmin, vmax = float(true_rewards.min()), float(true_rewards.max())
        order = np.argsort(true_rewards)
        sc0 = axs[0].scatter(xs[order], ys[order], c=true_rewards[order], vmin=vmin, vmax=vmax)
        sc1 = axs[1].scatter(xs[order], ys[order], c=pred_rewards[order], vmin=vmin, vmax=vmax)
        sc2 = axs[2].scatter(xs[order], ys[order], c=lin_rewards[order])
        for sc, ax in zip((sc0, sc1, sc2), axs[:3]):
            fig.colorbar(sc, ax=ax)

        # reward head prediction vs truth
        axs[3].scatter(true_rewards, pred_rewards, s=2, alpha=0.3)
        lim = [min(vmin, pred_rewards.min()), max(vmax, pred_rewards.max())]
        axs[3].plot(lim, lim, c='black', linewidth=1)
        axs[3].set_xlabel('true reward'); axs[3].set_ylabel('predicted reward')
        corr = np.corrcoef(true_rewards, pred_rewards)[0, 1] if true_rewards.std() > 0 and pred_rewards.std() > 0 else float('nan')
        mse  = float(np.mean((true_rewards - pred_rewards) ** 2))

        # agent trajectories
        if ENV_NAME == 'cheetah':
            axs[4].scatter(phys_traj[..., 0], time_axis, c='red', s=1)
            axs[4].set_xlim([-25, 25])
        elif ENV_NAME == 'walker':
            axs[4].scatter(phys_traj[..., 1], time_axis, c='red', s=1)
        elif ENV_NAME == 'quadruped':
            for j in range(aux_traj.shape[0]):
                axs[4].plot(aux_traj[j, :, 0], aux_traj[j, :, 3], c='red')
        elif ENV_NAME == 'antmaze':
            axs[4].scatter(obs_traj[..., 0], obs_traj[..., 1], c='red', s=5)
            for k in (0, 1, 2, 4):
                add_largest_maze_walls(axs[k])

        axs[0].set_title(f'{TASK}: true reward')
        axs[1].set_title('Reward head reconstruction R(h(s), a)')
        axs[2].set_title(f"Linear reconstruction phi(s')^T z (corr={lin_corr:.3f})")
        axs[3].set_title(f'Reward head: corr={corr:.3f}, MSE={mse:.4f}')
        axs[4].set_title(f'Agent Trajectory ({MODE_NAMES[mode]})')
        plt.tight_layout()
        plt.savefig(f"{LOGS_FOLDER}/task{tag}-steps:{steps}.png")
        plt.close()

    def run_eval(train_state, steps, make_plot, mode='mppi', tag=''):
        obs_traj, aux_traj, phys_traj = rollout_batch(train_state, next_eval_key(), mode)

        # same metric as the FB script: sum over the trajectory of the task reward of the visited states
        trajectory_rewards = np.asarray(reward_fn(jnp.asarray(aux_traj, dtype=jnp.float32), reward_param)).sum(axis=-1)
        if ENV_NAME == 'antmaze' and 'goal' in TASK:
            # goal tasks: 1 if the goal was reached at least once, else 0
            trajectory_rewards = np.where(trajectory_rewards != -EVAL_STEPS, 1.0, 0.0)

        print(TASK, f'[{MODE_NAMES[mode]}]', ':')
        print('\tRewards:', trajectory_rewards.tolist())
        print('\tmean:', trajectory_rewards.mean())
        print('\tstd:',  trajectory_rewards.std(ddof=1))

        if make_plot:
            plot_task(steps, train_state, obs_traj, aux_traj, phys_traj, mode, tag)
        return trajectory_rewards.mean(), trajectory_rewards.std(ddof=1)

    def save_agent(train_state, path):
        state = {
            'params':        train_state.params,
            'target_params': train_state.target_params,
            'pi_params':     train_state.pi_params,
        }
        state_dict = serialization.to_state_dict(jax.device_get(state))
        with open(path, "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # ── Training ──────────────────────────────────────────────────────────────────

    rewards_logs = {mode: [] for mode in EVAL_MODES}
    n_evals = 0

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
            start_time = time.time()
            results = {}
            for i, mode in enumerate(EVAL_MODES):
                results[mode] = run_eval(train_state, num_updates, make_plot, mode, tag='' if i == 0 else f'-{mode}')
            n_evals += 1

            first = EVAL_MODES[0]
            eval_row = {'step': num_updates, 'return_mean': float(results[first][0]),
                        'return_std': float(results[first][1]), 'time': time.time() - start_time}
            if EVAL_MODE == 'both':
                eval_row['policy_return_mean'] = float(results['policy'][0])
                eval_row['policy_return_std']  = float(results['policy'][1])
            eval_logger.log(eval_row)

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            for mode in EVAL_MODES:
                rewards_logs[mode].append(float(results[mode][0]))
                ax.plot(rewards_logs[mode], label=MODE_NAMES[mode])
            if len(EVAL_MODES) > 1:
                ax.legend()
            ax.set_title(TASK)
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/rewards.png")
            plt.close()

            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/tdmpc_latest.pickle")

        if (steps + 1) % SAVE_EACH_STEPS == 0:
            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/tdmpc_step:{num_updates}.pickle")

    mark_finished(f'{LOGS_FOLDER}/info.txt', NUM_OUTER_STEPS * SCAN_LENGTH)
