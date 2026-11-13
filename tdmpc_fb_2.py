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
    ENV_NAME = kwargs['env_name']
    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze']

    # ── Config ────────────────────────────────────────────────────────────────────
    # cheetah | walker | quadruped (dm_control)  ;  antmaze (d4rl antmaze-large-diverse-v2)
    # Zero-shot: nothing is trained on a reward. Every benchmark task of the env is evaluated.
    SEED = resolve_seed(kwargs)      # random unless seed= is given; recorded in info.txt

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = f'TDMPC-FB-sepB-{ENV_NAME}'

    LOGS_FOLDER       = f'./logs/{date_time_str}_{exp_name}'
    MODEL_SAVE_FOLDER = f'./models/{date_time_str}_{exp_name}'
    os.makedirs(LOGS_FOLDER)
    os.makedirs(MODEL_SAVE_FOLDER)
    print('LOGS_FOLDER      :', LOGS_FOLDER)
    print('MODEL_SAVE_FOLDER:', MODEL_SAVE_FOLDER)

    save_self_copy(dst_dir=LOGS_FOLDER)

    TRAIN_METRICS = ["model_loss", "consistency_loss", "ortho_loss", "model_grad_norm", "f_enc_grad_norm",
                     "fb_loss", "fb_diag", "fb_offdiag", "q_loss", "b_ortho_loss", "f_grad_norm", "b_grad_norm", "M1", "F_norm",
                     "pi_loss", "q_mean", "bhead_loss",
                     "latent_std", "feat_cov_l2", "feat_mean_norm", "b_std", "b_cov_l2", "b_mean_norm"]
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

    # ── Zero-shot setup ───────────────────────────────────────────────────────────
    # Training never sees a reward. It learns:
    #   * the world model on phi(s) (bounded, radius sqrt(d)): residual dynamics h' = d(h, a), trained with consistency +
    #     orthonormality (+ the gradient from F through its input),
    #   * B(s): the FB backward map on the raw observation, trained with the FB loss + orthonormality (as in default FB),
    #   * F(phi(s), a, z) with the FB loss  M[i, j] = F(phi(s_i), a_i, z_i)^T B(s'_j),  and pi(a | phi(s), z) maximising F^T z,
    #   * a small B-head that predicts B(s) from the latent (used only by the planner for the model reward).
    # At evaluation every benchmark task of the env is solved zero-shot from a task vector in B-space
    #     z = sqrt(d) * normalize( mean_s B(s) R(s) ),   s ~ dataset,
    # followed by the policy alone or by MPPI planning with the model, the B-head, F and pi.

    N_BENCH   = len(benchmarks)
    NUM_EVALS = 5
    print(NUM_TRAJECTORIES, TRAJECTORY_LEN, NUM_TRAJECTORIES * TRAJECTORY_LEN)
    print('Zero-shot evaluation tasks:', [b[1] for b in benchmarks])

    BATCH_SIZE      = int(kwargs['batch_size']) if 'batch_size' in kwargs else 1024
    HORIZON         = int(kwargs['horizon']) if 'horizon' in kwargs else 5
    EVAL_EACH_STEPS = 10      # in outer iterations (1 outer iteration = SCAN_LENGTH updates)
    SCAN_LENGTH     = 1_000
    NUM_OUTER_STEPS = 1_000   # 1M updates in total
    PLOT_EACH_EVALS = 5
    SAVE_EACH_STEPS = 100
    if 'eval_each_steps' in kwargs:
        EVAL_EACH_STEPS = int(kwargs['eval_each_steps'])
    # Number of benchmark tasks rolled out in parallel during eval (each one needs NUM_EVALS env instances)
    EVAL_CHUNK = int(kwargs['eval_chunk']) if 'eval_chunk' in kwargs else N_BENCH

    # How the agent acts at eval time:
    #   mppi   : MPPI planning in latent space
    #   policy : the policy prior pi(a | h, z) alone, deterministic (no planning, no noise)
    #   both   : (default) evaluate both at every eval; unprefixed columns = MPPI, `policy_` columns = policy prior
    EVAL_MODE = str(kwargs['eval_mode']) if 'eval_mode' in kwargs else 'both'
    assert EVAL_MODE in ['mppi', 'policy', 'both'], "eval_mode must be mppi | policy | both"
    EVAL_MODES = ['mppi', 'policy'] if EVAL_MODE == 'both' else [EVAL_MODE]
    MODE_NAMES = {'mppi': 'MPPI', 'policy': 'policy prior'}

    def eval_columns(prefix):
        return [f"{prefix}mean_over_tasks"] + [f"{prefix}{label}_{s}" for _, label, _ in benchmarks for s in ("mean", "std")]

    eval_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/eval_log.csv",
        fieldnames=["step"] + eval_columns('') + [f"{label}_corr" for _, label, _ in benchmarks]
                   + (eval_columns('policy_') if EVAL_MODE == 'both' else []) + ["time"],
    )

    # ── Replay buffer (sequences of length HORIZON) ───────────────────────────────
    # No rewards are stored: `aux` is only used at evaluation (task inference and plots).

    class OfflineReplayBuffer(flax.struct.PyTreeNode):
        obs:       jnp.ndarray
        actions:   jnp.ndarray
        aux:       Optional[jnp.ndarray]
        valid_idx: jnp.ndarray            # i such that (i, i+1) is inside a trajectory
        seq_idx:   jnp.ndarray            # i such that (i, ..., i+horizon) is inside a trajectory
        size:      int = flax.struct.field(pytree_node=False)
        seq_size:  int = flax.struct.field(pytree_node=False)
        horizon:   int = flax.struct.field(pytree_node=False)

        @classmethod
        def init(cls, observations, actions, trajectory_len, horizon, aux=None) -> "OfflineReplayBuffer":
            n_total = (len(observations) // trajectory_len) * trajectory_len
            idx     = np.arange(n_total)
            pos     = idx % trajectory_len
            valid_idx = idx[pos != trajectory_len - 1]
            seq_idx   = idx[pos <= trajectory_len - 1 - horizon]

            return cls(
                obs=jnp.asarray(observations[:n_total], dtype=jnp.float32),
                actions=jnp.asarray(actions[:n_total], dtype=jnp.float32),
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

        def sample_sequences(self, random_key, batch_size: int, horizon: Optional[int] = None):
            """Returns obs (H+1, B, obs_dim) and actions (H, B, action_dim)."""
            horizon = self.horizon if horizon is None else horizon
            assert horizon <= self.horizon
            i = self.seq_idx[jax.random.randint(random_key, (batch_size,), 0, self.seq_size)]
            t = jnp.arange(horizon)[:, None]
            obs     = self.obs[i[None, :] + jnp.arange(horizon + 1)[:, None]]
            actions = self.actions[i[None, :] + t]
            return obs, actions

    replay_buffer = OfflineReplayBuffer.init(dataset_obs, dataset_actions, TRAJECTORY_LEN, HORIZON, aux=dataset_aux)
    del dataset_obs, dataset_actions, dataset_aux

    # ── Neural networks ───────────────────────────────────────────────────────────

    import flax.linen as nn

    orth_init = nn.initializers.orthogonal()     # torch: nn.init.orthogonal_(weight), bias = 0

    class MLP(nn.Module):
        """TD-MPC `mlp`: Linear -> ELU -> Linear -> ELU -> Linear."""
        hidden:    Tuple[int, ...]
        out_dim:   int
        zero_last: bool = False

        @nn.compact
        def __call__(self, x):
            for width in self.hidden:
                x = nn.elu(nn.Dense(width, kernel_init=orth_init, bias_init=nn.initializers.zeros)(x))
            return nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros if self.zero_last else orth_init,
                            bias_init=nn.initializers.zeros)(x)

    class TorchDense(nn.Module):
        """nn.Dense with torch's default nn.Linear init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b (as in the FB script)."""
        features: int

        @nn.compact
        def __call__(self, x):
            fan_in = x.shape[-1]
            bound  = 1.0 / math.sqrt(fan_in)
            init   = lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound)
            kernel = self.param('kernel', init, (fan_in, self.features))
            bias   = self.param('bias', init, (self.features,))
            return x @ kernel + bias

    class NTanh(nn.Module):
        """FB 'ntanh': LayerNorm followed by Tanh."""
        @nn.compact
        def __call__(self, x):
            return jnp.tanh(nn.LayerNorm(epsilon=1e-5)(x))

    class ForwardMap(nn.Module):
        """FB ForwardMap (preprocess=True), used as F: (h, z, a) -> (F1, F2). Two embedding branches, a shared
        merge, and two heads."""
        hidden_dim:  int
        feature_dim: int
        z_dim:       int
        add_trunk:   bool = False

        @nn.compact
        def __call__(self, obs, z, action):
            # obs_action_net: Linear -> ntanh -> Linear -> relu
            oa = TorchDense(self.hidden_dim)(jnp.concatenate([obs, action], axis=-1))
            oa = NTanh()(oa)
            oa = nn.relu(TorchDense(self.feature_dim)(oa))
            # obs_z_net: Linear -> ntanh -> Linear -> relu
            oz = TorchDense(self.hidden_dim)(jnp.concatenate([obs, z], axis=-1))
            oz = NTanh()(oz)
            oz = nn.relu(TorchDense(self.feature_dim)(oz))

            h = jnp.concatenate([oa, oz], axis=-1)
            if self.add_trunk:
                h = nn.relu(TorchDense(self.hidden_dim)(h))

            def head(h):
                x = nn.relu(TorchDense(self.hidden_dim)(h))
                return TorchDense(self.z_dim)(x)

            return head(h), head(h)

    class Actor(nn.Module):
        """FB Actor (preprocess=True): pi(a | h, z), returns the mean action mu = tanh(policy(.))."""
        hidden_dim:  int
        feature_dim: int
        action_dim:  int
        add_trunk:   bool = False

        @nn.compact
        def __call__(self, obs, z):
            o = NTanh()(TorchDense(self.hidden_dim)(obs))
            o = nn.relu(TorchDense(self.feature_dim)(o))
            oz = NTanh()(TorchDense(self.hidden_dim)(jnp.concatenate([obs, z], axis=-1)))
            oz = nn.relu(TorchDense(self.feature_dim)(oz))

            h = jnp.concatenate([o, oz], axis=-1)
            if self.add_trunk:
                h = nn.relu(TorchDense(self.hidden_dim)(h))

            h = nn.relu(TorchDense(self.hidden_dim)(h))
            return jnp.tanh(TorchDense(self.action_dim)(h))

    def l2_normalize(x, eps=1e-8):
        """x / ||x|| along the last axis."""
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)

    class BackwardMap(nn.Module):
        """FB BackwardMap on the raw observation: Linear -> ntanh -> Linear -> relu -> Linear, then sqrt(z_dim) * normalize."""
        hidden_dim: int
        z_dim:      int

        @nn.compact
        def __call__(self, obs):
            x = NTanh()(TorchDense(self.hidden_dim)(obs))
            x = nn.relu(TorchDense(self.hidden_dim)(x))
            x = TorchDense(self.z_dim)(x)
            return math.sqrt(self.z_dim) * l2_normalize(x)

    def sample_truncated_normal(mu, random_key, std, clip, low=-1.0, high=1.0, eps=1e-6):
        """torch TruncatedNormal.sample(clip): clipped Gaussian noise, output clamped with a straight-through gradient."""
        noise   = jax.random.normal(random_key, mu.shape) * std
        noise   = jnp.clip(noise, -clip, clip)
        x       = mu + noise
        clamped = jnp.clip(x, low + eps, high - eps)
        return x + jax.lax.stop_gradient(clamped - x)

    # ── helpers ───────────────────────────────────────────────────────────────────

    def gram_ortho(X):
        """FB orthonormality loss on a batch of embeddings X (n, d): mean_{i!=j} (x_i^T x_j)^2 - 2 mean_i ||x_i||^2."""
        n        = X.shape[0]
        gram     = X @ X.T
        off_diag = 1.0 - jnp.eye(n)
        return ((gram ** 2) * off_diag).sum() / off_diag.sum() - 2 * jnp.diag(gram).mean()

    def feature_stats(X):
        """Health of a batch of embeddings X (n, d): (mean per-dimension std, ||X^T X / n - I||_F / sqrt(d), ||mean X||).
        Collapse -> std -> 0 and ||mean|| -> sqrt(d); whitened -> the covariance term sits at its sampling floor."""
        n, d = X.shape
        return jnp.std(X, axis=0).mean(), jnp.linalg.norm(X.T @ X / n - jnp.eye(d)) / math.sqrt(d), jnp.linalg.norm(X.mean(axis=0))

    def fb_losses(F1, F2, B, z, target_M, next_Q, discount, q_loss_coef, use_q_loss=True):
        """Forward-backward losses on a batch of n transitions (s_i, a_i, s'_i), as in the FB script.
        F1, F2: (n, d) forward embeddings F(phi(s_i), a_i, z_i). B: (n, d) backward embeddings B(s'_j) of the next states.
        target_M: (n, n) target successor measure. next_Q: (n,) target Q(s'_i, pi(s'_i, z_i), z_i)."""
        n        = B.shape[0]
        M1, M2   = F1 @ B.T, F2 @ B.T                                            # M[i, j] = F(phi(s_i), a_i, z_i)^T B(s'_j)
        off_diag = 1.0 - jnp.eye(n)
        n_off    = off_diag.sum()
        fb_offdiag = 0.5 * sum((((M - discount * target_M) ** 2) * off_diag).sum() / n_off for M in (M1, M2))
        fb_diag    = -sum(jnp.diag(M).mean() for M in (M1, M2))

        q_loss = jnp.zeros(())
        if use_q_loss:   # FB's auxiliary loss: Q = F^T z regressed onto the implicit reward B^T Cov^-1 z + discount * next_Q
            B_sg     = jax.lax.stop_gradient(B)
            cov      = B_sg.T @ B_sg / n
            inv_cov  = jnp.linalg.inv(cov)
            implicit_reward = ((B_sg @ inv_cov) * z).sum(axis=-1)
            target_Q = jax.lax.stop_gradient(implicit_reward + discount * next_Q)
            Q1, Q2   = (F1 * z).sum(-1), (F2 * z).sum(-1)
            q_loss   = jnp.mean((Q1 - target_Q) ** 2) + jnp.mean((Q2 - target_Q) ** 2)
        fb_loss = fb_offdiag + fb_diag + q_loss_coef * q_loss
        return fb_loss, {'fb_offdiag': fb_offdiag, 'fb_diag': fb_diag, 'q_loss': q_loss, 'M1': M1.mean()}

    # ── TD-MPC + FB agent with separate phi(s) and B(s) ───────────────────────────
    # phi(s) : encoder / latent of the world model (consistency + orthonormality + gradient from F)
    # B(s)   : FB backward map on the raw observation (FB loss + orthonormality, as in default FB)
    # F(phi(s), a, z)^T B(s'') : the FB successor measure;  pi(a | phi(s), z) : the policy

    @dataclasses.dataclass(frozen=True)
    class TDMPCFBConfig:
        # optimisation (FB values: lr 1e-4, batch 1024, target EMA at every update)
        lr:                  float = 1e-4
        grad_clip_norm:      float = 10.0    # model (encoder + dynamics) only; B, F, pi are not clipped, as in FB
        tau:                 float = 0.01    # EMA of the target encoder phi, target B and target F
        update_freq:         int   = 1
        batch_size:          int   = 1024
        # world model: latent_dim = dim of phi(s) = dim of B(s) = dim of F = dim of the task vector z
        latent_dim:          int   = 50
        enc_dim:             int   = 256     # phi
        mlp_dim:             int   = 512     # dynamics and the B-head used by the planner
        # F and pi: the FB architectures, applied on the latent phi(s)
        hidden_dim:          int   = 1024
        feature_dim:         int   = 512
        # B: the FB backward map on the raw observation
        backward_hidden_dim: int   = 526
        # losses
        horizon:             int   = 5       # length of the latent rollout of the consistency loss
        rho:                 float = 0.5
        discount:            float = 0.95
        consistency_coef:    float = 2.0
        ortho_coef:          float = 0.0     # orthonormality loss on phi(s)
        b_ortho_coef:        float = 1.0     # orthonormality loss on B(s'), inside the FB loss (FB's value)
        f_encoder_grad:      float = 0.1     # scale of the FB gradient that reaches phi through F's input
        q_loss:              bool  = True    # FB's auxiliary Q-loss
        q_loss_coef:         float = 0.01
        mix_ratio:           float = 0.5     # fraction of task vectors taken from B(random dataset state), as in FB
        # policy noise in the FB target action and in the actor loss
        stddev:              float = 0.2
        std_clip:            float = 0.3
        # MPPI planning
        min_std:             float = 0.05
        iterations:          int   = 6
        num_samples:         int   = 512
        num_elites:          int   = 64
        mixture_coef:        float = 0.05    # policy trajectories = int(mixture_coef * num_samples), added on top
        temperature:         float = 0.5
        momentum:            float = 0.1

    class TDMPCFBTrainingState(flax.struct.PyTreeNode):
        params:                Any          # world model: {'encoder' (phi), 'dynamics'}
        b_params:              Any          # B, the FB backward map on the raw observation
        f_params:              Any          # F, the FB forward map on the latent
        bh_params:             Any          # B-head: predicts B(s) from the latent (used by the planner for the model reward)
        target_params:         Any          # EMA of {'encoder', 'b', 'f'}
        pi_params:             Any
        optimizer_state:       optax.OptState
        b_optimizer_state:     optax.OptState
        f_optimizer_state:     optax.OptState
        bh_optimizer_state:    optax.OptState
        pi_optimizer_state:    optax.OptState
        random_key:            jnp.ndarray
        steps:                 jnp.ndarray

    class TDMPCFBAgent:

        def __init__(self, config: TDMPCFBConfig, obs_dim: int, action_dim: int):
            self.cfg        = config
            self.obs_dim    = obs_dim
            self.action_dim = action_dim
            c = config

            self.encoder      = MLP(hidden=(c.enc_dim, c.enc_dim), out_dim=c.latent_dim)        # phi
            self.dynamics     = MLP(hidden=(c.mlp_dim, c.mlp_dim), out_dim=c.latent_dim)
            self.backward_net = BackwardMap(c.backward_hidden_dim, c.latent_dim)                # B (raw obs -> sphere)
            self.forward_net  = ForwardMap(c.hidden_dim, c.feature_dim, c.latent_dim)           # (phi, z, a) -> (F1, F2)
            self.pi_net       = Actor(c.hidden_dim, c.feature_dim, action_dim)                  # (phi, z) -> mu
            self.b_head_net   = MLP(hidden=(c.mlp_dim, c.mlp_dim), out_dim=c.latent_dim)        # latent -> B (planner reward)

            self.optimizer    = optax.chain(optax.clip_by_global_norm(c.grad_clip_norm), optax.adam(c.lr))
            self.b_optimizer  = optax.adam(c.lr)
            self.f_optimizer  = optax.adam(c.lr)
            self.bh_optimizer = optax.adam(c.lr)
            self.pi_optimizer = optax.adam(c.lr)

        # ── init ──────────────────────────────────────────────────────────────

        def init(self, random_key) -> TDMPCFBTrainingState:
            c = self.cfg
            random_key, ke, kd, kb, kf, kbh, kpi = jax.random.split(random_key, 7)
            L = c.latent_dim
            obs = jnp.zeros((1, self.obs_dim))
            h_d, a_d, z_d = jnp.zeros((1, L)), jnp.zeros((1, self.action_dim)), jnp.zeros((1, L))

            params = {
                'encoder':  self.encoder.init(ke, obs)['params'],
                'dynamics': self.dynamics.init(kd, jnp.zeros((1, L + self.action_dim)))['params'],
            }
            b_params  = self.backward_net.init(kb, obs)['params']
            f_params  = self.forward_net.init(kf, h_d, z_d, a_d)['params']
            bh_params = self.b_head_net.init(kbh, h_d)['params']
            pi_params = self.pi_net.init(kpi, h_d, z_d)['params']
            target_params = {
                'encoder': jax.tree_util.tree_map(jnp.copy, params['encoder']),
                'b':       jax.tree_util.tree_map(jnp.copy, b_params),
                'f':       jax.tree_util.tree_map(jnp.copy, f_params),
            }

            return TDMPCFBTrainingState(
                params=params, b_params=b_params, f_params=f_params, bh_params=bh_params,
                target_params=target_params, pi_params=pi_params,
                optimizer_state=self.optimizer.init(params),
                b_optimizer_state=self.b_optimizer.init(b_params),
                f_optimizer_state=self.f_optimizer.init(f_params),
                bh_optimizer_state=self.bh_optimizer.init(bh_params),
                pi_optimizer_state=self.pi_optimizer.init(pi_params),
                random_key=random_key, steps=jnp.zeros((), dtype=jnp.int32),
            )

        # ── model ─────────────────────────────────────────────────────────────

        def _bound(self, h):
            """Project onto the sphere of radius sqrt(latent_dim), so each coordinate has RMS ~1."""
            return math.sqrt(self.cfg.latent_dim) * l2_normalize(h)

        def encode(self, encoder_params, obs):
            """phi(s) = h, on the sphere."""
            return self._bound(self.encoder.apply({'params': encoder_params}, obs))

        def next(self, params, h, a):
            """Residual dynamics h' = normalize(h + d(h, a))."""
            x = jnp.concatenate([h, a], axis=-1)
            return self._bound(h + self.dynamics.apply({'params': params['dynamics']}, x))

        def backward(self, b_params, obs):
            """B(s) on the sphere of radius sqrt(latent_dim) (FB backward map on the raw observation)."""
            return self.backward_net.apply({'params': b_params}, obs)

        def b_head(self, bh_params, h):
            """B-hat(h): prediction of B(s) from the latent h = phi(s) (or from an imagined latent), on the sphere."""
            return self._bound(self.b_head_net.apply({'params': bh_params}, h))

        def F(self, f_params, h, a, z):
            """(F1(h, a, z), F2(h, a, z)); h = phi(s). `f_params` can be the online or the target parameters."""
            return self.forward_net.apply({'params': f_params}, h, z, a)

        def score(self, x, z):
            """x^T z / d. For x = B(s) this is a cosine in [-1, 1]; for x = F it is the Q-value (per-dimension scale)."""
            return jnp.sum(x * z, axis=-1, keepdims=True) / self.cfg.latent_dim

        def pi(self, pi_params, h, z, key, std):
            """Deterministic policy pi(a | phi(s), z) (the FB Actor already applies tanh) + clipped Gaussian noise (std=0 -> none)."""
            mu = self.pi_net.apply({'params': pi_params}, h, z)
            return sample_truncated_normal(mu, key, std, self.cfg.std_clip) if std > 0 else mu

        def sample_z(self, random_key, size: int):
            """Task vectors uniform on the sphere of radius sqrt(latent_dim)."""
            return math.sqrt(self.cfg.latent_dim) * l2_normalize(jax.random.normal(random_key, (size, self.cfg.latent_dim)))

        # ── zero-shot task inference (in B-space, as in FB) ───────────────────

        @partial(jax.jit, static_argnames=("self",))
        def infer_z(self, b_params, obs, reward):
            """z = sqrt(d) * normalize(mean_s B(s) R(s)). Also returns corr(B(s)^T z, R(s)) on the same samples."""
            B_  = self.backward(b_params, obs)                                       # (N, L)
            z   = (reward[:, None] * B_).mean(axis=0)
            z   = math.sqrt(self.cfg.latent_dim) * l2_normalize(z)
            pred = B_ @ z
            pc, rc = pred - pred.mean(), reward - reward.mean()
            corr = (pc * rc).mean() / (pc.std() * rc.std() + 1e-8)
            return z, corr

        @partial(jax.jit, static_argnames=("self",))
        def reconstruct_reward(self, b_params, obs, z):
            """Linear reward reconstruction B(s)^T z / d (used for the plots)."""
            return self.score(self.backward(b_params, obs), z)[..., 0]

        # ── update ────────────────────────────────────────────────────────────

        @partial(jax.jit, static_argnames=("self",))
        def update(self, ts: TDMPCFBTrainingState, replay_buffer: OfflineReplayBuffer):
            c = self.cfg
            H, B, L = c.horizon, c.batch_size, c.latent_dim
            key, k_batch, k_z, k_perm, k_mix, k_next, k_pi = jax.random.split(ts.random_key, 7)

            obs, action = replay_buffer.sample_sequences(k_batch, B, H)              # (H+1,B,D) (H,B,A)

            # ── task vectors (B-space): uniform on the sphere, a fraction replaced by B(random state) as in FB ──
            z = self.sample_z(k_z, B)
            if c.mix_ratio > 0:
                perm     = jax.random.permutation(k_perm, B)
                z_data   = self.backward(ts.b_params, obs[0][perm])                  # already on the sphere
                mix_mask = jax.random.uniform(k_mix, (B, 1)) < c.mix_ratio
                z        = jnp.where(mix_mask, z_data, z)

            # ── constants: computed outside the differentiated functions below, so no gradient flows into them ──
            target_h = self.encode(ts.target_params['encoder'], obs[1:])             # (H,B,L) EMA phi: consistency target
            target_B = self.backward(ts.target_params['b'], obs[1])                  # (B,L)   EMA B(s'_j)
            h_all    = self.encode(ts.params['encoder'], obs)                        # (H+1,B,L) online phi of every state
            B_all    = self.backward(ts.b_params, obs)                               # (H+1,B,L) online B of every state (B-head targets)
            h_in, next_h = h_all[0], h_all[1]                                        # phi(s), phi(s'): policy inputs
            # FB target: target F at the target encoder's latent, with the next action of the (noisy) policy
            next_a   = self.pi(ts.pi_params, next_h, z, k_next, c.stddev)
            tF1, tF2 = self.F(ts.target_params['f'], target_h[0], next_a, z)         # (B,L)
            target_M = jnp.minimum(tF1 @ target_B.T, tF2 @ target_B.T)               # (B,B)
            next_Q   = jnp.minimum((tF1 * z).sum(-1), (tF2 * z).sum(-1))
            rho_t    = c.rho ** jnp.arange(H, dtype=jnp.float32)

            # ── phi + dynamics: consistency + orthonormality ──
            def model_loss_fn(params):
                h0 = self.encode(params['encoder'], obs[0])

                def rollout_step(h, xs):
                    a_t, target_h_t, rho = xs
                    h_next = self.next(params, h, a_t)
                    # mean squared error on the sphere of radius sqrt(d) == 2 - 2 cos(h_next, target) in [0, 4]
                    return h_next, (h_next, rho * jnp.mean((h_next - target_h_t) ** 2, axis=-1))    # (B,)

                _, (hs_next, cons) = jax.lax.scan(rollout_step, h0, (action, target_h, rho_t))
                hs = jnp.concatenate([h0[None], hs_next], axis=0)                                # (H+1,B,L) imagined latents
                consistency_loss = cons.sum(0).mean()
                ortho_loss = gram_ortho(h0)
                model_loss = c.consistency_coef * consistency_loss / H + c.ortho_coef * ortho_loss
                std, cov_l2, mean_norm = feature_stats(h0)
                aux = {'hs': hs, 'consistency_loss': consistency_loss, 'ortho_loss': ortho_loss,
                       'latent_std': std, 'feat_cov_l2': cov_l2, 'feat_mean_norm': mean_norm}
                return model_loss, aux

            # ── FB loss on the first transition of each sequence: M[i, j] = F(phi(s_i), a_i, z_i)^T B(s'_j) ──
            #  * F   : full gradient
            #  * B   : full gradient of the FB loss + its orthonormality loss (as in default FB)
            #  * phi : gradient through F's input only (phi(s_i) is F's input), scaled by `f_encoder_grad` below
            def fb_loss_fn(enc_params, b_params, f_params):
                h0     = self.encode(enc_params, obs[0])
                F1, F2 = self.F(f_params, h0, action[0], z)                                         # (B,L)
                B_s    = self.backward(b_params, obs[1])                                            # (B,L): B(s'_j)
                fb_loss, fb_aux = fb_losses(F1, F2, B_s, z, target_M, next_Q, c.discount, c.q_loss_coef, c.q_loss)
                b_ortho = gram_ortho(B_s)
                b_std, b_cov_l2, b_mean_norm = feature_stats(B_s)
                aux = dict(fb_aux, fb_loss=fb_loss, b_ortho_loss=b_ortho, F_norm=jnp.linalg.norm(F1, axis=-1).mean(),
                           b_std=b_std, b_cov_l2=b_cov_l2, b_mean_norm=b_mean_norm)
                return fb_loss + c.b_ortho_coef * b_ortho, aux

            (model_loss, m_aux), g_model = jax.value_and_grad(model_loss_fn, has_aux=True)(ts.params)
            (_, fb_aux), (g_fb_enc, g_b, g_f) = jax.value_and_grad(fb_loss_fn, argnums=(0, 1, 2), has_aux=True)(
                ts.params['encoder'], ts.b_params, ts.f_params)

            g_fb_enc = jax.tree_util.tree_map(lambda g: c.f_encoder_grad * g, g_fb_enc)
            grads = {'encoder':  jax.tree_util.tree_map(jnp.add, g_model['encoder'], g_fb_enc),
                     'dynamics': g_model['dynamics']}
            model_grad_norm = optax.global_norm(g_model)         # consistency + orthonormality
            f_enc_grad_norm = optax.global_norm(g_fb_enc)        # the gradient from F into phi (after scaling)
            f_grad_norm     = optax.global_norm(g_f)
            b_grad_norm     = optax.global_norm(g_b)

            updates, optimizer_state = self.optimizer.update(grads, ts.optimizer_state, ts.params)
            params = optax.apply_updates(ts.params, updates)
            b_updates, b_optimizer_state = self.b_optimizer.update(g_b, ts.b_optimizer_state, ts.b_params)
            b_params = optax.apply_updates(ts.b_params, b_updates)
            f_updates, f_optimizer_state = self.f_optimizer.update(g_f, ts.f_optimizer_state, ts.f_params)
            f_params = optax.apply_updates(ts.f_params, f_updates)

            # ── B-head (for the planner): predict B(s) from phi(s) and from the imagined latents; constants as inputs ──
            hs = m_aux['hs']

            def bhead_loss_fn(bh_params):
                p_real = self.b_head(bh_params, h_all)                                              # (H+1,B,L) vs B(s_t)
                p_imag = self.b_head(bh_params, hs[1:])                                             # (H,B,L)   imagined vs B(s_t)
                return jnp.mean((p_real - B_all) ** 2) + jnp.mean((p_imag - B_all[1:]) ** 2)

            bhead_loss, g_bh = jax.value_and_grad(bhead_loss_fn)(ts.bh_params)
            bh_updates, bh_optimizer_state = self.bh_optimizer.update(g_bh, ts.bh_optimizer_state, ts.bh_params)
            bh_params = optax.apply_updates(ts.bh_params, bh_updates)

            # ── policy: maximise min_i F_i(phi(s), a, z)^T z on the real latents (phi(s) is a constant here) ──
            def pi_loss_fn(pi_params):
                a = self.pi(pi_params, h_in, z, k_pi, c.stddev)
                F1, F2 = self.F(f_params, h_in, a, z)
                Q = jnp.minimum((F1 * z).sum(-1), (F2 * z).sum(-1))
                return -Q.mean(), Q.mean()

            (pi_loss, q_mean), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(ts.pi_params)
            pi_updates, pi_optimizer_state = self.pi_optimizer.update(pi_grads, ts.pi_optimizer_state, ts.pi_params)
            pi_params = optax.apply_updates(ts.pi_params, pi_updates)

            # ── EMA of the target encoder, target B and target F (every `update_freq` updates) ──
            tau = c.tau * (ts.steps % c.update_freq == 0)
            online = {'encoder': params['encoder'], 'b': b_params, 'f': f_params}
            target_params = jax.tree_util.tree_map(lambda p, t: t + tau * (p - t), online, ts.target_params)

            ts = ts.replace(params=params, b_params=b_params, f_params=f_params, bh_params=bh_params,
                            target_params=target_params, pi_params=pi_params,
                            optimizer_state=optimizer_state, b_optimizer_state=b_optimizer_state,
                            f_optimizer_state=f_optimizer_state, bh_optimizer_state=bh_optimizer_state,
                            pi_optimizer_state=pi_optimizer_state, random_key=key, steps=ts.steps + 1)
            metrics = {
                'model_loss': model_loss, 'consistency_loss': m_aux['consistency_loss'], 'ortho_loss': m_aux['ortho_loss'],
                'model_grad_norm': model_grad_norm, 'f_enc_grad_norm': f_enc_grad_norm,
                'fb_loss': fb_aux['fb_loss'], 'fb_diag': fb_aux['fb_diag'], 'fb_offdiag': fb_aux['fb_offdiag'],
                'q_loss': fb_aux['q_loss'], 'b_ortho_loss': fb_aux['b_ortho_loss'],
                'f_grad_norm': f_grad_norm, 'b_grad_norm': b_grad_norm, 'M1': fb_aux['M1'], 'F_norm': fb_aux['F_norm'],
                'pi_loss': pi_loss, 'q_mean': q_mean, 'bhead_loss': bhead_loss,
                'latent_std': m_aux['latent_std'], 'feat_cov_l2': m_aux['feat_cov_l2'], 'feat_mean_norm': m_aux['feat_mean_norm'],
                'b_std': fb_aux['b_std'], 'b_cov_l2': fb_aux['b_cov_l2'], 'b_mean_norm': fb_aux['b_mean_norm'],
            }
            return ts, metrics

        # ── MPPI planning in the latent space of phi, for a given task vector z ──────────

        def _estimate_value(self, params, f_params, bh_params, pi_params, h, z, actions, key):
            """Sum_t gamma^t B-hat(h_{t+1})^T z / d over the horizon (B-hat predicts B from the imagined latent), plus the
            bootstrapped min_i F_i(h_H, pi(h_H, z), z)^T z / d at the end. h (S,L), z (S,L), actions (H,S,A)."""
            c = self.cfg
            G, discount = 0.0, 1.0
            for t in range(c.horizon):
                h = self.next(params, h, actions[t])
                G = G + discount * self.score(self.b_head(bh_params, h), z)
                discount = discount * c.discount
            a = self.pi(pi_params, h, z, key, c.min_std)
            F1, F2 = self.F(f_params, h, a, z)
            q = jnp.minimum(self.score(F1, z), self.score(F2, z))
            return (G + discount * q)[..., 0]

        def _plan_single(self, ts, obs, z_task, prev_mean, use_prev, key):
            """MPPI for one env and one task vector. obs (obs_dim,), z_task (L,), prev_mean (H, A)."""
            c = self.cfg
            H, A = c.horizon, self.action_dim
            params, f_params, bh_params, pi_params = ts.params, ts.f_params, ts.bh_params, ts.pi_params
            num_pi = int(c.mixture_coef * c.num_samples)

            h0 = self.encode(params['encoder'], obs[None])[0]

            # trajectories sampled from the (noisy) policy; they are added on top of `num_samples`
            _h, _zt, pi_actions = jnp.tile(h0, (num_pi, 1)), jnp.tile(z_task, (num_pi, 1)), []
            for t in range(H):
                key, sub = jax.random.split(key)
                a = self.pi(pi_params, _h, _zt, sub, c.min_std)
                pi_actions.append(a)
                _h = self.next(params, _h, a)
            pi_actions = jnp.stack(pi_actions)                                      # (H, num_pi, A)

            h    = jnp.tile(h0, (c.num_samples + num_pi, 1))
            z    = jnp.tile(z_task, (c.num_samples + num_pi, 1))
            mean = jnp.where(use_prev, jnp.concatenate([prev_mean[1:], jnp.zeros((1, A))], axis=0), jnp.zeros((H, A)))
            std  = 2.0 * jnp.ones((H, A))

            for _ in range(c.iterations):
                key, k_noise, k_val = jax.random.split(key, 3)
                noise   = jax.random.normal(k_noise, (H, c.num_samples, A))
                sampled = jnp.clip(mean[:, None] + std[:, None] * noise, -1, 1)
                actions = jnp.concatenate([sampled, pi_actions], axis=1)            # (H, S + num_pi, A)

                value = jnp.nan_to_num(self._estimate_value(params, f_params, bh_params, pi_params, h, z, actions, k_val))
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
        def plan_batch(self, ts, obs, zs, prev_mean, use_prev, key):
            """MPPI for a batch of envs, env i conditioned on task vector zs[i]. obs (N, obs_dim), zs (N, L)."""
            keys = jax.random.split(key, obs.shape[0])
            return jax.vmap(self._plan_single, in_axes=(None, 0, 0, 0, None, 0))(ts, obs, zs, prev_mean, use_prev, keys)

        @partial(jax.jit, static_argnames=("self",))
        def act_policy_batch(self, ts, obs, zs):
            """Policy alone, no planning: a = pi(phi(s), z), deterministic (no exploration noise)."""
            h = self.encode(ts.params['encoder'], obs)
            return self.pi(ts.pi_params, h, zs, None, 0.0)

    tdmpc_config = TDMPCFBConfig(batch_size=BATCH_SIZE, horizon=HORIZON)
    for name in ('lr', 'discount', 'consistency_coef', 'ortho_coef', 'b_ortho_coef', 'f_encoder_grad', 'q_loss_coef',
                 'mix_ratio', 'stddev'):
        if name in kwargs:
            tdmpc_config = dataclasses.replace(tdmpc_config, **{name: float(kwargs[name])})
    print(tdmpc_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(tdmpc_config) + '\n')

    agent = TDMPCFBAgent(tdmpc_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)

    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key = jax.random.split(random_key)
    train_state = agent.init(init_key)
    print('parameters (phi + dynamics + B + F + B-head + pi):',
          sum(x.size for x in jax.tree_util.tree_leaves((train_state.params, train_state.b_params, train_state.f_params, train_state.bh_params, train_state.pi_params))))

    eval_random_state = {'key': jax.random.PRNGKey(SEED + 10_000)}
    def next_eval_key():
        eval_random_state['key'], sub = jax.random.split(eval_random_state['key'])
        return sub

    @jax.jit
    def train_chunk(train_state, replay_buffer):
        """SCAN_LENGTH consecutive updates."""
        def body(ts, _):
            return agent.update(ts, replay_buffer)
        return jax.lax.scan(body, train_state, None, length=SCAN_LENGTH)

    # ── Zero-shot evaluation on every benchmark task ──────────────────────────────

    eval_envs = []   # pool of env instances, grown on demand and reused across evals

    def rollout_batch(ts, zs, random_key, mode='mppi'):
        """len(zs) envs rolled out in lock-step, env i conditioned on task vector zs[i]. One batched jitted call per step.
        mode='mppi': MPPI planning. mode='policy': the policy prior alone (deterministic, no planning)."""
        n = zs.shape[0]
        while len(eval_envs) < n:
            eval_envs.append(make_eval_env())
        envs = eval_envs[:n]

        states    = [env_reset(e) for e in envs]
        prev_mean = jnp.zeros((n, tdmpc_config.horizon, ACTION_DIM))
        obs_l, aux_l, phys_l = [], [], []
        for step in tqdm(range(EVAL_STEPS), leave=False):
            obs = np.stack(states).astype(np.float32)
            obs_l.append(obs)
            aux_l.append(np.stack([env_aux(e, s) for e, s in zip(envs, states)]))
            if env_physics(envs[0]) is not None:
                phys_l.append(np.stack([env_physics(e) for e in envs]))

            if mode == 'policy':
                actions = agent.act_policy_batch(ts, jnp.asarray(obs), zs)
            else:
                random_key, sub = jax.random.split(random_key)
                actions, prev_mean = agent.plan_batch(ts, jnp.asarray(obs), zs, prev_mean, jnp.asarray(step > 0), sub)
            states = [env_step(e, a) for e, a in zip(envs, np.asarray(actions))]

        obs_traj  = np.stack(obs_l).transpose(1, 0, 2)
        aux_traj  = np.stack(aux_l).transpose(1, 0, 2)
        phys_traj = np.stack(phys_l).transpose(1, 0, 2) if phys_l else None
        return obs_traj, aux_traj, phys_traj

    def plot_benchmark(axs, b, z, obs_traj, aux_traj, phys_traj, b_params, mode):
        """3 panels: real reward on the dataset, its linear reconstruction B(s)^T z / d, and the agent trajectories."""
        reward_fn, label, param = benchmarks[b]
        eval_obs, _, _, eval_aux = replay_buffer.sample(next_eval_key(), 10_000)
        real_rewards  = np.asarray(reward_fn(eval_aux, param))
        recon_rewards = np.asarray(agent.reconstruct_reward(b_params, eval_obs, z))
        eval_obs, eval_aux = np.asarray(eval_obs), np.asarray(eval_aux)
        time_axis = np.tile(np.arange(EVAL_STEPS), (obs_traj.shape[0], 1))

        if ENV_NAME == 'cheetah':
            axs[b, 0].scatter(eval_obs[..., 8], eval_aux[..., 0], c=real_rewards)
            axs[b, 1].scatter(eval_obs[..., 8], eval_aux[..., 0], c=recon_rewards)
            axs[b, 2].scatter(phys_traj[..., 0], time_axis, c='red', s=1)
            axs[b, 2].set_xlim([-25, 25])
        elif ENV_NAME == 'walker':
            axs[b, 0].scatter(eval_obs[..., 16], eval_aux[..., 0], c=real_rewards)
            axs[b, 1].scatter(eval_obs[..., 16], eval_aux[..., 0], c=recon_rewards)
            axs[b, 2].scatter(phys_traj[..., 1], time_axis, c='red', s=1)
        elif ENV_NAME == 'quadruped':
            axs[b, 0].scatter(eval_aux[..., 0], eval_aux[..., 3], c=real_rewards)
            axs[b, 1].scatter(eval_aux[..., 0], eval_aux[..., 3], c=recon_rewards)
            for j in range(aux_traj.shape[0]):
                axs[b, 2].plot(aux_traj[j, :, 0], aux_traj[j, :, 3], c='red')
        elif ENV_NAME == 'antmaze':
            axs[b, 0].scatter(eval_obs[..., 0], eval_obs[..., 1], c=real_rewards)
            axs[b, 1].scatter(eval_obs[..., 0], eval_obs[..., 1], c=recon_rewards)
            axs[b, 2].scatter(obs_traj[..., 0], obs_traj[..., 1], c='red', s=5)
            for k in range(3):
                add_largest_maze_walls(axs[b, k])

        axs[b, 0].set_title(f'{label}')
        axs[b, 1].set_title('Linear reconstruction B(s)^T z')
        axs[b, 2].set_title(f'Agent Trajectory ({MODE_NAMES[mode]})')

    def run_benchmark(train_state, steps, make_plot, mode='mppi', tag=''):
        """Zero-shot evaluation on all tasks. For each task and each of the NUM_EVALS evals, a fresh task vector is
        inferred from 10k dataset states. Returns per-task (mean return, std, corr(B^T z, R))."""
        b_params = train_state.b_params

        if make_plot:
            fig, axs = plt.subplots(N_BENCH, 3, figsize=(15, N_BENCH * 4))

        means, stds, corrs = np.zeros(N_BENCH), np.zeros(N_BENCH), np.zeros(N_BENCH)
        for start in range(0, N_BENCH, EVAL_CHUNK):
            ids = list(range(start, min(start + EVAL_CHUNK, N_BENCH)))

            zs, cs = [], []
            for b in ids:
                reward_fn, _, param = benchmarks[b]
                for _ in range(NUM_EVALS):
                    eval_obs, _, _, eval_aux = replay_buffer.sample(next_eval_key(), 10_000)
                    z, corr = agent.infer_z(b_params, eval_obs, reward_fn(eval_aux, param))
                    zs.append(z); cs.append(corr)
            zs, cs = jnp.stack(zs), np.asarray(jnp.stack(cs))

            obs_traj, aux_traj, phys_traj = rollout_batch(train_state, zs, next_eval_key(), mode)

            for j, b in enumerate(ids):
                reward_fn, label, param = benchmarks[b]
                sl = slice(j * NUM_EVALS, (j + 1) * NUM_EVALS)

                # same metric as the FB script: sum over the trajectory of the task reward of the visited states
                trajectory_rewards = np.asarray(reward_fn(jnp.asarray(aux_traj[sl], dtype=jnp.float32), param)).sum(axis=-1)
                if ENV_NAME == 'antmaze' and 'goal' in label:
                    # goal tasks: 1 if the goal was reached at least once, else 0
                    trajectory_rewards = np.where(trajectory_rewards != -EVAL_STEPS, 1.0, 0.0)

                print(label, f'[{MODE_NAMES[mode]}]', ':')
                print('\tRewards:', trajectory_rewards.tolist())
                print('\tmean:', trajectory_rewards.mean())
                print('\tstd:',  trajectory_rewards.std(ddof=1))
                print('\tcorr(B^T z, R):', float(cs[sl].mean()))
                means[b], stds[b], corrs[b] = trajectory_rewards.mean(), trajectory_rewards.std(ddof=1), cs[sl].mean()

                if make_plot:
                    plot_benchmark(axs, b, zs[(j + 1) * NUM_EVALS - 1], obs_traj[sl], aux_traj[sl],
                                   None if phys_traj is None else phys_traj[sl], b_params, mode)

        if make_plot:
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/benchmark{tag}-steps:{steps}.png")
            plt.close()

        return means, stds, corrs

    def save_agent(train_state, path):
        state = {
            'params':        train_state.params,
            'b_params':      train_state.b_params,
            'f_params':      train_state.f_params,
            'bh_params':     train_state.bh_params,
            'target_params': train_state.target_params,
            'pi_params':     train_state.pi_params,
        }
        state_dict = serialization.to_state_dict(jax.device_get(state))
        with open(path, "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # ── Training ──────────────────────────────────────────────────────────────────

    rewards_logs = {mode: np.zeros((N_BENCH, 0)) for mode in EVAL_MODES}
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
                results[mode] = run_benchmark(train_state, num_updates, make_plot, mode, tag='' if i == 0 else f'-{mode}')
            n_evals += 1

            eval_row = {'step': num_updates, 'time': time.time() - start_time}
            def fill(prefix, means, stds):
                eval_row[f'{prefix}mean_over_tasks'] = float(means.mean())
                for i, (_, label, _) in enumerate(benchmarks):
                    eval_row[f'{prefix}{label}_mean'] = float(means[i])
                    eval_row[f'{prefix}{label}_std']  = float(stds[i])
            first = EVAL_MODES[0]
            fill('', results[first][0], results[first][1])
            for i, (_, label, _) in enumerate(benchmarks):
                eval_row[f'{label}_corr'] = float(results[first][2][i])
            if EVAL_MODE == 'both':
                fill('policy_', results['policy'][0], results['policy'][1])
            eval_logger.log(eval_row)

            num_cols = 4
            num_rows = N_BENCH // num_cols + int(N_BENCH % num_cols != 0)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
            axs = np.atleast_1d(axs).flatten()
            for mode in EVAL_MODES:
                rewards_logs[mode] = np.concatenate((rewards_logs[mode], results[mode][0].reshape(-1, 1)), axis=-1)
                for i, (_, label, _) in enumerate(benchmarks):
                    axs[i].plot(rewards_logs[mode][i], label=MODE_NAMES[mode])
                    axs[i].set_title(label)
            if len(EVAL_MODES) > 1:
                axs[0].legend()
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/rewards.png")
            plt.close()

            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/tdmpc_fb_sepB_latest.pickle")

        if (steps + 1) % SAVE_EACH_STEPS == 0:
            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/tdmpc_fb_sepB_step:{num_updates}.pickle")

    mark_finished(f'{LOGS_FOLDER}/info.txt', NUM_OUTER_STEPS * SCAN_LENGTH)
