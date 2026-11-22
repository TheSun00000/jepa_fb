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
    OGBENCH_ENVS = ['antmaze_medium', 'antmaze_large', 'cube_single']
    # Behaviour-cloning regulariser of the policy (OGBench antmaze / cube, as in offline_qd_jax.py): the policy loss is normalised by its own
    # magnitude and bc_coef * MSE(policy action, dataset action) is added. bc_coef=0 (default): unchanged.
    BC_COEF = float(kwargs['bc_coef']) if 'bc_coef' in kwargs else 0.0

    def bc_combine(main, bc, coef):
        if coef == 0:
            return main
        return main / (jax.lax.stop_gradient(jnp.abs(main)) + 1e-8) + coef * bc

    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze'] + OGBENCH_ENVS

    # ── Config ────────────────────────────────────────────────────────────────────
    # cheetah | walker | quadruped (dm_control)  ;  antmaze (d4rl antmaze-large-diverse-v2)
    # antmaze_medium | antmaze_large | cube_single (OGBench, 2026-11-21): dataset=explore|stitch|navigate (antmaze, default explore) or
    #   play|noisy (cube, default play); ogbench_dir= folder of the OGBench .npz files (default ~/.ogbench/data)
    # Zero-shot: nothing is trained on a reward. Every benchmark task of the env is evaluated.
    SEED = resolve_seed(kwargs)      # random unless seed= is given; recorded in info.txt
    # phi_identity=True: phi(s) = s. No encoder: F(s, a, z), B(s) = g(s) (the FB backward map on the raw observation) and
    # pi(a | s, z) act on the raw observation, trained with the FB loss -> plain FB. The world model stays ON: the dynamics
    # (h' = h + d(h, a), no sphere) learn to predict the next raw state with the consistency loss, and MPPI can be evaluated.
    PHI_IDENTITY = str(kwargs['phi_identity']).lower() in ('1', 'true', 'yes') if 'phi_identity' in kwargs else False
    # variant (required, no default):
    #   sepB : B(s) is FB's BackwardMap on the RAW observation (FB loss + orthonormality). phi (MLP encoder, enc_dim) feeds the dynamics,
    #          F and pi. A B-head regresses B(s) from real and imagined latents (its own parameters only) and gives MPPI its per-step
    #          reward B_hat(h)^T z. Same as the former tdmpc_fb_2.py.
    #   gphi : one trunk phi(s) (PhiMap, phi_hidden_dim) and B(s) = g(phi(s)), g = FB's BackwardMap on the latent. MPPI reward g(h)^T z.
    #          Options phi_identity, bcons_coef, b_encoder_grad. Same as the former tdmpc_fb_3.py.
    assert kwargs.get('variant') in ('sepB', 'gphi'), "variant=sepB or variant=gphi is required"
    VARIANT = kwargs['variant']
    assert not (PHI_IDENTITY and VARIANT == 'sepB'), "phi_identity is a gphi option"
    # wm_ckpt=<wm_final.pickle of tdmpc_adapt.py mode=pretrain> freeze_wm=true (gphi only; 2026-11-18): phi (the TD-MPC MLP encoder, ELU, on the
    # sphere) and the dynamics are loaded from a reward-free JEPA world model and never updated (no consistency, no FB gradient into them).
    # Only g (B = g(phi)), F and pi train, with the FB loss (+ bcons on g). Sizes must match the checkpoint (latent_dim 50, enc_dim 256, mlp_dim 512).
    WM_CKPT   = kwargs.get('wm_ckpt')
    FREEZE_WM = str(kwargs['freeze_wm']).lower() in ('1', 'true', 'yes') if 'freeze_wm' in kwargs else False
    assert FREEZE_WM == (WM_CKPT is not None), "wm_ckpt= and freeze_wm=true go together"
    assert not (FREEZE_WM and (PHI_IDENTITY or VARIANT != 'gphi')), "freeze_wm is a gphi option (no phi_identity)"
    # 2026-11-22: the frozen JEPA WM can have the TD-MPC MLP encoder (tdmpc_adapt.py encoder=mlp) or the PhiMap trunk (encoder=phimap);
    # the type and hidden width are read from the checkpoint.
    FROZEN_WM_PHIMAP, FROZEN_WM_WIDTH = False, None
    if FREEZE_WM:
        with open(WM_CKPT, 'rb') as f:
            _wm_enc = pickle.load(f)['encoder']
        FROZEN_WM_PHIMAP = 'TorchDense_0' in _wm_enc
        FROZEN_WM_WIDTH  = int(_wm_enc['TorchDense_0' if FROZEN_WM_PHIMAP else 'Dense_0']['kernel'].shape[1])
        del _wm_enc
    # Final evaluation after the last update, with more episodes than the periodic ones (reported number of the run)
    NUM_EVALS_FINAL = int(kwargs['num_evals_final']) if 'num_evals_final' in kwargs else 10

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = f'FBWM-{VARIANT}{"-identity" if PHI_IDENTITY else ""}{"-frozenwm" if FREEZE_WM else ""}-{ENV_NAME}'

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

    TRAIN_METRICS = ["model_loss", "consistency_loss", "bcons_loss", "ortho_loss",
                     "model_enc_grad_norm", "f_enc_grad_norm", "b_enc_grad_norm",
                     "fb_loss", "fb_diag", "fb_offdiag", "q_loss", "b_ortho_loss", "f_grad_norm", "b_grad_norm", "M1", "F_norm",
                     "pi_loss", "q_mean", "bhead_loss",
                     "latent_std", "feat_cov_l2", "feat_mean_norm", "b_std", "b_cov_l2", "b_mean_norm", "fcons_loss"]
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

    elif ENV_NAME == 'antmaze':
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

    else:  # OGBench: antmaze_medium | antmaze_large | cube_single
        import ogbench
        OGB_DIR = os.path.expanduser(kwargs.get('ogbench_dir', '~/.ogbench/data'))
        if ENV_NAME == 'cube_single':
            OGB_DATASET = kwargs.get('dataset', 'play')
            assert OGB_DATASET in ('play', 'noisy'), 'dataset=play|noisy for cube_single'
            ogb_name = f'cube-single-{OGB_DATASET}-v0'
            EVAL_STEPS = 200      # official episode length
        else:
            OGB_DATASET = kwargs.get('dataset', 'explore')
            assert OGB_DATASET in ('explore', 'stitch', 'navigate'), 'dataset=explore|stitch|navigate for antmaze'
            ogb_name = f"antmaze-{ENV_NAME.split('_')[1]}-{OGB_DATASET}-v0"
            EVAL_STEPS = 1000     # official episode length
        print('OGBench dataset:', ogb_name, 'from', OGB_DIR)
        _env0, dataset = ogbench.make_env_and_datasets(ogb_name, dataset_dir=OGB_DIR, compact_dataset=True)[:2]
        STATE_DIM, ACTION_DIM = _env0.observation_space.shape[0], _env0.action_space.shape[0]
        # OGBench trajectories are stored back to back, 'terminals' = 1 on the last row of each one
        # (compact dataset: 'terminals' is 1 on the last TWO rows of a trajectory, 'valids' is 0 on the last one only)
        NUM_TRAJECTORIES = int((np.asarray(dataset['valids']) == 0).sum())
        TRAJECTORY_LEN = dataset['terminals'].shape[0] // NUM_TRAJECTORIES
        assert NUM_TRAJECTORIES * TRAJECTORY_LEN == dataset['terminals'].shape[0], 'unexpected OGBench trajectory layout'
        N = NUM_TRAJECTORIES * TRAJECTORY_LEN
        dataset_obs     = np.asarray(dataset['observations'], dtype=np.float32)[:N].reshape(-1, STATE_DIM)
        dataset_actions = np.asarray(dataset['actions'], dtype=np.float32)[:N].reshape(-1, ACTION_DIM)
        dataset_aux     = dataset_obs       # rewards are computed from the full observation
        del dataset

        if ENV_NAME == 'cube_single':
            # obs[19:22] = block position, obs[12:15] = effector position, both rescaled by OGBench: x -> (x - 0.425) * 10,
            # y -> y * 10, z -> z * 10. Goals: the 5 official tasks (block goal position), success = block within 0.04 m.
            def _cube_scale(p): return np.array([(p[0] - 0.425) * 10, p[1] * 10, p[2] * 10], dtype=np.float32)
            def cube_goal_reward(state, goal):
                dist = jnp.linalg.norm(state[..., 19:22] - jnp.asarray(goal, dtype=jnp.float32), axis=-1)
                return (dist < 0.4).astype(jnp.float32) * 2 - 1
            def cube_block_dir(state, direction):   # dense: block displacement along an axis of the table
                return state[..., 19] * direction[0] + state[..., 20] * direction[1]
            benchmarks = [(cube_goal_reward, f'goal_{i + 1}', _cube_scale(t['goal_xyzs'][0])) for i, t in enumerate(_env0.unwrapped.task_infos)]
            benchmarks += [(cube_block_dir, 'block_x+', [1, 0]), (cube_block_dir, 'block_x-', [-1, 0]),
                           (cube_block_dir, 'block_y+', [0, 1]), (cube_block_dir, 'block_y-', [0, -1])]
        else:
            # obs[:2] = ant (x, y), obs[15:17] = (vx, vy): same layout as the d4rl antmaze. Goals: the 5 official tasks.
            benchmarks = [(goal_reaching_reward, f'goal_{i + 1}', np.array(t['goal_xy'])) for i, t in enumerate(_env0.unwrapped.task_infos)]
            benchmarks += [(velocity_reward, 'vel_left', [-1, 0]), (velocity_reward, 'vel_up', [0, 1]),
                           (velocity_reward, 'vel_down', [0, -1]), (velocity_reward, 'vel_right', [1, 0])]
        if kwargs.get('tasks') == 'goal':   # tasks=goal: evaluate the goal-reaching tasks only
            benchmarks = [b for b in benchmarks if b[1].startswith('goal_')]
        del _env0

        def make_eval_env():
            return ogbench.make_env_and_datasets(ogb_name, env_only=True)

        _goalk_init = None
        if 'goal_k' in kwargs:   # goal_k=K: one fixed (start state, goal K steps ahead) pair instead of the official tasks (utils/ogbench_tasks.py)
            from utils.ogbench_tasks import goalk_task
            assert ENV_NAME != 'cube_single', 'goal_k is implemented for antmaze only'
            _goalk_init, _goal_xy, _info = goalk_task(dataset_obs, NUM_TRAJECTORIES, TRAJECTORY_LEN, kwargs)
            benchmarks = [(goal_reaching_reward, f"goalk_{_info['k']}", _goal_xy)]
            print('goal_k task (fixed): start xy', np.round(_goalk_init[:2], 2), '| goal xy', np.round(_goal_xy, 2), '|', _info)

        def env_reset(e, label=None):
            from utils.ogbench_tasks import reset_env
            return reset_env(e, label, _goalk_init)

        def env_step(e, action):
            return np.asarray(e.step(action)[0])   # no autoreset / truncation handling: EVAL_STEPS = episode length

        def env_aux(e, state):
            return state

        def env_physics(e):
            return None

    def reset_for(e, label=None):
        """Env reset for the task `label` (OGBench: goal_k starts at the official start of task k); other envs: plain reset."""
        return env_reset(e, label) if ENV_NAME in ('antmaze_medium', 'antmaze_large', 'cube_single') else env_reset(e)

    # ── Zero-shot setup ───────────────────────────────────────────────────────────
    # Training never sees a reward. Both variants learn, on the latent phi(s) (bounded, radius sqrt(latent_dim)):
    #   * the world model: residual dynamics h' = d(h, a), trained with consistency (+ orthonormality),
    #   * F(phi(s), a, z) with the FB loss  M[i, j] = F(phi(s_i), a_i, z_i)^T B(s'_j),  and pi(a | phi(s), z) maximising F^T z.
    # They differ in B:
    #   sepB : B(s) is the FB backward map on the raw observation; a B-head B_hat(h) predicts B from (real and imagined) latents.
    #          phi gets gradient from the consistency loss and from F's input.
    #   gphi : B(s) = g(phi(s)), the backward map as a head on the trunk (defined on imagined latents too).
    #          phi gets gradient from the consistency losses, F's input, and B (through g).
    # At evaluation every benchmark task is solved zero-shot from a task vector in B-space
    #     z = sqrt(d) * normalize( mean_s B(s) R(s) ),   s ~ dataset,
    # followed by the policy alone or by MPPI planning, whose model reward is B_hat(h)^T z (sepB) or g(h)^T z (gphi) on imagined
    # latents h. Periodic evals use NUM_EVALS episodes per task; the final eval after the last update uses NUM_EVALS_FINAL (tag 'final').

    N_BENCH   = len(benchmarks)
    NUM_EVALS = 5
    # MPPI per-step reward (2026-11-15, approved by an independent reviewer agent): mean = g(h)^T z (as before);
    # lstsq = g(h)^T w with w = c (E[B B^T] + lam I)^-1 E[B R], the L2(rho) projection of R onto span(B), and c the factor that
    # puts z = c E[B R] on the sphere. F^T z is exactly the Q of that projected reward, so the per-step rewards and the
    # terminal value describe the same reward. The policy and the terminal value keep z = E[B R] (exact in FB).
    MPPI_REWARD = str(kwargs['mppi_reward']) if 'mppi_reward' in kwargs else 'mean'
    assert MPPI_REWARD in ('mean', 'lstsq'), 'mppi_reward=mean|lstsq'
    Z_RIDGE = float(kwargs['z_ridge']) if 'z_ridge' in kwargs else 1e-2   # relative to the mean eigenvalue of E[B B^T]
    # E[B B^T] of the lstsq weights: labeled = on the N labelled states (as before); dataset = on Z_COV_STATES unlabelled dataset
    # states (it involves no reward, so labels are not needed; only E[B R] uses the N labels)
    Z_COV = str(kwargs['z_cov']) if 'z_cov' in kwargs else 'labeled'
    assert Z_COV in ('labeled', 'dataset'), 'z_cov=labeled|dataset'
    Z_COV_STATES = 100_000
    Z_LABELS = int(kwargs['z_labels']) if 'z_labels' in kwargs else 10_000   # labelled states per inferred z (one fresh draw per episode)
    EVAL_CKPT = kwargs.get('eval_ckpt')   # eval only: load this checkpoint, run one eval (tag 'zlabels'), no training
    print(NUM_TRAJECTORIES, TRAJECTORY_LEN, NUM_TRAJECTORIES * TRAJECTORY_LEN)
    print('Zero-shot evaluation tasks:', [b[1] for b in benchmarks])

    BATCH_SIZE      = int(kwargs['batch_size']) if 'batch_size' in kwargs else 1024
    HORIZON         = int(kwargs['horizon']) if 'horizon' in kwargs else 5
    EVAL_EACH_STEPS = 10      # in outer iterations (1 outer iteration = SCAN_LENGTH updates)
    SCAN_LENGTH     = 1_000
    NUM_OUTER_STEPS = 500     # 500k updates in total (default 1M until 2026-11-14: returns plateau after ~500k)
    if 'num_outer_steps' in kwargs:   # shorter runs, for smoke tests only (recorded in the protocol line of info.txt)
        NUM_OUTER_STEPS = int(kwargs['num_outer_steps'])
    PLOT_EACH_EVALS = 5
    SAVE_EACH_STEPS = 100
    if 'eval_each_steps' in kwargs:
        EVAL_EACH_STEPS = int(kwargs['eval_each_steps'])
    # MPPI only on every MPPI_EVAL_EACH-th outer step (the other periodic evals are policy-only, MPPI columns empty).
    # Default: every periodic eval (unchanged). 2026-11-15 protocol: eval_each_steps=25 mppi_eval_each=100.
    MPPI_EVAL_EACH = int(kwargs['mppi_eval_each']) if 'mppi_eval_each' in kwargs else EVAL_EACH_STEPS
    DIAG_F = str(kwargs.get('diag_f', '0')).lower() in ('1', 'true')   # eval_ckpt diagnostic: F / B on imagined vs real latents
    # eval_ckpt diagnostic (2026-11-16): long open-loop world-model rollouts vs the real simulator (diag_wm.csv), then exit
    # without the usual eval. diag_wm_h = rollout length; diag_wm_envs = real episodes of the policy per task.
    DIAG_WM = str(kwargs.get('diag_wm', '0')).lower() in ('1', 'true')
    DIAG_WM_H = int(kwargs['diag_wm_h']) if 'diag_wm_h' in kwargs else 100
    DIAG_WM_ENVS = int(kwargs['diag_wm_envs']) if 'diag_wm_envs' in kwargs else 10
    # diag_wm_cf=1 (with diag_wm=1): counterfactual action ranking, see (4) in the DIAG_WM block (dm_control envs only)
    DIAG_WM_CF = str(kwargs.get('diag_wm_cf', '0')).lower() in ('1', 'true')
    # diag_profile=1 (eval_ckpt): time profile of real episodes (policy and MPPI): true reward, aux and, on cheetah, the torso pitch,
    # per 200-step segment (diag_profile.csv). Then exit.
    DIAG_PROFILE = str(kwargs.get('diag_profile', '0')).lower() in ('1', 'true')
    # fbgoal_k=K (eval_ckpt, OGBench, 2026-11-22): goal reaching on the near-goal pairs of wm_goal.py goal_k=K (the same pairs for the
    # same goal_n / goal_split / goal_min_dist / pair_seed / dataset: utils/ogbench_tasks.sample_goal_pairs). Pair i = dataset state s_t,
    # goal s_{t+K} of the same trajectory; the simulator starts at the stored (qpos, qvel) of s_t. Task vector of the goal STATE (FB's
    # goal reaching, the reward concentrated at s_{t+K}): z_i = sqrt(d) normalize(B(s_{t+K})); w_i = the lstsq per-step weights of
    # that z (E[B B^T] on Z_COV_STATES unlabelled dataset states, whatever z_cov=). Then each eval mode (mppi: the usual planner with
    # per-step reward g(h)^T w (mppi_reward=lstsq) or g(h)^T z, terminal F^T z, policy samples, mppi_root_check; policy: the actor
    # pi(phi(s), z_i) alone) for goal_budget steps (default 2K). Success: the goal coordinates (antmaze (x, y), cube block) come within
    # goal_tol (OGBench: 0.5 | 0.4 = 0.04 m) of those of s_{t+K} at any step. Writes fbgoal_log.csv (one row per mode),
    # fbgoal_episodes.csv, goal_pairs.csv, then exits.
    FBGOAL_K = int(kwargs.get('fbgoal_k', 0))
    GOAL_N = int(kwargs.get('goal_n', 50))
    GOAL_SPLIT = str(kwargs.get('goal_split', 'val'))
    GOAL_BUDGET = int(kwargs.get('goal_budget', 2 * FBGOAL_K))
    GOAL_TOL = float(kwargs['goal_tol']) if 'goal_tol' in kwargs else None          # default: OGBench's radius
    GOAL_MIN_DIST = float(kwargs['goal_min_dist']) if 'goal_min_dist' in kwargs else None
    PAIR_SEED = int(kwargs.get('pair_seed', 0))
    assert not FBGOAL_K or (EVAL_CKPT and ENV_NAME in OGBENCH_ENVS), 'fbgoal_k needs eval_ckpt= and an OGBench env'
    # fbgoal_official=1 (eval_ckpt, OGBench): the same evaluation on the 5 official tasks: goal_n_task (default 10) episodes per task,
    # each from the task's own start, z = sqrt(d) normalize(B(goal observation of OGBench's reset)), goal_budget steps (default: the
    # official episode length). One fbgoal_log.csv row per mode for all tasks ('official') and one per task.
    FBGOAL_OFFICIAL = str(kwargs.get('fbgoal_official', '0')).lower() in ('1', 'true')
    GOAL_N_TASK = int(kwargs.get('goal_n_task', 10))
    assert not (FBGOAL_K and FBGOAL_OFFICIAL), 'fbgoal_k and fbgoal_official are exclusive'
    assert not FBGOAL_OFFICIAL or (EVAL_CKPT and ENV_NAME in OGBENCH_ENVS), 'fbgoal_official needs eval_ckpt= and an OGBench env'
    # Number of benchmark tasks rolled out in parallel during eval (each one needs NUM_EVALS env instances)
    EVAL_CHUNK = int(kwargs['eval_chunk']) if 'eval_chunk' in kwargs else N_BENCH

    # How the agent acts at eval time:
    #   mppi   : MPPI planning in latent space
    #   policy : the policy pi(a | phi(s), z) alone, deterministic (no planning, no noise)
    #   both   : (default) evaluate both at every eval; unprefixed columns = MPPI, `policy_` columns = policy
    EVAL_MODE = str(kwargs['eval_mode']) if 'eval_mode' in kwargs else 'both'
    assert EVAL_MODE in ['mppi', 'policy', 'both'], "eval_mode must be mppi | policy | both"
    EVAL_MODES = ['mppi', 'policy'] if EVAL_MODE == 'both' else [EVAL_MODE]
    MODE_NAMES = {'mppi': 'MPPI', 'policy': 'policy prior'}

    def eval_columns(prefix):
        return [f"{prefix}mean_over_tasks"] + [f"{prefix}{label}_{s}" for _, label, _ in benchmarks for s in ("mean", "std")]

    eval_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/eval_log.csv",
        fieldnames=["step"] + eval_columns('') + [f"{label}_corr" for _, label, _ in benchmarks]
                   + (eval_columns('policy_') if EVAL_MODE == 'both' else []) + ["num_evals", "tag", "time"]
                   + [f"{m}_time" for m in EVAL_MODES] + ["z_labels", "mppi_elite_spread"]
                   + [f"{label}_corr_lstsq" for _, label, _ in benchmarks],   # wall-clock seconds of each eval mode (all tasks, num_evals episodes; first eval includes jit)
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
        merge, and two heads. `h` and `z` may have different sizes; the outputs have z_dim entries."""
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
        """FB BackwardMap: Linear -> ntanh -> Linear -> relu -> Linear, then sqrt(z_dim) * normalize.
        Used as g, with the latent phi(s) (or an imagined latent) as input instead of the raw observation: B(s) = g(phi(s))."""
        hidden_dim: int
        z_dim:      int

        @nn.compact
        def __call__(self, x):
            x = NTanh()(TorchDense(self.hidden_dim)(x))
            x = nn.relu(TorchDense(self.hidden_dim)(x))
            x = TorchDense(self.z_dim)(x)
            return math.sqrt(self.z_dim) * l2_normalize(x)

    class PhiMap(nn.Module):
        """phi(s): Linear -> NTanh -> [Linear -> relu] x (num_layers - 1) -> Linear.
        Linear output, no normalization or projection. The defaults (num_layers=2, ntanh=True, hidden_dim=64)
        are the trunk that was stable inside F in our experiments."""
        hidden_dim: int  = 64
        out_dim:    int  = 50
        num_layers: int  = 2       # number of hidden layers before the output layer
        ntanh:      bool = True    # False: the first activation is a plain relu (no LayerNorm)

        @nn.compact
        def __call__(self, obs):
            x = TorchDense(self.hidden_dim)(obs)
            x = NTanh()(x) if self.ntanh else nn.relu(x)
            for _ in range(self.num_layers - 1):
                x = nn.relu(TorchDense(self.hidden_dim)(x))
            return TorchDense(self.out_dim)(x)

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

    # ── TD-MPC + FB agent: one trunk phi(s), with B(s) = g(phi(s)) ────────────────

    @dataclasses.dataclass(frozen=True)
    class TDMPCFBConfig:
        variant:             str   = 'gphi'  # 'sepB' | 'gphi' (set from the required variant= flag; see the top of the script)
        frozen_wm:           bool  = False   # gphi only. phi = TD-MPC MLP encoder + dynamics loaded from a reward-free JEPA world model, never updated (see wm_ckpt=)
        frozen_wm_phimap:    bool  = False   # set from wm_ckpt: the frozen encoder is a PhiMap (width phi_hidden_dim) instead of the TD-MPC MLP (enc_dim)
        phi_identity:        bool  = False   # gphi only. True: phi(s) = s (no encoder): F, B, pi act on the raw observation; the world model predicts the next raw state
        # optimisation (FB values: lr 1e-4, batch 1024, target EMA at every update)
        lr:                  float = 1e-4
        grad_clip_norm:      float = 10.0    # phi + dynamics only; g, F, pi are not clipped, as in FB
        tau:                 float = 0.01    # EMA of the target phi, target g and target F
        update_freq:         int   = 1
        batch_size:          int   = 1024
        # sizes: latent_dim = size of phi(s); z_dim = size of B(s), of F's output and of the task vector z
        latent_dim:          int   = 50      # try 128 to give the trunk more room
        z_dim:               int   = 50
        phi_hidden_dim:      int   = 64      # gphi: phi = PhiMap (hidden width)
        enc_dim:             int   = 256     # sepB: phi = TD-MPC MLP encoder (Linear-ELU x2, hidden width); also the B-head width is mlp_dim
        mlp_dim:             int   = 512     # dynamics
        backward_hidden_dim: int   = 526     # g: the FB BackwardMap architecture (same width as in the FB script)
        # F and pi: the FB architectures, applied on the latent phi(s)
        hidden_dim:          int   = 1024
        feature_dim:         int   = 512
        # losses
        horizon:             int   = 5       # length of the latent rollout of the consistency losses
        rho:                 float = 0.5
        discount:            float = 0.95
        consistency_coef:    float = 2.0     # latent consistency: imagined h_t vs EMA phi(s_t)
        bcons_coef:          float = 0.0     # gphi only. B-space consistency: g(imagined h_t) vs EMA B(s_t); 0 = off (set 1.0 to enable)
        fcons_coef:          float = 0.0     # M1 (2026-11-15): F-consistency, F(imagined h_t, a_t, z) vs EMA F(EMA phi(s_t), a_t, z); 0 = off
        fcons_model_grad:    bool  = False   # M2: the F-consistency gradient also flows into phi and the dynamics (M1: into F only)
        ortho_coef:          float = 0.0     # orthonormality loss on phi(s): off (B's own orthonormality loss is kept, b_ortho_coef)
        b_ortho_coef:        float = 1.0     # orthonormality loss on B(s'), inside the FB loss (FB's value)
        f_encoder_grad:      float = 0.1     # scale of the FB gradient that reaches phi through F's input
        b_encoder_grad:      float = 1.0     # gphi only. Scale of the FB gradient that reaches phi through B = g(phi)
        q_loss:              bool  = True    # FB's auxiliary Q-loss
        q_loss_coef:         float = 0.01
        mix_ratio:           float = 0.5     # fraction of task vectors taken from B(random dataset state), as in FB
        # policy noise in the FB target action and in the actor loss
        stddev:              float = 0.2
        bc_coef:             float = 0.0     # behaviour-cloning term of the policy (see bc_combine)
        std_clip:            float = 0.3
        # MPPI planning
        min_std:             float = 0.05
        iterations:          int   = 6
        num_samples:         int   = 512
        num_elites:          int   = 64
        mixture_coef:        float = 0.05    # policy trajectories = int(mixture_coef * num_samples), added on top
        temperature:         float = 0.5
        momentum:            float = 0.1
        mppi_terminal_only:  bool  = False   # eval diagnostic: plan with the terminal F only (no per-step B reward)
        mppi_norm:           bool  = False   # MPPI softmax on elite values standardised by their std (scale-free temperature)
        mppi_root_check:     bool  = False   # (a) execute the policy's action if F at the real latent rates it above MPPI's first action
        mppi_init_policy:    bool  = False   # (b) initial MPPI mean = the policy's deterministic trajectory through the model
        mppi_init_std:       float = 2.0     # initial MPPI std (TD-MPC: 2.0)

    class TDMPCFBTrainingState(flax.struct.PyTreeNode):
        params:                Any          # trunk + dynamics: {'encoder' (phi), 'dynamics'}
        b_params:              Any          # B's network: gphi g (B(s) = g(phi(s))); sepB the BackwardMap on the raw observation
        bh_params:             Any          # sepB: B-head B_hat(h) used by the planner ({} for gphi)
        f_params:              Any          # F (the FB forward map, taking the latent phi(s))
        target_params:         Any          # EMA of {'encoder', 'b', 'f'}
        pi_params:             Any
        optimizer_state:       optax.OptState
        b_optimizer_state:     optax.OptState
        bh_optimizer_state:    optax.OptState
        f_optimizer_state:     optax.OptState
        pi_optimizer_state:    optax.OptState
        random_key:            jnp.ndarray
        steps:                 jnp.ndarray

    class TDMPCFBAgent:

        def __init__(self, config: TDMPCFBConfig, obs_dim: int, action_dim: int):
            self.cfg        = config
            self.obs_dim    = obs_dim
            self.action_dim = action_dim
            c = config
            self.latent_dim = obs_dim if c.phi_identity else c.latent_dim                          # size of phi(s)

            self.sepB = c.variant == 'sepB'
            if self.sepB:
                assert c.latent_dim == c.z_dim, "sepB: the B-head maps the latent to B, keep latent_dim == z_dim"
                self.encoder = MLP(hidden=(c.enc_dim, c.enc_dim), out_dim=c.latent_dim)              # phi: TD-MPC encoder
            elif c.frozen_wm and not c.frozen_wm_phimap:
                self.encoder = MLP(hidden=(c.enc_dim, c.enc_dim), out_dim=c.latent_dim)              # phi: the TD-MPC encoder of tdmpc_adapt.py
            else:
                self.encoder = PhiMap(c.phi_hidden_dim, c.latent_dim)                               # phi: the trunk
            self.dynamics    = MLP(hidden=(c.mlp_dim, c.mlp_dim), out_dim=self.latent_dim)
            # B's network, the FB BackwardMap: sepB on the raw observation, gphi on the latent (g)
            self.b_net       = BackwardMap(c.backward_hidden_dim, c.z_dim)
            self.b_head_net  = MLP(hidden=(c.mlp_dim, c.mlp_dim), out_dim=c.z_dim) if self.sepB else None   # sepB: latent -> B (planner)
            self.forward_net = ForwardMap(c.hidden_dim, c.feature_dim, c.z_dim)                   # (phi, z, a) -> (F1, F2)
            self.pi_net      = Actor(c.hidden_dim, c.feature_dim, action_dim)                     # (phi, z) -> mu

            self.optimizer    = optax.chain(optax.clip_by_global_norm(c.grad_clip_norm), optax.adam(c.lr))
            self.b_optimizer  = optax.adam(c.lr)
            self.bh_optimizer = optax.adam(c.lr)
            self.f_optimizer  = optax.adam(c.lr)
            self.pi_optimizer = optax.adam(c.lr)

        # ── init ──────────────────────────────────────────────────────────────

        def init(self, random_key) -> TDMPCFBTrainingState:
            c = self.cfg
            random_key, ke, kd, kg, kf, kpi, kbh = jax.random.split(random_key, 7)
            obs = jnp.zeros((1, self.obs_dim))
            h_d, a_d, z_d = jnp.zeros((1, self.latent_dim)), jnp.zeros((1, self.action_dim)), jnp.zeros((1, c.z_dim))

            params = {
                'encoder':  self.encoder.init(ke, obs)['params'],
                'dynamics': self.dynamics.init(kd, jnp.zeros((1, self.latent_dim + self.action_dim)))['params'],
            }
            b_params  = self.b_net.init(kg, obs if self.sepB else h_d)['params']
            bh_params = self.b_head_net.init(kbh, h_d)['params'] if self.sepB else {}
            f_params  = self.forward_net.init(kf, h_d, z_d, a_d)['params']
            pi_params = self.pi_net.init(kpi, h_d, z_d)['params']
            target_params = {
                'encoder': jax.tree_util.tree_map(jnp.copy, params['encoder']),
                'b':       jax.tree_util.tree_map(jnp.copy, b_params),
                'f':       jax.tree_util.tree_map(jnp.copy, f_params),
            }

            return TDMPCFBTrainingState(
                params=params, b_params=b_params, bh_params=bh_params, f_params=f_params, target_params=target_params,
                pi_params=pi_params,
                optimizer_state=self.optimizer.init(params),
                b_optimizer_state=self.b_optimizer.init(b_params),
                bh_optimizer_state=self.bh_optimizer.init(bh_params),
                f_optimizer_state=self.f_optimizer.init(f_params),
                pi_optimizer_state=self.pi_optimizer.init(pi_params),
                random_key=random_key, steps=jnp.zeros((), dtype=jnp.int32),
            )

        # ── model ─────────────────────────────────────────────────────────────

        def _sphere(self, x):
            """Project onto the sphere of radius sqrt(dim), so each coordinate has RMS ~1."""
            return math.sqrt(x.shape[-1]) * l2_normalize(x)
            # return x

        def encode(self, encoder_params, obs):
            """phi(s) = h, on the sphere of radius sqrt(latent_dim)."""
            if self.cfg.phi_identity:
                return obs                                                           # phi(s) = s
            return self._sphere(self.encoder.apply({'params': encoder_params}, obs))

        def next(self, params, h, a):
            """Residual dynamics h' = normalize(h + d(h, a))."""
            x = jnp.concatenate([h, a], axis=-1)
            h_next = h + self.dynamics.apply({'params': params['dynamics']}, x)
            return h_next if self.cfg.phi_identity else self._sphere(h_next)         # raw states are not on a sphere

        def b_from_latent(self, b_params, h):
            """gphi: B on a latent, g(h), on the sphere of radius sqrt(z_dim). Works on real latents phi(s) and on imagined ones."""
            return self.b_net.apply({'params': b_params}, h)                         # the BackwardMap normalises to the sphere

        def b_head(self, bh_params, h):
            """sepB: B_hat(h), the prediction of B(s) from the latent h = phi(s) (or an imagined latent), on the sphere."""
            return self._sphere(self.b_head_net.apply({'params': bh_params}, h))

        def backward(self, encoder_params, b_params, obs):
            """B(s): sepB the BackwardMap on the raw observation; gphi g(phi(s))."""
            if self.sepB:
                return self.b_net.apply({'params': b_params}, obs)
            return self.b_from_latent(b_params, self.encode(encoder_params, obs))

        def reward_feature(self, b_params, bh_params, h):
            """The feature whose dot product with z is MPPI's per-step model reward on an imagined latent h: sepB B_hat(h), gphi g(h)."""
            return self.b_head(bh_params, h) if self.sepB else self.b_from_latent(b_params, h)

        def F(self, f_params, h, a, z):
            """(F1(h, a, z), F2(h, a, z)); h = phi(s). `f_params` can be the online or the target parameters."""
            return self.forward_net.apply({'params': f_params}, h, z, a)

        def score(self, x, z):
            """x^T z / d. For x = B this is a cosine in [-1, 1]; for x = F it is the Q-value (per-dimension scale)."""
            return jnp.sum(x * z, axis=-1, keepdims=True) / self.cfg.z_dim

        def pi(self, pi_params, h, z, key, std):
            """Deterministic policy pi(a | phi(s), z) (the FB Actor already applies tanh) + clipped Gaussian noise (std=0 -> none)."""
            mu = self.pi_net.apply({'params': pi_params}, h, z)
            return sample_truncated_normal(mu, key, std, self.cfg.std_clip) if std > 0 else mu

        def sample_z(self, random_key, size: int):
            """Task vectors uniform on the sphere of radius sqrt(z_dim)."""
            return math.sqrt(self.cfg.z_dim) * l2_normalize(jax.random.normal(random_key, (size, self.cfg.z_dim)))

        # ── zero-shot task inference (in B-space, as in FB) ───────────────────

        @partial(jax.jit, static_argnames=("self",))
        def infer_z(self, encoder_params, b_params, obs, reward):
            """z = sqrt(d) * normalize(mean_s B(s) R(s)). Also returns corr(B(s)^T z, R(s)) on the same samples."""
            B_  = self.backward(encoder_params, b_params, obs)                       # (N, z_dim)
            z   = (reward[:, None] * B_).mean(axis=0)
            z   = math.sqrt(self.cfg.z_dim) * l2_normalize(z)
            pred = B_ @ z
            pc, rc = pred - pred.mean(), reward - reward.mean()
            corr = (pc * rc).mean() / (pc.std() * rc.std() + 1e-8)
            return z, corr

        @partial(jax.jit, static_argnames=("self",))
        def infer_zw(self, encoder_params, b_params, obs, reward, ridge, cov_obs=None):
            """(z, w, corr_lstsq). z = c E[B R] on the sphere (as infer_z); w = c (E[B B^T] + lam I)^-1 E[B R], same c.
            corr_lstsq: corr(R, centred least-squares fit of R on B) = how well span(B) (+ a constant) represents R."""
            B_  = self.backward(encoder_params, b_params, obs)
            d   = B_.shape[-1]
            b_r = (reward[:, None] * B_).mean(axis=0)
            c   = math.sqrt(self.cfg.z_dim) / jnp.maximum(jnp.linalg.norm(b_r), 1e-8)
            if cov_obs is None:
                cov = B_.T @ B_ / B_.shape[0]
            else:                                                       # unlabelled states: E_rho[B B^T] needs no reward
                B_u = self.backward(encoder_params, b_params, cov_obs)
                cov = B_u.T @ B_u / B_u.shape[0]
            lam = ridge * jnp.trace(cov) / d
            w   = c * jnp.linalg.solve(cov + lam * jnp.eye(d), b_r)
            Bc, Rc = B_ - B_.mean(0), reward - reward.mean()
            covc = Bc.T @ Bc / B_.shape[0]
            wc  = jnp.linalg.solve(covc + lam * jnp.eye(d), (Rc[:, None] * Bc).mean(0))
            pred = Bc @ wc
            corr = (pred * Rc).mean() / (pred.std() * Rc.std() + 1e-8)
            return c * b_r, w, corr

        @partial(jax.jit, static_argnames=("self",))
        def reconstruct_reward(self, encoder_params, b_params, obs, z):
            """Linear reward reconstruction B(s)^T z / d (used for the plots)."""
            return self.score(self.backward(encoder_params, b_params, obs), z)[..., 0]

        # ── update ────────────────────────────────────────────────────────────

        @partial(jax.jit, static_argnames=("self",))
        def update(self, ts: TDMPCFBTrainingState, replay_buffer: OfflineReplayBuffer):
            """One update. Both variants share the world model (consistency), F, pi and the FB loss; they differ in B:
              sepB : B = BackwardMap(s) (raw observation); phi gets consistency + f_encoder_grad x the FB gradient through F's input;
                     the B-head B_hat(h) is regressed on B(s_t) from real latents and from imagined ones (its own parameters only).
              gphi : B = g(phi(s)); phi also gets b_encoder_grad x the FB gradient through B, and g the B-space consistency (bcons)."""
            c = self.cfg
            H, B = c.horizon, c.batch_size
            key, k_batch, k_z, k_perm, k_mix, k_next, k_pi = jax.random.split(ts.random_key, 7)

            obs, action = replay_buffer.sample_sequences(k_batch, B, H)              # (H+1,B,D) (H,B,A)

            # ── task vectors (B-space): uniform on the sphere, a fraction replaced by B(random state) as in FB ──
            z = self.sample_z(k_z, B)
            if c.mix_ratio > 0:
                perm     = jax.random.permutation(k_perm, B)
                z_data   = self.backward(ts.params['encoder'], ts.b_params, obs[0][perm])      # already on the sphere
                mix_mask = jax.random.uniform(k_mix, (B, 1)) < c.mix_ratio
                z        = jnp.where(mix_mask, z_data, z)

            # ── constants: computed outside the differentiated functions below, so no gradient flows into them ──
            target_h = self.encode(ts.target_params['encoder'], obs[1:])                           # (H,B,L) EMA phi(s_t)
            if self.sepB:
                target_B     = self.backward(None, ts.target_params['b'], obs[1])                  # (B,D)   EMA B(s'_j)
                target_B_all = None
                h_all        = self.encode(ts.params['encoder'], obs)                              # (H+1,B,L) online phi of every state
                B_all        = self.backward(None, ts.b_params, obs)                               # (H+1,B,D) online B: B-head targets
            else:
                target_B_all = self.b_from_latent(ts.target_params['b'], target_h)                 # (H,B,D) EMA B(s_t)
                target_B     = target_B_all[0]                                                     # (B,D)   EMA B(s'_j)
                h_all        = self.encode(ts.params['encoder'], obs[:2])                          # (2,B,L) online phi(s), phi(s')
            h_in, next_h = h_all[0], h_all[1]                                                      # policy inputs
            # FB target: target F at the target encoder's latent, with the next action of the (noisy) policy
            next_a   = self.pi(ts.pi_params, next_h, z, k_next, c.stddev)
            tF1, tF2 = self.F(ts.target_params['f'], target_h[0], next_a, z)                       # (B,D)
            target_M = jnp.minimum(tF1 @ target_B.T, tF2 @ target_B.T)                             # (B,B)
            next_Q   = jnp.minimum((tF1 * z).sum(-1), (tF2 * z).sum(-1))
            rho_t    = c.rho ** jnp.arange(H, dtype=jnp.float32)
            # F-consistency targets (M1/M2): EMA F at the EMA latent of s_k, k = 1..H-1, with the dataset action a_k and the same z
            if c.fcons_coef > 0:
                z_rep = jnp.broadcast_to(z, (H - 1,) + z.shape)
                fc_t1, fc_t2 = self.F(ts.target_params['f'], target_h[:H - 1], action[1:], z_rep)           # (H-1,B,D)

            # ── phi + dynamics (+ g for gphi): latent consistency, B-space consistency (gphi), orthonormality on phi ──
            def model_loss_fn(params, b_params, f_params):
                h0 = self.encode(params['encoder'], obs[0])

                def rollout_step(h, xs):
                    a_t, target_h_t, target_B_t, rho = xs
                    h_next = self.next(params, h, a_t)
                    # mean squared error on the sphere == 2 - 2 cos(., target) in [0, 4]
                    cons = rho * jnp.mean((h_next - target_h_t) ** 2, axis=-1)                                # (B,)
                    if self.sepB:
                        bcons = jnp.zeros_like(cons)
                    else:
                        bcons = rho * jnp.mean((self.b_from_latent(b_params, h_next) - target_B_t) ** 2, axis=-1)
                    return h_next, (h_next, cons, bcons)

                xs_B = target_h if target_B_all is None else target_B_all       # placeholder of the right shape for sepB
                _, (hs_next, cons, bcons) = jax.lax.scan(rollout_step, h0, (action, target_h, xs_B, rho_t))   # (H,B,..)
                consistency_loss = cons.sum(0).mean()
                bcons_loss       = bcons.sum(0).mean()
                ortho_loss       = jnp.zeros(()) if c.phi_identity else gram_ortho(h0)
                if c.fcons_coef > 0:
                    # hs_next[k] is the imagined latent of s_{k+1}; M1: constant (gradient into F only), M2: into phi and the dynamics
                    h_im = hs_next[:H - 1] if c.fcons_model_grad else jax.lax.stop_gradient(hs_next[:H - 1])
                    f1, f2 = self.F(f_params, h_im, action[1:], z_rep)
                    fc = jnp.mean((f1 - fc_t1) ** 2, axis=-1) + jnp.mean((f2 - fc_t2) ** 2, axis=-1)       # (H-1,B)
                    fcons_loss = (rho_t[:H - 1, None] * fc).sum(0).mean()
                else:
                    fcons_loss = jnp.zeros(())
                model_loss = ((c.consistency_coef * consistency_loss + c.bcons_coef * bcons_loss + c.fcons_coef * fcons_loss) / H
                              + c.ortho_coef * ortho_loss)
                std, cov_l2, mean_norm = feature_stats(h0)
                aux = {'hs_next': hs_next, 'consistency_loss': consistency_loss, 'bcons_loss': bcons_loss, 'ortho_loss': ortho_loss,
                       'fcons_loss': fcons_loss,
                       'latent_std': std, 'feat_cov_l2': cov_l2, 'feat_mean_norm': mean_norm}
                return model_loss, aux

            # ── FB loss on the first transition of each sequence: M[i, j] = F(phi(s_i), a_i, z_i)^T B(s'_j) ──
            # phi enters through two separate copies of the encoder parameters (so each path gets its own gradient):
            #  * enc_f : phi(s_i), the input of F (scaled by f_encoder_grad below)
            #  * enc_b : gphi only, phi(s'_j), the input of g, i.e. the B side (scaled by b_encoder_grad); sepB: B ignores phi
            def fb_loss_fn(enc_f, enc_b, b_params, f_params):
                h0     = self.encode(enc_f, obs[0])
                F1, F2 = self.F(f_params, h0, action[0], z)                                         # (B,D)
                B_s    = self.backward(enc_b, b_params, obs[1])                                     # (B,D): B(s'_j)
                fb_loss, fb_aux = fb_losses(F1, F2, B_s, z, target_M, next_Q, c.discount, c.q_loss_coef, c.q_loss)
                b_ortho = gram_ortho(B_s)
                b_std, b_cov_l2, b_mean_norm = feature_stats(B_s)
                aux = dict(fb_aux, fb_loss=fb_loss, b_ortho_loss=b_ortho, F_norm=jnp.linalg.norm(F1, axis=-1).mean(),
                           b_std=b_std, b_cov_l2=b_cov_l2, b_mean_norm=b_mean_norm)
                return fb_loss + c.b_ortho_coef * b_ortho, aux

            (model_loss, m_aux), (grad_model, grad_b_model, grad_f_model) = jax.value_and_grad(
                model_loss_fn, argnums=(0, 1, 2), has_aux=True)(ts.params, ts.b_params, ts.f_params)
            (_, fb_aux), (grad_enc_f, grad_enc_b, grad_b_fb, grad_f) = jax.value_and_grad(
                fb_loss_fn, argnums=(0, 1, 2, 3), has_aux=True)(
                ts.params['encoder'], ts.params['encoder'], ts.b_params, ts.f_params)

            grad_enc_f = jax.tree_util.tree_map(lambda g: c.f_encoder_grad * g, grad_enc_f)
            grad_enc_b = jax.tree_util.tree_map(lambda g: c.b_encoder_grad * g, grad_enc_b)    # identically 0 for sepB
            grads = {
                'encoder':  jax.tree_util.tree_map(lambda a, b_, c_: a + b_ + c_, grad_model['encoder'], grad_enc_f, grad_enc_b),
                'dynamics': grad_model['dynamics'],
            }
            grad_b = jax.tree_util.tree_map(jnp.add, grad_b_model, grad_b_fb)                     # grad_b_model = 0 for sepB
            grad_f = jax.tree_util.tree_map(jnp.add, grad_f, grad_f_model)                         # grad_f_model = 0 unless fcons_coef > 0
            model_enc_grad_norm = optax.global_norm(grad_model['encoder'])     # consistency (+ B-consistency) + orthonormality
            f_enc_grad_norm     = optax.global_norm(grad_enc_f)                # from F's input (after scaling)
            b_enc_grad_norm     = optax.global_norm(grad_enc_b)                # from B = g(phi) (after scaling; 0 for sepB)
            f_grad_norm         = optax.global_norm(grad_f)
            b_grad_norm         = optax.global_norm(grad_b)

            updates, optimizer_state = self.optimizer.update(grads, ts.optimizer_state, ts.params)
            params = optax.apply_updates(ts.params, updates)
            if c.frozen_wm:                                                    # frozen world model: phi and the dynamics stay as loaded
                params, optimizer_state = ts.params, ts.optimizer_state
            b_updates, b_optimizer_state = self.b_optimizer.update(grad_b, ts.b_optimizer_state, ts.b_params)
            b_params = optax.apply_updates(ts.b_params, b_updates)
            f_updates, f_optimizer_state = self.f_optimizer.update(grad_f, ts.f_optimizer_state, ts.f_params)
            f_params = optax.apply_updates(ts.f_params, f_updates)

            # ── sepB: B-head (for the planner), predicts B(s_t) from phi(s_t) and from the imagined latents; inputs are constants ──
            if self.sepB:
                hs_next = jax.lax.stop_gradient(m_aux['hs_next'])                                  # (H,B,L) imagined latents

                def bhead_loss_fn(bh_params):
                    p_real = self.b_head(bh_params, h_all)                                          # (H+1,B,D) vs B(s_t)
                    p_imag = self.b_head(bh_params, hs_next)                                        # (H,B,D)   imagined vs B(s_t)
                    return jnp.mean((p_real - B_all) ** 2) + jnp.mean((p_imag - B_all[1:]) ** 2)

                bhead_loss, g_bh = jax.value_and_grad(bhead_loss_fn)(ts.bh_params)
                bh_updates, bh_optimizer_state = self.bh_optimizer.update(g_bh, ts.bh_optimizer_state, ts.bh_params)
                bh_params = optax.apply_updates(ts.bh_params, bh_updates)
            else:
                bhead_loss, bh_params, bh_optimizer_state = jnp.zeros(()), ts.bh_params, ts.bh_optimizer_state

            # ── policy: maximise min_i F_i(phi(s), a, z)^T z on the real latents (phi(s) is a constant here) ──
            def pi_loss_fn(pi_params):
                a = self.pi(pi_params, h_in, z, k_pi, c.stddev)
                F1, F2 = self.F(f_params, h_in, a, z)
                Q = jnp.minimum((F1 * z).sum(-1), (F2 * z).sum(-1))
                return bc_combine(-Q.mean(), jnp.mean((a - action[0]) ** 2), c.bc_coef), Q.mean()

            (pi_loss, q_mean), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(ts.pi_params)
            pi_updates, pi_optimizer_state = self.pi_optimizer.update(pi_grads, ts.pi_optimizer_state, ts.pi_params)
            pi_params = optax.apply_updates(ts.pi_params, pi_updates)

            # ── EMA of the target phi, target B network and target F (every `update_freq` updates) ──
            tau = c.tau * (ts.steps % c.update_freq == 0)
            online = {'encoder': params['encoder'], 'b': b_params, 'f': f_params}
            target_params = jax.tree_util.tree_map(lambda p, t: t + tau * (p - t), online, ts.target_params)

            ts = ts.replace(params=params, b_params=b_params, bh_params=bh_params, f_params=f_params, target_params=target_params,
                            pi_params=pi_params, optimizer_state=optimizer_state, b_optimizer_state=b_optimizer_state,
                            bh_optimizer_state=bh_optimizer_state, f_optimizer_state=f_optimizer_state,
                            pi_optimizer_state=pi_optimizer_state, random_key=key, steps=ts.steps + 1)
            metrics = {
                'model_loss': model_loss, 'consistency_loss': m_aux['consistency_loss'], 'bcons_loss': m_aux['bcons_loss'],
                'ortho_loss': m_aux['ortho_loss'],
                'model_enc_grad_norm': model_enc_grad_norm, 'f_enc_grad_norm': f_enc_grad_norm, 'b_enc_grad_norm': b_enc_grad_norm,
                'fb_loss': fb_aux['fb_loss'], 'fb_diag': fb_aux['fb_diag'], 'fb_offdiag': fb_aux['fb_offdiag'],
                'q_loss': fb_aux['q_loss'], 'b_ortho_loss': fb_aux['b_ortho_loss'],
                'f_grad_norm': f_grad_norm, 'b_grad_norm': b_grad_norm, 'M1': fb_aux['M1'], 'F_norm': fb_aux['F_norm'],
                'pi_loss': pi_loss, 'q_mean': q_mean, 'bhead_loss': bhead_loss,
                'latent_std': m_aux['latent_std'], 'feat_cov_l2': m_aux['feat_cov_l2'], 'feat_mean_norm': m_aux['feat_mean_norm'],
                'b_std': fb_aux['b_std'], 'b_cov_l2': fb_aux['b_cov_l2'], 'b_mean_norm': fb_aux['b_mean_norm'],
                'fcons_loss': m_aux['fcons_loss'],
            }
            return ts, metrics

        # ── MPPI planning in the latent space of phi, for a given task vector z ──────────

        def _estimate_value(self, params, b_params, bh_params, f_params, pi_params, h, z, actions, key, w=None):
            """Sum_t gamma^t r(h_{t+1})^T z / d over the horizon, r = B_hat (sepB) or g (gphi) on the imagined latent, plus the
            bootstrapped min_i F_i(h_H, pi(h_H, z), z)^T z / d at the end. h (S,L), z (S,D), actions (H,S,A)."""
            c = self.cfg
            G, discount = 0.0, 1.0
            for t in range(c.horizon):
                h = self.next(params, h, actions[t])
                if not c.mppi_terminal_only:
                    G = G + discount * self.score(self.reward_feature(b_params, bh_params, h), z if w is None else w)   # sepB B_hat(h), gphi g(h)
                discount = discount * c.discount
            a = self.pi(pi_params, h, z, key, c.min_std)
            F1, F2 = self.F(f_params, h, a, z)
            q = jnp.minimum(self.score(F1, z), self.score(F2, z))
            return (G + discount * q)[..., 0]

        def _plan_single(self, ts, obs, z_task, w_task, prev_mean, use_prev, key):
            """MPPI for one env and one task vector. obs (obs_dim,), z_task (D,), prev_mean (H, A)."""
            c = self.cfg
            H, A = c.horizon, self.action_dim
            params, b_params, bh_params, f_params, pi_params = ts.params, ts.b_params, ts.bh_params, ts.f_params, ts.pi_params
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

            if c.mppi_init_policy:   # (b) the policy's deterministic trajectory through the model
                _hd, pol_traj = h0[None], []
                for t in range(H):
                    a_d = self.pi(pi_params, _hd, z_task[None], None, 0.0)
                    pol_traj.append(a_d[0])
                    _hd = self.next(params, _hd, a_d)
                pol_traj = jnp.stack(pol_traj)                                      # (H, A)
            h    = jnp.tile(h0, (c.num_samples + num_pi, 1))
            z    = jnp.tile(z_task, (c.num_samples + num_pi, 1))
            w    = jnp.tile(w_task, (c.num_samples + num_pi, 1))       # per-step reward weights (= z unless mppi_reward=lstsq)
            mean = jnp.where(use_prev, jnp.concatenate([prev_mean[1:], jnp.zeros((1, A))], axis=0), jnp.zeros((H, A)))
            if c.mppi_init_policy:
                mean = pol_traj
            std  = c.mppi_init_std * jnp.ones((H, A))

            for _ in range(c.iterations):
                key, k_noise, k_val = jax.random.split(key, 3)
                noise   = jax.random.normal(k_noise, (H, c.num_samples, A))
                sampled = jnp.clip(mean[:, None] + std[:, None] * noise, -1, 1)
                actions = jnp.concatenate([sampled, pi_actions], axis=1)            # (H, S + num_pi, A)

                value = jnp.nan_to_num(self._estimate_value(params, b_params, bh_params, f_params, pi_params, h, z, actions, k_val, w))
                elite_value, elite_idx = jax.lax.top_k(value, c.num_elites)
                elite_actions = actions[:, elite_idx]                               # (H, E, A)

                spread = elite_value.std()
                logits = c.temperature * (elite_value - elite_value.max()) / ((spread + 1e-6) if c.mppi_norm else 1.0)
                score  = jax.nn.softmax(logits)
                _mean  = (score[None, :, None] * elite_actions).sum(axis=1)
                _std   = jnp.sqrt((score[None, :, None] * (elite_actions - _mean[:, None]) ** 2).sum(axis=1))
                _std   = jnp.clip(_std, c.min_std, 2.0)
                mean, std = c.momentum * mean + (1 - c.momentum) * _mean, _std

            key, sub = jax.random.split(key)
            chosen = jax.random.categorical(sub, logits)
            action = elite_actions[0, chosen]
            if c.mppi_root_check:   # (a) keep the policy's action unless F at the real latent rates MPPI's first action higher
                a_pol = self.pi(pi_params, h0[None], z_task[None], None, 0.0)[0]
                cand  = jnp.stack([action, a_pol])                                  # (2, A)
                F1, F2 = self.F(f_params, jnp.tile(h0, (2, 1)), cand, jnp.tile(z_task, (2, 1)))
                qv = jnp.minimum(self.score(F1, jnp.tile(z_task, (2, 1))), self.score(F2, jnp.tile(z_task, (2, 1))))[:, 0]
                action = jnp.where(qv[1] > qv[0], a_pol, action)
            return action, mean, spread                                             # eval mode: no exploration noise

        @partial(jax.jit, static_argnames=("self",))
        def plan_batch(self, ts, obs, zs, ws, prev_mean, use_prev, key):
            """MPPI for a batch of envs, env i conditioned on task vector zs[i] (per-step reward weights ws[i]). obs (N, obs_dim)."""
            keys = jax.random.split(key, obs.shape[0])
            return jax.vmap(self._plan_single, in_axes=(None, 0, 0, 0, 0, None, 0))(ts, obs, zs, ws, prev_mean, use_prev, keys)

        @partial(jax.jit, static_argnames=("self",))
        def act_policy_batch(self, ts, obs, zs):
            """Policy alone, no planning: a = pi(phi(s), z), deterministic (no exploration noise)."""
            h = self.encode(ts.params['encoder'], obs)
            return self.pi(ts.pi_params, h, zs, None, 0.0)

    tdmpc_config = TDMPCFBConfig(variant=VARIANT, bc_coef=BC_COEF, batch_size=BATCH_SIZE, horizon=HORIZON, phi_identity=PHI_IDENTITY, frozen_wm=FREEZE_WM)
    for name in ('lr', 'discount', 'consistency_coef', 'bcons_coef', 'ortho_coef', 'b_ortho_coef', 'f_encoder_grad',
                 'b_encoder_grad', 'q_loss_coef', 'mix_ratio', 'stddev', 'tau'):
        if name in kwargs:
            tdmpc_config = dataclasses.replace(tdmpc_config, **{name: float(kwargs[name])})
    if 'fcons_coef' in kwargs:
        tdmpc_config = dataclasses.replace(tdmpc_config, fcons_coef=float(kwargs['fcons_coef']))
    for name in ('mixture_coef', 'mppi_init_std', 'temperature', 'min_std', 'momentum'):   # planner settings (eval-time); defaults unchanged
        if name in kwargs:
            tdmpc_config = dataclasses.replace(tdmpc_config, **{name: float(kwargs[name])})
    for name in ('iterations', 'num_samples', 'num_elites'):
        if name in kwargs:
            tdmpc_config = dataclasses.replace(tdmpc_config, **{name: int(kwargs[name])})
    for name in ('fcons_model_grad', 'mppi_terminal_only', 'mppi_norm', 'mppi_root_check', 'mppi_init_policy'):
        if name in kwargs:
            tdmpc_config = dataclasses.replace(tdmpc_config, **{name: str(kwargs[name]).lower() in ('1', 'true')})
    for name in ('latent_dim', 'z_dim', 'enc_dim', 'phi_hidden_dim', 'hidden_dim', 'feature_dim', 'mlp_dim'):
        if name in kwargs:
            tdmpc_config = dataclasses.replace(tdmpc_config, **{name: int(kwargs[name])})
    if FREEZE_WM:
        tdmpc_config = dataclasses.replace(tdmpc_config, frozen_wm_phimap=FROZEN_WM_PHIMAP,
                                           **({'phi_hidden_dim': FROZEN_WM_WIDTH} if FROZEN_WM_PHIMAP else {'enc_dim': FROZEN_WM_WIDTH}))
    if VARIANT == 'sepB':
        assert 'bcons_coef' not in kwargs and 'b_encoder_grad' not in kwargs, "bcons_coef and b_encoder_grad are gphi options"
    print(tdmpc_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(tdmpc_config) + '\n')

    agent = TDMPCFBAgent(tdmpc_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)

    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key = jax.random.split(random_key)
    train_state = agent.init(init_key)
    if FREEZE_WM:
        with open(WM_CKPT, 'rb') as f:
            wm = pickle.load(f)
        enc = serialization.from_state_dict(train_state.params['encoder'], wm['encoder'])
        dyn = serialization.from_state_dict(train_state.params['dynamics'], wm['dynamics'])
        train_state = train_state.replace(
            params={'encoder': enc, 'dynamics': dyn},
            target_params=dict(train_state.target_params, encoder=jax.tree_util.tree_map(jnp.copy, enc)))
        train_state = train_state.replace(optimizer_state=agent.optimizer.init(train_state.params))
        print('loaded the frozen world model from', WM_CKPT)
    print('parameters (phi + dynamics + B + B-head + F + pi):',
          sum(x.size for x in jax.tree_util.tree_leaves(
              (train_state.params, train_state.b_params, train_state.bh_params, train_state.f_params, train_state.pi_params))))

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
    mppi_spread_log = []   # std of the final-iteration elite values at every MPPI step (reset at every eval row)
    corr_ls_log = {}       # task index -> corr of R with its least-squares fit on B (reset at every eval row)

    def rollout_batch(ts, zs, random_key, mode='mppi', ws=None, labels=None):
        """len(zs) envs rolled out in lock-step, env i conditioned on task vector zs[i]. One batched jitted call per step.
        mode='mppi': MPPI planning. mode='policy': the policy alone (deterministic, no planning)."""
        n = zs.shape[0]
        while len(eval_envs) < n:
            eval_envs.append(make_eval_env())
        envs = eval_envs[:n]

        states    = [reset_for(e, None if labels is None else labels[i]) for i, e in enumerate(envs)]
        prev_mean = jnp.zeros((n, tdmpc_config.horizon, ACTION_DIM))
        obs_l, aux_l, phys_l = [], [], []
        elite_spreads = mppi_spread_log
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
                actions, prev_mean, spread = agent.plan_batch(ts, jnp.asarray(obs), zs, zs if ws is None else ws, prev_mean,
                                                              jnp.asarray(step > 0), sub)
                elite_spreads.append(float(jnp.mean(spread)))
            states = [env_step(e, a) for e, a in zip(envs, np.asarray(actions))]

        obs_traj  = np.stack(obs_l).transpose(1, 0, 2)
        aux_traj  = np.stack(aux_l).transpose(1, 0, 2)
        phys_traj = np.stack(phys_l).transpose(1, 0, 2) if phys_l else None
        return obs_traj, aux_traj, phys_traj

    def plot_benchmark(axs, b, z, obs_traj, aux_traj, phys_traj, encoder_params, b_params, mode):
        """3 panels: real reward on the dataset, its linear reconstruction B(s)^T z / d, and the agent trajectories."""
        reward_fn, label, param = benchmarks[b]
        eval_obs, _, _, eval_aux = replay_buffer.sample(next_eval_key(), 10_000)
        real_rewards  = np.asarray(reward_fn(eval_aux, param))
        recon_rewards = np.asarray(agent.reconstruct_reward(encoder_params, b_params, eval_obs, z))
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

        elif ENV_NAME in OGBENCH_ENVS:   # antmaze: ant (x, y); cube: block (x, y)
            ix = (0, 1) if ENV_NAME != 'cube_single' else (19, 20)
            axs[b, 0].scatter(eval_obs[..., ix[0]], eval_obs[..., ix[1]], c=real_rewards, s=2)
            axs[b, 1].scatter(eval_obs[..., ix[0]], eval_obs[..., ix[1]], c=recon_rewards, s=2)
            axs[b, 2].scatter(obs_traj[..., ix[0]], obs_traj[..., ix[1]], c='red', s=2)

        axs[b, 0].set_title(f'{label}')
        axs[b, 1].set_title('Linear reconstruction B(s)^T z')
        axs[b, 2].set_title(f'Agent Trajectory ({MODE_NAMES[mode]})')

    def run_benchmark(train_state, steps, make_plot, mode='mppi', tag='', num_evals=NUM_EVALS):
        """Zero-shot evaluation on all tasks. For each task and each of the num_evals episodes, a fresh task vector is
        inferred from 10k dataset states. Returns per-task (mean return, std, corr(B^T z, R))."""
        encoder_params, b_params = train_state.params['encoder'], train_state.b_params

        if make_plot:
            fig, axs = plt.subplots(N_BENCH, 3, figsize=(15, N_BENCH * 4))

        means, stds, corrs = np.zeros(N_BENCH), np.zeros(N_BENCH), np.zeros(N_BENCH)
        for start in range(0, N_BENCH, EVAL_CHUNK):
            ids = list(range(start, min(start + EVAL_CHUNK, N_BENCH)))

            zs, ws, cs = [], [], []
            cov_obs = replay_buffer.sample(next_eval_key(), Z_COV_STATES)[0] if Z_COV == 'dataset' else None
            for b in ids:
                reward_fn, _, param = benchmarks[b]
                for _ in range(num_evals):
                    eval_obs, _, _, eval_aux = replay_buffer.sample(next_eval_key(), Z_LABELS)
                    r_lab = reward_fn(eval_aux, param)
                    z, corr = agent.infer_z(encoder_params, b_params, eval_obs, r_lab)
                    _, w, corr_ls = agent.infer_zw(encoder_params, b_params, eval_obs, r_lab, Z_RIDGE, cov_obs)
                    zs.append(z); ws.append(w); cs.append(corr); corr_ls_log.setdefault(b, []).append(float(corr_ls))
            zs, ws, cs = jnp.stack(zs), jnp.stack(ws), np.asarray(jnp.stack(cs))

            obs_traj, aux_traj, phys_traj = rollout_batch(train_state, zs, next_eval_key(), mode,
                                                          ws if MPPI_REWARD == 'lstsq' else None,
                                                          labels=[benchmarks[b][1] for b in ids for _ in range(num_evals)])

            for j, b in enumerate(ids):
                reward_fn, label, param = benchmarks[b]
                sl = slice(j * num_evals, (j + 1) * num_evals)

                # same metric as the FB script: sum over the trajectory of the task reward of the visited states
                trajectory_rewards = np.asarray(reward_fn(jnp.asarray(aux_traj[sl], dtype=jnp.float32), param)).sum(axis=-1)
                if (ENV_NAME == 'antmaze' or ENV_NAME in OGBENCH_ENVS) and 'goal' in label:
                    # goal tasks: 1 if the goal was reached at least once, else 0
                    trajectory_rewards = np.where(trajectory_rewards != -EVAL_STEPS, 1.0, 0.0)

                print(label, f'[{MODE_NAMES[mode]}]', ':')
                print('\tRewards:', trajectory_rewards.tolist())
                print('\tmean:', trajectory_rewards.mean())
                print('\tstd:',  trajectory_rewards.std(ddof=1))
                print('\tcorr(B^T z, R):', float(cs[sl].mean()))
                means[b], stds[b], corrs[b] = trajectory_rewards.mean(), trajectory_rewards.std(ddof=1), cs[sl].mean()

                if make_plot:
                    plot_benchmark(axs, b, zs[(j + 1) * num_evals - 1], obs_traj[sl], aux_traj[sl],
                                   None if phys_traj is None else phys_traj[sl], encoder_params, b_params, mode)

        if make_plot:
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/benchmark{tag}-steps:{steps}.png")
            plt.close()

        return means, stds, corrs

    def save_agent(train_state, path):
        state = {
            'params':        train_state.params,
            'b_params':      train_state.b_params,
            'bh_params':     train_state.bh_params,
            'f_params':      train_state.f_params,
            'target_params': train_state.target_params,
            'pi_params':     train_state.pi_params,
        }
        state_dict = serialization.to_state_dict(jax.device_get(state))
        with open(path, "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # ── Training ──────────────────────────────────────────────────────────────────

    rewards_logs = {mode: np.zeros((N_BENCH, 0)) for mode in EVAL_MODES}
    n_evals = 0

    def evaluate(train_state, num_updates, make_plot, num_evals, tag, modes=None):
        """Every eval mode on every task; one eval_log row (tag 'eval' = periodic, 'final' = after the last update)."""
        start_time = time.time()
        results, mode_times = {}, {}
        mppi_spread_log.clear(); corr_ls_log.clear()
        for i, mode in enumerate(modes or EVAL_MODES):
            plot_tag = ('' if i == 0 else f'-{mode}') + ('-final' if tag == 'final' else '')
            t0 = time.time()
            results[mode] = run_benchmark(train_state, num_updates, make_plot, mode, tag=plot_tag, num_evals=num_evals)
            mode_times[f'{mode}_time'] = time.time() - t0

        eval_row = {'step': num_updates, 'num_evals': num_evals, 'tag': tag, 'time': time.time() - start_time, **mode_times,
                    'z_labels': Z_LABELS, 'mppi_elite_spread': float(np.mean(mppi_spread_log)) if mppi_spread_log else ''}
        for i, (_, label, _) in enumerate(benchmarks):
            eval_row[f'{label}_corr_lstsq'] = float(np.mean(corr_ls_log[i])) if corr_ls_log.get(i) else ''
        def fill(prefix, means, stds):
            eval_row[f'{prefix}mean_over_tasks'] = float(means.mean())
            for i, (_, label, _) in enumerate(benchmarks):
                eval_row[f'{prefix}{label}_mean'] = float(means[i])
                eval_row[f'{prefix}{label}_std']  = float(stds[i])
        first = EVAL_MODES[0]
        if first in results:                       # MPPI columns stay empty on a policy-only eval
            fill('', results[first][0], results[first][1])
        any_mode = first if first in results else next(iter(results))
        for i, (_, label, _) in enumerate(benchmarks):
            eval_row[f'{label}_corr'] = float(results[any_mode][2][i])
        if EVAL_MODE == 'both' and 'policy' in results:
            fill('policy_', results['policy'][0], results['policy'][1])
        eval_logger.log(eval_row)
        return results

    add_text_to_file(f'{LOGS_FOLDER}/info.txt', run_info(globals(), z_labels=Z_LABELS))      # seed, job, git, command, protocol, data, versions, start time

    if EVAL_CKPT:
        with open(EVAL_CKPT, 'rb') as f:
            loaded = pickle.load(f)
        keys = ('params', 'b_params', 'bh_params', 'f_params', 'target_params', 'pi_params')
        restored = serialization.from_state_dict({k: getattr(train_state, k) for k in keys}, loaded)
        train_state = train_state.replace(**restored)
        print('loaded', EVAL_CKPT, '| z from', Z_LABELS, 'labelled states')
        if FBGOAL_K or FBGOAL_OFFICIAL:   # goal reaching with z = B(goal state), see FBGOAL_K / FBGOAL_OFFICIAL above
            from utils.ogbench_tasks import (sample_goal_pairs, describe_goal_pairs, write_goal_pairs, reset_to_goal_pair,
                                             default_goal_tol, goal_coords)
            g_coords = goal_coords(ENV_NAME)
            tol = default_goal_tol(ENV_NAME) if GOAL_TOL is None else GOAL_TOL
            if FBGOAL_K:
                P = sample_goal_pairs(ENV_NAME, ogb_name, OGB_DIR, FBGOAL_K, GOAL_N, GOAL_SPLIT, GOAL_MIN_DIST, PAIR_SEED)
                print(describe_goal_pairs(P))
                write_goal_pairs(f'{LOGS_FOLDER}/goal_pairs.csv', P)
                n_env, budget, tag = GOAL_N, GOAL_BUDGET, f'goal_k{FBGOAL_K}'
                labels = [tag] * n_env

                def reset_all(envs):    # (states, goal observations, reset error); the pairs are fixed
                    states = [reset_to_goal_pair(e, ENV_NAME, P['start_qpos'][i], P['start_qvel'][i], P['goal_coord'][i])
                              for i, e in enumerate(envs)]
                    return states, P['goal_obs'], np.abs(np.stack(states) - P['start_obs']).max(axis=-1)
            else:
                # the official tasks: start of task t, goal = OGBench's goal observation (fresh start and goal noise at every reset)
                n_env, budget, tag = 5 * GOAL_N_TASK, int(kwargs.get('goal_budget', EVAL_STEPS)), 'official'
                labels = [f'goal_{t}' for t in range(1, 6) for _ in range(GOAL_N_TASK)]

                def reset_all(envs):
                    out = [e.reset(options=dict(task_id=int(l.split('_')[1]))) for e, l in zip(envs, labels)]
                    return ([np.asarray(o[0]) for o in out], np.stack([np.asarray(o[1]['goal'], dtype=np.float32) for o in out]),
                            np.zeros(len(out)))
            enc, b_par = train_state.params['encoder'], train_state.b_params
            cov_obs = replay_buffer.sample(next_eval_key(), Z_COV_STATES)[0]          # E[B B^T] needs no label

            def goal_task_vectors(goal_obs):   # single labelled state with reward 1: E[B R] = B(goal), z = sqrt(d) normalize(B(goal))
                zs_l, ws_l = [], []
                for i in range(len(goal_obs)):
                    z_i, w_i, _ = agent.infer_zw(enc, b_par, jnp.asarray(goal_obs[i:i + 1]), jnp.ones((1,)), Z_RIDGE, cov_obs)
                    zs_l.append(z_i); ws_l.append(w_i)
                zs_ = jnp.stack(zs_l)
                return zs_, (jnp.stack(ws_l) if MPPI_REWARD == 'lstsq' else zs_)

            while len(eval_envs) < n_env:
                eval_envs.append(make_eval_env())
            envs = eval_envs[:n_env]
            gl_fields = ['mode', 'task', 'goal_k', 'n', 'goal_budget', 'goal_tol', 'horizon', 'success', 'success_std', 'gdist_start',
                         'gdist_end', 'gdist_min', 'reset_err', 'time']
            gl = CSVLogger(f"{LOGS_FOLDER}/fbgoal_log.csv", gl_fields)
            ge = CSVLogger(f"{LOGS_FOLDER}/fbgoal_episodes.csv", ['mode', 'task', 'episode', 'success', 'gdist_start', 'gdist_end',
                                                                  'gdist_min', 'reset_err'])
            for mode in EVAL_MODES:
                t0 = time.time()
                states, goal_obs, reset_err = reset_all(envs)
                goal_coord = goal_obs[:, g_coords]
                zs_g, ws_g = goal_task_vectors(goal_obs)
                prev_mean = jnp.zeros((n_env, tdmpc_config.horizon, ACTION_DIM))
                key, coords = next_eval_key(), []
                for step in tqdm(range(budget), leave=False):   # states before each action, as in wm_goal.py
                    obs = np.stack(states).astype(np.float32)
                    coords.append(obs[:, g_coords])
                    if mode == 'policy':
                        actions = agent.act_policy_batch(train_state, jnp.asarray(obs), zs_g)
                    else:
                        key, sub = jax.random.split(key)
                        actions, prev_mean, _ = agent.plan_batch(train_state, jnp.asarray(obs), zs_g, ws_g, prev_mean,
                                                                 jnp.asarray(step > 0), sub)
                    states = [env_step(e, a) for e, a in zip(envs, np.asarray(actions))]
                gdist = np.linalg.norm(np.stack(coords) - goal_coord[None], axis=-1).T                  # (n_env, T)
                succ = (gdist <= tol).any(axis=-1).astype(np.float64)
                groups = [(tag, np.arange(n_env))] + ([(l, np.array([i for i, x in enumerate(labels) if x == l]))
                                                       for l in sorted(set(labels))] if FBGOAL_OFFICIAL else [])
                for name, ix in groups:   # one row for all episodes, plus one per official task
                    gl.log({'mode': mode, 'task': name, 'goal_k': FBGOAL_K, 'n': len(ix), 'goal_budget': budget, 'goal_tol': tol,
                            'horizon': tdmpc_config.horizon, 'success': float(succ[ix].mean()),
                            'success_std': float(succ[ix].std(ddof=1)) if len(ix) > 1 else 0.0,
                            'gdist_start': float(gdist[ix, 0].mean()), 'gdist_end': float(gdist[ix, -1].mean()),
                            'gdist_min': float(gdist[ix].min(axis=-1).mean()), 'reset_err': float(reset_err[ix].mean()),
                            'time': time.time() - t0})
                    print(f'fbgoal {name} [{MODE_NAMES[mode]}]: success {succ[ix].mean():.2f} ({int(succ[ix].sum())}/{len(ix)}) | goal '
                          f'distance start {gdist[ix, 0].mean():.3f} min {gdist[ix].min(axis=-1).mean():.3f} end '
                          f'{gdist[ix, -1].mean():.3f} | reset error {reset_err[ix].mean():.2e}')
                for i in range(n_env):
                    ge.log({'mode': mode, 'task': labels[i], 'episode': i, 'success': float(succ[i]), 'gdist_start': float(gdist[i, 0]),
                            'gdist_end': float(gdist[i, -1]), 'gdist_min': float(gdist[i].min()), 'reset_err': float(reset_err[i])})
            print('wrote', f"{LOGS_FOLDER}/fbgoal_log.csv")
            mark_finished(f'{LOGS_FOLDER}/info.txt', 0)
            sys.exit(0)
        if DIAG_F:
            # F and B on the imagined latents h_hat_t (rolled out from phi(s_0) with the dataset actions) vs on phi(s_t), t = 1..H,
            # for each task's inferred z: corr and relative RMSE of F^T z (with pi's action on each side), cos(g(h_hat_t), B(s_t)).
            diag_logger = CSVLogger(f"{LOGS_FOLDER}/diag_f.csv",
                                    ["task", "t", "f_corr", "f_rel_rmse", "f_bias", "b_cos", "r_corr", "f_real_mean", "f_imag_mean"])
            ts_ = train_state
            obs_d, act_d = replay_buffer.sample_sequences(next_eval_key(), 10_000, tdmpc_config.horizon)
            enc = ts_.params['encoder']
            h_real = agent.encode(enc, obs_d)                                                 # (H+1,N,L)
            h_im = [h_real[0]]
            for t in range(tdmpc_config.horizon):
                h_im.append(agent.next(ts_.params, h_im[-1], act_d[t]))
            for b_i, (reward_fn, label, param) in enumerate(benchmarks):
                lab_obs, _, _, lab_aux = replay_buffer.sample(next_eval_key(), Z_LABELS)
                z_t, _ = agent.infer_z(enc, ts_.b_params, lab_obs, reward_fn(lab_aux, param))
                zN = jnp.broadcast_to(z_t, (h_real.shape[1], z_t.shape[0]))
                for t in range(1, tdmpc_config.horizon + 1):
                    def fq(h):
                        a = agent.pi(ts_.pi_params, h, zN, None, 0.0)
                        F1, F2 = agent.F(ts_.f_params, h, a, zN)
                        return np.asarray(jnp.minimum(agent.score(F1, zN), agent.score(F2, zN))[:, 0])
                    qr, qi = fq(h_real[t]), fq(h_im[t])
                    br = np.asarray(agent.reward_feature(ts_.b_params, ts_.bh_params, h_real[t]))
                    bi = np.asarray(agent.reward_feature(ts_.b_params, ts_.bh_params, h_im[t]))
                    rr, ri = br @ np.asarray(z_t), bi @ np.asarray(z_t)
                    cos = np.sum(br * bi, -1) / (np.linalg.norm(br, axis=-1) * np.linalg.norm(bi, axis=-1) + 1e-8)
                    diag_logger.log({'task': label, 't': t, 'f_corr': float(np.corrcoef(qr, qi)[0, 1]),
                                     'f_rel_rmse': float(np.sqrt(np.mean((qi - qr) ** 2)) / (qr.std() + 1e-8)),
                                     'f_bias': float(np.mean(qi - qr)), 'b_cos': float(cos.mean()),
                                     'r_corr': float(np.corrcoef(rr, ri)[0, 1]), 'f_real_mean': float(qr.mean()),
                                     'f_imag_mean': float(qi.mean())})
            print('wrote', f"{LOGS_FOLDER}/diag_f.csv")
        if DIAG_PROFILE:
            enc_p, b_par_p = train_state.params['encoder'], train_state.b_params
            n_ep, seg = DIAG_WM_ENVS, 200
            cov_obs_p = replay_buffer.sample(next_eval_key(), Z_COV_STATES)[0] if Z_COV == 'dataset' else None
            zs_l, ws_l = [], []
            for reward_fn, label, param in benchmarks:
                for _ in range(n_ep):
                    lab_obs, _, _, lab_aux = replay_buffer.sample(next_eval_key(), Z_LABELS)
                    z_t, _ = agent.infer_z(enc_p, b_par_p, lab_obs, reward_fn(lab_aux, param))
                    _, w_t, _ = agent.infer_zw(enc_p, b_par_p, lab_obs, reward_fn(lab_aux, param), Z_RIDGE, cov_obs_p)
                    zs_l.append(z_t); ws_l.append(w_t if MPPI_REWARD == 'lstsq' else z_t)
            zs_p, ws_p = jnp.stack(zs_l), jnp.stack(ws_l)
            prof_log = CSVLogger(f"{LOGS_FOLDER}/diag_profile.csv",
                                 ["source", "task", "seg", "reward", "aux0", "aux1", "pitch_abs", "ret"])
            for src in ('policy', 'mppi'):
                obs_p, aux_p, _ = rollout_batch(train_state, zs_p, next_eval_key(), src, ws_p if MPPI_REWARD == 'lstsq' else None)
                for b_i, (reward_fn, label, param) in enumerate(benchmarks):
                    sl = slice(b_i * n_ep, (b_i + 1) * n_ep)
                    r = np.asarray(reward_fn(jnp.asarray(aux_p[sl], dtype=jnp.float32), param))             # (n_ep, T)
                    ret = float(r.sum(1).mean())
                    for k in range(EVAL_STEPS // seg):
                        ss = slice(k * seg, (k + 1) * seg)
                        prof_log.log({'source': src, 'task': label, 'seg': k, 'reward': float(r[:, ss].mean()),
                                      'aux0': float(aux_p[sl][:, ss, 0].mean()), 'aux1': float(aux_p[sl][:, ss, 1].mean()),
                                      'pitch_abs': float(np.abs(obs_p[sl][:, ss, 1]).mean()) if ENV_NAME == 'cheetah' else '',
                                      'ret': ret})
                    print('diag_profile', src, label, 'return', round(ret, 1))
            print('wrote', f"{LOGS_FOLDER}/diag_profile.csv")
            mark_finished(f'{LOGS_FOLDER}/info.txt', 0)
            sys.exit(0)
        if DIAG_WM:
            # Open-loop accuracy of the world model far past its training horizon, against the real simulator. Two sources:
            #   'dataset': random dataset sub-trajectories with the dataset actions (latent metrics are task-independent and
            #              repeated on each task's rows; the reward metrics are per task);
            #   'policy' : real env episodes of the deterministic policy pi(.|phi(s), z_task), cut into windows every 25 steps.
            # From h_hat_0 = phi(s_0), h_hat_{t+1} = f(h_hat_t, a_t) with the real actions; at each t it is compared with phi(s_t):
            #   lat_cos, lat_nmse : cos(h_hat_t, phi(s_t)) and ||h_hat_t - phi(s_t)||^2 / E||phi(s) - E phi||^2 (1 = as bad as the mean latent)
            #   copy_nmse         : the same error for the persistence guess h_hat_t = phi(s_0) (the latent does not move)
            #   b_cos             : cos(g(h_hat_t), B(s_t))
            #   r_corr_imag/real  : corr over windows of the lstsq planner reward g(.)^T w with the true reward R(s_t), from h_hat_t / phi(s_t)
            #   r_nrmse           : rmse(g(h_hat_t)^T w - g(phi(s_t))^T w) / std over the dataset of g(phi(s))^T w
            #   obs_nmse(_real)   : MLP probe phi -> standardised (obs, aux), fitted on dataset latents, decoded from h_hat_t (from phi(s_t):
            #                       the probe's own floor); mean over obs dims of the squared error (1 = as bad as the dataset mean)
            #   <aux>_mae(_real)  : the probe's aux outputs (velocity, height, upright, ...), mean absolute error in physical units
            T = DIAG_WM_H
            enc, b_par = train_state.params['encoder'], train_state.b_params
            AUX_DIM = replay_buffer.aux.shape[-1]
            aux_names = {'walker': ['xvel', 'upright', 'height', 'angmom'], 'cheetah': ['xvel', 'angmom'],
                         'quadruped': ['vx', 'vy', 'vz', 'upright', 'com_h']}.get(ENV_NAME, [f'aux{i}' for i in range(AUX_DIM)])
            assert len(aux_names) == AUX_DIM, (aux_names, AUX_DIM)

            # dataset statistics: latent spread, probe target scaling, per-task scale of the planner reward
            s_obs, _, _, s_aux = replay_buffer.sample(next_eval_key(), 100_000)
            h_s = agent.encode(enc, s_obs)
            lat_var = float(jnp.mean(jnp.sum((h_s - h_s.mean(0)) ** 2, -1)))
            y_s = jnp.concatenate([s_obs, s_aux], -1)
            y_mu, y_sd = y_s.mean(0), y_s.std(0) + 1e-6
            Bs = agent.b_from_latent(b_par, h_s)

            # MLP probe phi(s) -> (obs, aux), 20k Adam steps on fresh dataset batches
            probe = MLP(hidden=(256, 256), out_dim=STATE_DIM + AUX_DIM)
            popt = optax.adam(1e-3)

            @jax.jit
            def train_probe(p, o, keys, rb):
                def step(carry, key):
                    p, o = carry
                    ob, _, _, ax = rb.sample(key, 1024)
                    h, tgt = agent.encode(enc, ob), (jnp.concatenate([ob, ax], -1) - y_mu) / y_sd
                    loss, g = jax.value_and_grad(lambda p: jnp.mean((probe.apply({'params': p}, h) - tgt) ** 2))(p)
                    up, o = popt.update(g, o, p)
                    return (optax.apply_updates(p, up), o), loss
                return jax.lax.scan(step, (p, o), keys)

            pp = probe.init(next_eval_key(), jnp.zeros((1, agent.latent_dim)))['params']
            (pp, _), probe_losses = train_probe(pp, popt.init(pp), jax.random.split(next_eval_key(), 20_000), replay_buffer)
            probe_loss = float(probe_losses[-1000:].mean())
            print('probe loss (standardised MSE, last 1k steps):', probe_loss)

            @jax.jit
            def wm_rollout(h0, acts):                                   # (N,L), (T,N,A) -> h_hat_1..T (T,N,L)
                def step(h, a):
                    h = agent.next(train_state.params, h, a)
                    return h, h
                return jax.lax.scan(step, h0, acts)[1]

            # one (z, w) per task, as in the eval (N = Z_LABELS labelled states, Cov(B) per z_cov)
            cov_obs = replay_buffer.sample(next_eval_key(), Z_COV_STATES)[0] if Z_COV == 'dataset' else None
            task_zw = []
            for reward_fn, label, param in benchmarks:
                lab_obs, _, _, lab_aux = replay_buffer.sample(next_eval_key(), Z_LABELS)
                r_lab = reward_fn(lab_aux, param)
                z_t, _ = agent.infer_z(enc, b_par, lab_obs, r_lab)
                _, w_t, _ = agent.infer_zw(enc, b_par, lab_obs, r_lab, Z_RIDGE, cov_obs)
                w_t = w_t if MPPI_REWARD == 'lstsq' else z_t
                task_zw.append((z_t, w_t, float(jnp.std(Bs @ w_t))))

            fields = (['source', 'task', 't', 'n', 'lat_cos', 'lat_nmse', 'copy_nmse', 'b_cos', 'r_corr_imag', 'r_corr_real', 'r_nrmse',
                       'obs_nmse', 'obs_nmse_real'] + [f'{a}_mae{s}' for a in aux_names for s in ('', '_real')]
                      + ['ep_return', 'probe_loss', 'horizon_train'])
            wm_logger = CSVLogger(f"{LOGS_FOLDER}/diag_wm.csv", fields)

            def corr_t(x, y):                                           # (T,N), (T,N) -> (T,) corr over N at each t
                xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
                return (xc * yc).mean(1) / (xc.std(1) * yc.std(1) + 1e-8)

            def cos_t(x, y):
                return (jnp.sum(x * y, -1) / (jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1) + 1e-8)).mean(1)

            def wm_metrics(source, obs_seq, act_seq, aux_seq, task_ids, ep_returns=None):
                """obs_seq (T+1,N,D), act_seq (T,N,A), aux_seq (T+1,N,X): one block of rows per task in task_ids."""
                obs_seq, act_seq, aux_seq = (jnp.asarray(x, dtype=jnp.float32) for x in (obs_seq, act_seq, aux_seq))
                h_real = agent.encode(enc, obs_seq)
                h_im, h_re = wm_rollout(h_real[0], act_seq), h_real[1:]
                lat_cos = cos_t(h_im, h_re)
                lat_nmse = jnp.mean(jnp.sum((h_im - h_re) ** 2, -1), 1) / lat_var
                copy_nmse = jnp.mean(jnp.sum((h_real[:1] - h_re) ** 2, -1), 1) / lat_var
                B_im, B_re = agent.b_from_latent(b_par, h_im), agent.b_from_latent(b_par, h_re)
                b_cos = cos_t(B_im, B_re)
                y_true = (jnp.concatenate([obs_seq[1:], aux_seq[1:]], -1) - y_mu) / y_sd
                y_im, y_re = probe.apply({'params': pp}, h_im), probe.apply({'params': pp}, h_re)
                obs_nmse = jnp.mean((y_im - y_true)[..., :STATE_DIM] ** 2, (1, 2))
                obs_nmse_real = jnp.mean((y_re - y_true)[..., :STATE_DIM] ** 2, (1, 2))
                aux_mae = jnp.mean(jnp.abs(y_im - y_true)[..., STATE_DIM:], 1) * y_sd[STATE_DIM:]       # (T,X), physical units
                aux_mae_real = jnp.mean(jnp.abs(y_re - y_true)[..., STATE_DIM:], 1) * y_sd[STATE_DIM:]
                base = {k: np.asarray(v) for k, v in dict(lat_cos=lat_cos, lat_nmse=lat_nmse, copy_nmse=copy_nmse, b_cos=b_cos,
                                                          obs_nmse=obs_nmse, obs_nmse_real=obs_nmse_real).items()}
                aux_mae, aux_mae_real = np.asarray(aux_mae), np.asarray(aux_mae_real)
                for b_i in task_ids:
                    reward_fn, label, param = benchmarks[b_i]
                    _, w_t, r_sd = task_zw[b_i]
                    r_true = reward_fn(aux_seq[1:], param)
                    r_im, r_re = B_im @ w_t, B_re @ w_t
                    r_ci, r_cr = np.asarray(corr_t(r_im, r_true)), np.asarray(corr_t(r_re, r_true))
                    r_nrmse = np.asarray(jnp.sqrt(jnp.mean((r_im - r_re) ** 2, 1)) / (r_sd + 1e-8))
                    for t in range(T):
                        row = {'source': source, 'task': label, 't': t + 1, 'n': int(obs_seq.shape[1]),
                               **{k: float(v[t]) for k, v in base.items()},
                               'r_corr_imag': float(r_ci[t]), 'r_corr_real': float(r_cr[t]), 'r_nrmse': float(r_nrmse[t]),
                               'ep_return': '' if ep_returns is None else float(ep_returns[b_i]),
                               'probe_loss': probe_loss, 'horizon_train': tdmpc_config.horizon}
                        for j, a in enumerate(aux_names):
                            row[f'{a}_mae'], row[f'{a}_mae_real'] = float(aux_mae[t, j]), float(aux_mae_real[t, j])
                        wm_logger.log(row)

            # (1) dataset sub-trajectories of length T (dataset actions)
            n_seq, tl = 4096, TRAJECTORY_LEN
            rng = np.random.default_rng(SEED + 20_000)
            starts = rng.integers(0, NUM_TRAJECTORIES, n_seq) * tl + rng.integers(0, tl - T, n_seq)
            idx = starts[None, :] + np.arange(T + 1)[:, None]
            wm_metrics('dataset', replay_buffer.obs[idx], replay_buffer.actions[idx[:-1]], replay_buffer.aux[idx],
                       range(N_BENCH))
            print('diag_wm: dataset done')

            # (2) real episodes of the policy, DIAG_WM_ENVS per task, all tasks in lock-step
            zs = jnp.concatenate([jnp.broadcast_to(task_zw[b][0], (DIAG_WM_ENVS, task_zw[b][0].shape[0])) for b in range(N_BENCH)])
            n = zs.shape[0]
            while len(eval_envs) < n:
                eval_envs.append(make_eval_env())
            envs = eval_envs[:n]
            states = [env_reset(e) for e in envs]
            obs_l, act_l, aux_l, phys_tr_l = [], [], [], []
            for step in tqdm(range(EVAL_STEPS), leave=False):
                obs = np.stack(states).astype(np.float32)
                aux_l.append(np.stack([env_aux(e, s) for e, s in zip(envs, states)]))
                if DIAG_WM_CF:
                    phys_tr_l.append(np.stack([env_physics(e) for e in envs]))
                actions = np.asarray(agent.act_policy_batch(train_state, jnp.asarray(obs), zs))
                obs_l.append(obs); act_l.append(actions)
                states = [env_step(e, a) for e, a in zip(envs, actions)]
            obs_tr, act_tr, aux_tr = np.stack(obs_l), np.stack(act_l), np.stack(aux_l)           # (EVAL_STEPS, n, .)
            win = np.arange(0, EVAL_STEPS - T, 25)
            for b_i, (reward_fn, label, param) in enumerate(benchmarks):
                sl = slice(b_i * DIAG_WM_ENVS, (b_i + 1) * DIAG_WM_ENVS)
                ret = np.asarray(reward_fn(jnp.asarray(aux_tr[:, sl], dtype=jnp.float32), param)).sum(0).mean()
                print(f'diag_wm: policy {label}: return {ret:.1f}')
                cut = lambda x, L: np.concatenate([x[s:s + L, sl] for s in win], axis=1)          # (L, windows * envs, .)
                wm_metrics('policy', cut(obs_tr, T + 1), cut(act_tr, T), cut(aux_tr, T + 1), [b_i], {b_i: ret})
            # (3) value check on the same real policy episodes: is F^T z the discounted return that is actually obtained?
            # Q_t = min_i F_i(phi(s_t), a_t, z) . z  (F^T z is the Q of the projected reward r_proj = B(s)^T w, exactly, w from the lstsq).
            # G_proj_t = sum_{k>=1} gamma^(k-1) r_proj(s_{t+k}) (K = 100 steps), G_true_t the same with the task's true reward.
            #   q_corr  = corr(Q, G_proj)  : accuracy of F on the policy's own states (1 = F ranks states correctly)
            #   q_slope = slope of G_proj on Q: 1 = unbiased, < 1 means F overestimates
            #   proj_corr = corr(G_proj, G_true): how much of the true return span(B) keeps
            #   q_corr_true = corr(Q, G_true): what the planner / policy actually sees
            qlog = CSVLogger(f"{LOGS_FOLDER}/diag_q.csv", ["task", "n", "q_corr", "q_slope", "q_mean", "g_proj_mean", "proj_corr",
                                                           "q_corr_true", "ep_return"])
            K, gam, t_max = 100, tdmpc_config.discount, EVAL_STEPS - 100
            for b_i, (reward_fn, label, param) in enumerate(benchmarks):
                sl = slice(b_i * DIAG_WM_ENVS, (b_i + 1) * DIAG_WM_ENVS)
                z_t, w_t, _ = task_zw[b_i]
                o = jnp.asarray(obs_tr[:, sl], dtype=jnp.float32).reshape(-1, STATE_DIM)
                a_ = jnp.asarray(act_tr[:, sl], dtype=jnp.float32).reshape(-1, ACTION_DIM)
                h_ = agent.encode(enc, o)
                zz = jnp.broadcast_to(z_t, (h_.shape[0], z_t.shape[0]))
                F1, F2 = agent.F(train_state.f_params, h_, a_, zz)
                Q = np.asarray(jnp.minimum((F1 * zz).sum(-1), (F2 * zz).sum(-1))).reshape(EVAL_STEPS, -1)
                r_proj = np.asarray(agent.b_from_latent(b_par, h_) @ w_t).reshape(EVAL_STEPS, -1)
                r_true = np.asarray(reward_fn(jnp.asarray(aux_tr[:, sl], dtype=jnp.float32), param))
                def disc(r):
                    G = np.zeros((t_max, r.shape[1]))
                    for k in range(1, K + 1):
                        G += gam ** (k - 1) * r[k:k + t_max]
                    return G
                Gp, Gt, Qm = disc(r_proj), disc(r_true), Q[:t_max]
                cc = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
                slope = float(np.polyfit(Qm.ravel(), Gp.ravel(), 1)[0])
                qlog.log({'task': label, 'n': int(Qm.size), 'q_corr': cc(Qm, Gp), 'q_slope': slope, 'q_mean': float(Qm.mean()),
                          'g_proj_mean': float(Gp.mean()), 'proj_corr': cc(Gp, Gt), 'q_corr_true': cc(Qm, Gt),
                          'ep_return': float(r_true.sum(0).mean())})
            print('wrote', f"{LOGS_FOLDER}/diag_q.csv")

            # (4) counterfactual action ranking: does Q(s, a', z) rank the first action correctly? From recorded states of the policy
            # episodes (10 per task x 8 times), the simulator is reset to the state and NC first actions are tried: the policy's
            # own, it plus noise (0.3, 0.6), a uniform random one; then the deterministic policy for K steps. G = real discounted
            # return of the TRUE task reward after the first action (same convention as Q = F^T z).
            #   spearman : mean over states of the rank correlation of Q and G across the NC actions
            #   top1     : fraction of states where argmax Q is also argmax G (chance 1/NC)
            #   gain_q   : mean [G(argmax Q action) - G(policy action)]; gain_best : mean [max G - G(policy action)] (the ceiling)
            if DIAG_WM_CF:
                phys_tr = np.stack(phys_tr_l)                                                    # (EVAL_STEPS, n, P)
                NC, K_cf, CHUNK = 4, 60, 25
                cf_times = list(range(100, 900, 100))
                cf_log = CSVLogger(f"{LOGS_FOLDER}/diag_cf.csv", ["task", "n_states", "spearman", "top1", "gain_q", "gain_best",
                                                                   "g_policy_mean", "q_policy_mean", "spearman_k3", "spearman_k10",
                                                                   "spearman_k30", "top1_k3", "top1_k10", "q_range", "g_range_k3", "g_range_k60"])
                cf_rng = np.random.default_rng(SEED + 30_000)
                while len(eval_envs) < CHUNK * NC:
                    eval_envs.append(make_eval_env())
                rank = lambda x: np.argsort(np.argsort(x))
                for b_i, (reward_fn, label, param) in enumerate(benchmarks):
                    z_t = task_zw[b_i][0]
                    pts = [(b_i * DIAG_WM_ENVS + j, t) for j in range(DIAG_WM_ENVS) for t in cf_times]
                    sp, hit, g_q, g_best, g_pi, q_pi = [], [], [], [], [], []
                    sp_k, hit_k, q_rng, g_rng3, g_rng60 = {3: [], 10: [], 30: []}, {3: [], 10: []}, [], [], []
                    for c0 in range(0, len(pts), CHUNK):
                        chunk = pts[c0:c0 + CHUNK]
                        n_c = len(chunk) * NC
                        cenvs = eval_envs[:n_c]
                        o0 = np.stack([obs_tr[t, i] for i, t in chunk]).astype(np.float32)                    # (S, D)
                        zc = jnp.broadcast_to(z_t, (len(chunk), z_t.shape[0]))
                        a_pi = np.asarray(agent.act_policy_batch(train_state, jnp.asarray(o0), zc))               # (S, A)
                        cand = np.stack([a_pi,
                                         np.clip(a_pi + 0.3 * cf_rng.standard_normal(a_pi.shape), -1, 1),
                                         np.clip(a_pi + 0.6 * cf_rng.standard_normal(a_pi.shape), -1, 1),
                                         cf_rng.uniform(-1, 1, a_pi.shape)], 1).astype(np.float32)           # (S, NC, A)
                        h0 = agent.encode(enc, jnp.asarray(o0))
                        Qc = np.zeros((len(chunk), NC))
                        for c in range(NC):
                            F1, F2 = agent.F(train_state.f_params, h0, jnp.asarray(cand[:, c]), zc)
                            Qc[:, c] = np.asarray(jnp.minimum((F1 * zc).sum(-1), (F2 * zc).sum(-1)))
                        for k, e in enumerate(cenvs):
                            e.reset()
                            e.physics.set_state(phys_tr[chunk[k // NC][1], chunk[k // NC][0]])
                            e.physics.forward()
                        zs_c = jnp.repeat(zc, NC, axis=0)
                        acts = cand.reshape(-1, ACTION_DIM)
                        G = np.zeros(n_c)
                        G_at = {}
                        for k_step in range(K_cf):
                            sts = [env_step(e, a) for e, a in zip(cenvs, acts)]
                            aux_k = np.stack([env_aux(e, s) for e, s in zip(cenvs, sts)])
                            G += gam ** k_step * np.asarray(reward_fn(jnp.asarray(aux_k, dtype=jnp.float32), param))
                            if k_step + 1 in (3, 10, 30):
                                G_at[k_step + 1] = G.copy().reshape(len(chunk), NC)
                            acts = np.asarray(agent.act_policy_batch(train_state, jnp.asarray(np.stack(sts).astype(np.float32)), zs_c))
                        G = G.reshape(len(chunk), NC)
                        for s_i in range(len(chunk)):
                            sp.append(float(np.corrcoef(rank(Qc[s_i]), rank(G[s_i]))[0, 1]) if G[s_i].std() > 1e-6 else np.nan)
                            hit.append(float(np.argmax(Qc[s_i]) == np.argmax(G[s_i])))
                            g_q.append(G[s_i, np.argmax(Qc[s_i])] - G[s_i, 0]); g_best.append(G[s_i].max() - G[s_i, 0])
                            g_pi.append(G[s_i, 0]); q_pi.append(Qc[s_i, 0])
                            q_rng.append(Qc[s_i].max() - Qc[s_i].min()); g_rng60.append(G[s_i].max() - G[s_i].min())
                            g_rng3.append(G_at[3][s_i].max() - G_at[3][s_i].min())
                            for kk in (3, 10, 30):
                                g_k = G_at[kk][s_i]
                                sp_k[kk].append(float(np.corrcoef(rank(Qc[s_i]), rank(g_k))[0, 1]) if g_k.std() > 1e-9 else np.nan)
                                if kk in hit_k:
                                    hit_k[kk].append(float(np.argmax(Qc[s_i]) == np.argmax(g_k)))
                    cf_log.log({'task': label, 'n_states': len(pts), 'spearman': float(np.nanmean(sp)), 'top1': float(np.mean(hit)),
                                'gain_q': float(np.mean(g_q)), 'gain_best': float(np.mean(g_best)),
                                'g_policy_mean': float(np.mean(g_pi)), 'q_policy_mean': float(np.mean(q_pi)),
                                'spearman_k3': float(np.nanmean(sp_k[3])), 'spearman_k10': float(np.nanmean(sp_k[10])),
                                'spearman_k30': float(np.nanmean(sp_k[30])), 'top1_k3': float(np.mean(hit_k[3])),
                                'top1_k10': float(np.mean(hit_k[10])), 'q_range': float(np.mean(q_rng)),
                                'g_range_k3': float(np.mean(g_rng3)), 'g_range_k60': float(np.mean(g_rng60))})
                    print(f'diag_cf {label}: spearman {np.nanmean(sp):.2f} top1 {np.mean(hit):.2f} gain_q {np.mean(g_q):.2f} best {np.mean(g_best):.2f}')
                print('wrote', f"{LOGS_FOLDER}/diag_cf.csv")
            print('wrote', f"{LOGS_FOLDER}/diag_wm.csv")
            mark_finished(f'{LOGS_FOLDER}/info.txt', 0)
            sys.exit(0)
        evaluate(train_state, 0, False, NUM_EVALS_FINAL, 'zlabels')
        mark_finished(f'{LOGS_FOLDER}/info.txt', 0)
        sys.exit(0)

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
            modes = EVAL_MODES if ((steps + 1) % MPPI_EVAL_EACH == 0 or EVAL_MODE != 'both') else ['policy']
            results = evaluate(train_state, num_updates, make_plot, NUM_EVALS, 'eval', modes)
            n_evals += 1

            num_cols = 4
            num_rows = N_BENCH // num_cols + int(N_BENCH % num_cols != 0)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
            axs = np.atleast_1d(axs).flatten()
            for mode in results:
                rewards_logs[mode] = np.concatenate((rewards_logs[mode], results[mode][0].reshape(-1, 1)), axis=-1)
                for i, (_, label, _) in enumerate(benchmarks):
                    axs[i].plot(rewards_logs[mode][i], label=MODE_NAMES[mode])
                    axs[i].set_title(label)
            if len(EVAL_MODES) > 1:
                axs[0].legend()
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/rewards.png")
            plt.close()

            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/fbwm_{VARIANT}_latest.pickle")

        if (steps + 1) % SAVE_EACH_STEPS == 0:
            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/fbwm_{VARIANT}_step:{num_updates}.pickle")

    # ── Final evaluation: NUM_EVALS_FINAL episodes per task on the last checkpoint (the reported number of the run) ──
    save_agent(train_state, f"{MODEL_SAVE_FOLDER}/fbwm_{VARIANT}_final.pickle")
    evaluate(train_state, NUM_OUTER_STEPS * SCAN_LENGTH, True, NUM_EVALS_FINAL, 'final')

    mark_finished(f'{LOGS_FOLDER}/info.txt', NUM_OUTER_STEPS * SCAN_LENGTH)
