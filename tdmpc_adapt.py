"""Reward-free JEPA world model + per-task TD-MPC heads: the C4 adaptation baseline (2026-11-14).

    mode=pretrain  env_name=walker                       # once per env: encoder + dynamics, no reward / value / FB terms
    mode=adapt     env_name=walker task=walk wm_ckpt=models/<run>/wm_final.pickle n_labels=10000 adapt_steps=100000

pretrain: TD-MPC v1 encoder and residual dynamics (same networks and sphere-projected latent as tdmpc_jax.py), trained with
the latent consistency loss (EMA target encoder, H-step rollout) plus VICReg variance / covariance terms on the encoded latents
(anti-collapse). Saves wm_final.pickle. encoder=phimap (2026-11-22): the encoder is fbwm.py's gphi trunk PhiMap (hidden width enc_dim, default
256) instead of the TD-MPC MLP, so the JEPA WM and FBWM gphi (phi_hidden_dim=256) share the encoder; run folder JEPAWM-phimap-<env>.
reg=sigreg target=online (2026-11-22): LeWM's loss on this backbone, SIGReg instead of VICReg and the online encoder as the consistency
target (no EMA, no stop-gradient); run folder JEPAWM-phimap-sigreg-online-<env>.
adapt: loads and freezes the encoder and dynamics. q_data=labeled: Q / policy from the N labelled (s, a, r, s') only, true r,
one-step TD (no dataset access at adaptation time; run folder TDMPC-adaptL-...). Otherwise: (1) fits a reward head R(h, a) on n_labels labelled transitions, (2) trains
Q1, Q2 and the policy prior with TD-MPC's value / policy losses for adapt_steps updates, with rewards from that head,
(3) evaluates MPPI (same planner as tdmpc_jax.py and fbwm.py) and the policy. An eval right after the reward fit (adapt step 0)
is pure H-step MPC with no value (Q starts at 0). Eval rows log the adaptation wall-clock.
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
    MODE = str(kwargs.get('mode', ''))
    assert MODE in ['pretrain', 'adapt'], "mode=pretrain|adapt is required"
    ENV_NAME = kwargs['env_name']
    if MODE == 'adapt':
        assert 'task' in kwargs and 'wm_ckpt' in kwargs, "mode=adapt needs task= and wm_ckpt="
    TASK     = str(kwargs['task']) if MODE == 'adapt' else None
    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze', 'antmaze_medium', 'antmaze_large', 'cube_single']
    # Behaviour-cloning regulariser of the policy (OGBench antmaze / cube, as in offline_qd_jax.py): the policy loss is normalised by its own
    # magnitude and bc_coef * MSE(policy action, dataset action) is added. bc_coef=0 (default): unchanged.
    BC_COEF = float(kwargs['bc_coef']) if 'bc_coef' in kwargs else 0.0
    # encoder=mlp|phimap (2026-11-22, same-encoder comparison with FBWM gphi): mlp (default) = the TD-MPC encoder Linear-ELU-Linear-ELU-Linear
    # (orthogonal init); phimap = fbwm.py's gphi trunk PhiMap, Linear -> LayerNorm+Tanh -> Linear -> ReLU -> Linear (torch init), copied
    # verbatim so the parameter names match. Hidden width enc_dim (default 256) for both. mode=adapt reads the encoder type and width from wm_ckpt.
    ENCODER = str(kwargs.get('encoder', 'mlp'))
    assert ENCODER in ('mlp', 'phimap'), "encoder=mlp|phimap"
    ENC_DIM = int(kwargs['enc_dim']) if 'enc_dim' in kwargs else 256
    ENC_LAYERS = int(kwargs['enc_layers']) if 'enc_layers' in kwargs else 2   # PhiMap depth (Linear layers = enc_layers + 1); default 2 = unchanged. mode=adapt reads it from wm_ckpt
    # reg=vicreg|sigreg, target=ema|online (2026-11-22, "LeWM's loss on our backbone", mode=pretrain only):
    #   reg=vicreg (default): VICReg variance + covariance on phi(s_0) (var_coef, cov_coef), as before.
    #   reg=sigreg: LeWM's SIGReg (port of wm_baselines.py / LeWM's module.py) instead of VICReg, sigreg_coef (default 0.09, LeWM's config)
    #     x num_proj random unit directions (default 1024, redrawn every update), on the sphere-projected online latents of every frame s_0..s_H.
    #   target=ema (default): consistency target = EMA encoder phi_bar(s_t), no gradient.
    #   target=online: consistency target = the online phi(s_t), gradient through it (LeWM: no EMA, no stop-gradient).
    #   The rollout loss itself is unchanged (H-step latent rollout, rho^t weights), LeWM predicts teacher-forced one step at a time.
    REG    = str(kwargs.get('reg', 'vicreg'))
    TARGET = str(kwargs.get('target', 'ema'))
    assert REG in ('vicreg', 'sigreg') and TARGET in ('ema', 'online'), "reg=vicreg|sigreg target=ema|online"
    assert MODE == 'pretrain' or ('reg' not in kwargs and 'target' not in kwargs), "reg= / target= are mode=pretrain options"
    SIGREG_COEF = float(kwargs['sigreg_coef']) if 'sigreg_coef' in kwargs else 0.09
    NUM_PROJ    = int(kwargs['num_proj']) if 'num_proj' in kwargs else 1024
    if MODE == 'adapt':
        with open(kwargs['wm_ckpt'], 'rb') as f:
            _wm_enc = pickle.load(f)['encoder']
        _ckpt_encoder = 'phimap' if 'TorchDense_0' in _wm_enc else 'mlp'
        assert 'encoder' not in kwargs or ENCODER == _ckpt_encoder, f"encoder={ENCODER} but wm_ckpt has a {_ckpt_encoder} encoder"
        ENCODER = _ckpt_encoder
        ENC_DIM = int(_wm_enc['TorchDense_0' if ENCODER == 'phimap' else 'Dense_0']['kernel'].shape[1])
        if ENCODER == 'phimap':
            ENC_LAYERS = len([k for k in _wm_enc if k.startswith('TorchDense_')]) - 1
        del _wm_enc

    def bc_combine(main, bc, coef):
        if coef == 0:
            return main
        return main / (jax.lax.stop_gradient(jnp.abs(main)) + 1e-8) + coef * bc


    # ── Config ────────────────────────────────────────────────────────────────────
    # cheetah | walker | quadruped (dm_control)  ;  antmaze (d4rl antmaze-large-diverse-v2)
    SEED = resolve_seed(kwargs)      # random unless seed= is given; recorded in info.txt

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = (f'JEPAWM{"-phimap" if ENCODER == "phimap" else ""}{"-sigreg" if REG == "sigreg" else ""}{"-online" if TARGET == "online" else ""}-{ENV_NAME}'
                if MODE == 'pretrain' else
                f'TDMPC-adapt{ {"labeled": "L", "labeled_model": "LM"}.get(kwargs.get("q_data"), "")}-{ENV_NAME}-{TASK}')   # adaptL / adaptLM: no dataset access

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

    TRAIN_METRICS = (["total_loss", "consistency_loss", "var_loss", "cov_loss", "grad_norm", "latent_std", "latent_rank", "sigreg_loss"]
                     if MODE == 'pretrain' else ["phase", "reward_loss", "value_loss", "pi_loss", "q_mean", "grad_norm"])
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


    else:  # OGBench: antmaze_medium | antmaze_large | cube_single (utils/ogbench_tasks.py; dataset=, ogbench_dir=)
        from utils.ogbench_tasks import load_ogbench
        _ogb = load_ogbench(ENV_NAME, kwargs)
        STATE_DIM, ACTION_DIM = _ogb['state_dim'], _ogb['action_dim']
        NUM_TRAJECTORIES, TRAJECTORY_LEN = _ogb['num_trajectories'], _ogb['trajectory_len']
        dataset_obs, dataset_actions = _ogb['obs'], _ogb['actions']
        dataset_aux = dataset_obs       # rewards are computed from the full observation
        benchmarks, EVAL_STEPS = _ogb['benchmarks'], _ogb['eval_steps']
        make_eval_env, env_reset, env_step = _ogb['make_eval_env'], _ogb['env_reset'], _ogb['env_step']
        del _ogb

        def env_aux(e, state):
            return state

        def env_physics(e):
            return None

    def reset_for(e, label=None):
        """Env reset for the task `label` (OGBench: goal_k starts at the official start of task k); other envs: plain reset."""
        return env_reset(e, label) if ENV_NAME in ('antmaze_medium', 'antmaze_large', 'cube_single') else env_reset(e)

    # ── Task selection (adapt only) and run settings ──────────────────────────────

    if MODE == 'adapt':
        labels = [b[1] for b in benchmarks]
        assert TASK in labels, f"task must be one of {labels}"
        reward_fn, _, reward_param = benchmarks[labels.index(TASK)]
        print('TASK:', TASK, '| reward param:', reward_param)

    NUM_EVALS = 5             # episodes per periodic eval (tag 'eval'), rolled out in parallel
    NUM_EVALS_FINAL = int(kwargs['num_evals_final']) if 'num_evals_final' in kwargs else 10
    assert NUM_EVALS_FINAL % NUM_EVALS == 0, 'num_evals_final must be a multiple of 5'

    BATCH_SIZE      = int(kwargs['batch_size']) if 'batch_size' in kwargs else 512
    HORIZON         = int(kwargs['horizon']) if 'horizon' in kwargs else 5
    SCAN_LENGTH     = 1_000
    PLOT_EACH_EVALS = 5
    if MODE == 'pretrain':
        NUM_OUTER_STEPS = int(kwargs['num_outer_steps']) if 'num_outer_steps' in kwargs else 500   # 500k updates
    else:
        N_LABELS     = int(kwargs['n_labels']) if 'n_labels' in kwargs else 10_000      # labelled transitions for the reward head
        REWARD_STEPS = int(kwargs['reward_steps']) if 'reward_steps' in kwargs else 10_000  # reward-head updates (multiple of 1000)
        ADAPT_STEPS  = int(kwargs['adapt_steps']) if 'adapt_steps' in kwargs else 100_000   # Q / policy updates (multiple of 1000)
        assert REWARD_STEPS % SCAN_LENGTH == 0 and ADAPT_STEPS % SCAN_LENGTH == 0
        NUM_OUTER_STEPS = ADAPT_STEPS // SCAN_LENGTH            # recorded as `updates` in the protocol line of info.txt
        EVAL_EACH_STEPS = int(kwargs['eval_each_steps']) if 'eval_each_steps' in kwargs else 10   # in chunks of 1000 Q updates
        Q_DATA = str(kwargs['q_data']) if 'q_data' in kwargs else 'dataset'   # dataset: TD on all transitions, rewards from the head
        assert Q_DATA in ['dataset', 'labeled', 'labeled_model'], "q_data=dataset|labeled|labeled_model"   # labeled: one-step TD on the N labelled transitions only
        EVAL_MODE  = str(kwargs['eval_mode']) if 'eval_mode' in kwargs else 'both'
        assert EVAL_MODE in ['mppi', 'policy', 'both']
        EVAL_MODES = ['mppi', 'policy'] if EVAL_MODE == 'both' else [EVAL_MODE]
        MODE_NAMES = {'mppi': 'MPPI', 'policy': 'policy prior'}
        eval_logger = CSVLogger(
            filepath=f"{LOGS_FOLDER}/eval_log.csv",
            fieldnames=["step", "return_mean", "return_std"]
                       + (["policy_return_mean", "policy_return_std"] if EVAL_MODE == 'both' else [])
                       + ["time", "num_evals", "tag", "reward_fit_time", "adapt_time", "reward_corr"],
        )

    # ── Rewards: none for pretrain; for adapt the true reward of the task, of which only N_LABELS transitions are used ──

    if MODE == 'adapt':
        def label_rewards(aux, reward_fn, param, chunk=1_000_000):
            f   = jax.jit(lambda a: reward_fn(a, param))
            out = np.empty(len(aux), dtype=np.float32)
            for s in range(0, len(aux), chunk):
                out[s:s + chunk] = np.asarray(f(jnp.asarray(aux[s:s + chunk], dtype=jnp.float32)))
            return out
        dataset_rewards = label_rewards(dataset_aux, reward_fn, reward_param)
    else:
        dataset_rewards = np.zeros(len(dataset_obs), dtype=np.float32)

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

    class TorchDense(nn.Module):
        """nn.Dense with torch's default nn.Linear init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b (copied from fbwm.py)."""
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
        """FB 'ntanh': LayerNorm followed by Tanh (copied from fbwm.py)."""
        @nn.compact
        def __call__(self, x):
            return jnp.tanh(nn.LayerNorm(epsilon=1e-5)(x))

    class PhiMap(nn.Module):
        """fbwm.py's gphi trunk (encoder=phimap): Linear -> NTanh -> [Linear -> relu] x (num_layers - 1) -> Linear (copied from fbwm.py)."""
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

    # SIGReg (LeWM's module.py; same code as wm_baselines.py)
    _SIG_KNOTS = 17
    _SIG_T     = jnp.linspace(0.0, 3.0, _SIG_KNOTS)
    _SIG_DT    = 3.0 / (_SIG_KNOTS - 1)
    _SIG_PHI   = jnp.exp(-_SIG_T ** 2 / 2.0)
    _SIG_W     = jnp.full((_SIG_KNOTS,), 2 * _SIG_DT).at[0].set(_SIG_DT).at[-1].set(_SIG_DT) * _SIG_PHI

    def sigreg(z, key):
        """z (T, B, D): Epps-Pulley statistic of NUM_PROJ random 1-d projections against N(0, 1), per frame, averaged."""
        A = jax.random.normal(key, (z.shape[-1], NUM_PROJ))
        A = A / jnp.linalg.norm(A, axis=0, keepdims=True)
        x_t = (z @ A)[..., None] * _SIG_T                                            # (T, B, M, K)
        err = (jnp.cos(x_t).mean(-3) - _SIG_PHI) ** 2 + jnp.sin(x_t).mean(-3) ** 2   # (T, M, K)
        return ((err @ _SIG_W) * z.shape[-2]).mean()

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

    # ── Agent: TD-MPC v1 networks; which ones train depends on the phase ──────────

    @dataclasses.dataclass(frozen=True)
    class AdaptConfig:
        lr:                float = 1e-3
        grad_clip_norm:    float = 10.0
        tau:               float = 0.01
        update_freq:       int   = 2
        batch_size:        int   = 512
        latent_dim:        int   = 50
        enc_dim:           int   = 256
        enc_layers:        int   = 2        # encoder=phimap: number of hidden layers (see ENC_LAYERS)
        encoder:           str   = 'mlp'    # mlp: TD-MPC encoder; phimap: fbwm.py's gphi trunk (see encoder=)
        reg:               str   = 'vicreg' # pretrain anti-collapse term: vicreg | sigreg (see reg=)
        target:            str   = 'ema'    # pretrain consistency target: ema (phi_bar, no gradient) | online (phi, with gradient)
        sigreg_coef:       float = 0.09
        num_proj:          int   = 1024
        mlp_dim:           int   = 512
        horizon:           int   = 5
        rho:               float = 0.5
        discount:          float = 0.99
        consistency_coef:  float = 2.0
        var_coef:          float = 1.0      # VICReg variance hinge: mean_j relu(1 - std_j(h)) on the batch of encoded latents
        cov_coef:          float = 0.04     # VICReg covariance: sum of squared off-diagonal covariances / latent_dim
        reward_coef:       float = 0.5
        value_coef:        float = 0.1
        min_std:           float = 0.05
        bc_coef:           float = 0.0     # behaviour-cloning term of the policy prior (see bc_combine)
        std_clip:          float = 0.3
        iterations:        int   = 6
        num_samples:       int   = 512
        num_elites:        int   = 64
        mixture_coef:      float = 0.05
        temperature:       float = 0.5
        momentum:          float = 0.1

    class AdaptState(flax.struct.PyTreeNode):
        params:             Any          # {'encoder', 'dynamics', 'reward', 'q1', 'q2'}
        target_params:      Any          # EMA of {'encoder', 'q1', 'q2'}
        pi_params:          Any
        optimizer_state:    optax.OptState
        pi_optimizer_state: optax.OptState
        random_key:         jnp.ndarray
        steps:              jnp.ndarray

    WM_KEYS   = ('encoder', 'dynamics')
    HEAD_KEYS = ('reward', 'q1', 'q2')

    class AdaptAgent:
        def __init__(self, config: AdaptConfig, obs_dim: int, action_dim: int):
            self.cfg, self.obs_dim, self.action_dim = config, obs_dim, action_dim
            c = config
            mlp = (c.mlp_dim, c.mlp_dim)
            self.encoder    = (PhiMap(c.enc_dim, c.latent_dim, num_layers=c.enc_layers) if c.encoder == 'phimap' else
                               MLP(hidden=(c.enc_dim, c.enc_dim), out_dim=c.latent_dim))
            self.dynamics   = MLP(hidden=mlp, out_dim=c.latent_dim)
            self.reward_net = MLP(hidden=mlp, out_dim=1, zero_last=True)
            self.pi_net     = MLP(hidden=mlp, out_dim=action_dim)
            self.q_net      = QNet(c.mlp_dim)
            self.optimizer    = optax.chain(optax.clip_by_global_norm(c.grad_clip_norm), optax.adam(c.lr))
            self.pi_optimizer = optax.chain(optax.clip_by_global_norm(c.grad_clip_norm), optax.adam(c.lr))

        def init(self, random_key) -> AdaptState:
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
            return AdaptState(params=params, target_params=target_params, pi_params=pi_params,
                              optimizer_state=self.optimizer.init(params), pi_optimizer_state=self.pi_optimizer.init(pi_params),
                              random_key=random_key, steps=jnp.zeros((), dtype=jnp.int32))

        def reset_optimizers(self, ts):
            return ts.replace(optimizer_state=self.optimizer.init(ts.params),
                              pi_optimizer_state=self.pi_optimizer.init(ts.pi_params), steps=jnp.zeros((), dtype=jnp.int32))

        # ── model (as tdmpc_jax.py) ───────────────────────────────────────────
        def _bound(self, h):
            return math.sqrt(self.cfg.latent_dim) * l2_normalize(h)

        def encode(self, encoder_params, obs):
            return self._bound(self.encoder.apply({'params': encoder_params}, obs))

        def next(self, params, h, a):
            x = jnp.concatenate([h, a], axis=-1)
            h_next = self._bound(h + self.dynamics.apply({'params': params['dynamics']}, x))
            return h_next, self.reward_net.apply({'params': params['reward']}, x)

        def q(self, params, h, a):
            x = jnp.concatenate([h, a], axis=-1)
            return self.q_net.apply({'params': params['q1']}, x), self.q_net.apply({'params': params['q2']}, x)

        def pi(self, pi_params, h, key, std):
            mu = jnp.tanh(self.pi_net.apply({'params': pi_params}, h))
            return sample_truncated_normal(mu, key, std, self.cfg.std_clip) if std > 0 else mu

        @staticmethod
        def _masked_step(optimizer, grads, opt_state, params, keys):
            """Adam step on `keys` of params only: the other subtrees get zero gradients and are kept as they are."""
            grads = {k: (g if k in keys else jax.tree_util.tree_map(jnp.zeros_like, g)) for k, g in grads.items()}
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new = optax.apply_updates(params, updates)
            return {k: (new[k] if k in keys else params[k]) for k in params}, opt_state, optax.global_norm(grads)

        # ── pretrain: consistency + VICReg, encoder and dynamics only ─────────
        @partial(jax.jit, static_argnames=("self",))
        def update_pretrain(self, ts: AdaptState, replay_buffer):
            c, H = self.cfg, self.cfg.horizon
            key, k_batch = jax.random.split(ts.random_key)
            obs, action, _ = replay_buffer.sample_sequences(k_batch, c.batch_size, H)
            key, k_sig = jax.random.split(key)
            ema_target_h = self.encode(ts.target_params['encoder'], obs[1:])
            rho_t = c.rho ** jnp.arange(H, dtype=jnp.float32)

            def loss_fn(params):
                h_all = self.encode(params['encoder'], obs)                        # (H+1, B, L) online latents
                h0 = h_all[0]
                target_h = h_all[1:] if c.target == 'online' else ema_target_h      # online: gradient through the target (LeWM)
                def step(h, xs):
                    a_t, tgt, rho = xs
                    h_next, _ = self.next(params, h, a_t)
                    return h_next, rho * jnp.mean((h_next - tgt) ** 2, axis=-1)
                _, cons = jax.lax.scan(step, h0, (action, target_h, rho_t))
                cons = jnp.minimum(cons.sum(0), 1e4).mean()
                hc   = h0 - h0.mean(axis=0)
                std  = jnp.sqrt(hc.var(axis=0) + 1e-4)
                var  = jnp.mean(nn.relu(1.0 - std))
                cov  = (hc.T @ hc) / (hc.shape[0] - 1)
                cov  = (jnp.sum(cov ** 2) - jnp.sum(jnp.diag(cov) ** 2)) / c.latent_dim
                sig  = sigreg(h_all, k_sig) if c.reg == 'sigreg' else jnp.zeros(())
                total = (c.consistency_coef * cons / H +
                         (c.sigreg_coef * sig if c.reg == 'sigreg' else c.var_coef * var + c.cov_coef * cov))
                s = jnp.linalg.svd(jax.lax.stop_gradient(hc), compute_uv=False)
                p = s / s.sum()
                rank = jnp.exp(-jnp.sum(p * jnp.log(p + 1e-12)))      # effective rank of the batch of latents
                return total, {'consistency_loss': cons, 'var_loss': var, 'cov_loss': cov,
                               'latent_std': std.mean(), 'latent_rank': rank, 'sigreg_loss': sig}

            (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
            params, opt_state, gnorm = self._masked_step(self.optimizer, grads, ts.optimizer_state, ts.params, WM_KEYS)
            tau = c.tau * (ts.steps % c.update_freq == 0)
            target_params = dict(ts.target_params)
            target_params['encoder'] = jax.tree_util.tree_map(lambda p, t: t + tau * (p - t), params['encoder'], ts.target_params['encoder'])
            ts = ts.replace(params=params, target_params=target_params, optimizer_state=opt_state, random_key=key, steps=ts.steps + 1)
            return ts, dict(aux, total_loss=total, grad_norm=gnorm)

        # ── adapt (1): reward head on the N labelled transitions, frozen encoder ──
        @partial(jax.jit, static_argnames=("self",))
        def update_reward(self, ts: AdaptState, lab_obs, lab_act, lab_rew):
            c = self.cfg
            key, k = jax.random.split(ts.random_key)
            i = jax.random.randint(k, (c.batch_size,), 0, lab_obs.shape[0])
            h = self.encode(ts.params['encoder'], lab_obs[i])

            def loss_fn(params):
                r = self.reward_net.apply({'params': params['reward']}, jnp.concatenate([h, lab_act[i]], axis=-1))
                return jnp.mean((r - lab_rew[i]) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(ts.params)
            params, opt_state, gnorm = self._masked_step(self.optimizer, grads, ts.optimizer_state, ts.params, ('reward',))
            ts = ts.replace(params=params, optimizer_state=opt_state, random_key=key, steps=ts.steps + 1)
            zero = jnp.zeros(())
            return ts, {'phase': zero, 'reward_loss': loss, 'value_loss': zero, 'pi_loss': zero, 'q_mean': zero, 'grad_norm': gnorm}

        # ── adapt (2): Q and policy prior (TD-MPC losses), rewards from the frozen reward head ──
        @partial(jax.jit, static_argnames=("self",))
        def update_value(self, ts: AdaptState, replay_buffer):
            c, H = self.cfg, self.cfg.horizon
            key, k_batch, k_td, k_pi = jax.random.split(ts.random_key, 4)
            obs, action, _ = replay_buffer.sample_sequences(k_batch, c.batch_size, H)
            enc = ts.params['encoder']
            h_real = self.encode(enc, obs)                                                   # (H+1,B,L), frozen
            reward = self.reward_net.apply({'params': ts.params['reward']},
                                           jnp.concatenate([h_real[:-1], action], axis=-1))   # learned reward of each transition
            next_a = self.pi(ts.pi_params, h_real[1:], k_td, c.min_std)
            tq1, tq2 = self.q(ts.target_params, h_real[1:], next_a)
            td_target = jax.lax.stop_gradient(reward + c.discount * jnp.minimum(tq1, tq2))
            rho_t = c.rho ** jnp.arange(H, dtype=jnp.float32)

            def loss_fn(params):
                h0 = self.encode(params['encoder'], obs[0])
                def step(h, xs):
                    a_t, td_t, rho = xs
                    q1, q2 = self.q(params, h, a_t)
                    h_next, _ = self.next(params, h, a_t)
                    return h_next, (h_next, rho * ((q1 - td_t) ** 2 + (q2 - td_t) ** 2)[..., 0], q1)
                _, (hs_next, val, q1s) = jax.lax.scan(step, h0, (action, td_target, rho_t))
                val = jnp.minimum(val.sum(0), 1e4).mean()
                return c.value_coef * val / H, {'hs': jnp.concatenate([h0[None], hs_next], 0), 'value_loss': val, 'q_mean': q1s[0].mean()}

            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
            params, opt_state, gnorm = self._masked_step(self.optimizer, grads, ts.optimizer_state, ts.params, ('q1', 'q2'))
            hs = aux['hs']

            def pi_loss_fn(pi_params):
                a = self.pi(pi_params, hs, k_pi, c.min_std)
                q1, q2 = self.q(params, hs, a)
                rho = c.rho ** jnp.arange(H + 1, dtype=jnp.float32)
                main = jnp.sum(-jnp.minimum(q1, q2).mean(axis=(1, 2)) * rho)
                return bc_combine(main, jnp.mean((a[:H] - action) ** 2), c.bc_coef)

            pi_loss, pi_grads = jax.value_and_grad(pi_loss_fn)(ts.pi_params)
            pi_updates, pi_opt_state = self.pi_optimizer.update(pi_grads, ts.pi_optimizer_state, ts.pi_params)
            pi_params = optax.apply_updates(ts.pi_params, pi_updates)
            tau = c.tau * (ts.steps % c.update_freq == 0)
            target_params = dict(ts.target_params)
            for k in ('q1', 'q2'):
                target_params[k] = jax.tree_util.tree_map(lambda p, t: t + tau * (p - t), params[k], ts.target_params[k])
            ts = ts.replace(params=params, target_params=target_params, pi_params=pi_params, optimizer_state=opt_state,
                            pi_optimizer_state=pi_opt_state, random_key=key, steps=ts.steps + 1)
            return ts, {'phase': jnp.ones(()), 'reward_loss': jnp.zeros(()), 'value_loss': aux['value_loss'],
                        'pi_loss': pi_loss, 'q_mean': aux['q_mean'], 'grad_norm': gnorm}

        # ── adapt (2'), q_data=labeled: Q and policy prior from the N labelled transitions only (no dataset access) ──
        @partial(jax.jit, static_argnames=("self",))
        def update_value_labeled(self, ts: AdaptState, lab_obs, lab_act, lab_rew, lab_next_obs):
            """One-step TD on (s, a, r, s') with the true r of the labelled transitions: there are no H-step sequences."""
            c = self.cfg
            key, k_batch, k_td, k_pi = jax.random.split(ts.random_key, 4)
            i = jax.random.randint(k_batch, (c.batch_size,), 0, lab_obs.shape[0])
            enc = ts.params['encoder']
            h, h_next = self.encode(enc, lab_obs[i]), self.encode(enc, lab_next_obs[i])      # frozen
            a, r = lab_act[i], lab_rew[i]
            next_a = self.pi(ts.pi_params, h_next, k_td, c.min_std)
            tq1, tq2 = self.q(ts.target_params, h_next, next_a)
            td_target = jax.lax.stop_gradient(r + c.discount * jnp.minimum(tq1, tq2))

            def loss_fn(params):
                q1, q2 = self.q(params, h, a)
                val = jnp.minimum(((q1 - td_target) ** 2 + (q2 - td_target) ** 2)[..., 0], 1e4).mean()
                return c.value_coef * val, {'value_loss': val, 'q_mean': q1.mean()}

            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
            params, opt_state, gnorm = self._masked_step(self.optimizer, grads, ts.optimizer_state, ts.params, ('q1', 'q2'))

            def pi_loss_fn(pi_params):
                a_pi = self.pi(pi_params, h, k_pi, c.min_std)
                q1, q2 = self.q(params, h, a_pi)
                return bc_combine(-jnp.minimum(q1, q2).mean(), jnp.mean((a_pi - a) ** 2), c.bc_coef)

            pi_loss, pi_grads = jax.value_and_grad(pi_loss_fn)(ts.pi_params)
            pi_updates, pi_opt_state = self.pi_optimizer.update(pi_grads, ts.pi_optimizer_state, ts.pi_params)
            pi_params = optax.apply_updates(ts.pi_params, pi_updates)
            tau = c.tau * (ts.steps % c.update_freq == 0)
            target_params = dict(ts.target_params)
            for k in ('q1', 'q2'):
                target_params[k] = jax.tree_util.tree_map(lambda p, t: t + tau * (p - t), params[k], ts.target_params[k])
            ts = ts.replace(params=params, target_params=target_params, pi_params=pi_params, optimizer_state=opt_state,
                            pi_optimizer_state=pi_opt_state, random_key=key, steps=ts.steps + 1)
            return ts, {'phase': jnp.ones(()), 'reward_loss': jnp.zeros(()), 'value_loss': aux['value_loss'],
                        'pi_loss': pi_loss, 'q_mean': aux['q_mean'], 'grad_norm': gnorm}

        # ── adapt (2''), q_data=labeled_model: Dyna-style, imagined rollouts from the N labelled states (no dataset access) ──
        @partial(jax.jit, static_argnames=("self",))
        def update_value_model(self, ts: AdaptState, lab_obs, lab_act, lab_rew, lab_next_obs):
            """Q on (a) the real labelled transitions with the true r (one-step TD) and (b) H-step rollouts of the frozen dynamics
            from the labelled states with the noisy policy, rewards from the reward head (TD at every imagined step)."""
            c, H = self.cfg, self.cfg.horizon
            key, k_batch, k_td, k_pi, k_roll = jax.random.split(ts.random_key, 5)
            i = jax.random.randint(k_batch, (c.batch_size,), 0, lab_obs.shape[0])
            enc = ts.params['encoder']
            h, h_next = self.encode(enc, lab_obs[i]), self.encode(enc, lab_next_obs[i])
            a, r = lab_act[i], lab_rew[i]
            # imagined rollout (constants): h_0 = phi(s), a_t ~ pi(h_t) + noise, h_{t+1} = dynamics, r_t = reward head
            hs, acts, rews = [h], [], []
            for t in range(H):
                k_roll, sub = jax.random.split(k_roll)
                a_t = self.pi(ts.pi_params, hs[-1], sub, c.min_std)
                h_n, r_t = self.next(ts.params, hs[-1], a_t)
                hs.append(h_n); acts.append(a_t); rews.append(r_t)
            hs, acts, rews = jnp.stack(hs), jnp.stack(acts), jnp.stack(rews)              # (H+1,B,L) (H,B,A) (H,B,1)
            k_td1, k_td2 = jax.random.split(k_td)
            na_real = self.pi(ts.pi_params, h_next, k_td1, c.min_std)
            tq1, tq2 = self.q(ts.target_params, h_next, na_real)
            td_real = jax.lax.stop_gradient(r + c.discount * jnp.minimum(tq1, tq2))
            na_im = self.pi(ts.pi_params, hs[1:], k_td2, c.min_std)
            tq1, tq2 = self.q(ts.target_params, hs[1:], na_im)
            td_im = jax.lax.stop_gradient(rews + c.discount * jnp.minimum(tq1, tq2))     # (H,B,1)
            hs, acts = jax.lax.stop_gradient(hs), jax.lax.stop_gradient(acts)

            def loss_fn(params):
                q1, q2 = self.q(params, h, a)
                val_real = jnp.mean((q1 - td_real) ** 2 + (q2 - td_real) ** 2)
                qi1, qi2 = self.q(params, hs[:-1], acts)
                val_im = jnp.mean((qi1 - td_im) ** 2 + (qi2 - td_im) ** 2)
                val = val_real + val_im
                return c.value_coef * val, {'value_loss': val, 'q_mean': q1.mean()}

            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
            params, opt_state, gnorm = self._masked_step(self.optimizer, grads, ts.optimizer_state, ts.params, ('q1', 'q2'))

            def pi_loss_fn(pi_params):
                q1, q2 = self.q(params, hs, self.pi(pi_params, hs, k_pi, c.min_std))
                # BC on the real labelled (h, a) pairs (the imagined latents have no dataset action)
                bc = jnp.mean((self.pi(pi_params, h, k_pi, c.min_std) - a) ** 2)
                return bc_combine(-jnp.minimum(q1, q2).mean(), bc, c.bc_coef)

            pi_loss, pi_grads = jax.value_and_grad(pi_loss_fn)(ts.pi_params)
            pi_updates, pi_opt_state = self.pi_optimizer.update(pi_grads, ts.pi_optimizer_state, ts.pi_params)
            pi_params = optax.apply_updates(ts.pi_params, pi_updates)
            tau = c.tau * (ts.steps % c.update_freq == 0)
            target_params = dict(ts.target_params)
            for k in ('q1', 'q2'):
                target_params[k] = jax.tree_util.tree_map(lambda p, t: t + tau * (p - t), params[k], ts.target_params[k])
            ts = ts.replace(params=params, target_params=target_params, pi_params=pi_params, optimizer_state=opt_state,
                            pi_optimizer_state=pi_opt_state, random_key=key, steps=ts.steps + 1)
            return ts, {'phase': jnp.ones(()), 'reward_loss': jnp.zeros(()), 'value_loss': aux['value_loss'],
                        'pi_loss': pi_loss, 'q_mean': aux['q_mean'], 'grad_norm': gnorm}

        # ── MPPI planning (as tdmpc_jax.py) ───────────────────────────────────

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

    adapt_config = AdaptConfig(batch_size=BATCH_SIZE, horizon=HORIZON, bc_coef=BC_COEF, encoder=ENCODER, enc_dim=ENC_DIM, enc_layers=ENC_LAYERS,
                               reg=REG, target=TARGET, sigreg_coef=SIGREG_COEF, num_proj=NUM_PROJ)
    for name in ('lr', 'discount', 'consistency_coef', 'var_coef', 'cov_coef'):
        if name in kwargs:
            adapt_config = dataclasses.replace(adapt_config, **{name: float(kwargs[name])})
    if 'update_freq' in kwargs:     # 2026-11-22: EMA target every update_freq updates (default 2; FBWM uses 1)
        adapt_config = dataclasses.replace(adapt_config, update_freq=int(kwargs['update_freq']))
    print(adapt_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(adapt_config) + '\n')

    agent = AdaptAgent(adapt_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)
    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key, label_key = jax.random.split(random_key, 3)
    train_state = agent.init(init_key)

    eval_random_state = {'key': jax.random.PRNGKey(SEED + 10_000)}
    def next_eval_key():
        eval_random_state['key'], sub = jax.random.split(eval_random_state['key'])
        return sub

    def save(tree, path):
        with open(path, "wb") as f:
            pickle.dump(serialization.to_state_dict(jax.device_get(tree)), f)

    add_text_to_file(f'{LOGS_FOLDER}/info.txt', run_info(globals()))      # seed, job, git, command, protocol, data, versions, start time

    # ══ mode=pretrain ═══════════════════════════════════════════════════════════
    if MODE == 'pretrain':
        @jax.jit
        def train_chunk(ts, rb):
            return jax.lax.scan(lambda t, _: agent.update_pretrain(t, rb), ts, None, length=SCAN_LENGTH)

        for steps in tqdm(range(NUM_OUTER_STEPS)):
            t0 = time.time()
            train_state, m = train_chunk(train_state, replay_buffer)
            m = jax.device_get(m)
            row = {'step': (steps + 1) * SCAN_LENGTH, 'time': time.time() - t0}
            row.update({k: float(np.mean(m[k])) for k in TRAIN_METRICS})
            train_logger.log(row)
            if (steps + 1) % 100 == 0:
                save({'encoder': train_state.params['encoder'], 'dynamics': train_state.params['dynamics'],
                      'target_encoder': train_state.target_params['encoder']}, f"{MODEL_SAVE_FOLDER}/wm_latest.pickle")

        save({'encoder': train_state.params['encoder'], 'dynamics': train_state.params['dynamics'],
              'target_encoder': train_state.target_params['encoder']}, f"{MODEL_SAVE_FOLDER}/wm_final.pickle")
        print('saved', f"{MODEL_SAVE_FOLDER}/wm_final.pickle")
        mark_finished(f'{LOGS_FOLDER}/info.txt', NUM_OUTER_STEPS * SCAN_LENGTH)
        sys.exit(0)

    # ══ mode=adapt ══════════════════════════════════════════════════════════════
    with open(kwargs['wm_ckpt'], 'rb') as f:
        wm = pickle.load(f)
    params = dict(train_state.params)
    params['encoder']  = serialization.from_state_dict(params['encoder'], wm['encoder'])
    params['dynamics'] = serialization.from_state_dict(params['dynamics'], wm['dynamics'])
    target_params = dict(train_state.target_params, encoder=params['encoder'])
    train_state = agent.reset_optimizers(train_state.replace(params=params, target_params=target_params))
    print('loaded the world model from', kwargs['wm_ckpt'])

    # the N labelled transitions: (s_t, a_t, r_t = R(aux_{t+1})), drawn once from the dataset (same draw for a given seed)
    lab_i   = replay_buffer.valid_idx[jax.random.choice(label_key, replay_buffer.size, (N_LABELS,), replace=False)]
    lab_obs, lab_act, lab_rew = replay_buffer.obs[lab_i], replay_buffer.actions[lab_i], replay_buffer.rewards[lab_i + 1][:, None]
    lab_next_obs = replay_buffer.obs[lab_i + 1]

    @jax.jit
    def reward_chunk(ts):
        return jax.lax.scan(lambda t, _: agent.update_reward(t, lab_obs, lab_act, lab_rew), ts, None, length=SCAN_LENGTH)

    @jax.jit
    def value_chunk(ts, rb):
        if Q_DATA == 'labeled':
            return jax.lax.scan(lambda t, _: agent.update_value_labeled(t, lab_obs, lab_act, lab_rew, lab_next_obs),
                                ts, None, length=SCAN_LENGTH)
        if Q_DATA == 'labeled_model':
            return jax.lax.scan(lambda t, _: agent.update_value_model(t, lab_obs, lab_act, lab_rew, lab_next_obs),
                                ts, None, length=SCAN_LENGTH)
        return jax.lax.scan(lambda t, _: agent.update_value(t, rb), ts, None, length=SCAN_LENGTH)

    @jax.jit
    def _reward_corr(params, rb):    # rb is an argument, not a closure: a captured dataset is compiled in as a constant (3.7 GB on quadruped, host OOM)
        """corr(learned reward, true reward) on 10k dataset transitions (mostly outside the labelled set)."""
        k = jax.random.PRNGKey(SEED + 20_000)
        i = rb.valid_idx[jax.random.randint(k, (10_000,), 0, rb.size)]
        h = agent.encode(params['encoder'], rb.obs[i])
        r = agent.reward_net.apply({'params': params['reward']}, jnp.concatenate([h, rb.actions[i]], -1))[:, 0]
        return jnp.corrcoef(r, rb.rewards[i + 1])[0, 1]

    def reward_corr(params):
        return _reward_corr(params, replay_buffer)

    def log_train(m, step):
        m = jax.device_get(m)
        row = {'step': step, 'time': 0.0}
        row.update({k: float(np.mean(m[k])) for k in TRAIN_METRICS})
        train_logger.log(row)

    # (1) reward head
    t0 = time.time()
    for s in range(REWARD_STEPS // SCAN_LENGTH):
        train_state, m = reward_chunk(train_state)
        log_train(m, -(REWARD_STEPS - (s + 1) * SCAN_LENGTH))      # negative steps: reward phase, before the Q updates
    jax.block_until_ready(train_state.params['reward'])
    reward_fit_time = time.time() - t0
    print(f'reward head: {reward_fit_time:.1f}s, corr with the true reward {float(reward_corr(train_state.params)):.3f}')
    train_state = agent.reset_optimizers(train_state)

    # ── Evaluation (MPPI planning and/or policy prior alone) ──────────────────────

    eval_envs = []   # pool of env instances, grown on demand and reused across evals

    def rollout_batch(ts, random_key, mode='mppi'):
        """NUM_EVALS envs rolled out in lock-step, one batched jitted call per step.
        mode='mppi': MPPI planning. mode='policy': the policy prior alone (deterministic, no planning)."""
        while len(eval_envs) < NUM_EVALS:
            eval_envs.append(make_eval_env())
        envs = eval_envs[:NUM_EVALS]

        states    = [reset_for(e, TASK) for e in envs]
        prev_mean = jnp.zeros((NUM_EVALS, adapt_config.horizon, ACTION_DIM))
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

    def run_eval(ts, mode, num_evals):
        all_rewards = []
        for _ in range(num_evals // NUM_EVALS):
            _, aux_traj, _ = rollout_batch(ts, next_eval_key(), mode)
            tr = np.asarray(reward_fn(jnp.asarray(aux_traj, dtype=jnp.float32), reward_param)).sum(axis=-1)
            if ENV_NAME in ('antmaze', 'antmaze_medium', 'antmaze_large', 'cube_single') and 'goal' in TASK:
                tr = np.where(tr != -EVAL_STEPS, 1.0, 0.0)
            all_rewards.append(tr)
        tr = np.concatenate(all_rewards)
        print(TASK, f'[{MODE_NAMES[mode]}]', 'mean', tr.mean(), 'std', tr.std(ddof=1))
        return tr.mean(), tr.std(ddof=1)

    def evaluate(ts, step, num_evals, tag, adapt_time):
        t0 = time.time()
        res = {mode: run_eval(ts, mode, num_evals) for mode in EVAL_MODES}
        first = EVAL_MODES[0]
        row = {'step': step, 'return_mean': float(res[first][0]), 'return_std': float(res[first][1]),
               'time': time.time() - t0, 'num_evals': num_evals, 'tag': tag, 'reward_fit_time': reward_fit_time,
               'adapt_time': adapt_time, 'reward_corr': float(reward_corr(ts.params))}
        if EVAL_MODE == 'both':
            row['policy_return_mean'], row['policy_return_std'] = float(res['policy'][0]), float(res['policy'][1])
        eval_logger.log(row)

    # adapt step 0: the reward head alone (Q = 0 at init, so MPPI is pure H-step MPC; the policy prior is untrained)
    evaluate(train_state, 0, NUM_EVALS, 'eval', 0.0)

    # (2) Q and policy prior
    adapt_time = 0.0
    for steps in tqdm(range(NUM_OUTER_STEPS)):
        t0 = time.time()
        train_state, m = value_chunk(train_state, replay_buffer)
        jax.block_until_ready(train_state.pi_params)
        adapt_time += time.time() - t0
        log_train(m, (steps + 1) * SCAN_LENGTH)
        if (steps + 1) % EVAL_EACH_STEPS == 0 and steps + 1 < NUM_OUTER_STEPS:
            evaluate(train_state, (steps + 1) * SCAN_LENGTH, NUM_EVALS, 'eval', adapt_time)

    save({'params': train_state.params, 'target_params': train_state.target_params, 'pi_params': train_state.pi_params},
         f"{MODEL_SAVE_FOLDER}/adapt_final.pickle")
    evaluate(train_state, ADAPT_STEPS, NUM_EVALS_FINAL, 'final', adapt_time)
    mark_finished(f'{LOGS_FOLDER}/info.txt', NUM_OUTER_STEPS * SCAN_LENGTH)
