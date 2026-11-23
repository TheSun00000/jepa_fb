"""FB as implemented in TD-JEPA (Bagatella et al., ICLR 2026), ported to JAX for OGBench (antmaze_medium | antmaze_large | cube_single).

Reference: github.com/facebookresearch/td_jepa, metamotivo/agents/fb/{agent.py, model.py, flow_bc/}, metamotivo/nn_models.py,
scripts/train/proprio/launch_fb_ogbench.py (BASE_CFG + sweep_antmaze / sweep_cube), metamotivo/evaluations/ogbench.py.
Their proprio OGBench numbers (Table 1, FB = FB_deep): antmaze-medium-navigate 73.0%, cube-single 49.6%; without the left encoder
(FB on raw states): 45.2% / 21.0%. Each number is the best of their 4-config sweep (ortho_coef {100, 1000} x lr_b {1e-4, 1e-5}).

What is ported (defaults = their OGBench proprio config):
- left encoder L(s) in front of F ('deep': Linear 512 -> LayerNorm -> Tanh -> 3 x [Linear 512 -> ReLU] -> Linear L_dim=50 -> sqrt(d)
  normalize; 'shallow': Linear L_dim=256 -> normalize; 'none': identity = plain FB). L is trained by the FB loss (F's optimizer), with
  its own EMA target used for the next state.
- B: same MLP as the deep encoder, on the raw next state, output z_dim=50, sqrt(d) normalize.
- F: two independent members (TD-JEPA's DenseParallel), each embed_sa(L(s), a) and embed_z(L(s), z) (Linear 512 -> LN -> Tanh ->
  Linear 256 -> ReLU), then 2 x [Linear 512 -> ReLU] -> Linear z_dim. Targets and Q use the MEAN of the two members (pessimism 0).
- FB loss: diff = F B^T - discount * target_M; 0.5 * mean off-diagonal diff^2 (summed over members) - mean diagonal diff (summed over
  members); orthonormality 0.5 * mean off-diag (B B^T)^2 - mean diag(B B^T), weight ortho_coef. No Q loss. discount 0.99, no terminals.
- z: Gaussian projected on the sphere of radius sqrt(z_dim); with probability 0.5 replaced by B(next_obs) of a shuffled batch state.
- Actor (flow BC, 'FBFlowBCAgent'): noise-conditioned actor pi(s, z, eps) on the raw state (embed(s, z, eps), embed(s, eps),
  2 x [Linear 512 -> ReLU] -> Linear -> tanh); a BC flow-matching vector field v(s, a_t, t) (4 x [Linear 512 -> GELU] -> Linear) trained on
  dataset actions; actor loss -Q / mean|Q| + bc_coef * ||pi(s, z, eps) - flow(s, eps)||^2 (flow: 10 Euler steps, clamped to [-1, 1]).
  Acting = pi(s, z, eps) with a fresh eps ~ N(0, I) at every step (also at evaluation, as in their act()).
- Orthogonal init (gain 1; sqrt(2) for F's parallel layers), zero biases. Adam: lr_f 1e-4 (F + L), lr_b 1e-4, lr_actor 1e-4,
  lr_actor_vf 3e-4. EMA tau 0.005 for F, L and B. Batch 256, 1M updates.
- Evaluation (theirs): the 5 official tasks; z = sqrt(d) normalize(sum_i r_i B(s'_i)) on z_labels=10000 dataset transitions, with
  r = OGBench's reward (0 at success, -1 otherwise; antmaze: ant (x, y) within 0.5 of the goal, cube: block within 0.04 m) shifted by
  +1 (shift_reward=1). Success = OGBench's info['success'] at any step of the episode. One z per task, num_evals_final episodes per task.
  NOTE: fb_jax.py / fbwm.py use +-1 rewards within 2 m (antmaze) for z inference and their periodic success; this script does not.

Flags (key=value): env_name (required), dataset (antmaze: navigate default; cube: play), seed, bc_coef (antmaze 0.3, cube 3.0),
ortho_coef (100), lr_b (1e-4), lr_f, lr_actor, lr_actor_vf, discount (0.99), batch_size (256), z_dim (50), left_encoder (deep|shallow|none),
tau (0.005), train_goal_ratio (0.5), flow_steps (10), z_labels (10000), shift_reward (1), num_outer_steps (1000 x 1000 updates),
eval_each_steps (in outer steps, 100), num_evals (periodic episodes per task, 5), num_evals_final (10), eval_ckpt (eval only).
Outputs: logs/<ts>_FBTDJ-<variant>-<env>/{training_log.csv, eval_log.csv, info.txt}, models/<run>/fbtdj_{latest,final,step:N}.pickle.
"""
import os
LOCAL = 'nazim' in os.getcwd()
if not LOCAL:
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

import csv
import sys
import shutil
import math
import time
import pickle
import dataclasses
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax import serialization
import optax

from utils.logs import parse_kwargs, resolve_seed, run_info, mark_finished, add_text_to_file


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath, self.fieldnames = filepath, fieldnames
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row):
        with open(self.filepath, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


if __name__ == '__main__':
    kwargs = parse_kwargs(sys.argv[1:])
    print(kwargs)
    print(jax.devices())

    assert 'env_name' in kwargs, 'env_name= is required'
    ENV_NAME = kwargs['env_name']
    assert ENV_NAME in ('antmaze_medium', 'antmaze_large', 'cube_single'), 'OGBench only: antmaze_medium | antmaze_large | cube_single'
    IS_CUBE = ENV_NAME == 'cube_single'
    if 'dataset' not in kwargs:
        kwargs['dataset'] = 'play' if IS_CUBE else 'navigate'
    LEFT_ENCODER = str(kwargs.get('left_encoder', 'deep'))
    assert LEFT_ENCODER in ('deep', 'shallow', 'none'), 'left_encoder=deep|shallow|none'
    SEED = resolve_seed(kwargs)

    # ── Logging ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    date_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = f'FBTDJ-{LEFT_ENCODER}-{ENV_NAME}'
    while True:   # two jobs started in the same second get the same name: append '_'
        LOGS_FOLDER, MODEL_SAVE_FOLDER = f'./logs/{date_time_str}_{exp_name}', f'./models/{date_time_str}_{exp_name}'
        try:
            os.makedirs(LOGS_FOLDER)
            break
        except FileExistsError:
            exp_name += '_'
    os.makedirs(MODEL_SAVE_FOLDER)
    print('LOGS_FOLDER      :', LOGS_FOLDER)
    print('MODEL_SAVE_FOLDER:', MODEL_SAVE_FOLDER)
    shutil.copy(os.path.abspath(__file__), os.path.join(LOGS_FOLDER, os.path.basename(__file__)))
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as f:
        f.write(str(kwargs) + '\n')

    # ── Data and tasks (utils/ogbench_tasks.py loader; only the 5 official goal tasks) ─────────────────────────────────────────
    from utils.ogbench_tasks import load_ogbench, reset_env
    _ogb = load_ogbench(ENV_NAME, kwargs)
    STATE_DIM, ACTION_DIM = _ogb['state_dim'], _ogb['action_dim']
    NUM_TRAJECTORIES, TRAJECTORY_LEN = _ogb['num_trajectories'], _ogb['trajectory_len']
    EVAL_STEPS, make_eval_env = _ogb['eval_steps'], _ogb['make_eval_env']
    GOALS = [(label, np.asarray(param, dtype=np.float32)) for _, label, param in _ogb['benchmarks'] if label.startswith('goal_')]
    assert len(GOALS) == 5, [g[0] for g in GOALS]
    dataset_obs, dataset_actions = _ogb['obs'], _ogb['actions']
    del _ogb
    GOAL_SLICE, GOAL_TOL = (slice(19, 22), 0.4) if IS_CUBE else (slice(0, 2), 0.5)   # OGBench success radius, observation units

    def ogbench_reward(obs, goal):
        """TD-JEPA's relabelled reward: 0 if the goal coordinates are within the OGBench radius, else -1 (before shift_reward)."""
        return (np.linalg.norm(obs[..., GOAL_SLICE] - goal, axis=-1) <= GOAL_TOL).astype(np.float32) - 1.0

    # ── Hyper-parameters (TD-JEPA OGBench proprio defaults) ───────────────────────────────────────────────────────────────────
    @dataclasses.dataclass(frozen=True)
    class Config:
        z_dim:            int   = 50
        L_dim:            int   = 50          # 256 for left_encoder=shallow (their 'shallow' setting)
        hidden_dim:       int   = 512
        b_layers:         int   = 4           # BackwardArchi hidden_layers (B and the deep left encoder)
        f_hidden_layers:  int   = 2
        actor_hidden_layers: int = 2
        vf_layers:        int   = 4
        num_parallel:     int   = 2
        lr_f:             float = 1e-4
        lr_b:             float = 1e-4
        lr_actor:         float = 1e-4
        lr_actor_vf:      float = 3e-4
        tau:              float = 0.005       # f_target_tau = b_target_tau
        ortho_coef:       float = 100.0       # their sweep: {100, 1000}
        train_goal_ratio: float = 0.5
        batch_size:       int   = 256
        discount:         float = 0.99
        bc_coef:          float = 0.3         # their sweep: antmaze 0.3, cube 3.0
        flow_steps:       int   = 10
        left_encoder:     str   = 'deep'

    cfg = Config(left_encoder=LEFT_ENCODER, bc_coef=3.0 if IS_CUBE else 0.3, L_dim=256 if LEFT_ENCODER == 'shallow' else 50)
    for name, typ in (('z_dim', int), ('L_dim', int), ('lr_f', float), ('lr_b', float), ('lr_actor', float), ('lr_actor_vf', float),
                      ('tau', float), ('ortho_coef', float), ('train_goal_ratio', float), ('batch_size', int), ('discount', float),
                      ('bc_coef', float), ('flow_steps', int)):
        if name in kwargs:
            cfg = dataclasses.replace(cfg, **{name: typ(kwargs[name])})
    print(cfg)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as f:
        f.write(str(cfg) + '\n')

    SCAN_LENGTH = 1_000
    NUM_OUTER_STEPS = int(kwargs.get('num_outer_steps', 1000))          # 1M updates (TD-JEPA OGBench)
    EVAL_EACH_STEPS = int(kwargs.get('eval_each_steps', 100))           # every 100k updates
    SAVE_EACH_STEPS = 250
    NUM_EVALS = int(kwargs.get('num_evals', 5))
    NUM_EVALS_FINAL = int(kwargs.get('num_evals_final', 10))            # theirs: 10 episodes per task
    Z_LABELS = int(kwargs.get('z_labels', 10_000))                      # theirs: num_inference_samples 10_000
    SHIFT_REWARD = float(kwargs.get('shift_reward', 1.0))               # theirs: shift_reward 1
    EVAL_CKPT = kwargs.get('eval_ckpt')

    TRAIN_METRICS = ['fb_loss', 'fb_diag', 'fb_offdiag', 'orth_loss', 'orth_loss_diag', 'orth_loss_offdiag', 'target_M', 'M1', 'F1',
                     'B_norm', 'z_norm', 'actor_loss', 'bc_flow_loss', 'bc_error', 'q']
    train_logger = CSVLogger(f'{LOGS_FOLDER}/training_log.csv', ['step'] + TRAIN_METRICS + ['time'])
    eval_logger = CSVLogger(f'{LOGS_FOLDER}/eval_log.csv',
                            ['step', 'success'] + [f'{g}_success' for g, _ in GOALS] + [f'{g}_corr' for g, _ in GOALS]
                            + ['time', 'num_evals', 'tag', 'z_labels', 'shift_reward'])

    # ── Replay buffer: transitions (s_t, a_t, s_{t+1}) inside a trajectory, no terminals (discount mask = 1) ──────────────────
    n_total = NUM_TRAJECTORIES * TRAJECTORY_LEN
    _idx = np.arange(n_total)
    VALID_IDX = jnp.asarray(_idx[(_idx % TRAJECTORY_LEN) != TRAJECTORY_LEN - 1], dtype=jnp.int32)
    OBS = jnp.asarray(dataset_obs[:n_total], dtype=jnp.float32)
    ACTIONS = jnp.asarray(dataset_actions[:n_total], dtype=jnp.float32)
    del dataset_obs, dataset_actions

    def sample_batch(key, n):
        i = VALID_IDX[jax.random.randint(key, (n,), 0, VALID_IDX.shape[0])]
        return OBS[i], ACTIONS[i], OBS[i + 1]

    # ── Networks (metamotivo/nn_models.py) ─────────────────────────────────────────────────────────────────────────────────────
    def dense(features, gain=1.0):
        return nn.Dense(features, kernel_init=nn.initializers.orthogonal(gain), bias_init=nn.initializers.zeros)

    def sphere(x):
        """Norm(): sqrt(d) * F.normalize(x)."""
        return math.sqrt(x.shape[-1]) * x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    class BackwardMLP(nn.Module):
        """BackwardArchi (norm=True): Linear -> LayerNorm -> Tanh -> (layers-1) x [Linear -> ReLU] -> Linear -> Norm.
        layers=0: Linear -> Norm (their 'shallow' left encoder)."""
        out_dim: int
        hidden_dim: int
        layers: int

        @nn.compact
        def __call__(self, x):
            if self.layers == 0:
                return sphere(dense(self.out_dim)(x))
            x = jnp.tanh(nn.LayerNorm(epsilon=1e-5)(dense(self.hidden_dim)(x)))
            for _ in range(self.layers - 1):
                x = nn.relu(dense(self.hidden_dim)(x))
            return sphere(dense(self.out_dim)(x))

    def embedding(x, hidden_dim, gain):
        """simple_embedding with 2 layers: Linear -> LayerNorm -> Tanh -> Linear(hidden/2) -> ReLU."""
        x = jnp.tanh(nn.LayerNorm(epsilon=1e-5)(dense(hidden_dim, gain)(x)))
        return nn.relu(dense(hidden_dim // 2, gain)(x))

    class ForwardMember(nn.Module):
        z_dim: int
        hidden_dim: int
        hidden_layers: int

        @nn.compact
        def __call__(self, h, z, a):
            g = math.sqrt(2.0)   # parallel_orthogonal_ with the relu gain (DenseParallel layers)
            x = jnp.concatenate([embedding(jnp.concatenate([h, a], -1), self.hidden_dim, g),
                                 embedding(jnp.concatenate([h, z], -1), self.hidden_dim, g)], -1)
            for _ in range(self.hidden_layers):
                x = nn.relu(dense(self.hidden_dim, g)(x))
            return dense(self.z_dim, g)(x)

    ForwardEnsemble = nn.vmap(ForwardMember, variable_axes={'params': 0}, split_rngs={'params': True},
                              in_axes=None, out_axes=0, axis_size=cfg.num_parallel)

    class NoiseActor(nn.Module):
        action_dim: int
        hidden_dim: int
        hidden_layers: int

        @nn.compact
        def __call__(self, s, z, eps):
            x = jnp.concatenate([embedding(jnp.concatenate([s, eps], -1), self.hidden_dim, 1.0),
                                 embedding(jnp.concatenate([s, z, eps], -1), self.hidden_dim, 1.0)], -1)
            for _ in range(self.hidden_layers):
                x = nn.relu(dense(self.hidden_dim)(x))
            return jnp.tanh(dense(self.action_dim)(x))

    class VectorField(nn.Module):
        action_dim: int
        hidden_dim: int
        layers: int

        @nn.compact
        def __call__(self, s, a, t):
            x = jnp.concatenate([s, a, t], -1)
            for _ in range(self.layers):
                x = nn.gelu(dense(self.hidden_dim)(x), approximate=False)
            return dense(self.action_dim)(x)

    class Identity(nn.Module):
        @nn.compact
        def __call__(self, x):
            return x

    left_net = {'deep': BackwardMLP(cfg.L_dim, cfg.hidden_dim, cfg.b_layers), 'shallow': BackwardMLP(cfg.L_dim, cfg.hidden_dim, 0),
                'none': Identity()}[cfg.left_encoder]
    b_net = BackwardMLP(cfg.z_dim, cfg.hidden_dim, cfg.b_layers)
    f_net = ForwardEnsemble(cfg.z_dim, cfg.hidden_dim, cfg.f_hidden_layers)
    actor_net = NoiseActor(ACTION_DIM, cfg.hidden_dim, cfg.actor_hidden_layers)
    vf_net = VectorField(ACTION_DIM, cfg.hidden_dim, cfg.vf_layers)
    L_OUT = STATE_DIM if cfg.left_encoder == 'none' else cfg.L_dim

    class TrainState(flax.struct.PyTreeNode):
        params: Any            # {'left', 'f', 'b', 'actor', 'vf'}
        target: Any            # {'left', 'f', 'b'}
        opt_state: Any         # {'f', 'b', 'actor', 'vf'}
        key: jnp.ndarray
        steps: jnp.ndarray

    optimizers = {'f': optax.adam(cfg.lr_f), 'b': optax.adam(cfg.lr_b), 'actor': optax.adam(cfg.lr_actor), 'vf': optax.adam(cfg.lr_actor_vf)}

    def init_state(key):
        kl, kf, kb, ka, kv, key = jax.random.split(key, 6)
        s, a, z = jnp.zeros((1, STATE_DIM)), jnp.zeros((1, ACTION_DIM)), jnp.zeros((1, cfg.z_dim))
        params = {'left': left_net.init(kl, s)['params'] if cfg.left_encoder != 'none' else {},
                  'f': f_net.init(kf, jnp.zeros((1, L_OUT)), z, a)['params'],
                  'b': b_net.init(kb, s)['params'],
                  'actor': actor_net.init(ka, s, z, a)['params'],
                  'vf': vf_net.init(kv, s, a, jnp.zeros((1, 1)))['params']}
        target = {k: jax.tree_util.tree_map(jnp.copy, params[k]) for k in ('left', 'f', 'b')}
        opt_state = {'f': optimizers['f'].init({'f': params['f'], 'left': params['left']}), 'b': optimizers['b'].init(params['b']),
                     'actor': optimizers['actor'].init(params['actor']), 'vf': optimizers['vf'].init(params['vf'])}
        return TrainState(params=params, target=target, opt_state=opt_state, key=key, steps=jnp.zeros((), jnp.int32))

    def L(p, s):
        return s if cfg.left_encoder == 'none' else left_net.apply({'params': p}, s)

    def B(p, s):
        return b_net.apply({'params': p}, s)

    def Fs(p, h, z, a):
        return f_net.apply({'params': p}, h, z, a)        # (num_parallel, batch, z_dim)

    def pi(p, s, z, eps):
        return actor_net.apply({'params': p}, s, z, eps)

    def flow_actions(p_vf, s, eps):
        a = eps
        for i in range(cfg.flow_steps):
            t = jnp.full((s.shape[0], 1), i / cfg.flow_steps)
            a = a + vf_net.apply({'params': p_vf}, s, a, t) / cfg.flow_steps
        return jnp.clip(a, -1.0, 1.0)

    def sample_z(key, n):
        return sphere(jax.random.normal(key, (n, cfg.z_dim)))

    # ── One update (FBAgent.update + FBFlowBCAgent.update_actor) ────────────────────────────────────────────────────────────────
    BS = cfg.batch_size
    OFF_DIAG = 1.0 - jnp.eye(BS)
    N_OFF = OFF_DIAG.sum()

    def update(ts: TrainState, _):
        key, kb, kz, kp, km, kn, kx0, kt, ke = jax.random.split(ts.key, 9)
        obs, action, next_obs = sample_batch(kb, BS)
        p = ts.params

        # z: sphere Gaussian; with prob. train_goal_ratio, B(goal) of a shuffled batch state (goal = next_obs), no gradient
        z = sample_z(kz, BS)
        goals_z = jax.lax.stop_gradient(sphere(B(p['b'], next_obs[jax.random.permutation(kp, BS)])))
        z = jnp.where(jax.random.uniform(km, (BS, 1)) < cfg.train_goal_ratio, goals_z, z)

        # target: mean over members of target_F(L_target(s'), z, a') target_B(s')^T, a' = pi(s', z, eps)
        next_h = L(ts.target['left'], next_obs)
        next_a = pi(p['actor'], next_obs, z, jax.random.normal(kn, (BS, ACTION_DIM)))
        target_M = jax.lax.stop_gradient(
            jnp.einsum('pbd,cd->pbc', Fs(ts.target['f'], next_h, z, next_a), B(ts.target['b'], next_obs)).mean(0))

        def fb_loss_fn(fp, bp):
            h = L(fp['left'], obs)
            F_ = Fs(fp['f'], h, z, action)                          # (P, bs, d)
            B_ = B(bp, next_obs)                                     # (bs, d)
            Ms = jnp.einsum('pbd,cd->pbc', F_, B_)                   # (P, bs, bs)
            diff = Ms - cfg.discount * target_M
            fb_offdiag = 0.5 * ((diff * OFF_DIAG) ** 2).sum() / N_OFF
            fb_diag = -jnp.diagonal(diff, axis1=1, axis2=2).mean() * Ms.shape[0]
            Cov = B_ @ B_.T
            orth_diag = -jnp.diag(Cov).mean()
            orth_offdiag = 0.5 * ((Cov * OFF_DIAG) ** 2).sum() / N_OFF
            orth = orth_offdiag + orth_diag
            loss = fb_offdiag + fb_diag + cfg.ortho_coef * orth
            m = {'fb_loss': loss, 'fb_diag': fb_diag, 'fb_offdiag': fb_offdiag, 'orth_loss': orth, 'orth_loss_diag': orth_diag,
                 'orth_loss_offdiag': orth_offdiag, 'target_M': target_M.mean(), 'M1': Ms[0].mean(), 'F1': F_[0].mean(),
                 'B_norm': jnp.linalg.norm(B_, axis=-1).mean(), 'z_norm': jnp.linalg.norm(z, axis=-1).mean()}
            return loss, m

        fparams = {'f': p['f'], 'left': p['left']}
        (_, metrics), (g_f, g_b) = jax.value_and_grad(fb_loss_fn, argnums=(0, 1), has_aux=True)(fparams, p['b'])
        up_f, os_f = optimizers['f'].update(g_f, ts.opt_state['f'], fparams)
        fparams = optax.apply_updates(fparams, up_f)
        up_b, os_b = optimizers['b'].update(g_b, ts.opt_state['b'], p['b'])
        bparams = optax.apply_updates(p['b'], up_b)

        # actor (after the FB step, with the updated F and L, as in torch): flow-matching BC on the vector field + -Q / |Q| + bc
        x1 = action
        x0 = jax.random.normal(kx0, x1.shape)
        t = jax.random.uniform(kt, (BS, 1))
        xt, vel = (1 - t) * x0 + t * x1, x1 - x0
        eps = jax.random.normal(ke, x1.shape)
        h_obs = jax.lax.stop_gradient(L(fparams['left'], obs))
        target_flow = jax.lax.stop_gradient(flow_actions(p['vf'], obs, eps))

        def actor_loss_fn(ap, vp):
            bc_flow = ((vf_net.apply({'params': vp}, obs, xt, t) - vel) ** 2).mean()
            a = pi(ap, obs, z, eps)
            Qs = (Fs(fparams['f'], h_obs, z, a) * z).sum(-1)         # (P, bs)
            Q = Qs.mean(0)
            loss = -Q.mean()
            bc_error = ((a - target_flow) ** 2).mean()
            if cfg.bc_coef > 0:
                loss = loss / jax.lax.stop_gradient(jnp.abs(Qs).mean()) + cfg.bc_coef * bc_error
            loss = loss + bc_flow
            return loss, {'actor_loss': loss, 'bc_flow_loss': bc_flow, 'bc_error': bc_error, 'q': Q.mean()}

        (_, am), (g_a, g_v) = jax.value_and_grad(actor_loss_fn, argnums=(0, 1), has_aux=True)(p['actor'], p['vf'])
        up_a, os_a = optimizers['actor'].update(g_a, ts.opt_state['actor'], p['actor'])
        up_v, os_v = optimizers['vf'].update(g_v, ts.opt_state['vf'], p['vf'])
        metrics.update(am)

        params = {'left': fparams['left'], 'f': fparams['f'], 'b': bparams,
                  'actor': optax.apply_updates(p['actor'], up_a), 'vf': optax.apply_updates(p['vf'], up_v)}
        ema = lambda new, old: jax.tree_util.tree_map(lambda x, y: cfg.tau * x + (1 - cfg.tau) * y, new, old)
        target = {k: ema(params[k], ts.target[k]) for k in ('left', 'f', 'b')}
        ts = ts.replace(params=params, target=target, opt_state={'f': os_f, 'b': os_b, 'actor': os_a, 'vf': os_v},
                        key=key, steps=ts.steps + 1)
        return ts, metrics

    @jax.jit
    def train_chunk(ts):
        return jax.lax.scan(update, ts, None, length=SCAN_LENGTH)

    # ── Evaluation (metamotivo/evaluations/ogbench.py) ─────────────────────────────────────────────────────────────────────────
    @jax.jit
    def b_of(bp, s):
        return B(bp, s)

    @jax.jit
    def act(ap, s, z, key):
        return pi(ap, s, z, jax.random.normal(key, (s.shape[0], ACTION_DIM)))

    eval_key = {'k': jax.random.PRNGKey(SEED + 10_000)}

    def next_key():
        eval_key['k'], sub = jax.random.split(eval_key['k'])
        return sub

    def infer_z(bp, goal):
        """z = sqrt(d) normalize(sum r_i B(s'_i)), r = OGBench reward (0 / -1) + shift on Z_LABELS random dataset transitions."""
        i = np.asarray(VALID_IDX[jax.random.randint(next_key(), (Z_LABELS,), 0, VALID_IDX.shape[0])]) + 1
        nxt = np.asarray(OBS[i])
        r = ogbench_reward(nxt, goal) + SHIFT_REWARD
        Bn = np.asarray(b_of(bp, jnp.asarray(nxt)))
        z = r @ Bn
        z = math.sqrt(cfg.z_dim) * z / max(np.linalg.norm(z), 1e-12)
        pred = Bn @ z
        corr = float(np.corrcoef(pred, r)[0, 1]) if r.std() > 0 else float('nan')
        return z.astype(np.float32), corr

    eval_envs = []

    def evaluate(ts, step, num_episodes, tag):
        t0 = time.time()
        row = {'step': step, 'num_evals': num_episodes, 'tag': tag, 'z_labels': Z_LABELS, 'shift_reward': SHIFT_REWARD}
        succ_all = []
        n = len(GOALS) * num_episodes
        while len(eval_envs) < n:
            eval_envs.append(make_eval_env())
        zs, labels = [], []
        for label, goal in GOALS:                         # one z per task, as theirs
            z, corr = infer_z(ts.params['b'], goal)
            row[f'{label}_corr'] = corr
            zs += [z] * num_episodes
            labels += [label] * num_episodes
        zs = jnp.asarray(np.stack(zs))
        obs = np.stack([reset_env(e, lab) for e, lab in zip(eval_envs[:n], labels)]).astype(np.float32)
        done, success = np.zeros(n, bool), np.zeros(n, bool)
        for _ in tqdm(range(EVAL_STEPS), leave=False):
            actions = np.asarray(act(ts.params['actor'], jnp.asarray(obs), zs, next_key()))
            for j, e in enumerate(eval_envs[:n]):
                if done[j]:
                    continue
                o, _, terminated, truncated, info = e.step(actions[j])
                obs[j] = o
                success[j] |= bool(info.get('success', False))
                done[j] = terminated or truncated
            if done.all():
                break
        for k, (label, _) in enumerate(GOALS):
            s = success[k * num_episodes:(k + 1) * num_episodes]
            row[f'{label}_success'] = float(s.mean())
            print(f'{label}: success {s.mean():.2f} | corr(B^T z, r) {row[f"{label}_corr"]:.3f}')
        row['success'] = float(success.mean())
        row['time'] = time.time() - t0
        print(f'[{tag}] step {step}: success {row["success"]:.3f} ({time.time() - t0:.0f}s)')
        eval_logger.log(row)

    def save(ts, path):
        with open(path, 'wb') as f:
            pickle.dump(serialization.to_state_dict(jax.device_get({'params': ts.params, 'target': ts.target})), f)

    # ── Run ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ts = init_state(jax.random.PRNGKey(SEED))
    add_text_to_file(f'{LOGS_FOLDER}/info.txt', run_info(globals(), z_labels=Z_LABELS))

    if EVAL_CKPT:
        with open(EVAL_CKPT, 'rb') as f:
            restored = serialization.from_state_dict({'params': ts.params, 'target': ts.target}, pickle.load(f))
        ts = ts.replace(**restored)
        print('loaded', EVAL_CKPT)
        evaluate(ts, 0, NUM_EVALS_FINAL, 'zlabels')
        mark_finished(f'{LOGS_FOLDER}/info.txt', 0)
        sys.exit(0)

    for steps in tqdm(range(NUM_OUTER_STEPS)):
        t0 = time.time()
        ts, m = train_chunk(ts)
        m = jax.device_get(m)
        row = {'step': (steps + 1) * SCAN_LENGTH, 'time': time.time() - t0}
        row.update({k: float(np.mean(m[k])) for k in TRAIN_METRICS})
        train_logger.log(row)
        if (steps + 1) % EVAL_EACH_STEPS == 0 and steps + 1 < NUM_OUTER_STEPS:
            evaluate(ts, (steps + 1) * SCAN_LENGTH, NUM_EVALS, 'eval')
            save(ts, f'{MODEL_SAVE_FOLDER}/fbtdj_latest.pickle')
        if (steps + 1) % SAVE_EACH_STEPS == 0:
            save(ts, f'{MODEL_SAVE_FOLDER}/fbtdj_step:{(steps + 1) * SCAN_LENGTH}.pickle')

    save(ts, f'{MODEL_SAVE_FOLDER}/fbtdj_final.pickle')
    evaluate(ts, NUM_OUTER_STEPS * SCAN_LENGTH, NUM_EVALS_FINAL, 'final')
    mark_finished(f'{LOGS_FOLDER}/info.txt', NUM_OUTER_STEPS * SCAN_LENGTH)
