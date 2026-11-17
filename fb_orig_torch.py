"""Minimal PyTorch port of the ORIGINAL Forward-Backward agent (facebookresearch/controllable_agent, fb_ddpg.py), cheetah only.

Why: our JAX fb_jax.py reproduces the paper on walker and quadruped but not on cheetah. This file keeps only what the original run does
(offline training on the ExoRL RND episodes, FB update, actor update, z inferred from labelled states, evaluation on the cheetah tasks)
and none of its framework (hydra, wandb, video, replay-buffer classes), so that the two implementations can be compared line by line.

Differences with fb_jax.py that this file keeps from the original on purpose:
  * orthogonal init (gain 1, zero bias) of every Linear of the actor / forward / backward networks (utils.weight_init); fb_jax.py uses
    torch's default init;
  * q_loss=False by default (fb_jax.py: on, coefficient 0.01);
  * discount 0.99, 1 update per step, batch 1024, lr 1e-4, tau 0.01, z_dim 50, mix_ratio 0.5, exploration std 0.2 clipped at 0.3.

Data: the ExoRL episodes, <data_dir>/*.npz with observation (1001, 17), action (1001, 6), discount, physics (1001, 18). Row i of 'action' is
the action that PRODUCED row i of 'observation', so a transition is (obs[i-1], action[i], obs[i]) for i = 1..1000 (as in the original).

Usage (flags are key=value, no dashes):
    python fb_orig_torch.py data_dir=/path/to/cheetah/rnd/buffer seed=1
Flags (default): num_updates (1000000), eval_every (50000), num_evals (5), num_evals_late (10, last 10% of training), num_evals_final (10),
    batch_size (1024), discount (0.99), z_dim (50), lr (1e-4), tau (0.01), hidden_dim (1024), feature_dim (512), backward_hidden_dim (526),
    mix_ratio (0.5), stddev (0.2), stddev_clip (0.3), q_loss (0), q_loss_coef (0.01), ortho_coef (1.0), ortho_init (1), num_inference (5120),
    max_episodes (all; smoke tests), resume (path to ckpt_latest.pt), ckpt_every (50000), seed (1).
"""
import csv
import glob
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────────── config ─────────────────────────────
DEFAULTS = dict(
    data_dir=None, seed=1, num_updates=1_000_000, eval_every=50_000, num_evals=5, num_evals_late=10, num_evals_final=10,
    batch_size=1024, discount=0.99, z_dim=50, lr=1e-4, tau=0.01, hidden_dim=1024, feature_dim=512, backward_hidden_dim=526,
    mix_ratio=0.5, stddev=0.2, stddev_clip=0.3, q_loss=0, q_loss_coef=0.01, ortho_coef=1.0, ortho_init=1, num_inference=5120,
    max_episodes=0, resume=None, ckpt_every=50_000, log_every=1000,
)


def parse_kwargs(argv):
    out = dict(DEFAULTS)
    for a in argv:
        k, v = a.split('=', 1)
        assert k in DEFAULTS, f'unknown flag {k}'
        d = DEFAULTS[k]
        out[k] = v if (d is None or isinstance(d, str)) else (float(v) if isinstance(d, float) else int(v))
    return out


# ───────────────────────────── networks (copied from agent/fb_modules.py, preprocess=True, add_trunk=False) ─────────────────────────────
def mlp(*layers):
    seq, prev = [], layers[0]
    for l in layers[1:]:
        if l == 'irelu':
            seq.append(nn.ReLU(inplace=True))
        elif l == 'relu':
            seq.append(nn.ReLU())
        elif l == 'ntanh':
            seq += [nn.LayerNorm(prev), nn.Tanh()]
        else:
            seq.append(nn.Linear(prev, l))
            prev = l
    return nn.Sequential(*seq)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim, ortho_init=True):
        super().__init__()
        self.obs_net = mlp(obs_dim, hidden_dim, 'ntanh', feature_dim, 'irelu')
        self.obs_z_net = mlp(obs_dim + z_dim, hidden_dim, 'ntanh', feature_dim, 'irelu')
        self.policy = mlp(2 * feature_dim, hidden_dim, 'irelu', action_dim)
        if ortho_init:
            self.apply(weight_init)

    def forward(self, obs, z):   # returns the mean action mu = tanh(policy(h))
        h = torch.cat([self.obs_net(obs), self.obs_z_net(torch.cat([obs, z], dim=-1))], dim=-1)
        return torch.tanh(self.policy(h))


class ForwardMap(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim, ortho_init=True):
        super().__init__()
        self.obs_action_net = mlp(obs_dim + action_dim, hidden_dim, 'ntanh', feature_dim, 'irelu')
        self.obs_z_net = mlp(obs_dim + z_dim, hidden_dim, 'ntanh', feature_dim, 'irelu')
        self.F1 = mlp(2 * feature_dim, hidden_dim, 'irelu', z_dim)
        self.F2 = mlp(2 * feature_dim, hidden_dim, 'irelu', z_dim)
        if ortho_init:
            self.apply(weight_init)

    def forward(self, obs, z, action):
        h = torch.cat([self.obs_action_net(torch.cat([obs, action], dim=-1)), self.obs_z_net(torch.cat([obs, z], dim=-1))], dim=-1)
        return self.F1(h), self.F2(h)


class BackwardMap(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, ortho_init=True):
        super().__init__()
        self.z_dim = z_dim
        self.B = mlp(obs_dim, hidden_dim, 'ntanh', hidden_dim, 'relu', z_dim)
        if ortho_init:
            self.apply(weight_init)

    def forward(self, obs):
        return math.sqrt(self.z_dim) * F.normalize(self.B(obs), dim=1)


def noisy_action(mu, std, clip):
    """utils.TruncatedNormal.sample(clip): Gaussian noise (already scaled by std) clipped to +-clip; output clamped, straight-through."""
    x = mu + torch.clamp(torch.randn_like(mu) * std, -clip, clip)
    clamped = torch.clamp(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return x - x.detach() + clamped.detach()


# ───────────────────────────── agent ─────────────────────────────
class FBAgent:
    def __init__(self, c, obs_dim, action_dim, device):
        self.c, self.dev = c, device
        o = bool(c['ortho_init'])
        z, h, f = c['z_dim'], c['hidden_dim'], c['feature_dim']
        self.actor = Actor(obs_dim, z, action_dim, f, h, o).to(device)
        self.forward_net = ForwardMap(obs_dim, z, action_dim, f, h, o).to(device)
        self.backward_net = BackwardMap(obs_dim, z, c['backward_hidden_dim'], o).to(device)
        self.forward_target = ForwardMap(obs_dim, z, action_dim, f, h, o).to(device)
        self.backward_target = BackwardMap(obs_dim, z, c['backward_hidden_dim'], o).to(device)
        self.forward_target.load_state_dict(self.forward_net.state_dict())
        self.backward_target.load_state_dict(self.backward_net.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c['lr'])
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()}, {'params': self.backward_net.parameters()}], lr=c['lr'])
        bs = c['batch_size']
        self.off_diag = 1.0 - torch.eye(bs, device=device)       # float mask (no boolean indexing: it forces a host sync)
        self.n_off = float(bs * (bs - 1))
        self.eye_z = torch.eye(c['z_dim'], device=device)

    def sample_z(self, n):
        return math.sqrt(self.c['z_dim']) * F.normalize(torch.randn(n, self.c['z_dim'], device=self.dev), dim=1)

    def update(self, obs, action, next_obs, discount):
        c, bs = self.c, obs.shape[0]
        z = self.sample_z(bs)
        # mix: part of the z are B of (shuffled) states
        perm = torch.randperm(bs, device=self.dev)
        mix = (torch.rand(bs, 1, device=self.dev) < c['mix_ratio'])
        with torch.no_grad():
            mz = self.backward_net(obs[perm])     # B is already sqrt(d) * normalize(.), as the original's mix_z
        z = torch.where(mix, mz, z)

        # ── FB loss
        with torch.no_grad():
            next_action = noisy_action(self.actor(next_obs, z), c['stddev'], c['stddev_clip'])
            tF1, tF2 = self.forward_target(next_obs, z, next_action)
            tB = self.backward_target(next_obs)
            target_M = torch.min(tF1 @ tB.T, tF2 @ tB.T)
        F1, F2 = self.forward_net(obs, z, action)
        B = self.backward_net(next_obs)
        M1, M2 = F1 @ B.T, F2 @ B.T
        off, n_off = self.off_diag, self.n_off
        fb_offdiag = 0.5 * sum((((M - discount * target_M) ** 2) * off).sum() / n_off for M in (M1, M2))
        fb_diag = -sum(M.diag().mean() for M in (M1, M2))
        fb_loss = fb_offdiag + fb_diag
        q_loss = torch.zeros((), device=self.dev)
        if c['q_loss']:
            with torch.no_grad():
                next_Q = torch.min((tF1 * z).sum(1), (tF2 * z).sum(1))
                inv_cov = torch.inverse(B.T @ B / bs)
                target_Q = (B @ inv_cov * z).sum(1) + discount.squeeze(1) * next_Q
            q_loss = F.mse_loss((F1 * z).sum(1), target_Q) + F.mse_loss((F2 * z).sum(1), target_Q)
            fb_loss = fb_loss + c['q_loss_coef'] * q_loss
        Cov = B @ B.T
        orth = ((Cov ** 2) * off).sum() / n_off - 2 * Cov.diag().mean()
        fb_loss = fb_loss + c['ortho_coef'] * orth
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()

        # ── actor loss (uses the freshly updated forward net, as in the original)
        a = noisy_action(self.actor(obs, z), c['stddev'], c['stddev_clip'])
        aF1, aF2 = self.forward_net(obs, z, a)
        Q = torch.min((aF1 * z).sum(1), (aF2 * z).sum(1))
        actor_loss = -Q.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # ── targets
        with torch.no_grad():
            for n, t in ((self.forward_net, self.forward_target), (self.backward_net, self.backward_target)):
                for p, tp in zip(n.parameters(), t.parameters()):
                    tp.data.copy_(c['tau'] * p.data + (1 - c['tau']) * tp.data)
        eye_diff = B.detach().T @ B.detach() / bs - self.eye_z
        return torch.stack([fb_loss.detach(), fb_offdiag.detach(), fb_diag.detach(), q_loss.detach(), orth.detach(),
                            eye_diff.norm() / math.sqrt(B.shape[1]), M1.detach().mean(), target_M.mean(), Q.detach().mean()])

    METRICS = ['fb_loss', 'fb_offdiag', 'fb_diag', 'q_loss', 'orth_loss', 'orth_l2', 'M1', 'target_M', 'q']

    def state(self):
        return {k: getattr(self, k).state_dict() for k in ('actor', 'forward_net', 'backward_net', 'forward_target', 'backward_target',
                                                             'actor_opt', 'fb_opt')}

    def load(self, s):
        for k, v in s.items():
            getattr(self, k).load_state_dict(v)


# ───────────────────────────── data ─────────────────────────────
class Data:
    """All episodes on the GPU: obs (E, 1001, 17), action (E, 1001, 6); physics (E, 1001, 18) stays on the CPU (z inference only)."""

    def __init__(self, data_dir, max_episodes, device):
        fs = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        assert fs, f'no .npz in {data_dir}'
        if max_episodes:
            fs = fs[:max_episodes]
        obs, act, phy = [], [], []
        for i, f in enumerate(fs):
            e = np.load(f)
            obs.append(e['observation']); act.append(e['action']); phy.append(e['physics'])
            if i < 50:   # the original multiplies the discount stored in the episodes; for ExoRL cheetah it is 1 everywhere
                assert np.all(e['discount'] == 1.0), 'discount != 1 in an episode: handle it before using this script'
        self.obs = torch.from_numpy(np.stack(obs)).to(device)
        self.action = torch.from_numpy(np.stack(act)).to(device)
        self.physics = np.stack(phy)
        self.E, self.T1, self.obs_dim = self.obs.shape
        self.action_dim = self.action.shape[-1]
        self.L = self.T1 - 1    # transitions per episode
        self.dev = device
        print(f'data: {self.E} episodes x {self.L} transitions, obs {self.obs_dim}, action {self.action_dim}', flush=True)

    def sample(self, n):
        ep = torch.randint(0, self.E, (n,), device=self.dev)
        t = torch.randint(1, self.L + 1, (n,), device=self.dev)      # 1..L, like the original (step_idx)
        return self.obs[ep, t - 1], self.action[ep, t], self.obs[ep, t], (ep, t)

    def sample_next_states(self, n):
        """n random next-states (obs, physics) for z inference."""
        ep = np.random.randint(0, self.E, n)
        t = np.random.randint(1, self.L + 1, n)
        return self.obs[torch.from_numpy(ep).to(self.dev), torch.from_numpy(t).to(self.dev)], self.physics[ep, t]


# ───────────────────────────── cheetah tasks (custom_dmc_tasks/cheetah.py) ─────────────────────────────
TASKS = [('walk', 'speed', 2.0), ('walk_backward', 'speed', -2.0), ('run', 'speed', 10.0), ('run_backward', 'speed', -10.0),
         ('flip', 'flip', 5.0)]


def task_reward(kind, p, speed, angmom):
    from dm_control.utils import rewards
    if kind == 'flip':
        return rewards.tolerance(angmom, bounds=(5.0, float('inf')), margin=5.0, value_at_margin=0, sigmoid='linear')
    sign, v = (1.0 if p > 0 else -1.0), abs(p)
    return rewards.tolerance(sign * speed, bounds=(v, float('inf')), margin=v, value_at_margin=0, sigmoid='linear')


def physics_features(ph, states):
    """(speed, angular momentum about y) of the torso for each physics state, as the original relabelling (set_state + get_reward)."""
    sp, am = np.zeros(len(states)), np.zeros(len(states))
    for i, s in enumerate(states):
        with ph.reset_context():
            ph.set_state(s)
        sp[i] = ph.named.data.sensordata['torso_subtreelinvel'][0]
        am[i] = ph.named.data.subtree_angmom['torso'][1]
    return sp, am


def make_env(seed):
    from dm_control import suite
    return suite.load('cheetah', 'run', task_kwargs=dict(random=seed), environment_kwargs=dict(flat_observation=True))


def flat(ts):
    return np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in ts.observation.values()])


@torch.no_grad()
def evaluate(agent, data, c, n_episodes, seed0):
    """For each task: n_episodes episodes (run in lock-step), each with its own z inferred from num_inference fresh labelled next-states."""
    dev, res = agent.dev, {}
    helper = make_env(0).physics
    for name, kind, p in TASKS:
        zs = []
        for _ in range(n_episodes):
            obs, phy = data.sample_next_states(c['num_inference'])
            sp, am = physics_features(helper, phy)
            r = torch.as_tensor(task_reward(kind, p, sp, am), dtype=torch.float32, device=dev).reshape(-1, 1)
            B = agent.backward_net(obs)
            z = (r.T @ B) / r.shape[0]
            zs.append(math.sqrt(c['z_dim']) * F.normalize(z, dim=1))
        z = torch.cat(zs, 0)
        envs = [make_env(seed0 + 1000 * j) for j in range(n_episodes)]
        steps = [e.reset() for e in envs]
        total = np.zeros(n_episodes)
        while not steps[0].last():
            o = torch.as_tensor(np.stack([flat(s) for s in steps]), device=dev)
            a = agent.actor(o, z).cpu().numpy()      # the mean action, as the original in eval mode
            steps = [e.step(a[j]) for j, e in enumerate(envs)]
            for j, e in enumerate(envs):
                ph = e.physics
                total[j] += float(task_reward(kind, p, ph.named.data.sensordata['torso_subtreelinvel'][0], ph.named.data.subtree_angmom['torso'][1]))
        res[name] = (float(total.mean()), float(total.std()))
    return res


# ───────────────────────────── main ─────────────────────────────
def main():
    c = parse_kwargs(sys.argv[1:])
    assert c['data_dir'], 'data_dir= is required'
    torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f'{stamp}_FBORIG-cheetah'
    while os.path.exists(f'logs/{name}'):
        name += '_'
    logd, modd = f'logs/{name}', f'models/{name}'
    os.makedirs(logd); os.makedirs(modd)
    with open(f'{logd}/info.txt', 'w') as f:
        f.write(str(c) + '\n')
    print(logd, dev, flush=True)

    data = Data(c['data_dir'], c['max_episodes'], dev)
    agent = FBAgent(c, data.obs_dim, data.action_dim, dev)
    start = 0
    if c['resume']:
        ck = torch.load(c["resume"], map_location=dev, weights_only=False)
        agent.load(ck['agent']); start = ck['step']
        print('resumed at', start, flush=True)

    tlog_f = open(f'{logd}/training_log.csv', 'w')
    tlog = csv.writer(tlog_f)
    tlog.writerow(['step'] + FBAgent.METRICS + ['time'])
    elog_f = open(f'{logd}/eval_log.csv', 'w')
    elog = csv.writer(elog_f)
    names = [t[0] for t in TASKS]
    elog.writerow(['step', 'mean_over_tasks'] + [f'{n}_{s}' for n in names for s in ('mean', 'std')] + ['num_evals', 'tag', 'time'])

    def do_eval(step, n, tag):
        t0 = time.time()
        r = evaluate(agent, data, c, n, seed0=12 * n + step % 1000)
        row = [step, float(np.mean([r[k][0] for k in names]))] + [x for k in names for x in r[k]] + [n, tag, time.time() - t0]
        elog.writerow(row); elog_f.flush()
        print(f'[eval {tag}] step {step}: mean {row[1]:.0f} ' + ' '.join(f'{k}={r[k][0]:.0f}' for k in names), flush=True)

    def save_ckpt(step, fn):
        torch.save({'agent': agent.state(), 'step': step, 'cfg': c}, f'{modd}/{fn}')

    acc, t0 = torch.zeros(len(FBAgent.METRICS), device=dev), time.time()
    N = c['num_updates']
    for step in range(start + 1, N + 1):
        obs, action, next_obs, _ = data.sample(c['batch_size'])
        acc += agent.update(obs, action, next_obs, torch.full((c['batch_size'], 1), c['discount'], device=dev))
        if step % c['log_every'] == 0:
            tlog_last = round(time.time() - t0, 2)
            tlog.writerow([step] + [round(x, 5) for x in (acc / c['log_every']).tolist()] + [tlog_last])
            tlog_f.flush()
            if step % (10 * c['log_every']) == 0:
                print(f'step {step}  {tlog_last} s/1000 updates', flush=True)
            acc.zero_(); t0 = time.time()
        if step % c['eval_every'] == 0 and step < N:
            do_eval(step, c['num_evals_late'] if step > 0.9 * N else c['num_evals'], 'eval')
        if step % c['ckpt_every'] == 0:
            save_ckpt(step, 'ckpt_latest.pt')
    save_ckpt(N, 'ckpt_final.pt')
    do_eval(N, c['num_evals_final'], 'final')
    print('done', flush=True)


if __name__ == '__main__':
    main()
