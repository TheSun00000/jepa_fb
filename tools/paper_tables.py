#!/usr/bin/env python3
"""The paper's main table and the label-budget figure, built from the synced logs in hpc/logs (2026-11-15).

    python tools/paper_tables.py      # writes hpc/status/paper_main_table.json and hpc/status/paper_label_curve.json

They go to the tracker database (collection `paper`, documents `main_table`, `label_curve`; `task_table` since 2026-11-17: the per-task
version of the main table, hpc/status/paper_task_table.json), shown in the overview tab.

Protocol of the main table (N = 10k labelled states): mean over the environment's tasks of the final evaluation, then mean and
95% CI (t-interval) over seeds. One fbwm configuration everywhere: gphi, z_dim 50, MPPI with the lstsq per-step reward and
Cov(B) from 100k dataset states (`mppi_reward=lstsq z_cov=dataset`), z from N labelled states, 20 episodes per task.
Each cell records where its runs come from and whether they deviate from the protocol (`note`).
"""
import ast, csv, glob, json, math, os, statistics as st
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
LOGS = os.path.join(ROOT, 'hpc', 'logs')
OUT = os.path.join(ROOT, 'hpc', 'status')
ENVS = ['walker', 'cheetah', 'quadruped']   # antmaze left out for now (user, 2026-11-15)
TASKS = {'walker': ['stand', 'walk', 'vel8', 'run', 'flip'], 'cheetah': ['vel10Back', 'vel2Back', 'vel2', 'vel10', 'flip'],
         'quadruped': ['stand', 'walk', 'run', 'jump']}
NS = [100, 300, 1000, 3000, 10000]
PAPER_CONS = '5'   # the paper configuration (2026-11-15, revised 23:20): gphi, z_dim 50, consistency_coef 5, lstsq, dataset Cov, root-check MPPI
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def rows(d):
    try:
        return list(csv.DictReader(open(os.path.join(d, 'eval_log.csv'))))
    except OSError:
        return []


def info(d):
    try:
        return ast.literal_eval(open(os.path.join(d, 'info.txt')).readline())
    except Exception:
        return None


def runs(pattern):
    for d in sorted(glob.glob(os.path.join(LOGS, pattern))):
        kw = info(d)
        if kw is not None:
            yield os.path.basename(d), kw, rows(d)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean_ci(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = st.mean(vals)
    ci = T95.get(len(vals), 2.0) * st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
    return {'m': round(m, 1), 'ci': None if ci is None else round(ci, 1), 'n': len(vals)}


# ── per-run scores: {(method, env, seed, N): (score, folder)} ──────────────────────────────────────────────────────
S = {}
TS = {}   # same keys as S: {task: score} of the chosen run, for the per-task table
NOTES = defaultdict(set)


def put(method, env, seed, n, score, folder, note=None, per_task=None):
    if score is None:
        return
    S[(method, env, int(seed), int(n))] = (score, folder)
    if per_task:
        TS[(method, env, int(seed), int(n))] = per_task
    if note:
        NOTES[(method, env, int(n))].add(note)


def row_tasks(row, prefix, env):
    """{task: score} from an eval_log row (columns <prefix><task>_mean)."""
    return {t: f(row.get(f'{prefix}{t}_mean')) for t in TASKS[env]}


# toplines: one run per (env, task, seed), final eval, mean over tasks per seed
for fam, method in (('TDMPC', 'tdmpc'), ('TD3', 'td3')):
    per = defaultdict(dict)
    for folder, kw, ev in runs(f'2026-11-1[45]_*_{fam}-*'):
        if 'num_outer_steps' in kw or 'mode' in kw or kw.get('env_name') not in TASKS:   # 'mode': tdmpc_adapt.py runs
            continue
        fin = [r for r in ev if r.get('tag') == 'final']
        if fin:
            per[(kw['env_name'], kw.get('seed', 1))][kw['task']] = (f(fin[-1]['return_mean']), folder)
            if fam == 'TDMPC' and fin[-1].get('policy_return_mean'):
                per[('pol:' + kw['env_name'], kw.get('seed', 1))][kw['task']] = (f(fin[-1]['policy_return_mean']), folder)
    for (env, seed), tasks in per.items():
        m = method
        if env.startswith('pol:'):
            env, m = env[4:], 'tdmpc_policy'
        if all(t in tasks for t in TASKS[env]):
            put(m, env, seed, 10000, st.mean(tasks[t][0] for t in TASKS[env]), tasks[TASKS[env][0]][1],
                per_task={t: tasks[t][0] for t in TASKS[env]})

# adaptation baselines (frozen reward-free JEPA world model, 6k Q updates): full dataset / labels only + imagined rollouts
for fam, method in (('TDMPC-adapt', 'adapt_full'), ('TDMPC-adaptLM', 'adapt_labels')):
    per = defaultdict(dict)
    for folder, kw, ev in runs(f'2026-11-1[4-9]_*_{fam}-*'):   # 11-18: quadruped N sweep (N = 100..3000)
        if str(kw.get('adapt_steps')) != '6000':
            continue
        if fam == 'TDMPC-adapt' and kw.get('q_data') not in (None, 'dataset'):
            continue
        fin = [r for r in ev if r.get('tag') == 'final']
        if fin:
            per[(kw['env_name'], kw.get('seed', 1), int(kw['n_labels']))][kw['task']] = (f(fin[-1]['return_mean']), folder)
    for (env, seed, n), tasks in per.items():
        if all(t in tasks for t in TASKS.get(env, [])):
            put(method, env, seed, n, st.mean(tasks[t][0] for t in TASKS[env]), tasks[TASKS[env][0]][1],
                per_task={t: tasks[t][0] for t in TASKS[env]})

# zero-shot, label sweep (eval-only, 20 episodes): fbwm (z50 + lstsq + dataset Cov, root-check MPPI preferred) and FB (500k preferred)
LATE = defaultdict(list)
for folder, kw, ev in runs('2026-11-1[5-9]_*'):
    if 'eval_ckpt' not in kw or str(kw.get('num_evals_final')) != '20' or 'z_labels' not in kw:
        continue
    r = [x for x in ev if x.get('tag') == 'zlabels']
    if not r:
        continue
    r, env, n = r[0], kw['env_name'], int(kw['z_labels'])
    if '_FB-' in folder:
        old_ckpt = any(t in kw['eval_ckpt'] for t in ('2026-11-13', '2026-11-14'))   # 1M-update walker checkpoints
        if env == 'cheetah' and 'step:500000' in kw['eval_ckpt'] and kw.get('label_align') == 1:   # 2026-11-18: corrected fb_jax, 500k checkpoint of a 1M run, 10000 episodes
            LATE['fb_new'].append(('fb', env, kw['seed'], n, f(r['mean_over_tasks']), folder, row_tasks(r, '', env)))
            continue
        LATE['fb_old' if old_ckpt else 'fb'].append(('fb', env, kw['seed'], n, f(r['mean_over_tasks']), folder, row_tasks(r, '', env)))
    elif (kw.get('variant') == 'gphi' and kw.get('mppi_reward') == 'lstsq' and kw.get('z_cov') == 'dataset' and 'z_dim' not in kw
          and str(kw.get('consistency_coef')) == PAPER_CONS and 'FBWMBTD' not in folder
          and not any(k in kw for k in ('horizon', 'mppi_init_policy', 'mixture_coef', 'mppi_terminal_only', 'mppi_norm', 'z_ridge', 'num_samples', 'iterations', 'num_elites', 'temperature', 'min_std', 'momentum', 'mppi_init_std', 'diag_profile', 'diag_wm'))):
        tier = 'root' if str(kw.get('mppi_root_check')) in ('1', 'True', 'true') else 'noroot'
        if env == 'cheetah' and 'step:500000' in kw['eval_ckpt'] and str(kw.get('target_mean')) == '1' and tier == 'root':
            tier = 'tdj'   # 2026-11-18: fbwm with the TD-JEPA FB changes (decisions.md), 500k checkpoint of a 1M run
        LATE[tier].append(('fbwm', env, kw['seed'], n, f(r['mean_over_tasks']), folder, row_tasks(r, '', env)))
        LATE[tier].append(('fbwm_policy', env, kw['seed'], n, f(r['policy_mean_over_tasks']), folder, row_tasks(r, 'policy_', env)))

TDJ_NOTE = '500k checkpoint of a 1M run, TD-JEPA FB changes (tau 0.001, discount 0.98, no Q loss, B 256, mean target, mix next, half ortho)'
FB_NEW_NOTE = '2 seeds; corrected fb_jax (discount 0.98, Q loss 1/d, no phi MLP), 10000 episodes; 500k checkpoint of a 1M run; no planner, policy return'
FB_NEW_ENVS = {e for (m, e, s, n, sc, fo, pt) in LATE['fb_new']}   # for these envs the new FB replaces the older sweeps and fallbacks entirely
for tier, note in (('tdj', TDJ_NOTE), ('fb_new', FB_NEW_NOTE), ('root', None), ('noroot', 'MPPI without the root check'), ('fb', None)):
    for (m, env, seed, n, score, folder, pt) in LATE[tier]:
        if tier in ('fb', 'fb_old') and env in FB_NEW_ENVS:
            continue
        if (m, env, int(seed), int(n)) not in S:
            put(m, env, seed, n, score, folder, note, pt)

# fallbacks for the main table only (N = 10k) where no eval-only sweep exists yet: training-run finals, flagged
for folder, kw, ev in runs('2026-11-15_*'):
    env = kw.get('env_name')
    if 'eval_ckpt' in kw or str(kw.get('num_evals_final')) != '20' or 'z_dim' in kw or env not in TASKS:
        continue
    fin = [x for x in ev if x.get('tag') == 'final']
    if not fin:
        continue
    fin, seed = fin[-1], kw.get('seed', 1)
    if '_FB-' in folder and env not in FB_NEW_ENVS and ('fb', env, int(seed), 10000) not in S:
        put('fb', env, seed, 10000, f(fin['mean_over_tasks']), folder, 'training-run final (no z sweep yet)', row_tasks(fin, '', env))
    if ('FBWM-gphi' in folder and kw.get('mppi_reward') == 'lstsq' and not kw.get('fcons_coef') and str(kw.get('consistency_coef')) == PAPER_CONS
            and not any(k in kw for k in ('hidden_dim', 'mix_ratio', 'discount', 'batch_size', 'lr', 'tau', 'q_loss_coef', 'b_ortho_coef'))
            and ('fbwm', env, int(seed), 10000) not in S):
        put('fbwm', env, seed, 10000, f(fin['mean_over_tasks']), folder, 'training-run final' + ('' if str(kw.get('mppi_root_check')) in ('1', 'True') else ', no root check') + ('' if kw.get('z_cov') == 'dataset' else ', Cov(B) from the labels'), row_tasks(fin, '', env))
        put('fbwm_policy', env, seed, 10000, f(fin['policy_mean_over_tasks']), folder, 'training-run final', row_tasks(fin, 'policy_', env))

# last resort: the 1M-update walker FB checkpoints
for (m, env, seed, n, score, folder, pt) in LATE['fb_old']:
    if env not in FB_NEW_ENVS and (m, env, int(seed), int(n)) not in S:
        put(m, env, seed, n, score, folder, 'FB checkpoints: 1M updates', pt)

# ── main table ─────────────────────────────────────────────────────────────────────────────────────────────────────
ROWS = [
    ('toplines: true reward, trained per task', [('tdmpc', 'TD-MPC (MPPI)', 'per task, ~35 min'),
                                                 ('tdmpc_policy', 'TD-MPC (policy)', 'per task, ~35 min'),
                                                 ('td3', 'offline TD3', 'per task, ~8 min')]),
    ('adaptation: reward-free world model + N labels', [('adapt_full', 'TD-MPC heads, full dataset', 'per task, ~20 s'),
                                                        ('adapt_labels', 'TD-MPC heads, labels only (Dyna)', 'per task, ~20 s')]),
    ('zero-shot: no training per task', [('fb', 'FB', 'none'), ('fbwm_policy', 'fbwm, policy', 'none'), ('fbwm', 'fbwm, MPPI (ours)', 'none')]),
]
table = {'title': 'Main results, N = 10k labelled states',
         'protocol': ('Mean over the environment\'s tasks of the final evaluation (20 episodes for zero-shot methods, 10 for the toplines '
                      'and adapted baselines), then mean ± 95% CI over seeds (n = seeds). fbwm: gphi, z_dim 50, consistency 5, MPPI with the lstsq '
                      'per-step reward, Cov(B) from 100k dataset states and the root check (never below the policy by F at the real state). 500k updates unless noted.'),
         'envs': ENVS, 'groups': []}
for gname, items in ROWS:
    g = {'name': gname, 'rows': []}
    for key, label, training in items:
        cells = {}
        for env in ENVS:
            vals = [S[k][0] for k in S if k[0] == key and k[1] == env and k[3] == 10000]
            c = mean_ci(vals)
            if c:
                if NOTES.get((key, env, 10000)):
                    c['note'] = '; '.join(sorted(NOTES[(key, env, 10000)]))
                cells[env] = c
        g['rows'].append({'key': key, 'label': label, 'training': training, 'ours': key == 'fbwm', 'cells': cells})
    table['groups'].append(g)

# ── per-task table (N = 10k): the same chosen runs as the main table, one column per (env, task), mean ± 95% CI over seeds ─────────────
TASK_ROWS = [('tdmpc', 'TD-MPC (MPPI)', 'per task, ~35 min'), ('td3', 'offline TD3', 'per task, ~10 min'),
             ('adapt_full', 'TD-MPC heads, full dataset', 'per task, ~20 s'), ('fb', 'FB', 'none'), ('fbwm_policy', 'fbwm, policy', 'none'),
             ('fbwm', 'fbwm, MPPI (ours)', 'none')]
task_table = {'title': 'Per-task results, N = 10k labelled states',
              'protocol': ('One column per task, not averaged. Mean ± 95% CI over seeds (n = seeds) of the final evaluation of each run '
                           '(20 episodes for zero-shot methods, 10 for the others); the runs are the ones behind the main table.'),
              'columns': [{'env': e, 'task': t} for e in ENVS for t in TASKS[e]], 'rows': []}
for key, label, training in TASK_ROWS:
    cells = {}
    for e in ENVS:
        for t in TASKS[e]:
            vals = [TS[k][t] for k in TS if k[0] == key and k[1] == e and k[3] == 10000 and TS[k].get(t) is not None]
            c = mean_ci(vals)
            if c:
                cells[f'{e}/{t}'] = c
    task_table['rows'].append({'key': key, 'label': label, 'training': training, 'ours': key == 'fbwm', 'cells': cells})

# ── label-budget figure: mean over envs of the per-env mean over seeds ─────────────────────────────────────────────
SERIES = [('fbwm', 'fbwm, MPPI (ours)'), ('fbwm_policy', 'fbwm, policy'), ('fb', 'FB'),
          ('adapt_full', 'adapted, full dataset'), ('adapt_labels', 'adapted, labels only')]
CURVE_ENVS = ['walker', 'cheetah', 'quadruped']   # quadruped added 2026-11-18 for the all / walker / cheetah / quadruped buttons of the figure


def env_mean(key, env, n):
    v = [S[k][0] for k in S if k[0] == key and k[1] == env and k[3] == n]
    return (st.mean(v), len(v)) if v else None


curve = {'title': 'Effect of the number of labelled states N', 'Ns': NS, 'envs': CURVE_ENVS, 'series': [], 'refs': [],
         'note': ('Mean over environments of the per-environment mean over seeds (and tasks). A point that lacks an environment is '
                  'marked partial. Dashed lines: per-task toplines trained on the true reward of the whole dataset (independent of N).')}
for key, label in SERIES:
    pts = []
    for n in NS:
        per = {e: env_mean(key, e, n) for e in CURVE_ENVS}
        have = {e: v for e, v in per.items() if v}
        if have:
            pts.append({'N': n, 'm': round(st.mean(v[0] for v in have.values()), 1), 'envs': sorted(have),
                        'seeds': {e: v[1] for e, v in have.items()}, 'per_env': {e: round(v[0], 1) for e, v in have.items()}})
    curve['series'].append({'key': key, 'label': label, 'points': pts})
for key, label in (('tdmpc', 'TD-MPC'), ('td3', 'offline TD3')):
    per = {e: env_mean(key, e, 10000) for e in CURVE_ENVS}
    have = {e: v for e, v in per.items() if v}
    if have:
        curve['refs'].append({'key': key, 'label': label, 'm': round(st.mean(v[0] for v in have.values()), 1), 'envs': sorted(have),
                              'per_env': {e: round(v[0], 1) for e, v in have.items()}})

import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
table['updated'] = curve['updated'] = task_table['updated'] = now
os.makedirs(OUT, exist_ok=True)
json.dump(table, open(os.path.join(OUT, 'paper_main_table.json'), 'w'), indent=1)
json.dump(curve, open(os.path.join(OUT, 'paper_label_curve.json'), 'w'), indent=1)
json.dump(task_table, open(os.path.join(OUT, 'paper_task_table.json'), 'w'), indent=1)
print('main table:', {r['key']: {e: c['m'] for e, c in r['cells'].items()} for g in table['groups'] for r in g['rows']})
print('curve:', {s['key']: [(p['N'], p['m'], ''.join(e[0] for e in p['envs'])) for p in s['points']] for s in curve['series']},
      '| refs:', [(r['label'], r['m'], r['envs']) for r in curve['refs']])
