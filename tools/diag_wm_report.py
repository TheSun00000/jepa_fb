#!/usr/bin/env python3
"""Summarise the diag_wm.csv files written by `fbwm.py eval_ckpt=... diag_wm=1` (open-loop world-model rollouts vs the simulator).

    python tools/diag_wm_report.py [--logs hpc/logs] [--since 2026-11-16] [--out hpc/status/diag_wm.json]

Per env and source (dataset | policy): mean over seeds and tasks of each metric at rollout steps t = 1, 5, 10, 25, 50, 100.
Metrics are described in fbwm.py (search DIAG_WM). Lower is better for *_nmse, *_mae, r_nrmse; higher for *_cos, r_corr_*.
"""
import argparse, ast, csv, glob, json, os
from collections import defaultdict
import numpy as np

STEPS = (1, 5, 10, 25, 50, 100)
METRICS = ['lat_cos', 'lat_nmse', 'copy_nmse', 'b_cos', 'r_corr_imag', 'r_corr_real', 'r_nrmse', 'obs_nmse', 'obs_nmse_real']

ap = argparse.ArgumentParser()
ap.add_argument('--logs', default='hpc/logs')
ap.add_argument('--since', default='2026-11-16')
ap.add_argument('--out', default=None)
a = ap.parse_args()

acc = defaultdict(lambda: defaultdict(list))   # (env, source, t) -> metric -> [per-seed means]
runs = defaultdict(list)
for path in sorted(glob.glob(os.path.join(a.logs, '*', 'diag_wm.csv'))):
    folder = os.path.basename(os.path.dirname(path))
    if folder < a.since:
        continue
    kw = ast.literal_eval(open(os.path.join(os.path.dirname(path), 'info.txt')).readline())
    env, seed = kw['env_name'], kw.get('seed')
    runs[env].append((seed, folder))
    rows = list(csv.DictReader(open(path)))
    aux_cols = [c for c in rows[0] if c.endswith('_mae') or c.endswith('_mae_real')]
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t = int(r['t'])
        if t in STEPS:
            for m in METRICS + aux_cols:
                if r[m] != '':
                    cell[(r['source'], t)][m].append(float(r[m]))
    for (src, t), d in cell.items():
        for m, v in d.items():
            acc[(env, src, t)][m].append(float(np.mean(v)))

out = {}
for env in ('walker', 'cheetah', 'quadruped'):
    if env not in runs:
        continue
    print(f'\n=== {env}: {len(runs[env])} seeds {sorted(s for s, _ in runs[env])}')
    for src in ('dataset', 'policy'):
        print(f'-- {src}')
        print('t     ' + ' '.join(f'{m:>13s}' for m in METRICS))
        for t in STEPS:
            d = acc.get((env, src, t))
            if not d:
                continue
            print(f'{t:<5d} ' + ' '.join(f'{np.mean(d[m]):13.3f}' for m in METRICS))
            out.setdefault(env, {}).setdefault(src, {})[t] = {m: float(np.mean(v)) for m, v in d.items()}
        aux = [m for m in (acc.get((env, src, STEPS[-1])) or {}) if m.endswith('_mae')]
        for m in aux:
            print(f'   {m}: ' + '  '.join(f't={t}: {np.mean(acc[(env, src, t)][m]):.3f} (probe floor {np.mean(acc[(env, src, t)][m + "_real"]):.3f})' for t in STEPS if (env, src, t) in acc))
if a.out:
    json.dump({'runs': runs, 'steps': out}, open(a.out, 'w'), indent=1)
    print('wrote', a.out)
