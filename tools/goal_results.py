#!/usr/bin/env python3
"""Goal-reaching results (OGBench near-goal pairs goal_k=K and official tasks) from the synced run folders in hpc/logs.

    python tools/goal_results.py                 # markdown tables per env: method x K, cell = successes / n
    python tools/goal_results.py --runs          # also list every (method, K) cell with its run folder(s)
    python tools/goal_results.py --env antmaze_medium

Sources (all on the same pairs for a given env / dataset / K / goal_n / pair_seed, see utils/ogbench_tasks.sample_goal_pairs):
  * wm_goal.py   (logs/*_WMGOAL-*): latent-distance MPPI on a JEPA WM (src wm) or an FBWM checkpoint (src fbwm); eval_log.csv
  * fbwm.py      eval_ckpt + fbgoal_k / fbgoal_official: FB-goal cost (z = B(goal)); fbgoal_log.csv, one row per mode (mppi, policy)
  * wm_baselines (logs/*_WMBASE-*): state-based LeWM / DINO-WM with CEM; goal_eval.csv, one row per K
Only reads CSV and info.txt files; never touches the cluster.
"""
import argparse, ast, csv, glob, os, re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hpc', 'logs')


def kwargs_of(run):
    try:
        with open(os.path.join(run, 'info.txt')) as f:
            return ast.literal_eval(f.readline().strip())
    except Exception:
        return {}


def rows_of(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def ckpt_tag(ckpt):   # 'models/<run>/file.pickle' -> '<run>' timestamp + seed hint
    m = re.search(r'models/([^/]+)/', str(ckpt))
    return m.group(1) if m else str(ckpt)


def collect():
    out = []   # dicts: env, method, k (int, or 'official'), succ, n, gmin, gstart, run, ckpt
    for run in sorted(glob.glob(os.path.join(ROOT, '*_WMGOAL-*'))):
        kw = kwargs_of(run)
        r = rows_of(os.path.join(run, 'eval_log.csv'))
        if not r or 'goal_k' not in kw or int(kw.get('goal_k', 0)) == 0 or not os.path.exists(os.path.join(run, 'goal_pairs.csv')):
            continue   # (no goal_pairs.csv: the earlier one-fixed-pair goal_k version, not comparable)
        k = int(kw['goal_k'])
        last = r[-1]
        col = f'goal_k{k}_mean'
        if col not in last:
            continue
        src = 'FBWM' if 'FBWM' in str(kw.get('ckpt')) else 'JEPA-WM'
        h = kw.get('horizon', 5)
        norm = str(kw.get('mppi_norm', 'false')).lower() in ('1', 'true')
        n = int(last.get('num_evals') or 50)
        out.append(dict(env=kw['env_name'], method=f'{src} distance MPPI H={h}{" norm" if norm else ""}', k=k,
                        succ=round(float(last[col]) * n), n=n, gmin=float(last.get(f'goal_k{k}_gdist_min') or 'nan'),
                        gstart=float(last.get(f'goal_k{k}_gdist_start') or 'nan'), run=os.path.basename(run), ckpt=ckpt_tag(kw.get('ckpt'))))
    for run in sorted(glob.glob(os.path.join(ROOT, '*_FBWM-*'))):
        rr = rows_of(os.path.join(run, 'fbgoal_log.csv'))
        if not rr:
            continue
        kw = kwargs_of(run)
        tags = []
        if str(kw.get('mppi_init_policy', 0)) in ('1', 'True', 'true'):
            tags.append('init-pol')
        if str(kw.get('mppi_norm', 0)) in ('1', 'True', 'true'):
            tags.append('norm')
        if str(kw.get('mppi_root_check', 0)) in ('1', 'True', 'true'):
            tags.append('root')
        for v in ('mppi_terminal_only', ):
            if str(kw.get(v, 0)) in ('1', 'True', 'true'):
                tags.append('terminal-only')
        if kw.get('mppi_reward') == 'mean':
            tags.append('mean-r')
        if 'horizon' in kw:
            tags.append(f"H={kw['horizon']}")
        for x in rr:
            if x.get('task', '') not in ('', 'official') and not x['task'].startswith('goal_k'):
                continue   # per-task rows of the official eval: only the 'official' total here
            k = 'official' if x.get('task') == 'official' else int(x['goal_k'])
            n = int(x.get('n') or x.get('goal_n') or 50)
            method = 'FBWM FB-goal actor' if x['mode'] == 'policy' else f"FBWM FB-goal MPPI ({' '.join(tags) or 'plain'})"
            out.append(dict(env=kw['env_name'], method=method, k=k, succ=round(float(x['success']) * n), n=n,
                            gmin=float(x['gdist_min']), gstart=float(x['gdist_start']), run=os.path.basename(run),
                            ckpt=ckpt_tag(kw.get('eval_ckpt'))))
    for run in sorted(glob.glob(os.path.join(ROOT, '*_WMBASE-*'))):
        kw = kwargs_of(run)
        if 'smoke' in str(kw.get('name', '')):
            continue
        for x in rows_of(os.path.join(run, 'goal_eval.csv')):
            n = int(x['n'])
            out.append(dict(env=kw['env_name'], method={'lewm': 'LeWM (state) CEM', 'dinowm': 'DINO-WM (state) CEM', 'lewm-mppi': f"LeWM (state) distance MPPI H={5 * int(kw.get('plan_horizon', 5))} norm",
                                                'dinowm-mppi': f"DINO-WM (state) distance MPPI H={5 * int(kw.get('plan_horizon', 5))} norm"}[x['model']],
                            k=int(x['goal_k']), succ=round(float(x['success']) * n), n=n, gmin=float(x['gdist_min']),
                            gstart=float(x['gdist_start']), run=os.path.basename(run), ckpt=os.path.basename(run)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default=None)
    ap.add_argument('--runs', action='store_true')
    a = ap.parse_args()
    rows = collect()
    envs = sorted({r['env'] for r in rows if a.env in (None, r['env'])})
    for env in envs:
        R = [r for r in rows if r['env'] == env]
        ks = sorted({r['k'] for r in R if r['k'] != 'official'}) + (['official'] if any(r['k'] == 'official' for r in R) else [])
        cells = defaultdict(list)
        for r in R:
            cells[(r['method'], r['k'])].append(r)
        methods = sorted({r['method'] for r in R})
        print(f'\n### {env}  (successes / episodes; several checkpoints are pooled; gain = start - closest goal distance)\n')
        print('| method | ' + ' | '.join(f'K={k}' if k != 'official' else 'official' for k in ks) + ' |')
        print('|---' * (len(ks) + 1) + '|')
        for m in methods:
            line = []
            for k in ks:
                c = cells.get((m, k))
                if not c:
                    line.append('')
                    continue
                s, n = sum(x['succ'] for x in c), sum(x['n'] for x in c)
                gain = sum(x['gstart'] - x['gmin'] for x in c) / len(c)
                line.append(f"{s}/{n} ({gain:.2f})" + (f" [{len({x['ckpt'] for x in c})} ckpt]" if len({x['ckpt'] for x in c}) > 1 else ''))
            print(f'| {m} | ' + ' | '.join(line) + ' |')
        if a.runs:
            for (m, k), c in sorted(cells.items(), key=lambda t: (t[0][0], str(t[0][1]))):
                print(f'  {m} K={k}: ' + ', '.join(f"{x['run']} (ckpt {x['ckpt']})" for x in c))


if __name__ == '__main__':
    main()
