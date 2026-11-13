#!/usr/bin/env python3
"""Index of every run folder synced into hpc/logs (run tools/sync_from_hpc.sh first): what ran, with which flags, and its headline numbers.

fb_jepa is run-centric: most runs (2026-09-29 -> 10-06) predate any job tracking, so the tracker lists RUNS (one per log folder),
and jobs only link to them. One record per folder:
    folder, date, family (the exp_name without the env: FB, FB-phi, TDMPC, TDMPC-SF, TDMPC-SF-FB, TDMPC-FB-sepB, TDMPC-FB-gphi, ...),
    env, task (single-task TD-MPC only), flags (info.txt kwargs minus env_name / eval_mode), variant (the flags as one string),
    mode ('both' = MPPI and policy columns, 'policy', 'mppi'), nEvals, lastStep, complete (lastStep >= 1M),
    final = mean of the last 10 evals {mppi, policy}, best (max single eval), tasks {task: {mppi, policy}} (last 10 evals), corr.
Eval-log formats:
    multi-task scripts (fb_jax, fb_modified, tdmpc_sf, tdmpc_fb*): mean_over_tasks, <task>_mean/_std, optional policy_* columns
        (eval_mode=both: unprefixed = MPPI, policy_ = policy), <task>_corr.
    tdmpc_jax.py (one task, true reward): return_mean (MPPI), policy_return_mean.

    tools/runs_index.py            # writes hpc/status/runs_<n>.json and curves_<n>.json (tracker shards, runs since TRACK_FROM only)
    tools/runs_index.py --all      # prints every synced run (old ones included) as JSON, for local analysis
"""
import ast
import csv
import json
import math
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LOGS = os.path.join(ROOT, 'hpc', 'logs')
OUT = os.path.join(ROOT, 'hpc', 'status')
SHARD_BYTES = 200_000            # a tracker database document is limited to 262144 bytes of JSON
FULL_STEPS = 1_000_000
LAST_K = 10                      # the "final" number = mean over the last LAST_K evals (single evals swing by +-150)
CURVE_POINTS = 60
IGNORE_FLAGS = ('env_name', 'eval_mode')
# The tracker starts empty: runs before this date (2026-09-29 -> 10-06) stay local in hpc/ and are never pushed (user, 2026-11-13).
TRACK_FROM = '2026-11-13'
POLICY_ONLY_FAMILIES = ('FB', 'FB-phi')     # scripts that only have a policy (no world model)
ENVS = ('cheetah', 'walker', 'quadruped', 'antmaze')


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def read_csv(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None


def read_info(folder):
    """(kwargs dict, config line) from info.txt: line 1 is the command-line kwargs, line 2 the config dataclass."""
    try:
        lines = open(os.path.join(LOGS, folder, 'info.txt')).read().splitlines()
    except OSError:
        return {}, ''
    kw = {}
    if lines:
        try:
            kw = ast.literal_eval(lines[0])
        except (ValueError, SyntaxError):
            kw = {}
    return kw if isinstance(kw, dict) else {}, (lines[1] if len(lines) > 1 else '')


def parse_folder(folder):
    """'2026-10-05_15-17-36_TDMPC-FB-gphi-cheetah_' -> (date, time, name without env, env guess)."""
    m = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(.*)$', folder)
    if not m:
        return None, None, folder, None
    name = m.group(3).rstrip('_')
    env = None
    for e in ENVS:
        if name.endswith('-' + e) or ('-' + e + '-') in name:
            env = e
    fam = name
    if env:
        fam = re.sub(r'-%s(-.*)?$' % env, '', name)
    return m.group(1), m.group(2).replace('-', ':'), fam, env


def fmt_val(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def index_run(folder):
    date, time, family, env_guess = parse_folder(folder)
    ev = read_csv(os.path.join(LOGS, folder, 'eval_log.csv'))
    kw, cfg = read_info(folder)
    rec = {'folder': folder, 'date': date, 'time': time, 'family': family, 'env': kw.get('env_name') or env_guess}
    flags = {k: fmt_val(v) for k, v in kw.items() if k not in IGNORE_FLAGS}
    if 'task' in flags:
        rec['task'] = flags['task']
    rec['flags'] = flags
    rec['variant'] = ' '.join('%s=%s' % kv for kv in sorted(flags.items()) if kv[0] not in ('task', 'seed')) or 'default'
    seed = flags.get('seed')
    if seed is None:
        m = re.search(r'seed=(\d+)', cfg)
        seed = m.group(1) if m else None
    rec['seed'] = seed
    if not ev:
        rec.update(nEvals=0, lastStep=0, complete=False)
        return rec, None
    cols = ev[0].keys()
    steps = [int(float(r['step'])) for r in ev if num(r.get('step')) is not None]
    rec['nEvals'], rec['lastStep'] = len(ev), (steps[-1] if steps else 0)
    rec['complete'] = rec['lastStep'] >= FULL_STEPS
    last = ev[-LAST_K:]

    if 'return_mean' in cols:                                  # tdmpc_jax.py, one task, true reward
        rec['mode'] = 'both' if 'policy_return_mean' in cols else 'mppi'
        m_col, p_col, tasks = 'return_mean', ('policy_return_mean' if rec['mode'] == 'both' else None), []
    elif 'mean_over_tasks' in cols:
        has_pol = 'policy_mean_over_tasks' in cols
        if has_pol:
            rec['mode'] = 'both'
        elif family in POLICY_ONLY_FAMILIES or 'identity' in family:
            rec['mode'] = 'policy'          # FB scripts; gphi-identity forced eval_mode=policy (no world model) in that version
        else:
            rec['mode'] = 'mppi'
        tasks = [c[:-5] for c in cols if c.endswith('_mean') and not c.startswith('policy_') and c != 'mean_over_tasks']
        m_col = 'mean_over_tasks'
        p_col = 'policy_mean_over_tasks' if has_pol else None
    else:
        rec['mode'] = 'unknown'
        return rec, None

    def pick(col_m, col_p):
        """{mppi, policy} from the columns, honouring the mode (a policy-only log stores the policy in the unprefixed column)."""
        vm = mean(num(r.get(col_m)) for r in last) if col_m else None
        vp = mean(num(r.get(col_p)) for r in last) if col_p else None
        if rec['mode'] == 'policy':
            return {'mppi': None, 'policy': vm}
        return {'mppi': vm, 'policy': vp}

    rec['final'] = pick(m_col, p_col)
    allm = [num(r.get(m_col)) for r in ev]
    allp = [num(r.get(p_col)) for r in ev] if p_col else []
    bm = max([x for x in allm if x is not None], default=None)
    bp = max([x for x in allp if x is not None], default=None)
    rec['best'] = {'mppi': None, 'policy': bm} if rec['mode'] == 'policy' else {'mppi': bm, 'policy': bp}
    rec['tasks'] = {t: pick(t + '_mean', ('policy_%s_mean' % t) if p_col else None) for t in tasks}
    if tasks and any((t + '_corr') in cols for t in tasks):
        rec['corr'] = {t: mean(num(r.get(t + '_corr')) for r in last) for t in tasks}

    stride = max(1, math.ceil(len(ev) / CURVE_POINTS))
    sel = ev[::stride] + ([] if (len(ev) - 1) % stride == 0 else [ev[-1]])
    curve = {'s': [int(float(r['step'])) for r in sel]}
    vm = [num(r.get(m_col)) for r in sel]
    vp = [num(r.get(p_col)) for r in sel] if p_col else None
    if rec['mode'] == 'policy':
        curve['p'] = [None if v is None else round(v) for v in vm]
    else:
        curve['m'] = [None if v is None else round(v) for v in vm]
        if vp:
            curve['p'] = [None if v is None else round(v) for v in vp]
    return rec, curve


def shards(items, key, extra=None):
    out, cur = [], {}
    for k in sorted(items):
        trial = dict(cur, **{k: items[k]})
        if cur and len(json.dumps(dict(extra or {}, **{key: trial}), separators=(',', ':'))) > SHARD_BYTES:
            out.append(cur)
            cur = {}
        cur[k] = items[k]
    out.append(cur)
    return out


def write_shards(prefix, key, items, extra=None):
    os.makedirs(OUT, exist_ok=True)
    for old in [f for f in os.listdir(OUT) if re.fullmatch(r'%s_\d+\.json' % prefix, f)]:
        os.remove(os.path.join(OUT, old))
    parts = shards(items, key, extra)
    for i, sh in enumerate(parts):
        with open(os.path.join(OUT, '%s_%d.json' % (prefix, i)), 'w') as f:
            json.dump(dict(extra or {}, **{key: sh}), f, separators=(',', ':'))
    return len(parts)


def build_all(since=TRACK_FROM):
    """Runs whose folder date is >= since (default TRACK_FROM: what the tracker gets). since=None: every synced run."""
    runs, curves = {}, {}
    if not os.path.isdir(LOGS):
        return runs, curves
    for folder in sorted(os.listdir(LOGS)):
        if not os.path.isdir(os.path.join(LOGS, folder)):
            continue
        if since and (parse_folder(folder)[0] or '') < since:
            continue
        rec, curve = index_run(folder)
        runs[folder] = rec
        if curve:
            curves[folder] = curve
    return runs, curves


def main():
    if '--all' in sys.argv:          # every synced run, for local analysis; nothing is written
        runs, _ = build_all(since=None)
        print(json.dumps(runs, indent=1))
        return
    runs, curves = build_all()
    n1 = write_shards('runs', 'runs', runs)
    n2 = write_shards('curves', 'curves', curves)
    fams = {}
    for r in runs.values():
        fams[r['family']] = fams.get(r['family'], 0) + 1
    print('%d runs (%d with curves) -> runs_0..%d.json, curves_0..%d.json' % (len(runs), len(curves), n1 - 1, n2 - 1))
    print('families:', ', '.join('%s %d' % kv for kv in sorted(fams.items())))


if __name__ == '__main__':
    sys.exit(main())
