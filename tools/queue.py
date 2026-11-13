#!/usr/bin/env python3
"""Local queue of fb_jepa experiments to sbatch on Jean Zay, submitted whenever job slots are free.

Ported from CLFB's tools/queue.py. The dev QoS allow at most 10 submitted jobs per user and per QoS (running + pending, every
array task counts as one), and CLFB uses the same account, so more experiments than fit wait here and are sent as slots free up.
Nothing runs locally: `submit` only calls `ssh jz` (git pull of the repo, squeue, sbatch).

Partitions (user, 2026-11-13): training on H100 dev (a full 1M-update run with MPPI evals takes 85-110 min, the dev limit is 2h);
eval-only jobs on V100 dev (`--gpu v100`, chosen automatically when every command has `eval_ckpt=`).

The list lives in queue/experiments.json (local, git-ignored). Each entry has an sbatch script scripts/queue/<name>.sh (committed
and pushed, because the cluster pulls it), written by `add` from a header and a list of shell commands (run one after the other).

    tools/queue.py add NAME --time 2:00:00 --cmd "python a ..." [--cmd "python b ..."] [--title "..."] [--gpu h100|v100|a100]
    tools/queue.py status            # the list, the jobs on the cluster, the free slots (read-only)
    tools/queue.py submit [--dry-run] [--notify]
    tools/queue.py sync              # update done / failed from Slurm
    tools/queue.py requeue NAME      # put a failed/cancelled entry back in the queue
    tools/queue.py prio NAME [NAME..]  # move queued entries to the front

`submit` refuses to run if the local scripts are not committed and pushed (the cluster would not have them).
"""
import argparse, fcntl, json, os, re, subprocess, sys
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
QFILE = os.path.join(ROOT, 'queue', 'experiments.json')
HOST, REMOTE = 'jz', '/lustre/fswork/projects/rech/vcj/uyo96iu/jepa_fb'
LIMIT = 10
APP = 'fb_jepa'

ENV_SETUP = """module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
"""

GPUS = {   # sbatch lines per partition; 'qos' is what squeue reports, used to count the slots per QoS
    'h100': {'qos': 'qos_gpu_h100-dev', 'lines': '#SBATCH --cpus-per-task=24\n#SBATCH --account=vcj@h100\n#SBATCH -C h100\n#SBATCH --qos=qos_gpu_h100-dev\n'},
    'a100': {'qos': 'qos_gpu_a100-dev', 'lines': '#SBATCH --cpus-per-task=8\n#SBATCH --account=vcj@a100\n#SBATCH -C a100\n#SBATCH --qos=qos_gpu_a100-dev\n'},
    'v100': {'qos': 'qos_gpu-dev',      'lines': '#SBATCH --cpus-per-task=16\n#SBATCH --account=vcj@v100\n#SBATCH --qos=qos_gpu-dev\n'},
}
QOS_TO_GPU = {v['qos']: k for k, v in GPUS.items()}

HEADER = """#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time={time}
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
{gpu_lines}
{env}
CMD="{joined}"
echo "Running: $CMD"
mkdir -p models logs
"""


def lock():
    """One queue.py that changes the queue at a time (the systemd timer and an interactive call must not submit twice)."""
    os.makedirs(os.path.dirname(QFILE), exist_ok=True)
    f = open(os.path.join(os.path.dirname(QFILE), '.lock'), 'w')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('another queue.py is running, skipped')
        sys.exit(0)
    return f


def notify(body):
    env = dict(os.environ)
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/run/user/%d/bus' % os.getuid())
    env.setdefault('DISPLAY', ':0')
    try:
        subprocess.run(['notify-send', '-a', APP, 'Jean Zay queue (fb_jepa)', body], env=env, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def load():
    if not os.path.exists(QFILE):
        return {'limit': LIMIT, 'entries': []}
    return json.load(open(QFILE))


def save(q):
    os.makedirs(os.path.dirname(QFILE), exist_ok=True)
    json.dump(q, open(QFILE, 'w'), indent=1)


def ssh(script, timeout=180):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, 'bash -l -s'], input=script,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-400:] or 'ssh failed')
    return r.stdout


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cluster_counts():
    """Submitted jobs of this user per GPU group, CLFB's included (the limit is per QoS and per user)."""
    out = ssh('squeue -u "$USER" -h -r -o "%q"').strip().splitlines()
    c = {g: 0 for g in GPUS}
    for q in out:
        g = QOS_TO_GPU.get(q.strip())
        if g:
            c[g] += 1
    return c


def group_of(e):
    if e.get('gpu'):
        return e['gpu']
    try:
        text = open(os.path.join(ROOT, e['script'])).read()
    except OSError:
        return 'h100'
    for g, v in GPUS.items():
        if v['qos'] in text:
            return g
    return 'h100'


def add(a):
    q = load()
    if any(e['name'] == a.name for e in q['entries']):
        sys.exit('entry %s already exists' % a.name)
    gpu = a.gpu or ('v100' if all('eval_ckpt=' in c for c in a.cmd) else 'h100')
    joined = '; '.join(a.cmd).replace('"', '\\"')
    body = HEADER.format(time=a.time, gpu_lines=GPUS[gpu]['lines'].rstrip('\n'), env=ENV_SETUP, joined=joined)
    body += '\n'.join(a.cmd) + '\n'
    rel = os.path.join('scripts', 'queue', a.name + '.sh')
    os.makedirs(os.path.join(ROOT, 'scripts', 'queue'), exist_ok=True)
    with open(os.path.join(ROOT, rel), 'w') as f:
        f.write(body)
    q['entries'].append({'name': a.name, 'script': rel, 'title': a.title, 'cmds': a.cmd, 'slots': a.slots, 'gpu': gpu,
                         'state': 'queued', 'job_id': None, 'added': datetime.now().isoformat(timespec='seconds')})
    save(q)
    print('queued %s (%s) -> %s' % (a.name, gpu, rel))


def status(a):
    q = load()
    try:
        counts = cluster_counts()
        print('cluster: ' + ', '.join('%s %d/%d' % (g, n, q.get('limit', LIMIT)) for g, n in counts.items()))
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print('cluster not reachable:', str(e)[:120])
    for i, e in enumerate(q['entries'], 1):
        print('%2d. %-28s %-5s %-10s %-8s %s' % (i, e['name'], e.get('gpu', '?'), e['state'], e['job_id'] or '-', e['title']))
    cnt = {}
    for e in q['entries']:
        cnt[e['state']] = cnt.get(e['state'], 0) + 1
    print(cnt)


def sync(a=None):
    q = load()
    ids = [e['job_id'] for e in q['entries'] if e['job_id'] and e['state'] in ('submitted', 'running')]
    if not ids:
        return
    out = ssh('sacct -j %s -X -n -P --format=JobID,State' % ','.join(ids))
    states = dict(l.split('|') for l in out.strip().splitlines() if '|' in l)
    for e in q['entries']:
        s = states.get(e['job_id'])
        if not s:
            continue
        s = s.split()[0].rstrip('+')
        e['state'] = {'COMPLETED': 'done', 'RUNNING': 'running', 'PENDING': 'submitted'}.get(
            s, 'failed' if s in ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL') else e['state'])
        if s.startswith('CANCELLED'):
            e['state'] = 'cancelled'
    save(q)


def submit(a):
    q = load()
    if not [e for e in q['entries'] if e['state'] == 'queued']:
        print('nothing queued')
        return
    sync()
    q = load()
    queued = [e for e in q['entries'] if e['state'] == 'queued']
    counts = cluster_counts()
    gfree = {g: q.get('limit', LIMIT) - counts[g] for g in counts}
    print('jobs per group: %s; %d queued' % (', '.join('%s %d' % kv for kv in counts.items()), len(queued)))
    # the cluster needs the scripts and the code: they must be committed and pushed
    dirty = git('status', '--porcelain', 'scripts', 'tools', 'utils', '*.py')
    subprocess.run(['git', 'fetch', '-q'], cwd=ROOT)
    unpushed = git('rev-list', '--count', 'origin/main..HEAD')
    if dirty or unpushed not in ('0', ''):
        sys.exit('not submitted: uncommitted changes in scripts/tools/utils/*.py or %s unpushed commit(s): commit and push first' % unpushed)
    todo = []
    for e in queued:
        g = group_of(e)
        if e['slots'] <= gfree.get(g, 0):
            todo.append(e)
            gfree[g] -= e['slots']
    if a.dry_run:
        print('would submit:', [e['name'] for e in todo])
        return
    if not todo:
        return
    script = 'cd "%s" || exit 1\ngit pull --ff-only -q 2>&1 | tail -2\nmkdir -p .slurm_logs\n' % REMOTE
    for e in todo:
        script += 'echo "@@%s $(sbatch %s)"\n' % (e['name'], e['script'])
    out = ssh(script)
    done, bad = [], []
    for line in out.splitlines():
        m = re.match(r'@@(\S+) Submitted batch job (\d+)', line)
        if m:
            for e in q['entries']:
                if e['name'] == m.group(1):
                    e.update(state='submitted', job_id=m.group(2), submitted=datetime.now().isoformat(timespec='seconds'))
            print('submitted %s -> job %s' % (m.group(1), m.group(2)))
            done.append('%s (%s)' % (m.group(1), m.group(2)))
        elif line.startswith('@@'):
            print('FAILED', line)
            bad.append(line)
    save(q)
    if a.notify and (done or bad):
        notify('submitted %d: %s%s' % (len(done), ', '.join(done), ('; %d failed' % len(bad)) if bad else ''))


def prio(a):
    q = load()
    by = {e['name']: e for e in q['entries']}
    front = [by[n] for n in a.names if n in by]
    q['entries'] = front + [e for e in q['entries'] if e['name'] not in a.names]
    save(q)
    print('front of the queue:', [e['name'] for e in q['entries'] if e['state'] == 'queued'][:8])


def requeue(a):
    q = load()
    for e in q['entries']:
        if e['name'] == a.name:
            e.update(state='queued', job_id=None)
    save(q)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('add'); p.add_argument('name'); p.add_argument('--time', default='2:00:00'); p.add_argument('--cmd', action='append', required=True)
    p.add_argument('--title', default=''); p.add_argument('--slots', type=int, default=1); p.add_argument('--gpu', choices=list(GPUS), default=''); p.set_defaults(f=add)
    sp.add_parser('status').set_defaults(f=status)
    sp.add_parser('sync').set_defaults(f=sync)
    p = sp.add_parser('submit'); p.add_argument('--dry-run', action='store_true'); p.add_argument('--notify', action='store_true'); p.set_defaults(f=submit)
    p = sp.add_parser('requeue'); p.add_argument('name'); p.set_defaults(f=requeue)
    p = sp.add_parser('prio'); p.add_argument('names', nargs='+'); p.set_defaults(f=prio)
    a = ap.parse_args()
    if a.cmd in ('add', 'submit', 'sync', 'requeue', 'prio'):
        _lock = lock()
    a.f(a)


if __name__ == '__main__':
    main()
