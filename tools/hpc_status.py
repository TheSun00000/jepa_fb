#!/usr/bin/env python3
"""Collect the state of the fb_jepa Slurm jobs launched from scripts/ on the HPC (read-only). Ported from CLFB.

One `ssh jz` call lists the jobs (sacct); a second one reads, for the jobs that are new or not final yet,
the first lines of their .slurm_logs/*.out (command, run folder) and the tail of the .err of failed jobs.

Jobs also get `results` (final MPPI / policy, see job_results.py) from the synced logs in hpc/logs.
Writes hpc/status/jobs.json (all known jobs, keyed by job id) and prints a short summary.
`--diff` also writes hpc/status/snapshot.json ({checkedAt, jobs}), the content of the tracker's document
snapshot/jobs, and rebuilds the run index (tools/runs_index.py: hpc/status/runs_<n>.json and curves_<n>.json, the documents
snapshot/runs_<n> and snapshot/curves_<n>); it says whether anything changed since the last push (`--mark-pushed` records the push).
The tracker flags (tracked, title, description) live apart, in the collection `tracking`, so a refresh never touches them.

A desktop notification is sent when jobs start, finish, fail or are cancelled (one per job array and kind of change).

Usage:
    tools/hpc_status.py [--since 2026-11-13]   (default: runs_index.TRACK_FROM; older jobs are not tracked) [--diff] [--mark-pushed] [--no-notify] [--notify-dry]
Exits quietly (code 0) when the HPC cannot be reached, so it is safe to run from a timer.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import job_results
import runs_index

HOST = 'jz'
REMOTE_DIR = '$WORK/jepa_fb'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'hpc', 'status')
JOBS = os.path.join(OUT, 'jobs.json')
PUSHED = os.path.join(OUT, 'pushed.json')       # hash of the job fields last pushed to the tracker
SNAPSHOT = os.path.join(OUT, 'snapshot.json')
RUNS_DIGEST_SRC = 'runs'                        # runs and curves are written in shards by runs_index
SHARD_BYTES = 200_000                           # a database document is limited to 262144 bytes of JSON

FINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'DEADLINE', 'BOOT_FAIL'}
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
SACCT_FIELDS = 'JobIDRaw,JobID,JobName,State,Elapsed,Submit,Start,End,ExitCode,Partition,NodeList,WorkDir'
PROJECT_DIR_NAME = 'jepa_fb'                     # CLFB runs on the same account with the same script names: keep only jobs whose WorkDir is this repo


def ssh(script, timeout=120):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, 'bash -l -s'],
                       input=script, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or 'ssh failed (%d)' % r.returncode)
    return r.stdout


def status_of(state):
    s = state.split()[0].rstrip('+')
    if s in ('PENDING', 'REQUEUED', 'RESIZING'):
        return 'pending'
    if s in ('RUNNING', 'COMPLETING', 'SUSPENDED'):
        return 'running'
    if s == 'COMPLETED':
        return 'finished'
    if s.startswith('CANCELLED'):
        return 'cancelled'
    return 'failed'


def list_jobs(since):
    out = ssh('cd "%s" || exit 1\necho @@SCRIPTS; (ls scripts; ls scripts/queue) 2>/dev/null\necho @@SACCT\n'
              'sacct -u "$USER" -S %s -X -n -P --format=%s\n'
              'echo @@SQUEUE; squeue -h -u "$USER" -t PENDING -o "%%i|%%R"\n' % (REMOTE_DIR, since, SACCT_FIELDS))
    scripts, jobs, mode, reasons = set(), {}, None, {}
    for line in out.splitlines():
        if line.startswith('@@'):
            mode = line[2:]
        elif mode == 'SCRIPTS' and line.strip():
            scripts.add(line.strip())
        elif mode == 'SQUEUE' and '|' in line:
            reasons[line.split('|')[0].strip()] = line.split('|', 1)[1].strip()
        elif mode == 'SACCT' and line.strip():
            f = line.split('|')
            if len(f) < len(SACCT_FIELDS.split(',')):
                continue
            raw, jid, name, state, elapsed, submit, start, end, exitc, part, node, workdir = f[:12]
            if name not in scripts or os.path.basename(workdir.rstrip('/')) != PROJECT_DIR_NAME:   # only this repo's jobs, launched from scripts/
                continue
            array_id, _, idx = jid.partition('_')
            if submit < since:                          # older than the window
                continue
            if idx.startswith('['):                     # array range whose tasks have not started: one row, keyed by its sacct id
                if status_of(state) != 'pending':       # (a cancelled range never ran: not listed)
                    continue
                raw = jid
            jobs[raw] = {
                'jobId': raw, 'arrayJob': array_id if idx else None, 'arrayIndex': idx or None,
                'script': name, 'state': state.split()[0], 'status': status_of(state), 'elapsed': elapsed,
                'submit': submit, 'start': start, 'end': end, 'exit': exitc, 'partition': part, 'node': node,
            }
    for raw, j in jobs.items():                         # why a pending job waits (QOSGrpCpuLimit, Priority, Resources, ...)
        if j['status'] == 'pending':
            j['reason'] = reasons.get(raw) or reasons.get(j['arrayJob'] or '', '') or reasons.get('%s_%s' % (j['arrayJob'], j['arrayIndex']), '')
    return jobs


def read_details(jobs, ids):
    """command, run folders, last output line (and .err tail if failed) for the given job ids."""
    if not ids:
        return {}
    parts = ['cd "%s" || exit 1' % REMOTE_DIR]
    for raw in ids:
        j = jobs[raw]
        # queue jobs log to %x_%j; the old array scripts logged to %x_%A_%a
        f = ('.slurm_logs/%s_%s_%s' % (j['script'], j['arrayJob'], j['arrayIndex'])) if j.get('arrayJob') and j.get('arrayIndex') else '.slurm_logs/%s_%s' % (j['script'], raw)
        parts.append(
            'echo "@@JOB %s"; if [ -f "%s.out" ]; then grep -a -m1 "^Running:" "%s.out"; '
            'grep -a "^LOGS_FOLDER" "%s.out" | sed "s/^/@@FOLDER /"; '
            'echo "@@LAST"; tail -c 3000 "%s.out" | tr "\\r" "\\n" | grep -a -v "^$" | tail -n 1 | cut -c1-240; fi; '
            'if [ -f "%s.err" ]; then echo "@@ERR"; tr "\\r" "\\n" < "%s.err" | grep -a -v -E "it/s|s/it|^[[:space:]]*$" | tail -n 8 | cut -c1-240; fi' % (raw, f, f, f, f, f, f))
    out = ssh('\n'.join(parts))
    res, cur, mode = {}, None, None
    for line in out.splitlines():
        if line.startswith('@@JOB '):
            cur = line.split()[1]
            res[cur] = {'command': None, 'runFolders': [], 'lastLine': '', 'err': ''}
            mode = None
        elif cur is None:
            continue
        elif mode and not line.startswith('@@'):      # inside @@LAST / @@ERR: never a command or a folder
            res[cur][mode] = (res[cur][mode] + '\n' + line).strip()
        elif line.startswith('Running:'):
            res[cur]['command'] = line[len('Running:'):].strip()
        elif line.startswith('@@FOLDER'):
            v = line.split(':', 1)[1].strip()
            res[cur]['runFolders'].append(v[2:].split('/', 1)[-1] if v.startswith('./') else v)
        elif line.startswith('@@LAST'):
            mode = 'lastLine'
        elif line.startswith('@@ERR'):
            mode = 'err'
        elif mode:
            res[cur][mode] = (res[cur][mode] + '\n' + line).strip()
    return res


def split_commands(cmd):
    """The python commands of a job whose script runs several of them in a row ('python a ...; python b ...')."""
    return [c.strip() for c in (cmd or '').split(';') if c.strip().startswith('python ')]


def expand_children(known):
    """A job that runs several commands (one run folder each) gets one child entry per command, so that the tracker can follow
    every run: id '<jobId>.<i>' (i from 0), same Slurm fields, its own command and run folder. The parent is kept and marked
    with its 'children'. Children are rebuilt at every check. A finished job whose folders do not match its commands is left whole."""
    for r in [r for r, j in known.items() if j.get('parentJob')]:
        del known[r]
    for r, j in list(known.items()):
        j.pop('children', None)
        cmds, folders = split_commands(j.get('command')), j.get('runFolders') or []
        if len(cmds) < 2 or (j['status'] not in ('running', 'pending') and len(folders) != len(cmds)):
            continue
        kids = []
        for i, c in enumerate(cmds):
            started = i < len(folders)
            last = started and i == len(folders) - 1
            if started and not last:
                st = 'finished'
            elif last:
                st = j['status']
            else:
                st = 'pending' if j['status'] in ('running', 'pending') else j['status']
            kid = dict(j, jobId='%s.%d' % (r, i), parentJob=r, command=c, runFolders=[folders[i]] if started else [],
                       status=st, lastLine=j.get('lastLine', '') if last and st == 'running' else '',
                       err=j.get('err', '') if last and st == 'failed' else '', results=None)
            kid.pop('children', None)
            if j.get('arrayIndex') is not None:
                kid['arrayIndex'] = '%s.%d' % (j['arrayIndex'], i)
            elif j.get('arrayJob'):
                kid['arrayIndex'] = str(i)
            known[kid['jobId']] = kid
            kids.append(kid['jobId'])
        j['children'] = kids


PAGE_RESULT_KEYS = ('text', 'final', 'folder', 'lastStep', 'complete')      # the keys of job['results'] that the tracker page reads
JOB_SHARD_BYTES = 200_000


def write_push_snapshot(now, jobs):
    """hpc/status/jobs_<n>.json = {checkedAt, jobs: {...}} of at most JOB_SHARD_BYTES each (the tracker documents snapshot/jobs (n = 0), snapshot/jobs_1, ...): the full
    snapshot passed the 262144 byte limit of a database document. The fields the page never reads are left out (exit, partition, node, an empty err, lastLine of
    jobs that are not running, the other keys of results)."""
    items = []
    for r in sorted(jobs):
        k = {a: b for a, b in jobs[r].items() if a not in ('exit', 'partition', 'node')}
        if not k.get('err'): k.pop('err', None)
        if k.get('status') != 'running': k.pop('lastLine', None)
        if k.get('results'): k['results'] = {a: b for a, b in k['results'].items() if a in PAGE_RESULT_KEYS}
        items.append((r, k))
    shards, cur = [], {}
    for r, k in items:
        if cur and len(json.dumps({'checkedAt': now, 'jobs': dict(cur, **{r: k})}, separators=(',', ':'))) > JOB_SHARD_BYTES:
            shards.append(cur)
            cur = {}
        cur[r] = k
    shards.append(cur)
    for old in [f for f in os.listdir(OUT) if re.fullmatch(r'jobs_\d+\.json', f)] + ['snapshot_push.json']:
        if os.path.exists(os.path.join(OUT, old)):
            os.remove(os.path.join(OUT, old))
    for i, sh in enumerate(shards):
        with open(os.path.join(OUT, 'jobs_%d.json' % i), 'w') as f:
            json.dump({'checkedAt': now, 'jobs': sh}, f, separators=(',', ':'))


def digest(job):
    keys = ('state', 'reason', 'children', 'parentJob', 'elapsed', 'start', 'end', 'exit', 'node', 'command', 'runFolders', 'lastLine', 'err', 'results')
    return hashlib.md5(json.dumps([job.get(k) for k in keys], sort_keys=True).encode()).hexdigest()


def whole_digest(jobs):
    return hashlib.md5(''.join(sorted(r + digest(j) for r, j in jobs.items())).encode()).hexdigest()


def curves_digest(c):
    return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def transitions(before, fresh):
    """{(array or job id, script, new status): [job ids]} for jobs whose status changed since the last check."""
    out = {}
    for raw, j in fresh.items():
        new = j['status']
        if before.get(raw) != new and new in ('running', 'finished', 'failed', 'cancelled'):
            out.setdefault((j['arrayJob'] or raw, j['script'], new), []).append(raw)
    return out


def notify(trans, dry=False):
    """One desktop notification per job array, listing how many jobs went to each status."""
    by_array = {}
    for (arr, script, status), ids in trans.items():
        by_array.setdefault((arr, script), {})[status] = len(ids)
    env = dict(os.environ)
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/run/user/%d/bus' % os.getuid())
    env.setdefault('DISPLAY', ':0')
    for (arr, script), counts in by_array.items():
        title = 'Jean Zay (fb_jepa): job %s' % arr
        body = '%s\n%s' % (script, ', '.join('%d %s' % (n, st) for st, n in sorted(counts.items())))
        urgency = 'critical' if 'failed' in counts else 'normal'
        if dry:
            print('notify [%s] %s | %s' % (urgency, title, body.replace('\n', ' | ')))
        else:
            try:
                subprocess.run(['notify-send', '-a', 'fb_jepa', '-u', urgency, title, body], env=env, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default=runs_index.TRACK_FROM, help='ignore jobs submitted before this date')
    ap.add_argument('--diff', action='store_true', help='write hpc/status/to_push.json')
    ap.add_argument('--no-notify', action='store_true', help='do not send desktop notifications')
    ap.add_argument('--notify-dry', action='store_true', help='print the notifications instead of sending them')
    ap.add_argument('--mark-pushed', action='store_true', help='record to_push.json as pushed to the tracker')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.mark_pushed:
        with open(PUSHED, 'w') as f:
            json.dump({'hash': whole_digest(load(JOBS, {})), 'curves': curves_digest(runs_index.build_all() + (load(os.path.join(ROOT, 'queue', 'experiments.json'), {'entries': []}),))}, f)
        print('marked as pushed')
        return

    try:
        fresh = list_jobs(args.since)
        known = {r: j for r, j in load(JOBS, {}).items() if (j.get('submit') or '') >= args.since and not str(j.get('arrayIndex') or '').startswith('[')
                 and not j.get('parentJob')}
        need = [r for r, j in fresh.items() if r not in known or known[r].get('state') not in FINAL
                or known[r].get('status') != j['status'] or not known[r].get('command')]
        details = read_details(fresh, need)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
        print('hpc_status: HPC not reachable, skipped (%s)' % str(e).splitlines()[0][:120], file=sys.stderr)
        return

    for raw, j in fresh.items():
        d = details.get(raw) or {k: known.get(raw, {}).get(k) for k in ('command', 'runFolders', 'lastLine', 'err')}
        err = '\n'.join(l.rstrip() for l in ANSI.sub('', d.get('err') or '').splitlines() if l.strip()) if j['status'] == 'failed' else ''
        j.update({'command': d.get('command'), 'runFolders': d.get('runFolders') or [],
                  'lastLine': ANSI.sub('', d.get('lastLine') or '')[-200:], 'err': err[-900:]})
    first_run = not os.path.exists(JOBS)
    trans = transitions({r: j.get('status') for r, j in known.items()}, fresh)
    known.update(fresh)
    expand_children(known)
    for j in known.values():          # headline numbers from the synced logs (run tools/sync_from_hpc.sh first)
        j['results'] = job_results.headline(j)
    with open(JOBS, 'w') as f:
        json.dump(known, f, indent=1, sort_keys=True)

    if trans and not first_run and not args.no_notify:
        notify(trans, dry=args.notify_dry)

    counts = {}
    for j in known.values():
        counts[j['status']] = counts.get(j['status'], 0) + 1
    print('%d jobs:' % len(known), ', '.join('%s %d' % kv for kv in sorted(counts.items())))

    if args.diff:
        runs, curves = runs_index.build_all()
        pushed = load(PUSHED, {})
        qstate = load(os.path.join(ROOT, 'queue', 'experiments.json'), {'entries': []})
        changed = pushed.get('hash') != whole_digest(known) or pushed.get('curves') != curves_digest((runs, curves, qstate))
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        with open(SNAPSHOT, 'w') as f:       # a job that ran several commands is represented by its children only
            json.dump({'checkedAt': now, 'jobs': {r: j for r, j in known.items() if not j.get('children')}}, f, separators=(',', ':'))
        write_push_snapshot(now, {r: j for r, j in known.items() if not j.get('children')})
        q = load(os.path.join(ROOT, 'queue', 'experiments.json'), {'entries': []})      # the local queue, shown on the Jobs tab
        queue_doc = {'checkedAt': now, 'entries': [{k: e.get(k) for k in ('name', 'title', 'gpu', 'state', 'job_id', 'added', 'submitted', 'cmds')}
                                                   for e in q.get('entries', [])]}
        with open(os.path.join(OUT, 'queue.json'), 'w') as f:
            json.dump(queue_doc, f, separators=(',', ':'))
        n_runs = runs_index.write_shards('runs', 'runs', runs)
        n_curves = runs_index.write_shards('curves', 'curves', curves)
        print('%d runs in %d shard(s), %d curves in %d shard(s)' % (len(runs), n_runs, len(curves), n_curves))
        print('changed since last push' if changed else 'no change since last push')

if __name__ == '__main__':
    main()
