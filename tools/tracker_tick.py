#!/usr/bin/env python3
"""One tracker refresh tick, made for a cheap Claude loop: everything except the ArtifactData push.

    tools/tracker_tick.py        # first tick, or the previous push failed
    tools/tracker_tick.py ok     # the previous tick's push succeeded: record it, then do this tick

It runs tools/sync_from_hpc.sh --tick (CSVs + info.txt of the tracked run folders only), tools/queue.py submit and tools/hpc_status.py --diff, then prints one line:
    UNCHANGED [IDLE]                 nothing to push
    PUSH [IDLE] <json writes>        the `writes` argument of one ArtifactData batch on the tracker (snapshot shards, with if_version)
IDLE = no fb_jepa job queued, submitted or running: the loop can stop.

Document versions of snapshot/* are kept in hpc/status/db_versions.json (only this tool writes those documents, so a successful
set bumps each by one). If a push is refused for a version, read the current versions with ArtifactData list (collection snapshot),
fix db_versions.json, and rerun without `ok`.
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
STATUS = os.path.join(ROOT, 'hpc', 'status')
VERSIONS = os.path.join(STATUS, 'db_versions.json')
LAST = os.path.join(STATUS, 'last_writes.json')
QFILE = os.path.join(ROOT, 'queue', 'experiments.json')


def run(*cmd, timeout=600):
    r = subprocess.run(list(cmd), cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def load(path, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def main():
    versions = load(VERSIONS, {})
    if sys.argv[1:] == ['ok'] and os.path.exists(LAST):
        run(sys.executable, 'tools/hpc_status.py', '--mark-pushed')
        for doc in load(LAST, []):
            versions[doc] = versions.get(doc, 0) + 1
        json.dump(versions, open(VERSIONS, 'w'))
        os.remove(LAST)

    import runs_index
    code, out = run('env', 'SINCE=' + runs_index.TRACK_FROM, 'bash', 'tools/sync_from_hpc.sh', '--tick')
    if code:
        print('SYNC FAILED ' + out.strip().splitlines()[-1:][0] if out.strip() else 'SYNC FAILED')
    run(sys.executable, 'tools/queue.py', 'submit', '--notify')
    code, out = run(sys.executable, 'tools/hpc_status.py', '--diff')
    if code:
        print('STATUS FAILED ' + ' | '.join(out.strip().splitlines()[-3:]))
        return

    q = load(QFILE, {'entries': []})
    idle = not any(e.get('state') in ('queued', 'submitted', 'running') for e in q['entries'])
    tag = ' IDLE' if idle else ''
    if 'changed since last push' not in out:
        print('UNCHANGED' + tag)
        return

    files = sorted(glob.glob(os.path.join(STATUS, 'runs_*.json')) + glob.glob(os.path.join(STATUS, 'curves_*.json'))
                   + glob.glob(os.path.join(STATUS, 'jobs_*.json'))) + [os.path.join(STATUS, 'queue.json')]
    writes, docs = [], []
    for f in files:
        doc = os.path.basename(f)[:-5]
        w = {'op': 'set', 'collection': 'snapshot', 'doc_id': doc, 'file_path': f}
        if doc in versions:
            w['if_version'] = versions[doc]
        writes.append(w)
        docs.append(doc)
    json.dump(docs, open(LAST, 'w'))
    print('PUSH' + tag + ' ' + json.dumps(writes, separators=(',', ':')))


if __name__ == '__main__':
    main()
