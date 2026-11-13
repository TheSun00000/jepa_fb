import os
import shutil
import csv
import random
import subprocess
import sys
from datetime import datetime


def auto_cast(x):
    for cast in (int, float):
        try:
            return cast(x)
        except:
            pass
    return x

def parse_kwargs(argv):
    out = {}
    for item in argv:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k] = auto_cast(v)
    return out


# Run metadata written to info.txt (after the kwargs and config lines) ### ### ### ### ### ###

_SEED_GIVEN = None
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# main dataset file per environment (the aux files are loaded next to it)
_DATASETS = {'cheetah': 'cheetah_rnd.npy', 'walker': 'walker_rnd.npy', 'quadruped': 'quadruped_rnd_without_physics.npy'}


def resolve_seed(kwargs, key='seed'):
    """The seed given as key=, else a random one. Written back into kwargs, so the kwargs line of info.txt records it."""
    global _SEED_GIVEN
    _SEED_GIVEN = key in kwargs
    seed = int(kwargs[key]) if _SEED_GIVEN else random.SystemRandom().randrange(10 ** 7)
    kwargs[key] = seed
    print(f'{key}: {seed}' + ('' if _SEED_GIVEN else ' (random)'))
    return seed


def _git_commit():
    """'<commit[:12]> (clean)' / '(uncommitted: a.py, b.py)'. Falls back to reading .git directly when git cannot run."""
    try:
        run = lambda *a: subprocess.run(['git', '-C', _REPO, *a], capture_output=True, text=True, timeout=20)
        head = run('rev-parse', 'HEAD')
        if head.returncode == 0 and head.stdout.strip():
            dirty = run('status', '--porcelain', '--untracked-files=no')
            files = [l[3:] for l in dirty.stdout.splitlines() if l.strip()] if dirty.returncode == 0 else None
            state = 'dirty state unknown' if files is None else ('clean' if not files else 'uncommitted: ' + ', '.join(files))
            return f'{head.stdout.strip()[:12]} ({state})'
    except Exception:
        pass
    try:
        ref = open(os.path.join(_REPO, '.git', 'HEAD')).read().strip()
        if ref.startswith('ref: '):
            ref_path = os.path.join(_REPO, '.git', ref[5:])
            if os.path.exists(ref_path):
                commit = open(ref_path).read().strip()
            else:
                commit = next((l.split()[0] for l in open(os.path.join(_REPO, '.git', 'packed-refs'))
                               if l.strip().endswith(ref[5:])), '')
        else:
            commit = ref
        return f'{commit[:12]} (dirty state unknown: git not runnable)' if commit else 'unknown'
    except Exception:
        return 'unknown'


def run_info(g, z_labels=10_000):
    """The run-metadata lines of info.txt, from the script's globals `g` (call `run_info(globals())` just before training)."""
    env = os.environ.get
    gpu = ''
    try:
        import jax
        gpu = jax.devices()[0].device_kind
    except Exception:
        pass
    job = [('slurm_job_id', env('SLURM_JOB_ID')), ('job_name', env('SLURM_JOB_NAME')), ('node', env('SLURMD_NODENAME')), ('gpu', gpu)]
    scan = g.get('SCAN_LENGTH', 1)
    proto = [('updates', g['NUM_OUTER_STEPS'] * scan if 'NUM_OUTER_STEPS' in g else None),
             ('eval_every', g['EVAL_EACH_STEPS'] * scan if 'EVAL_EACH_STEPS' in g else None),
             ('num_evals', g.get('NUM_EVALS')), ('z_labels', z_labels), ('horizon', g.get('HORIZON')), ('batch_size', g.get('BATCH_SIZE'))]
    env_name = g.get('ENV_NAME')
    if env_name == 'antmaze':
        data = 'd4rl antmaze-large-diverse-v2'
    else:
        data = os.path.join(g.get('DATASET_FOLDER', '').rstrip('/'), _DATASETS.get(env_name, '?'))
    if g.get('NUM_TRAJECTORIES') and g.get('TRAJECTORY_LEN'):
        data += f" ({g['NUM_TRAJECTORIES']} x {g['TRAJECTORY_LEN']} transitions)"
    versions = []
    for name in ('jax', 'flax', 'optax'):
        try:
            versions.append(f'{name}={__import__(name).__version__}')
        except Exception:
            versions.append(f'{name}=?')
    versions.append('python=' + sys.version.split()[0])
    seed = g.get('SEED')
    lines = [
        f"seed: {seed}" + ('' if _SEED_GIVEN is None else (' (given)' if _SEED_GIVEN else ' (random)')),
        'job: ' + (' '.join(f'{k}={v}' for k, v in job if v) or 'local'),
        'git: ' + _git_commit(),
        'command: ' + ' '.join(['python'] + sys.argv),
        'protocol: ' + ' '.join(f'{k}={v}' for k, v in proto if v is not None),
        'data: ' + data,
        'versions: ' + ' '.join(versions),
        'started: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    ]
    return '\n'.join(lines)


def mark_finished(path, steps):
    """Last line of info.txt, written only when training reaches its last update: a run cut by the time limit has none."""
    add_text_to_file(path, f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} steps={steps}")


def add_text_to_file(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        
        
        
# CSV ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

def init_csv_logger(path, fieldnames):
    file_exists = os.path.isfile(path)

    if file_exists:
        return  # don't overwrite existing file

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def log_csv_row(path, row_dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        writer.writerow(row_dict)