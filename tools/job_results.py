"""Headline numbers of a job, from the run folder(s) it wrote (hpc/logs, synced by tools/sync_from_hpc.sh), via tools/runs_index.py.

fb_jepa keeps the numbers per RUN (runs_index); a job only points at its run folder(s). headline(job) gives the page a short
summary: final MPPI / policy mean over tasks (mean of the last 10 evals), the last step, and whether the run reached 1M updates.
"""
import os

import runs_index


def headline(job):
    folders = job.get('runFolders') or []
    if not folders:
        return None
    folder = os.path.basename(folders[0].rstrip('/'))
    if not os.path.isdir(os.path.join(runs_index.LOGS, folder)):
        # some run folders were renamed by hand afterwards with trailing underscores (runs cut by the time limit)
        alt = sorted(f for f in os.listdir(runs_index.LOGS) if f.rstrip('_') == folder) if os.path.isdir(runs_index.LOGS) else []
        if not alt:
            return None
        folder = alt[0]
    rec, _ = runs_index.index_run(folder)
    f = rec.get('final') or {}
    parts = []
    if f.get('mppi') is not None:
        parts.append('MPPI %.0f' % f['mppi'])
    if f.get('policy') is not None:
        parts.append('policy %.0f' % f['policy'])
    return {'folder': folder, 'final': f, 'lastStep': rec.get('lastStep'), 'complete': rec.get('complete'),
            'text': ' / '.join(parts) + (' @%dk' % (rec.get('lastStep', 0) // 1000) if parts else '')}
