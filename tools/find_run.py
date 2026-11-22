#!/usr/bin/env python3
"""Print the checkpoint of the newest finished run that matches, or exit 1 (prints nothing) if none.

    python tools/find_run.py NAME ENV SEED FILE [cond ...]

NAME: the run-folder name before the env, e.g. JEPAWM-phimap, FBWM-gphi, FBWM-gphi-frozenwm (the folder must be
logs/<date>_<NAME>-<ENV> plus optional '_' suffixes, so JEPAWM-phimap does not match JEPAWM-phimap-online-walker).
FILE: the checkpoint name in models/<run>/ (wm_final.pickle, fbwm_gphi_final.pickle).
cond, on the command-line kwargs (info.txt line 1):  key=value (str equal),  key~text (text in the value),  !key (key absent).
Used to chain jobs on the cluster: `ck=$(python tools/find_run.py ...) && python fbwm.py ... eval_ckpt=$ck` never runs with an
empty path. Conditions avoid quotes so they fit in the queue scripts' CMD="..." line.
"""
import ast
import glob
import os
import re
import sys


def main(argv):
    if len(argv) < 4:
        sys.exit(__doc__)
    name, env, seed, fname, conds = argv[0], argv[1], argv[2], argv[3], argv[4:]
    pat = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_' + re.escape(f'{name}-{env}') + r'_*$')
    for d in sorted(glob.glob('logs/*'), reverse=True):
        base = os.path.basename(d)
        if not pat.match(base) or not os.path.isfile(f'{d}/info.txt'):
            continue
        try:
            kw = ast.literal_eval(open(f'{d}/info.txt').readline())
        except (ValueError, SyntaxError):
            continue
        if str(kw.get('env_name')) != env or str(kw.get('seed')) != seed:
            continue
        ok = True
        for c in conds:
            if c.startswith('!'):
                ok &= c[1:] not in kw
            elif '~' in c:
                k, v = c.split('~', 1)
                ok &= k in kw and v in str(kw[k])
            else:
                k, v = c.split('=', 1)
                ok &= k in kw and str(kw[k]) == v
        f = f'models/{base}/{fname}'
        if ok and os.path.isfile(f):
            print(f)
            return 0
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
