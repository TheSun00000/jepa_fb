#!/usr/bin/env bash
# One-way pull of logs and code from the HPC into ./hpc (never pushes, never deletes).
# Needs an open connection to the `jz` host (see ~/.ssh/config); without one it exits quietly.
# Usage: tools/sync_from_hpc.sh [--dry-run]
set -u

REMOTE_HOST=jz
REMOTE_DIR=/lustre/fswork/projects/rech/vcj/uyo96iu/jepa_fb/
DEST="$(cd "$(dirname "$0")/.." && pwd)/hpc/"
MAX_SIZE=20m

SSH_CMD="ssh -o BatchMode=yes -o ConnectTimeout=10"

if ! $SSH_CMD "$REMOTE_HOST" true 2>/dev/null; then
    echo "sync_from_hpc: no open connection to '$REMOTE_HOST', skipped (run 'ssh $REMOTE_HOST' in a terminal)" >&2
    exit 0
fi

mkdir -p "$DEST"

# --tick (tools/tracker_tick.py, every 2 min): only the CSVs and info.txt of the run folders since SINCE (the tracked ones),
# as one tar stream over one ssh call (~4 s; rsync took 10-110 s for the same 48 files). No .slurm_logs (hpc_status.py reads
# their tails on the cluster itself), no code, no scripts. Overwrites the local copies of those files, deletes nothing.
if [ "${1:-}" = "--tick" ]; then
    SINCE="${SINCE:-2026-11-13}"
    $SSH_CMD "$REMOTE_HOST" "cd $REMOTE_DIR && ls -d logs/*/ | awk -v s='logs/$SINCE' '\$0 >= s' | while read d; do ls \"\$d\"*.csv \"\${d}info.txt\" 2>/dev/null; done | tar czf - -T -" \
        | tar xmzf - -C "$DEST"   # -m: local mtimes (the cluster clock is ahead)
    exit $?
fi
exec rsync -a --info=stats1,progress0 --prune-empty-dirs --max-size="$MAX_SIZE" -e "$SSH_CMD" "$@" \
    --include='/logs/' --include='/logs/*/' \
    --include='/logs/*/*.csv' --include='/logs/*/info.txt' \
    --include='/.slurm_logs/' --include='/.slurm_logs/*.out' --include='/.slurm_logs/*.err' \
    --include='/scripts/' --include='/scripts/**' \
    --include='/utils/' --include='/utils/*.py' \
    --include='/*.py' --include='/*.md' --include='/*.csv' \
    --exclude='*' \
    "$REMOTE_HOST:$REMOTE_DIR" "$DEST"
