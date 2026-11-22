#!/usr/bin/env bash
# Prints models/<run>/fbwm_gphi_final.pickle of the newest finished fbwm.py gphi TRAINING run (no eval_ckpt) for env $1 and seed $2;
# optional $3 = key=value that must also appear in info.txt line 1 (e.g. bcons_coef=0 matches "'bcons_coef': 0,").
# Exits 1 (prints nothing) if none, so `ck=$(bash tools/find_fbwm.sh walker 1 bcons_coef=0) && python fbwm.py ... eval_ckpt=$ck`
# never runs the eval with an empty checkpoint (which would start a training run).
# Used to chain the N=10k MPPI eval after a policy-only training run in the same job.
env=$1; seed=$2; pat=""
[ -n "$3" ] && pat="'${3%%=*}': ${3#*=}"
for d in $(ls -d logs/*_FBWM-gphi-${env}* 2>/dev/null | sort -r); do
  [ -f "$d/info.txt" ] || continue
  l=$(head -1 "$d/info.txt")
  [[ "$l" == *"'env_name': '${env}'"* ]] || continue
  [[ "$l" == *"'seed': ${seed},"* || "$l" == *"'seed': ${seed}}"* ]] || continue
  [[ "$l" == *"'eval_ckpt'"* ]] && continue
  [[ -z "$pat" || "$l" == *"${pat},"* || "$l" == *"${pat}}"* ]] || continue
  f="models/$(basename "$d")/fbwm_gphi_final.pickle"
  [ -f "$f" ] && { echo "$f"; exit 0; }
done
exit 1
