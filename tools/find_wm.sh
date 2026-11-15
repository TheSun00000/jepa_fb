#!/usr/bin/env bash
# Prints models/<run>/wm_final.pickle of the finished reward-free JEPA world model for env $1 and seed $2 (empty if none yet).
env=$1; seed=$2
for d in logs/*_JEPAWM-${env}*; do
  [ -f "$d/info.txt" ] || continue
  grep -q "'seed': ${seed}[,}]" "$d/info.txt" || continue
  f="models/$(basename "$d")/wm_final.pickle"
  [ -f "$f" ] && { echo "$f"; exit 0; }
done
