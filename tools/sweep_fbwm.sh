#!/usr/bin/env bash
# Queue an eval-only label sweep (z from N = 100..10k, 20 episodes, root-check MPPI, lstsq, dataset Cov) of one fbwm checkpoint on V100.
# usage: tools/sweep_fbwm.sh NAME ENV SEED RUN_FOLDER "extra training flags (e.g. consistency_coef=2)"
set -e
name=$1; env=$2; seed=$3; folder=$4; extra=$5
C="python fbwm.py env_name=$env variant=gphi eval_mode=both bcons_coef=2 f_encoder_grad=1 $extra seed=$seed eval_ckpt=models/$folder/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset mppi_root_check=1 num_evals_final=20"
args=(); for n in 100 300 1000 3000 10000; do args+=(--cmd "$C z_labels=$n"); done
python tools/queue.py add "$name" --title "label sweep (paper config): $env s$seed $extra" --gpu v100 --time 2:00:00 "${args[@]}" >/dev/null
