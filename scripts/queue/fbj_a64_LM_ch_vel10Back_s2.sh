#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --account=vcj@v100
#SBATCH --qos=qos_gpu-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CMD="wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=100 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2; wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=300 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2; wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=1000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2; wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=3000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2; wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2"
echo "Running: $CMD"
mkdir -p models logs
wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=100 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2
wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=300 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2
wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=1000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2
wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=3000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2
wm=$(python tools/find_run.py JEPAWM-phimap cheetah 2 wm_final.pickle enc_dim=64 !var_coef) && python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=cheetah task=vel10Back wm_ckpt=$wm n_labels=10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2
