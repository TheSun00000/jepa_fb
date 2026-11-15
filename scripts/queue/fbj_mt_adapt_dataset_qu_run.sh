#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --account=vcj@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CMD="while [ -z \"$(ls -d models/*_JEPAWM-quadruped*/wm_final.pickle 2>/dev/null | head -1)\" ]; do sleep 30; done; python tdmpc_adapt.py mode=adapt q_data=dataset env_name=quadruped task=run wm_ckpt=$(ls -d models/*_JEPAWM-quadruped*/wm_final.pickle | head -1) n_labels=10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=1"
echo "Running: $CMD"
mkdir -p models logs
while [ -z "$(ls -d models/*_JEPAWM-quadruped*/wm_final.pickle 2>/dev/null | head -1)" ]; do sleep 30; done
python tdmpc_adapt.py mode=adapt q_data=dataset env_name=quadruped task=run wm_ckpt=$(ls -d models/*_JEPAWM-quadruped*/wm_final.pickle | head -1) n_labels=10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=1
