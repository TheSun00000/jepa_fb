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

CMD="while [ -z \"$(bash tools/find_wm.sh quadruped 2)\" ]; do sleep 30; done; python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=quadruped task=walk wm_ckpt=$(bash tools/find_wm.sh quadruped 2) n_labels=10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2"
echo "Running: $CMD"
mkdir -p models logs
while [ -z "$(bash tools/find_wm.sh quadruped 2)" ]; do sleep 30; done
python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=quadruped task=walk wm_ckpt=$(bash tools/find_wm.sh quadruped 2) n_labels=10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=2
