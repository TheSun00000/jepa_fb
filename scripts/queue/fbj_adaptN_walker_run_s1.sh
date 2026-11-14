#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:50:00
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

CMD="python tdmpc_adapt.py mode=adapt env_name=walker task=run wm_ckpt=models/2026-11-14_17-14-43_JEPAWM-walker/wm_final.pickle n_labels=100 300 1000 3000 10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=1"
echo "Running: $CMD"
mkdir -p models logs
python tdmpc_adapt.py mode=adapt env_name=walker task=run wm_ckpt=models/2026-11-14_17-14-43_JEPAWM-walker/wm_final.pickle n_labels=100 300 1000 3000 10000 reward_steps=2000 adapt_steps=6000 eval_each_steps=2 seed=1
