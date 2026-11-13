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

CMD="python fbwm.py env_name=cheetah variant=sepB eval_mode=both num_outer_steps=20 eval_each_steps=10; python fbwm.py env_name=cheetah variant=gphi eval_mode=both num_outer_steps=20 eval_each_steps=10"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=cheetah variant=sepB eval_mode=both num_outer_steps=20 eval_each_steps=10
python fbwm.py env_name=cheetah variant=gphi eval_mode=both num_outer_steps=20 eval_each_steps=10
