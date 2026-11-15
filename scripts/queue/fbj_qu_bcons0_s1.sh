#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CMD="python fbwm.py env_name=quadruped variant=gphi eval_mode=both f_encoder_grad=1 consistency_coef=5 bcons_coef=0 eval_each_steps=25 mppi_eval_each=100 mppi_reward=lstsq z_cov=dataset num_evals_final=20 seed=1"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=quadruped variant=gphi eval_mode=both f_encoder_grad=1 consistency_coef=5 bcons_coef=0 eval_each_steps=25 mppi_eval_each=100 mppi_reward=lstsq z_cov=dataset num_evals_final=20 seed=1
