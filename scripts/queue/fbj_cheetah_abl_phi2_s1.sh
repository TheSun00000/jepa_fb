#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
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

CMD="python fb_jax.py env_name=cheetah z_dim=50 num_outer_steps=1000 num_evals_final=10 q_loss_coef=0.02 label_align=1 discount=0.98 num_trajectories=5000 phi_layers=2 seed=1"
echo "Running: $CMD"
mkdir -p models logs
python fb_jax.py env_name=cheetah z_dim=50 num_outer_steps=1000 num_evals_final=10 q_loss_coef=0.02 label_align=1 discount=0.98 num_trajectories=5000 phi_layers=2 seed=1
