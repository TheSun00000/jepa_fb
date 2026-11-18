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

CMD="python fb_tdjepa.py env_name=cheetah num_outer_steps=1000 num_evals_final=10 num_trajectories=5000 left_enc=0 target_min=1 shared_trunk=1 ortho_orig=1 mix_obs=1 q_loss=1 tau=0.01 b_hidden=526 single_z=0 shared_trunk=0 seed=1"
echo "Running: $CMD"
mkdir -p models logs
python fb_tdjepa.py env_name=cheetah num_outer_steps=1000 num_evals_final=10 num_trajectories=5000 left_enc=0 target_min=1 shared_trunk=1 ortho_orig=1 mix_obs=1 q_loss=1 tau=0.01 b_hidden=526 single_z=0 shared_trunk=0 seed=1
