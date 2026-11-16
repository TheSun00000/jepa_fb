#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
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

CMD="python fbwm.py env_name=quadruped variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 latent_dim=100 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 z_labels=10000 diag_wm=1 diag_wm_h=100 diag_wm_envs=10 seed=3 eval_ckpt=models/2026-11-16_01-16-06_FBWM-gphi-quadruped/fbwm_gphi_final.pickle; python fbwm.py env_name=quadruped variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 latent_dim=100 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 z_labels=10000 diag_wm=1 diag_wm_h=100 diag_wm_envs=10 seed=1 eval_ckpt=models/2026-11-16_01-16-06_FBWM-gphi-quadruped_/fbwm_gphi_final.pickle; python fbwm.py env_name=quadruped variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 latent_dim=100 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 z_labels=10000 diag_wm=1 diag_wm_h=100 diag_wm_envs=10 seed=2 eval_ckpt=models/2026-11-16_01-16-06_FBWM-gphi-quadruped__/fbwm_gphi_final.pickle"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=quadruped variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 latent_dim=100 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 z_labels=10000 diag_wm=1 diag_wm_h=100 diag_wm_envs=10 seed=3 eval_ckpt=models/2026-11-16_01-16-06_FBWM-gphi-quadruped/fbwm_gphi_final.pickle
python fbwm.py env_name=quadruped variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 latent_dim=100 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 z_labels=10000 diag_wm=1 diag_wm_h=100 diag_wm_envs=10 seed=1 eval_ckpt=models/2026-11-16_01-16-06_FBWM-gphi-quadruped_/fbwm_gphi_final.pickle
python fbwm.py env_name=quadruped variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 latent_dim=100 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 z_labels=10000 diag_wm=1 diag_wm_h=100 diag_wm_envs=10 seed=2 eval_ckpt=models/2026-11-16_01-16-06_FBWM-gphi-quadruped__/fbwm_gphi_final.pickle
