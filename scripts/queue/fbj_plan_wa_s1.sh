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

CMD="python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mppi_root_check=1; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mppi_init_policy=1 mppi_init_std=0.5; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mixture_coef=0.25; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mixture_coef=0.5; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mppi_root_check=1 mppi_init_policy=1 mppi_init_std=0.5 mixture_coef=0.25"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mppi_root_check=1
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mppi_init_policy=1 mppi_init_std=0.5
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mixture_coef=0.25
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mixture_coef=0.5
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 z_dim=200 seed=1 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker_/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset num_evals_final=20 mppi_root_check=1 mppi_init_policy=1 mppi_init_std=0.5 mixture_coef=0.25
