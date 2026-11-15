#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:30:00
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

CMD="python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=100 z_ridge=0.01; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=100 z_ridge=0.1; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=100 z_ridge=1.0; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=300 z_ridge=0.01; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=300 z_ridge=0.1; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=300 z_ridge=1.0; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=1000 z_ridge=0.01; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=1000 z_ridge=0.1; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=1000 z_ridge=1.0; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=10000 z_ridge=0.01"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=100 z_ridge=0.01
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=100 z_ridge=0.1
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=100 z_ridge=1.0
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=300 z_ridge=0.01
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=300 z_ridge=0.1
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=300 z_ridge=1.0
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=1000 z_ridge=0.01
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=1000 z_ridge=0.1
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=1000 z_ridge=1.0
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 z_dim=200 seed=3 eval_ckpt=models/2026-11-15_03-01-48_FBWM-gphi-walker/fbwm_gphi_final.pickle mppi_reward=lstsq z_cov=dataset z_labels=10000 z_ridge=0.01
