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

CMD="python fbwm.py env_name=antmaze_medium dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle seed=1 mppi_root_check=1 fbgoal_official=1; python fbwm.py env_name=antmaze_medium dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle seed=1 mppi_root_check=0 eval_mode=mppi fbgoal_official=1"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=antmaze_medium dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle seed=1 mppi_root_check=1 fbgoal_official=1
python fbwm.py env_name=antmaze_medium dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle seed=1 mppi_root_check=0 eval_mode=mppi fbgoal_official=1
