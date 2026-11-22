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

CMD="python fbwm.py env_name=cube_single dataset=play variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle seed=1 eval_mode=policy z_dim=50 bc_coef=3.0 discount=0.99 fbgoal_k=10; python fbwm.py env_name=cube_single dataset=play variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle seed=1 eval_mode=policy z_dim=50 bc_coef=3.0 discount=0.99 fbgoal_k=50; python fbwm.py env_name=cube_single dataset=play variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle seed=1 eval_mode=policy z_dim=50 bc_coef=3.0 discount=0.99 fbgoal_official=1"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=cube_single dataset=play variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle seed=1 eval_mode=policy z_dim=50 bc_coef=3.0 discount=0.99 fbgoal_k=10
python fbwm.py env_name=cube_single dataset=play variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle seed=1 eval_mode=policy z_dim=50 bc_coef=3.0 discount=0.99 fbgoal_k=50
python fbwm.py env_name=cube_single dataset=play variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset eval_ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle seed=1 eval_mode=policy z_dim=50 bc_coef=3.0 discount=0.99 fbgoal_official=1
