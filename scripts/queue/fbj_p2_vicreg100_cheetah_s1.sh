#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
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

CMD="python fbwm.py env_name=cheetah eval_mode=policy variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 vicreg_var_coef=100 vicreg_cov_coef=10 eval_each_steps=25 mppi_reward=lstsq num_evals_final=20 seed=1; ck=$(python tools/find_run.py FBWM-gphi-vicreg cheetah 1 fbwm_gphi_final.pickle vicreg_var_coef=100 !eval_ckpt) && python fbwm.py env_name=cheetah eval_mode=both variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 vicreg_var_coef=100 vicreg_cov_coef=10 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 num_evals_final=20 z_labels=10000 seed=1 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi-vicreg cheetah 1 fbwm_gphi_final.pickle vicreg_var_coef=100 !eval_ckpt) && python wm_goal.py env_name=cheetah diag_geom=1 name=geom2_fbwm_vicreg100_s1 ckpt=$ck"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=cheetah eval_mode=policy variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 vicreg_var_coef=100 vicreg_cov_coef=10 eval_each_steps=25 mppi_reward=lstsq num_evals_final=20 seed=1
ck=$(python tools/find_run.py FBWM-gphi-vicreg cheetah 1 fbwm_gphi_final.pickle vicreg_var_coef=100 !eval_ckpt) && python fbwm.py env_name=cheetah eval_mode=both variant=gphi consistency_coef=5 bcons_coef=2 f_encoder_grad=1 vicreg_var_coef=100 vicreg_cov_coef=10 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 num_evals_final=20 z_labels=10000 seed=1 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi-vicreg cheetah 1 fbwm_gphi_final.pickle vicreg_var_coef=100 !eval_ckpt) && python wm_goal.py env_name=cheetah diag_geom=1 name=geom2_fbwm_vicreg100_s1 ckpt=$ck
