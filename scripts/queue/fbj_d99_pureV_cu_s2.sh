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

CMD="ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 mppi_terminal_only=1 fbgoal_k=10 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 mppi_terminal_only=1 fbgoal_k=50 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 mppi_terminal_only=1 fbgoal_k=200 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=policy fbgoal_k=10 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=policy fbgoal_k=50 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=policy fbgoal_k=200 eval_ckpt=$ck"
echo "Running: $CMD"
mkdir -p models logs
ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 mppi_terminal_only=1 fbgoal_k=10 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 mppi_terminal_only=1 fbgoal_k=50 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 mppi_terminal_only=1 fbgoal_k=200 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=policy fbgoal_k=10 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=policy fbgoal_k=50 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi cube_single 2 fbwm_gphi_final.pickle discount=0.99 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=cube_single dataset=play variant=gphi z_dim=200 bc_coef=3.0 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset num_evals_final=10 discount=0.99 seed=2 eval_mode=policy fbgoal_k=200 eval_ckpt=$ck
