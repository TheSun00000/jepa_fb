#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:45:00
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

CMD="ck=$(python tools/find_run.py FBWM-gphi antmaze_large 1 fbwm_gphi_final.pickle discount=0.99 ogbench_eval=1 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=antmaze_large dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset tasks=goal ogbench_eval=1 num_evals_final=10 discount=0.99 seed=1 eval_mode=both mppi_root_check=1 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi antmaze_large 1 fbwm_gphi_final.pickle discount=0.99 ogbench_eval=1 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=antmaze_large dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset tasks=goal ogbench_eval=1 num_evals_final=10 discount=0.99 seed=1 eval_mode=policy fbgoal_official=1 eval_ckpt=$ck; ck=$(python tools/find_run.py FBWM-gphi antmaze_large 1 fbwm_gphi_final.pickle discount=0.99 ogbench_eval=1 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=antmaze_large dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset tasks=goal ogbench_eval=1 num_evals_final=10 discount=0.99 seed=1 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 fbgoal_official=1 eval_ckpt=$ck"
echo "Running: $CMD"
mkdir -p models logs
ck=$(python tools/find_run.py FBWM-gphi antmaze_large 1 fbwm_gphi_final.pickle discount=0.99 ogbench_eval=1 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=antmaze_large dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset tasks=goal ogbench_eval=1 num_evals_final=10 discount=0.99 seed=1 eval_mode=both mppi_root_check=1 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi antmaze_large 1 fbwm_gphi_final.pickle discount=0.99 ogbench_eval=1 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=antmaze_large dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset tasks=goal ogbench_eval=1 num_evals_final=10 discount=0.99 seed=1 eval_mode=policy fbgoal_official=1 eval_ckpt=$ck
ck=$(python tools/find_run.py FBWM-gphi antmaze_large 1 fbwm_gphi_final.pickle discount=0.99 ogbench_eval=1 eval_each_steps=100 !eval_ckpt) && python fbwm.py env_name=antmaze_large dataset=navigate variant=gphi z_dim=200 bc_coef=0.3 consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset tasks=goal ogbench_eval=1 num_evals_final=10 discount=0.99 seed=1 eval_mode=mppi mppi_root_check=1 mppi_init_policy=1 fbgoal_official=1 eval_ckpt=$ck
