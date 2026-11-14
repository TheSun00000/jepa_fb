#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:40:00
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

CMD="python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 num_outer_steps=10 eval_each_steps=5 mppi_eval_each=10 num_evals_final=5 seed=7; python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 seed=1 eval_ckpt=models/2026-11-14_11-55-56_FBWM-gphi-walker/fbwm_gphi_final.pickle diag_f=1 mppi_norm=1 mppi_terminal_only=1 horizon=3 num_evals_final=5; python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=walker task=walk wm_ckpt=models/2026-11-14_17-14-43_JEPAWM-walker/wm_final.pickle n_labels=1000 reward_steps=2000 adapt_steps=2000 eval_each_steps=1 seed=1"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 fcons_coef=1 fcons_model_grad=1 num_outer_steps=10 eval_each_steps=5 mppi_eval_each=10 num_evals_final=5 seed=7
python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 seed=1 eval_ckpt=models/2026-11-14_11-55-56_FBWM-gphi-walker/fbwm_gphi_final.pickle diag_f=1 mppi_norm=1 mppi_terminal_only=1 horizon=3 num_evals_final=5
python tdmpc_adapt.py mode=adapt q_data=labeled_model env_name=walker task=walk wm_ckpt=models/2026-11-14_17-14-43_JEPAWM-walker/wm_final.pickle n_labels=1000 reward_steps=2000 adapt_steps=2000 eval_each_steps=1 seed=1
