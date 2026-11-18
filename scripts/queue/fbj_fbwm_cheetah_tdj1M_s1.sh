#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --account=vcj@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CMD="python fbwm.py env_name=cheetah variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 num_outer_steps=1000 eval_each_steps=50 mppi_eval_each=100 num_evals_final=20 tau=0.001 discount=0.98 ortho_init=1 q_loss=0 backward_hidden_dim=256 b_ortho_coef=0.5 target_mean=1 mix_next=1 seed=1"
echo "Running: $CMD"
mkdir -p models logs
python fbwm.py env_name=cheetah variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 mppi_reward=lstsq z_cov=dataset mppi_root_check=1 num_outer_steps=1000 eval_each_steps=50 mppi_eval_each=100 num_evals_final=20 tau=0.001 discount=0.98 ortho_init=1 q_loss=0 backward_hidden_dim=256 b_ortho_coef=0.5 target_mean=1 mix_next=1 seed=1
