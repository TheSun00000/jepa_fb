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

CMD="python tdmpc_adapt.py mode=pretrain encoder=phimap enc_dim=64 lr=1e-4 batch_size=1024 update_freq=1 consistency_coef=5 env_name=walker  var_coef=0 cov_coef=0 seed=1"
echo "Running: $CMD"
mkdir -p models logs
python tdmpc_adapt.py mode=pretrain encoder=phimap enc_dim=64 lr=1e-4 batch_size=1024 update_freq=1 consistency_coef=5 env_name=walker  var_coef=0 cov_coef=0 seed=1
