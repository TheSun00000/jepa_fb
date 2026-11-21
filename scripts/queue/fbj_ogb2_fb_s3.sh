#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:30:00
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

CMD="python fb_jax.py env_name=antmaze_medium dataset=navigate tasks=goal num_evals_final=50 z_labels=10000 eval_chunk=2 bc_coef=0.3 z_dim=200 seed=3 eval_ckpt=models/2026-11-21_14-45-53_FB-antmaze_medium/fb_agent_final.pickle"
echo "Running: $CMD"
mkdir -p models logs
python fb_jax.py env_name=antmaze_medium dataset=navigate tasks=goal num_evals_final=50 z_labels=10000 eval_chunk=2 bc_coef=0.3 z_dim=200 seed=3 eval_ckpt=models/2026-11-21_14-45-53_FB-antmaze_medium/fb_agent_final.pickle
