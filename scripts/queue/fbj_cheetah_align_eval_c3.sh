#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:40:00
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

CMD="python fb_jax.py env_name=cheetah z_dim=50 num_evals_final=20 seed=1 eval_ckpt=models/2026-11-17_19-56-08_FB-cheetah/fb_agent_final.pickle label_align=0; python fb_jax.py env_name=cheetah z_dim=50 num_evals_final=20 seed=1 eval_ckpt=models/2026-11-17_19-56-08_FB-cheetah/fb_agent_final.pickle label_align=1"
echo "Running: $CMD"
mkdir -p models logs
python fb_jax.py env_name=cheetah z_dim=50 num_evals_final=20 seed=1 eval_ckpt=models/2026-11-17_19-56-08_FB-cheetah/fb_agent_final.pickle label_align=0
python fb_jax.py env_name=cheetah z_dim=50 num_evals_final=20 seed=1 eval_ckpt=models/2026-11-17_19-56-08_FB-cheetah/fb_agent_final.pickle label_align=1
