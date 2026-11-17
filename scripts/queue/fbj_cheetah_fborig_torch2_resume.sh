#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CMD="python fb_orig_torch.py data_dir=/lustre/fswork/projects/rech/vcj/uyo96iu/controllable_agent/datasets/cheetah/rnd/buffer seed=1 resume=models/2026-11-17_23-31-45_FBORIG-cheetah/ckpt_latest.pt"
echo "Running: $CMD"
mkdir -p models logs
python fb_orig_torch.py data_dir=/lustre/fswork/projects/rech/vcj/uyo96iu/controllable_agent/datasets/cheetah/rnd/buffer seed=1 resume=models/2026-11-17_23-31-45_FBORIG-cheetah/ckpt_latest.pt
