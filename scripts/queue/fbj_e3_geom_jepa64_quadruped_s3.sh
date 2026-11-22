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

CMD="wm=$(python tools/find_run.py JEPAWM-phimap quadruped 3 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepa64_s3 ckpt=$wm"
echo "Running: $CMD"
mkdir -p models logs
wm=$(python tools/find_run.py JEPAWM-phimap quadruped 3 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepa64_s3 ckpt=$wm
