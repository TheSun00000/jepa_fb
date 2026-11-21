#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:30:00
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

CMD="python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=geom_fbwm_s1; python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle goal_k=0 diag_geom=1 name=geom_wm_s1"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=geom_fbwm_s1
python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle goal_k=0 diag_geom=1 name=geom_wm_s1
