#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
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

CMD="python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=ev_am_s2f_geom_fbwm; python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=10 horizon=5 mppi_norm=true seed=1 name=ev_am_s2f_fbwm_gk10_h5_norm; python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=50 horizon=5 mppi_norm=true seed=1 name=ev_am_s2f_fbwm_gk50_h5_norm; python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=200 horizon=5 mppi_norm=true seed=1 name=ev_am_s2f_fbwm_gk200_h5_norm"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=ev_am_s2f_geom_fbwm
python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=10 horizon=5 mppi_norm=true seed=1 name=ev_am_s2f_fbwm_gk10_h5_norm
python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=50 horizon=5 mppi_norm=true seed=1 name=ev_am_s2f_fbwm_gk50_h5_norm
python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-22_00-24-08_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle goal_k=200 horizon=5 mppi_norm=true seed=1 name=ev_am_s2f_fbwm_gk200_h5_norm
