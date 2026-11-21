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

CMD="python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk10_r25; python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk10_r1; python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk50_r25; python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk50_r1; python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk10_r25; python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk10_r1; python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk50_r25; python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk50_r1"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk10_r25
python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk10_r1
python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk50_r25
python wm_goal.py env_name=antmaze_medium navigate models/2026-11-21_13-23-49_FBWM-gphi-antmaze_medium/fbwm_gphi_final.pickle models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk50_r1
python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk10_r25
python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=10 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk10_r1
python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=25 seed=1 name=cem_wm_gk50_r25
python wm_goal.py env_name=cube_single play models/2026-11-21_22-24-09_FBWM-gphi-cube_single/fbwm_gphi_final.pickle models/2026-11-21_22-26-54_JEPAWM-cube_single/wm_final.pickle dataset= ckpt= goal_k=50 planner=cem horizon=25 receding=1 seed=1 name=cem_wm_gk50_r1
