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

CMD="python wm_baselines.py seed=1 planner=mppi plan_horizon=1 eval_k=10,50,200 model=dinowm env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_22-23-54_WMBASE-dinowm-antmaze_medium/wm_base_final.pickle; python wm_baselines.py seed=1 planner=mppi plan_horizon=1 eval_k=10,50,200 model=dinowm env_name=antmaze_large dataset=navigate ckpt=models/2026-11-21_22-23-56_WMBASE-dinowm-antmaze_large/wm_base_final.pickle; python wm_baselines.py seed=1 planner=mppi plan_horizon=1 eval_k=10,50,200 model=dinowm env_name=cube_single dataset=play ckpt=models/2026-11-21_22-23-57_WMBASE-dinowm-cube_single/wm_base_final.pickle"
echo "Running: $CMD"
mkdir -p models logs
python wm_baselines.py seed=1 planner=mppi plan_horizon=1 eval_k=10,50,200 model=dinowm env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_22-23-54_WMBASE-dinowm-antmaze_medium/wm_base_final.pickle
python wm_baselines.py seed=1 planner=mppi plan_horizon=1 eval_k=10,50,200 model=dinowm env_name=antmaze_large dataset=navigate ckpt=models/2026-11-21_22-23-56_WMBASE-dinowm-antmaze_large/wm_base_final.pickle
python wm_baselines.py seed=1 planner=mppi plan_horizon=1 eval_k=10,50,200 model=dinowm env_name=cube_single dataset=play ckpt=models/2026-11-21_22-23-57_WMBASE-dinowm-cube_single/wm_base_final.pickle
