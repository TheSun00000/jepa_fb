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

CMD="python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle tasks=goal_1 horizon=5 num_evals=10 seed=1 name=jepawm_goal1_h5"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=models/2026-11-21_13-23-49_JEPAWM-antmaze_medium/wm_final.pickle tasks=goal_1 horizon=5 num_evals=10 seed=1 name=jepawm_goal1_h5
