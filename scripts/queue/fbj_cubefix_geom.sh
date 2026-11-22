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

CMD="python wm_goal.py env_name=cube_single dataset=play ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=cubefix_d99_bc3_z50_geom; python wm_goal.py env_name=cube_single dataset=play ckpt=models/2026-11-22_09-55-48_FBWM-gphi-cube_single/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=cubefix_d99_bc3_geom; python wm_goal.py env_name=cube_single dataset=play ckpt=models/2026-11-22_09-34-43_FBWM-gphi-cube_single_/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=cubefix_d99_bc1_geom"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=cube_single dataset=play ckpt=models/2026-11-22_12-29-57_FBWM-gphi-cube_single/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=cubefix_d99_bc3_z50_geom
python wm_goal.py env_name=cube_single dataset=play ckpt=models/2026-11-22_09-55-48_FBWM-gphi-cube_single/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=cubefix_d99_bc3_geom
python wm_goal.py env_name=cube_single dataset=play ckpt=models/2026-11-22_09-34-43_FBWM-gphi-cube_single_/fbwm_gphi_final.pickle goal_k=0 diag_geom=1 name=cubefix_d99_bc1_geom
