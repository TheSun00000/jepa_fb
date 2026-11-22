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

CMD="python wm_goal.py env_name=quadruped diag_geom=1 name=geom_fbwm_s1 ckpt=models/2026-11-15_09-11-46_FBWM-gphi-quadruped/fbwm_gphi_final.pickle; python wm_goal.py env_name=quadruped diag_geom=1 name=geom_fbwm_s2 ckpt=models/2026-11-15_17-30-24_FBWM-gphi-quadruped__/fbwm_gphi_final.pickle; python wm_goal.py env_name=quadruped diag_geom=1 name=geom_fbwm_s3 ckpt=models/2026-11-15_17-30-40_FBWM-gphi-quadruped/fbwm_gphi_final.pickle; python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepamlp_s1 ckpt=models/2026-11-15_11-30-02_JEPAWM-quadruped/wm_final.pickle; python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepamlp_s2 ckpt=models/2026-11-15_12-31-51_JEPAWM-quadruped/wm_final.pickle; python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepamlp_s3 ckpt=models/2026-11-15_12-31-54_JEPAWM-quadruped/wm_final.pickle"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=quadruped diag_geom=1 name=geom_fbwm_s1 ckpt=models/2026-11-15_09-11-46_FBWM-gphi-quadruped/fbwm_gphi_final.pickle
python wm_goal.py env_name=quadruped diag_geom=1 name=geom_fbwm_s2 ckpt=models/2026-11-15_17-30-24_FBWM-gphi-quadruped__/fbwm_gphi_final.pickle
python wm_goal.py env_name=quadruped diag_geom=1 name=geom_fbwm_s3 ckpt=models/2026-11-15_17-30-40_FBWM-gphi-quadruped/fbwm_gphi_final.pickle
python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepamlp_s1 ckpt=models/2026-11-15_11-30-02_JEPAWM-quadruped/wm_final.pickle
python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepamlp_s2 ckpt=models/2026-11-15_12-31-51_JEPAWM-quadruped/wm_final.pickle
python wm_goal.py env_name=quadruped diag_geom=1 name=geom_jepamlp_s3 ckpt=models/2026-11-15_12-31-54_JEPAWM-quadruped/wm_final.pickle
