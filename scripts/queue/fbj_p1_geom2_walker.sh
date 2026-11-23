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

CMD="python wm_goal.py env_name=walker diag_geom=1 name=geom2_fbwm_s1 ckpt=models/2026-11-15_00-29-59_FBWM-gphi-walker_/fbwm_gphi_final.pickle; python wm_goal.py env_name=walker diag_geom=1 name=geom2_fbwm_s2 ckpt=models/2026-11-15_01-35-43_FBWM-gphi-walker/fbwm_gphi_final.pickle; python wm_goal.py env_name=walker diag_geom=1 name=geom2_fbwm_s3 ckpt=models/2026-11-15_01-37-21_FBWM-gphi-walker/fbwm_gphi_final.pickle; wm=$(python tools/find_run.py JEPAWM-phimap walker 1 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=walker diag_geom=1 name=geom2_jepa64_s1 ckpt=$wm; wm=$(python tools/find_run.py JEPAWM-phimap walker 2 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=walker diag_geom=1 name=geom2_jepa64_s2 ckpt=$wm; wm=$(python tools/find_run.py JEPAWM-phimap walker 3 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=walker diag_geom=1 name=geom2_jepa64_s3 ckpt=$wm"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=walker diag_geom=1 name=geom2_fbwm_s1 ckpt=models/2026-11-15_00-29-59_FBWM-gphi-walker_/fbwm_gphi_final.pickle
python wm_goal.py env_name=walker diag_geom=1 name=geom2_fbwm_s2 ckpt=models/2026-11-15_01-35-43_FBWM-gphi-walker/fbwm_gphi_final.pickle
python wm_goal.py env_name=walker diag_geom=1 name=geom2_fbwm_s3 ckpt=models/2026-11-15_01-37-21_FBWM-gphi-walker/fbwm_gphi_final.pickle
wm=$(python tools/find_run.py JEPAWM-phimap walker 1 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=walker diag_geom=1 name=geom2_jepa64_s1 ckpt=$wm
wm=$(python tools/find_run.py JEPAWM-phimap walker 2 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=walker diag_geom=1 name=geom2_jepa64_s2 ckpt=$wm
wm=$(python tools/find_run.py JEPAWM-phimap walker 3 wm_final.pickle enc_dim=64 !var_coef !target !reg) && python wm_goal.py env_name=walker diag_geom=1 name=geom2_jepa64_s3 ckpt=$wm
