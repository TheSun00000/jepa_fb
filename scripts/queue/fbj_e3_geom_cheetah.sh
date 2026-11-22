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

CMD="python wm_goal.py env_name=cheetah diag_geom=1 name=geom_fbwm_s1 ckpt=models/2026-11-15_07-30-35_FBWM-gphi-cheetah__/fbwm_gphi_final.pickle; python wm_goal.py env_name=cheetah diag_geom=1 name=geom_fbwm_s2 ckpt=models/2026-11-15_09-00-15_FBWM-gphi-cheetah/fbwm_gphi_final.pickle; python wm_goal.py env_name=cheetah diag_geom=1 name=geom_fbwm_s3 ckpt=models/2026-11-15_09-00-15_FBWM-gphi-cheetah_/fbwm_gphi_final.pickle; python wm_goal.py env_name=cheetah diag_geom=1 name=geom_jepamlp_s1 ckpt=models/2026-11-15_11-06-02_JEPAWM-cheetah/wm_final.pickle; python wm_goal.py env_name=cheetah diag_geom=1 name=geom_jepamlp_s2 ckpt=models/2026-11-15_12-25-07_JEPAWM-cheetah/wm_final.pickle; python wm_goal.py env_name=cheetah diag_geom=1 name=geom_jepamlp_s3 ckpt=models/2026-11-15_12-31-51_JEPAWM-cheetah/wm_final.pickle"
echo "Running: $CMD"
mkdir -p models logs
python wm_goal.py env_name=cheetah diag_geom=1 name=geom_fbwm_s1 ckpt=models/2026-11-15_07-30-35_FBWM-gphi-cheetah__/fbwm_gphi_final.pickle
python wm_goal.py env_name=cheetah diag_geom=1 name=geom_fbwm_s2 ckpt=models/2026-11-15_09-00-15_FBWM-gphi-cheetah/fbwm_gphi_final.pickle
python wm_goal.py env_name=cheetah diag_geom=1 name=geom_fbwm_s3 ckpt=models/2026-11-15_09-00-15_FBWM-gphi-cheetah_/fbwm_gphi_final.pickle
python wm_goal.py env_name=cheetah diag_geom=1 name=geom_jepamlp_s1 ckpt=models/2026-11-15_11-06-02_JEPAWM-cheetah/wm_final.pickle
python wm_goal.py env_name=cheetah diag_geom=1 name=geom_jepamlp_s2 ckpt=models/2026-11-15_12-25-07_JEPAWM-cheetah/wm_final.pickle
python wm_goal.py env_name=cheetah diag_geom=1 name=geom_jepamlp_s3 ckpt=models/2026-11-15_12-31-51_JEPAWM-cheetah/wm_final.pickle
