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

CMD="ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=0 diag_geom=1 name=cap_fbwm_am_s3_geom; ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=10 horizon=5 mppi_norm=true seed=1 name=cap_fbwm_am_s3_gk10_h5_norm; ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=50 horizon=5 mppi_norm=true seed=1 name=cap_fbwm_am_s3_gk50_h5_norm; ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=200 horizon=5 mppi_norm=true seed=1 name=cap_fbwm_am_s3_gk200_h5_norm"
echo "Running: $CMD"
mkdir -p models logs
ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=0 diag_geom=1 name=cap_fbwm_am_s3_geom
ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=10 horizon=5 mppi_norm=true seed=1 name=cap_fbwm_am_s3_gk10_h5_norm
ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=50 horizon=5 mppi_norm=true seed=1 name=cap_fbwm_am_s3_gk50_h5_norm
ck=$(python tools/find_run.py FBWM-gphi antmaze_medium 3 fbwm_gphi_final.pickle discount=0.99 phi_hidden_dim=512 !eval_ckpt) && python wm_goal.py env_name=antmaze_medium dataset=navigate ckpt=$ck goal_k=200 horizon=5 mppi_norm=true seed=1 name=cap_fbwm_am_s3_gk200_h5_norm
