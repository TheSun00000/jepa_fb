#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:50:00
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --account=vcj@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

CMD="(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/packtest_smi_1.txt; sleep 10; done) & SMI=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=101; kill $SMI; (while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/packtest_smi_2.txt; sleep 10; done) & SMI=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=201 & P1=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=202 & P2=$!; wait $P1 $P2; kill $SMI; (while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/packtest_smi_4.txt; sleep 10; done) & SMI=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=401 & P1=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=402 & P2=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=403 & P3=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=404 & P4=$!; wait $P1 $P2 $P3 $P4; kill $SMI"
echo "Running: $CMD"
mkdir -p models logs
(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/packtest_smi_1.txt; sleep 10; done) & SMI=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=101; kill $SMI
(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/packtest_smi_2.txt; sleep 10; done) & SMI=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=201 & P1=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=202 & P2=$!; wait $P1 $P2; kill $SMI
(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/packtest_smi_4.txt; sleep 10; done) & SMI=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=401 & P1=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=402 & P2=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=403 & P3=$!; XLA_PYTHON_CLIENT_PREALLOCATE=false python fbwm.py env_name=walker variant=gphi eval_mode=both consistency_coef=5 bcons_coef=2 f_encoder_grad=1 num_outer_steps=40 eval_each_steps=1000 num_evals_final=5 seed=404 & P4=$!; wait $P1 $P2 $P3 $P4; kill $SMI
