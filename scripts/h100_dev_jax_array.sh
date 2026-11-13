#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%A_%a.out
#SBATCH --error=.slurm_logs/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --array=0-2  # 3 experiments


module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env

nvidia-smi

ls /usr/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


# COMMANDS=(
#     "python tdmpc_fb_2.py env_name=cheetah eval_mode=both"
#     "python tdmpc_fb_2.py env_name=cheetah eval_mode=both"

#     "python tdmpc_fb_2.py env_name=walker eval_mode=both"
#     "python tdmpc_fb_2.py env_name=walker eval_mode=both"
#     "python tdmpc_fb_2.py env_name=walker eval_mode=both"
#     "python fb_jax.py env_name=walker"

#     "python tdmpc_fb_2.py env_name=quadruped eval_mode=both"
#     "python tdmpc_fb_2.py env_name=quadruped eval_mode=both"
#     "python tdmpc_fb_2.py env_name=quadruped eval_mode=both"
#     "python fb_jax.py env_name=quadruped"
# )

# COMMANDS=(
#     "python tdmpc_fb_2.py env_name=cheetah eval_mode=both f_encoder_grad=0.5"
#     "python tdmpc_fb_2.py env_name=cheetah eval_mode=both f_encoder_grad=1"
#     "python tdmpc_fb_2.py env_name=cheetah eval_mode=both f_encoder_grad=2"
#     "python tdmpc_fb_2.py env_name=cheetah eval_mode=both f_encoder_grad=5"
# )

# COMMANDS=(
    # "python tdmpc_jax.py env_name=cheetah task=vel2 eval_mode=both; python tdmpc_jax.py env_name=cheetah task=vel10 eval_mode=both"
    # "python tdmpc_jax.py env_name=cheetah task=vel10Back eval_mode=both; python tdmpc_jax.py env_name=cheetah task=vel2Back eval_mode=both"
    # "python tdmpc_jax.py env_name=cheetah task=flip eval_mode=both; python tdmpc_jax.py env_name=walker task=stand eval_mode=both"
    # "python tdmpc_jax.py env_name=walker task=walk eval_mode=both; python tdmpc_jax.py env_name=walker task=vel8 eval_mode=both"
    # "python tdmpc_jax.py env_name=walker task=run eval_mode=both; python tdmpc_jax.py env_name=walker task=flip eval_mode=both"

    # "python tdmpc_jax.py env_name=quadruped task=stand eval_mode=both; python tdmpc_jax.py env_name=quadruped task=walk eval_mode=both"
    # "python tdmpc_jax.py env_name=quadruped task=run eval_mode=both; python tdmpc_jax.py env_name=quadruped task=walk eval_mode=jump"
# )

# COMMANDS=(
#     "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both"
#     "python tdmpc_fb_3.py env_name=walker    eval_mode=both"
#     "python tdmpc_fb_3.py env_name=quadruped eval_mode=both"
# )


# COMMANDS=(
#     "python fb_modified.py env_name=cheetah phi_identity=True"
#     "python fb_modified.py env_name=walker phi_identity=True"
#     "python fb_modified.py env_name=quadruped phi_identity=True"
# )

# COMMANDS=(
#     "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both  phi_identity=True"
#     "python tdmpc_fb_3.py env_name=walker    eval_mode=both  phi_identity=True"
#     "python tdmpc_fb_3.py env_name=quadruped eval_mode=both  phi_identity=True"
# )

# COMMANDS=(
#     "python tdmpc_fb_3.py env_name=cheetah  eval_mode=both  f_phi=False b_phi=True  pi_phi=False"
#     "python tdmpc_fb_3.py env_name=cheetah  eval_mode=both  f_phi=True  b_phi=False pi_phi=False"
#     "python tdmpc_fb_3.py env_name=cheetah  eval_mode=both  f_phi=False b_phi=False  pi_phi=True"
# )

# COMMANDS=(
    # "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=1.0"
    # "python tdmpc_fb_3.py env_name=walker    eval_mode=both bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=1.0"
    # "python tdmpc_fb_3.py env_name=quadruped eval_mode=both bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=1.0"
# )

# COMMANDS=(
#     "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both consistency_coef=0.0 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=walker    eval_mode=both consistency_coef=0.0 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=quadruped eval_mode=both consistency_coef=0.0 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
# )

# COMMANDS=(
#     "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both consistency_coef=0.0 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=walker    eval_mode=both consistency_coef=0.0 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=quadruped eval_mode=both consistency_coef=0.0 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
# )

# COMMANDS=(
#     "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both consistency_coef=0.1 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=walker    eval_mode=both consistency_coef=0.1 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=quadruped eval_mode=both consistency_coef=0.1 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
# )

# COMMANDS=(
#     "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both consistency_coef=0.5 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=walker    eval_mode=both consistency_coef=0.5 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
#     "python tdmpc_fb_3.py env_name=quadruped eval_mode=both consistency_coef=0.5 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
# )

COMMANDS=(
    "python tdmpc_fb_3.py env_name=cheetah   eval_mode=both consistency_coef=5.0 bcons_coef=2.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
    # "python tdmpc_fb_3.py env_name=walker    eval_mode=both consistency_coef=5.0 bcons_coef=2.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
    "python tdmpc_fb_3.py env_name=quadruped eval_mode=both consistency_coef=5.0 bcons_coef=2.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0"
)



CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"
eval "$CMD"
