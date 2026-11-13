#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00                   # Adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env

nvidia-smi

ls /usr/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


# python tdmpc_jax.py env_name=cheetah task=vel2 eval_mode=both

# python tdmpc_sf.py env_name=cheetah eval_mode=both fb_networks=true

# python fb_jax.py env_name=cheetah

# python tdmpc_fb.py env_name=cheetah eval_mode=both
# python tdmpc_fb.py env_name=cheetah consistency_coef=0 fb_encoder_grad=1 ortho_coef=1 eval_mode=policy
# python tdmpc_fb.py env_name=cheetah eval_mode=both fb_encoder_grad=1


# python fb_modified.py env_name=cheetah


# python tdmpc_fb_2.py env_name=cheetah eval_mode=both


# python tdmpc_fb_2.py env_name=cheetah eval_mode=both f_encoder_grad=0

# python tdmpc_jax.py env_name=walker task=run eval_mode=both; python tdmpc_jax.py env_name=walker task=flip eval_mode=both

# python tdmpc_fb_3.py env_name=cheetah   eval_mode=both

# python tdmpc_fb_3.py env_name=cheetah   eval_mode=both

# 'consistency_coef', 'bcons_coef', 'ortho_coef', 'b_ortho_coef', 'f_encoder_grad', 'b_encoder_grad'

# python tdmpc_fb_3.py env_name=cheetah   eval_mode=both  bcons_coef=0 ortho_coef=0 b_ortho_coef=0 f_encoder_grad=0.1 b_encoder_grad=0
# python tdmpc_fb_3.py env_name=cheetah   eval_mode=both  bcons_coef=0 ortho_coef=0 b_ortho_coef=0 f_encoder_grad=0.1 b_encoder_grad=1.0
# python tdmpc_fb_3.py env_name=cheetah   eval_mode=both  bcons_coef=1.0 ortho_coef=0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=1.0
# python tdmpc_fb_3.py env_name=cheetah   eval_mode=both  bcons_coef=0 ortho_coef=0 b_ortho_coef=0 f_encoder_grad=1.0 b_encoder_grad=1.0
# python tdmpc_fb_3.py env_name=cheetah   eval_mode=both  bcons_coef=0 ortho_coef=0 b_ortho_coef=0 f_encoder_grad=0.1 b_encoder_grad=0


# 1. Reference: the current defaults, written out explicitly
# python tdmpc_fb_3.py env_name=cheetah eval_mode=both bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=1.0

# # 2. Weaker gradient from B into phi
# python tdmpc_fb_3.py env_name=cheetah eval_mode=both bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=0.1

# # 3. FB isolated from phi: the trunk is trained by consistency only (ortho on, to avoid the rank collapse we saw before)
# python tdmpc_fb_3.py env_name=cheetah eval_mode=both bcons_coef=0.0 ortho_coef=0.5 b_ortho_coef=1.0 f_encoder_grad=0.0 b_encoder_grad=0.0

# # 4. B-space consistency on
# python tdmpc_fb_3.py env_name=cheetah eval_mode=both bcons_coef=1.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=1.0

# python fb_jax.py env_name=cheetah


# python tdmpc_fb_2.py env_name=cheetah eval_mode=both

# python fb_modified.py env_name=cheetah

# python fb_modified.py env_name=cheetah  f_phi=False b_phi=True  pi_phi=False
# python fb_modified.py env_name=cheetah  f_phi=True  b_phi=False pi_phi=False
# python fb_modified.py env_name=cheetah  f_phi=False b_phi=False  pi_phi=True

# python fb_modified.py env_name=cheetah  f_phi=True b_phi=True  pi_phi=False


# python fb_jax.py env_name=walker

# python fb_modified.py env_name=cheetah  f_phi=True b_phi=True  pi_phi=False
# python fb_modified.py env_name=walker  f_phi=True b_phi=True  pi_phi=False
# python fb_modified.py env_name=quadruped  f_phi=True b_phi=True  pi_phi=False


# python tdmpc_fb_3.py env_name=cheetah eval_mode=both bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=0.1 b_encoder_grad=1.0


python tdmpc_fb_3.py env_name=walker    eval_mode=both consistency_coef=2.0 bcons_coef=0.0 ortho_coef=0.0 b_ortho_coef=1.0 f_encoder_grad=1.0 b_encoder_grad=1.0