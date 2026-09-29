import torch
import jax

import random


from brax.io import torch as io_torch
from brax.envs.wrappers.training import VmapWrapper 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cpu_device = jax.devices("cpu")[0]
gpu_device = jax.devices("gpu")[0]



class TorchWrapper:
    def __init__(self, env_name, env, num_envs, state_dim=None):
        
        self.env_name = env_name
        self.env = VmapWrapper(env, batch_size=num_envs)
        
        self.num_envs = num_envs
        self.state_dim = state_dim if state_dim is not None else env.observation_size
        self.action_dim = env.action_size
        
        
        # self.reset_fn = jax.jit(self.env.reset)
        # self.step_fn = jax.jit(self.env.step)
        
        
        self.reset_fn = jax.vmap(jax.jit(env.reset))
        self.step_fn = jax.vmap(jax.jit(env.step))
        
    
    def reset(self, seed=None):
        
        rng = jax.random.PRNGKey(random.randint(0, 99999999))
        
        reset_rng = jax.random.split(rng, self.num_envs)
        state = self.reset_fn(reset_rng)
        
        # state = self.reset_fn(rng)
        
        self.state = state
        
        
        return io_torch.jax_to_torch(state.obs), {}
    
    def step(self, action: torch.Tensor):
        
        action = io_torch.torch_to_jax(action)
        action = jax.device_put(action, gpu_device)
        next_state = self.step_fn(self.state, action)
        
        obs = self.state.obs
        reward, next_obs, done = next_state.reward, next_state.obs, next_state.done      
                  
        obs = io_torch.jax_to_torch(obs, device=device)
        action = io_torch.jax_to_torch(action, device=device)
        next_obs = io_torch.jax_to_torch(next_obs, device=device)
        reward = io_torch.jax_to_torch(reward, device=device)
        done = io_torch.jax_to_torch(done, device=device)
        
        info = next_state.info['state_descriptor']
            

        
        
        self.state = next_state
        
        
        return obs, action, reward, next_obs, done, info


