"""OGBench environments for the fb_jepa scripts (antmaze_medium, antmaze_large, cube_single): dataset, eval env and reward tasks.

    cfg = load_ogbench(env_name, kwargs)   # kwargs: dataset=, ogbench_dir=
cfg keys: ogb_name, state_dim, action_dim, num_trajectories, trajectory_len, obs, actions (float32, (N, dim)), eval_steps,
benchmarks [(reward_fn(aux, param), label, param)], make_eval_env, env_reset, env_step.
The aux of a state is the full observation. Reward-free: the dataset's own rewards are never used.
"""
import os
import numpy as np
import jax.numpy as jnp

OGBENCH_ENVS = ['antmaze_medium', 'antmaze_large', 'cube_single']
XY_VEL_SCALE = 0.33820298


def goal_reaching_reward(state, goal):          # antmaze: +1 within 2 m of the goal (x, y), else -1
    dist = jnp.linalg.norm(state[..., :2] - jnp.asarray(goal, dtype=jnp.float32), axis=-1)
    return (dist < 2).astype(jnp.float32) * 2 - 1


def velocity_reward(states, direction):         # antmaze: obs[15:17] = (vx, vy)
    xy_vels = states[..., 15:17] * XY_VEL_SCALE
    return jnp.sum(xy_vels * jnp.asarray(direction, dtype=jnp.float32), axis=-1)


def cube_goal_reward(state, goal):              # cube: obs[19:22] = block position (OGBench-rescaled); +1 within 0.04 m
    dist = jnp.linalg.norm(state[..., 19:22] - jnp.asarray(goal, dtype=jnp.float32), axis=-1)
    return (dist < 0.4).astype(jnp.float32) * 2 - 1


def cube_block_dir(state, direction):           # cube, dense: block displacement along a table axis
    return state[..., 19] * direction[0] + state[..., 20] * direction[1]


def _cube_scale(p):                             # OGBench observation scaling: x -> (x - 0.425) * 10, y -> y * 10, z -> z * 10
    return np.array([(p[0] - 0.425) * 10, p[1] * 10, p[2] * 10], dtype=np.float32)


def load_ogbench(env_name, kwargs):
    import ogbench
    ogb_dir = os.path.expanduser(kwargs.get('ogbench_dir', '~/.ogbench/data'))
    if env_name == 'cube_single':
        dataset_kind = kwargs.get('dataset', 'play')
        assert dataset_kind in ('play', 'noisy'), 'dataset=play|noisy for cube_single'
        ogb_name, eval_steps = f'cube-single-{dataset_kind}-v0', 200      # official episode length
    else:
        dataset_kind = kwargs.get('dataset', 'explore')
        assert dataset_kind in ('explore', 'stitch', 'navigate'), 'dataset=explore|stitch|navigate for antmaze'
        ogb_name, eval_steps = f"antmaze-{env_name.split('_')[1]}-{dataset_kind}-v0", 1000
    print('OGBench dataset:', ogb_name, 'from', ogb_dir)
    env0, dataset = ogbench.make_env_and_datasets(ogb_name, dataset_dir=ogb_dir, compact_dataset=True)[:2]
    state_dim, action_dim = env0.observation_space.shape[0], env0.action_space.shape[0]
    # compact dataset: 'terminals' is 1 on the last TWO rows of a trajectory, 'valids' is 0 on the last one only
    n_traj = int((np.asarray(dataset['valids']) == 0).sum())
    traj_len = dataset['terminals'].shape[0] // n_traj
    assert n_traj * traj_len == dataset['terminals'].shape[0], 'unexpected OGBench trajectory layout'
    n = n_traj * traj_len
    obs = np.asarray(dataset['observations'], dtype=np.float32)[:n].reshape(-1, state_dim)
    actions = np.asarray(dataset['actions'], dtype=np.float32)[:n].reshape(-1, action_dim)

    tasks = env0.unwrapped.task_infos
    if env_name == 'cube_single':
        benchmarks = [(cube_goal_reward, f'goal_{i + 1}', _cube_scale(t['goal_xyzs'][0])) for i, t in enumerate(tasks)]
        benchmarks += [(cube_block_dir, 'block_x+', [1, 0]), (cube_block_dir, 'block_x-', [-1, 0]),
                       (cube_block_dir, 'block_y+', [0, 1]), (cube_block_dir, 'block_y-', [0, -1])]
    else:
        benchmarks = [(goal_reaching_reward, f'goal_{i + 1}', np.array(t['goal_xy'])) for i, t in enumerate(tasks)]
        benchmarks += [(velocity_reward, 'vel_left', [-1, 0]), (velocity_reward, 'vel_up', [0, 1]),
                       (velocity_reward, 'vel_down', [0, -1]), (velocity_reward, 'vel_right', [1, 0])]
    del env0, dataset

    def make_eval_env():
        return ogbench.make_env_and_datasets(ogb_name, env_only=True)

    def env_reset(e):
        return np.asarray(e.reset()[0])

    def env_step(e, action):
        return np.asarray(e.step(action)[0])   # no autoreset / truncation handling: eval_steps = episode length

    return dict(ogb_name=ogb_name, state_dim=state_dim, action_dim=action_dim, num_trajectories=n_traj, trajectory_len=traj_len,
                obs=obs, actions=actions, eval_steps=eval_steps, benchmarks=benchmarks,
                make_eval_env=make_eval_env, env_reset=env_reset, env_step=env_step)
