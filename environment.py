import numpy as np
import torch
import envpool


def make_envpool_env(num_envs: int, seed: int = 0):
    """Create a vectorised envpool HalfCheetah env."""
    return envpool.make(
        "HalfCheetah-v4",
        env_type="gymnasium",
        num_envs=num_envs,
        seed=seed,
    )


def evaluate_policy_vectorized(policy_fn, env, num_steps: int, replay_buffer, device: str) -> float:
    """
    Roll out policy_fn in a vectorised envpool env for num_steps steps.

    policy_fn : callable (obs_tensor: Tensor (N, OBS_DIM) on device) → np.ndarray (N, ACTION_DIM)
    num_steps : number of vectorised steps; total interactions = num_steps × num_envs
    Returns   : per-env average extrinsic reward.

    Transitions are batched locally and inserted in one replay-buffer write.
    """
    obs, _ = env.reset()          # (N, OBS_DIM) float64
    num_envs  = obs.shape[0]
    obs_dim   = obs.shape[1]

    all_obs      = np.empty((num_steps * num_envs, obs_dim), dtype=np.float32)
    all_next_obs = np.empty((num_steps * num_envs, obs_dim), dtype=np.float32)
    all_actions  = np.empty((num_steps * num_envs, 6),       dtype=np.float32)
    all_done     = np.empty(num_steps * num_envs,             dtype=np.float32)

    total_reward = 0.0

    for step in range(num_steps):
        obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(device)

        with torch.no_grad():
            actions = policy_fn(obs_tensor)     # np.ndarray (N, 6)

        next_obs, reward, terminated, truncated, _ = env.step(actions)
        done = np.logical_or(terminated, truncated).astype(np.float32)

        sl = slice(step * num_envs, (step + 1) * num_envs)
        all_obs[sl]      = obs.astype(np.float32)
        all_next_obs[sl] = next_obs.astype(np.float32)
        all_actions[sl]  = actions.astype(np.float32)
        all_done[sl]     = done

        total_reward += float(reward.sum())
        obs = next_obs

    replay_buffer.add_batch(all_obs, all_actions, all_next_obs, all_done)
    return total_reward / num_envs
