import numpy as np
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


def make_env(seed: int = None) -> gym.Env:
    """Create a frame-stacked, preprocessed Breakout environment."""
    env = gym.make("ALE/Breakout-v5", frameskip=1)
    env = AtariPreprocessing(env, scale_obs=False)
    env = FrameStackObservation(env, stack_size=4)
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate_policy(policy_fn, env: gym.Env, num_steps: int, replay_buffer, device: str) -> float:
    """
    Roll out policy_fn in env for num_steps steps, storing transitions in replay_buffer.

    policy_fn : callable (obs_tensor: Tensor (1,4,84,84)) → int action
    Returns   : total extrinsic reward collected over num_steps steps.
    """
    obs, _ = env.reset()
    total_reward = 0.0

    for _ in range(num_steps):
        obs_np = np.array(obs, dtype=np.uint8)           # (4, 84, 84) uint8
        obs_f = obs_np.astype(np.float32) / 255.0
        obs_tensor = torch.tensor(obs_f, dtype=torch.float32, device=device).unsqueeze(0)

        action = policy_fn(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_obs_np = np.array(next_obs, dtype=np.uint8)
        replay_buffer.add(obs_np, action, next_obs_np, done)

        total_reward += float(reward)
        obs = next_obs

        if done:
            obs, _ = env.reset()

    return total_reward
