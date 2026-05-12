import numpy as np
import torch
import envpool


def make_envpool_env(seed: int = 0):
    """Create a single-env envpool HalfCheetah instance (Algorithm 3: one env per actor)."""
    return envpool.make(
        "HalfCheetah-v4",
        env_type="gymnasium",
        num_envs=1,
        seed=seed,
    )


def evaluate_policy(policy_fn, env, replay_buffer, device: str) -> tuple[float, int]:
    """
    Algorithm 3: run one complete episode, store transitions in replay buffer.
    Returns (episodic_return, num_steps).
    """
    obs, _ = env.reset()

    ep_obs, ep_actions, ep_next_obs, ep_done = [], [], [], []
    fitness = 0.0

    while True:
        obs_t = torch.from_numpy(obs.astype("float32")).to(device)
        with torch.no_grad():
            action = policy_fn(obs_t)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated[0] or truncated[0])

        ep_obs.append(obs[0].astype("float32"))
        ep_actions.append(action[0].astype("float32"))
        ep_next_obs.append(next_obs[0].astype("float32"))
        ep_done.append(float(done))
        fitness += float(reward[0])

        obs = next_obs
        if done:
            break

    replay_buffer.add_batch(
        np.array(ep_obs,      dtype=np.float32),
        np.array(ep_actions,  dtype=np.float32),
        np.array(ep_next_obs, dtype=np.float32),
        np.array(ep_done,     dtype=np.float32),
    )
    return fitness, len(ep_obs)


def evaluate_policy_deterministic(policy_fn, num_episodes: int, device: str) -> float:
    """
    Evaluate deterministically over complete episodes — paper-comparable metric.
    Returns mean episodic return.
    """
    env = envpool.make("HalfCheetah-v4", env_type="gymnasium", num_envs=1, seed=9999)
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        while True:
            obs_t = torch.from_numpy(obs.astype("float32")).to(device)
            with torch.no_grad():
                action = policy_fn(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward[0])
            if terminated[0] or truncated[0]:
                break
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))
