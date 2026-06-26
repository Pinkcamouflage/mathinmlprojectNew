import numpy as np
import torch
import envpool

import config as cfg


def make_envpool_env(seed: int = 0, num_envs: int | None = None,
                     num_threads: int | None = None):
    """Create an envpool HalfCheetah instance with `num_envs` parallel environments.

    `num_threads` overrides cfg.ENVPOOL_THREADS for callers that step one large
    pool (e.g. vectorized multi-learner evaluation) and want it spread over cores.
    """
    if num_envs is None:
        num_envs = cfg.NUM_EVAL_ENVS
    if num_threads is None:
        num_threads = cfg.ENVPOOL_THREADS
    return envpool.make(
        "HalfCheetah-v4",
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=num_threads,
        seed=seed,
    )


def evaluate_policy(policy_fn, env, replay_buffer, device: str) -> tuple[float, int]:
    """
    Algorithm 3 (vectorised): run one synchronised episode across all parallel
    envs and store every transition in the shared replay buffer.

    HalfCheetah has no early termination, so all envs truncate together at the
    time limit; the rollout stops on the first `done`. Fitness is the mean
    undiscounted episodic return across the parallel envs — a lower-variance
    estimate of the same quantity the paper uses.

    Returns (mean episodic return, frames collected = steps * num_envs).
    """
    obs, _ = env.reset()
    num_envs = obs.shape[0]

    ep_obs, ep_actions, ep_next_obs, ep_done = [], [], [], []
    returns = np.zeros(num_envs, dtype=np.float64)
    steps = 0

    while True:
        obs_t = torch.from_numpy(obs.astype("float32")).to(device)
        with torch.no_grad():
            action = policy_fn(obs_t)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = np.logical_or(terminated, truncated)

        ep_obs.append(obs.astype("float32"))
        ep_actions.append(action.astype("float32"))
        ep_next_obs.append(next_obs.astype("float32"))
        ep_done.append(done.astype("float32"))
        returns += reward
        steps += 1

        obs = next_obs
        if done.any():
            break

    replay_buffer.add_batch(
        np.concatenate(ep_obs,      axis=0),
        np.concatenate(ep_actions,  axis=0),
        np.concatenate(ep_next_obs, axis=0),
        np.concatenate(ep_done,     axis=0),
    )
    return float(returns.mean()), steps * num_envs


def evaluate_policy_deterministic(policy_fn, num_episodes: int, device: str) -> float:
    """
    Run `num_episodes` parallel episodes in one batched rollout — paper-comparable
    episodic return. Returns the mean return.
    """
    env = envpool.make(
        "HalfCheetah-v4",
        env_type="gymnasium",
        num_envs=num_episodes,
        num_threads=cfg.ENVPOOL_THREADS,
        seed=9999,
    )
    obs, _ = env.reset()
    returns = np.zeros(num_episodes, dtype=np.float64)
    while True:
        obs_t = torch.from_numpy(obs.astype("float32")).to(device)
        with torch.no_grad():
            action = policy_fn(obs_t)
        obs, reward, terminated, truncated, _ = env.step(action)
        returns += reward
        if np.logical_or(terminated, truncated).any():
            break
    env.close()
    return float(returns.mean())
