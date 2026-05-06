import threading
import numpy as np


class ReplayBuffer:
    """
    Circular replay buffer for continuous-action environments.

    Stores (obs, action, next_obs, done) as float32 vectors.
    Reward is not stored — it is computed symbolically at sample time.

    Observations are small vectors (e.g. 17 floats for HalfCheetah), so
    next_obs is stored explicitly rather than recovered via a stride trick.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.pos  = 0
        self.size = 0
        self._lock = threading.Lock()

        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.done     = np.zeros(capacity,               dtype=np.float32)

    def add_batch(self, obs_batch: np.ndarray, actions_batch: np.ndarray,
                  next_obs_batch: np.ndarray, done_batch: np.ndarray):
        """Thread-safe insert of N transitions in one lock acquisition."""
        n = len(obs_batch)
        with self._lock:
            indices = np.arange(self.pos, self.pos + n) % self.capacity
            self.obs[indices]      = obs_batch
            self.actions[indices]  = actions_batch
            self.next_obs[indices] = next_obs_batch
            self.done[indices]     = done_batch.astype(np.float32)
            self.pos  = int((self.pos + n) % self.capacity)
            self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        """Returns float32 numpy arrays: (obs, actions, next_obs, done)."""
        with self._lock:
            indices  = np.random.randint(0, self.size, size=batch_size)
            obs      = self.obs[indices].copy()
            actions  = self.actions[indices].copy()
            next_obs = self.next_obs[indices].copy()
            done     = self.done[indices].copy()
        return obs, actions, next_obs, done

    def __len__(self):
        return self.size
