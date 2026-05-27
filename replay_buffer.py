import threading
import numpy as np
import torch


class ReplayBuffer:
    """
    Circular replay buffer for continuous-action environments.

    Stores (obs, action, next_obs, done) as float32 tensors on `device`.
    Reward is not stored — it is computed symbolically at sample time.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device
        self.pos  = 0
        self.size = 0
        self._lock = threading.Lock()

        self.obs      = torch.zeros((capacity, obs_dim),    dtype=torch.float32, device=device)
        self.actions  = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim),    dtype=torch.float32, device=device)
        self.done     = torch.zeros(capacity,               dtype=torch.float32, device=device)

    def add_batch(self, obs_batch: np.ndarray, actions_batch: np.ndarray,
                  next_obs_batch: np.ndarray, done_batch: np.ndarray):
        """Thread-safe insert of N transitions in one lock acquisition."""
        n = len(obs_batch)
        with self._lock:
            indices = torch.arange(self.pos, self.pos + n, device=self.device) % self.capacity
            self.obs[indices]      = torch.as_tensor(obs_batch,    dtype=torch.float32, device=self.device)
            self.actions[indices]  = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
            self.next_obs[indices] = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
            self.done[indices]     = torch.as_tensor(done_batch.astype(np.float32), dtype=torch.float32, device=self.device)
            self.pos  = int((self.pos + n) % self.capacity)
            self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        """Returns float32 tensors on device: (obs, actions, next_obs, done)."""
        with self._lock:
            indices  = torch.randint(0, self.size, (batch_size,), device=self.device)
            obs      = self.obs[indices]
            actions  = self.actions[indices]
            next_obs = self.next_obs[indices]
            done     = self.done[indices]
        return obs, actions, next_obs, done

    def sample_vectorized(self, total: int):
        """Sample `total` transitions. Returns tensors on device."""
        return self.sample(total)

    def __len__(self):
        return self.size
