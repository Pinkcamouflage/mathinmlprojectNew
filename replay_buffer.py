import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Store as uint8 to save memory; normalize to float on sampling
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, next_obs: np.ndarray, done: bool):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.done[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)

        obs = torch.tensor(self.obs[indices], dtype=torch.float32, device=self.device) / 255.0
        next_obs = torch.tensor(self.next_obs[indices], dtype=torch.float32, device=self.device) / 255.0
        actions = torch.tensor(self.actions[indices], dtype=torch.long, device=self.device)
        done = torch.tensor(self.done[indices], dtype=torch.float32, device=self.device)

        # Also return CPU numpy copies for symbolic tree evaluation
        obs_np = obs.cpu().numpy()
        next_obs_np = next_obs.cpu().numpy()
        actions_np = self.actions[indices].copy()

        return obs, actions, next_obs, done, obs_np, actions_np, next_obs_np

    def __len__(self):
        return self.size
