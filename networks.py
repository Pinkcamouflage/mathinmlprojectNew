import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _mlp(in_dim: int, hidden_size: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, out_dim),
    )


class GaussianPolicy(nn.Module):
    """
    SAC stochastic policy with tanh squashing.
    forward() → (action, log_prob) using reparameterization.
    act()     → deterministic (mean) or stochastic sample, no grad.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    def forward(self, obs: torch.Tensor):
        """Returns (action (B,A), log_prob (B,)) via reparameterization."""
        h = self.trunk(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        dist  = torch.distributions.Normal(mean, std)
        x_t   = dist.rsample()
        action = torch.tanh(x_t)

        # log prob with tanh correction (numerically stable)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        h    = self.trunk(obs)
        mean = self.mean_head(h)
        if deterministic:
            return torch.tanh(mean)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        x_t = torch.distributions.Normal(mean, log_std.exp()).sample()
        return torch.tanh(x_t)


class DeterministicPolicy(nn.Module):
    """EA actor: deterministic tanh-bounded policy."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_size, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class MLPQNetwork(nn.Module):
    """Q(s, a) → scalar. Input is concat(obs, action)."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_size, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)
