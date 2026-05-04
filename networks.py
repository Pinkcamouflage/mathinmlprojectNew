import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariCNN(nn.Module):
    """
    Standard Atari CNN feature extractor.
    Input : (B, 4, 84, 84) float32
    Output: (B, 3136)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # → (B, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (B, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (B, 64, 7, 7)
            nn.ReLU(),
        )
        self.out_dim = 64 * 7 * 7  # 3136

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(start_dim=1)


class PolicyNetwork(nn.Module):
    """
    Discrete-action stochastic policy.
    Returns a probability distribution over actions.
    """
    def __init__(self, num_actions: int):
        super().__init__()
        self.cnn = AtariCNN()
        self.fc = nn.Sequential(
            nn.Linear(self.cnn.out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities (B, num_actions)."""
        logits = self.fc(self.cnn(x))
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Returns scalar action indices (B,)."""
        probs = self.forward(x)
        if deterministic:
            return probs.argmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class QNetwork(nn.Module):
    """
    Discrete-action Q-network.
    Returns Q(s, a) for all actions simultaneously.
    """
    def __init__(self, num_actions: int):
        super().__init__()
        self.cnn = AtariCNN()
        self.fc = nn.Sequential(
            nn.Linear(self.cnn.out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q values (B, num_actions)."""
        return self.fc(self.cnn(x))
