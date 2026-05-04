import copy
import numpy as np
import torch
import torch.nn.functional as F

from networks import PolicyNetwork, QNetwork
from symbolic_tree import SymbolicNode


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)


class SRLearner:
    """
    Symbolic-Reward Learner.
    Combines a symbolic reward tree with a Discrete SAC (SAC-D) agent.
    """

    def __init__(self, symbolic_tree: SymbolicNode, num_actions: int, cfg):
        self.symbolic_tree = symbolic_tree
        self.num_actions = num_actions
        self.cfg = cfg
        dev = cfg.DEVICE

        self.policy = PolicyNetwork(num_actions).to(dev)
        self.q1 = QNetwork(num_actions).to(dev)
        self.q2 = QNetwork(num_actions).to(dev)

        # Target networks (no gradients needed)
        self.policy_target = copy.deepcopy(self.policy).to(dev)
        self.q1_target = copy.deepcopy(self.q1).to(dev)
        self.q2_target = copy.deepcopy(self.q2).to(dev)
        for net in (self.policy_target, self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.LR_Q)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.LR_Q)
        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.LR_PI)

    def _intrinsic_reward(self, obs_np: np.ndarray, action_np: np.ndarray, next_obs_np: np.ndarray) -> torch.Tensor:
        r_hat = self.symbolic_tree.evaluate(obs_np, action_np, next_obs_np)
        return torch.tensor(r_hat, dtype=torch.float32, device=self.cfg.DEVICE)

    def update(self, batch) -> dict:
        """
        SAC-D update step.
        batch: (obs, actions, next_obs, done, obs_np, actions_np, next_obs_np)
        """
        obs, actions, next_obs, done, obs_np, actions_np, next_obs_np = batch

        # --- Intrinsic reward from symbolic tree (line 18) ---
        r_hat = self._intrinsic_reward(obs_np, actions_np, next_obs_np)

        # --- Compute TD target (line 19) ---
        with torch.no_grad():
            next_probs = self.policy_target(next_obs)          # (B, A)
            log_next = torch.log(next_probs + 1e-8)            # (B, A)
            next_q = torch.min(
                self.q1_target(next_obs),
                self.q2_target(next_obs),
            )                                                   # (B, A)
            # Discrete SAC soft-value: V(s') = Σ_a π(a|s')[Q(s',a) - α log π(a|s')]
            next_v = (next_probs * (next_q - self.cfg.ALPHA * log_next)).sum(dim=1)
            y = r_hat + self.cfg.GAMMA * (1.0 - done) * next_v  # (B,)

        # --- Update Q-networks (line 20) ---
        a_idx = actions.unsqueeze(1)  # (B, 1)

        q1_pred = self.q1(obs).gather(1, a_idx).squeeze(1)
        q1_loss = F.mse_loss(q1_pred, y)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        q2_pred = self.q2(obs).gather(1, a_idx).squeeze(1)
        q2_loss = F.mse_loss(q2_pred, y)
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # --- Update policy (line 21) ---
        probs = self.policy(obs)                                # (B, A)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            min_q = torch.min(self.q1(obs), self.q2(obs))      # (B, A)
        # Maximise E[Q - α log π]  ↔  minimise E[α log π - Q]
        pi_loss = (probs * (self.cfg.ALPHA * log_probs - min_q)).sum(dim=1).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # --- Soft-update target networks (lines 22-24) ---
        _soft_update(self.policy_target, self.policy, self.cfg.TAU)
        _soft_update(self.q1_target, self.q1, self.cfg.TAU)
        _soft_update(self.q2_target, self.q2, self.cfg.TAU)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "pi_loss": pi_loss.item(),
        }
