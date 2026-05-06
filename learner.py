import copy
import numpy as np
import torch
import torch.nn.functional as F

from networks import GaussianPolicy, MLPQNetwork
from symbolic_tree import SymbolicNode


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)


class SRLearner:
    """
    Symbolic-Reward Learner — SAC variant for continuous control.

    Uses a symbolic reward tree as the intrinsic reward signal in place of
    the environment's extrinsic reward. The SAC update then follows:
      Q target : y = r̂ + γ(1-d)(min_j Q'_j(s', a') − α log π(a'|s'))
      Actor    : maximize E[min_j Q_j(s, a_π) − α log π(a_π|s)]
      Alpha    : minimize −E[α (log π(a_π|s) + H_target)]
    """

    def __init__(self, symbolic_tree: SymbolicNode, cfg):
        self.symbolic_tree = symbolic_tree
        self.cfg = cfg
        dev = cfg.DEVICE

        self.actor    = GaussianPolicy(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(dev)
        self.q1       = MLPQNetwork(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(dev)
        self.q2       = MLPQNetwork(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(dev)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.log_alpha = torch.tensor(
            np.log(cfg.ALPHA_INIT), dtype=torch.float32, device=dev, requires_grad=True
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.LR_ACTOR)
        self.q1_opt    = torch.optim.Adam(self.q1.parameters(),    lr=cfg.LR_Q)
        self.q2_opt    = torch.optim.Adam(self.q2.parameters(),    lr=cfg.LR_Q)
        self.alpha_opt = torch.optim.Adam([self.log_alpha],        lr=cfg.LR_ALPHA)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _intrinsic_reward(self, obs_np: np.ndarray, action_np: np.ndarray,
                          next_obs_np: np.ndarray) -> torch.Tensor:
        r_hat = self.symbolic_tree.evaluate(obs_np, action_np, next_obs_np)
        return torch.tensor(r_hat, dtype=torch.float32, device=self.cfg.DEVICE)

    @torch.no_grad()
    def act(self, obs_tensor: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Returns continuous actions (B, ACTION_DIM) as numpy."""
        return self.actor.act(obs_tensor, deterministic=deterministic).cpu().numpy()

    def update(self, batch) -> dict:
        """
        SAC update step.
        batch: (obs, actions, next_obs, done) — all float32 numpy arrays.
        """
        obs_np, actions_np, next_obs_np, done_np = batch

        r_hat = self._intrinsic_reward(obs_np, actions_np, next_obs_np)

        dev      = self.cfg.DEVICE
        obs      = torch.from_numpy(obs_np).to(dev)
        actions  = torch.from_numpy(actions_np).to(dev)
        next_obs = torch.from_numpy(next_obs_np).to(dev)
        done     = torch.from_numpy(done_np).to(dev)

        # --- Q-network targets ---
        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_obs)
            min_q_next = torch.min(
                self.q1_target(next_obs, next_actions),
                self.q2_target(next_obs, next_actions),
            )
            y = r_hat + self.cfg.GAMMA * (1.0 - done) * (
                min_q_next - self.alpha.detach() * next_log_pi
            )

        # --- Update Q-networks ---
        q1_loss = F.mse_loss(self.q1(obs, actions), y)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()

        q2_loss = F.mse_loss(self.q2(obs, actions), y)
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # --- Update actor ---
        new_actions, log_pi = self.actor(obs)
        min_q_pi = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
        actor_loss = (self.alpha.detach() * log_pi - min_q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # --- Update temperature α ---
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.cfg.TARGET_ENTROPY)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # --- Soft-update target networks ---
        _soft_update(self.q1_target, self.q1, self.cfg.TAU)
        _soft_update(self.q2_target, self.q2, self.cfg.TAU)

        return {
            "q1_loss":    q1_loss.item(),
            "q2_loss":    q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha":      self.alpha.item(),
        }
