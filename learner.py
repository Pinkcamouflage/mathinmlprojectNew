import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, grad, vmap

from networks import GaussianPolicy, MLPQNetwork
from symbolic_tree import SymbolicNode


def _soft_update(target: nn.Module, source: nn.Module, tau: float):
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

        self.actor     = GaussianPolicy(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(dev)
        self.q1        = MLPQNetwork(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(dev)
        self.q2        = MLPQNetwork(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(dev)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(cfg.ALPHA_INIT))),
            dtype=torch.float32, device=dev, requires_grad=True
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.LR_ACTOR)
        self.q1_opt    = torch.optim.Adam(self.q1.parameters(),    lr=cfg.LR_Q)
        self.q2_opt    = torch.optim.Adam(self.q2.parameters(),    lr=cfg.LR_Q)
        self.alpha_opt = torch.optim.Adam([self.log_alpha],        lr=cfg.LR_ALPHA)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs_tensor: torch.Tensor, deterministic: bool = False):
        """Returns continuous actions (B, ACTION_DIM) as numpy."""
        return self.actor.act(obs_tensor, deterministic=deterministic).cpu().numpy()

    def update(self, batch) -> dict:
        """
        SAC update step.
        batch: (obs, actions, next_obs, done) — all float32 tensors on cfg.DEVICE.
        """
        obs, actions, next_obs, done = batch

        r_hat = self.symbolic_tree.evaluate(obs, actions, next_obs)
        clip = self.cfg.REWARD_CLIP
        r_hat = torch.nan_to_num(r_hat, nan=0.0, posinf=clip, neginf=-clip).clamp_(-clip, clip)

        # --- Q-network targets ---
        with torch.no_grad():
            noise = torch.randn(next_obs.shape[0], self.cfg.ACTION_DIM, device=self.cfg.DEVICE)
            next_actions, next_log_pi = self.actor(next_obs, noise)
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
        noise2 = torch.randn(obs.shape[0], self.cfg.ACTION_DIM, device=self.cfg.DEVICE)
        new_actions, log_pi = self.actor(obs, noise2)
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


# ---------------------------------------------------------------------------
# Vectorized SAC updater — runs all portfolio learners in parallel via vmap
# ---------------------------------------------------------------------------

class VectorizedSACUpdater:
    """
    Batches gradient computation across all portfolio learners simultaneously
    using torch.func.vmap. Replaces the nested
        for learner in portfolio:
            for _ in range(GRAD_STEPS_PER_GEN):
                learner.update(batch)
    loop with 3 vectorized forward/backward passes per step.

    Correctness: SAC requires separate gradient computations for each component
    to avoid cross-contamination (e.g., actor loss must not update Q-networks).
    This is achieved via 3 separate vmap'd grad passes:
      Pass 1 — targets: no-grad forward to compute y = r + γ*(min_Qt - α*log_π)
      Pass 2 — Q-grads: differentiate MSE(q_pred, y) w.r.t. Q-net params only
      Pass 3 — actor+α grads: differentiate actor+alpha losses w.r.t. actor/alpha only
    """

    def __init__(self, learners: list, cfg):
        self.learners = learners
        self.cfg = cfg
        self.n = len(learners)
        self.dev = cfg.DEVICE

        # Stateless base models used for functional_call shape reference
        self._base_q1    = MLPQNetwork(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(self.dev)
        self._base_q2    = MLPQNetwork(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(self.dev)
        self._base_actor = GaussianPolicy(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(self.dev)
        for net in (self._base_q1, self._base_q2, self._base_actor):
            for p in net.parameters():
                p.requires_grad_(False)

        # Stack all learner params into (N, *param_shape) tensors
        self._stacked_q1_params,    self._stacked_q1_bufs    = stack_module_state([l.q1    for l in learners])
        self._stacked_q2_params,    self._stacked_q2_bufs    = stack_module_state([l.q2    for l in learners])
        self._stacked_actor_params, self._stacked_actor_bufs = stack_module_state([l.actor for l in learners])
        self._stacked_q1t_params,   self._stacked_q1t_bufs   = stack_module_state([l.q1_target for l in learners])
        self._stacked_q2t_params,   self._stacked_q2t_bufs   = stack_module_state([l.q2_target for l in learners])

        # Reward computation is grouped by IDENTICAL tree: each distinct reward is
        # evaluated once over all learners that share it (one batched op group
        # instead of one per learner). Huge win on a grid where many points decode
        # to the same tree. Rebuilt in sync_from_learners() when trees change.
        self._build_reward_groups()

        # log_alpha per learner — shape (N,)
        self._log_alphas = torch.stack([l.log_alpha.detach().clone() for l in learners]).to(self.dev)
        self._log_alphas.requires_grad_(True)

        # Mega-optimizers: single Adam per net type holding all N learners' params
        self._mega_q1_opt    = torch.optim.Adam(list(self._stacked_q1_params.values()),    lr=cfg.LR_Q)
        self._mega_q2_opt    = torch.optim.Adam(list(self._stacked_q2_params.values()),    lr=cfg.LR_Q)
        self._mega_actor_opt = torch.optim.Adam(list(self._stacked_actor_params.values()), lr=cfg.LR_ACTOR)
        self._alpha_opt      = torch.optim.Adam([self._log_alphas],                        lr=cfg.LR_ALPHA)

        # Build the 3 vmap'd functions
        self._vmap_targets    = self._build_target_fn()
        self._vmap_q_grads    = self._build_q_grad_fn()
        self._vmap_actor_grad = self._build_actor_alpha_grad_fn()

    # ------------------------------------------------------------------
    # Pass 1: compute y targets for all learners (no grad needed)
    # ------------------------------------------------------------------

    def _build_target_fn(self):
        base_actor = self._base_actor
        base_q1    = self._base_q1
        base_q2    = self._base_q2
        gamma      = self.cfg.GAMMA

        def target_fn(actor_p, actor_b, q1t_p, q1t_b, q2t_p, q2t_b,
                      la, next_obs, done, r_hat, noise_next):
            next_a, next_lp = functional_call(base_actor, (actor_p, actor_b), (next_obs, noise_next))
            q1t_v = functional_call(base_q1, (q1t_p, q1t_b), (next_obs, next_a))
            q2t_v = functional_call(base_q2, (q2t_p, q2t_b), (next_obs, next_a))
            return r_hat + gamma * (1.0 - done) * (
                torch.minimum(q1t_v, q2t_v) - la.exp() * next_lp
            )

        return vmap(target_fn, in_dims=(0,) * 11)

    # ------------------------------------------------------------------
    # Pass 2: Q1 and Q2 gradients (targets are pre-computed and passed in)
    # ------------------------------------------------------------------

    def _build_q_grad_fn(self):
        base_q1 = self._base_q1
        base_q2 = self._base_q2

        def q_loss_fn(q1_p, q1_b, q2_p, q2_b, obs, actions, y):
            q1_pred = functional_call(base_q1, (q1_p, q1_b), (obs, actions))
            q2_pred = functional_call(base_q2, (q2_p, q2_b), (obs, actions))
            return F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)

        # argnums=(0, 2): differentiate w.r.t. q1_p and q2_p only
        return vmap(grad(q_loss_fn, argnums=(0, 2)), in_dims=(0,) * 7)

    # ------------------------------------------------------------------
    # Pass 3: Actor and alpha gradients
    # Actor gradient flows through Q-networks but Q-net params are NOT in argnums.
    # Alpha gradient uses detached log_pi so it doesn't affect actor params.
    # Combined scalar is safe: actor_p and la are independent.
    # ------------------------------------------------------------------

    def _build_actor_alpha_grad_fn(self):
        base_actor     = self._base_actor
        base_q1        = self._base_q1
        base_q2        = self._base_q2
        target_entropy = self.cfg.TARGET_ENTROPY

        def actor_alpha_loss_fn(actor_p, actor_b, q1_p, q1_b, q2_p, q2_b,
                                la, obs, noise_actor):
            alpha = la.exp()
            new_a, lp = functional_call(base_actor, (actor_p, actor_b), (obs, noise_actor))
            # q1_p and q2_p NOT in argnums: gradients flow through them to actor_p only
            q1_v = functional_call(base_q1, (q1_p, q1_b), (obs, new_a))
            q2_v = functional_call(base_q2, (q2_p, q2_b), (obs, new_a))
            actor_loss = (alpha.detach() * lp - torch.minimum(q1_v, q2_v)).mean()
            # lp detached: alpha grad does not affect actor params
            alpha_loss = -(la * (lp.detach() + target_entropy)).mean()
            return actor_loss + alpha_loss

        # argnums=(0, 6): differentiate w.r.t. actor_p and la only
        return vmap(grad(actor_alpha_loss_fn, argnums=(0, 6)), in_dims=(0,) * 9)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _tree_key(node):
        """Exact structural key (value + const + child keys) for grouping identical trees."""
        if not node.children:
            return (node.value, node.const_val)
        return (node.value, tuple(VectorizedSACUpdater._tree_key(c) for c in node.children))

    def _build_reward_groups(self):
        """Group learners by identical reward tree -> (compiled_fn, learner-index tensor)."""
        from collections import defaultdict
        groups = defaultdict(list)
        for i, l in enumerate(self.learners):
            groups[self._tree_key(l.symbolic_tree)].append(i)
        self._reward_groups = [
            (self.learners[idxs[0]].symbolic_tree.compile_eval(),
             torch.tensor(idxs, dtype=torch.long, device=self.dev))
            for idxs in groups.values()
        ]

    @torch.no_grad()
    def act_all(self, obs, deterministic: bool = False):
        """Vectorized policy for all N learners in one vmapped forward.

        obs: (N, P, OBS_DIM) tensor on device (P = envs/episodes per learner).
        Returns numpy (N, P, ACTION_DIM). Deterministic uses zero noise, i.e.
        tanh(mean) — identical to GaussianPolicy.act(deterministic=True).
        """
        base_actor = self._base_actor

        def act_fn(actor_p, actor_b, o, noise):
            a, _ = functional_call(base_actor, (actor_p, actor_b), (o, noise))
            return a

        P = obs.shape[1]
        if deterministic:
            noise = torch.zeros(self.n, P, self.cfg.ACTION_DIM, device=self.dev)
        else:
            noise = torch.randn(self.n, P, self.cfg.ACTION_DIM, device=self.dev)
        actions = vmap(act_fn, in_dims=(0, 0, 0, 0))(
            self._stacked_actor_params, self._stacked_actor_bufs, obs, noise)
        return actions.cpu().numpy()

    def update_all(self, batch) -> dict:
        """
        Run one SAC gradient step for all N learners in parallel.
        batch: (obs, actions, next_obs, done) tensors on device, shape (N*B, ...).
        """
        obs_flat, actions_flat, next_obs_flat, done_flat = batch
        B = obs_flat.shape[0] // self.n

        obs      = obs_flat.view(self.n, B, -1)
        actions  = actions_flat.view(self.n, B, -1)
        next_obs = next_obs_flat.view(self.n, B, -1)
        done     = done_flat.view(self.n, B)

        # Intrinsic reward, grouped by identical tree: evaluate each distinct reward
        # once over the stacked batches of all learners that share it (far fewer
        # kernel launches than one evaluation per learner). Elementwise ops over the
        # batch dim make the concatenation exact.
        r_hat = obs.new_empty((self.n, B))
        for fn, idx in self._reward_groups:
            g = idx.shape[0]
            oi = obs.index_select(0, idx).reshape(g * B, -1)
            ai = actions.index_select(0, idx).reshape(g * B, -1)
            ni = next_obs.index_select(0, idx).reshape(g * B, -1)
            r_hat.index_copy_(0, idx, fn(oi, ai, ni).reshape(g, B))
        # Symbolic rewards are unbounded; sanitise NaN/inf and clamp before use.
        clip = self.cfg.REWARD_CLIP
        r_hat = torch.nan_to_num(r_hat, nan=0.0, posinf=clip, neginf=-clip).clamp_(-clip, clip)

        noise_next  = torch.randn(self.n, B, self.cfg.ACTION_DIM, device=self.dev)
        noise_actor = torch.randn(self.n, B, self.cfg.ACTION_DIM, device=self.dev)

        # Pass 1: compute y targets (no grad)
        with torch.no_grad():
            y = self._vmap_targets(
                self._stacked_actor_params, self._stacked_actor_bufs,
                self._stacked_q1t_params,   self._stacked_q1t_bufs,
                self._stacked_q2t_params,   self._stacked_q2t_bufs,
                self._log_alphas,
                next_obs, done, r_hat, noise_next,
            )  # (N, B)

        # Pass 2: Q gradients
        grads_q1, grads_q2 = self._vmap_q_grads(
            self._stacked_q1_params, self._stacked_q1_bufs,
            self._stacked_q2_params, self._stacked_q2_bufs,
            obs, actions, y,
        )
        gc = self.cfg.GRAD_CLIP
        for param, g in zip(self._stacked_q1_params.values(), grads_q1.values()):
            param.grad = torch.nan_to_num(g, nan=0.0, posinf=gc, neginf=-gc).clamp_(-gc, gc)
        self._mega_q1_opt.step()
        self._mega_q1_opt.zero_grad()

        for param, g in zip(self._stacked_q2_params.values(), grads_q2.values()):
            param.grad = torch.nan_to_num(g, nan=0.0, posinf=gc, neginf=-gc).clamp_(-gc, gc)
        self._mega_q2_opt.step()
        self._mega_q2_opt.zero_grad()

        # Pass 3: Actor + alpha gradients
        grads_actor, grads_alpha = self._vmap_actor_grad(
            self._stacked_actor_params, self._stacked_actor_bufs,
            self._stacked_q1_params,   self._stacked_q1_bufs,
            self._stacked_q2_params,   self._stacked_q2_bufs,
            self._log_alphas,
            obs, noise_actor,
        )
        for param, g in zip(self._stacked_actor_params.values(), grads_actor.values()):
            param.grad = torch.nan_to_num(g, nan=0.0, posinf=gc, neginf=-gc).clamp_(-gc, gc)
        self._mega_actor_opt.step()
        self._mega_actor_opt.zero_grad()

        self._log_alphas.grad = torch.nan_to_num(grads_alpha, nan=0.0, posinf=gc, neginf=-gc).clamp_(-gc, gc)
        self._alpha_opt.step()
        self._alpha_opt.zero_grad()

        # Soft-update stacked target networks
        tau = self.cfg.TAU
        for name in self._stacked_q1_params:
            self._stacked_q1t_params[name].data.mul_(1.0 - tau).add_(
                tau * self._stacked_q1_params[name].data)
        for name in self._stacked_q2_params:
            self._stacked_q2t_params[name].data.mul_(1.0 - tau).add_(
                tau * self._stacked_q2_params[name].data)

        return {}

    def sync_to_learners(self):
        """Write stacked param tensors back to individual SRLearner networks.

        Must be called before: checkpointing, environment evaluation, portfolio
        evolution that reads learner network weights.
        """
        for i, learner in enumerate(self.learners):
            for name, stacked in self._stacked_q1_params.items():
                _get_param(learner.q1, name).data.copy_(stacked[i])
            for name, stacked in self._stacked_q1t_params.items():
                _get_param(learner.q1_target, name).data.copy_(stacked[i])
            for name, stacked in self._stacked_q2_params.items():
                _get_param(learner.q2, name).data.copy_(stacked[i])
            for name, stacked in self._stacked_q2t_params.items():
                _get_param(learner.q2_target, name).data.copy_(stacked[i])
            for name, stacked in self._stacked_actor_params.items():
                _get_param(learner.actor, name).data.copy_(stacked[i])
            learner.log_alpha.data.copy_(self._log_alphas[i])

    def sync_from_learners(self):
        """Reload stacked params from individual SRLearner networks.

        Must be called after portfolio evolution reassigns learners (e.g. after
        tree evolution swaps which SRLearner objects are in the portfolio list).
        """
        q1_p, _  = stack_module_state([l.q1        for l in self.learners])
        q2_p, _  = stack_module_state([l.q2        for l in self.learners])
        a_p,  _  = stack_module_state([l.actor     for l in self.learners])
        q1t_p, _ = stack_module_state([l.q1_target for l in self.learners])
        q2t_p, _ = stack_module_state([l.q2_target for l in self.learners])

        for name in self._stacked_q1_params:
            self._stacked_q1_params[name].data.copy_(q1_p[name])
        for name in self._stacked_q2_params:
            self._stacked_q2_params[name].data.copy_(q2_p[name])
        for name in self._stacked_actor_params:
            self._stacked_actor_params[name].data.copy_(a_p[name])
        for name in self._stacked_q1t_params:
            self._stacked_q1t_params[name].data.copy_(q1t_p[name])
        for name in self._stacked_q2t_params:
            self._stacked_q2t_params[name].data.copy_(q2t_p[name])

        new_alphas = torch.stack([l.log_alpha.detach().clone() for l in self.learners])
        self._log_alphas.data.copy_(new_alphas)

        # Trees may have changed with the portfolio — regroup + recompile rewards.
        self._build_reward_groups()

        # Rebuild mega-optimizers with fresh parameter references
        self._mega_q1_opt    = torch.optim.Adam(list(self._stacked_q1_params.values()),    lr=self.cfg.LR_Q)
        self._mega_q2_opt    = torch.optim.Adam(list(self._stacked_q2_params.values()),    lr=self.cfg.LR_Q)
        self._mega_actor_opt = torch.optim.Adam(list(self._stacked_actor_params.values()), lr=self.cfg.LR_ACTOR)
        self._alpha_opt      = torch.optim.Adam([self._log_alphas],                        lr=self.cfg.LR_ALPHA)


def _get_param(module: nn.Module, name: str) -> nn.Parameter:
    """Navigate dotted name like 'net.0.weight' to the actual parameter."""
    parts = name.split('.')
    obj = module
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return getattr(obj, parts[-1])
