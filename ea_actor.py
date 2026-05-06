import copy
import random
import numpy as np
import torch
from networks import DeterministicPolicy

import config as cfg


class EAActor:
    """EA-evolved deterministic actor for continuous control."""

    def __init__(self, device: str):
        self.device = device
        self.policy = DeterministicPolicy(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(device)

    @torch.no_grad()
    def act(self, obs_tensor: torch.Tensor) -> np.ndarray:
        """obs_tensor: (N, OBS_DIM) float32 → numpy (N, ACTION_DIM)"""
        return self.policy.act(obs_tensor).cpu().numpy()

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.policy.parameters()])

    def load_flat_params(self, flat: torch.Tensor):
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].reshape(p.shape))
            idx += n

    def clone(self) -> "EAActor":
        new = EAActor(self.device)
        new.policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))
        return new


def crossover(parent1: EAActor, parent2: EAActor) -> EAActor:
    """Single-point crossover of flattened parameter vectors. Returns one child."""
    p1 = parent1.flat_params()
    p2 = parent2.flat_params()
    cut = random.randint(0, len(p1))
    child = parent1.clone()
    child.load_flat_params(torch.cat([p1[:cut], p2[cut:]]))
    return child


@torch.no_grad()
def mutate(actor: EAActor, mutfrac: float, mutstrength: float,
           supermutprob: float, resetmutprob: float) -> EAActor:
    """
    Algorithm 4: per-weight-matrix mutation.

    For each weight matrix M (dim >= 2), sample mutfrac * |M| indices and apply:
      - with prob supermutprob : M[idx] *= N(0, 100 * mutstrength)
      - else with prob resetmutprob: M[idx] = N(0, 1)
      - else: M[idx] *= N(0, mutstrength)
    1D parameters (biases) are skipped per the algorithm spec.
    """
    mutant = actor.clone()
    for p in mutant.policy.parameters():
        if p.dim() < 2:
            continue
        num_elems     = p.numel()
        num_mutations = max(1, int(mutfrac * num_elems))
        flat    = p.data.view(-1)
        indices = torch.randint(0, num_elems, (num_mutations,), device=p.device)

        r1   = torch.rand(num_mutations, device=p.device)
        r2   = torch.rand(num_mutations, device=p.device)
        vals = flat[indices]

        super_mutated = vals * (torch.randn(num_mutations, device=p.device) * (100.0 * mutstrength))
        reset_vals    = torch.randn(num_mutations, device=p.device)
        reg_mutated   = vals * (torch.randn(num_mutations, device=p.device) * mutstrength)

        new_vals = reg_mutated
        new_vals = torch.where(r2 < resetmutprob, reset_vals, new_vals)
        new_vals = torch.where(r1 < supermutprob, super_mutated, new_vals)

        flat.scatter_(0, indices, new_vals)
    return mutant


def tournament_select(population: list, fitness: list, k: int) -> EAActor:
    """Return a clone of the winner of a k-way tournament."""
    candidates = random.sample(range(len(population)), min(k, len(population)))
    winner = max(candidates, key=lambda i: fitness[i])
    return population[winner].clone()
