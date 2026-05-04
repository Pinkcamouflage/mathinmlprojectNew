import copy
import random
import torch
from networks import PolicyNetwork


class EAActor:
    """EA-evolved neural network actor. Uses greedy (deterministic) action selection."""

    def __init__(self, num_actions: int, device: str):
        self.num_actions = num_actions
        self.device = device
        self.policy = PolicyNetwork(num_actions).to(device)

    @torch.no_grad()
    def act(self, obs_tensor: torch.Tensor) -> int:
        """obs_tensor: (1, 4, 84, 84) float32 → scalar action."""
        return int(self.policy.act(obs_tensor, deterministic=True).item())

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.policy.parameters()])

    def load_flat_params(self, flat: torch.Tensor):
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].reshape(p.shape))
            idx += n

    def clone(self) -> "EAActor":
        new = EAActor(self.num_actions, self.device)
        new.policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))
        return new


def crossover(parent1: EAActor, parent2: EAActor) -> EAActor:
    """Single-point crossover of flattened parameter vectors. Returns one child."""
    p1 = parent1.flat_params()
    p2 = parent2.flat_params()
    cut = random.randint(0, len(p1))
    child_params = torch.cat([p1[:cut], p2[cut:]])
    child = parent1.clone()
    child.load_flat_params(child_params)
    return child


def mutate(actor: EAActor, noise_std: float) -> EAActor:
    """Gaussian weight perturbation. Returns a new mutated actor."""
    mutant = actor.clone()
    params = mutant.flat_params()
    mutant.load_flat_params(params + torch.randn_like(params) * noise_std)
    return mutant


def tournament_select(population: list, fitness: list, k: int) -> EAActor:
    """Return a clone of the winner of a k-way tournament."""
    candidates = random.sample(range(len(population)), min(k, len(population)))
    winner = max(candidates, key=lambda i: fitness[i])
    return population[winner].clone()
