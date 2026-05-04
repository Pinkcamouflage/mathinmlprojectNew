import numpy as np
import random

OPERATORS = ["add", "sub", "mul", "div"]
TERMINALS = [
    "obs_mean", "obs_max", "obs_std",
    "next_obs_mean", "next_obs_max", "next_obs_std",
    "obs_diff_mean", "action", "const",
]


class SymbolicNode:
    def __init__(self, value, left=None, right=None, const_val=None):
        self.value = value
        self.left = left
        self.right = right
        self.const_val = const_val

    def is_leaf(self):
        return self.left is None and self.right is None

    def evaluate(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """
        obs, next_obs : (B, 4, 84, 84) float32 in [0, 1]
        action        : (B,) int64
        returns       : (B,) float32
        """
        if self.value == "add":
            return self.left.evaluate(obs, action, next_obs) + self.right.evaluate(obs, action, next_obs)
        if self.value == "sub":
            return self.left.evaluate(obs, action, next_obs) - self.right.evaluate(obs, action, next_obs)
        if self.value == "mul":
            return self.left.evaluate(obs, action, next_obs) * self.right.evaluate(obs, action, next_obs)
        if self.value == "div":
            denom = self.right.evaluate(obs, action, next_obs)
            return self.left.evaluate(obs, action, next_obs) / (np.abs(denom) + 1e-8)

        if self.value == "obs_mean":
            return obs.mean(axis=(1, 2, 3)).astype(np.float32)
        if self.value == "obs_max":
            return obs.max(axis=(1, 2, 3)).astype(np.float32)
        if self.value == "obs_std":
            return obs.std(axis=(1, 2, 3)).astype(np.float32)
        if self.value == "next_obs_mean":
            return next_obs.mean(axis=(1, 2, 3)).astype(np.float32)
        if self.value == "next_obs_max":
            return next_obs.max(axis=(1, 2, 3)).astype(np.float32)
        if self.value == "next_obs_std":
            return next_obs.std(axis=(1, 2, 3)).astype(np.float32)
        if self.value == "obs_diff_mean":
            return np.abs(next_obs - obs).mean(axis=(1, 2, 3)).astype(np.float32)
        if self.value == "action":
            # Normalize discrete action to [0, 1]
            return (action.astype(np.float32) / 3.0)
        if self.value == "const":
            return np.full(obs.shape[0], self.const_val, dtype=np.float32)

        raise ValueError(f"Unknown node value: {self.value}")

    def clone(self):
        node = SymbolicNode(
            self.value,
            left=self.left.clone() if self.left else None,
            right=self.right.clone() if self.right else None,
            const_val=self.const_val,
        )
        return node

    def all_nodes(self):
        nodes = [self]
        if self.left:
            nodes.extend(self.left.all_nodes())
        if self.right:
            nodes.extend(self.right.all_nodes())
        return nodes

    def __repr__(self):
        if self.is_leaf():
            if self.value == "const":
                return f"{self.const_val:.3f}"
            return self.value
        return f"({self.left} {self.value} {self.right})"


def generate_random_tree(max_depth: int, depth: int = 0) -> SymbolicNode:
    force_leaf = depth >= max_depth or (depth > 0 and random.random() < 0.3)
    if force_leaf:
        terminal = random.choice(TERMINALS)
        const_val = random.uniform(-1.0, 1.0) if terminal == "const" else None
        return SymbolicNode(terminal, const_val=const_val)

    op = random.choice(OPERATORS)
    left = generate_random_tree(max_depth, depth + 1)
    right = generate_random_tree(max_depth, depth + 1)
    return SymbolicNode(op, left, right)


def crossover(tree1: SymbolicNode, tree2: SymbolicNode):
    """Single-point GP crossover. Returns two children (clones with swapped subtrees)."""
    child1 = tree1.clone()
    child2 = tree2.clone()

    nodes1 = child1.all_nodes()
    nodes2 = child2.all_nodes()

    # Prefer non-root crossover points when possible
    point1 = random.choice(nodes1[1:] if len(nodes1) > 1 else nodes1)
    point2 = random.choice(nodes2[1:] if len(nodes2) > 1 else nodes2)

    # Swap subtree contents in-place
    point1.value, point2.value = point2.value, point1.value
    point1.const_val, point2.const_val = point2.const_val, point1.const_val
    point1.left, point2.left = point2.left, point1.left
    point1.right, point2.right = point2.right, point1.right

    return child1, child2


def mutate(tree: SymbolicNode, max_depth: int) -> SymbolicNode:
    """Replace a random subtree with a freshly generated one."""
    mutant = tree.clone()
    nodes = mutant.all_nodes()
    target = random.choice(nodes)
    replacement = generate_random_tree(max_depth)

    target.value = replacement.value
    target.const_val = replacement.const_val
    target.left = replacement.left
    target.right = replacement.right

    return mutant
