import numpy as np
import random

import torch

import config as cfg


# ------------------------------------------------------------------
# Operator catalogue (paper appendix)
# ------------------------------------------------------------------

ARITY = {
    # binary
    "add": 2, "subtract": 2, "multiply": 2,
    "max": 2, "min": 2,
    "pass_greater": 2, "pass_smaller": 2,
    "equal_to": 2, "protected_div": 2,
    # unary
    "cos": 1, "sin": 1, "tan": 1,
    "square": 1, "is_negative": 1,
    "div_by_100": 1, "div_by_10": 1,
    # ternary
    "gate": 3,
}

OPERATORS = list(ARITY.keys())

# Per-dimension state/action terminals (paper: tree maps s_i, a_i, s'_i to a scalar).
TERMINALS = (
    [f"obs_{i}"      for i in range(cfg.OBS_DIM)]
    + [f"next_obs_{i}" for i in range(cfg.OBS_DIM)]
    + [f"action_{i}"   for i in range(cfg.ACTION_DIM)]
    + ["const"]
)


# ------------------------------------------------------------------
# Vectorised primitive implementations (GPU-native via PyTorch)
# ------------------------------------------------------------------

def _add(a, b):           return a + b
def _subtract(a, b):      return a - b
def _multiply(a, b):      return a * b
def _max(a, b):           return torch.maximum(a, b)
def _min(a, b):           return torch.minimum(a, b)
def _pass_greater(a, b):  return torch.where(a > b, a, b)
def _pass_smaller(a, b):  return torch.where(a < b, a, b)
def _equal_to(a, b):      return (a == b).float()
def _protected_div(a, b): return torch.nan_to_num(a / b, nan=1.0, posinf=1.0, neginf=1.0)

def _cos(a):              return torch.cos(a)
def _sin(a):              return torch.sin(a)
def _tan(a):              return torch.tan(a)
def _square(a):           return a * a
def _is_negative(a):      return (a < 0).float()
def _div_by_100(a):       return a / 100.0
def _div_by_10(a):        return a / 10.0

def _gate(left, right, condition):
    return torch.where(condition <= 0, left, right)


_OP_FNS = {
    "add": _add, "subtract": _subtract, "multiply": _multiply,
    "max": _max, "min": _min,
    "pass_greater": _pass_greater, "pass_smaller": _pass_smaller,
    "equal_to": _equal_to, "protected_div": _protected_div,
    "cos": _cos, "sin": _sin, "tan": _tan,
    "square": _square, "is_negative": _is_negative,
    "div_by_100": _div_by_100, "div_by_10": _div_by_10,
    "gate": _gate,
}


# ------------------------------------------------------------------
# Tree node
# ------------------------------------------------------------------

class SymbolicNode:
    def __init__(self, value, children=None, const_val=None):
        self.value = value
        self.children = list(children) if children else []
        self.const_val = const_val

    def is_leaf(self):
        return len(self.children) == 0

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        obs, next_obs : (B, OBS_DIM) float32 tensor
        action        : (B, ACTION_DIM) float32 tensor in [-1, 1]
        returns       : (B,) float32 tensor
        """
        v = self.value
        if v in _OP_FNS:
            child_vals = [c.evaluate(obs, action, next_obs) for c in self.children]
            return _OP_FNS[v](*child_vals)

        if v == "const":
            return torch.full((obs.shape[0],), self.const_val, dtype=torch.float32, device=obs.device)
        if v.startswith("next_obs_"):
            return next_obs[:, int(v[9:])]
        if v.startswith("obs_"):
            return obs[:, int(v[4:])]
        if v.startswith("action_"):
            return action[:, int(v[7:])]

        raise ValueError(f"Unknown node value: {v}")

    def compile_eval(self):
        """Pre-resolve this tree to a fast callable fn(obs, action, next_obs) -> (B,).

        Resolves terminals to column selectors and operators to function refs ONCE,
        so per-step evaluation skips the startswith checks, int(v[...]) parsing,
        dict lookups, and torch.full re-allocation that `evaluate` repeats every
        call. Pure function of the current structure (nothing cached on the node),
        so after a genetic operator rewrites the tree you just recompile.
        Numerically identical to `evaluate`.
        """
        v = self.value
        if v in _OP_FNS:
            fn = _OP_FNS[v]
            cs = [c.compile_eval() for c in self.children]
            if len(cs) == 1:
                (c0,) = cs
                return lambda o, a, n: fn(c0(o, a, n))
            if len(cs) == 2:
                c0, c1 = cs
                return lambda o, a, n: fn(c0(o, a, n), c1(o, a, n))
            if len(cs) == 3:
                c0, c1, c2 = cs
                return lambda o, a, n: fn(c0(o, a, n), c1(o, a, n), c2(o, a, n))
            return lambda o, a, n: fn(*[c(o, a, n) for c in cs])
        if v == "const":
            c = float(self.const_val)
            return lambda o, a, n: o.new_full((o.shape[0],), c)
        if v.startswith("next_obs_"):
            i = int(v[9:]); return lambda o, a, n: n[:, i]
        if v.startswith("obs_"):
            i = int(v[4:]); return lambda o, a, n: o[:, i]
        if v.startswith("action_"):
            i = int(v[7:]); return lambda o, a, n: a[:, i]
        raise ValueError(f"Unknown node value: {v}")

    def clone(self):
        return SymbolicNode(
            self.value,
            children=[c.clone() for c in self.children],
            const_val=self.const_val,
        )

    def all_nodes(self):
        nodes = [self]
        for c in self.children:
            nodes.extend(c.all_nodes())
        return nodes

    def __repr__(self):
        if self.is_leaf():
            if self.value == "const":
                return f"{self.const_val:.3f}"
            return self.value
        args = ", ".join(repr(c) for c in self.children)
        return f"{self.value}({args})"


# ------------------------------------------------------------------
# Random tree generation
# ------------------------------------------------------------------

def _random_terminal() -> SymbolicNode:
    terminal = random.choice(TERMINALS)
    const_val = random.uniform(-1.0, 1.0) if terminal == "const" else None
    return SymbolicNode(terminal, const_val=const_val)


def generate_random_tree(max_depth: int, depth: int = 0) -> SymbolicNode:
    force_leaf = depth >= max_depth or (depth > 0 and random.random() < 0.3)
    if force_leaf:
        return _random_terminal()

    op = random.choice(OPERATORS)
    children = [generate_random_tree(max_depth, depth + 1) for _ in range(ARITY[op])]
    return SymbolicNode(op, children=children)


# ------------------------------------------------------------------
# Genetic operators
# ------------------------------------------------------------------

def _replace_node_in_place(target: SymbolicNode, source: SymbolicNode):
    target.value = source.value
    target.const_val = source.const_val
    target.children = source.children


def crossover(tree1: SymbolicNode, tree2: SymbolicNode):
    """Single-point GP crossover. Returns two children with swapped subtrees."""
    child1 = tree1.clone()
    child2 = tree2.clone()

    nodes1 = child1.all_nodes()
    nodes2 = child2.all_nodes()

    point1 = random.choice(nodes1[1:] if len(nodes1) > 1 else nodes1)
    point2 = random.choice(nodes2[1:] if len(nodes2) > 1 else nodes2)

    # Swap subtree contents in-place (each node keeps its own arity/value)
    point1.value, point2.value = point2.value, point1.value
    point1.const_val, point2.const_val = point2.const_val, point1.const_val
    point1.children, point2.children = point2.children, point1.children

    return child1, child2


def _cap_depth(node: SymbolicNode, max_depth: int, depth: int = 0) -> SymbolicNode:
    """Replace any node at or beyond max_depth with a random terminal."""
    if depth >= max_depth:
        return _random_terminal()
    node.children = [_cap_depth(c, max_depth, depth + 1) for c in node.children]
    return node


def mutate(tree: SymbolicNode, max_depth: int) -> SymbolicNode:
    """Replace a random subtree with a freshly generated one, capped at max_depth."""
    mutant = tree.clone()
    nodes = mutant.all_nodes()
    target = random.choice(nodes)
    replacement = generate_random_tree(max_depth)
    _replace_node_in_place(target, replacement)
    return _cap_depth(mutant, max_depth)
