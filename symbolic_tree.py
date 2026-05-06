import numpy as np
import random


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

TERMINALS = [
    "obs_mean", "obs_std", "obs_norm",
    "next_obs_mean", "next_obs_std", "next_obs_norm",
    "obs_diff_mean",
    "action_mean", "action_norm",
    "const",
]


# ------------------------------------------------------------------
# Vectorised primitive implementations
# ------------------------------------------------------------------

def _add(a, b):           return a + b
def _subtract(a, b):      return a - b
def _multiply(a, b):      return a * b
def _max(a, b):           return np.maximum(a, b)
def _min(a, b):           return np.minimum(a, b)
def _pass_greater(a, b):  return np.where(a > b, a, b)
def _pass_smaller(a, b):  return np.where(a < b, a, b)
def _equal_to(a, b):      return (a == b).astype(np.float32)
def _protected_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(a, b)
    if isinstance(x, np.ndarray):
        x = np.where(np.isfinite(x), x, 1.0).astype(np.float32)
    else:
        x = 1.0 if not np.isfinite(x) else x
    return x

def _cos(a):              return np.cos(a)
def _sin(a):              return np.sin(a)
def _tan(a):              return np.tan(a)
def _square(a):           return a * a
def _is_negative(a):      return (a < 0).astype(np.float32)
def _div_by_100(a):       return a / 100.0
def _div_by_10(a):        return a / 10.0

def _gate(left, right, condition):
    return np.where(condition <= 0, left, right)


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

    def evaluate(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """
        obs, next_obs : (B, OBS_DIM) float32
        action        : (B, ACTION_DIM) float32 in [-1, 1]
        returns       : (B,) float32
        """
        v = self.value
        if v in _OP_FNS:
            child_vals = [c.evaluate(obs, action, next_obs) for c in self.children]
            return _OP_FNS[v](*child_vals).astype(np.float32, copy=False)

        if v == "obs_mean":
            return obs.mean(axis=1).astype(np.float32)
        if v == "obs_std":
            return obs.std(axis=1).astype(np.float32)
        if v == "obs_norm":
            return (np.linalg.norm(obs, axis=1) / np.sqrt(obs.shape[1])).astype(np.float32)
        if v == "next_obs_mean":
            return next_obs.mean(axis=1).astype(np.float32)
        if v == "next_obs_std":
            return next_obs.std(axis=1).astype(np.float32)
        if v == "next_obs_norm":
            return (np.linalg.norm(next_obs, axis=1) / np.sqrt(next_obs.shape[1])).astype(np.float32)
        if v == "obs_diff_mean":
            return np.abs(next_obs - obs).mean(axis=1).astype(np.float32)
        if v == "action_mean":
            return action.mean(axis=1).astype(np.float32)
        if v == "action_norm":
            return (np.linalg.norm(action, axis=1) / np.sqrt(action.shape[1])).astype(np.float32)
        if v == "const":
            return np.full(obs.shape[0], self.const_val, dtype=np.float32)

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


def mutate(tree: SymbolicNode, max_depth: int) -> SymbolicNode:
    """Replace a random subtree with a freshly generated one."""
    mutant = tree.clone()
    nodes = mutant.all_nodes()
    target = random.choice(nodes)
    replacement = generate_random_tree(max_depth)
    _replace_node_in_place(target, replacement)
    return mutant
