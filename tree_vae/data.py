"""
Parse reward-function trees + fitnesses out of bestRun/training.csv.

Each learner cell is written by lisr.py as:  "<repr(tree)> (<fitness:.1f>)"
e.g.  max(next_obs_0, obs_8) (-160.3)

The grammar has fixed arity per operator (symbolic_tree.ARITY), so a plain
recursive-descent parser reconstructs the tree unambiguously. We reuse
symbolic_tree.SymbolicNode so parsed trees stay evaluable.
"""
import os
import sys
import csv
import re
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C                       # noqa: E402
from symbolic_tree import SymbolicNode, ARITY  # noqa: E402

# " <expr> (<float>)" -> split the trailing fitness. " (" never occurs inside an
# expr (operators print as "op(" with no leading space; args are joined ", ").
_FITNESS_RE = re.compile(r"^(.*) \((-?\d+\.?\d*)\)$")
_TOKEN_RE = re.compile(r"[(),]|[^\s(),]+")


@dataclass
class TreeSample:
    tree: SymbolicNode
    fitness: float
    generation: int
    repr_str: str


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_expr(expr: str) -> SymbolicNode:
    """Recursive-descent parse of a repr-style expression string."""
    toks = _TOKEN_RE.findall(expr)
    pos = 0

    def nxt():
        nonlocal pos
        t = toks[pos]
        pos += 1
        return t

    def parse():
        t = nxt()
        if t in ARITY:                       # operator node
            assert nxt() == "(", f"expected '(' after {t}"
            children = []
            for i in range(ARITY[t]):
                if i > 0:
                    assert nxt() == ",", "expected ','"
                children.append(parse())
            assert nxt() == ")", "expected ')'"
            return SymbolicNode(t, children=children)
        if _is_number(t):                    # const leaf (printed as a float)
            return SymbolicNode("const", const_val=float(t))
        return SymbolicNode(t)               # named terminal (obs_*, next_obs_*, action_*)

    node = parse()
    assert pos == len(toks), f"trailing tokens in {expr!r}"
    return node


def split_cell(cell: str):
    """Return (expr_str, fitness) from a learner CSV cell, or None if empty/bad."""
    cell = cell.strip()
    if not cell:
        return None
    m = _FITNESS_RE.match(cell)
    if not m:
        return None
    return m.group(1), float(m.group(2))


def load_samples(csv_path: str = None) -> list[TreeSample]:
    """Load every learner tree (all generations) from bestRun/training.csv."""
    csv_path = csv_path or C.BEST_RUN_CSV
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        tree_cols = [c for c in reader.fieldnames if c.endswith("_tree_fitness")]
        for row in reader:
            gen = int(float(row["generation"]))
            for col in tree_cols:
                parsed = split_cell(row.get(col, ""))
                if parsed is None:
                    continue
                expr, fit = parsed
                try:
                    tree = parse_expr(expr)
                except (AssertionError, IndexError, ValueError) as e:
                    print(f"  [warn] failed to parse gen={gen} {col}: {e}")
                    continue
                samples.append(TreeSample(tree, fit, gen, expr))
    return samples


def dedup(samples: list[TreeSample]) -> list[TreeSample]:
    """Keep the first occurrence of each distinct tree (by repr)."""
    seen = set()
    out = []
    for s in samples:
        if s.repr_str in seen:
            continue
        seen.add(s.repr_str)
        out.append(s)
    return out


if __name__ == "__main__":
    samples = load_samples()
    uniq = dedup(samples)
    print(f"loaded {len(samples)} trees, {len(uniq)} unique")
    # Round-trip check: parsed tree should re-repr to the original string.
    mismatches = sum(1 for s in samples if repr(s.tree) != s.repr_str)
    print(f"round-trip mismatches: {mismatches}")
    if samples:
        s = samples[0]
        print("example:", s.repr_str, "| fitness:", s.fitness, "| gen:", s.generation)
