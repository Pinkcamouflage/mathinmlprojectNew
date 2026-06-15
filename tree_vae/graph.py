"""
SymbolicNode -> tensor graph for the GNN-VAE.

Nodes are indexed in post-order (children before parents), so a simple forward
sweep over indices 0..N-1 visits every child before its parent — exactly the
leaf->root order the encoder needs. The root is therefore the last index.
"""
import os
import sys
from dataclasses import dataclass

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C                 # noqa: E402
from symbolic_tree import SymbolicNode  # noqa: E402


@dataclass
class TreeGraph:
    node_types: torch.Tensor   # LongTensor [N]
    const_vals: torch.Tensor   # FloatTensor [N] (0 for non-const nodes)
    children: list             # list[list[int]] ordered child indices per node
    root: int                  # index of the root node (== N-1)

    def to(self, device):
        self.node_types = self.node_types.to(device)
        self.const_vals = self.const_vals.to(device)
        return self


def tree_to_graph(node: SymbolicNode, device: str = None) -> TreeGraph:
    device = device or C.DEVICE
    types, consts, children = [], [], []

    def visit(n: SymbolicNode) -> int:
        child_idx = [visit(c) for c in n.children]   # post-order: children first
        idx = len(types)
        types.append(C.TYPE_TO_IDX[n.value])
        consts.append(float(n.const_val) if n.value == "const" else 0.0)
        children.append(child_idx)
        return idx

    root = visit(node)
    return TreeGraph(
        node_types=torch.tensor(types, dtype=torch.long, device=device),
        const_vals=torch.tensor(consts, dtype=torch.float32, device=device),
        children=children,
        root=root,
    )
