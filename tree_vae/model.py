"""
D-VAE-style variational autoencoder over reward-function trees.

Encoder : asynchronous message passing leaf->root (order-aware aggregation +
          GRU update); the root state -> (mu, logvar).
Decoder : top-down autoregressive generation. The number of children a node
          gets is fixed by the chosen node type's arity, so every decoded tree
          is structurally valid by construction.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C                 # noqa: E402
from symbolic_tree import SymbolicNode  # noqa: E402
from graph import TreeGraph            # noqa: E402


class TreeVAE(nn.Module):
    def __init__(self, hidden=C.HIDDEN_DIM, latent=C.LATENT_DIM):
        super().__init__()
        self.hidden = hidden
        self.latent = latent

        # ---- shared node featurization (encoder side) ----
        self.enc_type_emb = nn.Embedding(C.NUM_TYPES, hidden)
        self.enc_const    = nn.Linear(1, hidden)
        self.enc_pos      = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(C.MAX_ARITY)])
        self.enc_gru      = nn.GRUCell(hidden, hidden)
        self.fc_mu        = nn.Linear(hidden, latent)
        self.fc_logvar    = nn.Linear(hidden, latent)

        # ---- decoder ----
        self.z_to_h       = nn.Linear(latent, hidden)
        self.classifier   = nn.Linear(hidden, C.NUM_TYPES)
        self.dec_type_emb = nn.Embedding(C.NUM_TYPES, hidden)
        self.dec_const    = nn.Linear(1, hidden)
        self.dec_gru      = nn.GRUCell(hidden, hidden)
        self.child_init   = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(C.MAX_ARITY)])
        self.const_head   = nn.Linear(hidden, 1)

        # additive penalty that forbids operators once the depth budget is spent
        self.register_buffer("_op_penalty",
                             torch.tensor([float("-inf") if o else 0.0 for o in C.IS_OP]))

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def encode(self, g: TreeGraph):
        device = g.node_types.device
        N = g.node_types.shape[0]
        hs = [None] * N
        for i in range(N):                       # post-order: children done first
            x = self.enc_type_emb(g.node_types[i]) + self.enc_const(g.const_vals[i].view(1))
            kids = g.children[i]
            if kids:
                agg = sum(self.enc_pos[p](hs[c]) for p, c in enumerate(kids))
            else:
                agg = torch.zeros(self.hidden, device=device)
            hs[i] = self.enc_gru(x.view(1, -1), agg.view(1, -1)).view(-1)
        h_root = hs[g.root]
        return self.fc_mu(h_root), self.fc_logvar(h_root)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    # Decoder (teacher-forced reconstruction loss)
    # ------------------------------------------------------------------
    def recon_loss(self, z, g: TreeGraph):
        ce_terms = []
        const_terms = []

        def expand(idx, h_ctx):
            logits = self.classifier(h_ctx.view(1, -1))
            target = g.node_types[idx].view(1)
            ce_terms.append(F.cross_entropy(logits, target))

            cval = g.const_vals[idx].view(1)
            if g.node_types[idx].item() == C.CONST_IDX:
                pred = self.const_head(h_ctx.view(1, -1)).view(1)
                const_terms.append(F.mse_loss(pred, cval))

            x = self.dec_type_emb(g.node_types[idx]) + self.dec_const(cval)
            h_node = self.dec_gru(x.view(1, -1), h_ctx.view(1, -1)).view(-1)
            for p, c in enumerate(g.children[idx]):
                expand(c, torch.tanh(self.child_init[p](h_node)))

        h_root = torch.tanh(self.z_to_h(z))
        expand(g.root, h_root)

        ce = torch.stack(ce_terms).mean()
        const = torch.stack(const_terms).mean() if const_terms else torch.zeros((), device=z.device)
        return ce, const

    def forward(self, g: TreeGraph):
        mu, logvar = self.encode(g)
        z = self.reparam(mu, logvar)
        ce, const = self.recon_loss(z, g)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return ce, const, kl, mu, logvar

    # ------------------------------------------------------------------
    # Teacher-forced node stats (for honest reconstruction metrics)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def teacher_forced_stats(self, z, g: TreeGraph):
        """Walk the true structure; report per-node type hits and const abs-error.

        Returns (node_correct, node_total, const_abs_err_sum, const_count).
        """
        stats = [0, 0, 0.0, 0]   # correct, total, abs_err, n_const

        def expand(idx, h_ctx):
            logits = self.classifier(h_ctx.view(1, -1)).view(-1)
            pred = int(torch.argmax(logits).item())
            true = int(g.node_types[idx].item())
            stats[1] += 1
            stats[0] += int(pred == true)

            cval = g.const_vals[idx].view(1)
            if true == C.CONST_IDX:
                p = self.const_head(h_ctx.view(1, -1)).view(1)
                stats[2] += float(torch.abs(p - cval).item())
                stats[3] += 1

            x = self.dec_type_emb(g.node_types[idx]) + self.dec_const(cval)
            h_node = self.dec_gru(x.view(1, -1), h_ctx.view(1, -1)).view(-1)
            for p_, c in enumerate(g.children[idx]):
                expand(c, torch.tanh(self.child_init[p_](h_node)))

        expand(g.root, torch.tanh(self.z_to_h(z)))
        return tuple(stats)

    # ------------------------------------------------------------------
    # Free decoding: latent -> SymbolicNode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, z, max_depth=C.MAX_TREE_DEPTH, sample=False):
        def expand(h_ctx, depth):
            logits = self.classifier(h_ctx.view(1, -1)).view(-1)
            if depth >= max_depth:               # force a terminal: forbid operators
                logits = logits + self._op_penalty
            if sample:
                idx = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            else:
                idx = int(torch.argmax(logits).item())

            name = C.IDX_TO_TYPE[idx]
            if idx == C.CONST_IDX:
                cval = float(self.const_head(h_ctx.view(1, -1)).item())
                node = SymbolicNode("const", const_val=cval)
                cval_t = torch.tensor([cval], device=z.device)
            else:
                node = SymbolicNode(name)
                cval_t = torch.zeros(1, device=z.device)

            x = self.dec_type_emb(torch.tensor(idx, device=z.device)) + self.dec_const(cval_t)
            h_node = self.dec_gru(x.view(1, -1), h_ctx.view(1, -1)).view(-1)
            for p in range(C.ARITY_BY_IDX[idx]):
                node.children.append(expand(torch.tanh(self.child_init[p](h_node)), depth + 1))
            return node

        h_root = torch.tanh(self.z_to_h(z))
        return expand(h_root, 0)
