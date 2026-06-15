"""
Configuration + grammar vocabulary for the tree-VAE.

Named `tv_config` (not `config`) so it does not collide with the repo-root
`config.py`, which we import for OBS_DIM / ACTION_DIM and which `symbolic_tree`
depends on.
"""
import os
import sys

# Make the repo root importable (config.py, symbolic_tree.py live there).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as root_cfg                      # noqa: E402
from symbolic_tree import OPERATORS, TERMINALS, ARITY  # noqa: E402

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BEST_RUN_CSV = os.path.join(ROOT, "bestRun", "training.csv")
CKPT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
FIG_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# ------------------------------------------------------------------
# Node vocabulary (operators first, then terminals incl. "const")
# Order is taken straight from symbolic_tree so it matches the grammar exactly.
# ------------------------------------------------------------------
NODE_TYPES   = list(OPERATORS) + list(TERMINALS)
NUM_TYPES    = len(NODE_TYPES)
TYPE_TO_IDX  = {name: i for i, name in enumerate(NODE_TYPES)}
IDX_TO_TYPE  = {i: name for i, name in enumerate(NODE_TYPES)}
CONST_IDX    = TYPE_TO_IDX["const"]

ARITY_BY_IDX = [ARITY.get(name, 0) for name in NODE_TYPES]   # terminals -> 0
IS_OP        = [name in ARITY for name in NODE_TYPES]
MAX_ARITY    = max(ARITY_BY_IDX)                              # 3 (gate)

# Trees are restricted to this many operational layers (mirror the search).
MAX_TREE_DEPTH = root_cfg.MAX_TREE_DEPTH

# ------------------------------------------------------------------
# Model hyperparameters
# ------------------------------------------------------------------
HIDDEN_DIM = 128
LATENT_DIM = 32      # raised from 16: the log showed KL wanted ~19 nats of capacity
DEVICE     = "cpu"   # graphs are tiny; per-node ops are faster on CPU

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
SYNTH_CORPUS_SIZE = 5000     # random trees for grammar pretraining
EPOCHS_PRETRAIN   = 25
EPOCHS_FINETUNE   = 25
BATCH_SIZE        = 64
LR                = 1e-3
GRAD_CLIP         = root_cfg.GRAD_CLIP
KL_WARMUP_EPOCHS  = 10        # linear beta warmup 0 -> BETA_MAX
BETA_MAX          = 0.02       # lowered from 0.1: β=0.1 crushed reconstruction
                              # (ce rose 1.72 -> 2.18 as β ramped). Natural
                              # operating point was ce≈1.7, kl≈19.
