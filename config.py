import torch

ENV_ID = "HalfCheetah-v4"
OBS_DIM = 17
ACTION_DIM = 6

# EA actor population
EA_POP_SIZE = 25
EA_ELITE_SIZE = max(1, int(round(0.07 * EA_POP_SIZE)))  # e = 7% of k
TOURNAMENT_SIZE = 3

# Mutation parameters (Algorithm 4)
MUT_PROB = 0.9
MUT_FRAC = 0.1
MUT_STRENGTH = 0.1
SUPER_MUT_PROB = 0.05
RESET_MUT_PROB = 0.05

# Portfolio of SR learners
PORTFOLIO_SIZE = 25
PORTFOLIO_ELITE_SIZE = max(1, int(round(0.07 * (PORTFOLIO_SIZE + EA_POP_SIZE))))
MAX_TREE_DEPTH = 12

# SAC hyperparameters (continuous control)
GAMMA = 0.99
TAU = 5e-3
LR_Q = 3e-4
LR_ACTOR = 3e-4
LR_ALPHA = 3e-4
WEIGHT_DECAY = 0.0
HIDDEN_SIZE = 256
ALPHA_INIT = 0.2
TARGET_ENTROPY = -float(ACTION_DIM)  # standard SAC heuristic: -|A|

BATCH_SIZE = 256
EXPLORATION_STEPS = 5000

# Replay buffer
BUFFER_SIZE = 1_000_000

NUM_ENVS_PER_ACTOR = 8

EVAL_STEPS = 250
GRAD_STEPS_PER_GEN = 200
MAX_FRAMES = 20_000_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
