import torch

ENV_ID = "ALE/Breakout-v5"
FRAME_STACK = 4
IMG_SIZE = 84
NUM_ACTIONS = 4

# EA actor population
EA_POP_SIZE = 10       # k
EA_ELITE_SIZE = 1      # e
MUT_PROB = 0.9
MUT_NOISE_STD = 0.05
TOURNAMENT_SIZE = 3

# Portfolio of SR learners
PORTFOLIO_SIZE = 5     # m
PORTFOLIO_ELITE_SIZE = 1  # j
MAX_TREE_DEPTH = 3

# SAC-D hyperparameters
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2            # entropy temperature
LR_Q = 3e-4
LR_PI = 3e-4
BATCH_SIZE = 256       # T

# Replay buffer
BUFFER_SIZE = 100_000

# Training schedule
EVAL_STEPS = 500       # environment steps per evaluation per actor/learner
GRAD_STEPS_PER_GEN = 100  # gradient updates per learner per generation
NUM_GENERATIONS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
