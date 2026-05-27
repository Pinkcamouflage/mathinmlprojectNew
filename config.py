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
MAX_TREE_DEPTH = 3  # paper: trees restricted to 3 operational layers
TREE_IMMIGRANTS = 5  # fresh random trees injected each generation to fight premature convergence

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

# Numerical stability: symbolic rewards are unbounded (tan, square, multiply can
# explode), which destabilises the Q-targets and can NaN the networks.
REWARD_CLIP = 1000.0  # sanitise + clamp r_hat to [-REWARD_CLIP, REWARD_CLIP]
GRAD_CLIP   = 10.0     # clamp per-element gradients as a safety net

# Rollout parallelism
NUM_EVAL_ENVS   = 8    # parallel envs stepped per actor evaluation (all are used)
ENVPOOL_THREADS = 1    # C++ threads per envpool instance; cross-actor parallelism
                       # comes from the ThreadPoolExecutor below
NUM_EVAL_WORKERS = 12  # concurrent actor rollouts == physical cores
EVAL_EVERY    = 5      # run the deterministic champion eval every N generations
EVAL_EPISODES = 5      # parallel episodes per deterministic eval

# Replay buffer
BUFFER_SIZE = 1_000_000

GRAD_STEPS_PER_GEN = 1000
MAX_FRAMES = 150_000_000  # paper: 150M frames for continuous control

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
