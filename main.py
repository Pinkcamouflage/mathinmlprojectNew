import random
import numpy as np
import torch

from lisr import run_lisr
import config as cfg


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seeds(42)
    print(f"Running LISR on Breakout | device={cfg.DEVICE}")
    run_lisr(log_dir="./lisr_logs")
