"""
LISR — Learning Intrinsic Symbolic Rewards in Reinforcement Learning
Algorithm 1 implementation for HalfCheetah-v4 (continuous control).

Parallelism strategy
--------------------
- Each actor / learner owns an envpool env with NUM_ENVS_PER_ACTOR parallel instances.
- All actor evaluations within a phase run concurrently via ThreadPoolExecutor.
  * C++ env-stepping inside each thread is already parallel (envpool's thread pool).
  * GPU forward passes issued from each thread are queued by the CUDA driver and
    overlapped with the C++ env-stepping of the other threads.
- Replay-buffer writes are batched per actor (one lock acquisition per evaluation)
  to keep synchronisation overhead negligible.
"""

import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

import config as cfg
from ea_actor import EAActor, crossover as actor_crossover, mutate as actor_mutate, tournament_select
from environment import make_envpool_env, evaluate_policy, evaluate_policy_deterministic
from learner import SRLearner, VectorizedSACUpdater
from replay_buffer import ReplayBuffer
from symbolic_tree import crossover as tree_crossover, generate_random_tree, mutate as tree_mutate


# ---------------------------------------------------------------------------
# Portfolio initialisation (Algorithm 2)
# ---------------------------------------------------------------------------

def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def init_portfolio(num_learners: int) -> list:
    return [
        SRLearner(generate_random_tree(cfg.MAX_TREE_DEPTH), cfg)
        for _ in range(num_learners)
    ]


# ---------------------------------------------------------------------------
# Policy function factories — return numpy arrays for envpool compatibility
# ---------------------------------------------------------------------------

def _ea_policy(actor: EAActor):
    def fn(obs_tensor: torch.Tensor) -> np.ndarray:
        return actor.act(obs_tensor)        # np.ndarray (N, ACTION_DIM)
    return fn


def _learner_policy(learner: SRLearner):
    def fn(obs_tensor: torch.Tensor) -> np.ndarray:
        # Deterministic (mean) actions: low-variance fitness for cleaner tree
        # selection. Exploration comes from policy diversity across the population
        # (25 EA actors + 25 learners), not per-action noise.
        return learner.act(obs_tensor, deterministic=True)    # np.ndarray (N, ACTION_DIM)
    return fn


# ---------------------------------------------------------------------------
# Concurrent evaluation helpers
# ---------------------------------------------------------------------------

def _eval_concurrent(policy_fns, envs, replay_buffer, device, label):
    """Evaluate all actors concurrently. Returns (fitness list, total frames collected)."""
    fitness = [None] * len(policy_fns)
    frames  = [0]    * len(policy_fns)

    def _task(idx):
        return idx, *evaluate_policy(policy_fns[idx], envs[idx], replay_buffer, device)

    workers = min(len(policy_fns), cfg.NUM_EVAL_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, i): i for i in range(len(policy_fns))}
        for fut in as_completed(futures):
            idx, fit, n = fut.result()
            fitness[idx] = fit
            frames[idx]  = n

    print(f"  {label} fitness: {[f'{f:.1f}' for f in fitness]}")
    return fitness, sum(frames)


# ---------------------------------------------------------------------------
# Symbolic-tree tournament selection
# ---------------------------------------------------------------------------

def _tree_tournament(portfolio, fitness, k=3):
    candidates = random.sample(range(len(portfolio)), min(k, len(portfolio)))
    winner = max(candidates, key=lambda i: fitness[i])
    return portfolio[winner].symbolic_tree


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(learner: SRLearner, gen: int, log_dir: str):
    path = os.path.join(log_dir, f"best_learner_gen{gen}.pt")
    torch.save({
        "q1":           learner.q1.state_dict(),
        "q2":           learner.q2.state_dict(),
        "actor":        learner.actor.state_dict(),
        "log_alpha":    learner.log_alpha.item(),
        "symbolic_tree": repr(learner.symbolic_tree),
    }, path)
    print(f"  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_lisr(log_dir: str = "./lisr_logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.csv")

    print(f"Device            : {cfg.DEVICE}")
    print(f"EA pop={cfg.EA_POP_SIZE}, elites={cfg.EA_ELITE_SIZE}")
    print(f"Portfolio size={cfg.PORTFOLIO_SIZE}, elites={cfg.PORTFOLIO_ELITE_SIZE}")
    print(f"Eval mode         : {cfg.NUM_EVAL_ENVS} parallel envs/actor, "
          f"{cfg.NUM_EVAL_WORKERS} concurrent rollouts (Algorithm 3, vectorised)")

    # --- Initialise components (lines 1-3) ---
    portfolio     = init_portfolio(cfg.PORTFOLIO_SIZE)
    ea_pop        = [EAActor("cpu") for _ in range(cfg.EA_POP_SIZE)]  # inference-only, never gradient-trained
    replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE, obs_dim=cfg.OBS_DIM,
                                 action_dim=cfg.ACTION_DIM, device=cfg.DEVICE)
    vec_updater   = VectorizedSACUpdater(portfolio, cfg)

    ea_envs      = [make_envpool_env(seed=i)       for i in range(cfg.EA_POP_SIZE)]
    learner_envs = [make_envpool_env(seed=100 + i) for i in range(cfg.PORTFOLIO_SIZE)]

    learner_headers = [f"learner_{i}_tree_fitness" for i in range(cfg.PORTFOLIO_SIZE)]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["generation", "frames", "mean_ea_fitness", "best_ea_fitness",
             "mean_learner_fitness", "best_learner_fitness", "eval_return", "buffer_size"]
            + learner_headers
        )

    print(f"Frame budget      : {cfg.MAX_FRAMES:,}")

    best_learner_fitness = -float("inf")
    total_frames = 0
    gen = 0
    eval_return = 0.0  # refreshed every cfg.EVAL_EVERY generations
    start_time = time.monotonic()

    # -----------------------------------------------------------------------
    # Generation loop (line 5)
    # -----------------------------------------------------------------------
    while total_frames < cfg.MAX_FRAMES:
        gen += 1
        print(f"\n{'='*60}")
        print(f"Generation {gen}  |  frames={total_frames:,}/{cfg.MAX_FRAMES:,}  |  buffer={len(replay_buffer)}")

        # -------------------------------------------------------------------
        # EA actor evaluation — concurrent (lines 6-7)
        # -------------------------------------------------------------------
        ea_fitness, ea_frames = _eval_concurrent(
            [_ea_policy(a) for a in ea_pop],
            ea_envs, replay_buffer, "cpu",
            label="EA",
        )
        total_frames += ea_frames

        # -------------------------------------------------------------------
        # EA evolution (lines 8-15)
        # -------------------------------------------------------------------
        ranked = sorted(range(cfg.EA_POP_SIZE), key=lambda i: ea_fitness[i], reverse=True)
        elites = [ea_pop[i].clone() for i in ranked[: cfg.EA_ELITE_SIZE]]

        offspring = []
        while len(offspring) < cfg.EA_POP_SIZE - cfg.EA_ELITE_SIZE:
            parent = tournament_select(ea_pop, ea_fitness, cfg.TOURNAMENT_SIZE)
            child  = actor_crossover(random.choice(elites), parent)   # line 12
            if random.random() < cfg.MUT_PROB:                        # lines 13-15
                child = actor_mutate(
                    child,
                    mutfrac=cfg.MUT_FRAC,
                    mutstrength=cfg.MUT_STRENGTH,
                    supermutprob=cfg.SUPER_MUT_PROB,
                    resetmutprob=cfg.RESET_MUT_PROB,
                )
            offspring.append(child)

        ea_pop = elites + offspring

        # -------------------------------------------------------------------
        # Learner gradient updates (lines 16-24) — vectorised across all learners
        # -------------------------------------------------------------------
        if len(replay_buffer) >= max(cfg.BATCH_SIZE, cfg.EXPLORATION_STEPS):
            for _ in range(cfg.GRAD_STEPS_PER_GEN):
                batch = replay_buffer.sample_vectorized(cfg.BATCH_SIZE * cfg.PORTFOLIO_SIZE)
                vec_updater.update_all(batch)
            print(f"  Gradient updates: {cfg.GRAD_STEPS_PER_GEN} steps × {cfg.PORTFOLIO_SIZE} learners (vectorised)")

        # -------------------------------------------------------------------
        # Learner evaluation — concurrent (lines 25-26)
        # -------------------------------------------------------------------
        # Sync stacked params back to individual learner networks for inference
        vec_updater.sync_to_learners()
        learner_fitness, learner_frames = _eval_concurrent(
            [_learner_policy(l) for l in portfolio],
            learner_envs, replay_buffer, cfg.DEVICE,
            label="Learner",
        )
        total_frames += learner_frames
        learner_tree_strings = [repr(l.symbolic_tree) for l in portfolio]

        # Checkpoint best learner. Capture the champion reference now, before the
        # portfolio is reordered below — learner_fitness is indexed by the current
        # (pre-rebuild) order.
        best_idx = max(range(cfg.PORTFOLIO_SIZE), key=lambda i: learner_fitness[i])
        champion = portfolio[best_idx]
        if learner_fitness[best_idx] > best_learner_fitness:
            best_learner_fitness = learner_fitness[best_idx]
            _save_checkpoint(champion, gen, log_dir)

        # -------------------------------------------------------------------
        # Rank learners and evolve symbolic trees (lines 27-32)
        # -------------------------------------------------------------------
        ranked_l       = sorted(range(cfg.PORTFOLIO_SIZE), key=lambda i: learner_fitness[i], reverse=True)
        elite_learners = [portfolio[i] for i in ranked_l[: cfg.PORTFOLIO_ELITE_SIZE]]

        num_new        = cfg.PORTFOLIO_SIZE - cfg.PORTFOLIO_ELITE_SIZE
        num_immigrants = min(cfg.TREE_IMMIGRANTS, num_new)
        num_offspring  = num_new - num_immigrants

        new_trees = []
        while len(new_trees) < num_offspring:
            parent_tree = _tree_tournament(portfolio, learner_fitness).clone()
            elite_tree  = random.choice(elite_learners).symbolic_tree
            child_tree, _ = tree_crossover(elite_tree.clone(), parent_tree)  # line 31
            child_tree     = tree_mutate(child_tree, cfg.MAX_TREE_DEPTH)     # line 32
            new_trees.append(child_tree)
        # Random immigrants: fresh trees with no elite ancestry, assigned to the
        # worst non-elite slots to preserve population diversity.
        new_trees.extend(generate_random_tree(cfg.MAX_TREE_DEPTH) for _ in range(num_immigrants))

        # Rebuild portfolio: keep elite learners, assign evolved trees to non-elite slots
        new_portfolio = list(elite_learners)
        for idx, tree in zip(ranked_l[cfg.PORTFOLIO_ELITE_SIZE:], new_trees):
            portfolio[idx].symbolic_tree = tree
            new_portfolio.append(portfolio[idx])
        portfolio = new_portfolio

        # Reload stacked params in vectorized updater after portfolio reorder
        vec_updater.learners = portfolio
        vec_updater.sync_from_learners()

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        best_ea  = max(ea_fitness)
        mean_ea  = float(np.mean(ea_fitness))
        best_l   = max(learner_fitness)
        mean_l   = float(np.mean(learner_fitness))

        # Deterministic eval of best learner — paper-comparable episodic return
        if gen % cfg.EVAL_EVERY == 0:
            eval_return = evaluate_policy_deterministic(
                _learner_policy(champion), num_episodes=cfg.EVAL_EPISODES, device=cfg.DEVICE
            )

        print(f"  Best EA={best_ea:.1f}  Mean EA={mean_ea:.1f}  "
              f"Best Learner={best_l:.1f}  Mean Learner={mean_l:.1f}  "
              f"Eval return={eval_return:.1f}  Overall best={best_learner_fitness:.1f}")

        elapsed = time.monotonic() - start_time
        fps     = total_frames / max(elapsed, 1e-9)
        eta     = max(cfg.MAX_FRAMES - total_frames, 0) / fps if fps > 0 else 0.0
        print(f"  Elapsed={_fmt_hms(elapsed)}  FPS={fps:,.0f}  ETA={_fmt_hms(eta)}  "
              f"({100.0 * total_frames / cfg.MAX_FRAMES:.1f}% of frame budget)")
        learner_cells = [
            f"{learner_tree_strings[i]} ({learner_fitness[i]:.1f})"
            for i in range(cfg.PORTFOLIO_SIZE)
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [gen, total_frames, f"{mean_ea:.2f}", f"{best_ea:.2f}",
                 f"{mean_l:.2f}", f"{best_l:.2f}", f"{eval_return:.2f}", len(replay_buffer)]
                + learner_cells
            )

    # Cleanup
    for env in ea_envs + learner_envs:
        env.close()

    print("\nTraining complete.")
    print(f"Best learner extrinsic fitness: {best_learner_fitness:.2f}")
    return portfolio, ea_pop
