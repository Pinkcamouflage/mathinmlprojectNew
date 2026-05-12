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

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

import config as cfg
from ea_actor import EAActor, crossover as actor_crossover, mutate as actor_mutate, tournament_select
from environment import make_envpool_env, evaluate_policy_vectorized, evaluate_policy_deterministic
from learner import SRLearner
from replay_buffer import ReplayBuffer
from symbolic_tree import crossover as tree_crossover, generate_random_tree, mutate as tree_mutate


# ---------------------------------------------------------------------------
# Portfolio initialisation (Algorithm 2)
# ---------------------------------------------------------------------------

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
        return learner.act(obs_tensor, deterministic=False)   # np.ndarray (N, ACTION_DIM)
    return fn


# ---------------------------------------------------------------------------
# Concurrent evaluation helpers
# ---------------------------------------------------------------------------

def _eval_concurrent(policy_fns, envs, replay_buffer, eval_steps, device, label):
    """Evaluate all actors concurrently, return list of fitness values."""
    fitness = [None] * len(policy_fns)

    def _task(idx):
        return idx, evaluate_policy_vectorized(
            policy_fns[idx], envs[idx], eval_steps, replay_buffer, device
        )

    with ThreadPoolExecutor(max_workers=len(policy_fns)) as pool:
        futures = {pool.submit(_task, i): i for i in range(len(policy_fns))}
        for fut in as_completed(futures):
            idx, fit = fut.result()
            fitness[idx] = fit

    print(f"  {label} fitness: {[f'{f:.1f}' for f in fitness]}")
    return fitness


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
    print(f"Envs per actor    : {cfg.NUM_ENVS_PER_ACTOR}")
    print(f"Eval steps        : {cfg.EVAL_STEPS}  "
          f"(= {cfg.EVAL_STEPS * cfg.NUM_ENVS_PER_ACTOR} env interactions per actor)")

    # --- Initialise components (lines 1-3) ---
    portfolio     = init_portfolio(cfg.PORTFOLIO_SIZE)
    ea_pop        = [EAActor(cfg.DEVICE) for _ in range(cfg.EA_POP_SIZE)]
    replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE, obs_dim=cfg.OBS_DIM, action_dim=cfg.ACTION_DIM)

    ea_envs      = [make_envpool_env(cfg.NUM_ENVS_PER_ACTOR, seed=i)       for i in range(cfg.EA_POP_SIZE)]
    learner_envs = [make_envpool_env(cfg.NUM_ENVS_PER_ACTOR, seed=100 + i) for i in range(cfg.PORTFOLIO_SIZE)]

    with open(log_path, "w") as f:
        f.write("generation,frames,mean_ea_fitness,best_ea_fitness,mean_learner_fitness,best_learner_fitness,eval_return,buffer_size\n")

    frames_per_gen = (cfg.EA_POP_SIZE + cfg.PORTFOLIO_SIZE) * cfg.NUM_ENVS_PER_ACTOR * cfg.EVAL_STEPS
    print(f"Frames per gen    : {frames_per_gen:,}  (target: {cfg.MAX_FRAMES:,})")

    best_learner_fitness = -float("inf")
    total_frames = 0
    gen = 0

    # -----------------------------------------------------------------------
    # Generation loop (line 5)
    # -----------------------------------------------------------------------
    while total_frames < cfg.MAX_FRAMES:
        gen += 1
        total_frames += frames_per_gen
        print(f"\n{'='*60}")
        print(f"Generation {gen}  |  frames={total_frames:,}/{cfg.MAX_FRAMES:,}  |  buffer={len(replay_buffer)}")

        # -------------------------------------------------------------------
        # EA actor evaluation — concurrent (lines 6-7)
        # -------------------------------------------------------------------
        ea_fitness = _eval_concurrent(
            [_ea_policy(a) for a in ea_pop],
            ea_envs, replay_buffer, cfg.EVAL_STEPS, cfg.DEVICE,
            label="EA",
        )

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
        # Learner gradient updates (lines 16-24)
        # -------------------------------------------------------------------
        if len(replay_buffer) >= max(cfg.BATCH_SIZE, cfg.EXPLORATION_STEPS):
            total_losses = {"q1_loss": 0.0, "q2_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0}
            for learner in portfolio:
                for _ in range(cfg.GRAD_STEPS_PER_GEN):
                    batch = replay_buffer.sample(cfg.BATCH_SIZE)
                    for k, v in learner.update(batch).items():
                        total_losses[k] += v
            n = cfg.PORTFOLIO_SIZE * cfg.GRAD_STEPS_PER_GEN
            print(f"  Losses — q1={total_losses['q1_loss']/n:.4f}  "
                  f"q2={total_losses['q2_loss']/n:.4f}  "
                  f"actor={total_losses['actor_loss']/n:.4f}  "
                  f"α={total_losses['alpha']/n:.4f}")

        # -------------------------------------------------------------------
        # Learner evaluation — concurrent (lines 25-26)
        # -------------------------------------------------------------------
        learner_fitness = _eval_concurrent(
            [_learner_policy(l) for l in portfolio],
            learner_envs, replay_buffer, cfg.EVAL_STEPS, cfg.DEVICE,
            label="Learner",
        )

        # Checkpoint best learner
        best_idx = max(range(cfg.PORTFOLIO_SIZE), key=lambda i: learner_fitness[i])
        if learner_fitness[best_idx] > best_learner_fitness:
            best_learner_fitness = learner_fitness[best_idx]
            _save_checkpoint(portfolio[best_idx], gen, log_dir)

        # -------------------------------------------------------------------
        # Rank learners and evolve symbolic trees (lines 27-32)
        # -------------------------------------------------------------------
        ranked_l       = sorted(range(cfg.PORTFOLIO_SIZE), key=lambda i: learner_fitness[i], reverse=True)
        elite_learners = [portfolio[i] for i in ranked_l[: cfg.PORTFOLIO_ELITE_SIZE]]

        new_trees = []
        while len(new_trees) < cfg.PORTFOLIO_SIZE - cfg.PORTFOLIO_ELITE_SIZE:
            parent_tree = _tree_tournament(portfolio, learner_fitness).clone()
            elite_tree  = random.choice(elite_learners).symbolic_tree
            child_tree, _ = tree_crossover(elite_tree.clone(), parent_tree)  # line 31
            child_tree     = tree_mutate(child_tree, cfg.MAX_TREE_DEPTH)     # line 32
            new_trees.append(child_tree)

        # Rebuild portfolio: keep elite learners, assign evolved trees to non-elite slots
        new_portfolio = list(elite_learners)
        for idx, tree in zip(ranked_l[cfg.PORTFOLIO_ELITE_SIZE:], new_trees):
            portfolio[idx].symbolic_tree = tree
            new_portfolio.append(portfolio[idx])
        portfolio = new_portfolio

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        best_ea  = max(ea_fitness)
        mean_ea  = float(np.mean(ea_fitness))
        best_l   = max(learner_fitness)
        mean_l   = float(np.mean(learner_fitness))

        # Deterministic eval of best learner — paper-comparable episodic return
        best_learner = portfolio[max(range(cfg.PORTFOLIO_SIZE), key=lambda i: learner_fitness[i])]
        eval_return  = evaluate_policy_deterministic(
            _learner_policy(best_learner), num_episodes=5, device=cfg.DEVICE
        )

        print(f"  Best EA={best_ea:.1f}  Mean EA={mean_ea:.1f}  "
              f"Best Learner={best_l:.1f}  Mean Learner={mean_l:.1f}  "
              f"Eval return={eval_return:.1f}  Overall best={best_learner_fitness:.1f}")
        with open(log_path, "a") as f:
            f.write(f"{gen},{total_frames},{mean_ea:.2f},{best_ea:.2f},"
                    f"{mean_l:.2f},{best_l:.2f},{eval_return:.2f},{len(replay_buffer)}\n")

    # Cleanup
    for env in ea_envs + learner_envs:
        env.close()

    print("\nTraining complete.")
    print(f"Best learner extrinsic fitness: {best_learner_fitness:.2f}")
    return portfolio, ea_pop
