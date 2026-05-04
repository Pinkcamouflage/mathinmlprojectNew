"""
LISR — Learning Intrinsic Symbolic Rewards in Reinforcement Learning
Algorithm 1 implementation for Breakout (discrete actions).
"""

import random
import os
import torch

import config as cfg
from replay_buffer import ReplayBuffer
from symbolic_tree import generate_random_tree, crossover as tree_crossover, mutate as tree_mutate
from ea_actor import EAActor, crossover as actor_crossover, mutate as actor_mutate, tournament_select
from learner import SRLearner
from environment import make_env, evaluate_policy


# ---------------------------------------------------------------------------
# Portfolio initialisation (Algorithm 2)
# ---------------------------------------------------------------------------

def init_portfolio(num_learners: int) -> list:
    """Create m SR learners, each with a randomly generated symbolic tree."""
    return [
        SRLearner(generate_random_tree(cfg.MAX_TREE_DEPTH), cfg.NUM_ACTIONS, cfg)
        for _ in range(num_learners)
    ]


# ---------------------------------------------------------------------------
# Helper: tournament selection on symbolic trees
# ---------------------------------------------------------------------------

def _tree_tournament(portfolio: list, fitness: list, k: int = 3):
    """Return the symbolic tree of the tournament winner (no clone — caller clones)."""
    candidates = random.sample(range(len(portfolio)), min(k, len(portfolio)))
    winner = max(candidates, key=lambda i: fitness[i])
    return portfolio[winner].symbolic_tree


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_lisr(log_dir: str = "./lisr_logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.csv")

    print(f"Device: {cfg.DEVICE}")
    print(f"EA pop={cfg.EA_POP_SIZE}, elites={cfg.EA_ELITE_SIZE}")
    print(f"Portfolio size={cfg.PORTFOLIO_SIZE}, elites={cfg.PORTFOLIO_ELITE_SIZE}")

    # --- Initialise components (lines 1-3) ---
    portfolio = init_portfolio(cfg.PORTFOLIO_SIZE)
    ea_pop = [EAActor(cfg.NUM_ACTIONS, cfg.DEVICE) for _ in range(cfg.EA_POP_SIZE)]
    replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE, obs_shape=(4, 84, 84), device=cfg.DEVICE)

    # Separate environments for each actor / learner to allow independent resets
    ea_envs = [make_env(seed=i) for i in range(cfg.EA_POP_SIZE)]
    learner_envs = [make_env(seed=100 + i) for i in range(cfg.PORTFOLIO_SIZE)]

    with open(log_path, "w") as f:
        f.write("generation,best_ea_fitness,best_learner_fitness,buffer_size\n")

    best_learner_fitness = -float("inf")
    best_learner: SRLearner = None

    # -----------------------------------------------------------------------
    # Generation loop (line 5)
    # -----------------------------------------------------------------------
    for gen in range(1, cfg.NUM_GENERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"Generation {gen}/{cfg.NUM_GENERATIONS}  |  buffer={len(replay_buffer)}")

        # -------------------------------------------------------------------
        # EA actor evaluation (lines 6-7)
        # -------------------------------------------------------------------
        ea_fitness = []
        for i, actor in enumerate(ea_pop):
            policy_fn = _make_ea_policy(actor)
            fit = evaluate_policy(policy_fn, ea_envs[i], cfg.EVAL_STEPS, replay_buffer, cfg.DEVICE)
            ea_fitness.append(fit)
        print(f"  EA fitness:  {[f'{f:.1f}' for f in ea_fitness]}")

        # -------------------------------------------------------------------
        # Rank EA actors (line 8) and evolve (lines 9-15)
        # -------------------------------------------------------------------
        ranked = sorted(range(cfg.EA_POP_SIZE), key=lambda i: ea_fitness[i], reverse=True)

        elites = [ea_pop[i].clone() for i in ranked[: cfg.EA_ELITE_SIZE]]

        # Tournament-select parents, apply crossover with elites, then mutate
        offspring = []
        needed = cfg.EA_POP_SIZE - cfg.EA_ELITE_SIZE
        while len(offspring) < needed:
            parent = tournament_select(ea_pop, ea_fitness, cfg.TOURNAMENT_SIZE)
            elite = random.choice(elites)
            child = actor_crossover(elite, parent)          # single-point crossover (line 12)
            if random.random() < cfg.MUT_PROB:             # mutation (lines 13-15)
                child = actor_mutate(child, cfg.MUT_NOISE_STD)
            offspring.append(child)

        ea_pop = elites + offspring

        # -------------------------------------------------------------------
        # Learner gradient updates (lines 16-24)
        # -------------------------------------------------------------------
        if len(replay_buffer) >= cfg.BATCH_SIZE:
            total_losses = {"q1_loss": 0.0, "q2_loss": 0.0, "pi_loss": 0.0}
            for learner in portfolio:
                for _ in range(cfg.GRAD_STEPS_PER_GEN):
                    batch = replay_buffer.sample(cfg.BATCH_SIZE)
                    losses = learner.update(batch)
                    for k, v in losses.items():
                        total_losses[k] += v
            n = cfg.PORTFOLIO_SIZE * cfg.GRAD_STEPS_PER_GEN
            print(f"  Learner losses — q1={total_losses['q1_loss']/n:.4f}  "
                  f"q2={total_losses['q2_loss']/n:.4f}  pi={total_losses['pi_loss']/n:.4f}")

        # -------------------------------------------------------------------
        # Learner evaluation (lines 25-26)
        # -------------------------------------------------------------------
        learner_fitness = []
        for i, learner in enumerate(portfolio):
            policy_fn = _make_learner_policy(learner)
            fit = evaluate_policy(policy_fn, learner_envs[i], cfg.EVAL_STEPS, replay_buffer, cfg.DEVICE)
            learner_fitness.append(fit)
        print(f"  Learner fitness: {[f'{f:.1f}' for f in learner_fitness]}")

        # Track best learner for checkpointing
        best_idx = max(range(cfg.PORTFOLIO_SIZE), key=lambda i: learner_fitness[i])
        if learner_fitness[best_idx] > best_learner_fitness:
            best_learner_fitness = learner_fitness[best_idx]
            best_learner = portfolio[best_idx]
            _save_checkpoint(best_learner, gen, log_dir)
            print(f"  *** New best learner fitness: {best_learner_fitness:.1f} (gen {gen}) ***")

        # -------------------------------------------------------------------
        # Rank learners and evolve symbolic trees (lines 27-32)
        # -------------------------------------------------------------------
        ranked_l = sorted(range(cfg.PORTFOLIO_SIZE), key=lambda i: learner_fitness[i], reverse=True)

        elite_learners = [portfolio[i] for i in ranked_l[: cfg.PORTFOLIO_ELITE_SIZE]]

        # Build (m - j) evolved trees via tournament selection + crossover + mutation
        new_trees = []
        needed_trees = cfg.PORTFOLIO_SIZE - cfg.PORTFOLIO_ELITE_SIZE
        while len(new_trees) < needed_trees:
            parent_tree = _tree_tournament(portfolio, learner_fitness).clone()
            elite_tree = random.choice(elite_learners).symbolic_tree
            child_tree, _ = tree_crossover(elite_tree.clone(), parent_tree)  # crossover (line 31)
            child_tree = tree_mutate(child_tree, cfg.MAX_TREE_DEPTH)         # mutation  (line 32)
            new_trees.append(child_tree)

        # Assign new trees to non-elite learner slots (neural networks are kept)
        new_portfolio = list(elite_learners)
        for i, (idx, tree) in enumerate(zip(ranked_l[cfg.PORTFOLIO_ELITE_SIZE:], new_trees)):
            portfolio[idx].symbolic_tree = tree
            new_portfolio.append(portfolio[idx])
        portfolio = new_portfolio

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        best_ea = max(ea_fitness)
        best_l = max(learner_fitness)
        with open(log_path, "a") as f:
            f.write(f"{gen},{best_ea:.2f},{best_l:.2f},{len(replay_buffer)}\n")

    # Cleanup
    for env in ea_envs + learner_envs:
        env.close()

    print("\nTraining complete.")
    print(f"Best learner extrinsic fitness: {best_learner_fitness:.2f}")
    return portfolio, ea_pop


# ---------------------------------------------------------------------------
# Policy function factories (avoid closure-capture issues in loops)
# ---------------------------------------------------------------------------

def _make_ea_policy(actor: EAActor):
    def policy_fn(obs_tensor):
        return actor.act(obs_tensor)
    return policy_fn


def _make_learner_policy(learner: SRLearner):
    @torch.no_grad()
    def policy_fn(obs_tensor):
        return int(learner.policy.act(obs_tensor, deterministic=False).item())
    return policy_fn


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(learner: SRLearner, gen: int, log_dir: str):
    path = os.path.join(log_dir, f"best_learner_gen{gen}.pt")
    torch.save({
        "policy": learner.policy.state_dict(),
        "q1": learner.q1.state_dict(),
        "q2": learner.q2.state_dict(),
        "symbolic_tree": repr(learner.symbolic_tree),
    }, path)
