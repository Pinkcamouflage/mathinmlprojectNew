"""
Random-direction fitness-landscape surface.

Pipeline (per the request):
  1. Pick two RANDOM orthonormal latent directions (not PCA).
  2. Build a GRID x GRID grid over [-EXTENT, EXTENT]^2 along those directions.
  3. Decode every grid point z = center + a*d1 + b*d2 to an expression tree.
  4. Evaluate a fitness score for each decoded tree.
  5. Draw the height surface over the two directions (2-D contour + 3-D surface).
  6. Gaussian-smooth the height field for a nicer landscape.

True fitness = return of a SAC policy trained under the reward, which is far too
expensive for GRID*GRID points. So we use a cheap offline stand-in:

  --score proxy : decode -> evaluate the tree's reward on a fixed transition
                  batch -> SIGNED CORRELATION with the true HalfCheetah objective
                  (forward velocity - control cost). Aligned rewards score high.
  --score knn   : distance-weighted REAL logged fitness of the k nearest encoded
                  trees in latent space (uses the logged fitness labels).
"""
import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables projection='3d')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C            # noqa: E402
from graph import tree_to_graph  # noqa: E402
from visualize import load_model, encode_all  # noqa: E402
import data as data_mod          # noqa: E402


# ------------------------------------------------------------------
# Step 1: two random orthonormal directions
# ------------------------------------------------------------------
def random_directions(latent, seed):
    rng = np.random.default_rng(seed)
    d1 = rng.standard_normal(latent)
    d1 /= np.linalg.norm(d1)
    d2 = rng.standard_normal(latent)
    d2 -= (d2 @ d1) * d1          # Gram-Schmidt: make d2 orthogonal to d1
    d2 /= np.linalg.norm(d2)
    return d1.astype(np.float32), d2.astype(np.float32)


# ------------------------------------------------------------------
# Step 4 scorers
# ------------------------------------------------------------------
def make_proxy_scorer(batch=512, seed=0):
    """Signed correlation of a tree's reward with the true HalfCheetah objective."""
    g = torch.Generator().manual_seed(seed)
    obs      = torch.randn(batch, C.root_cfg.OBS_DIM, generator=g)
    next_obs = torch.randn(batch, C.root_cfg.OBS_DIM, generator=g)
    action   = torch.rand(batch, C.root_cfg.ACTION_DIM, generator=g) * 2 - 1
    # HalfCheetah reward ~ forward velocity (obs dim 8) - 0.1 * control cost
    target = next_obs[:, 8] - 0.1 * (action ** 2).sum(dim=1)
    target = (target - target.mean())
    tgt_std = target.std()

    def score(tree):
        try:
            r = tree.evaluate(obs, action, next_obs)
            r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            r = torch.clamp(r, -C.root_cfg.REWARD_CLIP, C.root_cfg.REWARD_CLIP)
        except Exception:
            return 0.0
        rc = r - r.mean()
        denom = rc.std() * tgt_std
        if float(denom) < 1e-8:
            return 0.0
        return float((rc * target).mean() / denom)

    return score


def make_knn_scorer(model, k=8):
    """Distance-weighted real logged fitness of nearest encoded trees."""
    samples = data_mod.load_samples()
    mu = encode_all(model, samples)                      # [N, latent]
    fit = np.array([s.fitness for s in samples], dtype=np.float64)

    def score_grid(zs):                                  # zs: [G, latent]
        d2 = ((zs[:, None, :] - mu[None, :, :]) ** 2).sum(-1)   # [G, N]
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        nd = np.take_along_axis(d2, idx, axis=1)
        w = 1.0 / (np.sqrt(nd) + 1e-6)
        nf = fit[idx]
        return (w * nf).sum(1) / w.sum(1)

    return score_grid


# ------------------------------------------------------------------
# Step 4 (actual fitness): train a SAC policy under each decoded reward and
# evaluate its real HalfCheetah return. Heavily vectorized to keep the GPU busy:
# `--chunk` learners are trained simultaneously in one vmap pass, each on its own
# `--batch` minibatch, sharing one replay buffer.
# ------------------------------------------------------------------
def _collect(env, buf, policy_fn, steps, dev, act_dim):
    """Roll out `policy_fn` for `steps` env-steps and push transitions to buf."""
    obs, _ = env.reset()
    eo, ea, en, ed = [], [], [], []
    for _ in range(steps):
        obs_t = torch.from_numpy(obs.astype("float32")).to(dev)
        a = policy_fn(obs_t)
        nobs, _, term, trunc, _ = env.step(a)
        done = np.logical_or(term, trunc)
        eo.append(obs.astype("float32")); ea.append(a.astype("float32"))
        en.append(nobs.astype("float32")); ed.append(done.astype("float32"))
        obs = nobs
        if done.any():
            obs, _ = env.reset()
    buf.add_batch(np.concatenate(eo), np.concatenate(ea),
                  np.concatenate(en), np.concatenate(ed))


@torch.no_grad()
def _eval_return(learner, env, dev):
    """Mean deterministic episodic return across the eval env's parallel episodes."""
    obs, _ = env.reset()
    returns = np.zeros(obs.shape[0], dtype=np.float64)
    while True:
        obs_t = torch.from_numpy(obs.astype("float32")).to(dev)
        a = learner.act(obs_t, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        returns += r
        if np.logical_or(term, trunc).any():
            break
    return float(returns.mean())


def score_true(model, zs, args):
    """Decode every grid point, train SAC under each reward, return real returns."""
    import config as root_cfg
    from learner import SRLearner, VectorizedSACUpdater
    from replay_buffer import ReplayBuffer
    from environment import make_envpool_env

    dev = root_cfg.DEVICE
    act_dim = root_cfg.ACTION_DIM

    print(f"decoding {len(zs)} grid points to trees ...")
    trees = [model.generate(torch.from_numpy(z)) for z in zs]

    # Shared replay buffer, pre-filled with random-policy transitions.
    buf = ReplayBuffer(max(args.prefill * 2, 200_000),
                       root_cfg.OBS_DIM, act_dim, device=dev)
    collect_env = make_envpool_env(seed=1234, num_envs=16)
    eval_env = make_envpool_env(seed=4321, num_envs=args.eval_episodes)
    rand_pol = lambda o: (np.random.uniform(-1, 1, (o.shape[0], act_dim))).astype("float32")
    print(f"pre-filling replay buffer (~{args.prefill} transitions) ...")
    while len(buf) < args.prefill:
        _collect(collect_env, buf, rand_pol, 512, dev, act_dim)

    heights = np.empty(len(trees), dtype=np.float64)
    n_chunks = (len(trees) + args.chunk - 1) // args.chunk
    for ci in range(n_chunks):
        sl = slice(ci * args.chunk, (ci + 1) * args.chunk)
        chunk = trees[sl]
        learners = [SRLearner(t, root_cfg) for t in chunk]
        updater = VectorizedSACUpdater(learners, root_cfg)
        n = len(learners)

        for step in range(args.steps):
            # Periodically add fresh on-policy-ish data (round-robin over learners).
            if step % args.collect_every == 0:
                updater.sync_to_learners()
                pol = learners[step // args.collect_every % n]
                _collect(collect_env, buf,
                         lambda o, _p=pol: _p.act(torch.as_tensor(o, device=dev)),
                         args.collect_steps, dev, act_dim)
            batch = buf.sample(n * args.batch)
            updater.update_all(batch)

        updater.sync_to_learners()
        for i, l in enumerate(learners):
            heights[sl.start + i] = _eval_return(l, eval_env, dev)
        done = min((ci + 1) * args.chunk, len(trees))
        print(f"  chunk {ci+1}/{n_chunks}: trained {n} learners x {args.steps} steps "
              f"({done}/{len(trees)} trees)  last-chunk return "
              f"min={heights[sl].min():.0f} max={heights[sl].max():.0f}")

    return heights.reshape(args.grid, args.grid)


# ------------------------------------------------------------------
# Step 6: dependency-free separable Gaussian smoothing
# ------------------------------------------------------------------
def gaussian_smooth(grid, sigma):
    if sigma <= 0:
        return grid
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()

    def conv_rows(a):
        pad = np.pad(a, ((0, 0), (radius, radius)), mode="reflect")
        return np.stack([np.convolve(row, k, mode="valid") for row in pad])

    return conv_rows(conv_rows(grid).T).T


# ------------------------------------------------------------------
# Plotting (steps 5 + 6)
# ------------------------------------------------------------------
def plot_surface(coords, height, label, out_prefix):
    lin = coords
    A, B = np.meshgrid(lin, lin, indexing="xy")

    # 2-D filled contour
    plt.figure(figsize=(8, 6.5))
    cf = plt.contourf(A, B, height, levels=40, cmap="viridis")
    plt.colorbar(cf, label=label)
    plt.contour(A, B, height, levels=12, colors="k", linewidths=0.3, alpha=0.4)
    plt.title("Fitness landscape over two random latent directions")
    plt.xlabel("random direction 1"); plt.ylabel("random direction 2")
    plt.tight_layout(); plt.savefig(out_prefix + "_2d.png", dpi=150); plt.close()
    print(f"saved {out_prefix}_2d.png")

    # 3-D surface
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A, B, height, cmap="viridis", linewidth=0,
                           antialiased=True)
    fig.colorbar(surf, shrink=0.6, label=label)
    ax.set_xlabel("random direction 1"); ax.set_ylabel("random direction 2")
    ax.set_zlabel(label)
    ax.set_title("Fitness landscape (random latent directions)")
    plt.tight_layout(); plt.savefig(out_prefix + "_3d.png", dpi=150); plt.close()
    print(f"saved {out_prefix}_3d.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--extent", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="random-direction seed")
    ap.add_argument("--score", choices=["proxy", "knn", "true"], default="proxy")
    ap.add_argument("--smooth", type=float, default=1.0, help="gaussian sigma (grid cells)")
    ap.add_argument("--center", choices=["origin", "data"], default="origin")
    # --score true (actual SAC fitness) knobs — bigger chunk/batch = busier GPU
    ap.add_argument("--chunk", type=int, default=128, help="learners trained per vmap pass")
    ap.add_argument("--batch", type=int, default=256, help="SAC minibatch per learner")
    ap.add_argument("--steps", type=int, default=1000, help="SAC grad steps per tree (fidelity knob)")
    ap.add_argument("--prefill", type=int, default=50_000, help="random transitions to seed the buffer")
    ap.add_argument("--collect-every", type=int, default=200, help="grad steps between data collection")
    ap.add_argument("--collect-steps", type=int, default=1000, help="env steps per collection round")
    ap.add_argument("--eval-episodes", type=int, default=10, help="parallel episodes per fitness eval")
    args = ap.parse_args()

    os.makedirs(C.FIG_DIR, exist_ok=True)
    model = load_model()
    latent = model.latent

    d1, d2 = random_directions(latent, args.seed)

    center = np.zeros(latent, dtype=np.float32)
    if args.center == "data":
        center = encode_all(model, data_mod.load_samples()).mean(0).astype(np.float32)

    # Step 2: grid of latent points
    lin = np.linspace(-args.extent, args.extent, args.grid).astype(np.float32)
    A, B = np.meshgrid(lin, lin, indexing="xy")
    zs = (center[None, :]
          + A.reshape(-1, 1) * d1[None, :]
          + B.reshape(-1, 1) * d2[None, :]).astype(np.float32)   # [G*G, latent]

    # Step 3 + 4: decode each point and score it
    if args.score == "true":
        height = score_true(model, zs, args)
        label = "fitness (actual SAC return)"
    elif args.score == "knn":
        knn = make_knn_scorer(model)
        height = knn(zs.astype(np.float64)).reshape(args.grid, args.grid)
        label = "fitness (kNN of logged returns)"
        # still decode for a validity readout
        n_unique = _decode_readout(model, zs)
        print(f"decoded grid: {n_unique} unique trees")
    else:
        scorer = make_proxy_scorer()
        heights = np.empty(len(zs), dtype=np.float64)
        uniq = set()
        for i, zv in enumerate(zs):
            tree = model.generate(torch.from_numpy(zv))
            uniq.add(repr(tree))
            heights[i] = scorer(tree)
            if (i + 1) % 500 == 0:
                print(f"  scored {i+1}/{len(zs)} points")
        height = heights.reshape(args.grid, args.grid)
        label = "fitness proxy (corr. with true objective)"
        print(f"decoded grid: {len(uniq)} unique trees")

    print(f"raw height: min={height.min():.3f} max={height.max():.3f} mean={height.mean():.3f}")

    # Step 6: smooth
    height_s = gaussian_smooth(height, args.smooth)

    out = os.path.join(C.FIG_DIR, f"surface_{args.score}_seed{args.seed}")
    plot_surface(lin, height_s, label, out)


@torch.no_grad()
def _decode_readout(model, zs):
    uniq = set()
    for zv in zs:
        uniq.add(repr(model.generate(torch.from_numpy(zv))))
    return len(uniq)


if __name__ == "__main__":
    main()
