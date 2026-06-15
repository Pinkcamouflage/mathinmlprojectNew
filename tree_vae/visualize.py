"""
Fitness-landscape visualization in the learned latent space.

  1. Encode every logged tree (bestRun) to its latent mean.
  2. Project to 2-D (UMAP if available, else PCA) and scatter, colored by fitness.
  3. Goodfellow-style interpolation: pick the worst- and best-fitness trees,
     linearly interpolate their latent codes, decode at each alpha, and plot how
     the decoded reward's output (on a fixed random (s,a,s') batch) changes.

Figures -> tree_vae/figures/
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C            # noqa: E402
from graph import tree_to_graph  # noqa: E402
from model import TreeVAE         # noqa: E402
import data as data_mod          # noqa: E402


def load_model():
    ckpt = torch.load(os.path.join(C.CKPT_DIR, "tree_vae.pt"), map_location=C.DEVICE)
    model = TreeVAE(hidden=ckpt["hidden"], latent=ckpt["latent"]).to(C.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def encode_all(model, samples):
    mus = []
    for s in samples:
        g = tree_to_graph(s.tree, device=C.DEVICE)
        mu, _ = model.encode(g)
        mus.append(mu.cpu().numpy())
    return np.stack(mus)


def latent_covariance(z):
    """Covariance matrix of the latent means + PCA spectrum.

    Returns (cov [D,D], eigvals [D] descending, evr [D] explained-variance ratio).
    """
    zc = z - z.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / (len(z) - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]      # ascending -> descending
    eigvals = np.clip(eigvals, 0.0, None)        # kill tiny negative round-off
    evr = eigvals / eigvals.sum()
    return cov, eigvals, evr


def report_variance(eigvals, evr, k=2):
    """Print the covariance spectrum and how much the top-k components explain."""
    print(f"\nLatent covariance: {len(eigvals)}x{len(eigvals)} matrix")
    print(f"  total variance (trace) = {eigvals.sum():.4f}")
    top = "  ".join(f"PC{i+1}={evr[i]:.1%}" for i in range(min(8, len(evr))))
    print(f"  explained variance per component: {top}")
    print(f"  >>> top-{k} components explain {evr[:k].sum():.1%} "
          f"of total latent variance ({' + '.join(f'{evr[i]:.1%}' for i in range(k))})")
    # how many PCs to reach 90% / 95%
    cum = np.cumsum(evr)
    for thr in (0.90, 0.95):
        n = int(np.searchsorted(cum, thr) + 1)
        print(f"  components needed for {thr:.0%} variance: {n}")


def plot_covariance(cov, path):
    plt.figure(figsize=(6, 5))
    lim = np.abs(cov).max()
    im = plt.imshow(cov, cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(im, label="covariance")
    plt.title("Latent covariance matrix")
    plt.xlabel("latent dim"); plt.ylabel("latent dim")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"saved {path}")


def project_2d(z):
    """Return (coords[N,2], method, axis_evr) where axis_evr is the explained-variance
    fraction of each plotted axis, or None for nonlinear (UMAP) projections."""
    if z.shape[1] == 2:
        var = z.var(axis=0)
        return z, "latent", var / var.sum()
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=0).fit_transform(z), "umap", None
    except Exception:
        pass
    zc = z - z.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA
        p = PCA(n_components=2, random_state=0)
        coords = p.fit_transform(z)
        return coords, "pca", p.explained_variance_ratio_
    except Exception:
        # numpy SVD PCA fallback, with explained-variance ratio from singular values
        u, s, vt = np.linalg.svd(zc, full_matrices=False)
        coords = zc @ vt[:2].T
        evr = (s ** 2) / (s ** 2).sum()
        return coords, "pca", evr[:2]


def plot_landscape(z2, fitness, path, method="pca", axis_evr=None):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(z2[:, 0], z2[:, 1], c=fitness, cmap="viridis", s=14, alpha=0.8)
    plt.colorbar(sc, label="fitness (HalfCheetah return)")
    if axis_evr is not None:
        xl = f"PC1 ({axis_evr[0]:.1%} var)"
        yl = f"PC2 ({axis_evr[1]:.1%} var)"
        title = (f"Reward-tree fitness landscape — top 2 PCs explain "
                 f"{axis_evr[0] + axis_evr[1]:.1%} of latent variance")
    else:
        xl, yl, title = "UMAP 1", "UMAP 2", "Reward-tree fitness landscape (UMAP)"
    plt.title(title); plt.xlabel(xl); plt.ylabel(yl)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"saved {path}")


@torch.no_grad()
def interpolate(model, tree_a, tree_b, steps=11):
    """Decode along a linear path between two trees' latent means."""
    mu_a, _ = model.encode(tree_to_graph(tree_a, device=C.DEVICE))
    mu_b, _ = model.encode(tree_to_graph(tree_b, device=C.DEVICE))

    # fixed random transition batch to read off each decoded reward's behavior
    B = 512
    torch.manual_seed(0)
    obs = torch.randn(B, C.root_cfg.OBS_DIM)
    next_obs = torch.randn(B, C.root_cfg.OBS_DIM)
    action = torch.rand(B, C.root_cfg.ACTION_DIM) * 2 - 1

    alphas = np.linspace(0, 1, steps)
    means = []
    print("\nInterpolation path (worst -> best):")
    for a in alphas:
        z = (1 - a) * mu_a + a * mu_b
        tree = model.generate(z, sample=False)
        try:
            r = tree.evaluate(obs, action, next_obs)
            r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            mval = float(r.mean())
        except Exception:
            mval = float("nan")
        means.append(mval)
        print(f"  a={a:.2f}  mean_r={mval:+.3f}  {repr(tree)}")
    return alphas, means


def plot_interpolation(alphas, means, path):
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, means, "o-")
    plt.xlabel("interpolation alpha (worst -> best)")
    plt.ylabel("mean decoded reward (random batch)")
    plt.title("Goodfellow-style latent interpolation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"saved {path}")


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    model = load_model()

    samples = data_mod.load_samples()
    print(f"encoding {len(samples)} trees ...")
    z = encode_all(model, samples)
    fitness = np.array([s.fitness for s in samples])

    # Latent covariance + how much variance the plotted dimensions explain.
    cov, eigvals, evr = latent_covariance(z)
    report_variance(eigvals, evr, k=2)
    plot_covariance(cov, os.path.join(C.FIG_DIR, "latent_covariance.png"))

    z2, method, axis_evr = project_2d(z)
    if method == "umap":
        print("\n[note] UMAP is nonlinear, so its axes have no explained-variance; "
              f"for reference the linear top-2 PCs explain {evr[:2].sum():.1%}.")
    plot_landscape(z2, fitness, os.path.join(C.FIG_DIR, "landscape.png"),
                   method=method, axis_evr=axis_evr)

    worst = min(samples, key=lambda s: s.fitness)
    best = max(samples, key=lambda s: s.fitness)
    print(f"\nworst fitness={worst.fitness:.1f}: {worst.repr_str}")
    print(f"best  fitness={best.fitness:.1f}: {best.repr_str}")
    alphas, means = interpolate(model, worst.tree, best.tree)
    plot_interpolation(alphas, means, os.path.join(C.FIG_DIR, "interpolation.png"))


if __name__ == "__main__":
    main()
