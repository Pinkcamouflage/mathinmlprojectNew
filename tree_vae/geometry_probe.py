"""
Does the set of GOOD reward trees lie on a low-dimensional geometry (helix /
torus) in the VAE latent space?

A bare "fit a shape, read off the residual" is not enough: a flexible surface
explains some variance of *any* blob. This probe judges each candidate shape
three ways, so the conclusion is honest:

  1. NULL BASELINE   - fit the same shape to a Gaussian with the data's covariance.
                       A shape that fits the data no better than matched noise is
                       capturing flexibility, not geometry. We report the ratio
                       FUE_data / FUE_null  (FUE = fraction of variance unexplained).
                       ~1 => no real structure;  <<1 => genuine manifold.
  2. HELD-OUT FIT    - helix/torus carry one angle per point, so they can overfit.
                       We K-fold cross-validate: fit the global shape on train,
                       solve only the angle for held-out points, measure distance.
  3. A SHAPE LADDER  - point < line < plane < circle < sphere < helix < torus.
                       A torus must beat a plane AND a sphere to mean anything.

Plus an intrinsic-dimension estimate (TwoNN) as a gate: a helix is ~1D, a torus
~2D. If the cloud is intrinsically ~10D, no 1-2D shape is the answer.

The decisive helix-vs-torus discriminator is TOPOLOGY (loops), not geometry:
circle/helix-loop -> b1=1; torus -> b1=2, b2=1; sphere -> b1=0,b2=1; blob -> none.
If `ripser` is importable we compute a persistence diagram and report Betti
numbers. It is optional (pip install ripser persim) and skipped if absent.

Usage:
  python geometry_probe.py --quantile 0.90        # probe the top-10% trees
  python geometry_probe.py --quantile 0.95 --dims 3 --cv 5 --null 8
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C                              # noqa: E402
from visualize import load_model, encode_all       # noqa: E402
import data as data_mod                             # noqa: E402


# ==================================================================
# Linear-algebra helpers
# ==================================================================
def pca_fit(X, n):
    """Fit PCA on X [m, D]; return (mean[D], components[n, D], explained_var_ratio[n])."""
    mean = X.mean(0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    return mean, Vt[:n], evr[:n]


def rodrigues(omega):
    """3-vector -> 3x3 rotation matrix (exponential map)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def total_ss(X, mean):
    """Sum over points of squared distance to `mean` (the variance to be explained)."""
    return float(((X - mean) ** 2).sum())


# ==================================================================
# Intrinsic dimension (TwoNN, Facco et al. 2017)
# ==================================================================
def intrinsic_dim_twonn(X):
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    d, _ = tree.query(X, k=3)            # self + two neighbors
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 1e-12
    mu = (r2[ok] / r1[ok])
    mu = mu[mu > 1.0]
    # d estimated from slope of -log(1-F(mu)) vs log(mu) through the origin
    n = len(mu)
    F = (np.arange(1, n + 1)) / (n + 1)
    x = np.log(np.sort(mu))
    y = -np.log(1 - F)
    return float((x @ y) / (x @ x))      # least-squares slope through origin


# ==================================================================
# Shape models. Each returns per-point SQUARED residual to the manifold.
# Shapes work in a `dims`-D PCA subspace fit on TRAIN; the variance a test point
# carries outside that subspace is added as residual (honest for held-out points).
# Every fitter exposes:  fit(Xtr) -> predictor;  predictor(Xte) -> sq_resid[te]
# ==================================================================
class ShapeFit:
    """Base: subtracts a `dims`-D PCA subspace; subclasses model within it."""
    n_params = 0          # free geometric params (for reference / AIC-style notes)

    def __init__(self, dims):
        self.dims = dims

    def _reduce(self, X):
        """Project to the (train) PCA subspace; return (coords[m, dims], tail_sq[m])."""
        Xc = X - self.mean
        coords = Xc @ self.comp.T                       # [m, dims]
        tail = (Xc ** 2).sum(1) - (coords ** 2).sum(1)  # variance outside subspace
        return coords, np.clip(tail, 0.0, None)

    def fit(self, Xtr):
        self.mean, self.comp, _ = pca_fit(Xtr, self.dims)
        self._fit_inner(Xtr)
        return self

    def _fit_inner(self, Xtr):
        pass

    def sq_resid(self, Xte):
        coords, tail = self._reduce(Xte)
        return self._within_sq(coords) + tail

    def _within_sq(self, coords):
        raise NotImplementedError


class Point(ShapeFit):
    """Just the centroid: residual = full distance to mean (the variance floor)."""
    def __init__(self):
        super().__init__(dims=0)
    def _reduce(self, X):
        Xc = X - self.mean
        return None, (Xc ** 2).sum(1)
    def _within_sq(self, coords):
        return 0.0


class Subspace(ShapeFit):
    """Best-fit affine subspace of dimension `dims` (line=1, plane=2)."""
    def _within_sq(self, coords):
        return np.zeros(len(coords))     # everything in-subspace is explained


class Circle(ShapeFit):
    """Circle in the best 2-plane (Kasa algebraic fit)."""
    n_params = 3
    def __init__(self):
        super().__init__(dims=2)
    def _fit_inner(self, Xtr):
        P, _ = self._reduce(Xtr)
        x, y = P[:, 0], P[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        b = x ** 2 + y ** 2
        cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
        self.center = np.array([cx, cy])
        self.radius = np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))
    def _within_sq(self, coords):
        d = np.linalg.norm(coords - self.center, axis=1)
        return (d - self.radius) ** 2


class Sphere(ShapeFit):
    """Sphere (2-sphere) in the best 3-space (algebraic fit)."""
    n_params = 4
    def __init__(self):
        super().__init__(dims=3)
    def _fit_inner(self, Xtr):
        P, _ = self._reduce(Xtr)
        A = np.c_[2 * P, np.ones(len(P))]
        b = (P ** 2).sum(1)
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        self.center = sol[:3]
        self.radius = np.sqrt(max(sol[3] + (self.center ** 2).sum(), 0.0))
    def _within_sq(self, coords):
        d = np.linalg.norm(coords - self.center, axis=1)
        return (d - self.radius) ** 2


def _helix_solve_t(a, b, h, r, p):
    """Per-point arclength param t minimizing |x - helix(t)|^2 (vectorized).

    The circular part wants t == atan2(b,a) (mod 2*pi); the axis part wants
    t == h/p. We pick the winding 2*pi*k that reconciles them by testing a few k
    around h/p, then do a clamped Newton refine. This is stable even as the fitted
    pitch p -> 0 (where a naive t=h/p diverges)."""
    phi = np.arctan2(b, a)
    if abs(p) < 1e-6:
        return phi                                   # degenerates to a circle
    def f(t):
        return (a - r * np.cos(t)) ** 2 + (b - r * np.sin(t)) ** 2 + (h - p * t) ** 2
    k0 = np.round((h / p - phi) / (2 * np.pi))
    best_t = phi + 2 * np.pi * k0
    best_f = f(best_t)
    for dk in (-2, -1, 1, 2):
        t = phi + 2 * np.pi * (k0 + dk)
        ft = f(t)
        m = ft < best_f
        best_t = np.where(m, t, best_t)
        best_f = np.where(m, ft, best_f)
    t = best_t
    for _ in range(3):                               # clamped Newton (no winding jumps)
        g = r * (a * np.sin(t) - b * np.cos(t)) + p * p * t - p * h
        gp = r * (a * np.cos(t) + b * np.sin(t)) + p * p
        step = g / np.where(np.abs(gp) < 1e-9, 1e-9, gp)
        t = t - np.clip(step, -np.pi, np.pi)
    return t


class Helix(ShapeFit):
    """c + r cos(t) u + r sin(t) v + p t w, in the best 3-space (orientation free)."""
    n_params = 8
    def __init__(self, n_restarts=3, seed=0):
        super().__init__(dims=3)
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
    def _frame(self, omega):
        return rodrigues(omega)                       # columns u,v,w
    def _resid_vec(self, params, P):
        omega, c = params[:3], params[3:6]
        r, ppitch = np.exp(params[6]), params[7]
        R = self._frame(omega)
        q = (P - c) @ R                               # coords in helix frame
        a, b, h = q[:, 0], q[:, 1], q[:, 2]
        t = _helix_solve_t(a, b, h, r, ppitch)
        return np.sqrt((a - r * np.cos(t)) ** 2 + (b - r * np.sin(t)) ** 2 + (h - ppitch * t) ** 2)
    def _fit_inner(self, Xtr):
        P, _ = self._reduce(Xtr)
        rad = np.linalg.norm(P[:, :2], axis=1).mean()
        base = np.array([0, 0, 0, 0, 0, 0, np.log(max(rad, 1e-3)),
                         np.std(P[:, 2]) / np.pi + 1e-3])
        best, best_cost = None, np.inf
        for i in range(self.n_restarts):
            x0 = base.copy()
            if i:
                x0[:3] = self.rng.normal(0, 1.0, 3)
            sol = least_squares(self._resid_vec, x0, args=(P,), method="lm", max_nfev=4000)
            if sol.cost < best_cost:
                best, best_cost = sol.x, sol.cost
        self.params = best
    def _within_sq(self, coords):
        return self._resid_vec(self.params, coords) ** 2


class Torus(ShapeFit):
    """(R + r cos phi)(cos th u + sin th v) + r sin phi w, best 3-space (closed-form resid)."""
    n_params = 8
    def __init__(self, n_restarts=3, seed=0):
        super().__init__(dims=3)
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
    def _resid_vec(self, params, P):
        omega, c = params[:3], params[3:6]
        Rmaj, rmin = np.exp(params[6]), np.exp(params[7])
        R = rodrigues(omega)
        q = (P - c) @ R
        a, b, h = q[:, 0], q[:, 1], q[:, 2]
        rho = np.sqrt(a ** 2 + b ** 2)                # distance from tube axis
        return np.sqrt((rho - Rmaj) ** 2 + h ** 2) - rmin
    def _fit_inner(self, Xtr):
        P, _ = self._reduce(Xtr)
        rho = np.linalg.norm(P[:, :2], axis=1)
        base = np.array([0, 0, 0, 0, 0, 0, np.log(max(rho.mean(), 1e-3)),
                         np.log(max(rho.std() + np.abs(P[:, 2]).std(), 1e-3))])
        best, best_cost = None, np.inf
        for i in range(self.n_restarts):
            x0 = base.copy()
            if i:
                x0[:3] = self.rng.normal(0, 1.0, 3)
            sol = least_squares(self._resid_vec, x0, args=(P,), method="lm", max_nfev=4000)
            if sol.cost < best_cost:
                best, best_cost = sol.x, sol.cost
        self.params = best
    def _within_sq(self, coords):
        return self._resid_vec(self.params, coords) ** 2


SHAPES = {
    "point":  lambda: Point(),
    "line":   lambda: Subspace(dims=1),
    "plane":  lambda: Subspace(dims=2),
    "circle": lambda: Circle(),
    "sphere": lambda: Sphere(),
    "helix":  lambda: Helix(),
    "torus":  lambda: Torus(),
}


# ==================================================================
# Evaluation: in-sample FUE, K-fold held-out FUE, Gaussian-null FUE
# ==================================================================
def fue_full(make, X):
    """Fraction of variance unexplained, fit and evaluated on all of X."""
    m = make().fit(X)
    return m.sq_resid(X).sum() / total_ss(X, X.mean(0))


def fue_cv(make, X, folds, rng):
    """Cross-validated FUE: fit on train, score held-out (angles solved per point)."""
    idx = rng.permutation(len(X))
    parts = np.array_split(idx, folds)
    num = den = 0.0
    gmean = X.mean(0)
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[j] for j in range(folds) if j != f])
        m = make().fit(X[tr])
        num += m.sq_resid(X[te]).sum()
        den += ((X[te] - gmean) ** 2).sum()
    return num / den


def fue_null(make, X, reps, rng):
    """Mean +/- std FUE of the shape fit to Gaussians matched to X's covariance."""
    mean = X.mean(0)
    cov = np.cov(X, rowvar=False)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[0]))
    vals = []
    for _ in range(reps):
        G = mean + rng.standard_normal((len(X), X.shape[1])) @ L.T
        vals.append(fue_full(make, G))
    return float(np.mean(vals)), float(np.std(vals))


# ==================================================================
# Optional topology (persistent homology) - decisive helix vs torus
# ==================================================================
def betti_report(X, dims=3, maxdim=2):
    try:
        from ripser import ripser
    except Exception:
        return None
    mean, comp, _ = pca_fit(X, dims)
    P = (X - mean) @ comp.T
    P = P / (P.std(0) + 1e-9)                          # scale-normalize
    dgms = ripser(P, maxdim=maxdim)["dgms"]
    out = {}
    for k, dg in enumerate(dgms):
        life = dg[:, 1] - dg[:, 0]
        life = life[np.isfinite(life)]
        if len(life) == 0:
            out[k] = (0, 0.0)
            continue
        thr = 0.5 * life.max()                         # "persistent" = >half the max bar
        out[k] = (int((life > thr).sum()), float(life.max()))
    return out


# ==================================================================
# Plotting
# ==================================================================
def plot_report(X, fit_results, helix, torus, fitness, out_path):
    fig = plt.figure(figsize=(16, 5))

    # (1) FUE bar chart: data (CV) vs null
    ax = fig.add_subplot(1, 3, 1)
    names = list(fit_results.keys())
    data = [fit_results[n]["cv"] for n in names]
    null = [fit_results[n]["null"][0] for n in names]
    nerr = [fit_results[n]["null"][1] for n in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, data, 0.4, label="data (held-out)")
    ax.bar(xs + 0.2, null, 0.4, yerr=nerr, capsize=3, label="Gaussian null", alpha=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("fraction of variance unexplained")
    ax.set_title("Lower = better fit. Data well below null = real geometry.")
    ax.legend()

    # (2) torus natural coords (theta, phi) colored by fitness
    ax = fig.add_subplot(1, 3, 2)
    P = (X - torus.mean) @ torus.comp.T
    R = rodrigues(torus.params[:3])
    q = (P - torus.params[3:6]) @ R
    th = np.arctan2(q[:, 1], q[:, 0])
    rho = np.linalg.norm(q[:, :2], axis=1)
    phi = np.arctan2(q[:, 2], rho - np.exp(torus.params[6]))
    sc = ax.scatter(th, phi, c=fitness, cmap="viridis", s=10, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="fitness")
    ax.set_xlabel("theta (around tube axis)"); ax.set_ylabel("phi (around tube)")
    ax.set_title("Torus angular coords (uniform fill ~ torus)")

    # (3) helix unrolled: arclength t vs radial residual
    ax = fig.add_subplot(1, 3, 3)
    P = (X - helix.mean) @ helix.comp.T
    R = rodrigues(helix.params[:3])
    q = (P - helix.params[3:6]) @ R
    r, p = np.exp(helix.params[6]), helix.params[7]
    t = _helix_solve_t(q[:, 0], q[:, 1], q[:, 2], r, p)
    sc = ax.scatter(t, q[:, 2], c=fitness, cmap="viridis", s=10, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="fitness")
    ax.set_xlabel("helix param t"); ax.set_ylabel("axis coordinate w")
    ax.set_title("Helix: linear axis-vs-t => real helix")

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"saved {out_path}")


# ==================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="keep trees with fitness >= this quantile (0.90 = top 10%%)")
    ap.add_argument("--max-points", type=int, default=2000, help="subsample cap for speed")
    ap.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    ap.add_argument("--null", type=int, default=8, help="Gaussian-null repetitions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-topology", action="store_true", help="skip persistent homology")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(C.FIG_DIR, exist_ok=True)

    # ---- encode the good trees ----
    model = load_model()
    samples = data_mod.dedup(data_mod.load_samples())
    fit_all = np.array([s.fitness for s in samples])
    thr = np.quantile(fit_all, args.quantile)
    good = [s for s in samples if s.fitness >= thr]
    print(f"{len(samples)} unique trees; fitness>= {thr:.1f} keeps {len(good)} good trees")

    X = encode_all(model, good).astype(np.float64)
    fitness = np.array([s.fitness for s in good])
    if len(X) > args.max_points:
        sel = rng.choice(len(X), args.max_points, replace=False)
        X, fitness = X[sel], fitness[sel]
        print(f"subsampled to {len(X)} points")

    # ---- how concentrated is the cloud? (manifold must live in few dims) ----
    _, _, evr = pca_fit(X, min(8, X.shape[1]))
    print("\nPCA explained variance (top 8): " + "  ".join(f"{v:.1%}" for v in evr))
    print(f"top-3 PCs hold {evr[:3].sum():.1%} of variance "
          f"(helix/torus live in the top-3 subspace here)")
    idim = intrinsic_dim_twonn(X)
    print(f"intrinsic dimension (TwoNN): {idim:.2f}  "
          f"(helix~1, torus~2; >>2 means no low-D shape fits)")

    # ---- fit the shape ladder ----
    print("\nfitting shape ladder (FUE = fraction of variance unexplained, lower=better)")
    print(f"{'shape':8}  {'in-sample':>10}  {'held-out':>10}  {'null(mean±sd)':>18}  {'data/null':>9}")
    results = {}
    fitted = {}
    for name, make in SHAPES.items():
        full = fue_full(make, X)
        cv = fue_cv(make, X, args.cv, np.random.default_rng(args.seed))
        nmean, nsd = fue_null(make, X, args.null, np.random.default_rng(args.seed + 1))
        ratio = cv / nmean if nmean > 1e-9 else float("nan")
        results[name] = {"full": full, "cv": cv, "null": (nmean, nsd), "ratio": ratio}
        fitted[name] = make().fit(X)
        print(f"{name:8}  {full:10.3f}  {cv:10.3f}  {nmean:8.3f} ± {nsd:5.3f}   {ratio:8.2f}")

    # ---- optional topology ----
    print()
    if not args.no_topology:
        betti = betti_report(X)
        if betti is None:
            print("topology: ripser not installed -> skipped "
                  "(pip install ripser persim to enable the decisive loop count)")
        else:
            desc = {0: "components b0", 1: "loops b1", 2: "voids b2"}
            print("persistent Betti numbers (scale-normalized top-3 PCA):")
            for k in sorted(betti):
                n, ml = betti[k]
                print(f"  {desc.get(k, f'H{k}'):14}: {n}   (max persistence {ml:.2f})")
            print("  reference: circle/helix-loop b1=1; torus b1=2,b2=1; sphere b1=0,b2=1; blob none")

    # ---- verdict ----
    print("\n--- verdict ---")
    best = min(results, key=lambda n: results[n]["ratio"])
    br = results[best]["ratio"]
    print(f"best data/null ratio: '{best}' at {br:.2f}")
    if br > 0.85:
        print("=> No shape beats matched Gaussian noise meaningfully: the good-tree")
        print("   cloud has no special helix/torus geometry (it is blob-like).")
    elif best in ("helix", "torus") and idim < 2.6:
        print(f"=> '{best}' fits substantially better than chance and intrinsic dim is low:")
        print(f"   evidence FOR a {best} geometry. Confirm with the topology/Betti numbers.")
    else:
        print(f"=> Structure exists but the strongest fit is '{best}', not helix/torus.")
        print("   The good trees are lower-D than random, but not specifically a helix/torus.")

    plot_report(X, results, fitted["helix"], fitted["torus"], fitness,
                os.path.join(C.FIG_DIR, f"geometry_probe_q{int(args.quantile*100)}.png"))


if __name__ == "__main__":
    main()
