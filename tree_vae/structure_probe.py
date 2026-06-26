"""
Is there ANY usable structure in the VAE latent space of reward trees?

`geometry_probe.py` answered a narrow question (do good trees lie on a helix /
torus / other named low-D manifold? -> no). This probe asks the broader one:
even without a clean geometric shape, does latent position carry information?

Three held-out tests, each against a null so the number means something:

  1. PREDICTABILITY  - regress fitness on z (linear Ridge + small MLP).
                       Cross-validated R^2. Null = shuffled fitness labels (R^2~0).
                       R^2 >> 0 => latent geometry is functionally meaningful.
  2. SEPARABILITY    - classify top-q vs bottom-q trees from z (logistic + boosting).
                       Cross-validated ROC-AUC. Null = shuffled labels (AUC~0.5).
                       AUC >> 0.5 => "good" is a localizable region of latent space.
  3. CLUSTERING      - GMM/BIC and KMeans silhouette over k, vs a matched-Gaussian
                       null. Data improving far more than the null => real clusters
                       (sub-families), not just one blob.

A "no structure" claim is only justified if ALL THREE come back near their nulls.

Usage:
  python structure_probe.py
  python structure_probe.py --tail 0.10 --cv 5 --kmax 10
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C                              # noqa: E402
from visualize import load_model, encode_all       # noqa: E402
import data as data_mod                             # noqa: E402


# ==================================================================
# 1. Predictability: latent -> fitness
# ==================================================================
def predictability(X, y, cv, seed):
    rng = np.random.default_rng(seed)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    out = {}

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred_lin = cross_val_predict(ridge, X, y, cv=kf)
    out["ridge_r2"] = r2_score(y, pred_lin)

    mlp = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                     early_stopping=True, random_state=0))
    pred_mlp = cross_val_predict(mlp, X, y, cv=kf)
    out["mlp_r2"] = r2_score(y, pred_mlp)

    # null: shuffled targets should give R^2 ~ 0 (often slightly negative)
    y_shuf = rng.permutation(y)
    out["null_r2"] = r2_score(y_shuf, cross_val_predict(ridge, X, y_shuf, cv=kf))

    out["pred_mlp"] = pred_mlp          # for the scatter plot
    return out


# ==================================================================
# 2. Separability: good vs bad
# ==================================================================
def separability(X, y_fit, tail, cv, seed):
    rng = np.random.default_rng(seed)
    lo = np.quantile(y_fit, tail)
    hi = np.quantile(y_fit, 1 - tail)
    good = y_fit >= hi
    bad = y_fit <= lo
    mask = good | bad
    Xs, ys = X[mask], good[mask].astype(int)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    out = {"n_good": int(good.sum()), "n_bad": int(bad.sum()), "lo": lo, "hi": hi}

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(logit, Xs, ys, cv=skf, method="predict_proba")[:, 1]
    out["logit_auc"] = roc_auc_score(ys, p)

    gb = HistGradientBoostingClassifier(random_state=0)
    p = cross_val_predict(gb, Xs, ys, cv=skf, method="predict_proba")[:, 1]
    out["gb_auc"] = roc_auc_score(ys, p)

    ys_shuf = rng.permutation(ys)
    p = cross_val_predict(logit, Xs, ys_shuf, cv=skf, method="predict_proba")[:, 1]
    out["null_auc"] = roc_auc_score(ys_shuf, p)
    return out


# ==================================================================
# 3. Clustering: GMM/BIC + KMeans silhouette, vs matched-Gaussian null
# ==================================================================
def _gmm_bic_curve(X, ks, rng):
    return np.array([GaussianMixture(k, covariance_type="diag", random_state=rng,
                                     reg_covar=1e-4).fit(X).bic(X) for k in ks])


def clustering(X, kmax, null_reps, rng):
    ks = list(range(1, kmax + 1))
    bic_data = _gmm_bic_curve(X, ks, 0)

    mean, cov = X.mean(0), np.cov(X, rowvar=False)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[0]))
    null_curves = []
    for r in range(null_reps):
        G = mean + rng.standard_normal(X.shape) @ L.T
        null_curves.append(_gmm_bic_curve(G, ks, 0))
    bic_null = np.mean(null_curves, axis=0)

    # KMeans silhouette (k>=2), data only
    sil = [silhouette_score(X, KMeans(k, n_init=5, random_state=0).fit_predict(X))
           for k in ks[1:]]

    # ΔBIC = improvement from k=1 to the best k (bigger = more cluster structure)
    out = {
        "ks": ks, "bic_data": bic_data, "bic_null": bic_null,
        "best_k_data": ks[int(np.argmin(bic_data))],
        "best_k_null": ks[int(np.argmin(bic_null))],
        "dbic_data": float(bic_data[0] - bic_data.min()),
        "dbic_null": float(bic_null[0] - bic_null.min()),
        "sil_ks": ks[1:], "sil": sil, "best_sil": float(np.max(sil)),
        "best_sil_k": ks[1:][int(np.argmax(sil))],
    }
    return out


# ==================================================================
# Plot
# ==================================================================
def plot(pred, sep, clu, y, out_path):
    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(y, pred["pred_mlp"], s=6, alpha=0.3)
    lim = [min(y.min(), pred["pred_mlp"].min()), max(y.max(), pred["pred_mlp"].max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("true fitness"); ax.set_ylabel("held-out predicted (MLP)")
    ax.set_title(f"Predictability: MLP R²={pred['mlp_r2']:.2f}, "
                 f"linear R²={pred['ridge_r2']:.2f} (null {pred['null_r2']:+.2f})")

    ax = fig.add_subplot(1, 3, 2)
    names = ["logistic", "boosting", "null"]
    vals = [sep["logit_auc"], sep["gb_auc"], sep["null_auc"]]
    ax.bar(names, vals, color=["C0", "C0", "C1"])
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("held-out ROC-AUC")
    ax.set_title(f"Separability good vs bad ({sep['n_good']}/{sep['n_bad']} trees)")

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(clu["ks"], clu["bic_data"], "o-", label="data")
    ax.plot(clu["ks"], clu["bic_null"], "s--", label="Gaussian null")
    ax.set_xlabel("GMM components k"); ax.set_ylabel("BIC (lower=better)")
    ax.set_title(f"Clustering: best k={clu['best_k_data']} "
                 f"(null {clu['best_k_null']}), max silhouette={clu['best_sil']:.2f}")
    ax.legend()

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"saved {out_path}")


# ==================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tail", type=float, default=0.10,
                    help="separability uses top-tail vs bottom-tail (0.10 = top/bottom 10%%)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=10, help="max GMM/KMeans components")
    ap.add_argument("--null-reps", type=int, default=5)
    ap.add_argument("--max-points", type=int, default=4000, help="subsample cap for speed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(C.FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    model = load_model()
    samples = data_mod.dedup(data_mod.load_samples())
    print(f"encoding {len(samples)} unique trees ...")
    X = encode_all(model, samples).astype(np.float64)
    y = np.array([s.fitness for s in samples], dtype=np.float64)
    if len(X) > args.max_points:
        sel = rng.choice(len(X), args.max_points, replace=False)
        X, y = X[sel], y[sel]
        print(f"subsampled to {len(X)} points")

    print("\n[1] PREDICTABILITY  latent -> fitness (cross-validated R^2)")
    pred = predictability(X, y, args.cv, args.seed)
    print(f"    linear (Ridge): R^2 = {pred['ridge_r2']:+.3f}")
    print(f"    MLP (64,64)   : R^2 = {pred['mlp_r2']:+.3f}")
    print(f"    shuffled null : R^2 = {pred['null_r2']:+.3f}  (should be ~0)")

    print(f"\n[2] SEPARABILITY  top-{args.tail:.0%} vs bottom-{args.tail:.0%} (cross-validated AUC)")
    sep = separability(X, y, args.tail, args.cv, args.seed)
    print(f"    {sep['n_good']} good (fit>= {sep['hi']:.0f}) vs {sep['n_bad']} bad (fit<= {sep['lo']:.0f})")
    print(f"    logistic (linear): AUC = {sep['logit_auc']:.3f}")
    print(f"    boosting (nonlin): AUC = {sep['gb_auc']:.3f}")
    print(f"    shuffled null    : AUC = {sep['null_auc']:.3f}  (should be ~0.5)")

    print(f"\n[3] CLUSTERING  GMM/BIC + KMeans silhouette vs matched-Gaussian null")
    clu = clustering(X, args.kmax, args.null_reps, np.random.default_rng(args.seed))
    print(f"    best k (data) = {clu['best_k_data']}   best k (Gaussian null) = {clu['best_k_null']}")
    print(f"    BIC improvement k=1->best:  data {clu['dbic_data']:.0f}   null {clu['dbic_null']:.0f}")
    print(f"    max KMeans silhouette = {clu['best_sil']:.3f} at k={clu['best_sil_k']}  "
          f"(>0.5 strong, <0.25 weak/none)")

    print("\n--- verdict ---")
    has_pred = pred["mlp_r2"] > 0.05
    has_sep = max(sep["logit_auc"], sep["gb_auc"]) > 0.6
    has_clu = (clu["best_k_data"] > 1 and clu["dbic_data"] > 3 * max(clu["dbic_null"], 1)
               and clu["best_sil"] > 0.25)
    print(f"    predictable: {has_pred}   separable: {has_sep}   clustered: {has_clu}")
    if has_pred or has_sep:
        print("    => Latent position DOES carry fitness information (even without a")
        print("       geometric manifold). 'No structure' would be too strong a claim.")
    else:
        print("    => Latent position carries little/no fitness information: combined with")
        print("       the geometry probe, 'no usable structure' is justified.")
    if not has_clu:
        print("    => No cluster structure beyond a single blob (k=1 or no better than null).")

    plot(pred, sep, clu, y, os.path.join(C.FIG_DIR, "structure_probe.png"))


if __name__ == "__main__":
    main()
