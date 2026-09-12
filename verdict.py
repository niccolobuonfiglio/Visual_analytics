"""
Cross-method comparison: which of PCA / MDS is the more faithful embedding?

Combines four independent criteria into a weighted composite, then attaches a
paired significance statement (Cohen's d plus a bootstrap confidence interval
on the per-point distortion gap) so the margin can be read for reliability.

Split out of the former ``analysis`` module. The supervised graph metrics, the
k-means ground-truth check and the simulated-annealing parameter search that
used to live alongside this code are not reachable from the interface; they are
kept in ``legacy/label_structure.py`` and are not imported.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, pairwise_distances,
                             silhouette_score)

RANDOM_STATE = 42


# ==========================================================================
# 4. MECHANICAL PCA vs MDS VERDICT
# ==========================================================================
def _trustworthiness_continuity(D_high: NDArray[np.float64],
                                D_low: NDArray[np.float64], k: int
                                ) -> tuple[float, float]:
    """Classical aggregate trustworthiness / continuity (Venna & Kaski).

    Included because they are the field-standard scalars: the verdict should
    not rest solely on this toolkit's own metric.
    """
    n = D_high.shape[0]
    k = max(1, min(k, (n - 1) // 2))
    def ranks(D):
        Dc = D.copy(); np.fill_diagonal(Dc, np.inf)
        return np.argsort(np.argsort(Dc, axis=1), axis=1) + 1
    rh, rl = ranks(D_high), ranks(D_low)
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    t_pen = c_pen = 0.0
    for i in range(n):
        intr = np.nonzero((rl[i] <= k) & (rh[i] > k))[0]
        extr = np.nonzero((rh[i] <= k) & (rl[i] > k))[0]
        t_pen += float(np.sum(rh[i][intr] - k))
        c_pen += float(np.sum(rl[i][extr] - k))
    return 1.0 - norm * t_pen, 1.0 - norm * c_pen


def _find_best_k_for_clustering(X: NDArray[np.float64], k_range: tuple[int, int] = (2, 15)) -> int:
    """Find k that maximizes silhouette score for k-means clustering.
    
    This is used to determine the natural number of clusters in the data,
    rather than relying on the number of class labels. Returns the best k
    found, or the middle of the range if silhouette fails.
    """
    n = X.shape[0]
    lo, hi = k_range
    hi = min(hi, n - 1)
    best_k = min(3, hi)  # fallback
    best_score = -1
    
    for k in range(max(2, lo), hi + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            labels = km.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except Exception:
            continue
    
    return best_k


def compare_methods(X_high: NDArray[np.float64], projections: dict,
                    errs: dict, fps: dict, fns: dict,
                    labels: Sequence[int], k: int = 10,
                    kmeans: Optional[dict] = None) -> dict:
    """Decide, mechanically, whether PCA or MDS is better -- and by how much.

    Rationale for a COMPOSITE score. Any single number can be gamed by a
    method's particular failure mode: mean signed error hides a projection that
    is equally bad in both directions (FP and FN cancel), trustworthiness alone
    ignores lost neighbours, and cluster agreement alone ignores per-point
    fidelity. We therefore combine four independent, same-direction criteria,
    each normalised to [0, 1] where 1 is better:

      1. ``fidelity``   -- 1 - mean(|FP| + |FN|). Total per-point distortion,
                           immune to the FP/FN cancellation that makes the mean
                           signed error look good for a badly distorted point.
      2. ``neighbourhood`` -- average of trustworthiness and continuity (classical
                           rank-based neighbourhood preservation in both directions).
      3. ``shepard``    -- global distance preservation (Pearson correlation of
                           all pairwise distances).
      4. ``cluster``    -- ARI between clustering in the projection and clustering
                           in high-D, using the BEST k by silhouette score (not
                           the number of class labels). This makes the criterion
                           data-driven and label-independent.

    The weights are stated explicitly rather than tuned, so the verdict is
    reproducible and auditable. A paired comparison over the per-point
    distortion supplies significance: we report Cohen's d and a paired
    bootstrap confidence interval on the mean difference, so "better" is
    accompanied by "by how much, and is it reliable?".
    """
    labels = np.asarray(labels)
    names = list(projections)
    D_high = pairwise_distances(X_high)

    # ---- Find the best k for clustering (data-driven, not label-driven) ----
    # This ensures the cluster preservation criterion is independent of the
    # class labels and reflects the natural structure of the data.
    best_k_cluster = _find_best_k_for_clustering(X_high)
    
    # Also compute the label-based k for comparison (reported for transparency)
    n_classes = len(np.unique(labels))
    
    # ---- Cluster the high-D data with the best k ----
    km_high = KMeans(n_clusters=best_k_cluster, n_init=10, random_state=RANDOM_STATE)
    cl_high = km_high.fit_predict(X_high)
    
    # ---- Compute cluster preservation for each projection ----
    cluster_scores = {}
    for m, xy in projections.items():
        km_low = KMeans(n_clusters=best_k_cluster, n_init=10, random_state=RANDOM_STATE)
        cl_low = km_low.fit_predict(xy)
        ari = adjusted_rand_score(cl_high, cl_low)
        cluster_scores[m] = float(np.clip(ari, 0.0, 1.0))
    
    # Four criteria weights
    WEIGHTS = {"fidelity": 0.30, "neighbourhood": 0.25, "shepard": 0.20, "cluster": 0.25}

    # upper-triangular high-D pairwise distances (for the global Shepard corr.)
    iu = np.triu_indices(D_high.shape[0], k=1)
    dh_flat = D_high[iu]
    dh_std = dh_flat.std()

    per_method, per_point_dist = {}, {}
    for m in names:
        D_low = pairwise_distances(projections[m])
        tw, co = _trustworthiness_continuity(D_high, D_low, k)
        dist = np.abs(fps[m]) + np.abs(fns[m])
        per_point_dist[m] = dist
        # global Shepard correlation of all pairwise distances
        dl_flat = D_low[iu]
        if dh_std > 1e-12 and dl_flat.std() > 1e-12:
            shepard = float(np.corrcoef(dh_flat, dl_flat)[0, 1])
        else:
            shepard = 0.0
        comp = {
            "fidelity": float(np.clip(1.0 - dist.mean(), 0.0, 1.0)),
            "neighbourhood": float(np.clip(0.5 * (tw + co), 0.0, 1.0)),
            "shepard": float(np.clip(shepard, 0.0, 1.0)),
            "cluster": cluster_scores.get(m, 0.0),
        }
        w = {c: WEIGHTS[c] for c in comp}
        tot = sum(w.values())
        score = sum(comp[c] * w[c] for c in comp) / tot
        per_method[m] = {
            "components": comp,
            "weights_used": {c: round(w[c] / tot, 4) for c in comp},
            "score": float(score),
            "mean_signed_error": float(np.mean(errs[m])),
            "mean_abs_distortion": float(dist.mean()),
            "fn_share": float(np.mean(errs[m] < 0)),
        }

    ranked = sorted(names, key=lambda m: per_method[m]["score"], reverse=True)
    win, lose = ranked[0], ranked[-1]
    s_win, s_lose = per_method[win]["score"], per_method[lose]["score"]
    margin = s_win - s_lose
    rel = (margin / s_lose * 100.0) if s_lose > 0 else float("inf")

    # paired significance on the per-point distortion (lower is better)
    diff = per_point_dist[lose] - per_point_dist[win]     # >0 favours winner
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    cohen_d = float(diff.mean() / sd) if sd > 0 else 0.0
    rng = np.random.default_rng(RANDOM_STATE)
    boot = [float(rng.choice(diff, diff.size, replace=True).mean())
            for _ in range(2000)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    decisive = bool(ci[0] > 0 and margin >= 0.02)

    # a short, honest english statement
    if margin < 0.005:
        verdict = (f"PCA and MDS are effectively tied "
                   f"(composite scores {s_win:.3f} vs {s_lose:.3f}).")
    else:
        strength = ("decisively" if decisive
                    else "but the margin is not statistically reliable;")
        verdict = (f"{win} is better than {lose} {strength} composite "
                   f"{s_win:.3f} vs {s_lose:.3f} "
                   f"(+{margin:.3f}, {rel:.1f}% relative).")

    return {
        "k": int(k),
        "weights": WEIGHTS,
        "per_method": per_method,
        "ranking": ranked,
        "winner": win,
        "margin_absolute": float(margin),
        "margin_relative_pct": float(rel),
        "effect_size_cohen_d": cohen_d,
        "bootstrap_ci_mean_distortion_diff": [ci[0], ci[1]],
        "decisive": decisive,
        "verdict": verdict,
        # Additional info for transparency
        "cluster_k_used": best_k_cluster,
        "n_classes": n_classes,
    }
