"""
Adjusted-distance MDS (retired).

Three ways of reweighting the feature space before MDS builds its layout:
mutual-information feature weights, Laplacian-Score locality weights, and
Mahalanobis whitening. The interface panel that displayed the plain-versus-
adjusted comparison has been removed, so none of this runs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import mutual_info_classif

RANDOM_STATE = 42

# =====================================================================
# QUARANTINED -- NOT IMPORTED BY THE RUNNING PIPELINE
# ---------------------------------------------------------------------
# Nothing in this file is reachable from the browser interface. It is kept
# for reference and for the report's history section only. Do not add an
# import of this module to pipeline.py without also adding the interface
# control that makes its output visible; the cost of each item is noted
# below.
# =====================================================================


def compute_feature_weights(X: NDArray[np.float64],
                            labels: Optional[Sequence[int]] = None,
                            method: str = "mutual_info",
                            custom: Optional[Sequence[float]] = None
                            ) -> NDArray[np.float64]:
    """Return a non-negative per-feature weight vector for weighted MDS.

    The weights scale each feature (as ``sqrt(w) * column``) before Euclidean
    distances are computed, so a feature with weight ``w`` contributes ``w`` times
    as much to the squared distance. Weights are normalised to average 1 so the
    overall distance scale is comparable to the unweighted case.

    Weighting is by **mutual information** with the class label: features that are
    more informative about the class are up-weighted, so the MDS layout is driven
    by the measurements that carry the class signal. (This is the only supported
    weighting scheme: correlation-based down-weighting was weaker and
    inverse-variance is a no-op under standardization, so both were removed.)

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix the weights are computed from (already standardized when
        standardization is on).
    labels : sequence of int
        Class labels; required.
    method : str
        Retained for API compatibility; only ``'mutual_info'`` is supported.
    custom : ignored
        Retained for API compatibility.

    Returns
    -------
    ndarray, shape (n_features,)
        Non-negative weights, normalised to mean 1.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]

    if labels is None:
        raise ValueError("Weighted MDS requires class labels for mutual-information "
                         "weighting.")
    w = mutual_info_classif(X, np.asarray(labels), random_state=RANDOM_STATE)

    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    if total <= 0:
        return np.ones(n_features)
    return w * (n_features / total)


def locality_feature_weights(X: NDArray[np.float64], k: int = 10
                             ) -> NDArray[np.float64]:
    """Label-free per-feature weights from LOCAL structure (Laplacian Score).

    Idea: a feature is useful for a projection if it varies *little between
    nearby points* (it respects the local manifold) relative to its overall
    variance. This is the Laplacian Score (He, Cai & Niyogi 2005): build a
    k-NN affinity graph, then for each feature f score

        L_f = (fᵀ L f) / (fᵀ D f)

    where L = D - W is the graph Laplacian of the affinity W and D its degree.
    A *low* L_f means the feature is smooth on the neighbourhood graph (good);
    we therefore weight features by ``1 - L_f`` (higher = more locality-
    preserving), normalised to mean 1. Unlike Mahalanobis (which whitens and can
    amplify noise directions), this *down-weights* noisy features that vary
    randomly between neighbours, which is what a projection wants.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n < 3 or p == 0:
        return np.ones(p)
    D_ = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D_, np.inf)
    kk = int(min(max(2, k), n - 1))
    order = np.argsort(D_, axis=1)[:, :kk]
    # heat-kernel affinity on the k-NN graph (symmetric)
    sigma = np.median(D_[np.isfinite(D_)]) or 1.0
    W = np.zeros((n, n))
    for i in range(n):
        for j in order[i]:
            wij = np.exp(-(D_[i, j] ** 2) / (2.0 * sigma ** 2))
            W[i, j] = max(W[i, j], wij)
            W[j, i] = W[i, j]
    deg = W.sum(axis=1)
    Dm = np.diag(deg)
    L = Dm - W
    scores = np.zeros(p)
    for f in range(p):
        fr = X[:, f].copy()
        # remove the weighted mean (projection onto the constant vector)
        denom_all = deg.sum() or 1.0
        fr = fr - (fr @ deg) / denom_all
        fDf = float(fr @ (deg * fr))
        if fDf <= 1e-12:
            scores[f] = 1.0            # constant feature -> max Laplacian score
            continue
        fLf = float(fr @ (L @ fr))
        scores[f] = fLf / fDf         # Laplacian score in [0, ~1]; low = good
    w = 1.0 - np.clip(scores, 0.0, 1.0)   # high = locality-preserving
    w = np.clip(w, 1e-6, None)
    total = w.sum()
    if total <= 0:
        return np.ones(p)
    return w * (p / total)            # normalise to mean 1


def build_projections_adjusted(X, standardize=True, weighted_mds=False,
                               feature_weights=None, mahalanobis=False):
    """The former adjusted-distance branches of ``build_projections``.

    COST WHEN ENABLED: one additional full MDS fit on a precomputed
    dissimilarity matrix, plus (for the comparison panel that consumed it) two
    extra ``per_point_neighbor_metrics`` passes, two silhouette scores and two
    full ``pdist`` calls. On a 569-point dataset this was the single most
    expensive block in the pipeline.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    RANDOM_STATE = 42

    Xp = StandardScaler().fit_transform(X) if standardize else np.asarray(X, float)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xp)

    if mahalanobis:
        # MAHALANOBIS distance = whiten by the inverse covariance, so correlated
        # and high-variance directions are down-weighted (a label-free
        # "adjusted distance", i.e. unsupervised metric learning).
        Xc = Xp - Xp.mean(axis=0)
        cov = np.atleast_2d(np.cov(Xc, rowvar=False))
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 1e-8, None)
            Xwhite = Xc @ (vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T)
        except np.linalg.LinAlgError:
            Xwhite = Xc
        D = pairwise_distances(Xwhite, metric="euclidean")
    elif weighted_mds and feature_weights is not None:
        w = np.clip(np.asarray(feature_weights, dtype=float), 0.0, None)
        D = pairwise_distances(Xp * np.sqrt(w), metric="euclidean")
    else:
        return {"PCA": pca, "MDS": MDS(n_components=2, dissimilarity="euclidean",
                                       n_init=4, random_state=RANDOM_STATE
                                       ).fit_transform(Xp)}, Xp

    mds = MDS(n_components=2, dissimilarity="precomputed", n_init=4,
              random_state=RANDOM_STATE).fit_transform(D)
    return {"PCA": pca, "MDS": mds}, Xp
