"""
Shared core for the PCA/MDS continuous error-scale project (dataset-agnostic).

No plotting and no matplotlib here on purpose, so both the static analysis script
and the interactive brushing tool can import it without backend conflicts.

Two metric families are provided:

Classic class-compactness scale (2D only)::
    error = -(alpha*d_same - beta*d_diff) / (alpha*d_same + beta*d_diff)   in [-1, +1]
        < 0  -> drawn toward OTHER classes    = FN (-1)
        > 0  -> tightly embedded in own class = FP (+1)
        = 0  -> equidistant (True)
    (alpha = beta = 1)

Intrusion / extrusion scale (compares high-D to 2D), the default::
    error = FP - FN   in [-1, +1]
    where FP (intrusion) and FN (extrusion) are rank-severity-weighted and can
    be returned separately for diagnostics. Distance weighting (as opposed to
    rank weighting) is available via ``weight='distance'``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import datasets as skds
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

from utils import describe  

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

Projections = dict[str, NDArray[np.float64]]
ErrorMap = dict[str, NDArray[np.float64]]

# Built-in sklearn loaders exposing .data / .target / .target_names
SKLEARN_LOADERS = {
    "wine": skds.load_wine,
    "iris": skds.load_iris,
    "breast_cancer": skds.load_breast_cancer,
    "digits": skds.load_digits,
}

# Synthetic datasets, each built to expose a DIFFERENT failure mode the real
# datasets above cannot show. Fixed seed so runs are reproducible. Each factory
# returns (X, y, target_names) like the sklearn loaders' relevant fields.
def _make_noisy():
    """Many irrelevant features: only 5 of 50 carry signal. Tests whether the
    metric can see structure buried in noise (and motivates weighted MDS)."""
    X, y = skds.make_classification(
        n_samples=300, n_features=50, n_informative=5, n_redundant=5,
        n_repeated=0, n_classes=3, n_clusters_per_class=1, class_sep=1.0,
        flip_y=0.01, random_state=RANDOM_STATE)
    return X, y, ["class 0", "class 1", "class 2"]

def _make_nonconvex():
    """Two concentric rings: perfectly separable but NOT by centroids, so
    k-means fails on data whose labels are a perfect (non-convex) clustering."""
    X, y = skds.make_circles(n_samples=300, noise=0.05, factor=0.5,
                             random_state=RANDOM_STATE)
    return X, y, ["outer ring", "inner ring"]

def _make_mislabeled():
    """Real geometric structure (6 blobs) but labels assigned at RANDOM to 3
    classes, so the labels carry no structure at all -- the extreme
    labels-are-not-clusters case."""
    X, _ = skds.make_blobs(n_samples=300, centers=6, n_features=10,
                           cluster_std=1.5, random_state=RANDOM_STATE)
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, size=X.shape[0])
    return X, y, ["label A", "label B", "label C"]

def _make_hierarchical():
    """Nested/hierarchical structure: 3 super-groups, each split into 2 sub-
    clusters, in a moderately high-D space with correlated features. Realistic
    for taxonomies (e.g. species within genera): the challenge is preserving
    BOTH scales at once -- a projection may keep the super-groups apart yet merge
    the sub-clusters, or vice-versa. 6 labels, 400 points, 20 features."""
    rng = np.random.RandomState(RANDOM_STATE)
    n_per = 400 // 6
    p = 20
    Xs, ys = [], []
    # three super-group centres, well separated in the first few dims
    super_centres = rng.randn(3, p) * 6.0
    label = 0
    for s in range(3):
        # two sub-clusters offset from the super centre by a smaller amount
        for _ in range(2):
            offset = rng.randn(p) * 1.6
            centre = super_centres[s] + offset
            # anisotropic, mildly correlated covariance
            A = rng.randn(p, p) * 0.35
            cov = A @ A.T + np.eye(p) * 0.4
            pts = rng.multivariate_normal(centre, cov, size=n_per)
            Xs.append(pts)
            ys.append(np.full(n_per, label))
            label += 1
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    names = [f"{g}{sub}" for g in ("A", "B", "C") for sub in ("1", "2")]
    return X, y, names

def _make_imbalanced_overlap():
    """Class-imbalanced, anisotropic, partially overlapping classes -- the kind
    of messiness real data has. Four classes of very different sizes (a big
    majority class, two mid, one rare), each an elongated (stretched) Gaussian,
    with two of them overlapping. Tests whether the metrics handle imbalance and
    whether a projection can separate unequal, overlapping clusters. 30 features,
    ~430 points."""
    rng = np.random.RandomState(RANDOM_STATE)
    p = 30
    sizes = [220, 110, 70, 30]          # strongly imbalanced
    centres = np.array([
        rng.randn(p) * 4.0,
        rng.randn(p) * 4.0,
        rng.randn(p) * 4.0,
        rng.randn(p) * 4.0,
    ])
    # make classes 0 and 1 overlap by pulling their centres together
    centres[1] = centres[0] + rng.randn(p) * 1.2
    Xs, ys = [], []
    for c, n_c in enumerate(sizes):
        # elongated covariance: a few directions stretched, others tight
        scales = np.ones(p)
        stretch = rng.choice(p, size=5, replace=False)
        scales[stretch] = rng.uniform(3.0, 6.0, size=5)
        A = (rng.randn(p, p) * 0.25) * scales[None, :]
        cov = A @ A.T + np.eye(p) * 0.5
        pts = rng.multivariate_normal(centres[c], cov, size=n_c)
        Xs.append(pts)
        ys.append(np.full(n_c, c))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # shuffle so classes aren't blocked in order
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    names = ["majority", "overlap-A", "medium", "rare"]
    return X, y, names

SYNTHETIC_LOADERS = {
    "synthetic_noisy": _make_noisy,
    "synthetic_circles": _make_nonconvex,
    "synthetic_mislabeled": _make_mislabeled,
    "hierarchical": _make_hierarchical,
    "imbalanced_overlap": _make_imbalanced_overlap,
}

# Public registry of every selectable dataset name (real + synthetic).
DATASET_NAMES = list(SKLEARN_LOADERS) + list(SYNTHETIC_LOADERS)


def load_dataset(name: str = "wine", csv_path: Optional[str] = None,
                 label_col: Optional[str] = None
                 ) -> tuple[NDArray[np.float64], NDArray[np.int_], list[str], str]:
    """Return (X, y, target_names, dataset_name).
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if label_col is None or label_col not in df.columns:
            raise ValueError(f"--label-col must name a column in {csv_path}. "
                             f"Available: {list(df.columns)}")
        cats = pd.Categorical(df[label_col])
        y = cats.codes.astype(int)
        X = (df.drop(columns=[label_col])
               .select_dtypes(include=[np.number])
               .to_numpy(dtype=float))
        return X, y, [str(c) for c in cats.categories], Path(csv_path).stem

    if name in SYNTHETIC_LOADERS:
        X, y, target_names = SYNTHETIC_LOADERS[name]()
        return X, y, target_names, name

    if name not in SKLEARN_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choices: {DATASET_NAMES} (or use a CSV).")
    data = SKLEARN_LOADERS[name]()
    return data.data, data.target, [str(t) for t in data.target_names], name


def maybe_subsample(X: NDArray[np.float64], y: NDArray[np.int_],
                    max_samples: Optional[int]
                    ) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]]:
    """Stratified subsample to keep MDS (O(n^2)) tractable on big data.
    Returns X, y, ids (ids = original row indices, for honest reporting).
    """
    ids = np.arange(len(y))
    if max_samples is None or len(y) <= max_samples:
        return X, y, ids
    Xs, _, ys, _, ids_s, _ = train_test_split(
        X, y, ids, train_size=max_samples, stratify=y, random_state=RANDOM_STATE)
    print(f"[subsampled] {len(y)} -> {len(ys)} points (stratified) for speed")
    return Xs, ys, ids_s


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


def build_projections(X: NDArray[np.float64], standardize: bool = True,
                      weighted_mds: bool = False,
                      feature_weights: Optional[Sequence[float]] = None,
                      mahalanobis: bool = False
                      ) -> tuple[Projections, NDArray[np.float64]]:
    """Standardize (optional) then return ({'PCA': xy, 'MDS': xy}, X_high).

    X_high is the (possibly standardized) high-D array the projections were built
    from; the intrusion/extrusion metric needs it to rank true neighbours.

    When ``weighted_mds`` is True and ``feature_weights`` are supplied, the MDS
    embedding is built from a **precomputed** weighted-Euclidean dissimilarity
    matrix (each feature scaled by ``sqrt(weight)``) rather than sklearn's plain
    Euclidean metric. PCA is always computed on the unweighted (standardized)
    array so it stays a faithful reference; only the MDS panel reacts to the
    weights. ``X_high`` is likewise returned unweighted so the error metric keeps
    ranking true neighbours in the original space.
    """
    Xp = StandardScaler().fit_transform(X) if standardize else np.asarray(X, float)
    import time as _time
    _t0 = _time.perf_counter()
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xp)
    _t1 = _time.perf_counter()

    if mahalanobis:
        # MAHALANOBIS distance = whiten by the inverse covariance, so correlated
        # and high-variance directions are down-weighted (a label-free "adjusted
        # distance", i.e. unsupervised metric learning). Equivalent to Euclidean
        # distance on the whitened data.
        Xc = Xp - Xp.mean(axis=0)
        cov = np.cov(Xc, rowvar=False)
        cov = np.atleast_2d(cov)
        # regularise for invertibility, then symmetric inverse-sqrt whitening
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 1e-8, None)
            W_white = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
            Xwhite = Xc @ W_white
        except np.linalg.LinAlgError:
            Xwhite = Xc
        D = pairwise_distances(Xwhite, metric="euclidean")
        mds = MDS(n_components=2, dissimilarity="precomputed", n_init=4,
                  random_state=RANDOM_STATE).fit_transform(D)
    elif weighted_mds and feature_weights is not None:
        w = np.clip(np.asarray(feature_weights, dtype=float), 0.0, None)
        Xw = Xp * np.sqrt(w)                       # feature i scaled by sqrt(w_i)
        D = pairwise_distances(Xw, metric="euclidean")
        mds = MDS(n_components=2, dissimilarity="precomputed", n_init=4,
                  random_state=RANDOM_STATE).fit_transform(D)
    else:
        mds = MDS(n_components=2, dissimilarity="euclidean", n_init=4,
                  random_state=RANDOM_STATE).fit_transform(Xp)
    _t2 = _time.perf_counter()
    build_projections.last_timings = {
        "pca_ms": round((_t1 - _t0) * 1000.0, 2),
        "mds_ms": round((_t2 - _t1) * 1000.0, 2),
    }
    return {"PCA": pca, "MDS": mds}, Xp


# Classic 2D class-compactness error (with alpha/beta weighting)
def compute_error_scale(X_2d: NDArray[np.float64], labels: Sequence[int],
                        alpha: float = 1.0, beta: float = 1.0
                        ) -> NDArray[np.float64]:
    """error_i = -(alpha*d_same - beta*d_diff) / (alpha*d_same + beta*d_diff).

    ``d_same`` / ``d_diff`` are point i's mean distance to same- / other-class
    points in the 2D projection. ``alpha`` up-weights within-class tightness and
    ``beta`` up-weights between-class separation; ``alpha = beta = 1`` is the
    original scale. Result is in ``[-1, +1]``.
    """
    lab = np.asarray(labels)
    n = X_2d.shape[0]
    D = pairwise_distances(X_2d, metric="euclidean")
    errors = np.zeros(n)
    for i in range(n):
        same = lab == lab[i]
        same[i] = False                 # exclude self
        diff = ~(lab == lab[i])
        d_same = D[i, same].mean() if same.any() else 0.0
        d_diff = D[i, diff].mean() if diff.any() else 0.0
        num = alpha * d_same - beta * d_diff
        denom = alpha * d_same + beta * d_diff
        errors[i] = -num / denom if denom != 0 else 0.0
    return errors


# Intrusion / extrusion error  (precision-vs-recall, FP - FN)
def _ranks(D: NDArray[np.float64]) -> NDArray[np.int_]:
    """1-based rank of every point as a neighbour of every other (self = inf)."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(np.argsort(D, axis=1), axis=1) + 1   # nearest non-self -> 1


def per_point_neighbor_metrics(X_high: NDArray[np.float64],
                               X_2d: NDArray[np.float64],
                               k: int = 10,
                               k_vec: Optional[NDArray[np.int_]] = None
                               ) -> dict:
    """Per-point neighbourhood-preservation metrics for one 2-D projection.

    Treats each point as a "classifier of its own neighbourhood": the high-D
    k-nearest-neighbour SET is the ground truth (the neighbours that should be
    kept), and the 2-D k-nearest-neighbour SET is the prediction (the neighbours
    the projection actually kept). From the overlap of the two sets:

        precision_i = |kNN_2D ∩ kNN_highD| / |kNN_2D|
            -- of the neighbours the projection asserts, how many are real
               (1 - precision is the intrusion / false-positive rate)
        recall_i    = |kNN_2D ∩ kNN_highD| / |kNN_highD|
            -- of the real neighbours, how many the projection kept
               (1 - recall is the extrusion / false-negative rate)
        f1_i        = 2 * precision * recall / (precision + recall)

    With a symmetric fixed k, |kNN_2D| = |kNN_highD| = k, so precision == recall
    == f1 for that point; the three still differ when adaptive per-point k is
    used (then each point's high-D and 2-D neighbourhoods can have different
    sizes). All three are in [0, 1]; 1 = the projection preserved this point's
    neighbourhood perfectly.

    Also computes a per-point **Jensen-Shannon divergence** between soft
    neighbour distributions, as an alternative to the hard k-NN F1. Following
    the t-SNE idea, each point's neighbours get a probability from a Gaussian
    over distances, in high-D (P) and in 2-D (Q); JS(P||Q) in [0, 1] (log base
    2) measures how differently the projection distributes a point's
    neighbour-affinity. 0 = identical distributions (faithful); 1 = maximally
    different. Reported as ``js`` (divergence) and ``js_fidelity = 1 - js`` so
    that, like F1, higher == better.

    Returns per-point arrays (lists) for ``precision``, ``recall``, ``f1``,
    ``js``, ``js_fidelity`` plus their global means.
    """
    Dh = pairwise_distances(X_high, metric="euclidean")
    Dl = pairwise_distances(X_2d, metric="euclidean")
    n = Dh.shape[0]
    np.fill_diagonal(Dh, np.inf)
    np.fill_diagonal(Dl, np.inf)

    # neighbour ORDER (nearest first) in each space
    order_h = np.argsort(Dh, axis=1)
    order_l = np.argsort(Dl, axis=1)

    prec = np.zeros(n); rec = np.zeros(n); f1 = np.zeros(n)
    for i in range(n):
        ki = int(k_vec[i]) if k_vec is not None else int(k)
        ki = max(1, min(ki, n - 1))
        set_h = set(order_h[i, :ki].tolist())
        set_l = set(order_l[i, :ki].tolist())
        inter = len(set_h & set_l)
        p = inter / len(set_l) if set_l else 0.0
        r = inter / len(set_h) if set_h else 0.0
        prec[i] = p; rec[i] = r
        f1[i] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # ---- Jensen-Shannon divergence between soft neighbour distributions ----
    # per-point Gaussian bandwidth = distance to the k-th neighbour, so the
    # softness scales with local density (a t-SNE-style local perplexity proxy)
    def _soft(D, order):
        P = np.zeros((n, n))
        for i in range(n):
            ki = int(k_vec[i]) if k_vec is not None else int(k)
            ki = max(1, min(ki, n - 1))
            sigma = D[i, order[i, ki - 1]]
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = np.mean(D[i, order[i, :ki]]) or 1.0
            w = np.exp(-(D[i] ** 2) / (2.0 * sigma ** 2))
            w[i] = 0.0
            s = w.sum()
            P[i] = w / s if s > 0 else w
        return P

    P = _soft(Dh, order_h)
    Q = _soft(Dl, order_l)
    M = 0.5 * (P + Q)
    eps = 1e-12
    def _kl(A, B):
        A = np.clip(A, eps, 1.0); B = np.clip(B, eps, 1.0)
        return np.sum(A * (np.log2(A) - np.log2(B)), axis=1)
    js = 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)     # in [0, 1] with log base 2
    js = np.clip(js, 0.0, 1.0)
    js_fid = 1.0 - js

    # ---- per-point distance correlation (Euclidean-preservation) ----
    # For each point, the Spearman rank correlation between its distances to all
    # other points in high-D vs in 2-D. 1 = the projection preserves this point's
    # ---- per-point symmetric Local Stress, as a fidelity ----
    # For each point i, the weighted squared error between its high-D distances
    # and its 2-D distances to its nearest neighbours, with weights that decay
    # with distance (1/rank), so closer neighbours matter more. Both spaces are
    # scaled to unit mean neighbour distance first, so the comparison is scale-
    # free and symmetric (neither space privileged). Fidelity = 1 - stress,
    # clamped to [0, 1]; higher = the point's local distances are well preserved.
    order_h_ls = np.argsort(Dh, axis=1)
    local_stress = np.zeros(n)
    for i in range(n):
        ki = int(k_vec[i]) if k_vec is not None else int(k)
        ki = max(1, min(ki, n - 1))
        neigh = order_h_ls[i, :ki]
        dh = Dh[i, neigh].astype(float)
        dl = Dl[i, neigh].astype(float)
        # scale each space to unit mean neighbour distance (scale-free, symmetric)
        mh = dh.mean() or 1.0
        ml = dl.mean() or 1.0
        dh_s = dh / mh
        dl_s = dl / ml
        # 1/rank weights (rank 1 = nearest), normalised to sum 1
        w = 1.0 / np.arange(1, ki + 1)
        w = w / w.sum()
        # weighted mean squared difference of the (scaled) distances
        stress = float(np.sum(w * (dh_s - dl_s) ** 2))
        local_stress[i] = 1.0 - stress
    local_stress = np.clip(local_stress, 0.0, 1.0)

    # ---- per-point Mean Relative Rank Error (MRRE), as a fidelity ----
    # MRRE penalises rank differences of a point's k nearest neighbours between
    # the two spaces (a smoother version of trustworthiness/continuity). We report
    # a per-point fidelity = 1 - normalised-MRRE in [0, 1].
    rank_l = np.argsort(np.argsort(Dl, axis=1), axis=1).astype(float)
    mrre_fid = np.zeros(n)
    order_h_i = np.argsort(Dh, axis=1)
    for i in range(n):
        ki = int(k_vec[i]) if k_vec is not None else int(k)
        ki = max(1, min(ki, n - 1))
        neigh = order_h_i[i, :ki]
        # rank of each high-D neighbour in the 2-D ordering (1-based)
        r_l = rank_l[i, neigh] + 1.0
        r_h = np.arange(1, ki + 1)
        err = np.sum(np.abs(r_l - r_h) / r_h)
        norm = np.sum(np.abs((n - np.arange(1, ki + 1)) ) / np.arange(1, ki + 1)) or 1.0
        mrre_fid[i] = float(1.0 - min(err / norm, 1.0))
    mrre_fid = np.clip(mrre_fid, 0.0, 1.0)

    # ---- per-point LINEAR distance preservation (Pearson) ----
    # For each point, the Pearson correlation between its ACTUAL high-D distances
    # and its ACTUAL 2-D distances to every other point. Unlike Local Stress
    # (a local, rank-weighted squared error) or MRRE (rank-based), this asks
    # whether the projection preserved the point's distances *linearly* across
    # the whole dataset: a high value means "twice as far in high-D => about
    # twice as far in 2-D". Rescaled from [-1, 1] to [0, 1] as (r + 1) / 2 so it
    # reads like the other fidelities (higher = better).
    lindist = np.zeros(n)
    for i in range(n):
        a = np.delete(Dh[i], i).astype(float)
        b = np.delete(Dl[i], i).astype(float)
        sa, sb = a.std(), b.std()
        if sa > 1e-12 and sb > 1e-12:
            lindist[i] = float(np.corrcoef(a, b)[0, 1])
    lindist = np.clip((lindist + 1.0) / 2.0, 0.0, 1.0)

    return {
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "js": js.tolist(),
        "js_fidelity": js_fid.tolist(),
        "localstress": local_stress.tolist(),
        "mrre": mrre_fid.tolist(),
        "lindist": lindist.tolist(),
        "mean": {
            "precision": float(prec.mean()), "recall": float(rec.mean()),
            "f1": float(f1.mean()), "js": float(js.mean()),
            "js_fidelity": float(js_fid.mean()),
            "localstress": float(local_stress.mean()), "mrre": float(mrre_fid.mean()),
            "lindist": float(lindist.mean()),
        },
    }


def adaptive_k(D_high: NDArray[np.float64], k_base: int,
               k_min: int = 3, k_max: Optional[int] = None
               ) -> NDArray[np.int_]:
    """Per-point neighbourhood size driven by local density.

    Dense points (small mean distance to their ``k_base`` nearest neighbours)
    get a smaller k; sparse/outlying points get a larger k. The mean local
    radius is mapped through the population median so the *median* point keeps
    roughly ``k_base``, then clamped to ``[k_min, k_max]``.

    Parameters
    ----------
    D_high : ndarray, shape (n, n)
        High-D pairwise distance matrix.
    k_base : int
        Target neighbourhood size for a typical (median-density) point.
    k_min, k_max : int
        Hard bounds; ``k_max`` defaults to ``2*k_base``.

    Returns
    -------
    ndarray of int, shape (n,)
        Per-point k.
    """
    n = D_high.shape[0]
    if k_max is None:
        k_max = 2 * k_base
    Dc = D_high.copy()
    np.fill_diagonal(Dc, np.inf)
    kk = max(1, min(k_base, n - 1))
    # mean distance to each point's kk nearest neighbours = local radius
    part = np.partition(Dc, kk - 1, axis=1)[:, :kk]
    local_radius = part.mean(axis=1)
    med = np.median(local_radius)
    if med <= 0:
        return np.full(n, k_base, dtype=int)
    scaled = np.rint(k_base * local_radius / med).astype(int)
    return np.clip(scaled, k_min, k_max)


def compute_intrusion_extrusion_error(
        X_high: NDArray[np.float64], X_2d: NDArray[np.float64], k: int = 10,
        labels: Optional[Sequence[int]] = None, class_aware: bool = True,
        weight: str = "rank", adaptive: bool = False,
        return_parts: bool = False):
    """Per-point projection-fidelity error ``FP - FN`` in ``[-1, 1]``.

    FP_i (intrusion): points in i's 2D k-neighbourhood whose TRUE high-D rank > k
                      (the projection placed them near i falsely).
    FN_i (extrusion): i's true high-D k-neighbours whose 2D rank > k
                      (the projection pushed real neighbours away).

    Parameters
    ----------
    weight : {'rank', 'distance'}
        ``rank`` weights each offender by ``rank_other_space - k`` (the original
        trustworthiness/continuity severity). ``distance`` instead weights each
        offender by the actual distance discrepancy -- intrusions by how *far*
        they truly are in high-D despite being pulled close in 2D, extrusions the
        mirror -- so a badly misplaced near-neighbour hurts more than a marginal
        one. Distance-weighted scores are normalised per point to ``[0, 1]``.
    adaptive : bool
        If True, ``k`` becomes a per-point value from :func:`adaptive_k`
        (density-driven), and ``k`` is used as the base/target.
    class_aware : bool
        Count only cross-class intrusions as FP and only same-class extrusions
        as FN, so the score reflects misrepresented *class* structure.
    return_parts : bool
        If True, return ``(err, fp, fn)`` with the non-negative FP and FN
        components separately (``err == fp - fn``); otherwise just ``err``.

    Returns
    -------
    ndarray, or (ndarray, ndarray, ndarray)
        See ``return_parts``.
    """
    n = X_2d.shape[0]
    Dh = pairwise_distances(X_high, metric="euclidean")
    Dl = pairwise_distances(X_2d, metric="euclidean")
    rh = _ranks(Dh)
    rl = _ranks(Dl)
    lab = np.asarray(labels) if labels is not None else None

    if adaptive:
        k_vec = adaptive_k(Dh, k)
    else:
        k_scalar = max(1, min(k, (2 * n - 2) // 3))
        k_vec = np.full(n, k_scalar, dtype=int)

    fp = np.zeros(n)
    fn = np.zeros(n)
    for i in range(n):
        ki = int(k_vec[i])
        ki = max(1, min(ki, (2 * n - 2) // 3))          # keep normaliser valid
        intruder = (rl[i] <= ki) & (rh[i] > ki)         # close in 2D, far high-D
        extruded = (rh[i] <= ki) & (rl[i] > ki)         # close high-D, far in 2D
        if class_aware and lab is not None:
            other = lab != lab[i]
            intruder &= other                            # only class-mixing
            extruded &= ~other                           # only lost class-mates

        if weight == "rank":
            W = ki * (2 * n - 3 * ki - 1) / 2.0          # max rank penalty
            W = W if W > 0 else 1.0
            fp[i] = np.sum(rh[i][intruder] - ki) / W
            fn[i] = np.sum(rl[i][extruded] - ki) / W
        elif weight == "distance":
            # intrusion severity: far in high-D despite being pulled close in 2D
            fp_raw = np.sum(Dh[i][intruder])
            # extrusion severity: far in 2D despite being a true high-D neighbour
            fn_raw = np.sum(Dl[i][extruded])
            row_h = Dh[i].sum()
            row_l = Dl[i].sum()
            fp[i] = fp_raw / row_h if row_h > 0 else 0.0
            fn[i] = fn_raw / row_l if row_l > 0 else 0.0
        else:
            raise ValueError(f"Unknown weight '{weight}'. Use rank or distance.")

    err = fp - fn
    if return_parts:
        return err, fp, fn
    return err


def compute_errors(proj: Projections, X_high: NDArray[np.float64],
                   labels: Sequence[int], metric: str = "intrusion", k: int = 10,
                   class_aware: bool = True, weight: str = "rank",
                   adaptive: bool = False, alpha: float = 1.0, beta: float = 1.0,
                   return_parts: bool = False):
    """Dispatch to a metric for every projection.

    ``metric='classic'`` -> 2D-only class compactness (honours ``alpha``/``beta``).
    ``metric='intrusion'`` -> FP-FN intrusion/extrusion (honours ``weight``,
    ``adaptive``, ``class_aware``).

    Returns an ``{name: err}`` dict, or if ``return_parts`` and the intrusion
    metric is used, ``(errors, fps, fns)`` -- three parallel dicts.
    """
    if metric == "classic":
        errs = {m: compute_error_scale(xy, labels, alpha=alpha, beta=beta)
                for m, xy in proj.items()}
        if return_parts:
            # classic has no separable FP/FN; expose positive/negative halves
            fps = {m: np.clip(e, 0, None) for m, e in errs.items()}
            fns = {m: np.clip(-e, 0, None) for m, e in errs.items()}
            return errs, fps, fns
        return errs

    if return_parts:
        errs = {}
        fps = {}
        fns = {}
        for m, xy in proj.items():
            e, fp, fn = compute_intrusion_extrusion_error(
                X_high, xy, k=k, labels=labels, class_aware=class_aware,
                weight=weight, adaptive=adaptive, return_parts=True)
            errs[m], fps[m], fns[m] = e, fp, fn
        return errs, fps, fns

    return {m: compute_intrusion_extrusion_error(
                X_high, xy, k=k, labels=labels, class_aware=class_aware,
                weight=weight, adaptive=adaptive)
            for m, xy in proj.items()}