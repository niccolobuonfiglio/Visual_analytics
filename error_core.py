"""
Shared core for the PCA/MDS continuous error-scale project (dataset-agnostic).

No plotting / no matplotlib here on purpose, so both the static analysis script
and the interactive brushing tool can import it without backend conflicts.

Two metric families are provided:

Classic class-compactness scale (2D only)::

    error = -(alpha*d_same - beta*d_diff) / (alpha*d_same + beta*d_diff)   in [-1, +1]
        < 0  -> drawn toward OTHER classes    = FN (-1)
        > 0  -> tightly embedded in own class = FP (+1)
        = 0  -> equidistant (True)

    (alpha = beta = 1 recovers the original professor scale.)

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

# re-exported so callers can keep importing describe from error_core if they like
from utils import describe  # noqa: F401

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


def load_dataset(name: str = "wine", csv_path: Optional[str] = None,
                 label_col: Optional[str] = None
                 ) -> tuple[NDArray[np.float64], NDArray[np.int_], list[str], str]:
    """Return (X, y, target_names, dataset_name).

    If ``csv_path`` is given, the named ``label_col`` becomes the class column
    and every remaining numeric column becomes a feature; otherwise a built-in
    sklearn loader named by ``name`` is used.
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

    if name not in SKLEARN_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choices: {list(SKLEARN_LOADERS)} (or use a CSV).")
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
                            method: str = "inverse_variance",
                            custom: Optional[Sequence[float]] = None
                            ) -> NDArray[np.float64]:
    """Return a non-negative per-feature weight vector for weighted MDS.

    The weights scale each feature (as ``sqrt(w) * column``) before Euclidean
    distances are computed, so a feature with weight ``w`` contributes ``w`` times
    as much to the squared distance. Weights are normalised to average 1 so the
    overall distance scale is comparable to the unweighted case.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix the weights are computed from (already standardized when
        standardization is on, so ``inverse_variance`` is ~uniform there).
    labels : sequence of int, optional
        Class labels; required for ``mutual_info``.
    method : {'inverse_variance', 'correlation', 'mutual_info', 'custom'}
        - ``inverse_variance`` : ``1 / var`` per feature — down-weights noisy,
          high-variance columns.
        - ``correlation`` : down-weight features that are redundant (highly
          correlated with the rest); weight ``1 - mean|corr to others|``.
        - ``mutual_info`` : up-weight features informative about the class label
          (mutual information, needs ``labels``).
        - ``custom`` : use the supplied ``custom`` vector verbatim.
    custom : sequence of float, optional
        Explicit weights, used when ``method='custom'``.

    Returns
    -------
    ndarray, shape (n_features,)
        Non-negative weights, normalised to mean 1.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]

    if method == "custom":
        if custom is None:
            raise ValueError("method='custom' requires a custom weight vector.")
        w = np.asarray(custom, dtype=float)
        if w.shape[0] != n_features:
            raise ValueError(f"custom weights length {w.shape[0]} != "
                             f"feature count {n_features}.")
    elif method == "inverse_variance":
        var = X.var(axis=0)
        w = 1.0 / np.where(var > 0, var, np.nan)
        w = np.nan_to_num(w, nan=0.0)
    elif method == "correlation":
        if n_features < 2:
            w = np.ones(n_features)
        else:
            C = np.corrcoef(X, rowvar=False)
            C = np.nan_to_num(C, nan=0.0)
            np.fill_diagonal(C, 0.0)
            redundancy = np.abs(C).sum(axis=1) / (n_features - 1)
            w = 1.0 - redundancy            # unique features weigh more
    elif method == "mutual_info":
        if labels is None:
            raise ValueError("method='mutual_info' requires labels.")
        w = mutual_info_classif(X, np.asarray(labels),
                                random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown weight method '{method}'. Use "
                         "inverse_variance, correlation, mutual_info or custom.")

    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    if total <= 0:                          # degenerate -> fall back to uniform
        return np.ones(n_features)
    return w * (n_features / total)         # normalise to mean 1


def build_projections(X: NDArray[np.float64], standardize: bool = True,
                      weighted_mds: bool = False,
                      feature_weights: Optional[Sequence[float]] = None
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
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xp)

    if weighted_mds and feature_weights is not None:
        w = np.clip(np.asarray(feature_weights, dtype=float), 0.0, None)
        Xw = Xp * np.sqrt(w)                       # feature i scaled by sqrt(w_i)
        D = pairwise_distances(Xw, metric="euclidean")
        mds = MDS(n_components=2, dissimilarity="precomputed", n_init=4,
                  random_state=RANDOM_STATE).fit_transform(D)
    else:
        mds = MDS(n_components=2, dissimilarity="euclidean", n_init=4,
                  random_state=RANDOM_STATE).fit_transform(Xp)
    return {"PCA": pca, "MDS": mds}, Xp


# ---------------------------------------------------------------------
# Classic 2D class-compactness error (now with alpha/beta weighting)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Intrusion / extrusion error  (precision-vs-recall, FP - FN)
# ---------------------------------------------------------------------
def _ranks(D: NDArray[np.float64]) -> NDArray[np.int_]:
    """1-based rank of every point as a neighbour of every other (self = inf)."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(np.argsort(D, axis=1), axis=1) + 1   # nearest non-self -> 1


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