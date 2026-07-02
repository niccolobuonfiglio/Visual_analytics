"""
Shared core for the PCA/MDS continuous error-scale project (dataset-agnostic).

No plotting / no matplotlib here on purpose, so both the static analysis script
and the interactive brushing tool can import it without backend conflicts.

Error metric (canonical "professor" scale):
    error = -(d_same - d_diff) / (d_same + d_diff)   in [-1, +1]
        < 0  -> drawn toward OTHER classes    = FN (-1)
        > 0  -> tightly embedded in own class = FP (+1)
        = 0  -> equidistant (True)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skds
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

# Built-in sklearn loaders exposing .data / .target / .target_names
SKLEARN_LOADERS = {
    "wine": skds.load_wine,
    "iris": skds.load_iris,
    "breast_cancer": skds.load_breast_cancer,
    "digits": skds.load_digits,
}


def load_dataset(name="wine", csv_path=None, label_col=None):
    """Return (X, y, target_names: list[str], dataset_name: str)."""
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


def describe(X, y, target_names, name):
    classes, counts = np.unique(y, return_counts=True)
    print("=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)
    print(f"Samples : {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes : {len(classes)}")
    for c, n in zip(classes, counts):
        label = target_names[c] if c < len(target_names) else str(c)
        print(f"  class {c} ({label}): {n}")


def maybe_subsample(X, y, max_samples):
    """Stratified subsample to keep MDS (O(n^2)) tractable on big data.
    Returns X, y, ids (ids = original row indices, for honest reporting)."""
    ids = np.arange(len(y))
    if max_samples is None or len(y) <= max_samples:
        return X, y, ids
    Xs, _, ys, _, ids_s, _ = train_test_split(
        X, y, ids, train_size=max_samples, stratify=y, random_state=RANDOM_STATE)
    print(f"[subsampled] {len(y)} -> {len(ys)} points (stratified) for speed")
    return Xs, ys, ids_s


def compute_error_scale(X_2d, labels):
    """error_i = -(d_same - d_diff) / (d_same + d_diff), in [-1, +1]."""
    labels = np.asarray(labels)
    n = X_2d.shape[0]
    D = pairwise_distances(X_2d, metric="euclidean")
    errors = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False                 # exclude self
        diff = ~(labels == labels[i])
        d_same = D[i, same].mean() if same.any() else 0.0
        d_diff = D[i, diff].mean() if diff.any() else 0.0
        denom = d_same + d_diff
        errors[i] = -(d_same - d_diff) / denom if denom != 0 else 0.0
    return errors


def build_projections(X, standardize=True):
    """Standardize (optional) then return ({'PCA': xy, 'MDS': xy}, X_high).

    X_high is the (possibly standardized) high-D array the projections were built
    from; the intrusion/extrusion metric needs it to rank true neighbours.
    """
    Xp = StandardScaler().fit_transform(X) if standardize else np.asarray(X, float)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xp)
    mds = MDS(n_components=2, dissimilarity="euclidean", n_init=4,
              random_state=RANDOM_STATE).fit_transform(Xp)
    return {"PCA": pca, "MDS": mds}, Xp


# ---------------------------------------------------------------------
# Intrusion / extrusion error  (precision-vs-recall, FP - FN)
# ---------------------------------------------------------------------
def _ranks(D):
    """1-based rank of every point as a neighbour of every other (self = inf)."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(np.argsort(D, axis=1), axis=1) + 1   # nearest non-self -> 1


def compute_intrusion_extrusion_error(X_high, X_2d, k=10, labels=None,
                                      class_aware=True):
    """error_i = FP_i - FN_i in [-1, 1], rank-severity weighted.

    FP_i (intrusion): points in i's 2D k-neighbourhood whose TRUE high-D rank > k
                      (the projection placed them near i falsely).
    FN_i (extrusion): i's true high-D k-neighbours whose 2D rank > k
                      (the projection pushed real neighbours away).
    Each weighted by how far off it is (rank - k) and normalised to [0, 1].

    class_aware=True counts only cross-class intrusions as FP and only same-class
    extrusions as FN, so the score reflects misrepresented *class* structure.
    """
    n = X_2d.shape[0]
    # need 2n - 3k - 1 > 0 for a valid normaliser; clamp k if necessary
    k = max(1, min(k, (2 * n - 2) // 3))
    rh = _ranks(pairwise_distances(X_high, metric="euclidean"))
    rl = _ranks(pairwise_distances(X_2d, metric="euclidean"))
    W = k * (2 * n - 3 * k - 1) / 2.0          # max possible per-point penalty
    labels = np.asarray(labels) if labels is not None else None

    err = np.zeros(n)
    for i in range(n):
        intruder = (rl[i] <= k) & (rh[i] > k)   # close in 2D, far in high-D
        extruded = (rh[i] <= k) & (rl[i] > k)   # close in high-D, far in 2D
        if class_aware and labels is not None:
            other = labels != labels[i]
            intruder &= other                   # only class-mixing intrusions
            extruded &= ~other                  # only lost class-mates
        fp = np.sum(rh[i][intruder] - k) / W
        fn = np.sum(rl[i][extruded] - k) / W
        err[i] = fp - fn
    return err


def compute_errors(proj, X_high, labels, metric="intrusion", k=10,
                   class_aware=True):
    """Dispatch: 'classic' (2D-only class compactness) or 'intrusion' (FP-FN)."""
    if metric == "classic":
        return {m: compute_error_scale(xy, labels) for m, xy in proj.items()}
    return {m: compute_intrusion_extrusion_error(X_high, xy, k=k, labels=labels,
                                                 class_aware=class_aware)
            for m, xy in proj.items()}