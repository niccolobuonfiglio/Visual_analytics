"""
Dataset acquisition for the projection-fidelity toolkit.

Everything that turns a name (or a CSV path) into ``(X, y, target_names,
dataset_name)`` lives here: the built-in scikit-learn loaders, the synthetic
generators written to expose specific failure modes, and stratified
subsampling.

Split out of the former ``error_core`` so that loading concerns are separable
from projection and metric concerns. Imports scikit-learn and pandas only; no
plotting, no web framework.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import datasets as skds
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


# Synthetic datasets, each built to expose a DIFFERENT failure mode the real
# datasets above cannot show. Fixed seed so runs are reproducible. Each factory
# returns (X, y, target_names) like the sklearn loaders' relevant fields.


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
