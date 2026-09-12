"""
Projection construction: standardise, then embed in two dimensions.

Only the plain Euclidean path is live. The adjusted-distance variants
(Mahalanobis whitening and locality/mutual-information feature weighting) are
not reachable from the interface and have been moved to
``legacy/adjusted_distance.py``; see the note in ``build_projections``.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

Projections = dict[str, NDArray[np.float64]]
ErrorMap = dict[str, NDArray[np.float64]]


def build_projections(X: NDArray[np.float64], standardize: bool = True
                      ) -> tuple[Projections, NDArray[np.float64]]:
    """Standardize (optional) then return ({'PCA': xy, 'MDS': xy}, X_high).

    ``X_high`` is the (possibly standardized) high-D array the projections were
    built from; the per-point metrics need it to rank true neighbours. Both
    embeddings are therefore measured against the same reference geometry.

    DISABLED: this function previously accepted ``weighted_mds`` /
    ``feature_weights`` / ``mahalanobis`` to build the MDS layout from an
    adjusted dissimilarity matrix. No interface control reaches that code, and
    each variant cost a second full MDS fit per run. The implementation is kept
    for reference in ``legacy/adjusted_distance.py`` and is not imported.
    """
    Xp = StandardScaler().fit_transform(X) if standardize else np.asarray(X, float)

    t0 = time.perf_counter()
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xp)
    t1 = time.perf_counter()
    mds = MDS(n_components=2, dissimilarity="euclidean", n_init=4,
              random_state=RANDOM_STATE).fit_transform(Xp)
    t2 = time.perf_counter()

    build_projections.last_timings = {
        "pca_ms": round((t1 - t0) * 1000.0, 2),
        "mds_ms": round((t2 - t1) * 1000.0, 2),
    }
    return {"PCA": pca, "MDS": mds}, Xp
