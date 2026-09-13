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


PROJECTORS = ("PCA", "MDS", "t-SNE", "UMAP")


def _embed(kind: str, Xp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a 2-D embedding of ``Xp`` under one named method."""
    n = Xp.shape[0]
    if kind == "PCA":
        return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xp)
    if kind == "MDS":
        return MDS(n_components=2, dissimilarity="euclidean", n_init=4,
                   random_state=RANDOM_STATE).fit_transform(Xp)
    if kind == "t-SNE":
        from sklearn.manifold import TSNE
        # perplexity must stay below n/3 or the algorithm is degenerate
        perp = max(5.0, min(30.0, (n - 1) / 3.0))
        return TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=RANDOM_STATE).fit_transform(Xp)
    if kind == "UMAP":
        try:
            import umap
        except ImportError as exc:                       # optional dependency
            raise RuntimeError(
                "UMAP was selected but umap-learn is not installed. "
                "Run: pip install umap-learn") from exc
        return umap.UMAP(n_components=2, n_neighbors=min(15, max(2, n - 1)),
                         random_state=RANDOM_STATE).fit_transform(Xp)
    raise ValueError(f"Unknown projection '{kind}'. Choose from {PROJECTORS}.")


def build_projections(X: NDArray[np.float64], standardize: bool = True,
                      method_a: str = "PCA", method_b: str = "MDS"
                      ) -> tuple[Projections, NDArray[np.float64]]:
    """Standardize (optional), then embed twice and return ({A, B}, X_high).

    The toolkit compares exactly TWO embeddings, because the verdict, the
    linked scatters and the Level/Evenness view are all inherently pairwise.
    Which two is now the user's choice: any pair drawn from PCA, MDS, t-SNE and
    UMAP, including a method against itself is refused upstream.

    The returned dict is keyed by SLOT -- ``"PCA"`` for the left panel and
    ``"MDS"`` for the right -- not by the method that filled it. Those key names
    are historical: every consumer downstream, from the verdict to the payload
    to the browser, addresses the two panels by those literals. The method
    actually used is carried alongside in ``last_methods`` and is what the
    interface labels things with.

    ``X_high`` is the (possibly standardized) high-D array the projections were
    built from; the per-point metrics need it to rank true neighbours. Both
    embeddings are therefore measured against the same reference geometry.

    DISABLED: this function previously accepted ``weighted_mds`` /
    ``feature_weights`` / ``mahalanobis`` to build the MDS layout from an
    adjusted dissimilarity matrix. No interface control reaches that code, and
    each variant cost a second full MDS fit per run. The implementation is kept
    for reference in ``legacy/adjusted_distance.py`` and is not imported.
    """
    for m in (method_a, method_b):
        if m not in PROJECTORS:
            raise ValueError(f"Unknown projection '{m}'. Choose from {PROJECTORS}.")

    Xp = StandardScaler().fit_transform(X) if standardize else np.asarray(X, float)

    t0 = time.perf_counter()
    a = _embed(method_a, Xp)
    t1 = time.perf_counter()
    b = _embed(method_b, Xp)
    t2 = time.perf_counter()

    build_projections.last_timings = {
        "pca_ms": round((t1 - t0) * 1000.0, 2),
        "mds_ms": round((t2 - t1) * 1000.0, 2),
    }
    build_projections.last_methods = {"PCA": method_a, "MDS": method_b}
    return {"PCA": a, "MDS": b}, Xp