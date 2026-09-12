"""
Assorted retired pieces.

  * ``compute_error_scale``       the original 2-D-only class-compactness score,
                                  superseded by the per-point neighbourhood
                                  metrics in ``metrics.py``.
  * centroid trajectories         class centroids traced across high-D, PCA and
                                  MDS on a shared normalised frame. Still cheap,
                                  but the view that rendered it was removed.
  * ``_report_untrustworthy``     the signed-error threshold report, superseded
                                  by the consensus-fidelity flag in
                                  ``pipeline.py``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# =====================================================================
# QUARANTINED -- NOT IMPORTED BY THE RUNNING PIPELINE
# ---------------------------------------------------------------------
# Nothing in this file is reachable from the browser interface. It is kept
# for reference and for the report's history section only. Do not add an
# import of this module to pipeline.py without also adding the interface
# control that makes its output visible; the cost of each item is noted
# below.
# =====================================================================


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


def _class_centroids(coords: np.ndarray, labels: np.ndarray,
                     classes: np.ndarray) -> np.ndarray:
    """Mean position of each class in ``classes`` for a 2-D coordinate array."""
    return np.array([coords[labels == c].mean(axis=0) for c in classes])


def _normalize_frame(cent: np.ndarray) -> np.ndarray:
    """Centre a set of 2-D centroids and scale so their RMS radius is 1.

    This puts every space (high-D reference, PCA, MDS) on a comparable frame so
    the *relative* class geometry: how spread out and how separated the classes
    are, can be compared across representations on one canvas.
    """
    c = cent - cent.mean(axis=0)
    rms = np.sqrt((c ** 2).sum(axis=1).mean())
    return c / rms if rms > 0 else c


def _centroid_trajectories(X_high: np.ndarray, pca_2d: np.ndarray,
                           mds_2d: np.ndarray, labels: np.ndarray,
                           target_names, classes: np.ndarray) -> dict:
    """Trace each class centroid across high-D (via PCA of the centroids), PCA
    and MDS, each frame normalised so between-class geometry is comparable.

    The high-D centroids live in feature space; to place them on the same 2-D
    canvas they are projected with a PCA fit on the centroids themselves (a
    faithful 2-D view of the between-class configuration). PCA and MDS centroids
    are taken directly from those embeddings. Every stage is then centred and
    RMS-normalised. Also returns mean pairwise between-centroid distances per
    stage (on the normalised frame) so preservation of class separation is
    quantifiable.
    """
    hi_cent = _class_centroids(X_high, labels, classes)
    if hi_cent.shape[0] >= 2 and hi_cent.shape[1] > 2:
        hi_2d = PCA(n_components=2, random_state=42).fit_transform(hi_cent)
    elif hi_cent.shape[1] == 2:
        hi_2d = hi_cent
    else:                                   
        hi_2d = np.column_stack([hi_cent[:, 0],
                                 np.zeros(hi_cent.shape[0])]) \
            if hi_cent.shape[1] else np.zeros((hi_cent.shape[0], 2))

    pca_cent = _class_centroids(pca_2d, labels, classes)
    mds_cent = _class_centroids(mds_2d, labels, classes)

    stages = {
        "High-D": _normalize_frame(hi_2d),
        "PCA": _normalize_frame(pca_cent),
        "MDS": _normalize_frame(mds_cent),
    }

    def _mean_pairwise(c: np.ndarray) -> float:
        if c.shape[0] < 2:
            return 0.0
        d = pairwise_distances(c)
        iu = np.triu_indices(c.shape[0], k=1)
        return float(d[iu].mean())

    return {
        "stage_names": list(stages),
        "classes": [target_names[c] if c < len(target_names) else str(c)
                    for c in classes],
        "class_codes": [int(c) for c in classes],
        "coords": {s: pts.tolist() for s, pts in stages.items()},
        "mean_between": {s: round(_mean_pairwise(pts), 4)
                         for s, pts in stages.items()},
    }


def _report_untrustworthy(pca_err, mds_err, labels, target_names, ids, thresh,
                          pca_fp, pca_fn, mds_fp, mds_fn) -> list[dict]:
    """Print the untrustworthy report (for the console) and return its rows."""
    # A point is "untrustworthy" when the projection asserts a FALSE CLOSENESS,
    # i.e. an INTRUSION (error > +thresh): points drawn together in 2-D that are
    # actually far apart -- these mislead the analyst into seeing structure that
    # is not there. A large NEGATIVE error is an EXTRUSION (a true neighbour
    # pushed away): a loss of information, but not a false claim, and on cleanly
    # separated data these dominate |error| while being mostly benign. Flagging
    # on |error| would therefore select the WRONG points. We flag intrusions as
    # the untrustworthy set and report extrusions separately.
    intr = (pca_err > thresh) | (mds_err > thresh)
    extr = ((pca_err < -thresh) | (mds_err < -thresh)) & ~intr
    sel = np.where(intr)[0]
    ext_sel = np.where(extr)[0]
    print("\n" + "=" * 60)
    print(f"UNTRUSTWORTHY POINTS  (false closeness: error > +{thresh} in PCA or MDS)")
    print("=" * 60)
    print(f"Count: {len(sel)}   (+ {len(ext_sel)} extrusion-only points, error < -{thresh})")
    rows = []
    for i in sel:
        rows.append({
            "index": int(ids[i]),
            "kind": "false closeness",
            "true_class": target_names[labels[i]] if labels[i] < len(target_names)
                          else str(labels[i]),
            "pca_error": round(float(pca_err[i]), 3),
            "mds_error": round(float(mds_err[i]), 3),
            "pca_FP": round(float(pca_fp[i]), 3),
            "pca_FN": round(float(pca_fn[i]), 3),
            "mds_FP": round(float(mds_fp[i]), 3),
            "mds_FN": round(float(mds_fn[i]), 3),
        })
    if len(sel):
        # compact printed table
        hdr = (f"{'index':>6} {'true_class':>12} {'pca_err':>8} {'mds_err':>8} "
               f"{'pca_FP':>7} {'pca_FN':>7} {'mds_FP':>7} {'mds_FN':>7}")
        print(hdr)
        for r in rows:
            print(f"{r['index']:>6} {r['true_class']:>12} "
                  f"{r['pca_error']:>8.3f} {r['mds_error']:>8.3f} "
                  f"{r['pca_FP']:>7.3f} {r['pca_FN']:>7.3f} "
                  f"{r['mds_FP']:>7.3f} {r['mds_FN']:>7.3f}")
    disagree = int(np.sum(((pca_err[sel] > 0) & (mds_err[sel] < 0)) |
                          ((pca_err[sel] < 0) & (mds_err[sel] > 0)))) if len(sel) else 0
    print(f"\nSign disagreements (PCA/MDS opposite signs): {disagree}")
    return rows
