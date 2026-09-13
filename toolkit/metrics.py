"""
Per-point projection-fidelity metrics.

Every measure here compares one point's high-dimensional neighbourhood with the
same point's neighbourhood in a 2-D embedding. Nothing in this module uses the
class labels for scoring.

Contents:
  * ``adaptive_k``                     per-point neighbourhood size from local density
  * ``per_point_neighbor_metrics``     neighbour F1, JS fidelity, local stress,
                                       normalised stress
  * ``compute_intrusion_extrusion_error``  rank- or distance-weighted intrusion
                                       and extrusion severities
  * ``compute_errors``                 dispatcher over a projection dict
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances

from toolkit.projections import Projections, ErrorMap


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

    # ---- per-point distance correlation ----
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

    # ---- per-point NORMALIZED STRESS (distance-magnitude, NOT rank-based) ----
    # Kruskal-style normalized stress over each point's neighbourhood, using the
    # ACTUAL distance magnitudes (no rank weighting, no per-neighbour scaling):
    #     stress_i = sum_j (d_hi(i,j) - s*d_2d(i,j))^2 / sum_j d_hi(i,j)^2
    # where s is the optimal single scale that minimises the numerator
    # (s = <d_hi,d_2d>/<d_2d,d_2d>), so it is invariant to the global size of the
    # 2-D layout but NOT to the relative stretching/squashing of individual
    # distances -- which is exactly the distortion we want to see. Reported as
    # fidelity = 1 - stress, clamped to [0, 1]. This produces a much wider,
    # better-distributed per-point (and per-class) score than the set/rank
    # metrics, which tend to saturate near 1.
    normstress = np.zeros(n)
    for i in range(n):
        ki = int(k_vec[i]) if k_vec is not None else int(k)
        ki = max(1, min(ki, n - 1))
        neigh = order_h[i, :ki]
        dh = Dh[i, neigh].astype(float)
        dl = Dl[i, neigh].astype(float)
        denom = float(np.dot(dh, dh))
        if denom <= 1e-12:
            normstress[i] = 1.0
            continue
        dd = float(np.dot(dl, dl))
        s = float(np.dot(dh, dl) / dd) if dd > 1e-12 else 0.0   # optimal scale
        num = float(np.sum((dh - s * dl) ** 2))
        normstress[i] = 1.0 - (num / denom)
    normstress = np.clip(normstress, 0.0, 1.0)

    # per-point neighbourhood size actually used (for display; especially useful
    # under adaptive k where every point has a different k)
    if k_vec is not None:
        kv_out = [int(max(1, min(int(k_vec[i]), n - 1))) for i in range(n)]
    else:
        kv_out = [int(k)] * n

    return {
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "js": js.tolist(),
        "js_fidelity": js_fid.tolist(),
        "localstress": local_stress.tolist(),
        "normstress": normstress.tolist(),
        "k_used": kv_out,
        "mean": {
            "precision": float(prec.mean()), "recall": float(rec.mean()),
            "f1": float(f1.mean()), "js": float(js.mean()),
            "js_fidelity": float(js_fid.mean()),
            "localstress": float(local_stress.mean()),
            "normstress": float(normstress.mean()),
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
                   class_aware: bool = False, weight: str = "rank",
                   adaptive: bool = False, return_parts: bool = False):
    """Compute the intrusion/extrusion severities for every projection.

    Returns an ``{name: err}`` dict, or ``(errors, fps, fns)`` -- three parallel
    dicts -- when ``return_parts`` is set.

    DISABLED: ``metric='classic'`` used to select a 2-D-only class-compactness
    score. It is not offered by the interface; the implementation now lives in
    ``legacy/legacy_misc.py`` and is not imported.
    """
    if metric == "classic":
        raise ValueError(
            "The 'classic' class-compactness metric is retired. Its "
            "implementation is kept in legacy/legacy_misc.py and is not wired "
            "into the pipeline; use metric='intrusion'.")

    if return_parts:
        errs, fps, fns = {}, {}, {}
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
