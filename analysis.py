"""
Analysis layer: supervised cluster-structure metrics, k-means validation,
parameter search and a mechanical PCA-vs-MDS verdict.

--------------------------------------------------------------------------
WHY THIS MODULE EXISTS
--------------------------------------------------------------------------
The toolkit's own intrusion/extrusion metric scores projection *fidelity*: it
compares the high-D neighbourhood of a point with its 2-D neighbourhood, so
it can only answer "did the projection distort this point?".

This module answers a different, complementary question: given the class
labels as the expected clusters, is that structure genuinely present in the
data at all? It builds a similarity graph over the high-D points and measures
supervised precision / recall / F-score on it, so that:

  - structure genuinely absent from the data can never be recovered by any
    projection -- a low score here is a property of the DATA, not the picture;
  - structure present in the data but missing from the picture is a projection
    artifact -- caught instead by the fidelity metric.

Having both families lets the analyst attribute a missing cluster to the right
cause. The module also supplies a k-means test of whether the labels really
are the cluster structure, a simulated-annealing search for the neighbourhood
parameter, and a composite PCA-vs-MDS verdict with a bootstrap confidence
interval.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             pairwise_distances, silhouette_score)
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42


# ==========================================================================
# 1. SIMILARITY GRAPH  (supervised precision / recall / F-score)
# ==========================================================================
def build_knn_graph(X: NDArray[np.float64], k: int = 15,
                    sigma: Optional[float] = None
                    ) -> tuple[list[set[int]], NDArray[np.float64]]:
    """Symmetric k-NN similarity graph G = (V, E, Omega) on the high-D data.

    Graph-based DR techniques (t-SNE, UMAP) build a neighbourhood graph
    internally, but PCA and MDS are *global* techniques and do not. To measure
    supervised cluster structure for them we construct the canonical local
    model: a symmetric k-nearest-neighbour graph with Gaussian-kernel edge
    weights

        omega_ij = exp( -d_ij^2 / (2 * sigma^2) )

    where ``sigma`` defaults to the mean k-NN distance (a standard local-scale
    heuristic). ``k`` is the one free parameter of this analysis -- the
    neighbourhood size -- analogous to ``n_neighbors`` in UMAP.

    Returns
    -------
    adj : list of set of int
        adj[i] is the neighbour set Gamma(v_i).
    W : ndarray (n, n)
        Edge weights; W[i, j] = 0 where there is no edge.
    """
    n = X.shape[0]
    k = max(1, min(int(k), n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)      # +1: self is returned
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]                  # drop self

    if sigma is None:
        sigma = float(dist.mean()) or 1.0

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for jj, j in enumerate(idx[i]):
            w = math.exp(-(dist[i, jj] ** 2) / (2.0 * sigma * sigma))
            # symmetrise: an edge exists if EITHER endpoint selected the other
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)

    adj = [set(np.nonzero(W[i])[0].tolist()) for i in range(n)]
    return adj, W


def _connected_components(adj: list[set[int]], labels: NDArray[np.int_]
                          ) -> NDArray[np.int_]:
    """Component id per vertex AFTER removing inter-label edges.

    Removing edges between differently-labelled vertices means each resulting
    component contains a single label, so a component is a set of same-label
    vertices that are mutually reachable. This underlies the lenient,
    component-based definition of False Negatives used by recall.
    """
    n = len(adj)
    comp = np.full(n, -1, dtype=int)
    cid = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        stack, comp[s] = [s], cid
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1 and labels[v] == labels[u]:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    return comp


def precision_recall_fscore(adj: list[set[int]], W: NDArray[np.float64],
                            labels: Sequence[int], alpha: float = 0.0,
                            beta: float = 1.0) -> dict:
    """Per-vertex precision, recall and F-score on the similarity graph.

    Precision -- WEIGHTED by the edge strengths, because a weak edge represents
    a weaker similarity and should not count as heavily as a strong one::

        P(v_i) = sum_{TP} omega_ij / ( sum_{TP} omega_ij + sum_{FP} omega_ij )

    with TP_i = same-label neighbours and FP_i = different-label neighbours.
    P = 1 by convention for an isolated vertex. High precision means a point is
    surrounded by its own class -- the classes are locally pure.

    Recall -- by CARDINALITY, not weight, because a *missing* edge contributes
    no weight, so a weighted recall would report a high value even when many
    neighbours are absent::

        R_alpha(v_i) = |TP| / ( |TP| + alpha*|FN_adj| + (1-alpha)*|FN_comp| )

    interpolating between a strict, adjacency-based FN -- every same-label
    vertex must be a direct neighbour (a clique per class) -- and a lenient,
    component-based FN -- it suffices to be *reachable* through same-label
    vertices (one connected component per class). alpha = 0 asks only for
    connectedness; alpha = 1 demands a fully connected clique. High recall
    means each class is held together rather than fragmented.

    F-score: the F_beta combination of the two (beta = 1 gives the harmonic
    mean, F1), averaged first within each label and then across labels, which
    stops a majority class from dominating the global figure.
    """
    labels = np.asarray(labels)
    n = len(adj)
    comp = _connected_components(adj, labels)

    prec = np.ones(n)
    rec = np.ones(n)
    for i in range(n):
        same = {j for j in adj[i] if labels[j] == labels[i]}
        diff = adj[i] - same
        w_tp = float(sum(W[i, j] for j in same))
        w_fp = float(sum(W[i, j] for j in diff))
        if (len(same) + len(diff)) > 0:
            prec[i] = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 1.0

        # FN, strict: same label but NOT adjacent
        same_label = np.nonzero(labels == labels[i])[0]
        fn_adj = len(same_label) - 1 - len(same)
        # FN, lenient: same label but in a DIFFERENT component
        fn_comp = int(np.sum(comp[same_label] != comp[i])) - 0
        tp = len(same)
        denom = tp + (alpha * fn_adj + (1.0 - alpha) * fn_comp)
        rec[i] = (tp / denom) if denom > 0 else 1.0

    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (b2 + 1.0) * (prec * rec) / (b2 * prec + rec)        # F_beta
        f1 = 2.0 * (prec * rec) / (prec + rec)                   # always F1
    f = np.nan_to_num(f, nan=0.0)
    f1 = np.nan_to_num(f1, nan=0.0)

    # global scores: mean over labels of the mean within label (so a majority
    # class cannot dominate). Reported for precision, recall, F_beta and F1.
    per_label = {}
    for c in np.unique(labels):
        m = labels == c
        per_label[int(c)] = {
            "precision": float(prec[m].mean()),
            "recall": float(rec[m].mean()),
            "f_score": float(f[m].mean()),
            "f1": float(f1[m].mean()),
            "size": int(m.sum()),
        }
    return {
        "per_point": {"precision": prec, "recall": rec, "f_score": f, "f1": f1},
        "per_label": per_label,
        "global": {
            # MACRO: average within each class, then across classes, so a
            # majority class cannot dominate.
            "precision": float(np.mean([v["precision"] for v in per_label.values()])),
            "recall": float(np.mean([v["recall"] for v in per_label.values()])),
            "f_score": float(np.mean([v["f_score"] for v in per_label.values()])),
            "f1": float(np.mean([v["f1"] for v in per_label.values()])),
        },
        "global_micro": {
            # MICRO: pool all points equally (no per-class averaging), so a large
            # class contributes proportionally. The macro/micro gap reveals class
            # imbalance -- a large gap means the per-class scores differ widely.
            "precision": float(prec.mean()),
            "recall": float(rec.mean()),
            "f_score": float(f.mean()),
            "f1": float(f1.mean()),
        },
        "alpha": alpha, "beta": beta,
    }


# ==========================================================================
# 2. K-MEANS  (is the label set actually the cluster structure?)
# ==========================================================================
def kmeans_analysis(X_high: NDArray[np.float64], projections: dict,
                    labels: Sequence[int], n_clusters: Optional[int] = None,
                    k_range: tuple[int, int] = (2, 10),
                    cluster_method: str = "kmeans") -> dict:
    """Cluster the data and test the 'classes are clusters' assumption.

    A central caveat of any label-supervised metric is that class labels
    frequently do NOT correspond to the cluster structure a distance function
    actually sees. When that happens, every supervised score is measuring
    against the wrong reference, and a projection can be blamed for "hiding"
    clusters that were never there. The remedy is to cluster the high-D space
    and compare that clustering both to the labels and to the clustering found
    in each 2-D projection.

    ``cluster_method`` selects the clusterer:

    * ``kmeans`` (default) -- the natural choice for the Euclidean geometry PCA
      and MDS assume, but it can only find convex, roughly isotropic clusters.
    * ``gmm`` -- Gaussian mixture; finds elliptical clusters of differing
      covariance, more flexible than k-means.
    * ``spectral`` -- spectral clustering on a nearest-neighbour affinity; can
      recover **non-convex** clusters (e.g. concentric rings) that k-means and
      GMM cannot, removing the false "labels are not clusters" verdict on such
      data.

    We run the chosen clusterer in three spaces -- high-D, the PCA plane and the
    MDS plane -- which yields two distinct kinds of evidence:

    * **ARI / NMI (clustering vs class labels) in high-D** answers "are the
      classes clusters at all?". Low agreement means a low supervised score is
      a property of the DATA, not a failure of the projection.
    * **ARI (clustering in 2-D vs clustering in high-D)** answers "does this
      projection preserve the cluster structure?" -- computed per method.

    ``n_clusters`` defaults to the number of classes.
    """
    labels = np.asarray(labels)
    n_classes = int(len(np.unique(labels)))
    if n_clusters is None:
        n_clusters = n_classes

    def _cluster(Z, k):
        """Fit the chosen clusterer with k clusters and return labels."""
        Z = np.asarray(Z)
        if cluster_method == "gmm":
            gm = GaussianMixture(n_components=k, covariance_type="full",
                                 n_init=3, random_state=RANDOM_STATE)
            return gm.fit_predict(Z)
        if cluster_method == "spectral":
            # affinity from nearest neighbours so non-convex shapes are captured;
            # cap neighbours below the sample size
            k_nn = int(min(10, max(2, Z.shape[0] // (k + 1))))
            sc = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                    n_neighbors=k_nn, assign_labels="kmeans",
                                    random_state=RANDOM_STATE)
            return sc.fit_predict(Z)
        # default: k-means
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        return km.fit_predict(Z)

    # optional model selection by silhouette, reported for transparency
    sil_scan = []
    lo, hi = k_range
    for k in range(max(2, lo), min(hi, len(X_high) - 1) + 1):
        try:
            lab = _cluster(X_high, k)
        except Exception:
            continue
        if len(np.unique(lab)) > 1:
            entry = {"k": k,
                     "silhouette": float(silhouette_score(X_high, lab))}
            sil_scan.append(entry)
    best_k = max(sil_scan, key=lambda r: r["silhouette"])["k"] if sil_scan else n_clusters

    def _fit(Z):
        return _cluster(Z, n_clusters)

    cl_high = _fit(X_high)
    out_methods = {}
    for m, xy in projections.items():
        cl_2d = _fit(xy)
        out_methods[m] = {
            # does the 2-D picture reproduce the high-D partition?
            "ari_vs_high": float(adjusted_rand_score(cl_high, cl_2d)),
            "nmi_vs_high": float(normalized_mutual_info_score(cl_high, cl_2d)),
            # does the 2-D picture reproduce the LABELS?
            "ari_vs_labels": float(adjusted_rand_score(labels, cl_2d)),
            "silhouette_2d": (float(silhouette_score(xy, labels))
                              if n_classes > 1 else 0.0),
            "assignments": cl_2d.tolist(),
        }

    # ---- match high-D clusters to labels for an interpretable accuracy ----
    # ARI is rotation/label-invariant but not intuitive; a matched accuracy
    # ("how many points would be right if we named each cluster by its majority
    # class?") is what a reader expects. Match clusters to classes optimally.
    match = _match_clusters_to_labels(labels, cl_high, n_classes, n_clusters)
    matched = np.array([match.get(int(c), -1) for c in cl_high])
    cluster_accuracy = float(np.mean(matched == labels))
    # contingency table (high-D cluster rows x class-label columns) for a
    # confusion-matrix heatmap in the UI
    uniq_labels = list(np.unique(labels))
    lab_index = {l: j for j, l in enumerate(uniq_labels)}
    contingency = np.zeros((n_clusters, n_classes), dtype=int)
    for cl, la in zip(cl_high, labels):
        contingency[int(cl), lab_index[int(la)]] += 1
    # per-class recall under that matching (what fraction of each class is
    # captured by the cluster assigned to it)
    per_class_capture = {}
    for c in np.unique(labels):
        m = labels == c
        per_class_capture[int(c)] = float(np.mean(matched[m] == c))

    return {
        "n_clusters": int(n_clusters),
        "n_classes": n_classes,
        "cluster_method": cluster_method,
        "best_k_by_silhouette": int(best_k),
        "contingency": contingency.tolist(),
        "matched_labels": [int(match.get(int(r), -1)) for r in range(n_clusters)],
        "silhouette_scan": sil_scan,
        "high_d": {
            "ari_vs_labels": float(adjusted_rand_score(labels, cl_high)),
            "nmi_vs_labels": float(normalized_mutual_info_score(labels, cl_high)),
            "silhouette": (float(silhouette_score(X_high, cl_high))
                           if n_clusters > 1 else 0.0),
            "cluster_accuracy": cluster_accuracy,
            "per_class_capture": per_class_capture,
            "assignments": cl_high.tolist(),
        },
        "methods": out_methods,
        "classes_are_clusters": bool(
            adjusted_rand_score(labels, cl_high) >= 0.5),
    }


def _match_clusters_to_labels(labels: NDArray[np.int_], clusters: NDArray[np.int_],
                              n_classes: int, n_clusters: int) -> dict:
    """Optimally map each cluster id to a class label (maximising agreement).

    Builds the cluster x class contingency table and solves the assignment
    problem so that the total number of correctly-labelled points is maximised.
    Falls back to majority vote if SciPy's linear_sum_assignment is absent.
    """
    labels = np.asarray(labels)
    cont = np.zeros((n_clusters, n_classes), dtype=int)
    uniq_labels = list(np.unique(labels))
    lab_index = {l: j for j, l in enumerate(uniq_labels)}
    for cl, la in zip(clusters, labels):
        cont[int(cl), lab_index[int(la)]] += 1
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-cont)          # maximise agreement
        return {int(r): int(uniq_labels[c]) for r, c in zip(rows, cols)}
    except Exception:
        # majority vote per cluster (may map two clusters to one class)
        return {int(r): int(uniq_labels[int(np.argmax(cont[r]))])
                for r in range(n_clusters)}


# ==========================================================================
# 3. SIMULATED ANNEALING  (search the neighbourhood parameter k)
# ==========================================================================
def simulated_annealing_k(objective, k_lo: int, k_hi: int,
                          n_iter: int = 60, t0: Optional[float] = None,
                          t_min_frac: float = 1e-3, seed: int = RANDOM_STATE,
                          n_probe: int = 5) -> dict:
    """Maximise a scalar objective over an integer neighbourhood parameter k.

    The F_beta-score as a function of k is *neither concave nor convex*:
    recall increases monotonically with k while precision tends to fall but not
    monotonically, so their combination can have several local maxima. Gradient
    methods and simple ternary search are therefore unsafe, which motivates a
    metaheuristic; simulated annealing is a lightweight, dependency-free choice
    well suited to a non-convex 1-D objective.

    We implement simulated annealing because it is the lightest-weight option
    with no extra dependency, it copes with the non-convexity by accepting
    worsening moves early on, and its behaviour is easy to audit in a report:
    every evaluation is recorded in the returned trace.

    Acceptance is the standard Metropolis rule for a maximisation problem::

        accept if  delta >= 0,  else with probability exp(delta / T)

    with a geometric cooling schedule T <- T * gamma.

    TEMPERATURE CALIBRATION (important). The Metropolis rule is *not* scale
    invariant: exp(delta / T) depends on the magnitude of the objective. A
    fixed T0 = 1 accepts virtually every uphill move when the objective lives
    in [0, 1] (as an f-score does), and virtually none when the objective
    spans tens. Either way the search degenerates -- into a random walk or
    into a plain hill climber that cannot escape a local optimum, which
    defeats the reason for choosing SA. We therefore probe a few k values
    first and set T0 to the standard deviation of those samples (falling back
    to their range), so that a "typical" worsening move is accepted with
    probability about e^-1 at the start and almost never at the end. Pass an
    explicit ``t0`` to override.

    ``objective`` is any callable k -> float (cached, so repeated k is free).
    Returns the best k, its score, the calibrated schedule and the full trace.
    """
    rng = np.random.default_rng(seed)
    k_lo, k_hi = int(k_lo), int(k_hi)
    if k_hi <= k_lo:
        val = float(objective(k_lo))
        return {"best_k": k_lo, "best_score": val, "trace": [],
                "n_evaluations": 1, "t0": 0.0, "cache": {k_lo: val}}

    cache: dict[int, float] = {}

    def f(k: int) -> float:
        k = int(np.clip(k, k_lo, k_hi))
        if k not in cache:
            cache[k] = float(objective(k))
        return cache[k]

    # --- probe to calibrate the temperature to the objective's own scale ---
    # Probe densely so a few unlucky samples cannot mislead the calibration
    # (a low-spread probe set was the root cause of premature convergence: it
    # produced a tiny T0 and the search froze in a local corner).
    n_probe = max(n_probe, min(9, k_hi - k_lo + 1))
    probes = np.unique(np.linspace(k_lo, k_hi, n_probe).round().astype(int))
    pv = [f(int(k)) for k in probes]
    if t0 is None:
        # base T0 on the full observed RANGE of the probes (not just the std),
        # and floor it, so a flat-looking probe set still permits exploration
        rng_obs = float(np.ptp(pv))
        std_obs = float(np.std(pv))
        t0 = max(std_obs, 0.5 * rng_obs)
        if t0 <= 0:
            t0 = 1.0
    t_min = max(t0 * t_min_frac, 1e-9)

    gamma = (t_min / t0) ** (1.0 / max(1, n_iter - 1))
    # initial proposal reach spans a large fraction of the WHOLE range, so early
    # moves can traverse from one end to the other (a quarter-range cap could
    # never reach a far optimum)
    span = max(1, (k_hi - k_lo) // 2)

    cur = int(probes[int(np.argmax(pv))])       # start from the best probe
    cur_v = f(cur)
    best, best_v = cur, cur_v
    T = t0
    trace, uphill = [], 0
    stall = 0                                   # iterations since best improved
    reheats = 0
    for it in range(n_iter):
        # proposal width shrinks with temperature: broad early, fine late
        width = max(1, int(round(span * (T / t0))))
        cand = int(np.clip(cur + rng.integers(-width, width + 1), k_lo, k_hi))
        cand_v = f(cand)
        delta = cand_v - cur_v
        if delta >= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(delta / max(T, 1e-12))
            if accept:
                uphill += 1                     # a deliberately worsening move
        if accept:
            cur, cur_v = cand, cand_v
            if cur_v > best_v:
                best, best_v = cur, cur_v
                stall = 0
            else:
                stall += 1
        else:
            stall += 1
        # --- REHEAT on stall: if the best has not improved for a while, warm
        # the temperature back up and jump to the best, so the search takes a
        # fresh exploratory pass instead of dying in a local optimum ---
        if stall >= max(6, n_iter // 6) and it < n_iter - 2:
            T = max(T, 0.5 * t0)
            cur, cur_v = best, best_v
            stall = 0
            reheats += 1
        trace.append({"iter": it, "T": round(T, 6), "k": cand,
                      "score": round(cand_v, 6), "accepted": bool(accept),
                      "uphill": bool(accept and delta < 0),
                      "best_k": best, "best_score": round(best_v, 6)})
        T *= gamma

    return {"best_k": int(best), "best_score": float(best_v), "trace": trace,
            "n_evaluations": len(cache), "t0": float(t0),
            "uphill_moves_accepted": int(uphill), "reheats": int(reheats),
            "cache": {int(k): float(v) for k, v in sorted(cache.items())}}


def optimise_relationship_k(X_high: NDArray[np.float64], labels: Sequence[int],
                            k_lo: int = 2, k_hi: int = 60,
                            alpha: float = 0.0, beta: float = 1.0,
                            n_iter: int = 45) -> dict:
    """Search for the neighbourhood parameter k maximising the global F-score.

    Note what this does NOT require: no PCA or MDS embedding is ever computed
    inside the loop. We are asking only whether the high-D similarity structure
    could support the expected clusters, so each objective evaluation rebuilds
    just the k-NN graph -- which is exactly why the search is affordable.
    """
    def obj(k: int) -> float:
        adj, W = build_knn_graph(X_high, k=int(k))
        return precision_recall_fscore(adj, W, labels, alpha=alpha,
                                       beta=beta)["global"]["f1"]

    res = simulated_annealing_k(obj, k_lo, k_hi, n_iter=n_iter)
    adj, W = build_knn_graph(X_high, k=res["best_k"])
    res["metrics_at_best"] = precision_recall_fscore(
        adj, W, labels, alpha=alpha, beta=beta)["global"]
    res["alpha"], res["beta"] = alpha, beta
    # true size of an exhaustive scan over the SAME range, so the UI reports
    # "N / exhaustive_count" honestly instead of a hard-coded constant
    res["k_lo"], res["k_hi"] = int(k_lo), int(k_hi)
    res["exhaustive_count"] = int(k_hi - k_lo + 1)
    return res


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
      2. ``trust``      -- classical trustworthiness (few false neighbours).
      3. ``continuity`` -- classical continuity (few lost neighbours).
      4. ``cluster``    -- ARI between k-means in the plane and k-means in
                           high-D: does the picture preserve the partition?
                           Omitted (weights renormalised) if unavailable.

    The weights are stated explicitly rather than tuned, so the verdict is
    reproducible and auditable. A paired comparison over the per-point
    distortion supplies significance: we report Cohen's d and a paired
    bootstrap confidence interval on the mean difference, so "better" is
    accompanied by "by how much, and is it reliable?".
    """
    labels = np.asarray(labels)
    names = list(projections)
    D_high = pairwise_distances(X_high)

    WEIGHTS = {"fidelity": 0.40, "trust": 0.20, "continuity": 0.20, "cluster": 0.20}

    per_method, per_point_dist = {}, {}
    for m in names:
        D_low = pairwise_distances(projections[m])
        tw, co = _trustworthiness_continuity(D_high, D_low, k)
        dist = np.abs(fps[m]) + np.abs(fns[m])
        per_point_dist[m] = dist
        comp = {
            "fidelity": float(np.clip(1.0 - dist.mean(), 0.0, 1.0)),
            "trust": float(np.clip(tw, 0.0, 1.0)),
            "continuity": float(np.clip(co, 0.0, 1.0)),
        }
        if kmeans and m in kmeans.get("methods", {}):
            comp["cluster"] = float(np.clip(
                kmeans["methods"][m]["ari_vs_high"], 0.0, 1.0))
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
    }