"""
Flask backend for the PCA/MDS error-analysis toolkit.

Wraps error_core directly (it is matplotlib-free), captures all console output
for the UI's console panel, and returns raw arrays so the browser can build the
Plotly figures itself (small payloads + native brushing). Runs are persisted to
disk as JSON so results survive a restart.
"""

from __future__ import annotations

import io
import json
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from error_core import (SKLEARN_LOADERS, DATASET_NAMES, load_dataset, maybe_subsample,
                        build_projections, compute_errors,
                        compute_feature_weights)
from utils import describe, build_fname_tag, nice_vmax, normalize_errors
from analysis import (build_knn_graph, precision_recall_fscore, kmeans_analysis,
                      optimise_relationship_k, compare_methods)

APP_DIR = Path(__file__).parent
STORE = APP_DIR / "runs_store.json"          
UPLOAD_DIR = APP_DIR / "user_csvs"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(APP_DIR / "static"), static_url_path="")


# persistence helpers
def _load_store() -> dict[str, Any]:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text())
        except json.JSONDecodeError:
            return {"runs": []}
    return {"runs": []}


def _save_store(data: dict[str, Any]) -> None:
    STORE.write_text(json.dumps(data))


# core analysis
def _run_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    """Execute the static pipeline, capturing stdout. Returns a JSON-ready dict."""
    buf = io.StringIO()
    t0 = time.time()

    dataset = params.get("dataset", "wine")
    csv_path = params.get("csv_path") or None
    label_col = params.get("label_col") or None
    metric = params.get("metric", "intrusion")
    neighbors = int(params.get("neighbors", 10))
    # Fidelity metric is GEOMETRIC (label-free) by default. Class-awareness is
    # an explicit opt-in: it restricts FP to cross-class intruders and FN to
    # same-class extrusions, i.e. it measures class-structure fidelity instead
    # of pure neighbourhood fidelity. Accept the new 'class_aware' flag; fall
    # back to the legacy 'geometric' flag for older payloads.
    if "class_aware" in params:
        class_aware = bool(params.get("class_aware"))
    else:
        class_aware = not bool(params.get("geometric", False)) if "geometric" in params else False
    geometric = not class_aware
    weight = params.get("weight", "rank")
    adaptive = bool(params.get("adaptive", True))
    normalize = params.get("normalize", "none")
    norm_scale = float(params.get("norm_scale", 3.0))
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 1.0))
    max_samples = params.get("max_samples")
    max_samples = int(max_samples) if max_samples else None
    standardize = not bool(params.get("no_standardize", False))
    threshold = float(params.get("threshold", 0.5))
    weighted_mds = bool(params.get("weighted_mds", False))
    weight_method = params.get("weight_method", "mahalanobis")
    use_maha = weighted_mds and (weight_method == "mahalanobis")
    custom_weights = params.get("custom_weights") or None
    weight_info = None
    # analysis layer (paper-grounded) toggles
    do_kmeans = bool(params.get("kmeans", True))
    do_relmetrics = bool(params.get("rel_metrics", True))
    do_anneal = bool(params.get("anneal", False))
    rel_k = int(params.get("rel_k", 15))
    rel_alpha = float(params.get("rel_alpha", 0.5))
    rel_beta = float(params.get("rel_beta", 1.0))

    with redirect_stdout(buf):
        X, y, target_names, name = load_dataset(dataset, csv_path, label_col)
        describe(X, y, target_names, name)
        X, y, ids = maybe_subsample(X, y, max_samples)

        feature_weights = None
        if weighted_mds and not use_maha:
            try:
                # weights are computed on the (optionally standardized) array so
                # they match the space MDS actually embeds
                from sklearn.preprocessing import StandardScaler as _SS
                X_for_w = _SS().fit_transform(X) if standardize else np.asarray(X, float)
                feature_weights = compute_feature_weights(
                    X_for_w, labels=y, method=weight_method,
                    custom=custom_weights)
                weight_info = {
                    "method": weight_method,
                    "weights": [round(float(w), 4) for w in feature_weights],
                    "min": round(float(np.min(feature_weights)), 4),
                    "max": round(float(np.max(feature_weights)), 4),
                }
                print(f"\n[weighted MDS] method={weight_method}  "
                      f"weights min={feature_weights.min():.3f} "
                      f"max={feature_weights.max():.3f} (mean 1.0)")
            except Exception as we:
                print(f"[weighted MDS] could not compute weights ({we}); "
                      "falling back to plain Euclidean MDS")
                feature_weights = None
                weighted_mds = False

        proj, X_high = build_projections(
            X, standardize=standardize, weighted_mds=weighted_mds,
            feature_weights=feature_weights, mahalanobis=use_maha)
        timings = dict(getattr(build_projections, "last_timings", {"pca_ms": 0, "mds_ms": 0}))

        # When weighted MDS is active, also compute the PLAIN MDS layout so the
        # UI can show both side by side, and quantify how much class separation
        # changed (silhouette of the labels in each 2-D layout).
        mds_compare = None
        if weighted_mds and (feature_weights is not None or use_maha):
            try:
                plain_proj, _ = build_projections(
                    X, standardize=standardize, weighted_mds=False)
                from sklearn.metrics import silhouette_score as _sil
                n_lab = len(np.unique(y))
                sil_plain = float(_sil(plain_proj["MDS"], y)) if n_lab > 1 else 0.0
                sil_weight = float(_sil(proj["MDS"], y)) if n_lab > 1 else 0.0
                # neighbourhood-preservation F1: does the ADJUSTED (weighted)
                # distance actually improve projection fidelity -- i.e. reduce
                # false positives (intrusions) and false negatives (extrusions)?
                from error_core import per_point_neighbor_metrics as _ppnm2
                f1_plain = _ppnm2(X_high, plain_proj["MDS"], k=neighbors)["mean"]
                f1_weight = _ppnm2(X_high, proj["MDS"], k=neighbors)["mean"]
                mds_compare = {
                    "method": "Mahalanobis" if use_maha else weight_method,
                    "plain": {"x": plain_proj["MDS"][:, 0].tolist(),
                              "y": plain_proj["MDS"][:, 1].tolist()},
                    "silhouette_plain": round(sil_plain, 4),
                    "silhouette_weighted": round(sil_weight, 4),
                    "delta": round(sil_weight - sil_plain, 4),
                    "f1_plain": round(f1_plain["f1"], 4),
                    "f1_weighted": round(f1_weight["f1"], 4),
                    "f1_delta": round(f1_weight["f1"] - f1_plain["f1"], 4),
                    "fp_plain": round(1 - f1_plain["precision"], 4),
                    "fp_weighted": round(1 - f1_weight["precision"], 4),
                    "fn_plain": round(1 - f1_plain["recall"], 4),
                    "fn_weighted": round(1 - f1_weight["recall"], 4),
                }
                print(f"[weighted MDS] class silhouette: plain={sil_plain:.3f} "
                      f"-> weighted={sil_weight:.3f} (delta {sil_weight-sil_plain:+.3f})")
            except Exception as ce:
                print(f"[weighted MDS] could not build comparison ({ce})")
                mds_compare = None

        errs, fps, fns = compute_errors(
            proj, X_high, y, metric=metric, k=neighbors,
            class_aware=class_aware, weight=weight, adaptive=adaptive,
            alpha=alpha, beta=beta, return_parts=True)

        # measure how much the geometric and class-aware error modes differ on
        # this data, so the UI can report it as a concrete number. Compute the
        # OTHER mode's per-point error and compare.
        class_aware_delta = None
        try:
            other_errs, _, _ = compute_errors(
                proj, X_high, y, metric=metric, k=neighbors,
                class_aware=(not class_aware), weight=weight, adaptive=adaptive,
                alpha=alpha, beta=beta, return_parts=True)
            # aggregate across both methods
            import numpy as _np
            diffs, signflips, n_tot = [], 0, 0
            for m in ("PCA", "MDS"):
                a = _np.asarray(errs[m]); b = _np.asarray(other_errs[m])
                diffs.append(_np.abs(a - b))
                signflips += int(_np.sum(_np.sign(a) != _np.sign(b)))
                n_tot += a.size
            diffs = _np.concatenate(diffs)
            corr = float(_np.corrcoef(
                _np.concatenate([_np.asarray(errs[m]) for m in ("PCA","MDS")]),
                _np.concatenate([_np.asarray(other_errs[m]) for m in ("PCA","MDS")]))[0,1])
            class_aware_delta = {
                "active_mode": "class-aware" if class_aware else "geometric",
                "mean_abs_diff": round(float(_np.mean(diffs)), 4),
                "sign_flip_pct": round(100.0 * signflips / max(1, n_tot), 1),
                "correlation": round(corr, 3),
            }
        except Exception as cae:
            print(f"[class-aware delta] skipped ({cae})")

        # per-point k distribution (only meaningful when adaptive k is on)
        adaptive_k_dist = None
        _kv = None
        if adaptive:
            try:
                from error_core import adaptive_k as _adaptive_k
                from sklearn.metrics import pairwise_distances as _pdist
                kv = _adaptive_k(_pdist(X_high, metric="euclidean"), neighbors)
                kv = np.asarray(kv, dtype=int)
                _kv = kv
                # bin into the requested ranges
                bins = [(3, 5), (6, 8), (9, 12), (13, 100)]
                bin_labels = ["3-5", "6-8", "9-12", "13+"]
                counts = []
                for lo, hi in bins:
                    counts.append(int(np.sum((kv >= lo) & (kv <= hi))))
                total = max(1, len(kv))
                adaptive_k_dist = {
                    "bin_labels": bin_labels,
                    "counts": counts,
                    "pct": [round(100.0 * c / total, 1) for c in counts],
                    "k_min": int(kv.min()), "k_max": int(kv.max()),
                    "k_median": int(np.median(kv)), "k_base": int(neighbors),
                    "n": total,
                }
            except Exception as ke:
                print(f"[adaptive k] distribution skipped ({ke})")

        # per-point neighbourhood-preservation metrics (F1 + Jensen-Shannon),
        # computed for BOTH projections. This is the "each point is a classifier
        # of its own neighbourhood" view the projection-quality analysis uses.
        neighbor_metrics = None
        try:
            from error_core import per_point_neighbor_metrics as _ppnm
            npm_pca = _ppnm(X_high, proj["PCA"], k=neighbors, k_vec=_kv)
            npm_mds = _ppnm(X_high, proj["MDS"], k=neighbors, k_vec=_kv)
            neighbor_metrics = {
                "k": int(neighbors), "adaptive": bool(adaptive),
                "PCA": npm_pca, "MDS": npm_mds,
            }
        except Exception as nme:
            print(f"[neighbour metrics] skipped ({nme})")

        if normalize != "none":
            errs = {m: normalize_errors(e, normalize, norm_scale)
                    for m, e in errs.items()}

        pca_err, mds_err = errs["PCA"], errs["MDS"]
        print("\nFirst 5 PCA errors:", np.round(pca_err[:5], 3))
        print("First 5 MDS errors:", np.round(mds_err[:5], 3))

        # colour scale
        if metric == "classic" and normalize == "none":
            vmax = 1.0
        else:
            vmax = nice_vmax(pca_err, mds_err)

        # ---- analysis layer: k-means, relationship metrics, SA, verdict ----
        kmeans_res = rel_res = anneal_res = verdict_res = None
        try:
            if do_kmeans:
                cluster_method = params.get("cluster_method", "kmeans")
                if cluster_method not in ("kmeans", "gmm", "spectral"):
                    cluster_method = "kmeans"
                kmeans_res = kmeans_analysis(X_high, proj, y,
                                             cluster_method=cluster_method)
                print(f"\n[k-means] ARI(high-D clusters vs labels) = "
                      f"{kmeans_res['high_d']['ari_vs_labels']:.3f} -> "
                      f"classes {'ARE' if kmeans_res['classes_are_clusters'] else 'are NOT'} clusters")
                for _m, _v in kmeans_res["methods"].items():
                    print(f"          {_m}: ARI(2-D vs high-D) = {_v['ari_vs_high']:.3f}")
        except Exception as ke:
            print(f"[k-means] skipped ({ke})")
        try:
            if do_relmetrics:
                _adj, _W = build_knn_graph(X_high, k=rel_k)
                _pr = precision_recall_fscore(_adj, _W, y, alpha=rel_alpha, beta=rel_beta)
                rel_res = {"global": _pr["global"], "global_micro": _pr["global_micro"],
                           "per_label": _pr["per_label"],
                           "k": rel_k, "alpha": rel_alpha, "beta": rel_beta,
                           "target_names": target_names}
                g = _pr["global"]
                print(f"\n[cluster-structure metrics] k={rel_k} alpha={rel_alpha} -> "
                      f"precision {g['precision']:.3f}  recall {g['recall']:.3f}  "
                      f"F1 {g['f1']:.3f}")
        except Exception as re_:
            print(f"[cluster-structure metrics] skipped ({re_})")
        try:
            if do_anneal:
                anneal_res = optimise_relationship_k(
                    X_high, y, k_lo=2, k_hi=min(60, max(3, len(y) // 3)),
                    alpha=rel_alpha, beta=rel_beta, n_iter=30)
                print(f"\n[simulated annealing] best k={anneal_res['best_k']} "
                      f"f-score={anneal_res['best_score']:.4f} "
                      f"({anneal_res['n_evaluations']} evaluations, "
                      f"T0={anneal_res['t0']:.4f})")
        except Exception as ae:
            print(f"[simulated annealing] skipped ({ae})")
        try:
            verdict_res = compare_methods(X_high, proj, errs, fps, fns, y,
                                          k=neighbors, kmeans=kmeans_res)
            print(f"\n[verdict] {verdict_res['verdict']}")
            # --- MDS variability: the bootstrap CI captures per-point variation
            # for ONE pair of layouts, not MDS's run-to-run stochasticity. Re-run
            # MDS a few times with different seeds, recompute the MDS fidelity
            # component each time, and report the spread so "decisive" can be read
            # in the light of how much the layout itself wobbles. ---
            try:
                from sklearn.manifold import MDS as _MDS
                from sklearn.metrics import pairwise_distances as _pd
                base_fid = 1.0 - float(np.mean(np.abs(fps["MDS"]) + np.abs(fns["MDS"])))
                Xs_for_mds = X_high
                D_high = _pd(Xs_for_mds)
                fids = [base_fid]
                for seed in range(1, 4):          # 3 extra layouts
                    m = _MDS(n_components=2, dissimilarity="precomputed",
                             random_state=seed, n_init=1, normalized_stress="auto")
                    xy = m.fit_transform(D_high)
                    _e, _fp, _fn = compute_errors(
                        {"MDS": xy}, X_high, y, metric=metric, k=neighbors,
                        class_aware=class_aware, weight=weight,
                        adaptive=adaptive, return_parts=True)
                    fids.append(1.0 - float(np.mean(np.abs(_fp["MDS"]) + np.abs(_fn["MDS"]))))
                verdict_res["mds_stability"] = {
                    "fidelity_mean": round(float(np.mean(fids)), 4),
                    "fidelity_std": round(float(np.std(fids)), 4),
                    "fidelity_min": round(float(np.min(fids)), 4),
                    "fidelity_max": round(float(np.max(fids)), 4),
                    "n_layouts": len(fids),
                    "margin": verdict_res.get("margin_absolute"),
                }
                _sd = verdict_res["mds_stability"]["fidelity_std"]
                print(f"[verdict] MDS fidelity across {len(fids)} layouts: "
                      f"{np.mean(fids):.3f} +/- {_sd:.3f}")
            except Exception as se:
                print(f"[verdict] MDS stability skipped ({se})")
        except Exception as ve:
            print(f"[verdict] skipped ({ve})")

        # untrustworthy report — NOW based on the per-point neighbourhood F1
        # (a low-F1 point cannot be trusted: its neighbourhood in the picture is
        # wrong), which is far more reliable than the old FP/FN threshold.
        report_rows = []
        untrust_desc = "low F1"
        try:
            if neighbor_metrics is not None:
                import numpy as _np2
                f1_pca = _np2.asarray(neighbor_metrics["PCA"]["f1"])
                f1_mds = _np2.asarray(neighbor_metrics["MDS"]["f1"])
                js_pca = _np2.asarray(neighbor_metrics["PCA"]["js_fidelity"])
                js_mds = _np2.asarray(neighbor_metrics["MDS"]["js_fidelity"])
                # worst F1 across the two methods per point
                worst = _np2.minimum(f1_pca, f1_mds)
                # flag the bottom points: F1 below a data-driven cut
                # (25th percentile OR < 0.5, whichever flags fewer, capped)
                cut = min(0.5, float(_np2.percentile(worst, 25)))
                sel = _np2.where(worst < cut)[0]
                sel = sel[_np2.argsort(worst[sel])]      # worst first
                untrust_desc = f"F1 < {cut:.2f} (worst of PCA/MDS)"
                for i in sel:
                    report_rows.append({
                        "index": int(ids[i]),
                        "true_class": target_names[y[i]] if y[i] < len(target_names) else str(y[i]),
                        "pca_f1": round(float(f1_pca[i]), 3),
                        "mds_f1": round(float(f1_mds[i]), 3),
                        "pca_js": round(float(js_pca[i]), 3),
                        "mds_js": round(float(js_mds[i]), 3),
                        "worst_f1": round(float(worst[i]), 3),
                    })
        except Exception as ue:
            print(f"[untrustworthy] F1-based flag skipped ({ue})")

    elapsed = time.time() - t0
    console = buf.getvalue() + f"\n[done] pipeline finished in {elapsed:.2f}s"

    classes, counts = np.unique(y, return_counts=True)
    # --- k/n degeneracy guard ---
    # The rank normaliser W = k(2n-3k-1)/2 requires k < (2n-1)/3; well before
    # that hard limit, a k that is a large fraction of n makes almost every 2-D
    # neighbourhood pull in unrelated points (heavy FP) while extrusion is
    # suppressed, so the signed-error balance degenerates toward 100% FP and the
    # colour scale loses contrast. Warn so the result is not silently misread.
    n_now = int(X.shape[0])
    ratio = neighbors / max(1, n_now)
    hard_limit = (2 * n_now - 1) / 3.0
    warnings = []
    if neighbors >= hard_limit:
        warnings.append({"level": "error", "code": "k_over_limit",
            "message": (f"k = {neighbors} is at/above the metric's limit for "
                        f"n = {n_now} (needs k < {hard_limit:.0f}). Results are "
                        f"degenerate; reduce k or raise the sample size.")})
    elif ratio >= 0.15:
        warnings.append({"level": "warning", "code": "k_over_n",
            "message": (f"k = {neighbors} is {ratio*100:.0f}% of n = {n_now}. Above "
                        f"~15% the intrusion/extrusion balance skews heavily toward "
                        f"false positives and the colour scale loses contrast. "
                        f"Consider k \u2264 {int(0.15*n_now)} or a larger sample.")})

    info = {
        "name": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(classes)),
        "class_counts": [
            {"label": target_names[c] if c < len(target_names) else str(c),
             "code": int(c), "count": int(n)}
            for c, n in zip(classes, counts)],
        "standardized": standardize,
        "elapsed": round(elapsed, 3),
        "k_over_n_ratio": round(ratio, 3),
        "warnings": warnings,
    }

    fname = build_fname_tag(name, metric, neighbors, geometric,
                            adaptive=adaptive, normalize=normalize)

    # full per-point table (index, class, coords, err, fp, fn for both methods)
    labels_list = y.tolist()
    # feature matrix (standardized high-D values) + names, so the UI can filter
    # points by attribute windows. Round to keep the payload compact.
    _feat = np.round(np.asarray(X_high, dtype=float), 4)
    _p = _feat.shape[1]
    # keep feature payload bounded: cap at 30 columns for very wide data
    _fcols = list(range(min(_p, 30)))
    feature_names = [f"f{j}" for j in _fcols]
    feature_values = _feat[:, _fcols].tolist()

    table = {
        "ids": ids.tolist(),
        "labels": labels_list,
        "class_names": [target_names[c] if c < len(target_names) else str(c)
                        for c in labels_list],
        "feature_names": feature_names,
        "feature_values": feature_values,
        "pca": {"x": proj["PCA"][:, 0].tolist(), "y": proj["PCA"][:, 1].tolist(),
                "err": errs["PCA"].tolist(), "fp": fps["PCA"].tolist(),
                "fn": fns["PCA"].tolist()},
        "mds": {"x": proj["MDS"][:, 0].tolist(), "y": proj["MDS"][:, 1].tolist(),
                "err": errs["MDS"].tolist(), "fp": fps["MDS"].tolist(),
                "fn": fns["MDS"].tolist()},
    }

    summary = {
        "pca_mean": float(errs["PCA"].mean()),
        "mds_mean": float(errs["MDS"].mean()),
        "pca_fn_share": float(np.mean(errs["PCA"] < 0)),
        "mds_fn_share": float(np.mean(errs["MDS"] < 0)),
        "untrustworthy_count": len(report_rows),
    }

    # class-centroid trajectories (high-D -> PCA -> MDS, comparable frames)
    centroids = _centroid_trajectories(
        X_high, proj["PCA"], proj["MDS"], y, target_names, classes)

    return {
        "ok": True,
        "params": params,
        "info": info,
        "target_names": target_names,
        "vmax": vmax,
        "fname": fname,
        "table": table,
        "untrustworthy": report_rows,
        "summary": summary,
        "centroids": centroids,
        "weight_info": weight_info,
        "mds_compare": mds_compare,
        "adaptive_k_dist": adaptive_k_dist,
        "class_aware_delta": class_aware_delta,
        "neighbor_metrics": neighbor_metrics,
        "timings": timings,
        "untrust_desc": untrust_desc,
        "kmeans": kmeans_res,
        "rel_metrics": rel_res,
        "anneal": anneal_res,
        "verdict": verdict_res,
        "console": console,
        "metric": metric,
        "timestamp": time.time(),
    }


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


# routes
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/datasets")
def datasets():
    return jsonify({"datasets": DATASET_NAMES})


@app.route("/api/run", methods=["POST"])
def run():
    params = request.get_json(force=True)
    try:
        result = _run_pipeline(params)
    except Exception as exc:  # surface errors to the console panel
        tb = traceback.format_exc()
        return jsonify({
            "ok": False,
            "error": str(exc),
            "console": f"[error] {exc}\n\n{tb}",
        }), 200

    # keep last 20 runs
    store = _load_store()
    store["runs"].insert(0, result)
    store["runs"] = store["runs"][:20]
    store["last"] = result
    _save_store(store)
    return jsonify(result)


@app.route("/api/last")
def last():
    store = _load_store()
    return jsonify(store.get("last") or {"ok": False, "empty": True})


@app.route("/api/history")
def history():
    store = _load_store()
    runs = store.get("runs", [])
    # lightweight index only
    idx = [{
        "timestamp": r.get("timestamp"),
        "name": r.get("info", {}).get("name"),
        "metric": r.get("metric"),
        "params": r.get("params"),
        "summary": r.get("summary"),
    } for r in runs]
    return jsonify({"runs": idx})


@app.route("/api/clear", methods=["POST"])
def clear():
    _save_store({"runs": []})
    return jsonify({"ok": True})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no file"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"ok": False, "error": "must be a .csv"}), 400
    dest = UPLOAD_DIR / f.filename
    f.save(dest)
    # report columns so the UI can offer a label-col picker
    import pandas as pd
    cols = list(pd.read_csv(dest, nrows=1).columns)
    return jsonify({"ok": True, "path": str(dest), "columns": cols})


if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=_port, debug=False, threaded=True)