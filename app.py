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

from error_core import (SKLEARN_LOADERS, load_dataset, maybe_subsample,
                        build_projections, compute_errors,
                        compute_feature_weights)
from utils import describe, build_fname_tag, nice_vmax, normalize_errors

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
    geometric = bool(params.get("geometric", False))
    weight = params.get("weight", "rank")
    adaptive = bool(params.get("adaptive", False))
    normalize = params.get("normalize", "none")
    norm_scale = float(params.get("norm_scale", 3.0))
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 1.0))
    max_samples = params.get("max_samples")
    max_samples = int(max_samples) if max_samples else None
    standardize = not bool(params.get("no_standardize", False))
    threshold = float(params.get("threshold", 0.5))
    weighted_mds = bool(params.get("weighted_mds", False))
    weight_method = params.get("weight_method", "mutual_info")
    custom_weights = params.get("custom_weights") or None
    weight_info = None

    with redirect_stdout(buf):
        X, y, target_names, name = load_dataset(dataset, csv_path, label_col)
        describe(X, y, target_names, name)
        X, y, ids = maybe_subsample(X, y, max_samples)

        feature_weights = None
        if weighted_mds:
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
            feature_weights=feature_weights)
        class_aware = not geometric

        errs, fps, fns = compute_errors(
            proj, X_high, y, metric=metric, k=neighbors,
            class_aware=class_aware, weight=weight, adaptive=adaptive,
            alpha=alpha, beta=beta, return_parts=True)

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

        # untrustworthy report 
        report_rows = _report_untrustworthy(
            pca_err, mds_err, y, target_names, ids, threshold,
            fps["PCA"], fns["PCA"], fps["MDS"], fns["MDS"])

    elapsed = time.time() - t0
    console = buf.getvalue() + f"\n[done] pipeline finished in {elapsed:.2f}s"

    classes, counts = np.unique(y, return_counts=True)
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
    }

    fname = build_fname_tag(name, metric, neighbors, geometric,
                            adaptive=adaptive, normalize=normalize)

    # full per-point table (index, class, coords, err, fp, fn for both methods)
    labels_list = y.tolist()
    table = {
        "ids": ids.tolist(),
        "labels": labels_list,
        "class_names": [target_names[c] if c < len(target_names) else str(c)
                        for c in labels_list],
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
    mask = (np.abs(pca_err) > thresh) | (np.abs(mds_err) > thresh)
    sel = np.where(mask)[0]
    print("\n" + "=" * 60)
    print(f"UNTRUSTWORTHY POINTS  (|error| > {thresh} in PCA or MDS)")
    print("=" * 60)
    print(f"Count: {len(sel)}")
    rows = []
    for i in sel:
        rows.append({
            "index": int(ids[i]),
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
    return jsonify({"datasets": list(SKLEARN_LOADERS)})


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