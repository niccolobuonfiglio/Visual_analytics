"""
The analysis pipeline: one parameter dict in, one JSON-ready result dict out.

Extracted from the former ``app.py`` so that the Flask layer is only routing and
this module is only computation. Nothing here imports Flask, so the pipeline can
be driven from a script or a test without starting a server.

WHAT RUNS, AND WHAT NO LONGER DOES
----------------------------------
Everything below produces something the browser interface actually displays.
Six blocks that used to run on every request have been disabled because no
interface control reaches them; each is marked ``DISABLED`` at the point where
it used to execute, with a pointer to the quarantined implementation:

  * adjusted-distance MDS comparison ..... legacy/adjusted_distance.py
  * mutual-information feature weights ... legacy/adjusted_distance.py
  * geometric vs class-aware delta ....... (second full metric pass, removed)
  * k-means ground-truth check ........... legacy/label_structure.py
  * supervised graph precision/recall .... legacy/label_structure.py
  * simulated-annealing k search ......... legacy/label_structure.py
  * MDS layout stability (extra seeds) ... (three extra MDS fits, removed)
  * class-centroid trajectories .......... legacy/legacy_misc.py
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from typing import Any

import numpy as np

from datasets import load_dataset, maybe_subsample
from projections import build_projections
from metrics import adaptive_k, per_point_neighbor_metrics, compute_errors
from verdict import compare_methods
from utils import describe, build_fname_tag, nice_vmax, normalize_errors

# The three per-point metrics averaged into the consensus fidelity that drives
# the untrustworthy flag. Normalised stress is deliberately not included; see
# the note in _flag_untrustworthy.
CONSENSUS_KEYS = ["f1", "js_fidelity", "localstress"]


# ---------------------------------------------------------------------------
# parameter handling
# ---------------------------------------------------------------------------
def _read_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalise the raw request body into the values the pipeline uses.

    Kept separate so the long list of defaults does not clutter the run itself.
    """
    max_samples = params.get("max_samples")
    return {
        "dataset": params.get("dataset", "wine"),
        "csv_path": params.get("csv_path") or None,
        "label_col": params.get("label_col") or None,
        "neighbors": int(params.get("neighbors", 10)),
        "adaptive": bool(params.get("adaptive", True)),
        "max_samples": int(max_samples) if max_samples else None,
        "standardize": not bool(params.get("no_standardize", False)),
        # which two embeddings to compare; either slot may be any of PROJECTORS
        "method_a": params.get("method_a") or "PCA",
        "method_b": params.get("method_b") or "MDS",
        # The fidelity metric is GEOMETRIC: it counts any broken neighbour, with
        # no assumption that the class labels are correct. The class-aware mode
        # (cross-class intrusions and same-class extrusions only) is no longer
        # offered by the interface, so this is fixed off.
        "class_aware": False,
        "weight": "rank",
        "normalize": params.get("normalize", "none"),
        "norm_scale": float(params.get("norm_scale", 3.0)),
    }


# ---------------------------------------------------------------------------
# adaptive k reporting
# ---------------------------------------------------------------------------
def _adaptive_k_report(X_high: np.ndarray, k_base: int) -> tuple[dict | None,
                                                                np.ndarray | None]:
    """Per-point k vector plus the binned distribution the slider indicator shows.

    Adaptive k silently changes what every downstream number means, so the
    resulting spread is reported rather than left implicit.
    """
    from sklearn.metrics import pairwise_distances

    try:
        kv = np.asarray(adaptive_k(pairwise_distances(X_high, metric="euclidean"),
                                   k_base), dtype=int)
    except Exception as exc:
        print(f"[adaptive k] distribution skipped ({exc})")
        return None, None

    bins = [(3, 5), (6, 8), (9, 12), (13, 100)]
    labels = ["3-5", "6-8", "9-12", "13+"]
    counts = [int(np.sum((kv >= lo) & (kv <= hi))) for lo, hi in bins]
    total = max(1, len(kv))
    dist = {
        "bin_labels": labels,
        "counts": counts,
        "pct": [round(100.0 * c / total, 1) for c in counts],
        "k_min": int(kv.min()), "k_max": int(kv.max()),
        "k_median": int(np.median(kv)), "k_base": int(k_base),
        "n": total,
    }
    return dist, kv


# ---------------------------------------------------------------------------
# untrustworthy points: consensus fidelity + adaptive fence
# ---------------------------------------------------------------------------
def _flag_untrustworthy(neighbor_metrics: dict, ids: np.ndarray,
                        y: np.ndarray, target_names) -> list[dict]:
    """Flag points whose CONSENSUS fidelity is low in the worse of PCA / MDS.

    Flagging on any single metric inherits that metric's blind spot, so three
    are averaged. Taking the worse of the two methods is conservative on
    purpose: an analyst reading either picture would be misled.

    The threshold is a Tukey-style fence on the consensus distribution with an
    absolute floor beneath it, and the cascade below keeps the table readable on
    both very clean and very distorted runs. It is a usability heuristic, not a
    calibrated test -- a run that flags exactly the cap is reporting the cap.
    """
    rows: list[dict] = []
    if not neighbor_metrics:
        return rows

    per = {}
    for method in ("PCA", "MDS"):
        stack = np.vstack([np.asarray(neighbor_metrics[method][k])
                           for k in CONSENSUS_KEYS])
        per[method] = np.mean(stack, axis=0)

    worse = np.minimum(per["PCA"], per["MDS"])

    q1 = float(np.percentile(worse, 25))
    q3 = float(np.percentile(worse, 75))
    iqr = q3 - q1

    # (fence multiplier, absolute floor), tried in order
    cascade = [(0.5, 0.32), (0.8, 0.28), (1.0, 0.25)]
    fence, floor = q1 - cascade[0][0] * iqr, cascade[0][1]
    flagged = np.where((worse < fence) | (worse < floor))[0]
    for mult, flr in cascade[1:]:
        if len(flagged) <= 0.3 * len(worse):
            break
        fence, floor = q1 - mult * iqr, flr
        flagged = np.where((worse < fence) | (worse < floor))[0]
    if len(flagged) < 3:                       # too few: relax instead
        fence, floor = q1 - 0.3 * iqr, 0.35
        flagged = np.where((worse < fence) | (worse < floor))[0]

    flagged = flagged[np.argsort(worse[flagged])]          # worst first
    max_display = min(20, int(0.15 * len(worse)))
    if len(flagged) > max_display:
        flagged = flagged[:max_display]

    bad_thr = max(fence, floor)
    short = {"f1": "F1", "js_fidelity": "JS", "localstress": "stress"}
    for i in flagged:
        pca_c, mds_c = float(per["PCA"][i]), float(per["MDS"][i])
        which = ("both" if (pca_c < bad_thr and mds_c < bad_thr)
                 else ("PCA" if pca_c <= mds_c else "MDS"))
        worse_method = "PCA" if pca_c <= mds_c else "MDS"
        agree = [short[k] for k in CONSENSUS_KEYS
                 if float(neighbor_metrics[worse_method][k][i]) < bad_thr]
        rows.append({
            "index": int(ids[i]),
            "true_class": (target_names[y[i]] if y[i] < len(target_names)
                           else str(y[i])),
            "consensus": round(float(worse[i]), 3),
            "pca_consensus": round(pca_c, 3),
            "mds_consensus": round(mds_c, 3),
            "which": which,
            "agree": ", ".join(agree) if agree else "--",
            "pca_f1": round(float(neighbor_metrics["PCA"]["f1"][i]), 3),
            "mds_f1": round(float(neighbor_metrics["MDS"]["f1"][i]), 3),
        })

    print(f"[untrustworthy] Found {len(rows)} points with consensus "
          f"< {bad_thr:.2f}")
    return rows


# ---------------------------------------------------------------------------
# parameter-degeneracy guard
# ---------------------------------------------------------------------------
def _degeneracy_warnings(neighbors: int, n: int) -> list[dict]:
    """Warn when k is too large for n.

    The rank normaliser W = k(2n - 3k - 1)/2 is positive only for k < (2n-1)/3.
    Well before that hard limit, a k that is a large fraction of n makes almost
    every 2-D neighbourhood pull in unrelated points, so intrusion dominates and
    the scores lose contrast.
    """
    warnings: list[dict] = []
    hard_limit = (2 * n - 1) / 3.0
    ratio = neighbors / max(1, n)
    if neighbors >= hard_limit:
        warnings.append({
            "level": "error", "code": "k_over_limit",
            "message": (f"k = {neighbors} is at/above the metric's limit for "
                        f"n = {n} (needs k < {hard_limit:.0f}). Results are "
                        f"degenerate; reduce k or raise the sample size.")})
    elif ratio >= 0.15:
        warnings.append({
            "level": "warning", "code": "k_over_n",
            "message": (f"k = {neighbors} is {ratio*100:.0f}% of n = {n}. Above "
                        f"~15% the intrusion/extrusion balance skews heavily "
                        f"toward false positives and the colour scale loses "
                        f"contrast. Consider k \u2264 {int(0.15*n)} or a larger "
                        f"sample.")})
    return warnings


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    """Execute one analysis, capturing stdout for the interface console panel."""
    cfg = _read_params(params)
    buf = io.StringIO()
    t0 = time.time()

    with redirect_stdout(buf):
        X, y, target_names, name = load_dataset(
            cfg["dataset"], cfg["csv_path"], cfg["label_col"])
        describe(X, y, target_names, name)
        X, y, ids = maybe_subsample(X, y, cfg["max_samples"])

        # DISABLED: mutual-information feature weights for a weighted MDS.
        # No interface control reaches this. See legacy/adjusted_distance.py.

        # Projections always use the PLAIN Euclidean distance, so the per-point
        # metrics and the verdict are always measured against the true geometry.
        proj, X_high = build_projections(
            X, standardize=cfg["standardize"],
            method_a=cfg["method_a"], method_b=cfg["method_b"])
        timings = dict(getattr(build_projections, "last_timings",
                               {"pca_ms": 0, "mds_ms": 0}))
        # slot -> method actually used, so the interface can label the panels
        methods = dict(getattr(build_projections, "last_methods",
                               {"PCA": "PCA", "MDS": "MDS"}))
        print(f"[projections] left = {methods['PCA']}, right = {methods['MDS']}")

        # DISABLED: plain-vs-adjusted MDS comparison (Mahalanobis whitening or
        # Laplacian-Score locality weighting). This was the most expensive block
        # in the pipeline -- a second full MDS fit plus two extra metric passes,
        # two silhouette scores and two pdist calls per run -- and the panel that
        # displayed it has been removed. See legacy/adjusted_distance.py.

        errs, fps, fns = compute_errors(
            proj, X_high, y, k=cfg["neighbors"],
            class_aware=cfg["class_aware"], weight=cfg["weight"],
            adaptive=cfg["adaptive"], return_parts=True)

        # DISABLED: geometric-vs-class-aware divergence. Computing it meant a
        # second full compute_errors pass over both projections, and the readout
        # that displayed the result has been removed.

        adaptive_k_dist, k_vec = (None, None)
        if cfg["adaptive"]:
            adaptive_k_dist, k_vec = _adaptive_k_report(X_high, cfg["neighbors"])

        # Per-point neighbourhood preservation for both projections: the "each
        # point is a classifier of its own neighbourhood" view the interface
        # colours by, filters on and flags from.
        neighbor_metrics = None
        try:
            neighbor_metrics = {
                "k": int(cfg["neighbors"]), "adaptive": bool(cfg["adaptive"]),
                "PCA": per_point_neighbor_metrics(X_high, proj["PCA"],
                                                  k=cfg["neighbors"], k_vec=k_vec),
                "MDS": per_point_neighbor_metrics(X_high, proj["MDS"],
                                                  k=cfg["neighbors"], k_vec=k_vec),
            }
        except Exception as exc:
            print(f"[neighbour metrics] skipped ({exc})")

        if cfg["normalize"] != "none":
            errs = {m: normalize_errors(e, cfg["normalize"], cfg["norm_scale"])
                    for m, e in errs.items()}

        pca_err, mds_err = errs["PCA"], errs["MDS"]
        print("\nFirst 5 PCA errors:", np.round(pca_err[:5], 3))
        print("First 5 MDS errors:", np.round(mds_err[:5], 3))

        vmax = nice_vmax(pca_err, mds_err)

        # DISABLED: k-means ground-truth check, supervised graph
        # precision/recall/F-score, and the simulated-annealing search for the
        # graph's neighbourhood parameter. Together these were roughly half the
        # wall-clock time of a run on a few hundred points, and none of the three
        # cards that displayed them remains in the interface.
        # See legacy/label_structure.py.

        verdict_res = None
        try:
            verdict_res = compare_methods(X_high, proj, errs, fps, fns, y,
                                          k=cfg["neighbors"], kmeans=None,
                                          slot_names=methods)
            print(f"\n[verdict] {verdict_res['verdict']}")
        except Exception as exc:
            print(f"[verdict] skipped ({exc})")

        # DISABLED: MDS layout stability across extra random seeds. Three more
        # MDS fits plus a compute_errors pass each, and the figure was never
        # consumed by the interface.

        report_rows = _flag_untrustworthy(neighbor_metrics, ids, y, target_names)

    elapsed = time.time() - t0
    console = buf.getvalue() + f"\n[done] pipeline finished in {elapsed:.2f}s"

    classes, counts = np.unique(y, return_counts=True)
    n_now = int(X.shape[0])
    info = {
        "name": name,
        "n_samples": n_now,
        "n_features": int(X.shape[1]),
        "n_classes": int(len(classes)),
        "class_counts": [
            {"label": target_names[c] if c < len(target_names) else str(c),
             "code": int(c), "count": int(n)}
            for c, n in zip(classes, counts)],
        "standardized": cfg["standardize"],
        "elapsed": round(elapsed, 3),
        "k_over_n_ratio": round(cfg["neighbors"] / max(1, n_now), 3),
        "methods": methods,
        "warnings": _degeneracy_warnings(cfg["neighbors"], n_now),
    }

    fname = build_fname_tag(name, "intrusion", cfg["neighbors"], geometric=True,
                            adaptive=cfg["adaptive"], normalize=cfg["normalize"])

    # Feature matrix (standardised high-D values) so the interface can filter
    # points by attribute window. Rounded and column-capped to keep the payload
    # small; the browser rebuilds every figure from these arrays.
    feat = np.round(np.asarray(X_high, dtype=float), 4)
    fcols = list(range(min(feat.shape[1], 30)))
    labels_list = y.tolist()

    table = {
        "ids": ids.tolist(),
        "labels": labels_list,
        "class_names": [target_names[c] if c < len(target_names) else str(c)
                        for c in labels_list],
        "feature_names": [f"f{j}" for j in fcols],
        "feature_values": feat[:, fcols].tolist(),
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

    return {
        "ok": True,
        "params": params,
        "info": info,
        "target_names": target_names,
        "vmax": vmax,
        "fname": fname,
        "table": table,
        "untrustworthy": report_rows,
        "untrust_desc": "consensus",
        "summary": summary,
        "adaptive_k_dist": adaptive_k_dist,
        "neighbor_metrics": neighbor_metrics,
        "timings": timings,
        "verdict": verdict_res,
        "console": console,
        "metric": "intrusion",
        "timestamp": time.time(),
    }