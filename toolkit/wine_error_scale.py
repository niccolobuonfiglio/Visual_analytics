"""
Static PCA-vs-MDS continuous error-scale analysis (dataset-agnostic).

Wine is the default starting point; any built-in sklearn classification dataset
or a CSV with a label column works too. Shared logic lives in error_core.py,
helpers in utils.py, and all figures in plotting.py.

Usage:
    python wine_error_scale.py                          # Wine (default)
    python wine_error_scale.py --dataset iris
    python wine_error_scale.py --dataset digits --max-samples 400
    python wine_error_scale.py --csv mydata.csv --label-col species
    python wine_error_scale.py --dataset wine --no-standardize --threshold 0.6
    python wine_error_scale.py --weight distance --adaptive --normalize tanh
    python wine_error_scale.py --metric classic --alpha 1 --beta 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")  # drop this line if you want interactive windows

from toolkit.error_core import (SKLEARN_LOADERS, load_dataset, maybe_subsample,
                        build_projections, compute_errors)
from toolkit.utils import describe, resolve_outdir, build_fname_tag, nice_vmax, \
    normalize_errors
from toolkit.plotting import (plot_projections, plot_error_comparison,
                      plot_error_histograms, plot_parallel_coordinates)


# UNTRUSTWORTHY POINTS
def report_untrustworthy(pca_err: NDArray[np.float64],
                         mds_err: NDArray[np.float64], labels: NDArray[np.int_],
                         target_names: Sequence[str], ids: NDArray[np.int_],
                         thresh: float,
                         pca_fp: Optional[NDArray[np.float64]] = None,
                         pca_fn: Optional[NDArray[np.float64]] = None,
                         mds_fp: Optional[NDArray[np.float64]] = None,
                         mds_fn: Optional[NDArray[np.float64]] = None
                         ) -> pd.DataFrame:
    """Print points whose |error| exceeds ``thresh`` in either projection.

    When the FP/FN component arrays are supplied they are shown alongside the
    net error so it is clear whether intrusion or extrusion drove each point.
    """
    mask = (np.abs(pca_err) > thresh) | (np.abs(mds_err) > thresh)
    sel = np.where(mask)[0]
    cols = {
        "index": ids[sel],
        "true_class": [target_names[labels[i]] for i in sel],
        "pca_error": np.round(pca_err[sel], 3),
        "mds_error": np.round(mds_err[sel], 3),
    }
    if pca_fp is not None:
        cols["pca_FP"] = np.round(pca_fp[sel], 3)
        cols["pca_FN"] = np.round(pca_fn[sel], 3)
        cols["mds_FP"] = np.round(mds_fp[sel], 3)
        cols["mds_FN"] = np.round(mds_fn[sel], 3)
    df = pd.DataFrame(cols)
    print("\n" + "=" * 60)
    print(f"UNTRUSTWORTHY POINTS  (|error| > {thresh} in PCA or MDS)")
    print("=" * 60)
    print(f"Count: {len(sel)}")
    if len(sel):
        print(df.to_string(index=False))
    disagree = ((pca_err[sel] > 0) & (mds_err[sel] < 0)) | \
               ((pca_err[sel] < 0) & (mds_err[sel] > 0))
    print(f"\nSign disagreements (PCA/MDS opposite signs): {int(disagree.sum())}")
    return df


# MAIN
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous PCA/MDS error scale.")
    p.add_argument("--dataset", default="wine", choices=list(SKLEARN_LOADERS),
                   help="built-in sklearn dataset (default: wine)")
    p.add_argument("--csv", default=None, help="path to a CSV instead of a built-in set")
    p.add_argument("--label-col", default=None, help="label column name (with --csv)")
    p.add_argument("--neighbors", type=int, default=10,
                   help="neighbourhood size k for the intrusion metric")
    p.add_argument("--geometric", action="store_true",
                   help="intrusion metric ignores labels (pure neighbour ranks)")
    p.add_argument("--weight", default="rank", choices=["rank", "distance"],
                   help="intrusion offender weighting: rank-severity or distance")
    p.add_argument("--adaptive", action="store_true",
                   help="per-point density-driven k (intrusion metric)")
    p.add_argument("--normalize", default="none",
                   choices=["none", "tanh", "sigmoid", "sumabs"],
                   help="squash/transform final errors for display")
    p.add_argument("--norm-scale", type=float, default=3.0,
                   help="steepness for tanh/sigmoid normalization")
    p.add_argument("--no-standardize", action="store_true",
                   help="skip StandardScaler (default: standardize)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="stratified subsample cap (helps MDS on big data)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="|error| cutoff for 'untrustworthy' points")
    p.add_argument("--outdir", default="outputs", help="base output folder "
                   "(figures go to <outdir>/<dataset>/<standardized|raw>/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X, y, target_names, name = load_dataset(args.dataset, args.csv, args.label_col)
    describe(X, y, target_names, name)

    X, y, ids = maybe_subsample(X, y, args.max_samples)

    standardize = not args.no_standardize
    outdir = resolve_outdir(args.outdir, name, standardize)

    proj, X_high = build_projections(X, standardize=standardize)
    pca_2d, mds_2d = proj["PCA"], proj["MDS"]
    class_aware = not args.geometric

    # compute errors + FP/FN parts for asymmetric reporting
    errs, fps, fns = compute_errors(
        proj, X_high, y, k=args.neighbors,
        class_aware=class_aware, weight=args.weight, adaptive=args.adaptive,
        return_parts=True)

    # optional display normalization (applied to net error only)
    if args.normalize != "none":
        errs = {m: normalize_errors(e, args.normalize, args.norm_scale)
                for m, e in errs.items()}

    pca_err, mds_err = errs["PCA"], errs["MDS"]
    print("\nFirst 5 PCA errors:", np.round(pca_err[:5], 3))
    print("First 5 MDS errors:", np.round(mds_err[:5], 3))

    # filename tag + colour scale
    fname = build_fname_tag(name, "intrusion", args.neighbors, args.geometric,
                            adaptive=args.adaptive, normalize=args.normalize)
    vmax = nice_vmax(pca_err, mds_err)

    plot_projections(pca_2d, pca_err, mds_2d, mds_err, name, outdir, fname, vmax)
    plot_error_comparison(pca_err, mds_err, y, target_names, name, outdir,
                          fname, vmax)
    plot_error_histograms(errs, name, outdir, fname, vmax)
    plot_parallel_coordinates(errs, y, target_names, name, outdir, fname, vmax)

    report_untrustworthy(pca_err, mds_err, y, target_names, ids, args.threshold,
                         pca_fp=fps["PCA"], pca_fn=fns["PCA"],
                         mds_fp=fps["MDS"], mds_fn=fns["MDS"])


if __name__ == "__main__":
    main()
