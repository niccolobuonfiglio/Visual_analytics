"""
Static PCA-vs-MDS continuous error-scale analysis (dataset-agnostic).

Wine is the default starting point; any built-in sklearn classification dataset
or a CSV with a label column works too. Shared logic lives in error_core.py.

Usage:
    python wine_error_scale.py                          # Wine (default)
    python wine_error_scale.py --dataset iris
    python wine_error_scale.py --dataset digits --max-samples 400
    python wine_error_scale.py --csv mydata.csv --label-col species
    python wine_error_scale.py --dataset wine --no-standardize --threshold 0.6
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # drop this line if you want interactive windows
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from error_core import (SKLEARN_LOADERS, load_dataset, describe,
                        maybe_subsample, compute_error_scale, build_projections)


# =====================================================================
# SIDE-BY-SIDE PROJECTION PLOT
# =====================================================================
def plot_projections(pca_2d, pca_err, mds_2d, mds_err, name, outdir):
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = "RdBu"   # low(-1)->red, 0->white, high(+1)->blue
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, proj, err, m in [(axes[0], pca_2d, pca_err, "PCA"),
                             (axes[1], mds_2d, mds_err, "MDS")]:
        ax.scatter(proj[:, 0], proj[:, 1], c=err, cmap=cmap, norm=norm,
                   s=55, edgecolors="k", linewidths=0.4)
        ax.set_title(m)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["FN (-1)", "True (0)", "FP (+1)"])
    fig.suptitle(f"{name} — Error scale: FN (red) – True (white/green) – FP (blue)",
                 fontsize=14)
    out = Path(outdir) / f"{name}_projections.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# =====================================================================
# PCA-ERROR vs MDS-ERROR SCATTER
# =====================================================================
def plot_error_comparison(pca_err, mds_err, labels, target_names, name, outdir):
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(pca_err, mds_err, c=labels, cmap="viridis",
                    s=55, edgecolors="k", linewidths=0.4)
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("PCA error")
    ax.set_ylabel("MDS error")
    ax.set_title(f"{name} — PCA error vs MDS error (per point)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # quadrants (negated scale: <0 = FN, >0 = FP)
    ax.text(-0.95, -0.95, "both FN", ha="left", va="bottom", fontsize=9, alpha=0.7)
    ax.text(0.95, 0.95, "both FP", ha="right", va="top", fontsize=9, alpha=0.7)
    ax.text(-0.95, 0.95, "PCA-FN / MDS-FP", ha="left", va="top", fontsize=9, alpha=0.7)
    ax.text(0.95, -0.95, "PCA-FP / MDS-FN", ha="right", va="bottom", fontsize=9, alpha=0.7)
    handles, _ = sc.legend_elements()
    ax.legend(handles, list(target_names), title="class", loc="upper center",
              ncol=min(len(target_names), 5), fontsize=8)
    out = Path(outdir) / f"{name}_error_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# =====================================================================
# UNTRUSTWORTHY POINTS
# =====================================================================
def report_untrustworthy(pca_err, mds_err, labels, target_names, ids, thresh):
    mask = (np.abs(pca_err) > thresh) | (np.abs(mds_err) > thresh)
    sel = np.where(mask)[0]
    df = pd.DataFrame({
        "index": ids[sel],
        "true_class": [target_names[labels[i]] for i in sel],
        "pca_error": np.round(pca_err[sel], 3),
        "mds_error": np.round(mds_err[sel], 3),
    })
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


# =====================================================================
# MAIN
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Continuous PCA/MDS error scale.")
    p.add_argument("--dataset", default="wine", choices=list(SKLEARN_LOADERS),
                   help="built-in sklearn dataset (default: wine)")
    p.add_argument("--csv", default=None, help="path to a CSV instead of a built-in set")
    p.add_argument("--label-col", default=None, help="label column name (with --csv)")
    p.add_argument("--no-standardize", action="store_true",
                   help="skip StandardScaler (default: standardize)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="stratified subsample cap (helps MDS on big data)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="|error| cutoff for 'untrustworthy' points")
    p.add_argument("--outdir", default=".", help="where to write the PNGs")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    X, y, target_names, name = load_dataset(args.dataset, args.csv, args.label_col)
    describe(X, y, target_names, name)

    X, y, ids = maybe_subsample(X, y, args.max_samples)

    proj = build_projections(X, standardize=not args.no_standardize)
    pca_2d, mds_2d = proj["PCA"], proj["MDS"]
    pca_err = compute_error_scale(pca_2d, y)
    mds_err = compute_error_scale(mds_2d, y)
    print("\nFirst 5 PCA errors:", np.round(pca_err[:5], 3))
    print("First 5 MDS errors:", np.round(mds_err[:5], 3))

    plot_projections(pca_2d, pca_err, mds_2d, mds_err, name, args.outdir)
    plot_error_comparison(pca_err, mds_err, y, target_names, name, args.outdir)
    report_untrustworthy(pca_err, mds_err, y, target_names, ids, args.threshold)


if __name__ == "__main__":
    main()
