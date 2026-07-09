"""
Plotting layer for the PCA/MDS error-scale project.

All matplotlib lives here (plus the brushing tool). Split out from the static
analysis script so both front-ends can share the same figures. Every function
takes already-computed errors and returns the saved path (or the Figure).

Colour convention throughout: RdBu diverging, low->red (FN), 0->white,
high->blue (FP), centred at 0 with a symmetric limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

CMAP = "RdBu"   # low->red (FN), 0->white, high->blue (FP)


def _diverging_norm(vmax: float) -> TwoSlopeNorm:
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def _add_error_colorbar(fig: Figure, axes, norm: TwoSlopeNorm,
                        vmax: float) -> None:
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04,
                        ticks=[-vmax, 0, vmax])
    cbar.ax.set_yticklabels([f"FN (-{vmax:g})", "True (0)", f"FP (+{vmax:g})"])


# =====================================================================
# SIDE-BY-SIDE PROJECTION PLOT
# =====================================================================
def plot_projections(pca_2d: NDArray[np.float64], pca_err: NDArray[np.float64],
                     mds_2d: NDArray[np.float64], mds_err: NDArray[np.float64],
                     name: str, outdir: str | Path, fname: str,
                     vmax: float = 1.0) -> Path:
    """Two panels (PCA, MDS), points coloured by signed error. Returns path."""
    norm = _diverging_norm(vmax)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, proj, err, m in [(axes[0], pca_2d, pca_err, "PCA"),
                             (axes[1], mds_2d, mds_err, "MDS")]:
        ax.scatter(proj[:, 0], proj[:, 1], c=err, cmap=CMAP, norm=norm,
                   s=55, edgecolors="k", linewidths=0.4)
        ax.set_title(m)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    _add_error_colorbar(fig, axes, norm, vmax)
    fig.suptitle(f"{name} — Error scale: FN (red) - FP (blue)",
                 fontsize=14)
    out = Path(outdir) / f"{fname}_projections.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# =====================================================================
# PCA-ERROR vs MDS-ERROR SCATTER
# =====================================================================
def plot_error_comparison(pca_err: NDArray[np.float64],
                          mds_err: NDArray[np.float64], labels: ArrayLike,
                          target_names: Sequence[str], name: str,
                          outdir: str | Path, fname: str,
                          vmax: float = 1.0) -> Path:
    """Per-point PCA error vs MDS error, coloured by class. Returns path."""
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(pca_err, mds_err, c=labels, cmap="viridis",
                    s=55, edgecolors="k", linewidths=0.4)
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("PCA error")
    ax.set_ylabel("MDS error")
    ax.set_title(f"{name} — PCA error vs MDS error (per point)")
    ax.set_xlim(-vmax, vmax)
    ax.set_ylim(-vmax, vmax)
    # quadrants (negated scale: <0 = FN, >0 = FP)
    ax.text(-0.95 * vmax, -0.95 * vmax, "both FN", ha="left", va="bottom",
            fontsize=9, alpha=0.7)
    ax.text(0.95 * vmax, 0.95 * vmax, "both FP", ha="right", va="top",
            fontsize=9, alpha=0.7)
    ax.text(-0.95 * vmax, 0.95 * vmax, "PCA-FN / MDS-FP", ha="left", va="top",
            fontsize=9, alpha=0.7)
    ax.text(0.95 * vmax, -0.95 * vmax, "PCA-FP / MDS-FN", ha="right",
            va="bottom", fontsize=9, alpha=0.7)
    handles, _ = sc.legend_elements()
    ax.legend(handles, list(target_names), title="class", loc="upper center",
              ncol=min(len(target_names), 5), fontsize=8)
    out = Path(outdir) / f"{fname}_error_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# =====================================================================
# ERROR HISTOGRAMS (PCA vs MDS, side by side)
# =====================================================================
def plot_error_histograms(errors: Mapping[str, NDArray[np.float64]], name: str,
                          outdir: str | Path, fname: str,
                          vmax: float = 1.0, bins: int = 30) -> Path:
    """Distribution of signed errors for each method, one panel per method.

    Bars are tinted by their bin centre on the same FN->True->FP scale, so the
    red/blue balance of a method is visible at a glance. A dashed line marks 0.
    """
    names = list(errors)
    norm = _diverging_norm(vmax)
    cmap = plt.get_cmap(CMAP)
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    edges = np.linspace(-vmax, vmax, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for ax, m in zip(axes, names):
        e = np.asarray(errors[m], dtype=float)
        counts, _ = np.histogram(np.clip(e, -vmax, vmax), bins=edges)
        ax.bar(centres, counts, width=(edges[1] - edges[0]) * 0.9,
               color=cmap(norm(centres)), edgecolor="k", linewidth=0.3)
        ax.axvline(0, color="grey", ls="--", lw=1)
        mean_e = float(e.mean())
        ax.axvline(mean_e, color="black", ls=":", lw=1.2,
                   label=f"mean {mean_e:+.3f}")
        ax.set_title(f"{m}  (n={e.size})")
        ax.set_xlabel("signed error  (FN < 0 < FP)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle(f"{name} — error distribution per method", fontsize=14)
    out = Path(outdir) / f"{fname}_histograms.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# =====================================================================
# PARALLEL COORDINATES (mean error per class across methods)
# =====================================================================
def plot_parallel_coordinates(errors: Mapping[str, NDArray[np.float64]],
                              labels: ArrayLike, target_names: Sequence[str],
                              name: str, outdir: str | Path, fname: str,
                              vmax: float = 1.0) -> Path:
    """One polyline per class tracing its mean error across each method axis.

    Axes (verticals) are the methods (PCA, MDS, ...); each class is a line whose
    height on each axis is that class's mean signed error under that method. A
    light band shows +/- one within-class standard deviation. This exposes, for
    instance, a class that PCA embeds faithfully but MDS distorts.
    """
    labels = np.asarray(labels)
    names = list(errors)
    classes = np.unique(labels)
    xs = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 4, 6))
    cmap = plt.get_cmap("viridis", len(classes))
    for ci, c in enumerate(classes):
        mean_list: list[float] = []
        std_list: list[float] = []
        for m in names:
            e = np.asarray(errors[m], dtype=float)[labels == c]
            mean_list.append(float(e.mean()) if e.size else 0.0)
            std_list.append(float(e.std()) if e.size else 0.0)
        means = np.asarray(mean_list)
        stds = np.asarray(std_list)
        cls_name = target_names[c] if c < len(target_names) else str(c)
        ax.plot(xs, means, "-o", color=cmap(ci), lw=2, label=cls_name, zorder=3)
        ax.fill_between(xs, means - stds, means + stds, color=cmap(ci),
                        alpha=0.15, zorder=1)

    for x in xs:                      # vertical axis lines
        ax.axvline(float(x), color="grey", lw=0.8, alpha=0.5)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(-vmax, vmax)
    ax.set_ylabel("mean signed error  (FN < 0 < FP)")
    ax.set_title(f"{name} — mean error per class across methods")
    ax.legend(title="class", fontsize=8, loc="best")
    out = Path(outdir) / f"{fname}_parallel.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return out
