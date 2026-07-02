"""
Linked brushing for PCA <-> MDS projections (dataset-agnostic).

Drag a rectangle on EITHER plot; the same points light up on BOTH plots and
their error values are printed. Works in both directions. Wine is the default.

Run from a terminal with a GUI backend (TkAgg / QtAgg):
    python brushing.py
    python brushing.py --dataset iris
    python brushing.py --dataset digits --max-samples 300
    python brushing.py --csv mydata.csv --label-col species

It will NOT work with the Agg backend or with plain Jupyter inline; in a
notebook run `%matplotlib qt` (or `%matplotlib widget`) first. Shared data /
metric logic lives in error_core.py.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.widgets import RectangleSelector

from error_core import (SKLEARN_LOADERS, load_dataset, maybe_subsample,
                        build_projections, compute_errors)

CMAP = "RdBu"


def points_in_rect(coords, x0, x1, y0, y1):
    """Indices of coords (N,2) inside the axis-aligned rectangle. Pure -> testable."""
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))
    m = ((coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) &
         (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax))
    return np.where(m)[0]


class LinkedBrush:
    """Bidirectional rectangle brushing across two projection axes."""

    def __init__(self, projections, errors, labels, target_names, name):
        self.names = list(projections)            # e.g. ["PCA", "MDS"]
        self.proj = projections
        self.err = errors
        self.labels = np.asarray(labels)
        self.target_names = target_names
        peak = max(max(np.abs(e).max() for e in errors.values()), 0.1)
        vmax = float(np.ceil(peak * 10) / 10)
        self.norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        self._vmax = vmax

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        self.ax = dict(zip(self.names, self.axes))
        self.hl = {}          # highlight (ring) scatter per axis
        self.selectors = []

        for m in self.names:
            ax = self.ax[m]
            xy = self.proj[m]
            ax.scatter(xy[:, 0], xy[:, 1], c=self.err[m], cmap=CMAP,
                       norm=self.norm, s=55, edgecolors="k", linewidths=0.4, zorder=1)
            self.hl[m] = ax.scatter([], [], s=160, facecolors="none",
                                    edgecolors="lime", linewidths=2.2, zorder=3)
            ax.set_title(m)
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            sel = RectangleSelector(
                ax, self._make_callback(m), useblit=True, button=[1],
                minspanx=1e-6, minspany=1e-6, spancoords="data", interactive=False)
            self.selectors.append(sel)

        sm = ScalarMappable(norm=self.norm, cmap=CMAP)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.axes, fraction=0.046, pad=0.04,
                                 ticks=[-self._vmax, 0, self._vmax])
        cbar.ax.set_yticklabels([f"FN (-{self._vmax:g})", "True (0)",
                                 f"FP (+{self._vmax:g})"])
        self.fig.suptitle(f"{name} — linked brushing (drag a box on either plot; "
                          "FN red / True white / FP blue)", fontsize=13)

    def _make_callback(self, source):
        def on_select(eclick, erelease):
            idx = points_in_rect(self.proj[source], eclick.xdata, erelease.xdata,
                                 eclick.ydata, erelease.ydata)
            self._highlight(idx, source)
        return on_select

    def _highlight(self, idx, source):
        for m in self.names:
            self.hl[m].set_offsets(self.proj[m][idx] if len(idx)
                                   else np.empty((0, 2)))
        self.fig.canvas.draw_idle()
        self._report(idx, source)

    def _report(self, idx, source):
        print("\n" + "-" * 56)
        if len(idx) == 0:
            print(f"[{source}] empty selection")
            return
        print(f"[brushed on {source}] {len(idx)} point(s):")
        print(f"{'idx':>4} {'class':>12} " +
              " ".join(f"{m+'_err':>9}" for m in self.names))
        for i in idx:
            cls = self.target_names[self.labels[i]]
            vals = " ".join(f"{self.err[m][i]:>9.3f}" for m in self.names)
            print(f"{i:>4} {cls:>12} {vals}")

    def show(self):
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Linked PCA/MDS brushing.")
    p.add_argument("--dataset", default="wine", choices=list(SKLEARN_LOADERS))
    p.add_argument("--csv", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--metric", default="intrusion", choices=["intrusion", "classic"])
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--geometric", action="store_true",
                   help="intrusion metric ignores labels (pure neighbour ranks)")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--max-samples", type=int, default=None,
                   help="stratified subsample cap (MDS is slow on big data)")
    return p.parse_args()


def build(args):
    X, y, target_names, name = load_dataset(args.dataset, args.csv, args.label_col)
    X, y, _ = maybe_subsample(X, y, args.max_samples)
    proj, X_high = build_projections(X, standardize=not args.no_standardize)
    errors = compute_errors(proj, X_high, y, metric=args.metric,
                            k=args.neighbors, class_aware=not args.geometric)
    return proj, errors, y, target_names, name


def main():
    args = parse_args()
    proj, errors, y, target_names, name = build(args)
    LinkedBrush(proj, errors, y, target_names, name).show()


if __name__ == "__main__":
    main()