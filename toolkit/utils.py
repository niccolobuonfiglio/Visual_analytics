"""
Shared helpers for the PCA/MDS error-scale project.

Pure, side-effect-light utilities pulled out of the analysis scripts so the
metric engine (error_core), the plotting layer (plotting) and the two CLI
front-ends can all reuse them without dragging in matplotlib or sklearn.

Kept deliberately free of any plotting imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def describe(X: ArrayLike, y: ArrayLike, target_names: Sequence[str],
             name: str) -> None:
    """Short human-readable summary of a labelled dataset.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (only its shape is used).
    y : array-like, shape (n_samples,)
        Integer class labels.
    target_names : sequence of str
        Human-readable class names, indexable by label value.
    name : str
        Dataset name, printed as the header.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    print("=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)
    print(f"Samples : {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes : {len(classes)}")
    for c, n in zip(classes, counts):
        label = target_names[c] if c < len(target_names) else str(c)
        print(f"  class {c} ({label}): {n}")


def resolve_outdir(base: str | Path, name: str, standardize: bool) -> Path:
    """Figures are bucketed per dataset and per preprocessing choice so repeated
    runs with different settings never overwrite each other.
    """
    sub = "standardized" if standardize else "raw"
    ds_dir = Path(base) / name / sub
    ds_dir.mkdir(parents=True, exist_ok=True)
    return ds_dir


def build_fname_tag(name: str, metric: str, neighbors: int, geometric: bool,
                    adaptive: bool = False, normalize: str = "none") -> str:
    """Build a filename stem encoding the metric and its settings.
    """
    if metric == "classic":
        tag = f"{name}_classic"
    else:
        mode = "geom" if geometric else "class"
        tag = f"{name}_intrusion_k{'auto' if adaptive else neighbors}_{mode}"
    if normalize and normalize != "none":
        tag += f"_{normalize}"
    return tag


def nice_vmax(*error_arrays: ArrayLike, floor: float = 0.1) -> float:
    """Symmetric colour-scale limit rounded up to the next 0.1.

    Takes the largest absolute error across every array passed in (so PCA and
    MDS share one scale) and rounds up so the colour bar has clean ticks.
    """
    peak = floor
    for e in error_arrays:
        e = np.asarray(e, dtype=float)
        if e.size:
            peak = max(peak, float(np.abs(e).max()))
    return float(np.ceil(peak * 10) / 10)


def normalize_errors(err: NDArray[np.float64], method: str = "none",
                     scale: float = 1.0) -> NDArray[np.float64]:
    """Optionally squash a signed error array while preserving its sign.

    Parameters
    ----------
    err : ndarray
        Signed errors, nominally in ``[-1, 1]``.
    method : {'none', 'tanh', 'sigmoid', 'sumabs'}
        - ``none``   : returned unchanged.
        - ``tanh``   : ``tanh(scale * err)`` ; smooth, sign-preserving, keeps
          small values roughly linear and saturates large ones.
        - ``sigmoid``: signed logistic ``2*sigmoid(scale*err) - 1``; similar
          to tanh but sharper near 0 for the same ``scale``.
        - ``sumabs`` : divide by the mean absolute error so the typical
          magnitude is ~1 (a soft, data-driven rescale rather than a squash).
    scale : float
        Steepness for ``tanh`` / ``sigmoid``; ignored otherwise.

    Returns
    -------
    ndarray
        Transformed errors, same shape as ``err``.
    """
    err = np.asarray(err, dtype=float)
    if method == "none":
        return err
    if method == "tanh":
        return np.tanh(scale * err)
    if method == "sigmoid":
        return 2.0 / (1.0 + np.exp(-scale * err)) - 1.0
    if method == "sumabs":
        denom = np.mean(np.abs(err))
        return err / denom if denom > 0 else err
    raise ValueError(f"Unknown normalize method '{method}'. "
                     "Choose none, tanh, sigmoid, or sumabs.")
