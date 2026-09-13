"""
Compatibility facade for the former monolithic ``error_core`` module.

The contents were split into three focused modules:

    datasets.py     loading, synthetic generators, stratified subsampling
    projections.py  standardise + PCA + MDS
    metrics.py      adaptive k and the per-point fidelity metrics

Existing imports such as ``from error_core import load_dataset`` keep working
through this file. New code should import from the specific module instead.

Two former members are no longer re-exported because they are retired:
``compute_feature_weights`` and ``locality_feature_weights`` (now in
legacy/adjusted_distance.py) and ``compute_error_scale`` (legacy/legacy_misc.py).
"""

from datasets import (SKLEARN_LOADERS, SYNTHETIC_LOADERS, DATASET_NAMES,
                      RANDOM_STATE, load_dataset, maybe_subsample)
from projections import Projections, ErrorMap, build_projections
from metrics import (adaptive_k, per_point_neighbor_metrics,
                     compute_intrusion_extrusion_error, compute_errors)

__all__ = [
    "SKLEARN_LOADERS", "SYNTHETIC_LOADERS", "DATASET_NAMES", "RANDOM_STATE",
    "load_dataset", "maybe_subsample",
    "Projections", "ErrorMap", "build_projections",
    "adaptive_k", "per_point_neighbor_metrics",
    "compute_intrusion_extrusion_error", "compute_errors",
]
