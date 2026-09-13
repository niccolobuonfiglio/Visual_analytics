"""
Compatibility facade for the former ``analysis`` module.

The live part -- the PCA-vs-MDS composite verdict and the classical
trustworthiness / continuity it rests on -- now lives in ``verdict.py``.

The label- and cluster-structure family that used to share this file (k-NN
graph precision/recall/F-score, the k-means ground-truth check and the
simulated-annealing parameter search) is not reachable from the interface and
has been moved to ``legacy/label_structure.py``. It is intentionally NOT
re-exported here, so that importing it again is a deliberate act.
"""

from toolkit.verdict import compare_methods

__all__ = ["compare_methods"]
