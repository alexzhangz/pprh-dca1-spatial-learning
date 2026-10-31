"""Manuscript definitions collected from position_posterior.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import numpy as np

from scipy.sparse import csr_matrix

def mean_posterior_matrix(truth_bins, posterior):
    """Mean P(candidate bin | activity) conditional on each true bin.

    Return a square matrix (true bin, candidate bin) and frame counts by true
    bin. Unobserved true-bin rows are NaN. Float64 sums preserve stored soft
    probabilities; no hard classification or extra row normalization is used.
    """
    y, p = np.asarray(truth_bins), np.asarray(posterior, dtype=float)
    if y.ndim != 1 or p.ndim != 2 or len(y) != len(p) or p.shape[1] < 1:
        raise ValueError('matching 1D labels and 2D posterior required')
    n_bins = p.shape[1]
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= n_bins)):
        raise ValueError('true bins must be valid integer indices')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('posterior must contain finite probabilities')
    if not np.allclose(p.sum(axis=1), 1, atol=2e-7, rtol=0):
        raise ValueError('posterior rows must sum to one')
    counts = np.bincount(y, minlength=n_bins)
    assignment = csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(n_bins, len(y)))
    sums = assignment @ p
    mean = np.divide(sums, counts[:, None], out=np.full((n_bins, n_bins), np.nan), where=counts[:, None] > 0)
    return mean, counts

def average_mouse_matrices(matrices):
    """Equal-mouse mean per true-bin row, ignoring wholly missing rows."""
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or len(matrices) == 0 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError('nonempty mouse by bin by bin array required')
    finite = np.isfinite(matrices)
    valid = finite.all(axis=2)
    if np.any(finite.any(axis=2) != valid) or np.isinf(matrices).any():
        raise ValueError('missing data must be whole NaN rows')
    n_mice = valid.sum(axis=0)
    means = np.divide(np.nansum(matrices, axis=0), n_mice[:, None],
                     out=np.full(matrices.shape[1:], np.nan), where=n_mice[:, None] > 0)
    return means, n_mice
