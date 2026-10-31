"""Manuscript definitions collected from correlation.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import warnings

import numpy as np

from scipy.stats import rankdata, spearmanr

def safe_corr(x, y, method="pearson", min_n=3):
    """Return a scalar Pearson or Spearman correlation with pairwise finite filtering."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < min_n:
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]
    if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
        return np.nan

    if method == "pearson":
        return float(np.corrcoef(x_valid, y_valid)[0, 1])
    corr_value, _ = spearmanr(x_valid, y_valid)
    return float(corr_value)

def rowwise_corr(matrix_a, matrix_b, method="pearson", min_obs=3):
    """Return row-wise correlations along the last axis."""
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.shape != b.shape:
        raise ValueError(f"Input arrays must have the same shape, got {a.shape} and {b.shape}")

    valid = np.isfinite(a) & np.isfinite(b)
    n_valid = valid.sum(axis=-1)

    if method == "spearman":
        a = rankdata(np.where(valid, a, np.nan), axis=-1, nan_policy="omit")
        b = rankdata(np.where(valid, b, np.nan), axis=-1, nan_policy="omit")
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a_mean = np.nanmean(np.where(valid, a, np.nan), axis=-1, keepdims=True)
        b_mean = np.nanmean(np.where(valid, b, np.nan), axis=-1, keepdims=True)

    a_centered = np.where(valid, a - a_mean, 0.0)
    b_centered = np.where(valid, b - b_mean, 0.0)
    numerator = np.sum(a_centered * b_centered, axis=-1)
    denominator = np.sqrt(np.sum(a_centered ** 2, axis=-1) * np.sum(b_centered ** 2, axis=-1))
    return np.divide(
        numerator,
        denominator,
        out=np.full(n_valid.shape, np.nan, dtype=float),
        where=(n_valid >= min_obs) & (denominator > 0),
    )

def pc_corr(vector_a, vector_b, method="pearson", min_obs=1):
    """Return place-cell spatial correlation for two one-dimensional maps."""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < min_obs:
        return np.nan

    a_valid = a[valid]
    b_valid = b[valid]
    if np.sum(a_valid) == 0 or np.sum(b_valid) == 0:
        return np.nan
    return safe_corr(a_valid, b_valid, method=method, min_n=max(2, min_obs))

def subsampled_pv_corr(snarm_a, snarm_b, target_n, n_iter=1000, seed=2026, method="spearman"):
    """Average binwise PV correlations over repeated fixed-neuron subsamples."""
    snarm_a = np.asarray(snarm_a, dtype=float)
    snarm_b = np.asarray(snarm_b, dtype=float)
    num_bins = snarm_a.shape[1]
    if np.isnan(target_n) or target_n < 3 or snarm_a.shape[0] < target_n:
        return np.full(num_bins, np.nan)

    target_n = int(target_n)
    rng = np.random.default_rng(seed)
    sampled_indices = np.asarray(
        [rng.choice(snarm_a.shape[0], size=target_n, replace=False) for _ in range(n_iter)]
    )
    sampled_a = np.transpose(snarm_a[sampled_indices, :], (0, 2, 1))
    sampled_b = np.transpose(snarm_b[sampled_indices, :], (0, 2, 1))
    pv_cor_iter = rowwise_corr(sampled_a, sampled_b, method=method)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(pv_cor_iter, axis=0)

def subsampled_pair_pv_corr(
    prev_maps,
    curr_maps,
    target_n,
    n_iter=1000,
    seed=2026,
    method="spearman",
    chunk_size=50,
):
    """Average per-pair/bin PV correlations over repeated fixed-neuron subsamples."""
    prev_maps = np.asarray(prev_maps, dtype=float)
    curr_maps = np.asarray(curr_maps, dtype=float)
    if prev_maps.shape != curr_maps.shape:
        raise ValueError(f"Input shapes must match, got {prev_maps.shape} and {curr_maps.shape}")

    n_pair, n_neuron, n_bin = prev_maps.shape
    target_n = int(target_n)
    if target_n < 3 or n_neuron < target_n or n_pair == 0:
        return np.full((n_pair, n_bin), np.nan)

    valid_neuron = np.isfinite(prev_maps).any(axis=(0, 2)) & np.isfinite(curr_maps).any(axis=(0, 2))
    prev_valid = prev_maps[:, valid_neuron, :]
    curr_valid = curr_maps[:, valid_neuron, :]
    n_valid = prev_valid.shape[1]
    if n_valid < target_n:
        return np.full((n_pair, n_bin), np.nan)

    rng = np.random.default_rng(seed)
    corr_sum = np.zeros((n_pair, n_bin), dtype=float)
    corr_count = np.zeros((n_pair, n_bin), dtype=float)

    for start in range(0, n_iter, chunk_size):
        this_chunk = min(chunk_size, n_iter - start)
        sample_idx = np.vstack(
            [rng.choice(n_valid, size=target_n, replace=False) for _ in range(this_chunk)]
        )
        sampled_prev = np.take(prev_valid, sample_idx, axis=1).transpose(1, 0, 3, 2)
        sampled_curr = np.take(curr_valid, sample_idx, axis=1).transpose(1, 0, 3, 2)
        corr = rowwise_corr(
            sampled_prev.reshape(this_chunk * n_pair * n_bin, target_n),
            sampled_curr.reshape(this_chunk * n_pair * n_bin, target_n),
            method=method,
        ).reshape(this_chunk, n_pair, n_bin)
        finite = np.isfinite(corr)
        corr_sum += np.where(finite, corr, 0.0).sum(axis=0)
        corr_count += finite.sum(axis=0)

    return np.divide(
        corr_sum,
        corr_count,
        out=np.full((n_pair, n_bin), np.nan),
        where=corr_count > 0,
    )
