"""Manuscript definitions collected from manifold.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import warnings

from pathlib import Path

import numpy as np

from sklearn.covariance import LedoitWolf

from sklearn.decomposition import PCA

from sklearn.manifold import SpectralEmbedding

from .subjects import GROUP_ORDER

from .trials import should_skip_mouse_day

CONTRAST_NAMES = [
    "left_vs_right",
    "previous_vs_current_left",
    "previous_vs_current_right",
]

DEFAULT_TARGET_NEURON_COUNT = 50

DEFAULT_EMBEDDING_DIM = 15

DEFAULT_REDUCER = "spectral"

DPRIME_COVARIANCE_WITHIN_CLASS = "within_class_ledoit_wolf"

DPRIME_COVARIANCE_LEGACY_POOLED = "legacy_pooled"

DEFAULT_DPRIME_COVARIANCE = DPRIME_COVARIANCE_WITHIN_CLASS

LEGACY_CACHE_FILENAME = "manifold_quantification_50all_dim15.npy"

LEGACY_ZSCORE_CACHE_FILENAME = "manifold_quantification_50all_dim15_zscore.npy"

DEFAULT_UNZSCORE_CACHE_FILENAME = "manifold_quantification_50all_dim15_within_class_ledoit_wolf.npy"

DEFAULT_CACHE_FILENAME = "manifold_quantification_50all_dim15_zscore_within_class_ledoit_wolf.npy"

CONTRAST_BIN_LABELS = {
    "left_vs_right": tuple(
        [f"previous_bin_{idx:02d}" for idx in range(56)]
        + [f"current_bin_{idx:02d}" for idx in range(56)]
    ),
    "previous_vs_current_left": tuple(f"bin_{idx:02d}" for idx in range(56)),
    "previous_vs_current_right": tuple(f"bin_{idx:02d}" for idx in range(56)),
}

CONTRAST_BIN_COUNTS = {
    contrast_name: len(labels)
    for contrast_name, labels in CONTRAST_BIN_LABELS.items()
}

def normalize_dprime_covariance(dprime_covariance=None):
    """Return the canonical d-prime covariance estimator name."""
    key = DEFAULT_DPRIME_COVARIANCE if dprime_covariance is None else str(dprime_covariance)
    key = key.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        DPRIME_COVARIANCE_WITHIN_CLASS: DPRIME_COVARIANCE_WITHIN_CLASS,
        "within_class": DPRIME_COVARIANCE_WITHIN_CLASS,
        "within_class_ledoitwolf": DPRIME_COVARIANCE_WITHIN_CLASS,
        "ledoit_wolf": DPRIME_COVARIANCE_WITHIN_CLASS,
        "lda": DPRIME_COVARIANCE_WITHIN_CLASS,
        "regularized_lda": DPRIME_COVARIANCE_WITHIN_CLASS,
        DPRIME_COVARIANCE_LEGACY_POOLED: DPRIME_COVARIANCE_LEGACY_POOLED,
        "pooled": DPRIME_COVARIANCE_LEGACY_POOLED,
        "pooled_covariance": DPRIME_COVARIANCE_LEGACY_POOLED,
        "legacy": DPRIME_COVARIANCE_LEGACY_POOLED,
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported dprime_covariance {dprime_covariance!r}. "
            f"Expected {DPRIME_COVARIANCE_WITHIN_CLASS!r} or {DPRIME_COVARIANCE_LEGACY_POOLED!r}."
        )
    return aliases[key]

def cache_filename_for_dprime_covariance(dprime_covariance=None, *, zscore=True):
    """Return the default manifold cache filename for a covariance estimator."""
    covariance = normalize_dprime_covariance(dprime_covariance)
    if covariance == DPRIME_COVARIANCE_WITHIN_CLASS:
        return DEFAULT_CACHE_FILENAME if zscore else DEFAULT_UNZSCORE_CACHE_FILENAME
    if covariance == DPRIME_COVARIANCE_LEGACY_POOLED:
        return LEGACY_ZSCORE_CACHE_FILENAME if zscore else LEGACY_CACHE_FILENAME
    raise AssertionError(f"Unhandled covariance estimator {covariance!r}")

def compute_day_manifold_quantification(
    snarm_day,
    labels_day,
    target_neuron_count=DEFAULT_TARGET_NEURON_COUNT,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    n_repeat=100,
    n_shuffle=100,
    seed=2026,
    reducer=DEFAULT_REDUCER,
    min_finite_fraction=0.80,
    show_progress=False,
    progress_desc=None,
    progress_position=0,
    segment_ids=None,
    is_context_trial=None,
    zscore=True,
    dprime_covariance=DEFAULT_DPRIME_COVARIANCE,
):
    """Compute per-trial embedding and variable-bin d-prime for one day."""
    reducer = _normalize_reducer(reducer)
    dprime_covariance = normalize_dprime_covariance(dprime_covariance)
    target_neuron_count = int(target_neuron_count)
    embedding_dim = int(embedding_dim)
    n_repeat = int(n_repeat)
    n_shuffle = int(n_shuffle)
    zscore = bool(zscore)

    snarm_day = np.asarray(snarm_day, dtype=float)
    if snarm_day.ndim != 3:
        raise ValueError(f"snarm_day must be 3D (trials, neurons, bins), got {snarm_day.shape}")
    n_trials, _n_neurons, n_bins = snarm_day.shape
    if n_bins != 56:
        raise ValueError(f"Expected 56 spatial bins, got {n_bins}")

    labels = np.asarray(labels_day, dtype=bool)
    if labels.ndim != 2 or labels.shape[0] < 3:
        raise ValueError(f"labels_day must be (3, n_trials), got {labels.shape}")

    labels_aligned, n_trial_aligned = _align_labels_to_snarm(labels, n_trials)
    snarm_trial = snarm_day[:n_trial_aligned].transpose(0, 2, 1)
    valid_trial = np.any(labels_aligned[:3], axis=0)

    if is_context_trial is not None:
        ctx_aligned = _align_trial_vector(
            is_context_trial,
            n_trial_aligned,
            labels.shape[1],
            dtype=bool,
            name="is_context_trial",
        )
        valid_trial &= ~ctx_aligned

    outcome = labels_aligned[2]
    rewarded_left = labels_aligned[1]
    lc_idx = np.where(rewarded_left & outcome & valid_trial)[0]
    rc_idx = np.where((~rewarded_left) & outcome & valid_trial)[0]

    all_idx = np.where(valid_trial)[0]
    n_all = int(all_idx.size)
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(all_idx)}

    current_correct = outcome & valid_trial
    curr_idx = np.where(current_correct)[0]
    curr_idx = curr_idx[curr_idx > 0]
    prev_idx = curr_idx - 1
    pair_valid = valid_trial[prev_idx] & valid_trial[curr_idx]

    if segment_ids is not None:
        seg_aligned = _align_trial_vector(
            segment_ids,
            n_trial_aligned,
            labels.shape[1],
            dtype=int,
            name="segment_ids",
        )
        pair_valid &= seg_aligned[prev_idx] == seg_aligned[curr_idx]

    prev_idx = prev_idx[pair_valid]
    curr_idx = curr_idx[pair_valid]
    current_left = rewarded_left[curr_idx]

    left_pair_trials = (
        np.column_stack([prev_idx[current_left], curr_idx[current_left]])
        if np.any(current_left)
        else np.empty((0, 2), dtype=int)
    )
    right_pair_trials = (
        np.column_stack([prev_idx[~current_left], curr_idx[~current_left]])
        if np.any(~current_left)
        else np.empty((0, 2), dtype=int)
    )
    left_pair_pos = _map_pair_positions(left_pair_trials, idx_to_pos)
    right_pair_pos = _map_pair_positions(right_pair_trials, idx_to_pos)
    pl_pos = left_pair_pos[:, 0]
    cl_pos = left_pair_pos[:, 1]
    pr_pos = right_pair_pos[:, 0]
    cr_pos = right_pair_pos[:, 1]

    min_trials_per_contrast = 3
    valid_lvr = len(left_pair_pos) >= min_trials_per_contrast and len(right_pair_pos) >= min_trials_per_contrast
    valid_pvcl = len(pl_pos) >= min_trials_per_contrast and len(cl_pos) >= min_trials_per_contrast
    valid_pvcr = len(pr_pos) >= min_trials_per_contrast and len(cr_pos) >= min_trials_per_contrast
    any_valid = valid_lvr or valid_pvcl or valid_pvcr

    base_counts = {
        "n_lc": int(lc_idx.size),
        "n_rc": int(rc_idx.size),
        "n_pl": int(pl_pos.size),
        "n_cl": int(cl_pos.size),
        "n_pr": int(pr_pos.size),
        "n_cr": int(cr_pos.size),
        "n_total": n_all,
    }
    pair_indices = {
        "left": left_pair_trials.astype(int),
        "right": right_pair_trials.astype(int),
    }

    if n_all < 6 or not any_valid:
        result = _empty_per_trial_result(
            n_repeat,
            n_shuffle,
            zscore=zscore,
            dprime_covariance=dprime_covariance,
        )
        result.update(base_counts)
        result["full_result"]["valid_trial_indices"] = all_idx.astype(int)
        result["full_result"]["pair_indices"] = pair_indices
        return result

    n_finite = np.isfinite(snarm_trial[all_idx]).sum(axis=(0, 1))
    keep = n_finite >= (n_all * n_bins * min_finite_fraction)
    candidate_indices = np.where(keep)[0]
    candidate_neuron_count = int(candidate_indices.size)

    if candidate_neuron_count < target_neuron_count:
        result = _empty_per_trial_result(
            n_repeat,
            n_shuffle,
            zscore=zscore,
            dprime_covariance=dprime_covariance,
        )
        result.update(base_counts)
        result["candidate_neuron_count"] = candidate_neuron_count
        result["target_neuron_count"] = target_neuron_count
        result["embedding_dim"] = embedding_dim
        result["reducer"] = reducer
        result["full_result"]["valid_trial_indices"] = all_idx.astype(int)
        result["full_result"]["pair_indices"] = pair_indices
        return result

    embedding_pool = snarm_trial[all_idx][:, :, candidate_indices]
    if zscore:
        embedding_pool = _zscore_columns(
            embedding_pool.reshape(n_all * n_bins, candidate_neuron_count)
        ).reshape(n_all, n_bins, candidate_neuron_count)

    rng = np.random.default_rng(seed)
    n_pos = n_all * n_bins
    per_bin_real = {
        name: np.full((n_repeat, CONTRAST_BIN_COUNTS[name]), np.nan, dtype=np.float32)
        for name in CONTRAST_NAMES
    }
    per_bin_shuffle = {
        name: np.full((n_repeat, n_shuffle, CONTRAST_BIN_COUNTS[name]), np.nan, dtype=np.float32)
        for name in CONTRAST_NAMES
    }
    embeddings = np.full((n_repeat, n_all, n_bins, embedding_dim), np.nan, dtype=np.float32)
    selected_neuron_indices = np.full((n_repeat, target_neuron_count), -1, dtype=int)
    embedding_seeds = np.full(n_repeat, -1, dtype=np.int64)
    shuffle_seeds = np.full((n_repeat, n_shuffle), -1, dtype=np.int64)

    repeat_iter = _progress(
        range(n_repeat),
        enabled=show_progress,
        desc=f"{progress_desc or 'per-bin'} repeats",
        total=n_repeat,
        position=progress_position,
        leave=False,
    )
    for repeat_idx in repeat_iter:
        selected = rng.choice(candidate_indices, size=target_neuron_count, replace=False)
        selected_neuron_indices[repeat_idx] = selected.astype(int)
        selected_pool = np.searchsorted(candidate_indices, selected)
        X = embedding_pool[:, :, selected_pool].reshape(n_pos, target_neuron_count)
        emb_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        embedding_seeds[repeat_idx] = emb_seed
        embedding = fit_embedding(X, embedding_dim=embedding_dim, reducer=reducer, seed=emb_seed)
        emb_3d = embedding.reshape(n_all, n_bins, embedding_dim)
        embeddings[repeat_idx] = emb_3d.astype(np.float32)

        emb_left = emb_right = None
        lvr_precision = None
        if valid_lvr:
            emb_left = _paired_trajectory_embedding(emb_3d, left_pair_pos)
            emb_right = _paired_trajectory_embedding(emb_3d, right_pair_pos)
            lvr_precision = _precision_by_bin(
                emb_left,
                emb_right,
                dprime_covariance=dprime_covariance,
            )
            for bin_i in range(CONTRAST_BIN_COUNTS["left_vs_right"]):
                per_bin_real["left_vs_right"][repeat_idx, bin_i] = _d_prime_from_precision(
                    emb_left[:, bin_i, :],
                    emb_right[:, bin_i, :],
                    lvr_precision[bin_i],
                )

        emb_pl = emb_cl = None
        pvcl_precision = None
        if valid_pvcl:
            emb_pl = emb_3d[pl_pos]
            emb_cl = emb_3d[cl_pos]
            pvcl_precision = _precision_by_bin(
                emb_pl,
                emb_cl,
                dprime_covariance=dprime_covariance,
            )
            for bin_i in range(n_bins):
                per_bin_real["previous_vs_current_left"][repeat_idx, bin_i] = _d_prime_from_precision(
                    emb_pl[:, bin_i, :],
                    emb_cl[:, bin_i, :],
                    pvcl_precision[bin_i],
                )

        emb_pr = emb_cr = None
        pvcr_precision = None
        if valid_pvcr:
            emb_pr = emb_3d[pr_pos]
            emb_cr = emb_3d[cr_pos]
            pvcr_precision = _precision_by_bin(
                emb_pr,
                emb_cr,
                dprime_covariance=dprime_covariance,
            )
            for bin_i in range(n_bins):
                per_bin_real["previous_vs_current_right"][repeat_idx, bin_i] = _d_prime_from_precision(
                    emb_pr[:, bin_i, :],
                    emb_cr[:, bin_i, :],
                    pvcr_precision[bin_i],
                )

        for shuffle_i in range(n_shuffle):
            shuf_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            shuffle_seeds[repeat_idx, shuffle_i] = shuf_seed
            shuf_rng = np.random.default_rng(shuf_seed)

            if valid_lvr:
                emb_left_s, emb_right_s = _shuffle_rows(emb_left, emb_right, shuf_rng)
                for bin_i in range(CONTRAST_BIN_COUNTS["left_vs_right"]):
                    per_bin_shuffle["left_vs_right"][repeat_idx, shuffle_i, bin_i] = _d_prime_from_precision(
                        emb_left_s[:, bin_i, :],
                        emb_right_s[:, bin_i, :],
                        lvr_precision[bin_i],
                    )

            if valid_pvcl:
                emb_pl_s, emb_cl_s = _shuffle_rows(emb_pl, emb_cl, shuf_rng)
                for bin_i in range(n_bins):
                    per_bin_shuffle["previous_vs_current_left"][repeat_idx, shuffle_i, bin_i] = _d_prime_from_precision(
                        emb_pl_s[:, bin_i, :],
                        emb_cl_s[:, bin_i, :],
                        pvcl_precision[bin_i],
                    )

            if valid_pvcr:
                emb_pr_s, emb_cr_s = _shuffle_rows(emb_pr, emb_cr, shuf_rng)
                for bin_i in range(n_bins):
                    per_bin_shuffle["previous_vs_current_right"][repeat_idx, shuffle_i, bin_i] = _d_prime_from_precision(
                        emb_pr_s[:, bin_i, :],
                        emb_cr_s[:, bin_i, :],
                        pvcr_precision[bin_i],
                    )

    per_bin_dprime_mean = {}
    per_bin_shuffle_z = {}
    for contrast_name in CONTRAST_NAMES:
        real_values = per_bin_real[contrast_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            real_mean = np.nanmean(real_values, axis=0)
            shuf_flat = per_bin_shuffle[contrast_name].reshape(-1, real_values.shape[1])
            shuf_mean = np.nanmean(shuf_flat, axis=0)
            shuf_std = np.nanstd(shuf_flat, axis=0, ddof=1)
        per_bin_dprime_mean[contrast_name] = real_mean.astype(np.float32)
        per_bin_shuffle_z[contrast_name] = np.divide(
            real_mean - shuf_mean,
            shuf_std,
            out=np.full(real_values.shape[1], np.nan, dtype=np.float32),
            where=(shuf_std > 0) & np.isfinite(shuf_std),
        ).astype(np.float32)

    result = {
        "valid": True,
        **base_counts,
        "candidate_neuron_count": candidate_neuron_count,
        "target_neuron_count": target_neuron_count,
        "embedding_dim": embedding_dim,
        "reducer": reducer,
        "zscore": zscore,
        "dprime_covariance": dprime_covariance,
        "n_repeat": n_repeat,
        "n_shuffle": n_shuffle,
        "per_bin_dprime_mean_by_contrast": per_bin_dprime_mean,
        "per_bin_shuffle_z_by_contrast": per_bin_shuffle_z,
        "bin_count_by_contrast": dict(CONTRAST_BIN_COUNTS),
        "bin_labels_by_contrast": {
            name: list(labels)
            for name, labels in CONTRAST_BIN_LABELS.items()
        },
        "contrast_names": list(CONTRAST_NAMES),
        "full_result": {
            "embeddings": embeddings,
            "selected_neuron_indices": selected_neuron_indices,
            "embedding_seeds": embedding_seeds,
            "shuffle_seeds": shuffle_seeds,
            "real_dprime_by_contrast": per_bin_real,
            "shuffle_dprime_by_contrast": per_bin_shuffle,
            "valid_trial_indices": all_idx.astype(int),
            "pair_indices": pair_indices,
        },
    }
    return result

def _align_labels_to_snarm(labels, n_trials):
    """Align day-level trial labels to snarm_1d trial rows."""
    if labels.shape[1] >= n_trials + 1:
        n_trial_aligned = min(int(n_trials), labels.shape[1] - 1)
        return labels[:, 1 : 1 + n_trial_aligned], n_trial_aligned
    if labels.shape[1] >= n_trials:
        n_trial_aligned = int(n_trials)
        return labels[:, :n_trial_aligned], n_trial_aligned
    raise ValueError(
        f"labels_day has {labels.shape[1]} columns, fewer than snarm trials {n_trials}"
    )

def _align_trial_vector(values, n_trial_aligned, labels_width, dtype=None, name="metadata"):
    """Align 1D per-trial metadata to the same trial rows as labels[:, 1:]."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    if arr.size >= labels_width and labels_width == n_trial_aligned + 1:
        aligned = arr[1 : 1 + n_trial_aligned]
    elif arr.size >= n_trial_aligned:
        aligned = arr[:n_trial_aligned]
    else:
        raise ValueError(
            f"{name} has {arr.size} entries, fewer than aligned trials {n_trial_aligned}"
        )
    if dtype is not None:
        aligned = aligned.astype(dtype)
    return aligned

def _map_pair_positions(pair_trials, idx_to_pos):
    pair_trials = np.asarray(pair_trials, dtype=int)
    if pair_trials.size == 0:
        return np.empty((0, 2), dtype=int)
    rows = []
    for prev_idx, curr_idx in pair_trials:
        prev_pos = idx_to_pos.get(int(prev_idx), -1)
        curr_pos = idx_to_pos.get(int(curr_idx), -1)
        if prev_pos >= 0 and curr_pos >= 0:
            rows.append((prev_pos, curr_pos))
    if not rows:
        return np.empty((0, 2), dtype=int)
    return np.asarray(rows, dtype=int)

def _paired_trajectory_embedding(emb_3d, pair_pos):
    pair_pos = np.asarray(pair_pos, dtype=int)
    if pair_pos.size == 0:
        return np.empty((0, 112, emb_3d.shape[2]), dtype=emb_3d.dtype)
    previous = emb_3d[pair_pos[:, 0]]
    current = emb_3d[pair_pos[:, 1]]
    return np.concatenate([previous, current], axis=1)

def _shuffle_rows(class_a, class_b, rng):
    class_a = np.asarray(class_a)
    class_b = np.asarray(class_b)
    combined = np.concatenate([class_a, class_b], axis=0)
    order = rng.permutation(combined.shape[0])
    n_a = class_a.shape[0]
    return combined[order[:n_a]], combined[order[n_a:]]

def _empty_per_trial_result(
    n_repeat,
    n_shuffle,
    zscore=True,
    dprime_covariance=DEFAULT_DPRIME_COVARIANCE,
):
    dprime_covariance = normalize_dprime_covariance(dprime_covariance)
    dprime = {
        name: np.full(CONTRAST_BIN_COUNTS[name], np.nan, dtype=np.float32)
        for name in CONTRAST_NAMES
    }
    shuffle_z = {
        name: np.full(CONTRAST_BIN_COUNTS[name], np.nan, dtype=np.float32)
        for name in CONTRAST_NAMES
    }
    real_dprime = {
        name: np.full((int(n_repeat), CONTRAST_BIN_COUNTS[name]), np.nan, dtype=np.float32)
        for name in CONTRAST_NAMES
    }
    shuffle_dprime = {
        name: np.full((int(n_repeat), int(n_shuffle), CONTRAST_BIN_COUNTS[name]), np.nan, dtype=np.float32)
        for name in CONTRAST_NAMES
    }
    return {
        "valid": False,
        "n_lc": 0, "n_rc": 0, "n_pl": 0, "n_cl": 0, "n_pr": 0, "n_cr": 0,
        "n_total": 0,
        "candidate_neuron_count": 0,
        "target_neuron_count": 0,
        "embedding_dim": 0,
        "reducer": DEFAULT_REDUCER,
        "zscore": bool(zscore),
        "dprime_covariance": dprime_covariance,
        "n_repeat": n_repeat,
        "n_shuffle": n_shuffle,
        "per_bin_dprime_mean_by_contrast": dprime,
        "per_bin_shuffle_z_by_contrast": shuffle_z,
        "bin_count_by_contrast": dict(CONTRAST_BIN_COUNTS),
        "bin_labels_by_contrast": {
            name: list(labels)
            for name, labels in CONTRAST_BIN_LABELS.items()
        },
        "contrast_names": list(CONTRAST_NAMES),
        "full_result": {
            "embeddings": np.empty((0, 0, 56, 0), dtype=np.float32),
            "selected_neuron_indices": np.empty((0, 0), dtype=int),
            "embedding_seeds": np.empty(0, dtype=np.int64),
            "shuffle_seeds": np.empty((0, 0), dtype=np.int64),
            "real_dprime_by_contrast": real_dprime,
            "shuffle_dprime_by_contrast": shuffle_dprime,
            "valid_trial_indices": np.empty(0, dtype=int),
            "pair_indices": {
                "left": np.empty((0, 2), dtype=int),
                "right": np.empty((0, 2), dtype=int),
            },
        },
    }

def compute_manifold_quantification_for_subject(
    load_path,
    mouse_name,
    num_day=7,
    target_neuron_count=DEFAULT_TARGET_NEURON_COUNT,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    n_repeat=100,
    n_shuffle=100,
    seed=2026,
    reducer=DEFAULT_REDUCER,
    min_finite_fraction=0.80,
    save_cache=True,
    cache_filename=None,
    show_progress=False,
    progress_position=0,
    save_full_cache=True,
    zscore=True,
    dprime_covariance=DEFAULT_DPRIME_COVARIANCE,
):
    """Compute and optionally save a reproducible manifold cache for one mouse."""
    load_path = Path(load_path)
    snarm_path = load_path / "snarm_1d.npy"
    labels_path = load_path / "trial_labels.npy"
    if not (snarm_path.exists() and labels_path.exists()):
        return None

    snarm_1d = np.load(snarm_path, allow_pickle=True)
    trial_labels = np.load(labels_path, allow_pickle=True)

    # Load optional segment/context files
    segment_ids = None
    is_context_trial = None
    for candidate in (load_path / "trial_segment_ids.npy",
                       load_path / "trial_segment_ids_auto.npy"):
        if candidate.exists():
            segment_ids = np.load(candidate, allow_pickle=True)
            break
    for candidate in (load_path / "is_context_trial.npy",
                       load_path / "is_context_trial_auto.npy"):
        if candidate.exists():
            is_context_trial = np.load(candidate, allow_pickle=True)
            break

    num_day = int(num_day)
    num_day_min = min(num_day, len(snarm_1d), len(trial_labels))
    target_neuron_count = int(target_neuron_count)
    embedding_dim = int(embedding_dim)
    reducer = _normalize_reducer(reducer)
    zscore = bool(zscore)
    dprime_covariance = normalize_dprime_covariance(dprime_covariance)
    cache_filename = cache_filename or cache_filename_for_dprime_covariance(
        dprime_covariance,
        zscore=zscore,
    )
    full_cache_filename = f"{Path(cache_filename).stem}_full.npz"
    day_results = np.empty(num_day_min, dtype=object)
    day_results[:] = None
    full_day_results = np.empty(num_day_min, dtype=object)
    full_day_results[:] = None
    cache = {
        "mouse_name": mouse_name,
        "neuron_mode": f"all{target_neuron_count}",
        "neuron_selection": "all",
        "target_neuron_count": target_neuron_count,
        "embedding_dim": embedding_dim,
        "reducer": reducer,
        "zscore": zscore,
        "dprime_covariance": dprime_covariance,
        "n_repeat": int(n_repeat),
        "n_shuffle": int(n_shuffle),
        "seed": int(seed),
        "min_finite_fraction": float(min_finite_fraction),
        "contrast_names": list(CONTRAST_NAMES),
        "bin_count_by_contrast": dict(CONTRAST_BIN_COUNTS),
        "bin_labels_by_contrast": {
            name: list(labels)
            for name, labels in CONTRAST_BIN_LABELS.items()
        },
        "full_cache_filename": full_cache_filename,
        "valid_day": np.zeros(num_day_min, dtype=bool),
        "n_lc": np.zeros(num_day_min, dtype=int),
        "n_rc": np.zeros(num_day_min, dtype=int),
        "n_pl": np.zeros(num_day_min, dtype=int),
        "n_cl": np.zeros(num_day_min, dtype=int),
        "n_pr": np.zeros(num_day_min, dtype=int),
        "n_cr": np.zeros(num_day_min, dtype=int),
        "candidate_neuron_count": np.full(num_day_min, np.nan),
        "selected_neuron_count": np.full(num_day_min, np.nan),
        "per_bin_dprime_mean_by_contrast": {
            name: np.full((num_day_min, CONTRAST_BIN_COUNTS[name]), np.nan, dtype=np.float32)
            for name in CONTRAST_NAMES
        },
        "per_bin_shuffle_z_by_contrast": {
            name: np.full((num_day_min, CONTRAST_BIN_COUNTS[name]), np.nan, dtype=np.float32)
            for name in CONTRAST_NAMES
        },
        "day_results": day_results,
    }

    day_iter = _progress(
        range(num_day_min),
        enabled=show_progress,
        desc=f"{mouse_name} days",
        total=num_day_min,
        position=progress_position,
        leave=False,
    )
    for day_idx in day_iter:
        if should_skip_mouse_day(day_idx, mouse_name):
            continue
        if day_idx >= len(snarm_1d) or day_idx >= len(trial_labels):
            continue

        day_seg = segment_ids[day_idx] if segment_ids is not None and day_idx < len(segment_ids) else None
        day_ctx = is_context_trial[day_idx] if is_context_trial is not None and day_idx < len(is_context_trial) else None

        day_seed = int(seed) + day_idx * 100003
        result = compute_day_manifold_quantification(
            snarm_day=snarm_1d[day_idx],
            labels_day=trial_labels[day_idx],
            target_neuron_count=target_neuron_count,
            embedding_dim=embedding_dim,
            n_repeat=n_repeat,
            n_shuffle=n_shuffle,
            seed=day_seed,
            reducer=reducer,
            min_finite_fraction=min_finite_fraction,
            show_progress=show_progress,
            progress_desc=f"{mouse_name} day {day_idx + 1}",
            progress_position=progress_position + 1,
            segment_ids=day_seg,
            is_context_trial=day_ctx,
            zscore=zscore,
            dprime_covariance=dprime_covariance,
        )
        full_day_results[day_idx] = result.get("full_result")
        light_result = dict(result)
        light_result.pop("full_result", None)
        day_results[day_idx] = light_result

        cache["n_lc"][day_idx] = result["n_lc"]
        cache["n_rc"][day_idx] = result["n_rc"]
        cache["n_pl"][day_idx] = result["n_pl"]
        cache["n_cl"][day_idx] = result["n_cl"]
        cache["n_pr"][day_idx] = result["n_pr"]
        cache["n_cr"][day_idx] = result["n_cr"]
        cache["candidate_neuron_count"][day_idx] = result["candidate_neuron_count"]
        if result["valid"]:
            cache["selected_neuron_count"][day_idx] = target_neuron_count
        if not result["valid"]:
            continue
        cache["valid_day"][day_idx] = True
        for contrast_name in CONTRAST_NAMES:
            cache["per_bin_dprime_mean_by_contrast"][contrast_name][day_idx] = (
                result["per_bin_dprime_mean_by_contrast"][contrast_name]
            )
            cache["per_bin_shuffle_z_by_contrast"][contrast_name][day_idx] = (
                result["per_bin_shuffle_z_by_contrast"][contrast_name]
            )

    if save_cache:
        np.save(load_path / cache_filename, cache, allow_pickle=True)
        if save_full_cache:
            np.savez_compressed(
                load_path / full_cache_filename,
                day_results=full_day_results,
                contrast_names=np.asarray(CONTRAST_NAMES, dtype=object),
                bin_count_by_contrast=np.asarray([dict(CONTRAST_BIN_COUNTS)], dtype=object),
                bin_labels_by_contrast=np.asarray([cache["bin_labels_by_contrast"]], dtype=object),
            )
    return cache

def manifold_cache_to_long_rows(cache, group=None, num_day=None, include_invalid=False):
    """Convert one manifold cache dict to long-form metric rows.

    Supports variable-bin, fixed per-bin, and legacy aggregate cache formats.
    """
    mouse_name = cache["mouse_name"]
    contrast_names = list(cache["contrast_names"])
    valid_day = np.asarray(cache["valid_day"], dtype=bool)
    if num_day is None:
        num_day = valid_day.size
    n_contrast = len(contrast_names)

    per_bin_by_contrast = cache.get("per_bin_dprime_mean_by_contrast", None)
    if per_bin_by_contrast is not None:
        per_bin_z_by_contrast = cache.get("per_bin_shuffle_z_by_contrast", {})
        nd = min(int(num_day), valid_day.size)
        rows = []
        for day_idx in range(nd):
            if not include_invalid and not valid_day[day_idx]:
                continue
            for contrast_name in contrast_names:
                bin_vals = np.asarray(per_bin_by_contrast[contrast_name][day_idx], dtype=float)
                bin_z = np.asarray(per_bin_z_by_contrast[contrast_name][day_idx], dtype=float)
                finite = np.isfinite(bin_vals)
                finite_z = np.isfinite(bin_z)
                rows.append({
                    "group": group,
                    "mouse": mouse_name,
                    "day": day_idx + 1,
                    "day_idx": day_idx,
                    "contrast": contrast_name,
                    "valid_day": bool(valid_day[day_idx]),
                    "candidate_neuron_count": _array_day_value(
                        cache.get("candidate_neuron_count", np.nan), day_idx,
                    ),
                    "selected_neuron_count": _array_day_value(
                        cache.get("selected_neuron_count", np.nan), day_idx,
                    ),
                    "embedding_dim": int(cache.get("embedding_dim", DEFAULT_EMBEDDING_DIM)),
                    "target_neuron_count": int(
                        cache.get("target_neuron_count", DEFAULT_TARGET_NEURON_COUNT)
                    ),
                    "reducer": cache.get("reducer", DEFAULT_REDUCER),
                    "zscore": bool(cache.get("zscore", False)),
                    "dprime_covariance": cache.get(
                        "dprime_covariance",
                        DPRIME_COVARIANCE_LEGACY_POOLED,
                    ),
                    "n_repeat": int(cache.get("n_repeat", 0)),
                    "n_shuffle": int(cache.get("n_shuffle", 0)),
                    "n_bins": int(bin_vals.size),
                    "dprime_mean": float(np.mean(bin_vals[finite])) if finite.any() else np.nan,
                    "dprime_std": float(np.std(bin_vals[finite], ddof=1)) if finite.sum() > 1 else np.nan,
                    "shuffle_z": float(np.mean(bin_z[finite_z])) if finite_z.any() else np.nan,
                })
        return rows

    # Fixed per-bin format fallback.
    per_bin = cache.get("per_bin_dprime_mean", None)
    if per_bin is not None:
        # New format: per_bin_dprime_mean (n_days, 56, 3), per_bin_shuffle_z (n_days, 56, 3)
        per_bin = np.asarray(per_bin, dtype=float)
        per_bin_z = np.asarray(cache.get("per_bin_shuffle_z", np.full_like(per_bin, np.nan)), dtype=float)
        nd = min(int(num_day), per_bin.shape[0])
        rows = []
        for day_idx in range(nd):
            if not include_invalid and not valid_day[day_idx]:
                continue
            for contrast_idx, contrast_name in enumerate(contrast_names):
                bin_vals = per_bin[day_idx, :, contrast_idx]
                bin_z = per_bin_z[day_idx, :, contrast_idx]
                finite = np.isfinite(bin_vals)
                finite_z = np.isfinite(bin_z)
                rows.append({
                    "group": group,
                    "mouse": mouse_name,
                    "day": day_idx + 1,
                    "day_idx": day_idx,
                    "contrast": contrast_name,
                    "valid_day": bool(valid_day[day_idx]),
                    "candidate_neuron_count": _array_day_value(
                        cache.get("candidate_neuron_count", np.nan), day_idx,
                    ),
                    "selected_neuron_count": _array_day_value(
                        cache.get("selected_neuron_count", np.nan), day_idx,
                    ),
                    "embedding_dim": int(cache.get("embedding_dim", DEFAULT_EMBEDDING_DIM)),
                    "target_neuron_count": int(
                        cache.get("target_neuron_count", DEFAULT_TARGET_NEURON_COUNT)
                    ),
                    "reducer": cache.get("reducer", DEFAULT_REDUCER),
                    "zscore": bool(cache.get("zscore", False)),
                    "dprime_covariance": cache.get(
                        "dprime_covariance",
                        DPRIME_COVARIANCE_LEGACY_POOLED,
                    ),
                    "n_repeat": int(cache.get("n_repeat", 0)),
                    "n_shuffle": int(cache.get("n_shuffle", 0)),
                    # Aggregate metrics from per-bin data
                    "dprime_mean": float(np.mean(bin_vals[finite])) if finite.any() else np.nan,
                    "dprime_std": float(np.std(bin_vals[finite], ddof=1)) if finite.sum() > 1 else np.nan,
                    "shuffle_z": float(np.mean(bin_z[finite_z])) if finite_z.any() else np.nan,
                })
        return rows

    # Legacy format fallback
    metric_arrays = {
        key: np.asarray(cache[key], dtype=float)
        for key in [
            "dprime_mean",
            "dprime_std",
            "shuffle_mean",
            "shuffle_std",
            "shuffle_p95",
            "shuffle_z",
        ]
        if key in cache
    }
    rows = []
    for day_idx in range(min(int(num_day), valid_day.size)):
        if not include_invalid and not valid_day[day_idx]:
            continue
        for contrast_idx, contrast_name in enumerate(contrast_names):
            row = {
                "group": group,
                "mouse": mouse_name,
                "day": day_idx + 1,
                "day_idx": day_idx,
                "contrast": contrast_name,
                "valid_day": bool(valid_day[day_idx]),
                "candidate_neuron_count": _array_day_value(
                    cache.get("candidate_neuron_count", np.nan),
                    day_idx,
                ),
                "selected_neuron_count": _array_day_value(
                    cache.get("selected_neuron_count", np.nan),
                    day_idx,
                ),
                "embedding_dim": int(cache.get("embedding_dim", DEFAULT_EMBEDDING_DIM)),
                "target_neuron_count": int(
                    cache.get("target_neuron_count", DEFAULT_TARGET_NEURON_COUNT)
                ),
                "reducer": cache.get("reducer", DEFAULT_REDUCER),
                "zscore": bool(cache.get("zscore", False)),
                "dprime_covariance": cache.get(
                    "dprime_covariance",
                    DPRIME_COVARIANCE_LEGACY_POOLED,
                ),
                "n_repeat": int(cache.get("n_repeat", 0)),
                "n_shuffle": int(cache.get("n_shuffle", 0)),
            }
            for metric_name, values in metric_arrays.items():
                row[metric_name] = float(values[day_idx, contrast_idx])
            rows.append(row)
    return rows

def _zscore_columns(points):
    """Z-score a point-by-neuron matrix without changing the selected columns."""
    points = np.asarray(points, dtype=float)
    mean = np.nanmean(points, axis=0, keepdims=True)
    std = np.nanstd(points, axis=0, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    z = (points - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

def fit_embedding(points, embedding_dim, reducer=DEFAULT_REDUCER, seed=2026):
    """Fit a deterministic reducer to the selected point-by-neuron matrix."""
    reducer = _normalize_reducer(reducer)
    points = np.asarray(points, dtype=float)
    if reducer == "spectral":
        n_neighbors = min(10, max(points.shape[0] - 1, 1))
        model = SpectralEmbedding(
            n_components=int(embedding_dim),
            n_neighbors=n_neighbors,
            n_jobs=1,
            random_state=int(seed),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not find the number of physical cores.*",
                category=UserWarning,
            )
            return model.fit_transform(points)
    if reducer == "pca":
        model = PCA(n_components=int(embedding_dim), random_state=int(seed))
        return model.fit_transform(points)
    raise ValueError(f"Unsupported reducer {reducer!r}")

def _precision_by_bin(class_a, class_b, dprime_covariance=DEFAULT_DPRIME_COVARIANCE):
    dprime_covariance = normalize_dprime_covariance(dprime_covariance)
    class_a = np.asarray(class_a, dtype=float)
    class_b = np.asarray(class_b, dtype=float)
    return [
        _precision_for_classes(
            class_a[:, bin_i, :],
            class_b[:, bin_i, :],
            dprime_covariance=dprime_covariance,
        )
        for bin_i in range(class_a.shape[1])
    ]

def _precision_for_classes(class_a, class_b, dprime_covariance=DEFAULT_DPRIME_COVARIANCE):
    covariance = normalize_dprime_covariance(dprime_covariance)
    if covariance == DPRIME_COVARIANCE_LEGACY_POOLED:
        return _pooled_precision(np.vstack([class_a, class_b]))
    if covariance == DPRIME_COVARIANCE_WITHIN_CLASS:
        return _within_class_precision(class_a, class_b)
    raise AssertionError(f"Unhandled covariance estimator {covariance!r}")

def _pooled_precision(combined):
    combined = _finite_rows(combined)
    if combined.shape[0] < 2 or combined.shape[1] == 0:
        return None
    if combined.shape[0] <= combined.shape[1]:
        covariance = np.cov(combined, rowvar=False)
    else:
        covariance = LedoitWolf().fit(combined).covariance_
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    covariance += np.eye(covariance.shape[0]) * 1e-9
    precision = np.linalg.pinv(covariance)
    if not np.isfinite(precision).all():
        return None
    return precision

def _within_class_precision(class_a, class_b):
    a = _finite_rows(class_a)
    b = _finite_rows(class_b)
    if a.shape[0] < 2 or b.shape[0] < 2 or a.shape[1] == 0 or a.shape[1] != b.shape[1]:
        return None
    residuals = np.vstack(
        [
            a - np.nanmean(a, axis=0, keepdims=True),
            b - np.nanmean(b, axis=0, keepdims=True),
        ]
    )
    residuals = _finite_rows(residuals)
    if residuals.shape[0] < 2:
        return None
    covariance = LedoitWolf().fit(residuals).covariance_
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    covariance += np.eye(covariance.shape[0]) * 1e-9
    precision = np.linalg.pinv(covariance)
    if not np.isfinite(precision).all():
        return None
    return precision

def _d_prime_from_precision(class_a, class_b, precision):
    if precision is None:
        return np.nan
    a = _finite_rows(class_a)
    b = _finite_rows(class_b)
    if a.shape[0] < 2 or b.shape[0] < 2 or a.shape[1] != b.shape[1]:
        return np.nan
    delta = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    value = float(delta @ precision @ delta.T)
    if not np.isfinite(value):
        return np.nan
    return float(np.sqrt(max(value, 0.0)))

def _progress(iterable, enabled=False, **kwargs):
    if not enabled:
        return iterable
    return tqdm(iterable, **kwargs)

def _normalize_reducer(reducer):
    key = str(reducer).lower().replace("-", "_")
    aliases = {
        "spectral": "spectral",
        "spectral_embedding": "spectral",
        "spectralembedding": "spectral",
        "pca": "pca",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported reducer {reducer!r}. Expected one of: spectral, pca."
        )
    return aliases[key]

def _finite_rows(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        return np.empty((0, 0))
    return values[np.isfinite(values).all(axis=1)]

def _array_day_value(values, day_idx):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return float(values)
    if day_idx >= values.shape[0]:
        return np.nan
    return float(values[day_idx])

def load_manifold_per_bin_data(
    mouse_names,
    base_path="../8-maze/Aligned_data",
    cache_filename=DEFAULT_CACHE_FILENAME,
    num_day=7,
):
    """Load variable-bin d-prime + shuffle_z arrays per contrast."""
    base_path = Path(base_path)
    num_day = int(num_day)

    ordered_groups = [group for group in GROUP_ORDER if group in mouse_names]
    ordered_groups.extend(group for group in mouse_names if group not in ordered_groups)
    mouse_list = []
    for group in ordered_groups:
        for mouse_name in mouse_names.get(group, []):
            mouse_list.append((group, mouse_name))
    n_mice = len(mouse_list)

    results = {}
    for contrast_name in CONTRAST_NAMES:
        n_bins = CONTRAST_BIN_COUNTS[contrast_name]
        results[contrast_name] = {
            "data": np.full((n_mice, num_day, n_bins), np.nan, dtype=np.float32),
            "shuffle_z": np.full((n_mice, num_day, n_bins), np.nan, dtype=np.float32),
            "n_bins": n_bins,
            "bin_labels": list(CONTRAST_BIN_LABELS[contrast_name]),
        }

    for mouse_idx, (_group, mouse_name) in enumerate(mouse_list):
        cache_path = base_path / mouse_name / cache_filename
        if not cache_path.exists():
            continue
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
        except (OSError, EOFError, ValueError):
            continue

        nd = min(num_day, np.asarray(cache.get("valid_day", np.zeros(0))).size)
        valid = np.asarray(cache.get("valid_day", np.zeros(0)), dtype=bool)[:nd]
        for day_idx in range(nd):
            if not valid[day_idx]:
                continue
            if "per_bin_dprime_mean_by_contrast" in cache:
                for contrast_name in CONTRAST_NAMES:
                    dvals = np.asarray(
                        cache["per_bin_dprime_mean_by_contrast"][contrast_name][day_idx],
                        dtype=np.float32,
                    )
                    zvals = np.asarray(
                        cache["per_bin_shuffle_z_by_contrast"][contrast_name][day_idx],
                        dtype=np.float32,
                    )
                    n = min(results[contrast_name]["n_bins"], dvals.size)
                    results[contrast_name]["data"][mouse_idx, day_idx, :n] = dvals[:n]
                    results[contrast_name]["shuffle_z"][mouse_idx, day_idx, :n] = zvals[:n]
            elif "per_bin_dprime_mean" in cache:
                dvals = np.asarray(cache["per_bin_dprime_mean"][day_idx], dtype=np.float32)
                zvals = np.asarray(cache["per_bin_shuffle_z"][day_idx], dtype=np.float32)
                for contrast_idx, contrast_name in enumerate(CONTRAST_NAMES):
                    n = min(results[contrast_name]["n_bins"], dvals.shape[0])
                    results[contrast_name]["data"][mouse_idx, day_idx, :n] = dvals[:n, contrast_idx]
                    results[contrast_name]["shuffle_z"][mouse_idx, day_idx, :n] = zvals[:n, contrast_idx]

    results["mouse_list"] = mouse_list
    return results
