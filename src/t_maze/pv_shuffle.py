"""Manuscript definitions collected from pv_shuffle.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import warnings

from pathlib import Path

from typing import Mapping, Sequence

import numpy as np

from scipy.stats import rankdata

from .manifold import (
    CONDITION_ALIASES,
    CONDITION_LABELS,
    CONDITION_ORDER,
    CONTRAST_LABELS,
    CONTRAST_SPECS,
    DEFAULT_MIN_TRIALS_PER_CONDITION,
    DEFAULT_TARGET_NEURON_COUNT,
    DPRIME_COVARIANCE,
    PAIR_CONTRAST_SPECS,
    InferenceSpec,
    _as_trial_bin_neuron,
    _balanced_sample_choice_classes,
    _d_prime_from_precision,
    _precision_by_bin,
    _sample_vs_choice_balance_count,
    _shuffle_rows,
    _shuffle_sample_choice_within_side,
    _validated_condition_trials,
    _zscore_columns,
    build_group_day_summary,
    build_spatial_bin_long_table,
    build_spatial_section_long_table,
    build_tmaze_manifold_condition_trials,
    discover_group_mice,
    finalize_inference_results,
    fit_longitudinal_effects_by,
    tmaze_path_sections,
)

DEFAULT_CACHE_FILENAME = "tmaze_pv_shuffle_50all_zscore.npy"

LEGACY_CACHE_FILENAME = "tmaze_pv_shuffle_50all.npy"

DEFAULT_METRICS = ("spearman", "cosine", "dprime")

DEFAULT_N_REPEAT = 100

DEFAULT_N_SHUFFLE = 100

DEFAULT_MIN_FINITE_FRACTION = 0.80

def compute_day_pv_shuffle_control(
    condition_trials: Mapping[str, np.ndarray],
    *,
    target_neuron_count: int = DEFAULT_TARGET_NEURON_COUNT,
    n_repeat: int = DEFAULT_N_REPEAT,
    n_shuffle: int = DEFAULT_N_SHUFFLE,
    seed: int = 2026,
    min_finite_fraction: float = DEFAULT_MIN_FINITE_FRACTION,
    min_trials_per_condition: int = DEFAULT_MIN_TRIALS_PER_CONDITION,
    zscore: bool = True,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict:
    """Compute per-bin PV shuffle controls for one mouse-day.

    ``spearman`` and ``cosine`` compare condition-mean population vectors at
    each spatial bin. ``dprime`` uses the original selected-neuron space as a
    high-dimensional control for the manifold d-prime analysis.
    """
    metrics = _normalize_metrics(metrics)
    target_neuron_count = int(target_neuron_count)
    n_repeat = int(n_repeat)
    n_shuffle = int(n_shuffle)
    min_trials_per_condition = int(min_trials_per_condition)

    trials = _validated_condition_trials(condition_trials)
    first = trials[CONDITION_ORDER[0]]
    n_bin = int(first.shape[1])
    counts = {f"n_{label.lower()}": int(trials[label].shape[0]) for label in CONDITION_ORDER}
    sample_vs_choice_balance_count = _sample_vs_choice_balance_count(trials)
    sample_vs_choice_class_count = 2 * sample_vs_choice_balance_count

    valid_contrasts = {
        contrast: (
            trials[a].shape[0] >= min_trials_per_condition
            and trials[b].shape[0] >= min_trials_per_condition
        )
        for contrast, (a, b) in PAIR_CONTRAST_SPECS.items()
    }
    valid_contrasts["sample_vs_choice"] = (
        sample_vs_choice_balance_count >= min_trials_per_condition
    )

    if not any(valid_contrasts.values()):
        result = _empty_day_result(n_bin, n_repeat, n_shuffle, metrics=metrics, zscore=zscore)
        result.update(counts)
        result["sample_vs_choice_balance_count"] = sample_vs_choice_balance_count
        result["sample_vs_choice_class_count"] = sample_vs_choice_class_count
        return result

    condition_slices = {}
    start = 0
    pieces = []
    for label in CONDITION_ORDER:
        values = trials[label]
        stop = start + values.shape[0]
        condition_slices[label] = slice(start, stop)
        pieces.append(values)
        start = stop
    pooled = np.concatenate(pieces, axis=0)
    n_total_trials = int(pooled.shape[0])

    finite_count = np.isfinite(pooled).sum(axis=(0, 1))
    keep = finite_count >= (n_total_trials * n_bin * float(min_finite_fraction))
    candidate_indices = np.where(keep)[0]
    candidate_neuron_count = int(candidate_indices.size)

    if candidate_neuron_count < target_neuron_count:
        result = _empty_day_result(n_bin, n_repeat, n_shuffle, metrics=metrics, zscore=zscore)
        result.update(counts)
        result["sample_vs_choice_balance_count"] = sample_vs_choice_balance_count
        result["sample_vs_choice_class_count"] = sample_vs_choice_class_count
        result["candidate_neuron_count"] = candidate_neuron_count
        result["target_neuron_count"] = target_neuron_count
        return result

    pooled_candidates = pooled[:, :, candidate_indices]
    if zscore:
        pooled_candidates = _zscore_columns(
            pooled_candidates.reshape(n_total_trials * n_bin, candidate_neuron_count)
        ).reshape(n_total_trials, n_bin, candidate_neuron_count)

    per_bin_real = _empty_metric_contrast_arrays(metrics, (n_repeat, n_bin))
    per_bin_shuffle = _empty_metric_contrast_arrays(metrics, (n_repeat, n_shuffle, n_bin))
    selected_neuron_indices = np.full((n_repeat, target_neuron_count), -1, dtype=int)
    shuffle_seeds = np.full((n_repeat, n_shuffle), -1, dtype=np.int64)
    sample_vs_choice_balance_counts = np.zeros(n_repeat, dtype=int)

    rng = np.random.default_rng(seed)
    for repeat_idx in range(n_repeat):
        selected = rng.choice(candidate_indices, size=target_neuron_count, replace=False)
        selected_neuron_indices[repeat_idx] = selected.astype(int)
        selected_pool = np.searchsorted(candidate_indices, selected)
        selected_values = pooled_candidates[:, :, selected_pool]
        selected_by_condition = {
            label: selected_values[condition_slices[label]]
            for label in CONDITION_ORDER
        }

        precision_by_contrast = {}
        svc_selected = None
        svc_precision = None

        for contrast, (label_a, label_b) in PAIR_CONTRAST_SPECS.items():
            if not valid_contrasts[contrast]:
                continue
            class_a = selected_by_condition[label_a]
            class_b = selected_by_condition[label_b]
            precision = _precision_by_bin(class_a, class_b)
            precision_by_contrast[contrast] = precision
            values = _metric_values_by_bin(class_a, class_b, precision, metrics)
            for metric in metrics:
                per_bin_real[metric][contrast][repeat_idx, :] = values[metric]

        if valid_contrasts["sample_vs_choice"]:
            class_sample, class_choice, balance_count, svc_selected = _balanced_sample_choice_classes(
                selected_by_condition,
                rng,
            )
            sample_vs_choice_balance_counts[repeat_idx] = int(balance_count)
            svc_precision = _precision_by_bin(class_sample, class_choice)
            precision_by_contrast["sample_vs_choice"] = svc_precision
            values = _metric_values_by_bin(class_sample, class_choice, svc_precision, metrics)
            for metric in metrics:
                per_bin_real[metric]["sample_vs_choice"][repeat_idx, :] = values[metric]

        shuf_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n_shuffle, dtype=np.uint32)
        shuffle_seeds[repeat_idx, :] = shuf_seeds.astype(np.int64)
        for contrast, (label_a, label_b) in PAIR_CONTRAST_SPECS.items():
            if not valid_contrasts[contrast]:
                continue
            values = _shuffle_pair_metric_values_many(
                selected_by_condition[label_a],
                selected_by_condition[label_b],
                precision_by_contrast[contrast],
                metrics,
                shuf_seeds,
            )
            for metric in metrics:
                per_bin_shuffle[metric][contrast][repeat_idx, :, :] = values[metric]

        if valid_contrasts["sample_vs_choice"] and svc_selected is not None:
            values = _shuffle_sample_choice_metric_values_many(
                svc_selected,
                svc_precision,
                metrics,
                shuf_seeds,
            )
            for metric in metrics:
                per_bin_shuffle[metric]["sample_vs_choice"][repeat_idx, :, :] = values[metric]

    summarized = _summarize_real_and_shuffle(per_bin_real, per_bin_shuffle, metrics, n_bin)
    return {
        "valid": True,
        **counts,
        "n_total": n_total_trials,
        "sample_vs_choice_balance_count": sample_vs_choice_balance_count,
        "sample_vs_choice_class_count": sample_vs_choice_class_count,
        "candidate_neuron_count": candidate_neuron_count,
        "selected_neuron_count": target_neuron_count,
        "target_neuron_count": target_neuron_count,
        "zscore": bool(zscore),
        "dprime_covariance": DPRIME_COVARIANCE,
        "n_repeat": n_repeat,
        "n_shuffle": n_shuffle,
        "metrics": tuple(metrics),
        "contrast_names": list(CONTRAST_SPECS),
        "contrast_labels": dict(CONTRAST_LABELS),
        "bin_count_by_contrast": {name: n_bin for name in CONTRAST_SPECS},
        **summarized,
        "full_result": {
            "real_by_metric_contrast": per_bin_real,
            "shuffle_by_metric_contrast": per_bin_shuffle,
            "selected_neuron_indices": selected_neuron_indices,
            "shuffle_seeds": shuffle_seeds,
            "condition_slices": condition_slices,
            "sample_vs_choice_balance_counts": sample_vs_choice_balance_counts,
        },
    }

def compute_pv_shuffle_control_for_subject(
    subject_dir: str | Path,
    *,
    mouse_name: str | None = None,
    group: str | None = None,
    num_day: int | None = None,
    target_neuron_count: int = DEFAULT_TARGET_NEURON_COUNT,
    n_repeat: int = DEFAULT_N_REPEAT,
    n_shuffle: int = DEFAULT_N_SHUFFLE,
    seed: int = 2026,
    min_finite_fraction: float = DEFAULT_MIN_FINITE_FRACTION,
    min_trials_per_condition: int = DEFAULT_MIN_TRIALS_PER_CONDITION,
    condition_aliases: Mapping[str, str] | None = None,
    save_cache: bool = True,
    cache_filename: str | None = None,
    save_full_cache: bool = False,
    zscore: bool = True,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict | None:
    """Compute and optionally save a PV shuffle cache for one T-maze subject."""
    subject = Path(subject_dir)
    mouse_name = subject.name if mouse_name is None else str(mouse_name)
    aliases = dict(CONDITION_ALIASES if condition_aliases is None else condition_aliases)
    metrics = _normalize_metrics(metrics)
    cache_filename = cache_filename or (DEFAULT_CACHE_FILENAME if zscore else LEGACY_CACHE_FILENAME)
    trial_path = subject / "Snarm_trial_v2.npy"
    index_path = subject / "index_all.npy"
    if not (trial_path.exists() and index_path.exists()):
        return None

    trial_maps = np.load(trial_path, allow_pickle=True).item()
    sample_days = np.asarray(trial_maps["S"], dtype=object)
    choice_days = np.asarray(trial_maps["C"], dtype=object)
    index_all = np.load(index_path, allow_pickle=True)
    max_day = min(sample_days.shape[0], choice_days.shape[0], np.asarray(index_all).shape[1])
    if num_day is not None:
        max_day = min(max_day, int(num_day))
    if max_day <= 0:
        return None

    first_sample = _as_trial_bin_neuron(sample_days[0], "Snarm_trial_v2['S'][0]")
    n_bin = int(first_sample.shape[1])
    day_results = np.empty(max_day, dtype=object)
    day_results[:] = None
    full_day_results = np.empty(max_day, dtype=object)
    full_day_results[:] = None

    cache = {
        "group": group,
        "mouse_name": mouse_name,
        "neuron_mode": f"all{int(target_neuron_count)}",
        "neuron_selection": "all",
        "condition_aliases": aliases,
        "condition_labels": dict(CONDITION_LABELS),
        "target_neuron_count": int(target_neuron_count),
        "zscore": bool(zscore),
        "dprime_covariance": DPRIME_COVARIANCE,
        "n_repeat": int(n_repeat),
        "n_shuffle": int(n_shuffle),
        "seed": int(seed),
        "min_finite_fraction": float(min_finite_fraction),
        "min_trials_per_condition": int(min_trials_per_condition),
        "metrics": tuple(metrics),
        "contrast_names": list(CONTRAST_SPECS),
        "contrast_labels": dict(CONTRAST_LABELS),
        "bin_count_by_contrast": {name: n_bin for name in CONTRAST_SPECS},
        "valid_day": np.zeros(max_day, dtype=bool),
        "n_sl": np.zeros(max_day, dtype=int),
        "n_sr": np.zeros(max_day, dtype=int),
        "n_cl": np.zeros(max_day, dtype=int),
        "n_cr": np.zeros(max_day, dtype=int),
        "sample_vs_choice_balance_count": np.zeros(max_day, dtype=int),
        "sample_vs_choice_class_count": np.zeros(max_day, dtype=int),
        "candidate_neuron_count": np.full(max_day, np.nan),
        "selected_neuron_count": np.full(max_day, np.nan),
        "per_bin_real_by_metric_contrast": _empty_metric_contrast_arrays(
            metrics, (max_day, n_bin)
        ),
        "per_bin_shuffle_mean_by_metric_contrast": _empty_metric_contrast_arrays(
            metrics, (max_day, n_bin)
        ),
        "per_bin_shuffle_std_by_metric_contrast": _empty_metric_contrast_arrays(
            metrics, (max_day, n_bin)
        ),
        "per_bin_shuffle_z_by_metric_contrast": _empty_metric_contrast_arrays(
            metrics, (max_day, n_bin)
        ),
        "per_bin_separation_z_by_metric_contrast": _empty_metric_contrast_arrays(
            metrics, (max_day, n_bin)
        ),
        "day_results": day_results,
    }

    for day_idx in range(max_day):
        condition_trials = build_tmaze_manifold_condition_trials(
            sample_days[day_idx],
            choice_days[day_idx],
            index_all,
            day_idx=day_idx,
            condition_aliases=aliases,
        )
        day_seed = int(seed) + day_idx * 100003
        result = compute_day_pv_shuffle_control(
            condition_trials,
            target_neuron_count=target_neuron_count,
            n_repeat=n_repeat,
            n_shuffle=n_shuffle,
            seed=day_seed,
            min_finite_fraction=min_finite_fraction,
            min_trials_per_condition=min_trials_per_condition,
            zscore=zscore,
            metrics=metrics,
        )
        full_day_results[day_idx] = result.get("full_result")
        light_result = dict(result)
        light_result.pop("full_result", None)
        day_results[day_idx] = light_result

        for label in CONDITION_ORDER:
            cache[f"n_{label.lower()}"][day_idx] = int(result.get(f"n_{label.lower()}", 0))
        cache["sample_vs_choice_balance_count"][day_idx] = int(
            result.get("sample_vs_choice_balance_count", 0)
        )
        cache["sample_vs_choice_class_count"][day_idx] = int(
            result.get("sample_vs_choice_class_count", 0)
        )
        cache["candidate_neuron_count"][day_idx] = result.get("candidate_neuron_count", np.nan)
        cache["selected_neuron_count"][day_idx] = result.get("selected_neuron_count", np.nan)
        if not result["valid"]:
            continue
        cache["valid_day"][day_idx] = True
        for key in (
            "per_bin_real_by_metric_contrast",
            "per_bin_shuffle_mean_by_metric_contrast",
            "per_bin_shuffle_std_by_metric_contrast",
            "per_bin_shuffle_z_by_metric_contrast",
            "per_bin_separation_z_by_metric_contrast",
        ):
            for metric in metrics:
                for contrast in CONTRAST_SPECS:
                    cache[key][metric][contrast][day_idx] = result[key][metric][contrast]

    if save_cache:
        np.save(subject / cache_filename, cache, allow_pickle=True)
        if save_full_cache:
            full_cache_filename = f"{Path(cache_filename).stem}_full.npz"
            np.savez_compressed(
                subject / full_cache_filename,
                day_results=full_day_results,
                contrast_names=np.asarray(list(CONTRAST_SPECS), dtype=object),
                metrics=np.asarray(metrics, dtype=object),
                bin_count_by_contrast=np.asarray([cache["bin_count_by_contrast"]], dtype=object),
            )
    return cache

def _normalize_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(metric).lower() for metric in metrics)
    allowed = {"spearman", "cosine", "dprime"}
    unknown = [metric for metric in normalized if metric not in allowed]
    if unknown:
        raise ValueError(f"unknown PV shuffle metric(s): {unknown}")
    return normalized

def _empty_metric_contrast_arrays(metrics: Sequence[str], shape: tuple[int, ...]) -> dict:
    return {
        metric: {
            contrast: np.full(shape, np.nan, dtype=np.float32)
            for contrast in CONTRAST_SPECS
        }
        for metric in metrics
    }

def _empty_day_result(
    n_bin: int,
    n_repeat: int,
    n_shuffle: int,
    *,
    metrics: Sequence[str],
    zscore: bool = True,
) -> dict:
    metric_arrays = _empty_metric_contrast_arrays(metrics, (int(n_bin),))
    return {
        "valid": False,
        "n_sl": 0,
        "n_sr": 0,
        "n_cl": 0,
        "n_cr": 0,
        "n_total": 0,
        "sample_vs_choice_balance_count": 0,
        "sample_vs_choice_class_count": 0,
        "candidate_neuron_count": np.nan,
        "selected_neuron_count": np.nan,
        "target_neuron_count": DEFAULT_TARGET_NEURON_COUNT,
        "zscore": bool(zscore),
        "dprime_covariance": DPRIME_COVARIANCE,
        "n_repeat": int(n_repeat),
        "n_shuffle": int(n_shuffle),
        "metrics": tuple(metrics),
        "contrast_names": list(CONTRAST_SPECS),
        "contrast_labels": dict(CONTRAST_LABELS),
        "bin_count_by_contrast": {name: int(n_bin) for name in CONTRAST_SPECS},
        "per_bin_real_by_metric_contrast": metric_arrays,
        "per_bin_shuffle_mean_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (int(n_bin),)),
        "per_bin_shuffle_std_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (int(n_bin),)),
        "per_bin_shuffle_z_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (int(n_bin),)),
        "per_bin_separation_z_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (int(n_bin),)),
        "full_result": {
            "real_by_metric_contrast": _empty_metric_contrast_arrays(
                metrics, (int(n_repeat), int(n_bin))
            ),
            "shuffle_by_metric_contrast": _empty_metric_contrast_arrays(
                metrics, (int(n_repeat), int(n_shuffle), int(n_bin))
            ),
            "sample_vs_choice_balance_counts": np.zeros(int(n_repeat), dtype=int),
        },
    }

def _metric_values_by_bin(
    class_a: np.ndarray,
    class_b: np.ndarray,
    precision_by_bin: Sequence[np.ndarray | None] | None,
    metrics: Sequence[str],
) -> dict[str, np.ndarray]:
    class_a = np.asarray(class_a, dtype=float)
    class_b = np.asarray(class_b, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_a = np.nanmean(class_a, axis=0)
        mean_b = np.nanmean(class_b, axis=0)
    values: dict[str, np.ndarray] = {}
    if "spearman" in metrics:
        values["spearman"] = _rowwise_corr(mean_a, mean_b, method="spearman").astype(np.float32)
    if "cosine" in metrics:
        values["cosine"] = _rowwise_cosine_similarity(mean_a, mean_b).astype(np.float32)
    if "dprime" in metrics:
        if precision_by_bin is None:
            precision_by_bin = _precision_by_bin(class_a, class_b)
        values["dprime"] = np.asarray(
            [
                _d_prime_from_precision(class_a[:, bin_idx, :], class_b[:, bin_idx, :], precision)
                for bin_idx, precision in enumerate(precision_by_bin)
            ],
            dtype=np.float32,
        )
    return values

def _shuffle_pair_metric_values_many(
    class_a: np.ndarray,
    class_b: np.ndarray,
    precision_by_bin: Sequence[np.ndarray | None] | None,
    metrics: Sequence[str],
    seeds: np.ndarray,
) -> dict[str, np.ndarray]:
    a = np.asarray(class_a, dtype=float)
    b = np.asarray(class_b, dtype=float)
    combined = np.concatenate([a, b], axis=0)
    n_a = a.shape[0]
    orders = np.asarray(
        [np.random.default_rng(int(seed)).permutation(combined.shape[0]) for seed in seeds],
        dtype=int,
    )
    a_s = combined[orders[:, :n_a]]
    b_s = combined[orders[:, n_a:]]
    return _metric_values_many(a_s, b_s, precision_by_bin, metrics)

def _shuffle_sample_choice_metric_values_many(
    selected_by_condition: Mapping[str, np.ndarray],
    precision_by_bin: Sequence[np.ndarray | None] | None,
    metrics: Sequence[str],
    seeds: np.ndarray,
) -> dict[str, np.ndarray]:
    selected = {label: np.asarray(selected_by_condition[label], dtype=float) for label in CONDITION_ORDER}
    balance_count = selected["SL"].shape[0]
    sample_parts = []
    choice_parts = []
    for side_idx, (sample_label, choice_label) in enumerate((("SL", "CL"), ("SR", "CR"))):
        combined = np.concatenate([selected[sample_label], selected[choice_label]], axis=0)
        orders = np.asarray(
            [
                np.random.default_rng(int(seed) + side_idx * 1000003).permutation(combined.shape[0])
                for seed in seeds
            ],
            dtype=int,
        )
        sample_parts.append(combined[orders[:, :balance_count]])
        choice_parts.append(combined[orders[:, balance_count:]])
    sample = np.concatenate(sample_parts, axis=1)
    choice = np.concatenate(choice_parts, axis=1)
    return _metric_values_many(sample, choice, precision_by_bin, metrics)

def _metric_values_many(
    class_a: np.ndarray,
    class_b: np.ndarray,
    precision_by_bin: Sequence[np.ndarray | None] | None,
    metrics: Sequence[str],
) -> dict[str, np.ndarray]:
    class_a = np.asarray(class_a, dtype=float)
    class_b = np.asarray(class_b, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_a = np.nanmean(class_a, axis=1)
        mean_b = np.nanmean(class_b, axis=1)
    values: dict[str, np.ndarray] = {}
    if "spearman" in metrics:
        values["spearman"] = _rowwise_corr(mean_a, mean_b, method="spearman").astype(np.float32)
    if "cosine" in metrics:
        values["cosine"] = _rowwise_cosine_similarity(mean_a, mean_b).astype(np.float32)
    if "dprime" in metrics:
        values["dprime"] = _dprime_from_mean_delta_many(mean_a - mean_b, precision_by_bin).astype(np.float32)
    return values

def _dprime_from_mean_delta_many(
    delta: np.ndarray,
    precision_by_bin: Sequence[np.ndarray | None] | None,
) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 3:
        raise ValueError(f"delta must have shape (shuffle, bin, neuron), got {delta.shape}")
    n_shuffle, n_bin, n_neuron = delta.shape
    precision_stack = np.full((n_bin, n_neuron, n_neuron), np.nan, dtype=float)
    if precision_by_bin is not None:
        for bin_idx, precision in enumerate(precision_by_bin):
            if precision is None:
                continue
            arr = np.asarray(precision, dtype=float)
            if arr.shape == (n_neuron, n_neuron) and np.isfinite(arr).all():
                precision_stack[bin_idx] = arr
    valid_bins = np.isfinite(precision_stack).all(axis=(1, 2))
    out = np.full((n_shuffle, n_bin), np.nan, dtype=float)
    if not valid_bins.any():
        return out
    values = np.einsum(
        "sbn,bnm,sbm->sb",
        delta[:, valid_bins, :],
        precision_stack[valid_bins],
        delta[:, valid_bins, :],
        optimize=True,
    )
    values[~np.isfinite(values)] = np.nan
    out[:, valid_bins] = np.sqrt(np.maximum(values, 0.0))
    return out

def _rowwise_corr(matrix_a: np.ndarray, matrix_b: np.ndarray, *, method: str) -> np.ndarray:
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"rowwise correlation inputs must match, got {a.shape} and {b.shape}")
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
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
        where=(n_valid >= 3) & (denominator > 0),
    )

def _rowwise_cosine_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"cosine inputs must match, got {a.shape} and {b.shape}")
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    valid = np.isfinite(a) & np.isfinite(b)
    a_masked = np.where(valid, a, 0.0)
    b_masked = np.where(valid, b, 0.0)
    numerator = np.sum(a_masked * b_masked, axis=-1)
    denominator = np.sqrt(np.sum(a_masked ** 2, axis=-1) * np.sum(b_masked ** 2, axis=-1))
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=(valid.sum(axis=-1) >= 3) & (denominator > 0),
    )

def _summarize_real_and_shuffle(
    real: Mapping[str, Mapping[str, np.ndarray]],
    shuffle: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Sequence[str],
    n_bin: int,
) -> dict:
    out = {
        "per_bin_real_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (n_bin,)),
        "per_bin_shuffle_mean_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (n_bin,)),
        "per_bin_shuffle_std_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (n_bin,)),
        "per_bin_shuffle_z_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (n_bin,)),
        "per_bin_separation_z_by_metric_contrast": _empty_metric_contrast_arrays(metrics, (n_bin,)),
    }
    for metric in metrics:
        for contrast in CONTRAST_SPECS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                real_mean = np.nanmean(real[metric][contrast], axis=0)
                shuf_flat = shuffle[metric][contrast].reshape(-1, n_bin)
                shuf_mean = np.nanmean(shuf_flat, axis=0)
                shuf_std = np.nanstd(shuf_flat, axis=0, ddof=1)
            shuffle_z = np.divide(
                real_mean - shuf_mean,
                shuf_std,
                out=np.full(n_bin, np.nan, dtype=np.float32),
                where=(shuf_std > 0) & np.isfinite(shuf_std),
            )
            if metric in {"spearman", "cosine"}:
                separation_z = -shuffle_z
            else:
                separation_z = shuffle_z
            out["per_bin_real_by_metric_contrast"][metric][contrast] = real_mean.astype(np.float32)
            out["per_bin_shuffle_mean_by_metric_contrast"][metric][contrast] = shuf_mean.astype(np.float32)
            out["per_bin_shuffle_std_by_metric_contrast"][metric][contrast] = shuf_std.astype(np.float32)
            out["per_bin_shuffle_z_by_metric_contrast"][metric][contrast] = shuffle_z.astype(np.float32)
            out["per_bin_separation_z_by_metric_contrast"][metric][contrast] = separation_z.astype(np.float32)
    return out
