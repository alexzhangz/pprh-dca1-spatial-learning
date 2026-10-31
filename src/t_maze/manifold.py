"""Manuscript definitions collected from manifold.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from collections import OrderedDict

from typing import Mapping, Sequence

import numpy as np

import pandas as pd

from sklearn.covariance import LedoitWolf

from .stability import (
    GROUP_ORDER,
    InferenceSpec,
    build_group_day_summary,
    condition_trial_mask,
    discover_group_mice,
    finalize_inference_results,
    fit_longitudinal_effects_by,
    nanmean_no_warning,
    tmaze_path_sections,
)

DEFAULT_TARGET_NEURON_COUNT = 50

DEFAULT_MIN_TRIALS_PER_CONDITION = 3

DEFAULT_DPRIME_COVARIANCE = "within_class_ledoit_wolf"

LEGACY_DPRIME_COVARIANCE = "legacy_pooled"

DPRIME_COVARIANCE_OPTIONS = (
    DEFAULT_DPRIME_COVARIANCE,
    LEGACY_DPRIME_COVARIANCE,
)

DPRIME_COVARIANCE = DEFAULT_DPRIME_COVARIANCE

CONDITION_ALIASES = {
    "SL": "SLC",
    "SR": "SRC",
    "CL": "CLC",
    "CR": "CRC",
}

CONDITION_LABELS = {
    "SL": "Sample left",
    "SR": "Sample right",
    "CL": "Choice left",
    "CR": "Choice right",
}

CONDITION_PHASE = {
    "SLC": "S",
    "SRC": "S",
    "SLA": "S",
    "SRA": "S",
    "CLC": "C",
    "CRC": "C",
    "CL": "C",
    "CR": "C",
}

CONDITION_ORDER = ("SL", "SR", "CL", "CR")

PAIR_CONTRAST_SPECS = OrderedDict(
    [
        ("sample_vs_choice_left", ("SL", "CL")),
        ("sample_vs_choice_right", ("SR", "CR")),
        ("sample_left_vs_right", ("SL", "SR")),
        ("choice_left_vs_right", ("CL", "CR")),
    ]
)

AGGREGATE_CONTRAST_SPECS = OrderedDict(
    [
        ("sample_vs_choice", ("S", "C")),
    ]
)

CONTRAST_SPECS = OrderedDict(
    [
        *PAIR_CONTRAST_SPECS.items(),
        *AGGREGATE_CONTRAST_SPECS.items(),
    ]
)

CONTRAST_LABELS = {
    "sample_vs_choice_left": "SL vs CL",
    "sample_vs_choice_right": "SR vs CR",
    "sample_left_vs_right": "SL vs SR",
    "choice_left_vs_right": "CL vs CR",
    "sample_vs_choice": "S vs C",
}

def build_tmaze_manifold_condition_trials(
    sample_day: np.ndarray,
    choice_day: np.ndarray,
    index_all: np.ndarray,
    *,
    day_idx: int,
    condition_aliases: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Return trial clouds for the manifold labels SL/SR/CL/CR.

    Parameters
    ----------
    sample_day, choice_day
        Arrays shaped ``(trial, bin, neuron)`` from ``Snarm_trial_v2``.
    index_all
        T-maze v2 trial index array shaped ``(3, day, trial)``.
    condition_aliases
        Mapping from user-facing labels to v2 condition keys. The default maps
        SL/SR/CL/CR to SLC/SRC/CLC/CRC, so all default contrasts use correct
        trials only.
    """
    aliases = dict(CONDITION_ALIASES if condition_aliases is None else condition_aliases)
    sample_day = _as_trial_bin_neuron(sample_day, "sample_day")
    choice_day = _as_trial_bin_neuron(choice_day, "choice_day")
    if sample_day.shape[1:] != choice_day.shape[1:]:
        raise ValueError(
            "sample_day and choice_day must share bin/neuron axes, "
            f"got {sample_day.shape} and {choice_day.shape}"
        )

    source_by_phase = {"S": sample_day, "C": choice_day}
    result: dict[str, np.ndarray] = {}
    for label in CONDITION_ORDER:
        key = aliases[label]
        phase = CONDITION_PHASE.get(key)
        if phase not in source_by_phase:
            raise ValueError(f"condition key {key!r} does not map to S or C phase")
        source = source_by_phase[phase]
        mask = condition_trial_mask(index_all, int(day_idx), key)[: source.shape[0]]
        result[label] = source[mask].astype(float, copy=False)
    return result

def build_spatial_bin_long_table(
    metric_arrays_by_contrast: Mapping[str, Mapping[str, np.ndarray]],
    mouse_list: Sequence[tuple[str, str]],
    contrasts: Sequence[str],
    *,
    groups: Sequence[str],
    num_day: int,
    metric_cols: Sequence[str],
    contrast_labels: Mapping[str, str] | None = None,
    first_day: int = 1,
    last_day: int | None = None,
) -> pd.DataFrame:
    """Build a long mouse/day/bin table from per-bin manifold arrays."""
    columns = [
        "group",
        "mouse",
        "day",
        "day_idx",
        "day_centered",
        "contrast",
        "contrast_label",
        "bin",
        *metric_cols,
    ]
    contrast_labels = dict(CONTRAST_LABELS if contrast_labels is None else contrast_labels)
    group_set = set(groups)
    if last_day is None:
        last_day = int(num_day)
    center = (float(first_day) + float(last_day)) / 2.0
    rows = []
    for contrast in contrasts:
        arrays = {metric: np.asarray(metric_arrays_by_contrast[contrast][metric], dtype=float) for metric in metric_cols}
        shape = next(iter(arrays.values())).shape
        if len(shape) != 3:
            raise ValueError("metric arrays must have shape (mouse, day, bin)")
        n_mouse, n_day, n_bin = shape
        for mouse_idx, (group, mouse) in enumerate(mouse_list[:n_mouse]):
            if group not in group_set:
                continue
            for day_idx in range(min(int(num_day), n_day)):
                for bin_idx in range(n_bin):
                    row = {
                        "group": group,
                        "mouse": mouse,
                        "day": day_idx + 1,
                        "day_idx": day_idx,
                        "day_centered": (day_idx + 1) - center,
                        "contrast": contrast,
                        "contrast_label": contrast_labels.get(contrast, contrast),
                        "bin": bin_idx,
                    }
                    has_finite = False
                    for metric, values in arrays.items():
                        value = float(values[mouse_idx, day_idx, bin_idx])
                        row[metric] = value
                        has_finite = has_finite or np.isfinite(value)
                    if has_finite:
                        rows.append(row)
    return pd.DataFrame(rows, columns=columns)

def build_spatial_section_long_table(
    metric_arrays_by_contrast: Mapping[str, Mapping[str, np.ndarray]],
    mouse_list: Sequence[tuple[str, str]],
    contrasts: Sequence[str],
    *,
    groups: Sequence[str],
    num_day: int,
    metric_cols: Sequence[str],
    section_definitions_by_contrast: Mapping[str, Mapping[str, Sequence[int]]],
    contrast_labels: Mapping[str, str] | None = None,
    first_day: int = 1,
    last_day: int | None = None,
) -> pd.DataFrame:
    """Build a long mouse/day/section table by averaging bins."""
    columns = [
        "group",
        "mouse",
        "day",
        "day_idx",
        "day_centered",
        "contrast",
        "contrast_label",
        "section",
        "section_idx",
        "n_section_bin",
        *metric_cols,
    ]
    contrast_labels = dict(CONTRAST_LABELS if contrast_labels is None else contrast_labels)
    group_set = set(groups)
    if last_day is None:
        last_day = int(num_day)
    center = (float(first_day) + float(last_day)) / 2.0
    rows = []
    for contrast in contrasts:
        arrays = {metric: np.asarray(metric_arrays_by_contrast[contrast][metric], dtype=float) for metric in metric_cols}
        shape = next(iter(arrays.values())).shape
        if len(shape) != 3:
            raise ValueError("metric arrays must have shape (mouse, day, bin)")
        n_mouse, n_day, n_bin = shape
        sections = section_definitions_by_contrast[contrast]
        _validate_sections(sections, n_bin)
        for mouse_idx, (group, mouse) in enumerate(mouse_list[:n_mouse]):
            if group not in group_set:
                continue
            for day_idx in range(min(int(num_day), n_day)):
                for section_idx, (section, bins) in enumerate(sections.items()):
                    bins = np.asarray(bins, dtype=int)
                    row = {
                        "group": group,
                        "mouse": mouse,
                        "day": day_idx + 1,
                        "day_idx": day_idx,
                        "day_centered": (day_idx + 1) - center,
                        "contrast": contrast,
                        "contrast_label": contrast_labels.get(contrast, contrast),
                        "section": section,
                        "section_idx": int(section_idx),
                        "n_section_bin": int(bins.size),
                    }
                    has_finite = False
                    for metric, values in arrays.items():
                        value = nanmean_no_warning(values[mouse_idx, day_idx, bins])
                        row[metric] = float(value)
                        has_finite = has_finite or np.isfinite(value)
                    if has_finite:
                        rows.append(row)
    return pd.DataFrame(rows, columns=columns)

def _as_trial_bin_neuron(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (trial, bin, neuron), got {arr.shape}")
    return arr

def _validated_condition_trials(condition_trials: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [label for label in CONDITION_ORDER if label not in condition_trials]
    if missing:
        raise KeyError(f"condition_trials missing labels: {missing}")
    result = {label: _as_trial_bin_neuron(condition_trials[label], label) for label in CONDITION_ORDER}
    shapes = {values.shape[1:] for values in result.values()}
    if len(shapes) != 1:
        raise ValueError(f"all condition trial clouds must share bin/neuron axes, got {shapes}")
    return result

def _zscore_columns(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mean = np.nanmean(points, axis=0, keepdims=True)
    std = np.nanstd(points, axis=0, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    values = (points - mean) / std
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

def _precision_by_bin(
    class_a: np.ndarray,
    class_b: np.ndarray,
    *,
    dprime_covariance: str = DEFAULT_DPRIME_COVARIANCE,
) -> list[np.ndarray | None]:
    class_a = np.asarray(class_a, dtype=float)
    class_b = np.asarray(class_b, dtype=float)
    return [
        _precision_for_classes(
            class_a[:, bin_idx, :],
            class_b[:, bin_idx, :],
            dprime_covariance=dprime_covariance,
        )
        for bin_idx in range(class_a.shape[1])
    ]

def _precision_for_classes(
    class_a: np.ndarray,
    class_b: np.ndarray,
    *,
    dprime_covariance: str = DEFAULT_DPRIME_COVARIANCE,
) -> np.ndarray | None:
    covariance = _normalize_dprime_covariance(dprime_covariance)
    if covariance == DEFAULT_DPRIME_COVARIANCE:
        return _within_class_precision(class_a, class_b)
    if covariance == LEGACY_DPRIME_COVARIANCE:
        return _pooled_precision(np.vstack([class_a, class_b]))
    raise ValueError(f"Unsupported dprime_covariance {dprime_covariance!r}")

def _within_class_precision(class_a: np.ndarray, class_b: np.ndarray) -> np.ndarray | None:
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
    return _precision_from_covariance(covariance)

def _pooled_precision(combined: np.ndarray) -> np.ndarray | None:
    combined = _finite_rows(combined)
    if combined.shape[0] < 2 or combined.shape[1] == 0:
        return None
    if combined.shape[0] <= combined.shape[1]:
        covariance = np.cov(combined, rowvar=False)
    else:
        covariance = LedoitWolf().fit(combined).covariance_
    return _precision_from_covariance(covariance)

def _precision_from_covariance(covariance: np.ndarray) -> np.ndarray | None:
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    covariance += np.eye(covariance.shape[0]) * 1e-9
    precision = np.linalg.pinv(covariance)
    if not np.isfinite(precision).all():
        return None
    return precision

def _d_prime_from_precision(class_a: np.ndarray, class_b: np.ndarray, precision: np.ndarray | None) -> float:
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

def _shuffle_rows(class_a: np.ndarray, class_b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(class_a)
    b = np.asarray(class_b)
    combined = np.concatenate([a, b], axis=0)
    order = rng.permutation(combined.shape[0])
    shuffled = combined[order]
    return shuffled[: a.shape[0]], shuffled[a.shape[0] :]

def _sample_vs_choice_balance_count(condition_trials: Mapping[str, np.ndarray]) -> int:
    return int(min(np.asarray(condition_trials[label]).shape[0] for label in CONDITION_ORDER))

def _balanced_sample_choice_classes(
    emb_by_condition: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray]]:
    """Return S and C classes with left/right labels balanced by undersampling."""
    balance_count = _sample_vs_choice_balance_count(emb_by_condition)
    selected: dict[str, np.ndarray] = {}
    for label in CONDITION_ORDER:
        values = np.asarray(emb_by_condition[label])
        idx = rng.choice(values.shape[0], size=balance_count, replace=False)
        selected[label] = values[idx]
    sample = np.concatenate([selected["SL"], selected["SR"]], axis=0)
    choice = np.concatenate([selected["CL"], selected["CR"]], axis=0)
    return sample, choice, int(balance_count), selected

def _shuffle_sample_choice_within_side(
    selected_by_condition: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle S/C labels separately within left and right balanced subsets."""
    sample_left, choice_left = _shuffle_rows(
        selected_by_condition["SL"],
        selected_by_condition["CL"],
        rng,
    )
    sample_right, choice_right = _shuffle_rows(
        selected_by_condition["SR"],
        selected_by_condition["CR"],
        rng,
    )
    return (
        np.concatenate([sample_left, sample_right], axis=0),
        np.concatenate([choice_left, choice_right], axis=0),
    )

def _finite_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        return np.empty((0, 0))
    return values[np.isfinite(values).all(axis=1)]

def _normalize_dprime_covariance(dprime_covariance: str) -> str:
    key = str(dprime_covariance).lower().replace("-", "_")
    aliases = {
        DEFAULT_DPRIME_COVARIANCE: DEFAULT_DPRIME_COVARIANCE,
        "within_class": DEFAULT_DPRIME_COVARIANCE,
        "within_class_ledoitwolf": DEFAULT_DPRIME_COVARIANCE,
        "ledoit_wolf": DEFAULT_DPRIME_COVARIANCE,
        "ledoitwolf": DEFAULT_DPRIME_COVARIANCE,
        LEGACY_DPRIME_COVARIANCE: LEGACY_DPRIME_COVARIANCE,
        "legacy": LEGACY_DPRIME_COVARIANCE,
        "pooled": LEGACY_DPRIME_COVARIANCE,
        "pooled_regularized": LEGACY_DPRIME_COVARIANCE,
    }
    if key not in aliases:
        expected = ", ".join(DPRIME_COVARIANCE_OPTIONS)
        raise ValueError(
            f"Unsupported dprime_covariance {dprime_covariance!r}. "
            f"Expected one of: {expected}."
        )
    return aliases[key]

def _validate_sections(sections: Mapping[str, Sequence[int]], n_bin: int) -> None:
    for section, bins in sections.items():
        arr = np.asarray(bins, dtype=int).ravel()
        if arr.size == 0:
            raise ValueError(f"section {section!r} has no bins")
        if np.any(arr < 0) or np.any(arr >= int(n_bin)):
            raise ValueError(f"section {section!r} contains bins outside 0..{int(n_bin) - 1}")
