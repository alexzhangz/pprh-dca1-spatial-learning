"""Manuscript definitions collected from stability.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import warnings

from collections import OrderedDict

from dataclasses import dataclass

from pathlib import Path

from typing import Mapping, Sequence

import numpy as np

import pandas as pd

from scipy import stats

from scipy.stats import rankdata, sem

GROUP_ORDER = ("C", "I", "P", "N")

GROUP_MICE = {
    "C": tuple(f"C{i}" for i in range(3, 12)),
    "I": ("I1", "I2", "I3", "I4", "I5", "I7", "I8", "I9"),
    "P": tuple(f"P{i}" for i in range(1, 6)),
    "N": ("N2", "N3"),
}

CONDITION_PAIR_MAP = {
    "choice_correct": {"left": "CLC", "right": "CRC"},
    "choice_all": {"left": "CL", "right": "CR"},
    "sample_correct": {"left": "SLC", "right": "SRC"},
    "sample_rewarded": {"left": "SLA", "right": "SRA"},
    "choice_rewarded": {"left": "CLA", "right": "CRA"},
}

INTERDAY_PC_COLUMNS = [
    "group",
    "mouse",
    "pair_mode",
    "pair_index",
    "day_a",
    "day_b",
    "day_pair_label",
    "condition",
    "condition_key",
    "neuron",
    "pc_interday_corr",
]

CANONICAL_RESULT_COLUMNS = (
    "analysis",
    "view",
    "metric",
    "contrast",
    "effect_type",
    "test_family",
    "test",
    "reference",
    "comparison",
    "group",
    "effect",
    "statistic",
    "p_value",
    "q_value",
    "significant",
    "sampling_unit",
    "aggregation_level",
    "independence_assumption",
    "n_mouse",
    "n_mouse_day",
    "n_neuron",
    "n_observation",
    "model_formula",
    "model_status",
)

@dataclass(frozen=True)
class InferenceSpec:
    analysis: str
    view: str
    metric: str
    value_col: str
    contrast: str
    sampling_unit: str
    aggregation_level: str
    independence_assumption: str
    reference_group: str
    comparison_group: str
    unit_id_cols: tuple[str, ...]
    repeat_id_cols: tuple[str, ...]
    group_col: str = "group"
    day_col: str = "day"

def resolve_condition_parts(condition_pair: str | Mapping[str, str]) -> dict[str, str]:
    """Return canonical left/right condition keys for a T-maze contrast."""
    if isinstance(condition_pair, Mapping):
        result = dict(condition_pair)
    else:
        key = str(condition_pair).lower().replace("-", "_").replace("/", "_").replace(" ", "_")
        if key not in CONDITION_PAIR_MAP:
            raise ValueError(
                f"unknown condition_pair {condition_pair!r}; expected one of "
                f"{sorted(CONDITION_PAIR_MAP)}"
            )
        result = dict(CONDITION_PAIR_MAP[key])
    if set(result) != {"left", "right"}:
        raise ValueError("condition parts must contain exactly 'left' and 'right'")
    return result

def tmaze_path_sections(n_bins: int, *, n_stem: int = 13) -> OrderedDict[str, np.ndarray]:
    """Return physically meaningful T-maze path sections for section-level plots."""
    n_bins = int(n_bins)
    n_stem = min(int(n_stem), n_bins)
    sections = OrderedDict()
    if n_stem > 0:
        sections["Stem"] = np.arange(0, n_stem, dtype=int)
    if n_bins > n_stem:
        sections["Choice arm"] = np.arange(n_stem, n_bins, dtype=int)
    if not sections:
        sections["All bins"] = np.arange(0, n_bins, dtype=int)
    return sections

def condition_trial_mask(index_all: np.ndarray, day_idx: int, condition_key: str) -> np.ndarray:
    """Return trial mask for a v2 condition key using `generate_spatial_maps` labels."""
    labels = np.asarray(index_all, dtype=bool)
    if labels.ndim != 3 or labels.shape[0] < 3:
        raise ValueError(f"index_all must have shape (3, day, trial), got {labels.shape}")
    day_idx = int(day_idx)
    if day_idx < 0 or day_idx >= labels.shape[1]:
        raise IndexError(f"day_idx {day_idx} outside index_all day axis")
    left_index = labels[0, day_idx]
    correct_index = labels[1, day_idx]
    real_left_index = labels[2, day_idx]
    all_trials = np.ones(left_index.shape, dtype=bool)
    masks = {
        "C": all_trials,
        "S": all_trials,
        "CL": ~real_left_index,
        "CR": real_left_index,
        "CLC": correct_index & left_index,
        "CRC": correct_index & ~left_index,
        "CLI": ~correct_index & left_index,
        "CRI": ~correct_index & ~left_index,
        "CLA": left_index,
        "CRA": ~left_index,
        "SLC": correct_index & left_index,
        "SRC": correct_index & ~left_index,
        "SLA": left_index,
        "SRA": ~left_index,
    }
    key = str(condition_key)
    if key not in masks:
        raise ValueError(f"unknown condition key {condition_key!r}")
    return masks[key].copy()

def discover_group_mice(
    aligned_root: str | Path,
    *,
    groups: Sequence[str] = ("C", "I", "P"),
    require_v2: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Discover available subjects in group order."""
    root = Path(aligned_root)
    result: dict[str, tuple[str, ...]] = {}
    for group in GROUP_ORDER:
        if group not in set(groups):
            continue
        mice = []
        for mouse in GROUP_MICE.get(group, ()):
            subject = root / mouse
            if not subject.is_dir():
                continue
            if require_v2 and not (
                (subject / "Snarm_trial_v2.npy").exists()
                and (subject / "Snarm_1d_v2.npy").exists()
            ):
                continue
            mice.append(mouse)
        result[group] = tuple(mice)
    return result

def load_subject_condition_maps(
    subject_dir: str | Path,
    *,
    condition_keys: Sequence[str] | None = None,
    filename: str = "Snarm_1d_v2.npy",
) -> dict[str, np.ndarray]:
    """Load day-level condition maps as `(day, neuron, bin)` arrays."""
    path = Path(subject_dir) / filename
    raw = np.load(path, allow_pickle=True).item()
    keys = list(raw) if condition_keys is None else list(condition_keys)
    return {key: _condition_object_to_day_neuron_bin(raw[key]) for key in keys}

def load_pc_day_mask(
    subject_dir: str | Path,
    day_idx: int,
    n_neuron: int,
    *,
    pc_key: str = "C",
    filename: str = "pc_index_v2.npy",
) -> np.ndarray:
    """Load a day PC mask from new-key `pc_index_v2.npy` only."""
    pc_index = np.load(Path(subject_dir) / filename, allow_pickle=True).item()
    if pc_key not in pc_index:
        raise KeyError(f"{filename} missing new key {pc_key!r}")
    values = np.asarray(pc_index[pc_key][int(day_idx)], dtype=bool).ravel()
    return align_mask(values, int(n_neuron), fill=False)

def align_mask(mask: np.ndarray | None, n_items: int, *, fill: bool = True) -> np.ndarray:
    if mask is None:
        return np.full(int(n_items), bool(fill), dtype=bool)
    source = np.asarray(mask, dtype=bool).ravel()
    aligned = np.zeros(int(n_items), dtype=bool)
    n_copy = min(source.size, aligned.size)
    aligned[:n_copy] = source[:n_copy]
    return aligned

def safe_corr(x, y, *, method: str = "pearson", min_n: int = 3) -> float:
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < int(min_n):
        return np.nan
    xv = x[valid]
    yv = y[valid]
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(xv, yv)[0, 1])
    corr, _ = stats.spearmanr(xv, yv)
    return float(corr)

def rowwise_corr(matrix_a, matrix_b, *, method: str = "pearson", min_obs: int = 3) -> np.ndarray:
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.shape != b.shape:
        raise ValueError(f"Input arrays must have the same shape, got {a.shape} and {b.shape}")

    valid = np.isfinite(a) & np.isfinite(b)
    n_valid = np.sum(valid, axis=-1)
    if method == "spearman":
        a = rankdata(np.where(valid, a, np.nan), axis=-1, nan_policy="omit")
        b = rankdata(np.where(valid, b, np.nan), axis=-1, nan_policy="omit")
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        am = np.nanmean(np.where(valid, a, np.nan), axis=-1, keepdims=True)
        bm = np.nanmean(np.where(valid, b, np.nan), axis=-1, keepdims=True)
    ac = np.where(valid, a - am, 0.0)
    bc = np.where(valid, b - bm, 0.0)
    numerator = np.sum(ac * bc, axis=-1)
    denominator = np.sqrt(np.sum(ac * ac, axis=-1) * np.sum(bc * bc, axis=-1))
    return np.divide(
        numerator,
        denominator,
        out=np.full(n_valid.shape, np.nan, dtype=float),
        where=(n_valid >= int(min_obs)) & (denominator > 0),
    )

def pc_corr(vector_a, vector_b, *, method: str = "pearson", min_obs: int = 1) -> float:
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < int(min_obs):
        return np.nan
    av = a[valid]
    bv = b[valid]
    if np.nansum(av) == 0 or np.nansum(bv) == 0:
        return np.nan
    return safe_corr(av, bv, method=method, min_n=max(2, int(min_obs)))

def subsampled_pv_corr(
    map_a,
    map_b,
    selected_neuron_count: int | None,
    *,
    n_iter: int = 100,
    seed: int = 2026,
    method: str = "spearman",
) -> np.ndarray:
    """Return binwise PV correlations for `(neuron, bin)` maps."""
    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"maps must share shape (neuron, bin), got {a.shape} and {b.shape}")
    if selected_neuron_count is None:
        return rowwise_corr(a.T, b.T, method=method)
    target = int(selected_neuron_count)
    n_neuron, n_bin = a.shape
    valid_neuron = np.isfinite(a).any(axis=1) & np.isfinite(b).any(axis=1)
    a = a[valid_neuron]
    b = b[valid_neuron]
    if target < 3 or a.shape[0] < target:
        return np.full(n_bin, np.nan, dtype=float)
    rng = np.random.default_rng(int(seed))
    values = np.full((int(n_iter), n_bin), np.nan, dtype=float)
    for iter_idx in range(int(n_iter)):
        idx = rng.choice(a.shape[0], size=target, replace=False)
        values[iter_idx] = rowwise_corr(a[idx].T, b[idx].T, method=method)
    return nanmean_no_warning(values, axis=0)

def interday_day_pairs(num_day: int, pair_mode: str) -> list[dict[str, object]]:
    mode = str(pair_mode).lower().replace("-", "_").replace(" ", "_")
    if mode not in {"adjacent", "anchor"}:
        raise ValueError("pair_mode must be 'adjacent' or 'anchor'")
    num_day = int(num_day)
    if mode == "adjacent":
        pairs = [(idx, idx + 1) for idx in range(max(num_day - 1, 0))]
    else:
        pairs = [(0, idx) for idx in range(1, num_day)]
    return [
        {
            "pair_mode": mode,
            "pair_index": int(pair_idx),
            "day_a": int(day_a),
            "day_b": int(day_b),
            "day_pair_label": f"D{day_a + 1}-D{day_b + 1}",
        }
        for pair_idx, (day_a, day_b) in enumerate(pairs)
    ]

def compute_interday_pv_stability(
    condition_maps: Mapping[str, np.ndarray],
    condition_parts: Mapping[str, str],
    day_pairs: Sequence[Mapping[str, object]],
    *,
    neuron_mask: np.ndarray | None = None,
    selected_neuron_count: int | None = 50,
    n_subsample_iter: int = 100,
    seed: int = 2026,
    method: str = "spearman",
    bin_indices: Sequence[int] | None = None,
    valid_day_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute same-condition interday binwise PV stability for one subject."""
    condition_parts = resolve_condition_parts(condition_parts)
    day_pairs = list(day_pairs)
    first_key = next((key for key in condition_parts.values() if key in condition_maps), None)
    if first_key is None:
        return _empty_pv_result(condition_parts, len(day_pairs), 0)
    first = np.asarray(condition_maps[first_key], dtype=float)
    if first.ndim != 3:
        raise ValueError(f"condition map {first_key!r} must have shape (day, neuron, bin)")
    n_day, n_neuron, n_bin = first.shape
    bins = _validate_bin_indices(bin_indices, n_bin)
    mask = align_mask(neuron_mask, n_neuron, fill=True)
    valid_days = _valid_day_mask(valid_day_mask, n_day)
    result = _empty_pv_result(condition_parts, len(day_pairs), bins.size)
    result["n_neuron"] = int(np.sum(mask))

    for pair_idx, pair in enumerate(day_pairs):
        day_a = int(pair["day_a"])
        day_b = int(pair["day_b"])
        if not _pair_is_valid(day_a, day_b, n_day, valid_days):
            continue
        for part, key in condition_parts.items():
            if key not in condition_maps:
                continue
            values = np.asarray(condition_maps[key], dtype=float)
            if values.ndim != 3 or values.shape[0] <= max(day_a, day_b):
                continue
            if values.shape[1] != n_neuron:
                raise ValueError("all condition maps must share neuron count")
            map_a = values[day_a, mask, :][:, bins]
            map_b = values[day_b, mask, :][:, bins]
            result[part][pair_idx] = subsampled_pv_corr(
                map_a,
                map_b,
                selected_neuron_count,
                n_iter=n_subsample_iter,
                seed=int(seed) + pair_idx * 10_000 + _seed_offset(key),
                method=method,
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result["averaged"] = np.nanmean(np.stack([result["left"], result["right"]]), axis=0)
    return result

def compute_interday_pc_long_table(
    condition_maps: Mapping[str, np.ndarray],
    condition_parts: Mapping[str, str],
    day_pairs: Sequence[Mapping[str, object]],
    *,
    group: str,
    mouse: str,
    bin_indices: Sequence[int] | None = None,
    valid_day_mask: np.ndarray | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return one interday PC spatial correlation row per neuron and condition."""
    condition_parts = resolve_condition_parts(condition_parts)
    first_key = next((key for key in condition_parts.values() if key in condition_maps), None)
    if first_key is None:
        return pd.DataFrame(columns=INTERDAY_PC_COLUMNS)
    first = np.asarray(condition_maps[first_key], dtype=float)
    n_day, n_neuron, n_bin = first.shape
    bins = _validate_bin_indices(bin_indices, n_bin)
    valid_days = _valid_day_mask(valid_day_mask, n_day)
    rows = []
    for pair in day_pairs:
        day_a = int(pair["day_a"])
        day_b = int(pair["day_b"])
        if not _pair_is_valid(day_a, day_b, n_day, valid_days):
            continue
        values_by_part = {}
        for part, key in condition_parts.items():
            if key not in condition_maps:
                continue
            maps = np.asarray(condition_maps[key], dtype=float)
            if maps.shape[1] != n_neuron:
                raise ValueError("all condition maps must share neuron count")
            map_a = maps[day_a, :, :][:, bins]
            map_b = maps[day_b, :, :][:, bins]
            corr = np.asarray([pc_corr(map_a[idx], map_b[idx], method=method) for idx in range(n_neuron)])
            values_by_part[part] = corr
            rows.extend(_interday_pc_rows(corr, pair, group, mouse, part, key))
        if {"left", "right"}.issubset(values_by_part):
            averaged = nanmean_no_warning(np.vstack([values_by_part["left"], values_by_part["right"]]), axis=0)
            rows.extend(_interday_pc_rows(averaged, pair, group, mouse, "averaged", "averaged"))
    return pd.DataFrame(rows, columns=INTERDAY_PC_COLUMNS)

def compute_interday_stability_subject(
    subject_dir: str | Path,
    *,
    mouse: str,
    group: str,
    condition_pair: str | Mapping[str, str] = "choice_correct",
    pair_modes: Sequence[str] = ("adjacent", "anchor"),
    neuron_set: str = "all",
    pc_key: str = "C",
    selected_neuron_count: int | None = 50,
    n_subsample_iter: int = 100,
    pv_method: str = "spearman",
    pc_method: str = "pearson",
    seed: int = 2026,
) -> dict[str, object]:
    subject = Path(subject_dir)
    parts = resolve_condition_parts(condition_pair)
    maps = load_subject_condition_maps(subject, condition_keys=tuple(parts.values()))
    first = next(iter(maps.values()))
    n_day, n_neuron, n_bin = first.shape
    payload: dict[str, object] = {
        "pv_arrays": {},
        "candidate_neuron_count": {},
        "pc_long": {},
        "n_day": n_day,
        "n_bin": n_bin,
    }
    valid_days = np.ones(n_day, dtype=bool)
    for mode in pair_modes:
        pairs = interday_day_pairs(n_day, mode)
        arrays = {part: np.full((len(pairs), n_bin), np.nan, dtype=float) for part in ("left", "right", "averaged")}
        candidates = np.full(len(pairs), np.nan, dtype=float)
        pc_table = compute_interday_pc_long_table(
            maps,
            parts,
            pairs,
            group=group,
            mouse=mouse,
            method=pc_method,
            valid_day_mask=valid_days,
        )
        for pair_idx, pair in enumerate(pairs):
            mask = _interday_neuron_mask(
                subject,
                int(pair["day_a"]),
                int(pair["day_b"]),
                n_neuron,
                neuron_set=neuron_set,
                pc_key=pc_key,
            )
            candidates[pair_idx] = int(np.sum(mask))
            result = compute_interday_pv_stability(
                maps,
                parts,
                [pair],
                neuron_mask=mask,
                selected_neuron_count=selected_neuron_count,
                n_subsample_iter=n_subsample_iter,
                seed=int(seed) + pair_idx * 1000 + _seed_offset(mouse),
                method=pv_method,
                valid_day_mask=valid_days,
            )
            for part in arrays:
                arrays[part][pair_idx] = result[part][0]
        payload["pv_arrays"][mode] = arrays
        payload["candidate_neuron_count"][mode] = candidates
        payload["pc_long"][mode] = pc_table
    return payload

def nanmean_no_warning(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)

def sem_no_warning(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sem(values, axis=axis, nan_policy="omit")

def build_group_day_summary(
    table: pd.DataFrame,
    value_col: str,
    *,
    by: Sequence[str] = (),
    group_col: str = "group",
    day_col: str = "day",
) -> pd.DataFrame:
    keys = list(dict.fromkeys([*tuple(by), group_col, day_col]))
    required = set(keys + [value_col])
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"table is missing columns: {missing}")
    rows = []
    grouper = keys[0] if len(keys) == 1 else keys
    for key, subset in table.groupby(grouper, observed=True, dropna=False, sort=False):
        key_values = (key,) if len(keys) == 1 else tuple(key)
        values = pd.to_numeric(subset[value_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row = dict(zip(keys, key_values))
        row["mean"] = float(np.mean(values)) if values.size else np.nan
        row["sem"] = float(sem_no_warning(values)) if values.size else np.nan
        row["n_observation"] = int(values.size)
        rows.append(row)
    return pd.DataFrame(rows, columns=keys + ["mean", "sem", "n_observation"])

def fit_longitudinal_effects_by(
    table: pd.DataFrame,
    spec: InferenceSpec,
    *,
    by: Sequence[str] = (),
    first_day: int,
    last_day: int,
    include_interaction: bool = True,
) -> pd.DataFrame:
    """Fit canonical 8-maze-style longitudinal effects.

    This implementation uses OLS with cluster-robust covariance by the declared
    repeat id. It keeps the same result columns/effect names used by the
    migrated notebooks.
    """
    rows = []
    group_cols = tuple(by)
    if group_cols:
        grouped = table.groupby(list(group_cols), observed=True, dropna=False, sort=False)
    else:
        grouped = [((), table)]
    center = (float(first_day) + float(last_day)) / 2.0
    for key, subset in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        locator = dict(zip(group_cols, key_tuple)) if group_cols else {}
        model = subset.loc[
            subset[spec.group_col].isin([spec.reference_group, spec.comparison_group])
        ].copy()
        model[spec.day_col] = pd.to_numeric(model[spec.day_col], errors="coerce")
        model[spec.value_col] = pd.to_numeric(model[spec.value_col], errors="coerce")
        model = model.loc[np.isfinite(model[spec.day_col]) & np.isfinite(model[spec.value_col])]
        if model.empty:
            continue
        model["day_centered"] = model[spec.day_col] - center
        rows.extend(_between_and_slope_rows(model, spec, locator, include_interaction))
        rows.extend(_endpoint_rows(model, spec, locator, first_day, last_day))
    return finalize_inference_results(pd.DataFrame(rows))

def finalize_inference_results(results: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=CANONICAL_RESULT_COLUMNS)
    frame = results.copy()
    for column in CANONICAL_RESULT_COLUMNS:
        if column not in frame:
            frame[column] = False if column == "significant" else np.nan
    frame["q_value"] = np.nan
    frame["significant"] = False
    family_cols = ["analysis", "view", "metric", "contrast", "effect_type", "test_family"]
    for _, index in frame.groupby(family_cols, dropna=False, observed=True).groups.items():
        q_values, significant = benjamini_hochberg(frame.loc[index, "p_value"].to_numpy(dtype=float), alpha=alpha)
        frame.loc[index, "q_value"] = q_values
        frame.loc[index, "significant"] = significant
    return frame.loc[:, list(CANONICAL_RESULT_COLUMNS) + [c for c in frame.columns if c not in CANONICAL_RESULT_COLUMNS]]

def benjamini_hochberg(p_values, *, alpha: float = 0.05):
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    significant = np.zeros(p_values.shape, dtype=bool)
    finite = np.isfinite(p_values)
    if not np.any(finite):
        return q_values, significant
    finite_values = p_values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    adjusted = ranked * ranked.size / np.arange(1, ranked.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q_finite = np.empty_like(adjusted)
    q_finite[order] = adjusted
    q_values[finite] = q_finite
    significant[finite] = q_finite <= alpha
    return q_values, significant

def _condition_object_to_day_neuron_bin(values) -> np.ndarray:
    days = np.asarray(values, dtype=object)
    converted = []
    for day_idx, value in enumerate(days):
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"condition day {day_idx} must have shape (bin, neuron), got {arr.shape}")
        converted.append(arr.T)
    return np.stack(converted, axis=0)

def _interday_neuron_mask(subject, day_a, day_b, n_neuron, *, neuron_set, pc_key):
    if str(neuron_set).lower() == "all":
        return np.ones(int(n_neuron), dtype=bool)
    mask_a = load_pc_day_mask(subject, day_a, n_neuron, pc_key=pc_key)
    mask_b = load_pc_day_mask(subject, day_b, n_neuron, pc_key=pc_key)
    return mask_a | mask_b

def _interday_pc_rows(values, pair, group, mouse, condition, condition_key):
    return [
        {
            "group": group,
            "mouse": mouse,
            "pair_mode": pair["pair_mode"],
            "pair_index": int(pair["pair_index"]),
            "day_a": int(pair["day_a"]) + 1,
            "day_b": int(pair["day_b"]) + 1,
            "day_pair_label": pair["day_pair_label"],
            "condition": condition,
            "condition_key": condition_key,
            "neuron": int(neuron_idx),
            "pc_interday_corr": float(value) if np.isfinite(value) else np.nan,
        }
        for neuron_idx, value in enumerate(np.asarray(values, dtype=float).ravel())
    ]

def _validate_bin_indices(bin_indices, n_bin):
    if bin_indices is None:
        return np.arange(int(n_bin), dtype=int)
    bins = np.asarray(bin_indices, dtype=int).ravel()
    if np.any(bins < 0) or np.any(bins >= int(n_bin)):
        raise ValueError(f"bin_indices must index bins 0..{int(n_bin) - 1}")
    return bins

def _valid_day_mask(valid_day_mask, n_day):
    if valid_day_mask is None:
        return np.ones(int(n_day), dtype=bool)
    return align_mask(valid_day_mask, int(n_day), fill=False)

def _pair_is_valid(day_a, day_b, n_day, valid_days):
    return (
        0 <= int(day_a) < int(n_day)
        and 0 <= int(day_b) < int(n_day)
        and bool(valid_days[int(day_a)])
        and bool(valid_days[int(day_b)])
    )

def _empty_pv_result(condition_parts, n_pair, n_bin):
    return {
        "left": np.full((int(n_pair), int(n_bin)), np.nan, dtype=float),
        "right": np.full((int(n_pair), int(n_bin)), np.nan, dtype=float),
        "averaged": np.full((int(n_pair), int(n_bin)), np.nan, dtype=float),
        "condition_keys": np.array(dict(condition_parts), dtype=object),
        "n_neuron": 0,
    }

def _seed_offset(value):
    return sum((idx + 1) * ord(char) for idx, char in enumerate(str(value)))

def _inference_counts(table, spec):
    finite = np.isfinite(pd.to_numeric(table[spec.value_col], errors="coerce"))
    fit = table.loc[finite]
    return {
        "n_mouse": int(fit[["mouse"]].drop_duplicates().shape[0]) if "mouse" in fit else 0,
        "n_mouse_day": int(fit[["mouse", spec.day_col]].drop_duplicates().shape[0])
        if "mouse" in fit and spec.day_col in fit
        else 0,
        "n_neuron": int(fit[["mouse", "neuron"]].drop_duplicates().shape[0])
        if "mouse" in fit and "neuron" in fit
        else 0,
        "n_observation": int(finite.sum()),
    }

def _ols_params(table, formula, repeat_id_cols):
    import statsmodels.formula.api as smf

    if table.shape[0] < 2:
        return None
    try:
        model = smf.ols(formula, data=table)
        if repeat_id_cols:
            groups = table.loc[:, list(repeat_id_cols)].astype(str).agg("|".join, axis=1)
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        else:
            fit = model.fit(cov_type="HC3")
        return fit
    except Exception:
        return None

def _result_row(spec, table, effect_type, test_family, fit, term, locator):
    row = {column: np.nan for column in CANONICAL_RESULT_COLUMNS}
    row.update(
        {
            "analysis": spec.analysis,
            "view": spec.view,
            "metric": spec.metric,
            "contrast": spec.contrast,
            "effect_type": effect_type,
            "test_family": test_family,
            "test": "cluster OLS" if spec.repeat_id_cols else "OLS HC3",
            "reference": spec.reference_group,
            "comparison": spec.comparison_group,
            "sampling_unit": spec.sampling_unit,
            "aggregation_level": spec.aggregation_level,
            "independence_assumption": spec.independence_assumption,
            "model_formula": getattr(getattr(fit, "model", None), "formula", ""),
            "model_status": "ok" if fit is not None and term in fit.params else "insufficient observations",
            **_inference_counts(table, spec),
        }
    )
    if fit is not None and term in fit.params:
        row["effect"] = float(fit.params[term])
        row["statistic"] = float(fit.tvalues[term]) if np.isfinite(fit.tvalues[term]) else np.nan
        row["p_value"] = float(fit.pvalues[term]) if np.isfinite(fit.pvalues[term]) else np.nan
    row.update(locator)
    return row

def _between_and_slope_rows(model, spec, locator, include_interaction):
    rows = []
    group_formula = f"{spec.value_col} ~ day_centered + C({spec.group_col})"
    group_term = f"C({spec.group_col})[T.{spec.comparison_group}]"
    fit = _ols_params(model, group_formula, spec.repeat_id_cols)
    rows.append(_result_row(spec, model, "between_group_overall", "repeated_linear_model", fit, group_term, locator))
    for group_name in (spec.reference_group, spec.comparison_group):
        group_table = model.loc[model[spec.group_col] == group_name]
        slope_fit = _ols_params(group_table, f"{spec.value_col} ~ day_centered", spec.repeat_id_cols)
        row = _result_row(
            spec,
            group_table,
            "within_group_day_slope",
            "repeated_linear_model",
            slope_fit,
            "day_centered",
            {**locator, "group": group_name},
        )
        row["reference"] = np.nan
        row["comparison"] = np.nan
        rows.append(row)
    if include_interaction:
        formula = f"{spec.value_col} ~ day_centered * C({spec.group_col})"
        term = f"day_centered:C({spec.group_col})[T.{spec.comparison_group}]"
        fit = _ols_params(model, formula, spec.repeat_id_cols)
        rows.append(_result_row(spec, model, "group_day_interaction", "repeated_linear_model", fit, term, locator))
    return rows

def _endpoint_rows(model, spec, locator, first_day, last_day):
    rows = []
    for day, effect_type in ((first_day, "between_group_first_day"), (last_day, "between_group_last_day")):
        day_table = model.loc[model[spec.day_col] == day].copy()
        ref = day_table.loc[day_table[spec.group_col] == spec.reference_group, spec.value_col].to_numpy(dtype=float)
        comp = day_table.loc[day_table[spec.group_col] == spec.comparison_group, spec.value_col].to_numpy(dtype=float)
        row = {column: np.nan for column in CANONICAL_RESULT_COLUMNS}
        effect = float(np.nanmean(comp) - np.nanmean(ref)) if ref.size and comp.size else np.nan
        if ref.size >= 2 and comp.size >= 2:
            statistic, p_value = stats.ttest_ind(comp, ref, equal_var=False)
            status = "ok"
        else:
            statistic, p_value, status = np.nan, np.nan, "insufficient observations"
        row.update(
            {
                "analysis": spec.analysis,
                "view": spec.view,
                "metric": spec.metric,
                "contrast": spec.contrast,
                "effect_type": effect_type,
                "test_family": "independent_two_group",
                "test": "Welch independent t-test",
                "reference": spec.reference_group,
                "comparison": spec.comparison_group,
                "effect": effect,
                "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "sampling_unit": spec.sampling_unit,
                "aggregation_level": spec.aggregation_level,
                "independence_assumption": spec.independence_assumption,
                "model_status": status,
                **_inference_counts(day_table, spec),
            }
        )
        row.update(locator)
        rows.append(row)
    return rows
