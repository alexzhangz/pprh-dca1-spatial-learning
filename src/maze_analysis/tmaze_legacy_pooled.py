"""Manuscript definitions collected from tmaze_legacy_pooled.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import warnings

from collections import OrderedDict

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from sklearn.covariance import LedoitWolf

from sklearn.manifold import SpectralEmbedding

HISTORICAL_GROUP_MICE = OrderedDict(
    [
        ("C", tuple(f"C{i}" for i in range(3, 12))),
        ("I", ("I1", "I2", "I3", "I4", "I5", "I7", "I8", "I9")),
        ("P", tuple(f"P{i}" for i in range(1, 6))),
    ]
)

HISTORICAL_SUBJECT_ORDER = tuple(
    mouse for mice in HISTORICAL_GROUP_MICE.values() for mouse in mice
)

CONDITION_ALIASES = {
    "SL": "SLC",
    "SR": "SRC",
    "CL": "CLC",
    "CR": "CRC",
}

CONDITION_ORDER = ("SL", "SR", "CL", "CR")

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

PAIR_CONTRAST_SPECS = OrderedDict(
    [
        ("sample_vs_choice_left", ("SL", "CL")),
        ("sample_vs_choice_right", ("SR", "CR")),
        ("sample_left_vs_right", ("SL", "SR")),
        ("choice_left_vs_right", ("CL", "CR")),
    ]
)

CONTRAST_SPECS = OrderedDict(
    [
        *PAIR_CONTRAST_SPECS.items(),
        ("sample_vs_choice", ("S", "C")),
    ]
)

CONTRAST_LABELS = {
    "sample_vs_choice_left": "SL vs CL",
    "sample_vs_choice_right": "SR vs CR",
    "sample_left_vs_right": "SL vs SR",
    "choice_left_vs_right": "CL vs CR",
    "sample_vs_choice": "S vs C",
}

TMAZE_SECTIONS = OrderedDict(
    [
        ("Stem", np.arange(0, 13, dtype=int)),
        ("Choice arm", np.arange(13, 29, dtype=int)),
    ]
)

LEGACY_DPRIME_COVARIANCE = "legacy_pooled"

BIN_TABLE_COLUMNS = (
    "group",
    "mouse",
    "day",
    "day_idx",
    "day_centered",
    "contrast",
    "contrast_label",
    "bin",
    "dprime_mean",
    "shuffle_z",
)

SECTION_TABLE_COLUMNS = (
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
    "dprime_mean",
    "shuffle_z",
)

@dataclass(frozen=True)
class LegacyPooledConfig:
    """Frozen numerical contract for the manuscript P1--P5 analysis."""

    days: tuple[int, ...] = (1, 2, 3, 4, 5)
    target_neuron_count: int = 50
    embedding_dim: int = 15
    n_repeat: int = 100
    n_shuffle: int = 100
    reducer: str = "spectral"
    base_seed: int = 2026
    subject_seed_stride: int = 1_000_003
    day_seed_stride: int = 100_003
    min_finite_fraction: float = 0.80
    min_trials_per_condition: int = 3
    zscore: bool = True
    dprime_covariance: str = LEGACY_DPRIME_COVARIANCE

    def validate(self) -> None:
        if self.days != tuple(range(1, len(self.days) + 1)):
            raise ValueError("days must be the consecutive one-based prefix D1..Dn")
        if self.reducer != "spectral":
            raise ValueError("historical contract requires SpectralEmbedding")
        if self.dprime_covariance != LEGACY_DPRIME_COVARIANCE:
            raise ValueError("historical contract requires legacy_pooled covariance")
        for name in ("target_neuron_count", "embedding_dim", "n_repeat", "n_shuffle"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

def historical_subject_seed(mouse: str, config: LegacyPooledConfig | None = None) -> int:
    """Return the full-cohort notebook seed for one mouse."""

    cfg = LegacyPooledConfig() if config is None else config
    try:
        index = HISTORICAL_SUBJECT_ORDER.index(str(mouse))
    except ValueError as exc:
        raise ValueError(f"mouse {mouse!r} is outside the historical subject order") from exc
    return int(cfg.base_seed + index * cfg.subject_seed_stride)

def compute_subject_legacy_pooled(
    subject_dir: str | Path,
    *,
    mouse: str,
    group: str,
    config: LegacyPooledConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute D1--Dn legacy-pooled manifold metrics for one subject in memory."""

    cfg = LegacyPooledConfig() if config is None else config
    cfg.validate()
    subject = Path(subject_dir)
    trial_path = subject / "Snarm_trial_v2.npy"
    index_path = subject / "index_all.npy"
    if not trial_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"missing T-maze v2 inputs under {subject}")

    trial_maps = np.load(trial_path, allow_pickle=True).item()
    sample_days = np.asarray(trial_maps["S"], dtype=object)
    choice_days = np.asarray(trial_maps["C"], dtype=object)
    index_all = np.load(index_path, allow_pickle=True)
    n_day = min(len(cfg.days), sample_days.shape[0], choice_days.shape[0], index_all.shape[1])
    if n_day != len(cfg.days):
        raise ValueError(
            f"{mouse} has only {n_day} usable days but contract requires {len(cfg.days)}"
        )

    first = _as_trial_bin_neuron(sample_days[0], "Snarm_trial_v2['S'][0]")
    n_bin = int(first.shape[1])
    subject_seed = historical_subject_seed(mouse, cfg) if seed is None else int(seed)
    cache: dict[str, Any] = {
        "group": str(group),
        "mouse_name": str(mouse),
        "neuron_mode": f"all{cfg.target_neuron_count}",
        "neuron_selection": "all",
        "condition_aliases": dict(CONDITION_ALIASES),
        "target_neuron_count": int(cfg.target_neuron_count),
        "embedding_dim": int(cfg.embedding_dim),
        "reducer": cfg.reducer,
        "zscore": bool(cfg.zscore),
        "dprime_covariance": cfg.dprime_covariance,
        "n_repeat": int(cfg.n_repeat),
        "n_shuffle": int(cfg.n_shuffle),
        "seed": subject_seed,
        "day_seed_stride": int(cfg.day_seed_stride),
        "min_finite_fraction": float(cfg.min_finite_fraction),
        "min_trials_per_condition": int(cfg.min_trials_per_condition),
        "contrast_names": list(CONTRAST_SPECS),
        "contrast_labels": dict(CONTRAST_LABELS),
        "bin_count_by_contrast": {name: n_bin for name in CONTRAST_SPECS},
        "valid_day": np.zeros(n_day, dtype=bool),
        "n_sl": np.zeros(n_day, dtype=int),
        "n_sr": np.zeros(n_day, dtype=int),
        "n_cl": np.zeros(n_day, dtype=int),
        "n_cr": np.zeros(n_day, dtype=int),
        "sample_vs_choice_balance_count": np.zeros(n_day, dtype=int),
        "sample_vs_choice_class_count": np.zeros(n_day, dtype=int),
        "candidate_neuron_count": np.full(n_day, np.nan),
        "selected_neuron_count": np.full(n_day, np.nan),
        "per_bin_dprime_mean_by_contrast": {
            name: np.full((n_day, n_bin), np.nan, dtype=np.float32)
            for name in CONTRAST_SPECS
        },
        "per_bin_shuffle_z_by_contrast": {
            name: np.full((n_day, n_bin), np.nan, dtype=np.float32)
            for name in CONTRAST_SPECS
        },
    }

    for day_idx in range(n_day):
        trials = _build_condition_trials(
            sample_days[day_idx], choice_days[day_idx], index_all, day_idx=day_idx
        )
        result = compute_day_legacy_pooled(
            trials,
            config=cfg,
            seed=subject_seed + day_idx * cfg.day_seed_stride,
        )
        for label in CONDITION_ORDER:
            cache[f"n_{label.lower()}"][day_idx] = int(result[f"n_{label.lower()}"])
        for name in (
            "sample_vs_choice_balance_count",
            "sample_vs_choice_class_count",
            "candidate_neuron_count",
            "selected_neuron_count",
        ):
            cache[name][day_idx] = result[name]
        if not result["valid"]:
            continue
        cache["valid_day"][day_idx] = True
        for contrast in CONTRAST_SPECS:
            cache["per_bin_dprime_mean_by_contrast"][contrast][day_idx] = (
                result["per_bin_dprime_mean_by_contrast"][contrast]
            )
            cache["per_bin_shuffle_z_by_contrast"][contrast][day_idx] = (
                result["per_bin_shuffle_z_by_contrast"][contrast]
            )
    return cache

def compute_day_legacy_pooled(
    condition_trials: Mapping[str, np.ndarray],
    *,
    config: LegacyPooledConfig | None = None,
    seed: int,
) -> dict[str, Any]:
    """Compute one mouse-day with the historical pooled precision matrices."""

    cfg = LegacyPooledConfig() if config is None else config
    cfg.validate()
    trials = _validated_condition_trials(condition_trials)
    n_bin = int(trials[CONDITION_ORDER[0]].shape[1])
    counts = {f"n_{label.lower()}": int(trials[label].shape[0]) for label in CONDITION_ORDER}
    balance_count = min(counts[f"n_{label.lower()}"] for label in CONDITION_ORDER)
    valid_pair = {
        contrast: (
            trials[a].shape[0] >= cfg.min_trials_per_condition
            and trials[b].shape[0] >= cfg.min_trials_per_condition
        )
        for contrast, (a, b) in PAIR_CONTRAST_SPECS.items()
    }
    valid_svc = balance_count >= cfg.min_trials_per_condition
    if not any((*valid_pair.values(), valid_svc)):
        return _empty_day_result(n_bin, cfg, counts, balance_count)

    slices: dict[str, slice] = {}
    pieces: list[np.ndarray] = []
    start = 0
    for label in CONDITION_ORDER:
        values = trials[label]
        stop = start + values.shape[0]
        slices[label] = slice(start, stop)
        pieces.append(values)
        start = stop
    pooled = np.concatenate(pieces, axis=0)
    n_total_trials = int(pooled.shape[0])
    finite_count = np.isfinite(pooled).sum(axis=(0, 1))
    keep = finite_count >= (n_total_trials * n_bin * cfg.min_finite_fraction)
    candidate_indices = np.where(keep)[0]
    if candidate_indices.size < cfg.target_neuron_count:
        return _empty_day_result(
            n_bin, cfg, counts, balance_count, candidate_count=int(candidate_indices.size)
        )

    embedding_pool = pooled[:, :, candidate_indices]
    if cfg.zscore:
        embedding_pool = _zscore_columns(
            embedding_pool.reshape(n_total_trials * n_bin, candidate_indices.size)
        ).reshape(n_total_trials, n_bin, candidate_indices.size)

    per_bin_real = {
        name: np.full((cfg.n_repeat, n_bin), np.nan, dtype=np.float32)
        for name in CONTRAST_SPECS
    }
    per_bin_shuffle = {
        name: np.full((cfg.n_repeat, cfg.n_shuffle, n_bin), np.nan, dtype=np.float32)
        for name in CONTRAST_SPECS
    }
    rng = np.random.default_rng(int(seed))
    for repeat_idx in range(cfg.n_repeat):
        selected = rng.choice(
            candidate_indices, size=cfg.target_neuron_count, replace=False
        )
        selected_pool = np.searchsorted(candidate_indices, selected)
        points = embedding_pool[:, :, selected_pool].reshape(
            n_total_trials * n_bin, cfg.target_neuron_count
        )
        embedding_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        embedding = _fit_spectral_embedding(points, cfg.embedding_dim, embedding_seed)
        emb_3d = embedding.reshape(n_total_trials, n_bin, cfg.embedding_dim)
        emb_by_condition = {label: emb_3d[slices[label]] for label in CONDITION_ORDER}

        precision_by_contrast: dict[str, list[np.ndarray | None]] = {}
        for contrast, (label_a, label_b) in PAIR_CONTRAST_SPECS.items():
            if not valid_pair[contrast]:
                continue
            class_a = emb_by_condition[label_a]
            class_b = emb_by_condition[label_b]
            precision = _precision_by_bin(class_a, class_b)
            precision_by_contrast[contrast] = precision
            for bin_idx in range(n_bin):
                per_bin_real[contrast][repeat_idx, bin_idx] = _d_prime_from_precision(
                    class_a[:, bin_idx, :], class_b[:, bin_idx, :], precision[bin_idx]
                )

        svc_selected: dict[str, np.ndarray] | None = None
        svc_precision: list[np.ndarray | None] | None = None
        if valid_svc:
            class_sample, class_choice, svc_selected = _balanced_sample_choice_classes(
                emb_by_condition, balance_count, rng
            )
            svc_precision = _precision_by_bin(class_sample, class_choice)
            for bin_idx in range(n_bin):
                per_bin_real["sample_vs_choice"][repeat_idx, bin_idx] = (
                    _d_prime_from_precision(
                        class_sample[:, bin_idx, :],
                        class_choice[:, bin_idx, :],
                        svc_precision[bin_idx],
                    )
                )

        for shuffle_idx in range(cfg.n_shuffle):
            shuffle_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            shuffle_rng = np.random.default_rng(shuffle_seed)
            for contrast, (label_a, label_b) in PAIR_CONTRAST_SPECS.items():
                if not valid_pair[contrast]:
                    continue
                class_a_s, class_b_s = _shuffle_rows(
                    emb_by_condition[label_a], emb_by_condition[label_b], shuffle_rng
                )
                precision = precision_by_contrast[contrast]
                for bin_idx in range(n_bin):
                    per_bin_shuffle[contrast][repeat_idx, shuffle_idx, bin_idx] = (
                        _d_prime_from_precision(
                            class_a_s[:, bin_idx, :],
                            class_b_s[:, bin_idx, :],
                            precision[bin_idx],
                        )
                    )
            if svc_selected is not None and svc_precision is not None:
                sample_s, choice_s = _shuffle_sample_choice_within_side(
                    svc_selected, shuffle_rng
                )
                for bin_idx in range(n_bin):
                    per_bin_shuffle["sample_vs_choice"][repeat_idx, shuffle_idx, bin_idx] = (
                        _d_prime_from_precision(
                            sample_s[:, bin_idx, :],
                            choice_s[:, bin_idx, :],
                            svc_precision[bin_idx],
                        )
                    )

    real_mean_by_contrast: dict[str, np.ndarray] = {}
    shuffle_z_by_contrast: dict[str, np.ndarray] = {}
    for contrast in CONTRAST_SPECS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            real_mean = np.nanmean(per_bin_real[contrast], axis=0)
            shuf_flat = per_bin_shuffle[contrast].reshape(-1, n_bin)
            shuf_mean = np.nanmean(shuf_flat, axis=0)
            shuf_std = np.nanstd(shuf_flat, axis=0, ddof=1)
        real_mean_by_contrast[contrast] = real_mean.astype(np.float32)
        shuffle_z_by_contrast[contrast] = np.divide(
            real_mean - shuf_mean,
            shuf_std,
            out=np.full(n_bin, np.nan, dtype=np.float32),
            where=(shuf_std > 0) & np.isfinite(shuf_std),
        ).astype(np.float32)

    return {
        "valid": True,
        **counts,
        "sample_vs_choice_balance_count": int(balance_count),
        "sample_vs_choice_class_count": int(2 * balance_count),
        "candidate_neuron_count": int(candidate_indices.size),
        "selected_neuron_count": int(cfg.target_neuron_count),
        "per_bin_dprime_mean_by_contrast": real_mean_by_contrast,
        "per_bin_shuffle_z_by_contrast": shuffle_z_by_contrast,
    }

def cache_to_bin_table(
    cache: Mapping[str, Any], *, days: Sequence[int] = (1, 2, 3, 4, 5)
) -> pd.DataFrame:
    """Convert one in-memory subject cache to the historical bin-table schema."""

    selected_days = tuple(int(day) for day in days)
    center = (min(selected_days) + max(selected_days)) / 2.0
    valid_day = np.asarray(cache["valid_day"], dtype=bool)
    rows: list[dict[str, Any]] = []
    for day in selected_days:
        day_idx = day - 1
        if day_idx >= valid_day.size or not valid_day[day_idx]:
            continue
        for contrast in cache["contrast_names"]:
            dprime = np.asarray(
                cache["per_bin_dprime_mean_by_contrast"][contrast][day_idx], dtype=float
            )
            shuffle_z = np.asarray(
                cache["per_bin_shuffle_z_by_contrast"][contrast][day_idx], dtype=float
            )
            for bin_idx, (d_value, z_value) in enumerate(zip(dprime, shuffle_z)):
                if not (np.isfinite(d_value) or np.isfinite(z_value)):
                    continue
                rows.append(
                    {
                        "group": str(cache["group"]),
                        "mouse": str(cache["mouse_name"]),
                        "day": day,
                        "day_idx": day_idx,
                        "day_centered": day - center,
                        "contrast": contrast,
                        "contrast_label": cache["contrast_labels"][contrast],
                        "bin": int(bin_idx),
                        "dprime_mean": float(d_value),
                        "shuffle_z": float(z_value),
                    }
                )
    return pd.DataFrame(rows, columns=BIN_TABLE_COLUMNS)

def bin_to_section_table(
    bin_table: pd.DataFrame,
    *,
    sections: Mapping[str, Sequence[int]] = TMAZE_SECTIONS,
) -> pd.DataFrame:
    """Average the two physical T-maze sections using the historical schema."""

    if bin_table.empty:
        return pd.DataFrame(columns=SECTION_TABLE_COLUMNS)
    group_cols = (
        "group",
        "mouse",
        "day",
        "day_idx",
        "day_centered",
        "contrast",
        "contrast_label",
    )
    rows: list[dict[str, Any]] = []
    for keys, sub in bin_table.groupby(list(group_cols), sort=False, dropna=False):
        base = dict(zip(group_cols, keys))
        for section_idx, (section, bins) in enumerate(sections.items()):
            bin_values = np.asarray(tuple(int(value) for value in bins), dtype=int)
            selected = sub.loc[sub["bin"].isin(bin_values)]
            if selected.empty:
                continue
            rows.append(
                {
                    **base,
                    "section": str(section),
                    "section_idx": int(section_idx),
                    "n_section_bin": int(bin_values.size),
                    "dprime_mean": _nanmean(selected["dprime_mean"]),
                    "shuffle_z": _nanmean(selected["shuffle_z"]),
                }
            )
    return pd.DataFrame(rows, columns=SECTION_TABLE_COLUMNS)

def _build_condition_trials(
    sample_day: np.ndarray,
    choice_day: np.ndarray,
    index_all: np.ndarray,
    *,
    day_idx: int,
) -> dict[str, np.ndarray]:
    sample = _as_trial_bin_neuron(sample_day, "sample_day")
    choice = _as_trial_bin_neuron(choice_day, "choice_day")
    if sample.shape[1:] != choice.shape[1:]:
        raise ValueError("sample and choice trial maps must share bin/neuron axes")
    sources = {"S": sample, "C": choice}
    result: dict[str, np.ndarray] = {}
    for label in CONDITION_ORDER:
        key = CONDITION_ALIASES[label]
        source = sources[CONDITION_PHASE[key]]
        mask = _condition_trial_mask(index_all, day_idx, key)[: source.shape[0]]
        result[label] = source[mask].astype(float, copy=False)
    return result

def _condition_trial_mask(index_all: np.ndarray, day_idx: int, key: str) -> np.ndarray:
    labels = np.asarray(index_all, dtype=bool)
    if labels.ndim != 3 or labels.shape[0] < 3:
        raise ValueError(f"index_all must have shape (3, day, trial), got {labels.shape}")
    left_index = labels[0, int(day_idx)]
    correct_index = labels[1, int(day_idx)]
    real_left_index = labels[2, int(day_idx)]
    masks = {
        "CLC": correct_index & left_index,
        "CRC": correct_index & ~left_index,
        "SLC": correct_index & left_index,
        "SRC": correct_index & ~left_index,
        "CL": ~real_left_index,
        "CR": real_left_index,
        "SLA": left_index,
        "SRA": ~left_index,
    }
    if key not in masks:
        raise ValueError(f"unsupported condition key {key!r}")
    return masks[key].copy()

def _validated_condition_trials(
    condition_trials: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = [label for label in CONDITION_ORDER if label not in condition_trials]
    if missing:
        raise KeyError(f"condition_trials missing labels: {missing}")
    result = {
        label: _as_trial_bin_neuron(condition_trials[label], label)
        for label in CONDITION_ORDER
    }
    if len({values.shape[1:] for values in result.values()}) != 1:
        raise ValueError("all condition trial clouds must share bin/neuron axes")
    return result

def _as_trial_bin_neuron(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (trial, bin, neuron), got {arr.shape}")
    return arr

def _fit_spectral_embedding(points: np.ndarray, embedding_dim: int, seed: int) -> np.ndarray:
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

def _zscore_columns(points: np.ndarray) -> np.ndarray:
    mean = np.nanmean(points, axis=0, keepdims=True)
    std = np.nanstd(points, axis=0, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return np.nan_to_num(
        (points - mean) / std, nan=0.0, posinf=0.0, neginf=0.0
    )

def _precision_by_bin(class_a: np.ndarray, class_b: np.ndarray) -> list[np.ndarray | None]:
    return [
        _pooled_precision(np.vstack([class_a[:, bin_idx, :], class_b[:, bin_idx, :]]))
        for bin_idx in range(class_a.shape[1])
    ]

def _pooled_precision(combined: np.ndarray) -> np.ndarray | None:
    values = _finite_rows(combined)
    if values.shape[0] < 2 or values.shape[1] == 0:
        return None
    if values.shape[0] <= values.shape[1]:
        covariance = np.cov(values, rowvar=False)
    else:
        covariance = LedoitWolf().fit(values).covariance_
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    covariance += np.eye(covariance.shape[0]) * 1e-9
    precision = np.linalg.pinv(covariance)
    return precision if np.isfinite(precision).all() else None

def _d_prime_from_precision(
    class_a: np.ndarray, class_b: np.ndarray, precision: np.ndarray | None
) -> float:
    if precision is None:
        return np.nan
    a = _finite_rows(class_a)
    b = _finite_rows(class_b)
    if a.shape[0] < 2 or b.shape[0] < 2 or a.shape[1] != b.shape[1]:
        return np.nan
    delta = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    value = float(delta @ precision @ delta.T)
    return float(np.sqrt(max(value, 0.0))) if np.isfinite(value) else np.nan

def _finite_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return np.empty((0, 0))
    return arr[np.isfinite(arr).all(axis=1)]

def _balanced_sample_choice_classes(
    embedded: Mapping[str, np.ndarray],
    balance_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    selected: dict[str, np.ndarray] = {}
    for label in CONDITION_ORDER:
        values = np.asarray(embedded[label])
        indices = rng.choice(values.shape[0], size=int(balance_count), replace=False)
        selected[label] = values[indices]
    sample = np.concatenate([selected["SL"], selected["SR"]], axis=0)
    choice = np.concatenate([selected["CL"], selected["CR"]], axis=0)
    return sample, choice, selected

def _shuffle_rows(
    class_a: np.ndarray, class_b: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([class_a, class_b], axis=0)
    shuffled = combined[rng.permutation(combined.shape[0])]
    return shuffled[: class_a.shape[0]], shuffled[class_a.shape[0] :]

def _shuffle_sample_choice_within_side(
    selected: Mapping[str, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    sample_left, choice_left = _shuffle_rows(selected["SL"], selected["CL"], rng)
    sample_right, choice_right = _shuffle_rows(selected["SR"], selected["CR"], rng)
    return (
        np.concatenate([sample_left, sample_right], axis=0),
        np.concatenate([choice_left, choice_right], axis=0),
    )

def _empty_day_result(
    n_bin: int,
    config: LegacyPooledConfig,
    counts: Mapping[str, int],
    balance_count: int,
    *,
    candidate_count: int | float = np.nan,
) -> dict[str, Any]:
    empty = {
        name: np.full(int(n_bin), np.nan, dtype=np.float32)
        for name in CONTRAST_SPECS
    }
    return {
        "valid": False,
        **counts,
        "sample_vs_choice_balance_count": int(balance_count),
        "sample_vs_choice_class_count": int(2 * balance_count),
        "candidate_neuron_count": candidate_count,
        "selected_neuron_count": np.nan,
        "per_bin_dprime_mean_by_contrast": {k: v.copy() for k, v in empty.items()},
        "per_bin_shuffle_z_by_contrast": {k: v.copy() for k, v in empty.items()},
    }

def _nanmean(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else np.nan
