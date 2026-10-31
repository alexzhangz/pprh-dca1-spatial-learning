"""Manuscript definitions collected from fig4_stability.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from .pv_pc_incremental import CACHE_SCHEMA_VERSION, analysis_cache_path

from .manuscript_cohort import (
    DEFAULT_MANUSCRIPT_COHORT_PATH,
    group_mice,
    load_manuscript_cohort,
)

from .spatial import normalize_spatial_rows, spatial_peak_sort_index

DAYS = (1, 2, 3, 4, 5)

GROUP_ORDER = ("C", "I")

LOCKED_GROUP_MICE = group_mice(
    load_manuscript_cohort(DEFAULT_MANUSCRIPT_COHORT_PATH),
    "figure_8_maze",
)

PART_TO_CONDITION = {"left": "CL", "right": "CR"}

CONDITION_TO_PART = {value: key for key, value in PART_TO_CONDITION.items()}

EXPECTED_WITHIN_MISSING = frozenset({("I", "B7", 4)})

SPLIT_STRATEGY = "condition_ordinal_odd_even_v1"

ADJACENT_DAY_PAIRS = ((1, 2), (2, 3), (3, 4), (4, 5))

ANCHOR_DAY_PAIRS = ((1, 2), (1, 3), (1, 4), (1, 5))

INTERDAY_PAIRS = {
    "adjacent": ADJACENT_DAY_PAIRS,
    "anchor": ANCHOR_DAY_PAIRS,
}

ANALYSIS_CONFIGS = {
    "14": {
        "pv_pair": "CL_CR",
        "pv_neuron_mode": "all50",
        "num_day": 5,
        "split_strategy": SPLIT_STRATEGY,
        "min_split_trials": 4,
        "n_subsample_iter": 100,
        "pv_method": "spearman",
        "distance_threshold": 6.0,
        "transition_penalty": 0.25,
        "smooth": True,
    },
    "16": {
        "num_day": 5,
        "condition_keys": ("CL", "CR"),
        "split_strategy": SPLIT_STRATEGY,
        "distance_threshold": 6.0,
        "smooth": True,
        "transition_penalty": 0.25,
        "min_split_trials": 4,
        "pc_method": "pearson",
    },
    "17": {
        "num_day": 5,
        "pv_pair": "CL_CR",
        "pv_neuron_mode": "all50",
        "pair_modes": ("adjacent", "anchor"),
        "n_subsample_iter": 100,
        "pv_method": "spearman",
        "pc_method": "pearson",
        "random_seed": 2026,
    },
}

REQUIRED_PAYLOAD_FIELDS = {
    "14": (
        "left",
        "right",
        "averaged",
        "candidate_neuron_count",
        "valid_day_left",
        "valid_day_right",
        "valid_day_averaged",
        "trial_count_left",
        "trial_count_right",
        "odd_trial_count_left",
        "odd_trial_count_right",
        "even_trial_count_left",
        "even_trial_count_right",
        "split_strategy",
        "pv_method",
    ),
    "16": ("long_table", "qc_table", "map_store", "split_strategy"),
    "17": ("pv_arrays", "candidate_neuron_count", "pc_long"),
}

SHARED_COVARIATE_COLUMNS = (
    "mean_speed_cm_s",
    "route_relative_lateral_deviation_cm",
    "mean_speed_z",
    "route_relative_lateral_deviation_z",
)

def configure_manuscript_cohort(
    path: str | Path = DEFAULT_MANUSCRIPT_COHORT_PATH,
) -> dict[str, Any]:
    """Apply one validated manuscript cohort contract to every Fig. 4 helper.

    ``LOCKED_GROUP_MICE`` is mutated in place so modules that imported the
    mapping before this call continue to observe the configured cohort.  This
    is deliberately a runner-level configuration hook: source construction,
    heatmaps, covariate grids, cache manifests, and validation must enumerate
    the same animals.
    """

    contract = load_manuscript_cohort(path)
    contract_days = tuple(int(day) for day in contract["days"])
    if contract_days != DAYS:
        raise ValueError(
            f"Fig. 4 requires D1-D5, but cohort contract declares {contract_days}"
        )
    configured = group_mice(contract, "figure_8_maze")
    if tuple(configured) != GROUP_ORDER:
        raise ValueError(
            "Fig. 4 cohort contract must provide C and I in project group order"
        )
    LOCKED_GROUP_MICE.clear()
    LOCKED_GROUP_MICE.update(
        {group: tuple(configured[group]) for group in GROUP_ORDER}
    )
    return contract

@dataclass(frozen=True)
class HeatmapPairData:
    """A paired, peak-sorted heatmap payload with exportable row labels."""

    panel: str
    group: str
    condition: str
    day_a: int
    day_b: int
    pair_label: str
    map_a_label: str
    map_b_label: str
    map_a_raw: np.ndarray
    map_b_raw: np.ndarray
    map_a_display: np.ndarray
    map_b_display: np.ndarray
    row_labels: pd.DataFrame

def analysis_config(analysis: str | int) -> dict[str, Any]:
    """Return a defensive copy of the locked exact configuration."""

    key = str(analysis)
    if key not in ANALYSIS_CONFIGS:
        raise ValueError(f"unsupported Fig. 4 analysis {analysis!r}; expected 14, 16, or 17")
    return dict(ANALYSIS_CONFIGS[key])

def load_locked_subject_caches(
    cache_root: str | Path,
    analysis: str | int,
) -> list[dict[str, Any]]:
    """Load and strictly validate every configured manuscript cache envelope.

    Cache paths are derived from the exact D1--D5 configuration, so a cache
    generated with a different split, day range, PV mode, or method cannot be
    silently substituted.
    """

    key = str(analysis)
    config = analysis_config(key)
    records: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        for mouse in LOCKED_GROUP_MICE[group]:
            path = analysis_cache_path(cache_root, key, config, mouse)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing exact-config analysis-{key} cache for {group}/{mouse}: {path}"
                )
            try:
                envelope = np.load(path, allow_pickle=True).item()
            except (OSError, EOFError, ValueError, AttributeError) as error:
                raise ValueError(f"unreadable subject-cache envelope: {path}") from error
            _validate_envelope(envelope, key, group, mouse, path)
            records.append(envelope)
    _validate_locked_record_set(records, key)
    return records

def build_within_pv_bin_source(
    records: Sequence[Mapping[str, Any]],
    *,
    num_bin: int = 56,
) -> pd.DataFrame:
    """Return one analysis-14 row per mouse/day/condition/spatial bin."""

    _validate_locked_record_set(records, "14")
    rows: list[dict[str, Any]] = []
    for record in records:
        group, mouse = str(record["group"]), str(record["mouse"])
        payload = record["payload"]
        _validate_analysis14_payload(payload, group, mouse, num_bin=num_bin)
        for part, condition in PART_TO_CONDITION.items():
            values = np.asarray(payload[part], dtype=float)
            for day_index, day in enumerate(DAYS):
                valid_day = bool(payload[f"valid_day_{part}"][day_index])
                for bin_index, raw_r in enumerate(values[day_index]):
                    rows.append(
                        {
                            "group": group,
                            "mouse": mouse,
                            "day": day,
                            "day_c": float(day - 3),
                            "condition_part": part,
                            "condition": condition,
                            "spatial_bin": int(bin_index),
                            "raw_r": _finite_or_nan(raw_r),
                            "fisher_z": _scalar_fisher_z(raw_r),
                            "valid_day": valid_day,
                            "trial_count": int(payload[f"trial_count_{part}"][day_index]),
                            "odd_trial_count": int(
                                payload[f"odd_trial_count_{part}"][day_index]
                            ),
                            "even_trial_count": int(
                                payload[f"even_trial_count_{part}"][day_index]
                            ),
                            "candidate_neuron_count": _finite_or_nan(
                                payload["candidate_neuron_count"][day_index]
                            ),
                            "fixed_neuron_target": 50,
                            "neuron_subsample_iterations": 100,
                            "pv_method": "spearman",
                            "split_strategy": SPLIT_STRATEGY,
                            "missing_reason": (
                                "insufficient_odd_even_trials" if not valid_day else ""
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    _assert_unique(
        result,
        ["group", "mouse", "day", "condition", "spatial_bin"],
        "within-session PV-bin source",
    )
    _validate_within_missingness(result, value_col="raw_r")
    return result

def build_within_pv_mouse_day_source(pv_bin_source: pd.DataFrame) -> pd.DataFrame:
    """Average raw bin correlations, then Fisher-transform, per mouse-day."""

    required = {
        "group",
        "mouse",
        "day",
        "day_c",
        "condition_part",
        "condition",
        "spatial_bin",
        "raw_r",
        "valid_day",
        "trial_count",
        "odd_trial_count",
        "even_trial_count",
        "candidate_neuron_count",
    }
    _require_columns(pv_bin_source, required, "within-session PV-bin source")
    _assert_unique(
        pv_bin_source,
        ["group", "mouse", "day", "condition", "spatial_bin"],
        "within-session PV-bin source",
    )
    rows: list[dict[str, Any]] = []
    keys = ["group", "mouse", "day", "day_c", "condition_part", "condition"]
    for key, table in pv_bin_source.groupby(keys, observed=True, sort=False):
        raw = pd.to_numeric(table["raw_r"], errors="coerce").to_numpy(dtype=float)
        finite = raw[np.isfinite(raw)]
        raw_mean = float(np.mean(finite)) if finite.size else np.nan
        rows.append(
            {
                **dict(zip(keys, key)),
                "metric": "within_session_pv_correlation",
                "raw_r": raw_mean,
                "fisher_z": _scalar_fisher_z(raw_mean),
                "n_bin_finite": int(finite.size),
                "valid_day": bool(table["valid_day"].iloc[0]),
                "trial_count": int(table["trial_count"].iloc[0]),
                "odd_trial_count": int(table["odd_trial_count"].iloc[0]),
                "even_trial_count": int(table["even_trial_count"].iloc[0]),
                "candidate_neuron_count": _finite_or_nan(
                    table["candidate_neuron_count"].iloc[0]
                ),
                "split_strategy": SPLIT_STRATEGY,
                "missing_reason": (
                    "" if finite.size else "insufficient_odd_even_trials"
                ),
            }
        )
    result = pd.DataFrame(rows)
    _assert_unique(
        result,
        ["group", "mouse", "day", "condition"],
        "within-session PV mouse-day source",
    )
    _validate_within_missingness(result, value_col="raw_r")
    return result

def build_within_single_neuron_source(
    records: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return analysis-16 neuron rows and their unique mouse-day QC table."""

    _validate_locked_record_set(records, "16")
    source_frames: list[pd.DataFrame] = []
    qc_frames: list[pd.DataFrame] = []
    for record in records:
        group, mouse = str(record["group"]), str(record["mouse"])
        payload = record["payload"]
        _validate_analysis16_payload(payload, group, mouse)
        table = payload["long_table"].copy()
        table = table.loc[
            table["day"].isin(DAYS) & table["condition"].isin(PART_TO_CONDITION.values())
        ].copy()
        table["group"] = group
        table["mouse"] = mouse
        table["day"] = pd.to_numeric(table["day"], errors="raise").astype(int)
        table["day_c"] = table["day"] - 3.0
        table = table.rename(columns={"pc_within_corr": "raw_r"})
        table["fisher_z"] = fisher_z(table["raw_r"].to_numpy(dtype=float))
        table["condition_part"] = table["condition"].map(CONDITION_TO_PART)
        table["split_strategy"] = SPLIT_STRATEGY
        source_frames.append(
            table[
                [
                    "group",
                    "mouse",
                    "day",
                    "day_c",
                    "condition_part",
                    "condition",
                    "neuron",
                    "raw_r",
                    "fisher_z",
                    "split_strategy",
                ]
            ]
        )

        qc = payload["qc_table"].copy()
        qc = qc.loc[
            qc["day"].isin(DAYS) & qc["condition"].isin(PART_TO_CONDITION.values())
        ].copy()
        qc["group"] = group
        qc["mouse"] = mouse
        qc["condition_part"] = qc["condition"].map(CONDITION_TO_PART)
        qc_frames.append(qc)

    source = pd.concat(source_frames, ignore_index=True)
    qc = pd.concat(qc_frames, ignore_index=True)
    unit = ["group", "mouse", "day", "condition", "neuron"]
    _assert_unique(source, unit, "within-session single-neuron source")
    _assert_unique(qc, ["group", "mouse", "day", "condition"], "analysis-16 QC source")
    source = source.merge(
        qc[
            [
                "group",
                "mouse",
                "day",
                "condition",
                "n_neuron",
                "trial_count",
                "odd_trial_count",
                "even_trial_count",
                "finite_pc_corr_count",
            ]
        ],
        on=["group", "mouse", "day", "condition"],
        how="left",
        validate="many_to_one",
    )
    source["missing_reason"] = np.where(
        np.isfinite(source["raw_r"]), "", "insufficient_odd_even_trials"
    )
    _validate_within_missingness(source, value_col="raw_r")
    return source, qc

def build_interday_pv_bin_source(
    records: Sequence[Mapping[str, Any]],
    *,
    num_bin: int = 56,
) -> pd.DataFrame:
    """Return one analysis-17 row per mouse/pair/condition/spatial bin."""

    _validate_locked_record_set(records, "17")
    rows: list[dict[str, Any]] = []
    for record in records:
        group, mouse = str(record["group"]), str(record["mouse"])
        payload = record["payload"]
        _validate_analysis17_payload(payload, group, mouse, num_bin=num_bin)
        for pair_mode, pairs in INTERDAY_PAIRS.items():
            for part, condition in PART_TO_CONDITION.items():
                values = np.asarray(payload["pv_arrays"][pair_mode][part], dtype=float)
                candidates = np.asarray(
                    payload["candidate_neuron_count"][pair_mode], dtype=float
                )
                for pair_index, (day_a, day_b) in enumerate(pairs):
                    for bin_index, raw_r in enumerate(values[pair_index]):
                        rows.append(
                            {
                                "group": group,
                                "mouse": mouse,
                                "pair_mode": pair_mode,
                                "pair_index": pair_index,
                                "day_a": day_a,
                                "day_b": day_b,
                                "day_b_c": float(day_b - 3.5),
                                "day_pair_label": f"D{day_a}-D{day_b}",
                                "condition_part": part,
                                "condition": condition,
                                "spatial_bin": int(bin_index),
                                "raw_r": _finite_or_nan(raw_r),
                                "fisher_z": _scalar_fisher_z(raw_r),
                                "candidate_neuron_count": _finite_or_nan(
                                    candidates[pair_index]
                                ),
                                "fixed_neuron_target": 50,
                                "neuron_subsample_iterations": 100,
                                "pv_method": "spearman",
                            }
                        )
    result = pd.DataFrame(rows)
    unit = [
        "group",
        "mouse",
        "pair_mode",
        "day_a",
        "day_b",
        "condition",
        "spatial_bin",
    ]
    _assert_unique(result, unit, "inter-day PV-bin source")
    _validate_one_based_interday_rows(result, "inter-day PV-bin source")
    return result

def build_interday_pv_mouse_pair_source(pv_bin_source: pd.DataFrame) -> pd.DataFrame:
    """Average raw bin correlations, then Fisher-transform, per mouse-pair."""

    required = {
        "group",
        "mouse",
        "pair_mode",
        "pair_index",
        "day_a",
        "day_b",
        "day_b_c",
        "day_pair_label",
        "condition_part",
        "condition",
        "spatial_bin",
        "raw_r",
        "candidate_neuron_count",
    }
    _require_columns(pv_bin_source, required, "inter-day PV-bin source")
    _validate_one_based_interday_rows(pv_bin_source, "inter-day PV-bin source")
    keys = [
        "group",
        "mouse",
        "pair_mode",
        "pair_index",
        "day_a",
        "day_b",
        "day_b_c",
        "day_pair_label",
        "condition_part",
        "condition",
    ]
    rows: list[dict[str, Any]] = []
    for key, table in pv_bin_source.groupby(keys, observed=True, sort=False):
        raw = pd.to_numeric(table["raw_r"], errors="coerce").to_numpy(dtype=float)
        finite = raw[np.isfinite(raw)]
        raw_mean = float(np.mean(finite)) if finite.size else np.nan
        rows.append(
            {
                **dict(zip(keys, key)),
                "metric": "interday_pv_correlation",
                "raw_r": raw_mean,
                "fisher_z": _scalar_fisher_z(raw_mean),
                "n_bin_finite": int(finite.size),
                "candidate_neuron_count": _finite_or_nan(
                    table["candidate_neuron_count"].iloc[0]
                ),
            }
        )
    result = pd.DataFrame(rows)
    _assert_unique(
        result,
        ["group", "mouse", "pair_mode", "day_a", "day_b", "condition"],
        "inter-day PV mouse-pair source",
    )
    _validate_one_based_interday_rows(result, "inter-day PV mouse-pair source")
    return result

def build_interday_single_neuron_source(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return analysis-17 neuron rows without modifying one-based day labels."""

    _validate_locked_record_set(records, "17")
    frames: list[pd.DataFrame] = []
    for record in records:
        group, mouse = str(record["group"]), str(record["mouse"])
        payload = record["payload"]
        _validate_analysis17_payload(payload, group, mouse)
        for pair_mode, pairs in INTERDAY_PAIRS.items():
            table = payload["pc_long"][pair_mode].copy()
            table = table.loc[table["condition"].isin(PART_TO_CONDITION)].copy()
            # day_a/day_b in this table are already one-based; never add 1 here.
            table["group"] = group
            table["mouse"] = mouse
            table["pair_mode"] = pair_mode
            table["day_a"] = pd.to_numeric(table["day_a"], errors="raise").astype(int)
            table["day_b"] = pd.to_numeric(table["day_b"], errors="raise").astype(int)
            expected = set(pairs)
            observed = set(table[["day_a", "day_b"]].itertuples(index=False, name=None))
            if observed != expected:
                raise ValueError(
                    f"analysis-17 {group}/{mouse}/{pair_mode} has noncanonical one-based "
                    f"pairs: expected={sorted(expected)}, observed={sorted(observed)}"
                )
            table["day_b_c"] = table["day_b"] - 3.5
            table["condition_part"] = table["condition"]
            table["condition"] = table["condition_part"].map(PART_TO_CONDITION)
            table = table.rename(columns={"pc_interday_corr": "raw_r"})
            table["fisher_z"] = fisher_z(table["raw_r"].to_numpy(dtype=float))
            frames.append(
                table[
                    [
                        "group",
                        "mouse",
                        "pair_mode",
                        "pair_index",
                        "day_a",
                        "day_b",
                        "day_b_c",
                        "day_pair_label",
                        "condition_part",
                        "condition",
                        "condition_key",
                        "neuron",
                        "raw_r",
                        "fisher_z",
                    ]
                ]
            )
    result = pd.concat(frames, ignore_index=True)
    unit = [
        "group",
        "mouse",
        "pair_mode",
        "day_a",
        "day_b",
        "condition",
        "neuron",
    ]
    _assert_unique(result, unit, "inter-day single-neuron source")
    _validate_one_based_interday_rows(result, "inter-day single-neuron source")
    return result

def prepare_shared_within_day_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the configured Figure-8 D1--D5 covariate grid."""

    required = {"group", "mouse", "day", *SHARED_COVARIATE_COLUMNS}
    _require_columns(covariates, required, "shared behavior covariates")
    table = covariates.copy()
    if "project" in table:
        table = table.loc[table["project"].astype(str).eq("Figure-8")].copy()
    allowed_mice = {mouse for mice in LOCKED_GROUP_MICE.values() for mouse in mice}
    table = table.loc[
        table["group"].isin(GROUP_ORDER)
        & table["mouse"].isin(allowed_mice)
        & table["day"].isin(DAYS)
    ].copy()
    table["day"] = pd.to_numeric(table["day"], errors="raise").astype(int)
    _assert_unique(table, ["group", "mouse", "day"], "shared behavior covariates")
    expected = {
        (group, mouse, day)
        for group in GROUP_ORDER
        for mouse in LOCKED_GROUP_MICE[group]
        for day in DAYS
    }
    observed = set(table[["group", "mouse", "day"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            "shared behavior covariates do not match the configured D1-D5 cohort grid; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for column in SHARED_COVARIATE_COLUMNS:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"shared behavior covariate {column!r} contains nonfinite values")
    keep = ["group", "mouse", "day", *SHARED_COVARIATE_COLUMNS]
    optional = [
        column
        for column in (
            "covariate_raw_frame_count",
            "covariate_running_frame_count",
            "covariate_valid_frame_count",
            "covariate_status",
        )
        if column in table
    ]
    return table[keep + optional].sort_values(["group", "mouse", "day"]).reset_index(
        drop=True
    )

def attach_within_day_covariates(
    source: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach shared mouse-day movement covariates to any within-day source."""

    _require_columns(source, {"group", "mouse", "day"}, "within-day source")
    prepared = prepare_shared_within_day_covariates(covariates)
    overlap = [column for column in prepared.columns if column in source and column not in {"group", "mouse", "day"}]
    if overlap:
        raise ValueError(f"within-day source already contains covariate columns: {overlap}")
    result = source.merge(
        prepared,
        on=["group", "mouse", "day"],
        how="left",
        validate="many_to_one",
    )
    if result[list(SHARED_COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("within-day covariate merge produced unmatched rows")
    return result

def build_interday_pair_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    """Average endpoint session z-scores for each canonical inter-day pair.

    Endpoint raw values and endpoint z-scores are retained.  The unsuffixed
    ``mean_speed_z`` and ``route_relative_lateral_deviation_z`` columns are the
    prespecified pair covariates: the arithmetic mean of the two session-level
    z-scores.
    """

    within = prepare_shared_within_day_covariates(covariates)
    indexed = within.set_index(["group", "mouse", "day"], verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        for mouse in LOCKED_GROUP_MICE[group]:
            for pair_mode, pairs in INTERDAY_PAIRS.items():
                for pair_index, (day_a, day_b) in enumerate(pairs):
                    endpoint_a = indexed.loc[(group, mouse, day_a)]
                    endpoint_b = indexed.loc[(group, mouse, day_b)]
                    row: dict[str, Any] = {
                        "group": group,
                        "mouse": mouse,
                        "pair_mode": pair_mode,
                        "pair_index": pair_index,
                        "day_a": day_a,
                        "day_b": day_b,
                        "day_b_c": float(day_b - 3.5),
                        "day_pair_label": f"D{day_a}-D{day_b}",
                        "covariate_pair_rule": "mean_of_two_session_z_scores",
                    }
                    for column in SHARED_COVARIATE_COLUMNS:
                        row[f"{column}_day_a"] = float(endpoint_a[column])
                        row[f"{column}_day_b"] = float(endpoint_b[column])
                    for column in (
                        "mean_speed_z",
                        "route_relative_lateral_deviation_z",
                    ):
                        row[column] = float(
                            np.mean([endpoint_a[column], endpoint_b[column]])
                        )
                    rows.append(row)
    result = pd.DataFrame(rows)
    _assert_unique(
        result,
        ["group", "mouse", "pair_mode", "day_a", "day_b"],
        "inter-day pair covariates",
    )
    _validate_one_based_interday_rows(result, "inter-day pair covariates")
    return result

def attach_interday_pair_covariates(
    source: pd.DataFrame,
    pair_covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach canonical pair-mean covariates without changing day labels."""

    keys = ["group", "mouse", "pair_mode", "day_a", "day_b"]
    _require_columns(source, set(keys), "inter-day source")
    _require_columns(
        pair_covariates,
        {
            *keys,
            "mean_speed_z",
            "route_relative_lateral_deviation_z",
            "mean_speed_z_day_a",
            "mean_speed_z_day_b",
            "route_relative_lateral_deviation_z_day_a",
            "route_relative_lateral_deviation_z_day_b",
        },
        "inter-day pair covariates",
    )
    _assert_unique(pair_covariates, keys, "inter-day pair covariates")
    protected_covariates = {
        "mean_speed_z",
        "route_relative_lateral_deviation_z",
        "mean_speed_z_day_a",
        "mean_speed_z_day_b",
        "route_relative_lateral_deviation_z_day_a",
        "route_relative_lateral_deviation_z_day_b",
    }
    overlap = sorted(protected_covariates.intersection(source.columns))
    if overlap:
        raise ValueError(f"inter-day source already contains pair-covariate columns: {overlap}")
    # Pair identity fields such as pair_index/day_pair_label are intentionally
    # present in both tables.  Keep the source copy and merge only new
    # covariate/provenance columns to avoid suffixing equivalent metadata.
    merge_columns = keys + [
        column
        for column in pair_covariates.columns
        if column not in keys and column not in source.columns
    ]
    result = source.merge(
        pair_covariates[merge_columns], on=keys, how="left", validate="many_to_one"
    )
    if result[["mean_speed_z", "route_relative_lateral_deviation_z"]].isna().any().any():
        raise ValueError("inter-day covariate merge produced unmatched rows")
    _validate_one_based_interday_rows(result, "inter-day source with covariates")
    return result

def prepare_within_left_heatmaps(
    records: Sequence[Mapping[str, Any]],
    *,
    num_bin: int = 56,
) -> list[HeatmapPairData]:
    """Prepare Fig. 4D left-turn odd/even heatmaps for every D1--D5 day."""

    _validate_locked_record_set(records, "16")
    for record in records:
        _validate_analysis16_payload(
            record["payload"], str(record["group"]), str(record["mouse"]), num_bin=num_bin
        )
    output: list[HeatmapPairData] = []
    for group in GROUP_ORDER:
        group_records = [record for record in records if str(record["group"]) == group]
        for day in DAYS:
            map_a_parts: list[np.ndarray] = []
            map_b_parts: list[np.ndarray] = []
            labels: list[dict[str, Any]] = []
            for record in group_records:
                mouse = str(record["mouse"])
                entry = record["payload"]["map_store"].get((day, "CL"))
                if entry is None:
                    if (group, mouse, day) not in EXPECTED_WITHIN_MISSING:
                        raise ValueError(
                            f"analysis-16 map_store unexpectedly lacks {group}/{mouse}/D{day}/CL"
                        )
                    continue
                odd = _stack_subject_map_entry(entry, "odd", num_bin=num_bin)
                even = _stack_subject_map_entry(entry, "even", num_bin=num_bin)
                if odd.shape != even.shape:
                    raise ValueError(f"odd/even heatmap shape mismatch for {group}/{mouse}/D{day}")
                entry_mice = list(entry["mouse"])
                entry_neurons = list(entry["neuron"])
                if len(entry_mice) != odd.shape[0] or len(entry_neurons) != odd.shape[0]:
                    raise ValueError(f"map_store row labels mismatch for {group}/{mouse}/D{day}")
                map_a_parts.append(odd)
                map_b_parts.append(even)
                labels.extend(
                    {
                        "group": group,
                        "mouse": str(label_mouse),
                        "day": day,
                        "condition": "CL",
                        "neuron": int(neuron),
                    }
                    for label_mouse, neuron in zip(entry_mice, entry_neurons)
                )
            map_a = np.vstack(map_a_parts)
            map_b = np.vstack(map_b_parts)
            output.append(
                prepare_sorted_heatmap_pair(
                    map_a,
                    map_b,
                    pd.DataFrame(labels),
                    panel="D",
                    group=group,
                    condition="CL",
                    day_a=day,
                    day_b=day,
                    pair_label=f"D{day}",
                    map_a_label="Odd trials",
                    map_b_label="Even trials",
                )
            )
    return output

def load_registered_left_maps(
    records: Sequence[Mapping[str, Any]],
    aligned_root: str | Path,
    *,
    filename: str = "snarm_all.npy",
    condition: str = "CL",
    num_bin: int = 56,
) -> dict[tuple[str, str], np.ndarray]:
    """Load registered D1--D5 left-condition maps for the locked cohort."""

    _validate_locked_record_set(records, "17")
    aligned_root = Path(aligned_root)
    maps: dict[tuple[str, str], np.ndarray] = {}
    for record in records:
        group, mouse = str(record["group"]), str(record["mouse"])
        path = aligned_root / mouse / filename
        if not path.exists():
            raise FileNotFoundError(path)
        payload = np.load(path, allow_pickle=True).item()
        if not isinstance(payload, Mapping) or condition not in payload:
            raise ValueError(f"{path} does not contain condition {condition!r}")
        values = np.asarray(payload[condition], dtype=float)
        if values.ndim != 3 or values.shape[0] < len(DAYS) or values.shape[2] < num_bin:
            raise ValueError(
                f"{path}/{condition} must have shape (>=5, neuron, >=56), got {values.shape}"
            )
        maps[(group, mouse)] = values[: len(DAYS), :, :num_bin]
    return maps

def prepare_adjacent_left_heatmaps(
    subject_maps: Mapping[tuple[str, str], np.ndarray],
    *,
    num_bin: int = 56,
) -> list[HeatmapPairData]:
    """Prepare Fig. 4H registered-neuron heatmaps for all adjacent day pairs."""

    expected_subjects = {
        (group, mouse) for group in GROUP_ORDER for mouse in LOCKED_GROUP_MICE[group]
    }
    observed_subjects = {(str(group), str(mouse)) for group, mouse in subject_maps}
    if observed_subjects != expected_subjects:
        raise ValueError(
            "registered-map subjects do not match the configured manuscript cohort; "
            f"missing={sorted(expected_subjects-observed_subjects)}, "
            f"extra={sorted(observed_subjects-expected_subjects)}"
        )
    output: list[HeatmapPairData] = []
    for group in GROUP_ORDER:
        for day_a, day_b in ADJACENT_DAY_PAIRS:
            earlier_parts: list[np.ndarray] = []
            later_parts: list[np.ndarray] = []
            labels: list[dict[str, Any]] = []
            for mouse in LOCKED_GROUP_MICE[group]:
                values = np.asarray(subject_maps[(group, mouse)], dtype=float)
                if values.ndim != 3 or values.shape[0] < len(DAYS) or values.shape[2] < num_bin:
                    raise ValueError(
                        f"registered maps for {group}/{mouse} must have shape "
                        f"(>=5, neuron, >={num_bin}), got {values.shape}"
                    )
                earlier = values[day_a - 1, :, :num_bin]
                later = values[day_b - 1, :, :num_bin]
                if earlier.shape != later.shape:
                    raise ValueError(f"registered map shape mismatch for {group}/{mouse}")
                earlier_parts.append(earlier)
                later_parts.append(later)
                labels.extend(
                    {
                        "group": group,
                        "mouse": mouse,
                        "day_a": day_a,
                        "day_b": day_b,
                        "condition": "CL",
                        "neuron": int(neuron),
                    }
                    for neuron in range(earlier.shape[0])
                )
            output.append(
                prepare_sorted_heatmap_pair(
                    np.vstack(earlier_parts),
                    np.vstack(later_parts),
                    pd.DataFrame(labels),
                    panel="H",
                    group=group,
                    condition="CL",
                    day_a=day_a,
                    day_b=day_b,
                    pair_label=f"D{day_a}-D{day_b}",
                    map_a_label=f"D{day_a}",
                    map_b_label=f"D{day_b}",
                )
            )
    return output

def prepare_sorted_heatmap_pair(
    map_a: np.ndarray,
    map_b: np.ndarray,
    row_labels: pd.DataFrame,
    *,
    panel: str,
    group: str,
    condition: str,
    day_a: int,
    day_b: int,
    pair_label: str,
    map_a_label: str,
    map_b_label: str,
) -> HeatmapPairData:
    """Filter paired rows, sort on map A's peak, and normalize each map row."""

    map_a = np.asarray(map_a, dtype=float)
    map_b = np.asarray(map_b, dtype=float)
    if map_a.ndim != 2 or map_b.ndim != 2 or map_a.shape != map_b.shape:
        raise ValueError(
            f"paired heatmaps must be equal 2D arrays, got {map_a.shape} and {map_b.shape}"
        )
    labels = row_labels.copy().reset_index(drop=True)
    if len(labels) != map_a.shape[0]:
        raise ValueError("heatmap row_labels length does not match matrix rows")
    finite_rows = np.isfinite(map_a).any(axis=1) & np.isfinite(map_b).any(axis=1)
    map_a = map_a[finite_rows]
    map_b = map_b[finite_rows]
    labels = labels.loc[finite_rows].reset_index(drop=True)
    if map_a.shape[0] == 0:
        raise ValueError(f"no jointly finite rows for heatmap {panel}/{group}/{pair_label}")
    order = spatial_peak_sort_index(map_a)
    map_a_display = normalize_spatial_rows(map_a)[order]
    map_b_display = normalize_spatial_rows(map_b)[order]
    map_a_raw = map_a[order]
    map_b_raw = map_b[order]
    labels = labels.iloc[order].reset_index(drop=True)
    labels.insert(0, "display_row", np.arange(len(labels), dtype=int))
    labels["earlier_peak_bin"] = [
        int(np.nanargmax(row)) if np.isfinite(row).any() else -1 for row in map_a_display
    ]
    _assert_unique(
        labels,
        [column for column in ("group", "mouse", "day", "day_a", "day_b", "condition", "neuron") if column in labels],
        f"heatmap {panel}/{group}/{pair_label} row labels",
    )
    return HeatmapPairData(
        panel=str(panel),
        group=str(group),
        condition=str(condition),
        day_a=int(day_a),
        day_b=int(day_b),
        pair_label=str(pair_label),
        map_a_label=str(map_a_label),
        map_b_label=str(map_b_label),
        map_a_raw=map_a_raw,
        map_b_raw=map_b_raw,
        map_a_display=map_a_display,
        map_b_display=map_b_display,
        row_labels=labels,
    )

def fisher_z(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Fisher ``atanh(r)`` while retaining NaNs and bounding |r| at 1."""

    result = np.asarray(values, dtype=float)
    finite = np.isfinite(result)
    if np.any(np.abs(result[finite]) > 1.0 + 1e-12):
        raise ValueError("correlation values must lie within [-1, 1]")
    bound = np.nextafter(1.0, 0.0)
    clipped = np.clip(result, -bound, bound)
    transformed = np.full(result.shape, np.nan, dtype=float)
    transformed[finite] = np.arctanh(clipped[finite])
    return transformed

def _validate_envelope(
    envelope: Any,
    analysis: str,
    group: str,
    mouse: str,
    path: Path,
) -> None:
    if not isinstance(envelope, Mapping):
        raise ValueError(f"subject cache is not a mapping: {path}")
    required = {
        "schema_version",
        "analysis",
        "mouse",
        "group",
        "config",
        "source_fingerprint",
        "status",
        "payload",
    }
    missing = sorted(required.difference(envelope))
    if missing:
        raise ValueError(f"subject cache is missing envelope fields {missing}: {path}")
    if envelope["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(f"subject cache schema mismatch: {path}")
    if str(envelope["analysis"]) != analysis:
        raise ValueError(f"subject cache analysis mismatch: {path}")
    if str(envelope["group"]) != group or str(envelope["mouse"]) != mouse:
        raise ValueError(
            f"subject cache identity mismatch at {path}: "
            f"expected {group}/{mouse}, got {envelope.get('group')}/{envelope.get('mouse')}"
        )
    if envelope["status"] != "complete":
        raise ValueError(f"subject cache is not complete: {path}")
    if _canonical(envelope["config"]) != _canonical(ANALYSIS_CONFIGS[analysis]):
        raise ValueError(f"subject cache exact configuration mismatch: {path}")
    if not isinstance(envelope["payload"], Mapping):
        raise ValueError(f"subject cache payload is not a mapping: {path}")
    missing_payload = sorted(
        set(REQUIRED_PAYLOAD_FIELDS[analysis]).difference(envelope["payload"])
    )
    if missing_payload:
        raise ValueError(f"subject cache payload missing {missing_payload}: {path}")
    if analysis == "14":
        _validate_analysis14_payload(envelope["payload"], group, mouse)
    elif analysis == "16":
        _validate_analysis16_payload(envelope["payload"], group, mouse)
    else:
        _validate_analysis17_payload(envelope["payload"], group, mouse)

def _validate_locked_record_set(
    records: Sequence[Mapping[str, Any]],
    analysis: str,
) -> None:
    expected = {
        (group, mouse) for group in GROUP_ORDER for mouse in LOCKED_GROUP_MICE[group]
    }
    identities: list[tuple[str, str]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("cache records must be mappings")
        if str(record.get("analysis")) != analysis:
            raise ValueError(f"expected analysis-{analysis} cache records")
        if _canonical(record.get("config")) != _canonical(ANALYSIS_CONFIGS[analysis]):
            raise ValueError(f"analysis-{analysis} record configuration mismatch")
        identities.append((str(record.get("group")), str(record.get("mouse"))))
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate analysis-{analysis} subject-cache records")
    observed = set(identities)
    if observed != expected:
        raise ValueError(
            f"analysis-{analysis} records do not match the configured manuscript cohort; "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )

def _validate_analysis14_payload(
    payload: Mapping[str, Any],
    group: str,
    mouse: str,
    *,
    num_bin: int = 56,
) -> None:
    _require_mapping_fields(payload, REQUIRED_PAYLOAD_FIELDS["14"], "analysis-14 payload")
    if payload["split_strategy"] != SPLIT_STRATEGY or payload["pv_method"] != "spearman":
        raise ValueError(f"analysis-14 split/method mismatch for {group}/{mouse}")
    for part in ("left", "right", "averaged"):
        values = np.asarray(payload[part], dtype=float)
        if values.shape != (len(DAYS), num_bin):
            raise ValueError(
                f"analysis-14 {group}/{mouse}/{part} shape {values.shape}; "
                f"expected {(len(DAYS), num_bin)}"
            )
        valid = np.asarray(payload[f"valid_day_{part}"], dtype=bool)
        if valid.shape != (len(DAYS),):
            raise ValueError(f"analysis-14 valid-day shape mismatch for {group}/{mouse}/{part}")
        finite = np.isfinite(values).any(axis=1)
        if not np.array_equal(valid, finite):
            raise ValueError(f"analysis-14 valid-day flag disagrees with data for {group}/{mouse}/{part}")
    for key in (
        "candidate_neuron_count",
        "trial_count_left",
        "trial_count_right",
        "odd_trial_count_left",
        "odd_trial_count_right",
        "even_trial_count_left",
        "even_trial_count_right",
    ):
        if np.asarray(payload[key]).shape != (len(DAYS),):
            raise ValueError(f"analysis-14 {key} shape mismatch for {group}/{mouse}")
    expected_missing = EXPECTED_WITHIN_MISSING.intersection({(group, mouse, day) for day in DAYS})
    for part in ("left", "right"):
        observed_missing = {
            (group, mouse, day)
            for day, valid in zip(DAYS, payload[f"valid_day_{part}"])
            if not bool(valid)
        }
        if observed_missing != expected_missing:
            raise ValueError(
                f"analysis-14 unexpected {part} within-session missingness for "
                f"{group}/{mouse}: expected={sorted(expected_missing)}, "
                f"observed={sorted(observed_missing)}"
            )

def _validate_analysis16_payload(
    payload: Mapping[str, Any],
    group: str,
    mouse: str,
    *,
    num_bin: int = 56,
) -> None:
    _require_mapping_fields(payload, REQUIRED_PAYLOAD_FIELDS["16"], "analysis-16 payload")
    if payload["split_strategy"] != SPLIT_STRATEGY:
        raise ValueError(f"analysis-16 split strategy mismatch for {group}/{mouse}")
    long_table = payload["long_table"]
    qc = payload["qc_table"]
    if not isinstance(long_table, pd.DataFrame) or not isinstance(qc, pd.DataFrame):
        raise ValueError(f"analysis-16 tables must be DataFrames for {group}/{mouse}")
    _require_columns(
        long_table,
        {"group", "mouse", "day", "condition", "neuron", "pc_within_corr"},
        f"analysis-16 long table {group}/{mouse}",
    )
    _require_columns(
        qc,
        {
            "group",
            "mouse",
            "day",
            "condition",
            "n_neuron",
            "trial_count",
            "odd_trial_count",
            "even_trial_count",
            "finite_pc_corr_count",
        },
        f"analysis-16 QC {group}/{mouse}",
    )
    selected = long_table.loc[
        long_table["day"].isin(DAYS) & long_table["condition"].isin(PART_TO_CONDITION.values())
    ]
    if not selected["group"].astype(str).eq(group).all() or not selected["mouse"].astype(str).eq(mouse).all():
        raise ValueError(f"analysis-16 long-table identity mismatch for {group}/{mouse}")
    _assert_unique(selected, ["day", "condition", "neuron"], f"analysis-16 {group}/{mouse}")
    expected_units = {(day, condition) for day in DAYS for condition in PART_TO_CONDITION.values()}
    observed_long_units = set(
        selected[["day", "condition"]].itertuples(index=False, name=None)
    )
    if observed_long_units != expected_units:
        raise ValueError(f"analysis-16 long table does not cover D1-D5 CL/CR for {group}/{mouse}")
    qc_selected = qc.loc[
        qc["day"].isin(DAYS) & qc["condition"].isin(PART_TO_CONDITION.values())
    ]
    _assert_unique(qc_selected, ["day", "condition"], f"analysis-16 QC {group}/{mouse}")
    observed_units = set(qc_selected[["day", "condition"]].itertuples(index=False, name=None))
    if observed_units != expected_units:
        raise ValueError(f"analysis-16 QC does not cover D1-D5 CL/CR for {group}/{mouse}")
    missing = set()
    for (day, _condition), table in selected.groupby(["day", "condition"], observed=True):
        if not np.isfinite(pd.to_numeric(table["pc_within_corr"], errors="coerce")).any():
            missing.add((group, mouse, int(day)))
    expected_missing = EXPECTED_WITHIN_MISSING.intersection({(group, mouse, day) for day in DAYS})
    if missing != expected_missing:
        raise ValueError(
            f"analysis-16 unexpected within-session missingness for {group}/{mouse}: "
            f"expected={sorted(expected_missing)}, observed={sorted(missing)}"
        )
    map_store = payload["map_store"]
    if not isinstance(map_store, Mapping):
        raise ValueError(f"analysis-16 map_store is not a mapping for {group}/{mouse}")
    for day in DAYS:
        for condition in PART_TO_CONDITION.values():
            entry = map_store.get((day, condition))
            is_expected_missing = (group, mouse, day) in EXPECTED_WITHIN_MISSING
            if entry is None:
                if not is_expected_missing:
                    raise ValueError(
                        f"analysis-16 map_store lacks {group}/{mouse}/D{day}/{condition}"
                    )
                continue
            if is_expected_missing:
                raise ValueError(
                    f"analysis-16 map_store should omit expected missing {group}/{mouse}/D{day}"
                )
            odd = _stack_subject_map_entry(entry, "odd", num_bin=num_bin)
            even = _stack_subject_map_entry(entry, "even", num_bin=num_bin)
            if odd.shape != even.shape:
                raise ValueError(
                    f"analysis-16 odd/even map shape mismatch for {group}/{mouse}/D{day}/{condition}"
                )
            if "mouse" not in entry or "neuron" not in entry:
                raise ValueError(
                    f"analysis-16 map_store row labels missing for {group}/{mouse}/D{day}/{condition}"
                )
            if len(entry["mouse"]) != odd.shape[0] or len(entry["neuron"]) != odd.shape[0]:
                raise ValueError(
                    f"analysis-16 map_store row-label count mismatch for "
                    f"{group}/{mouse}/D{day}/{condition}"
                )
            if any(str(label) != mouse for label in entry["mouse"]):
                raise ValueError(
                    f"analysis-16 map_store mouse labels mismatch for {group}/{mouse}/D{day}/{condition}"
                )

def _validate_analysis17_payload(
    payload: Mapping[str, Any],
    group: str,
    mouse: str,
    *,
    num_bin: int = 56,
) -> None:
    _require_mapping_fields(payload, REQUIRED_PAYLOAD_FIELDS["17"], "analysis-17 payload")
    for pair_mode, pairs in INTERDAY_PAIRS.items():
        if pair_mode not in payload["pv_arrays"] or pair_mode not in payload["pc_long"]:
            raise ValueError(f"analysis-17 lacks {pair_mode} payload for {group}/{mouse}")
        for part in ("left", "right", "averaged"):
            if part not in payload["pv_arrays"][pair_mode]:
                raise ValueError(f"analysis-17 lacks {pair_mode}/{part} PV values for {group}/{mouse}")
            values = np.asarray(payload["pv_arrays"][pair_mode][part], dtype=float)
            if values.shape != (len(pairs), num_bin):
                raise ValueError(
                    f"analysis-17 {group}/{mouse}/{pair_mode}/{part} shape {values.shape}; "
                    f"expected {(len(pairs), num_bin)}"
                )
        candidates = np.asarray(payload["candidate_neuron_count"][pair_mode])
        if candidates.shape != (len(pairs),):
            raise ValueError(f"analysis-17 candidate-neuron shape mismatch for {group}/{mouse}")
        table = payload["pc_long"][pair_mode]
        if not isinstance(table, pd.DataFrame):
            raise ValueError(f"analysis-17 PC table is not a DataFrame for {group}/{mouse}")
        _require_columns(
            table,
            {
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
            },
            f"analysis-17 PC table {group}/{mouse}/{pair_mode}",
        )
        if not table["group"].astype(str).eq(group).all() or not table["mouse"].astype(str).eq(mouse).all():
            raise ValueError(f"analysis-17 PC-table identity mismatch for {group}/{mouse}")
        if not table["pair_mode"].astype(str).eq(pair_mode).all():
            raise ValueError(f"analysis-17 PC-table pair-mode mismatch for {group}/{mouse}")
        selected = table.loc[table["condition"].isin(PART_TO_CONDITION)]
        observed = set(selected[["day_a", "day_b"]].itertuples(index=False, name=None))
        if observed != set(pairs):
            raise ValueError(
                f"analysis-17 {group}/{mouse}/{pair_mode} day labels are not canonical "
                f"one-based pairs: {sorted(observed)}"
            )
        _assert_unique(
            selected,
            ["day_a", "day_b", "condition", "neuron"],
            f"analysis-17 PC {group}/{mouse}/{pair_mode}",
        )
        expected_condition_pairs = {
            (day_a, day_b, condition)
            for day_a, day_b in pairs
            for condition in PART_TO_CONDITION
        }
        observed_condition_pairs = set(
            selected[["day_a", "day_b", "condition"]].itertuples(
                index=False, name=None
            )
        )
        if observed_condition_pairs != expected_condition_pairs:
            raise ValueError(
                f"analysis-17 PC table does not cover both conditions for every "
                f"{pair_mode} pair in {group}/{mouse}"
            )

def _validate_within_missingness(table: pd.DataFrame, *, value_col: str) -> None:
    keys = ["group", "mouse", "day", "condition"]
    _require_columns(table, {*keys, value_col}, "within-session source")
    missing = set()
    for (group, mouse, day, condition), unit in table.groupby(keys, observed=True):
        values = pd.to_numeric(unit[value_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).any():
            missing.add((str(group), str(mouse), int(day), str(condition)))
    expected = {
        (group, mouse, day, condition)
        for group, mouse, day in EXPECTED_WITHIN_MISSING
        for condition in PART_TO_CONDITION.values()
    }
    if missing != expected:
        raise ValueError(
            "within-session source missingness does not match expected B7 D4-only contract; "
            f"expected={sorted(expected)}, observed={sorted(missing)}"
        )

def _validate_one_based_interday_rows(table: pd.DataFrame, label: str) -> None:
    _require_columns(table, {"pair_mode", "day_a", "day_b"}, label)
    for pair_mode, pairs in INTERDAY_PAIRS.items():
        selected = table.loc[table["pair_mode"].astype(str).eq(pair_mode)]
        if selected.empty:
            raise ValueError(f"{label} lacks pair mode {pair_mode!r}")
        observed = set(
            selected[["day_a", "day_b"]]
            .astype(int)
            .itertuples(index=False, name=None)
        )
        if observed != set(pairs):
            raise ValueError(
                f"{label}/{pair_mode} must preserve canonical one-based pairs; "
                f"expected={sorted(pairs)}, observed={sorted(observed)}"
            )

def _stack_subject_map_entry(
    entry: Mapping[str, Any],
    key: str,
    *,
    num_bin: int,
) -> np.ndarray:
    if key not in entry:
        raise ValueError(f"map_store entry lacks {key!r} maps")
    parts = [np.asarray(part, dtype=float) for part in entry[key]]
    if not parts:
        raise ValueError(f"map_store entry has no {key!r} maps")
    if any(part.ndim != 2 or part.shape[1] < num_bin for part in parts):
        raise ValueError(f"map_store {key!r} maps must have shape (neuron, >={num_bin})")
    return np.vstack([part[:, :num_bin] for part in parts])

def _scalar_fisher_z(value: Any) -> float:
    value = float(value)
    if not np.isfinite(value):
        return np.nan
    return float(fisher_z(np.asarray([value]))[0])

def _finite_or_nan(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else np.nan

def _assert_unique(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    columns = list(columns)
    _require_columns(table, set(columns), label)
    duplicate = table.duplicated(columns, keep=False)
    if duplicate.any():
        example = table.loc[duplicate, columns].head(5).to_dict("records")
        raise ValueError(f"{label} contains duplicate sampling units: {example}")

def _require_columns(table: pd.DataFrame, columns: set[str], label: str) -> None:
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{label} must be a pandas DataFrame")
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")

def _require_mapping_fields(
    mapping: Mapping[str, Any], fields: Sequence[str], label: str
) -> None:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a mapping")
    missing = sorted(set(fields).difference(mapping))
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")

def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_canonical(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
