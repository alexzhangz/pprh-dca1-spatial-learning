"""Manuscript definitions collected from fig6_figs3_stability.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import hashlib

import json

import sys

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from .behavior_covariates import pooled_z_standardize_covariates

from .fig3_8maze_behavior import build_fig3_8maze_behavior_tables

from .manuscript_lmm import fit_adjusted_unadjusted_lmm

from .odd_even_reliability import (
    SPLIT_STRATEGY,
    build_pv_bin_source,
    build_pv_mouse_day_table,
    build_single_neuron_source,
)

from .pv_pc_incremental import CACHE_SCHEMA_VERSION, analysis_cache_path

DAYS = (1, 2, 3, 4, 5)

ADJACENT_DAY_PAIRS = ((1, 2), (2, 3), (3, 4), (4, 5))

ANCHOR_DAY_PAIRS = ((1, 2), (1, 3), (1, 4), (1, 5))

INTERDAY_PAIRS = {
    "adjacent": ADJACENT_DAY_PAIRS,
    "anchor": ANCHOR_DAY_PAIRS,
}

FIG6_COHORT = {
    "C": ("A4", "A5", "A7", "A8", "A10"),
    "P": ("H1", "H2", "H3", "H5", "H6"),
}

FIGS3_COHORT = {
    "C": ("C3", "C4", "C7", "C8", "C9", "C10", "C11"),
    "I": ("I2", "I3", "I4", "I5", "I7", "I8", "I9"),
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

COVARIATE_COLUMNS = (
    "mean_speed_z",
    "route_relative_lateral_deviation_z",
)

FIG6_PPRH_COVARIATE_COLUMNS = (
    "route_relative_lateral_deviation_z",
)

FIGS3_PATHWAY_COVARIATE_COLUMNS = COVARIATE_COLUMNS

SINGLE_NEURON_CLUSTER_COL = "globally_unique_neuron"

SINGLE_NEURON_PSEUDOREPLICATION_WARNING = (
    "Author-requested primary single-neuron LMM uses only a globally unique "
    "mouse::neuron random intercept and omits mouse and mouse-by-time random "
    "effects; treatment is assigned at mouse level, so between-group inference "
    "is pseudoreplication-sensitive."
)

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def fisher_z(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    bound = np.nextafter(1.0, 0.0)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    result[finite] = np.arctanh(np.clip(values[finite], -bound, bound))
    return result

def validate_exact_cohort(
    table: pd.DataFrame,
    cohort: Mapping[str, Sequence[str]],
    *,
    label: str,
) -> None:
    required = {"group", "mouse"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing cohort columns: {missing}")
    observed = {
        str(group): tuple(
            sorted(
                table.loc[table["group"].astype(str).eq(str(group)), "mouse"]
                .astype(str)
                .unique()
            )
        )
        for group in cohort
    }
    expected = {
        str(group): tuple(sorted(str(mouse) for mouse in mice))
        for group, mice in cohort.items()
    }
    if observed != expected:
        raise ValueError(f"{label} cohort mismatch: expected={expected}, observed={observed}")
    unexpected_groups = sorted(set(table["group"].astype(str)).difference(cohort))
    if unexpected_groups:
        raise ValueError(f"{label} contains unexpected groups: {unexpected_groups}")

def load_8maze_cache_records(
    cache_root: str | Path,
    analysis: str | int,
    *,
    cohort: Mapping[str, Sequence[str]] = FIG6_COHORT,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Load exact D1-D5 cache envelopes for the declared Fig. 6 cohort."""

    key = str(analysis)
    if key not in ANALYSIS_CONFIGS:
        raise ValueError(f"unsupported analysis {analysis!r}")
    cache_root = Path(cache_root)
    config = ANALYSIS_CONFIGS[key]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for group, mice in cohort.items():
        for mouse in mice:
            path = analysis_cache_path(cache_root, key, config, str(mouse))
            if not path.exists():
                raise FileNotFoundError(
                    f"missing exact-config analysis-{key} cache for {group}/{mouse}: {path}"
                )
            envelope = np.load(path, allow_pickle=True).item()
            if envelope.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError(f"cache schema mismatch: {path}")
            if str(envelope.get("analysis")) != key:
                raise ValueError(f"analysis identity mismatch: {path}")
            if str(envelope.get("group")) != str(group) or str(envelope.get("mouse")) != str(mouse):
                raise ValueError(f"cache cohort identity mismatch: {path}")
            if _canonical(envelope.get("config")) != _canonical(config):
                raise ValueError(f"cache config mismatch: {path}")
            if envelope.get("status") != "complete":
                raise ValueError(f"cache is not complete: {path}")
            records.append(envelope)
            rows.append(
                {
                    "analysis": key,
                    "group": str(group),
                    "mouse": str(mouse),
                    "input_path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size),
                    "config_sha256": sha256_json(config),
                    "status": "validated",
                }
            )
    manifest = pd.DataFrame(rows)
    validate_exact_cohort(manifest, cohort, label=f"analysis-{key} cache manifest")
    return records, manifest

def build_8maze_within_sources(
    records14: Sequence[Mapping[str, Any]],
    records16: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    pv_bin = build_pv_bin_source(records14, num_day=5, num_bin=56)
    pv_mouse_day = build_pv_mouse_day_table(pv_bin)
    single, single_qc = build_single_neuron_source(records16, num_day=5)
    single = single.rename(columns={"raw_r": "raw_r", "fisher_z": "fisher_z"})
    for table, label in (
        (pv_bin, "Fig6 within PV-bin source"),
        (pv_mouse_day, "Fig6 within PV mouse-day source"),
        (single, "Fig6 within single-neuron source"),
    ):
        validate_exact_cohort(table, FIG6_COHORT, label=label)
    return {
        "pv_bin": pv_bin,
        "pv_mouse_day": pv_mouse_day,
        "single_neuron": single,
        "single_neuron_qc": single_qc,
    }

def build_8maze_interday_sources(
    records17: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    pv_rows: list[dict[str, Any]] = []
    single_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    for record in records17:
        group = str(record["group"])
        mouse = str(record["mouse"])
        payload = record["payload"]
        for mode, pairs in INTERDAY_PAIRS.items():
            arrays = payload["pv_arrays"][mode]
            candidates = np.asarray(payload["candidate_neuron_count"][mode], dtype=float)
            if candidates.shape != (4,):
                raise ValueError(f"analysis-17 candidate-count shape mismatch for {mouse}/{mode}")
            for pair_index, (day_a, day_b) in enumerate(pairs):
                candidate_rows.append(
                    {
                        "group": group,
                        "mouse": mouse,
                        "pair_mode": mode,
                        "pair_index": int(pair_index),
                        "time_index": int(pair_index + 1),
                        "day_a": int(day_a),
                        "day_b": int(day_b),
                        "day_pair_label": f"D{day_a}-D{day_b}",
                        "candidate_neuron_count": float(candidates[pair_index]),
                    }
                )
                for part, condition in (("left", "CL"), ("right", "CR")):
                    values = np.asarray(arrays[part], dtype=float)
                    if values.shape != (4, 56):
                        raise ValueError(
                            f"analysis-17 PV shape mismatch for {mouse}/{mode}/{part}: {values.shape}"
                        )
                    for spatial_bin, raw_r in enumerate(values[pair_index]):
                        pv_rows.append(
                            {
                                "group": group,
                                "mouse": mouse,
                                "pair_mode": mode,
                                "pair_index": int(pair_index),
                                "time_index": int(pair_index + 1),
                                "day_a": int(day_a),
                                "day_b": int(day_b),
                                "day_pair_label": f"D{day_a}-D{day_b}",
                                "condition": condition,
                                "condition_part": part,
                                "spatial_bin": int(spatial_bin),
                                "raw_r": _finite_or_nan(raw_r),
                            }
                        )
            pc = payload["pc_long"][mode].copy()
            pc = pc.loc[
                pc["condition"].isin(["left", "right"])
                & pd.to_numeric(pc["day_b"], errors="coerce").le(5)
            ].copy()
            pc["condition"] = pc["condition"].map({"left": "CL", "right": "CR"})
            pc["time_index"] = pd.to_numeric(pc["pair_index"], errors="raise").astype(int) + 1
            pc = pc.rename(columns={"pc_interday_corr": "raw_r"})
            pc["fisher_z"] = fisher_z(pc["raw_r"])
            single_frames.append(pc)

    pv_bin = pd.DataFrame(pv_rows)
    pv_bin["fisher_z"] = fisher_z(pv_bin["raw_r"])
    pv_mouse_pair = _collapse_pv_bins(
        pv_bin,
        keys=(
            "group",
            "mouse",
            "pair_mode",
            "pair_index",
            "time_index",
            "day_a",
            "day_b",
            "day_pair_label",
            "condition",
            "condition_part",
        ),
    )
    single = pd.concat(single_frames, ignore_index=True)
    candidate = pd.DataFrame(candidate_rows)
    for table, label in (
        (pv_bin, "Fig6 interday PV-bin source"),
        (pv_mouse_pair, "Fig6 interday PV mouse-pair source"),
        (single, "Fig6 interday single-neuron source"),
        (candidate, "Fig6 interday candidate source"),
    ):
        validate_exact_cohort(table, FIG6_COHORT, label=label)
    return {
        "pv_bin": pv_bin,
        "pv_mouse_pair": pv_mouse_pair,
        "single_neuron": single,
        "candidate_neuron_count": candidate,
    }

def build_figure8_cp_covariates(
    aligned_root: str | Path,
    *,
    cohort: Mapping[str, Sequence[str]] = FIG6_COHORT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    atomic = build_fig3_8maze_behavior_tables(
        aligned_root,
        group_mice=cohort,
        days=DAYS,
    )
    table = atomic["covariates_mouse_day"].copy()
    required = {
        "group",
        "mouse",
        "day",
        "mean_speed_cm_s",
        "route_relative_lateral_deviation_cm",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Figure-8 C/P covariates are missing columns: {missing}")
    validate_exact_cohort(table, cohort, label="Figure-8 C/P raw covariates")
    expected = sum(len(mice) for mice in cohort.values()) * len(DAYS)
    if len(table) != expected or table.duplicated(["group", "mouse", "day"]).any():
        raise ValueError("Figure-8 C/P covariates do not form the complete locked grid")
    output_names = {
        "mean_speed_cm_s": "mean_speed_z",
        "route_relative_lateral_deviation_cm": "route_relative_lateral_deviation_z",
    }
    standardized, qc = pooled_z_standardize_covariates(
        table,
        columns=tuple(output_names),
        output_columns=output_names,
        ddof=1,
    )
    if not (qc["status"] == "ok").all():
        raise ValueError("Figure-8 C/P covariate standardization failed")
    if standardized[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("Figure-8 C/P covariates contain nonfinite standardized values")
    standardized["day_c"] = pd.to_numeric(standardized["day"], errors="raise") - 3.0
    standardized["project"] = "CA Figure-8 maze"
    standardized["comparison"] = "dCA1_vs_pPRH"
    return standardized, qc

def prepare_tmaze_ci_covariates(
    shared_covariates: pd.DataFrame,
    *,
    cohort: Mapping[str, Sequence[str]] = FIGS3_COHORT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "project",
        "group",
        "mouse",
        "day",
        *COVARIATE_COLUMNS,
        "mean_speed_cm_s",
        "route_relative_lateral_deviation_cm",
    }
    missing = sorted(required.difference(shared_covariates.columns))
    if missing:
        raise ValueError(f"shared T-maze covariates are missing columns: {missing}")
    table = shared_covariates.loc[
        shared_covariates["project"].astype(str).eq("DNMP T-maze")
        & shared_covariates["group"].astype(str).isin(cohort)
        & pd.to_numeric(shared_covariates["day"], errors="coerce").isin(DAYS)
    ].copy()
    allowed = {str(mouse) for mice in cohort.values() for mouse in mice}
    table = table.loc[table["mouse"].astype(str).isin(allowed)].copy()
    validate_exact_cohort(table, cohort, label="T-maze C/I shared covariates")
    expected = sum(len(mice) for mice in cohort.values()) * len(DAYS)
    if len(table) != expected or table.duplicated(["group", "mouse", "day"]).any():
        raise ValueError("T-maze C/I covariates do not form the complete locked grid")
    # Never reuse z values created for another cohort or comparison.  The raw
    # columns are re-standardized on the exact locked C7/I7 joint-complete
    # pool, even when the input file happens to carry older z columns.
    output_names = {
        "mean_speed_cm_s": "mean_speed_z",
        "route_relative_lateral_deviation_cm": "route_relative_lateral_deviation_z",
    }
    table = table.drop(columns=list(COVARIATE_COLUMNS), errors="ignore")
    table, qc = pooled_z_standardize_covariates(
        table,
        columns=tuple(output_names),
        output_columns=output_names,
        ddof=1,
    )
    if not (qc["status"] == "ok").all():
        raise ValueError("T-maze C/I covariate re-standardization failed")
    if table[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("T-maze C/I re-standardized covariates contain nonfinite values")
    table["day_c"] = pd.to_numeric(table["day"], errors="raise") - 3.0
    table["comparison"] = "control_vs_inhibition"
    return table, qc

def build_pair_covariates(
    within_covariates: pd.DataFrame,
    *,
    cohort: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    required = {"group", "mouse", "day", *COVARIATE_COLUMNS}
    missing = sorted(required.difference(within_covariates.columns))
    if missing:
        raise ValueError(f"within-day covariates are missing columns: {missing}")
    validate_exact_cohort(within_covariates, cohort, label="pair-covariate input")
    rows: list[dict[str, Any]] = []
    indexed = within_covariates.set_index(["group", "mouse", "day"])
    for group, mice in cohort.items():
        for mouse in mice:
            for mode, pairs in INTERDAY_PAIRS.items():
                for pair_index, (day_a, day_b) in enumerate(pairs):
                    a = indexed.loc[(str(group), str(mouse), int(day_a))]
                    b = indexed.loc[(str(group), str(mouse), int(day_b))]
                    rows.append(
                        {
                            "group": str(group),
                            "mouse": str(mouse),
                            "pair_mode": mode,
                            "pair_index": int(pair_index),
                            "time_index": int(pair_index + 1),
                            "day_a": int(day_a),
                            "day_b": int(day_b),
                            "day_pair_label": f"D{day_a}-D{day_b}",
                            "mean_speed_z_day_a": float(a["mean_speed_z"]),
                            "mean_speed_z_day_b": float(b["mean_speed_z"]),
                            "route_relative_lateral_deviation_z_day_a": float(
                                a["route_relative_lateral_deviation_z"]
                            ),
                            "route_relative_lateral_deviation_z_day_b": float(
                                b["route_relative_lateral_deviation_z"]
                            ),
                            "mean_speed_z": float(
                                (a["mean_speed_z"] + b["mean_speed_z"]) / 2.0
                            ),
                            "route_relative_lateral_deviation_z": float(
                                (
                                    a["route_relative_lateral_deviation_z"]
                                    + b["route_relative_lateral_deviation_z"]
                                )
                                / 2.0
                            ),
                            "covariate_pair_rule": "mean_of_two_session_z_scores",
                        }
                    )
    result = pd.DataFrame(rows)
    if result.duplicated(["group", "mouse", "pair_mode", "pair_index"]).any():
        raise ValueError("pair covariates contain duplicate mouse-pair rows")
    validate_exact_cohort(result, cohort, label="pair covariates")
    return result

def attach_within_covariates(
    source: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    overlap = sorted(set(COVARIATE_COLUMNS).intersection(source.columns))
    if overlap:
        raise ValueError(f"within-day source already contains covariates: {overlap}")
    columns = ["group", "mouse", "day", *COVARIATE_COLUMNS]
    result = source.merge(
        covariates[columns],
        on=["group", "mouse", "day"],
        how="left",
        validate="many_to_one",
    )
    if result[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("within-day covariate merge produced unmatched rows")
    return result

def attach_pair_covariates(
    source: pd.DataFrame,
    pair_covariates: pd.DataFrame,
) -> pd.DataFrame:
    overlap = sorted(set(COVARIATE_COLUMNS).intersection(source.columns))
    if overlap:
        raise ValueError(f"interday source already contains covariates: {overlap}")
    keys = ["group", "mouse", "pair_mode", "pair_index"]
    columns = [*keys, *COVARIATE_COLUMNS]
    result = source.merge(
        pair_covariates[columns],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if result[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("interday covariate merge produced unmatched rows")
    return result

def build_tmaze_interday_sources(
    tmaze_repo_root: str | Path,
    *,
    cohort: Mapping[str, Sequence[str]] = FIGS3_COHORT,
    selected_neuron_count: int = 50,
    n_subsample_iter: int = 100,
) -> dict[str, pd.DataFrame]:
    """Rebuild accepted Fig. S3 D1-D5 metrics from T-maze spatial maps.

    The migrated subject helper is used to preserve the accepted deterministic
    per-pair seed policy.  Although the stored maps contain seven sessions,
    only pairs with ``day_b <= 5`` are admitted to the returned release.
    """

    root = Path(tmaze_repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from t_maze.stability import compute_interday_stability_subject  # type: ignore

    aligned_root = root / "Data" / "Aligened_data"
    pv_rows: list[dict[str, Any]] = []
    single_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for group, mice in cohort.items():
        for mouse in mice:
            subject = aligned_root / str(mouse)
            map_path = subject / "Snarm_1d_v2.npy"
            if not map_path.exists():
                raise FileNotFoundError(map_path)
            input_rows.append(
                {
                    "group": str(group),
                    "mouse": str(mouse),
                    "input_path": str(map_path.resolve()),
                    "sha256": sha256_file(map_path),
                    "bytes": int(map_path.stat().st_size),
                    "status": "validated",
                }
            )
            payload = compute_interday_stability_subject(
                subject,
                mouse=str(mouse),
                group=str(group),
                condition_pair="choice_correct",
                pair_modes=("adjacent", "anchor"),
                neuron_set="all",
                selected_neuron_count=int(selected_neuron_count),
                n_subsample_iter=int(n_subsample_iter),
                pv_method="spearman",
                pc_method="pearson",
                seed=2026,
            )
            if int(payload["n_bin"]) != 29:
                raise ValueError(f"T-maze path-bin mismatch for {mouse}: {payload['n_bin']}")
            for mode, pairs in INTERDAY_PAIRS.items():
                arrays = payload["pv_arrays"][mode]
                candidates = np.asarray(payload["candidate_neuron_count"][mode], dtype=float)
                for pair_index, (day_a, day_b) in enumerate(pairs):
                    candidate_rows.append(
                        {
                            "group": str(group),
                            "mouse": str(mouse),
                            "pair_mode": mode,
                            "pair_index": int(pair_index),
                            "time_index": int(pair_index + 1),
                            "day_a": int(day_a),
                            "day_b": int(day_b),
                            "day_pair_label": f"D{day_a}-D{day_b}",
                            "candidate_neuron_count": float(candidates[pair_index]),
                        }
                    )
                    for part, condition in (
                        ("left", "left"),
                        ("right", "right"),
                        ("averaged", "averaged"),
                    ):
                        values = np.asarray(arrays[part], dtype=float)
                        if values.shape[1] != 29:
                            raise ValueError(
                                f"T-maze PV shape mismatch for {mouse}/{mode}/{part}: {values.shape}"
                            )
                        for spatial_bin, raw_r in enumerate(values[pair_index]):
                            pv_rows.append(
                                {
                                    "group": str(group),
                                    "mouse": str(mouse),
                                    "pair_mode": mode,
                                    "pair_index": int(pair_index),
                                    "time_index": int(pair_index + 1),
                                    "day_a": int(day_a),
                                    "day_b": int(day_b),
                                    "day_pair_label": f"D{day_a}-D{day_b}",
                                    "condition": condition,
                                    "condition_part": part,
                                    "spatial_bin": int(spatial_bin),
                                    "raw_r": _finite_or_nan(raw_r),
                                }
                            )
                pc = payload["pc_long"][mode].copy()
                pc = pc.loc[
                    pd.to_numeric(pc["day_b"], errors="coerce").le(5)
                    & pc["condition"].isin(["left", "right", "averaged"])
                ].copy()
                pc["time_index"] = pd.to_numeric(pc["pair_index"], errors="raise").astype(int) + 1
                pc = pc.rename(columns={"pc_interday_corr": "raw_r"})
                pc["fisher_z"] = fisher_z(pc["raw_r"])
                single_frames.append(pc)

    pv_bin = pd.DataFrame(pv_rows)
    pv_bin["fisher_z"] = fisher_z(pv_bin["raw_r"])
    pv_mouse_pair = _collapse_pv_bins(
        pv_bin,
        keys=(
            "group",
            "mouse",
            "pair_mode",
            "pair_index",
            "time_index",
            "day_a",
            "day_b",
            "day_pair_label",
            "condition",
            "condition_part",
        ),
    )
    single = pd.concat(single_frames, ignore_index=True)
    candidate = pd.DataFrame(candidate_rows)
    inputs = pd.DataFrame(input_rows)
    for table, label in (
        (pv_bin, "FigS3 PV-bin source"),
        (pv_mouse_pair, "FigS3 PV mouse-pair source"),
        (single, "FigS3 single-neuron source"),
        (candidate, "FigS3 candidate-neuron source"),
        (inputs, "FigS3 input manifest"),
    ):
        validate_exact_cohort(table, cohort, label=label)
    if not candidate["candidate_neuron_count"].ge(selected_neuron_count).all():
        failed = candidate.loc[
            candidate["candidate_neuron_count"].lt(selected_neuron_count)
        ]
        raise ValueError(
            "T-maze fixed-50 PV contract failed: "
            + failed[["group", "mouse", "pair_mode", "pair_index", "candidate_neuron_count"]]
            .to_dict("records")
            .__repr__()
        )
    return {
        "pv_bin": pv_bin,
        "pv_mouse_pair": pv_mouse_pair,
        "single_neuron": single,
        "candidate_neuron_count": candidate,
        "input_manifest": inputs,
    }

def prepare_globally_unique_neuron_units(table: pd.DataFrame) -> pd.DataFrame:
    required = {"mouse", "neuron"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"single-neuron table is missing columns: {missing}")
    result = table.loc[table["mouse"].notna() & table["neuron"].notna()].copy()
    neuron_numeric = pd.to_numeric(result["neuron"], errors="raise")
    if not np.isfinite(neuron_numeric.to_numpy(dtype=float)).all():
        raise ValueError("single-neuron table contains nonfinite neuron identifiers")
    neuron_label = neuron_numeric.map(
        lambda value: str(int(value)) if float(value).is_integer() else str(float(value))
    )
    result[SINGLE_NEURON_CLUSTER_COL] = result["mouse"].astype(str) + "::" + neuron_label
    identity = result[[SINGLE_NEURON_CLUSTER_COL, "mouse", "neuron"]].drop_duplicates()
    if identity[SINGLE_NEURON_CLUSTER_COL].duplicated(keep=False).any():
        raise ValueError("globally unique neuron identifier is not one-to-one")
    return result

def fit_stability_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    reference_group: str,
    comparison_group: str,
    project: str,
    figure: str,
    maxiter: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Fit adjusted/unadjusted LMMs for declared population/neuron jobs."""

    contrast_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for job in jobs:
        metric = str(job["metric"])
        analysis = str(job["analysis"])
        continuous_covariates = tuple(
            str(column)
            for column in job.get("continuous_covariates", COVARIATE_COLUMNS)
        )
        if len(set(continuous_covariates)) != len(continuous_covariates):
            raise ValueError(
                f"duplicate continuous covariates for {figure}/{analysis}/{metric}: "
                f"{continuous_covariates}"
            )
        unsupported_covariates = sorted(
            set(continuous_covariates).difference(COVARIATE_COLUMNS)
        )
        if unsupported_covariates:
            raise ValueError(
                f"unsupported continuous covariates for {figure}/{analysis}/{metric}: "
                f"{unsupported_covariates}"
            )
        time_col = str(job["time_col"])
        first_time = float(job["first_time"])
        last_time = float(job["last_time"])
        source = job["table"]
        for condition in tuple(job["conditions"]):
            table = source.loc[source["condition"].astype(str).eq(str(condition))].copy()
            if table.empty:
                raise ValueError(f"empty model table for {figure}/{analysis}/{metric}/{condition}")
            if metric == "single_neuron":
                model_table = prepare_globally_unique_neuron_units(table)
                cluster_col = SINGLE_NEURON_CLUSTER_COL
                fits = fit_adjusted_unadjusted_lmm(
                    model_table,
                    continuous_covariates=continuous_covariates,
                    value_col="fisher_z",
                    time_col=time_col,
                    group_col="group",
                    mouse_col=cluster_col,
                    reference_group=reference_group,
                    comparison_group=comparison_group,
                    sampling_unit_cols=(cluster_col, time_col),
                    first_time=first_time,
                    last_time=last_time,
                    maxiter=int(maxiter),
                )
                cluster_definition = "globally unique mouse::neuron"
                random_effect = "(1|mouse::neuron)"
                warning = SINGLE_NEURON_PSEUDOREPLICATION_WARNING
            elif metric == "pv":
                model_table = table
                cluster_col = "mouse"
                fits = fit_adjusted_unadjusted_lmm(
                    model_table,
                    continuous_covariates=continuous_covariates,
                    value_col="fisher_z",
                    time_col=time_col,
                    group_col="group",
                    mouse_col="mouse",
                    reference_group=reference_group,
                    comparison_group=comparison_group,
                    sampling_unit_cols=("mouse", time_col),
                    first_time=first_time,
                    last_time=last_time,
                    maxiter=int(maxiter),
                )
                cluster_definition = "mouse"
                random_effect = "(1|mouse)"
                warning = ""
            else:
                raise ValueError(f"unsupported metric type: {metric}")

            biological_mice = {
                group: tuple(
                    sorted(
                        model_table.loc[model_table["group"].astype(str).eq(group), "mouse"]
                        .astype(str)
                        .unique()
                    )
                )
                for group in (reference_group, comparison_group)
            }
            for role, result in fits.items():
                diagnostic = dict(result.diagnostics)
                if diagnostic.get("model_status") != "ok":
                    raise RuntimeError(
                        f"LMM failed for {figure}/{analysis}/{metric}/{condition}/{role}: "
                        f"{diagnostic.get('failure_reason')}"
                    )
                if role == "adjusted" and continuous_covariates:
                    formula = (
                        "fisher_z ~ time_c*group + "
                        + " + ".join(continuous_covariates)
                        + " + "
                        + random_effect
                    )
                else:
                    formula = "fisher_z ~ time_c*group + " + random_effect
                diagnostic.update(
                    {
                        "project": project,
                        "figure": figure,
                        "analysis": analysis,
                        "metric": metric,
                        "condition": str(condition),
                        "model_role": role,
                        "model_cluster_definition": cluster_definition,
                        "model_formula_with_random_effect": formula,
                        "continuous_covariates": list(continuous_covariates),
                        "n_biological_mice": int(
                            sum(len(mice) for mice in biological_mice.values())
                        ),
                        "biological_mice": biological_mice,
                        "mouse_random_effect_included": metric == "pv",
                        "mouse_by_time_random_effect_included": False,
                        "pseudoreplication_warning": warning,
                        "reported_probability": "raw two-sided P",
                        "outcome_scale": "Fisher z of correlation",
                        "plotted_scale": "raw correlation r",
                    }
                )
                diagnostics.append(diagnostic)

                contrasts = result.contrasts.copy()
                contrasts.insert(0, "model_role", role)
                contrasts.insert(0, "condition", str(condition))
                contrasts.insert(0, "metric", metric)
                contrasts.insert(0, "analysis", analysis)
                contrasts.insert(0, "figure", figure)
                contrasts.insert(0, "project", project)
                contrasts["model_cluster_definition"] = cluster_definition
                contrasts["model_formula_with_random_effect"] = formula
                contrasts["continuous_covariates"] = ";".join(
                    continuous_covariates
                )
                contrasts["mouse_random_effect_included"] = metric == "pv"
                contrasts["mouse_by_time_random_effect_included"] = False
                contrasts["pseudoreplication_warning"] = warning
                contrasts["reported_probability"] = "P"
                contrasts["reported_probability_value"] = pd.to_numeric(
                    contrasts["p_value"], errors="coerce"
                )
                contrasts["significant"] = contrasts["reported_probability_value"] < 0.05
                contrasts["multiplicity_correction"] = "none"
                contrasts["outcome_scale"] = "Fisher z of correlation"
                contrasts["plotted_scale"] = "raw correlation r"
                contrasts["n_biological_mice_reference"] = len(
                    biological_mice[reference_group]
                )
                contrasts["n_biological_mice_comparison"] = len(
                    biological_mice[comparison_group]
                )
                contrast_frames.append(contrasts)

                coefficients = result.coefficients.copy()
                coefficients.insert(0, "model_role", role)
                coefficients.insert(0, "condition", str(condition))
                coefficients.insert(0, "metric", metric)
                coefficients.insert(0, "analysis", analysis)
                coefficients.insert(0, "figure", figure)
                coefficients.insert(0, "project", project)
                coefficients["model_cluster_definition"] = cluster_definition
                coefficients["model_formula_with_random_effect"] = formula
                coefficients["continuous_covariates"] = ";".join(
                    continuous_covariates
                )
                coefficients["pseudoreplication_warning"] = warning
                coefficient_frames.append(coefficients)

    contrasts = pd.concat(contrast_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    return contrasts, coefficients, diagnostics

def build_raw_summary(
    table: pd.DataFrame,
    *,
    time_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in table.groupby(
        ["group", "condition", time_col],
        observed=True,
        sort=True,
    ):
        values = pd.to_numeric(frame["raw_r"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "group": str(key[0]),
                "condition": str(key[1]),
                time_col: key[2],
                "mean_raw_r": float(np.mean(values)) if values.size else np.nan,
                "sem_raw_r": (
                    float(np.std(values, ddof=1) / np.sqrt(values.size))
                    if values.size > 1
                    else np.nan
                ),
                "n_finite": int(values.size),
            }
        )
    return pd.DataFrame(rows)

def _collapse_pv_bins(
    table: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in table.groupby(list(keys), observed=True, sort=False):
        values = pd.to_numeric(frame["raw_r"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        raw_r = float(np.mean(values)) if values.size else np.nan
        rows.append(
            {
                **dict(zip(keys, key)),
                "raw_r": raw_r,
                "fisher_z": float(fisher_z(raw_r)) if np.isfinite(raw_r) else np.nan,
                "n_bin_finite": int(values.size),
            }
        )
    return pd.DataFrame(rows)

def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def _finite_or_nan(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else np.nan
