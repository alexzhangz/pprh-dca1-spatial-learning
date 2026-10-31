"""Manuscript definitions collected from fig7_figure8_pprh.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import hashlib

import json

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from .behavior_covariates import pooled_z_standardize_covariates

from .fig3_8maze_behavior import build_fig3_8maze_behavior_tables

from .manuscript_lmm import LongitudinalLMMResult, fit_adjusted_unadjusted_lmm

DAYS = (1, 2, 3, 4, 5)

DAY_CENTER = 3.0

GROUP_ORDER = ("C", "P")

COHORT = {
    "C": ("A4", "A5", "A7", "A8", "A10"),
    "P": ("H1", "H2", "H3", "H5", "H6"),
}

PLOTTED_SECTIONS = ("Start arm", "Side arm", "Reward arm", "End arm")

PV_SECTIONS = (*PLOTTED_SECTIONS, "Reward location")

MANIFOLD_SECTION_MAP = {
    "Bins 0-13": "Start arm",
    "Bins 14-27": "Side arm",
    "Bins 28-41": "Reward arm",
    "Bins 42-55": "End arm",
}

COVARIATE_COLUMNS = (
    "route_relative_lateral_deviation_z",
)

RAW_COVARIATE_COLUMNS = (
    "mean_speed_cm_s",
    "route_relative_lateral_deviation_cm",
)

PLANNED_ESTIMANDS = (
    "reference_group_day_slope",
    "comparison_group_day_slope",
    "group_by_day_interaction",
    "centered_overall_group_effect",
    "between_group_D1",
    "between_group_D5",
)

MODEL_FORMULA = (
    "value ~ (day - 3) * group + "
    "route_relative_lateral_deviation_z + (1 | mouse)"
)

EXPECTED_SOURCE_SHA256 = {
    "pv_bin": "9b95574254e0e5fa2f9d6c99e1b39439d780b3657cc609c043b979f33beff74f",
    "pv_section": "39e08ddf3cbedee15dc0a410ccab03e5ab0220b50cd1e09120c50a88bcc71e0f",
    "manifold_section": "df25aaad9cff4aeb6074ea46270f18267643e3223b9358c9dcd01699fc05e348",
}

@dataclass(frozen=True)
class Figure8PprhSourcePaths:
    pv_bin: Path
    pv_section: Path
    manifold_section: Path

@dataclass(frozen=True)
class Figure8PprhBundle:
    pv_bin: pd.DataFrame
    pv_section: pd.DataFrame
    manifold_section: pd.DataFrame
    covariates: pd.DataFrame
    covariate_qc: pd.DataFrame
    source_qc: pd.DataFrame

@dataclass(frozen=True)
class Figure8PprhStats:
    contrasts: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    support: pd.DataFrame
    fit_results: Mapping[tuple[str, str], Mapping[str, LongitudinalLMMResult]]

def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def prepare_figure8_cp_covariates(
    aligned_root: str | Path,
    *,
    cohort: Mapping[str, Sequence[str]] = COHORT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute and pooled-z covariates in the exact C5/P5 comparison."""

    atomic = build_fig3_8maze_behavior_tables(
        aligned_root,
        group_mice=cohort,
        days=DAYS,
    )
    table = atomic["covariates_mouse_day"].copy()
    _validate_exact_cohort(table, cohort, label="Figure-8 C/P raw covariates")
    expected_rows = sum(len(mice) for mice in cohort.values()) * len(DAYS)
    if len(table) != expected_rows or table.duplicated(["group", "mouse", "day"]).any():
        raise ValueError("Figure-8 C/P covariates do not form the complete 50-row grid")
    output_names = {
        "route_relative_lateral_deviation_cm": (
            "route_relative_lateral_deviation_z"
        ),
    }
    table, qc = pooled_z_standardize_covariates(
        table,
        columns=tuple(output_names),
        output_columns=output_names,
        ddof=1,
    )
    if not (qc["status"] == "ok").all():
        raise ValueError("Figure-8 C/P pooled covariate standardization failed")
    if not (qc["reference_n"] == expected_rows).all():
        raise ValueError("Figure-8 C/P covariate reference is not the exact 50 mouse-days")
    if table[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("Figure-8 C/P standardized covariates contain nonfinite values")
    table["day_idx"] = pd.to_numeric(table["day"], errors="raise").astype(int) - 1
    table["day_c"] = pd.to_numeric(table["day"], errors="raise") - DAY_CENTER
    table["day_centered"] = table["day_c"]
    table["comparison"] = "dCA1_control_vs_projection_defined_pPRH"
    return table, qc

def load_figure8_pprh_bundle(
    paths: Figure8PprhSourcePaths,
    *,
    aligned_root: str | Path,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_SHA256,
) -> Figure8PprhBundle:
    """Load, normalize and fail-closed validate Fig. 7A/B/E source tables."""

    source_rows: list[dict[str, Any]] = []
    raw: dict[str, pd.DataFrame] = {}
    for key in ("pv_bin", "pv_section", "manifold_section"):
        path = Path(getattr(paths, key)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        expected = str(expected_hashes[key]).lower()
        if digest.lower() != expected:
            raise ValueError(f"{key} source SHA-256 mismatch: {digest} != {expected}")
        raw[key] = pd.read_csv(path)
        source_rows.append(
            {
                "source_key": key,
                "path": str(path),
                "sha256": digest,
                "bytes": int(path.stat().st_size),
                "status": "validated",
            }
        )

    pv_bin = _normalize_pv(raw["pv_bin"], level="bin")
    pv_section = _normalize_pv(raw["pv_section"], level="section")
    manifold_section = _normalize_manifold(raw["manifold_section"])
    covariates, covariate_qc = prepare_figure8_cp_covariates(aligned_root)

    _validate_complete_support(
        pv_bin,
        keys=("group", "mouse", "day", "spatial_bin"),
        expected_rows=10 * 5 * 56,
        label="Fig7A PV-bin",
    )
    _validate_complete_support(
        pv_section,
        keys=("group", "mouse", "day", "section"),
        expected_rows=10 * 5 * len(PV_SECTIONS),
        label="Fig7B PV-section",
    )
    _validate_complete_support(
        manifold_section,
        keys=("group", "mouse", "day", "section"),
        expected_rows=10 * 5 * len(PLOTTED_SECTIONS),
        label="Fig7E manifold-section",
    )
    return Figure8PprhBundle(
        pv_bin=pv_bin,
        pv_section=pv_section,
        manifold_section=manifold_section,
        covariates=covariates,
        covariate_qc=covariate_qc,
        source_qc=pd.DataFrame(source_rows),
    )

def build_longitudinal_table(bundle: Figure8PprhBundle) -> pd.DataFrame:
    """Combine section sources and attach exact-comparison covariates."""

    source = pd.concat(
        [bundle.pv_section, bundle.manifold_section],
        ignore_index=True,
        sort=False,
    )
    covariate_columns = [
        "group",
        "mouse",
        "day",
        *RAW_COVARIATE_COLUMNS,
        *COVARIATE_COLUMNS,
    ]
    overlap = sorted(
        set(covariate_columns[3:]).intersection(source.columns)
    )
    if overlap:
        raise ValueError(f"source table already contains covariates: {overlap}")
    table = source.merge(
        bundle.covariates[covariate_columns],
        on=["group", "mouse", "day"],
        how="left",
        validate="many_to_one",
    )
    if table[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("Fig7A/B/E covariate merge produced unmatched rows")
    table["day_idx"] = pd.to_numeric(table["day"], errors="raise").astype(int) - 1
    table["day_c"] = pd.to_numeric(table["day"], errors="raise") - DAY_CENTER
    table["day_centered"] = table["day_c"]
    _assert_unique(
        table,
        ("metric", "section", "group", "mouse", "day"),
        "Fig7A/B/E model table",
    )
    return table.sort_values(
        ["metric", "section_idx", "group", "mouse", "day"],
        kind="mergesort",
    ).reset_index(drop=True)

def fit_figure8_pprh_models(
    table: pd.DataFrame,
    *,
    maxiter: int = 1000,
) -> Figure8PprhStats:
    """Fit adjusted manuscript LMMs and paired unadjusted audit models."""

    required = {
        "metric",
        "section",
        "group",
        "mouse",
        "day",
        "value",
        *COVARIATE_COLUMNS,
    }
    _require_columns(table, required, "Fig7A/B/E model table")
    _assert_unique(
        table,
        ("metric", "section", "group", "mouse", "day"),
        "Fig7A/B/E model table",
    )
    contrast_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    fit_results: dict[
        tuple[str, str], Mapping[str, LongitudinalLMMResult]
    ] = {}
    for raw_key, subset in table.groupby(["metric", "section"], observed=True, sort=True):
        key = tuple(str(value) for value in raw_key)
        _validate_exact_cohort(subset, COHORT, label=f"Fig7A/B/E stratum {key}")
        for group in GROUP_ORDER:
            days = set(
                pd.to_numeric(
                    subset.loc[subset["group"].eq(group), "day"], errors="coerce"
                ).dropna().astype(int)
            )
            if days != set(DAYS):
                raise ValueError(f"Fig7A/B/E stratum {key}/{group} lacks D1-D5")
        paired = fit_adjusted_unadjusted_lmm(
            subset,
            value_col="value",
            time_col="day",
            group_col="group",
            mouse_col="mouse",
            reference_group="C",
            comparison_group="P",
            continuous_covariates=COVARIATE_COLUMNS,
            sampling_unit_cols=("mouse", "day"),
            first_time=1,
            last_time=5,
            maxiter=int(maxiter),
        )
        fit_results[key] = paired
        if paired["adjusted"].diagnostics.get("nobs") != paired["unadjusted"].diagnostics.get("nobs"):
            raise RuntimeError(f"paired LMM samples differ for Fig7A/B/E stratum {key}")
        for role, result in paired.items():
            if not result.succeeded:
                raise RuntimeError(f"Fig7A/B/E LMM failed for {key}/{role}")
            contrasts = result.contrasts.rename(
                columns={"contrast": "estimand_contrast"}
            ).copy()
            if set(contrasts["estimand_contrast"]) != set(PLANNED_ESTIMANDS):
                raise RuntimeError(f"unexpected estimands for Fig7A/B/E stratum {key}")
            contrasts.insert(0, "model_role", role)
            contrasts.insert(0, "section", key[1])
            contrasts.insert(0, "metric", key[0])
            contrasts["reported_probability"] = "P"
            contrasts["reported_probability_value"] = pd.to_numeric(
                contrasts["p_value"], errors="coerce"
            )
            contrasts["multiple_comparison_correction"] = "none"
            contrasts["manuscript_facing"] = role == "adjusted"
            contrasts["sampling_unit"] = "mouse/day"
            contrasts["biological_replicate"] = "mouse"
            contrasts["model_formula_contract"] = MODEL_FORMULA
            contrasts["panel"] = "B" if key[0] == "high_dimensional_pv" else "E"
            contrast_frames.append(contrasts)

            coefficients = result.coefficients.copy()
            coefficients.insert(0, "model_role", role)
            coefficients.insert(0, "section", key[1])
            coefficients.insert(0, "metric", key[0])
            coefficient_frames.append(coefficients)
            diagnostic_rows.append(
                {
                    "metric": key[0],
                    "section": key[1],
                    "model_role": role,
                    "model_status": result.diagnostics.get("model_status"),
                    "optimizer": result.diagnostics.get("optimizer"),
                    "converged": result.diagnostics.get("converged"),
                    "covariance_finite": result.diagnostics.get("covariance_finite"),
                    "fallback_used": result.diagnostics.get("fallback_used"),
                    "boundary_random_intercept": result.diagnostics.get(
                        "boundary_random_intercept"
                    ),
                    "nobs": result.diagnostics.get("nobs"),
                    "n_mice": result.diagnostics.get("n_mice"),
                    "n_mice_C": result.diagnostics.get("n_mice_reference"),
                    "n_mice_P": result.diagnostics.get("n_mice_comparison"),
                    "continuous_covariates": json.dumps(
                        list(result.diagnostics.get("continuous_covariates", ()))
                    ),
                    "sampling_unit_cols": json.dumps(
                        list(result.diagnostics.get("sampling_unit_cols", ()))
                    ),
                    "formula": result.diagnostics.get("formula"),
                    "common_complete_case": result.diagnostics.get(
                        "common_complete_case"
                    ),
                }
            )
    contrasts = pd.concat(contrast_frames, ignore_index=True)
    if any(column.startswith("q") or "bh_" in column for column in contrasts):
        raise RuntimeError("Fig7A/B/E manuscript statistics must contain raw P only")
    return Figure8PprhStats(
        contrasts=contrasts,
        coefficients=pd.concat(coefficient_frames, ignore_index=True),
        diagnostics=pd.DataFrame(diagnostic_rows),
        support=_build_support(table),
        fit_results=fit_results,
    )

def build_group_day_summary(table: pd.DataFrame) -> pd.DataFrame:
    finite = table.loc[np.isfinite(pd.to_numeric(table["value"], errors="coerce"))]
    grouped = finite.groupby(
        ["metric", "section", "section_idx", "group", "day"],
        observed=True,
        sort=True,
    )["value"]
    result = grouped.agg(mean="mean", std="std", n="size").reset_index()
    result["sem"] = result["std"] / np.sqrt(result["n"])
    return result

def build_annotation_payloads(
    contrasts: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    selected = contrasts.loc[contrasts["model_role"].eq("adjusted")].copy()
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_key, subset in selected.groupby(["metric", "section"], observed=True, sort=True):
        indexed = subset.set_index("estimand_contrast")
        missing = sorted(set(PLANNED_ESTIMANDS).difference(indexed.index))
        if missing:
            raise ValueError(f"Fig7A/B/E annotation stratum {raw_key} lacks {missing}")
        payloads[tuple(str(value) for value in raw_key)] = {
            "overall": raw_p_to_symbol(
                indexed.loc["centered_overall_group_effect", "p_value"]
            ),
            "slopes": {
                "C": raw_p_to_symbol(
                    indexed.loc["reference_group_day_slope", "p_value"]
                ),
                "P": raw_p_to_symbol(
                    indexed.loc["comparison_group_day_slope", "p_value"]
                ),
            },
            "interaction": raw_p_to_symbol(
                indexed.loc["group_by_day_interaction", "p_value"]
            ),
            "reported_probability": "P",
            "multiple_comparison_correction": "none",
        }
    return payloads

def raw_p_to_symbol(value: Any) -> str:
    probability = float(value)
    if not np.isfinite(probability) or probability < 0 or probability > 1:
        raise ValueError(f"invalid raw P value: {value!r}")
    if probability < 0.001:
        return "***"
    if probability < 0.01:
        return "**"
    if probability < 0.05:
        return "*"
    return "n.s."

def _normalize_pv(frame: pd.DataFrame, *, level: str) -> pd.DataFrame:
    unit = "bin" if level == "bin" else "section"
    _require_columns(
        frame,
        {"group", "mouse", "day", "metric", "value", unit},
        f"Figure-8 PV {level}",
    )
    table = _filter_locked_cohort(frame)
    table = table.loc[table["metric"].astype(str).eq("pv_separation_z")].copy()
    table["metric"] = "high_dimensional_pv"
    table["contrast"] = "correct_left_vs_right_route"
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    if level == "bin":
        table = table.rename(columns={"bin": "spatial_bin"})
        table["spatial_bin"] = pd.to_numeric(
            table["spatial_bin"], errors="raise"
        ).astype(int)
    else:
        table["section"] = table["section"].astype(str)
        section_index = {section: index for index, section in enumerate(PV_SECTIONS)}
        if not set(table["section"]).issubset(section_index):
            raise ValueError("Figure-8 PV source contains unknown sections")
        table["section_idx"] = table["section"].map(section_index).astype(int)
    return _rebuild_day_columns(table)

def _normalize_manifold(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        frame,
        {"group", "mouse", "day", "contrast", "section", "shuffle_z"},
        "Figure-8 manifold section",
    )
    table = _filter_locked_cohort(frame)
    table = table.loc[
        table["contrast"].astype(str).eq("left_vs_right_current")
    ].copy()
    table["source_section"] = table["section"].astype(str)
    table["section"] = table["source_section"].map(MANIFOLD_SECTION_MAP)
    if table["section"].isna().any():
        raise ValueError("Figure-8 manifold source contains unknown sections")
    table["section_idx"] = table["section"].map(
        {section: index for index, section in enumerate(PLOTTED_SECTIONS)}
    ).astype(int)
    table["metric"] = "neural_manifold"
    table["contrast"] = "correct_left_vs_right_route"
    table["value"] = pd.to_numeric(table["shuffle_z"], errors="coerce")
    return _rebuild_day_columns(table)

def _filter_locked_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    allowed = {
        str(group): {str(mouse) for mouse in mice}
        for group, mice in COHORT.items()
    }
    table = frame.loc[
        frame["group"].astype(str).isin(GROUP_ORDER)
        & pd.to_numeric(frame["day"], errors="coerce").isin(DAYS)
    ].copy()
    keep = np.zeros(len(table), dtype=bool)
    for group, mice in allowed.items():
        keep |= table["group"].astype(str).eq(group).to_numpy() & table[
            "mouse"
        ].astype(str).isin(mice).to_numpy()
    table = table.loc[keep].copy()
    _validate_exact_cohort(table, COHORT, label="Figure-8 locked source")
    return table

def _rebuild_day_columns(table: pd.DataFrame) -> pd.DataFrame:
    result = table.drop(
        columns=["day_idx", "day_centered", "day_c"], errors="ignore"
    ).copy()
    result["day"] = pd.to_numeric(result["day"], errors="raise").astype(int)
    result["day_idx"] = result["day"] - 1
    result["day_c"] = result["day"] - DAY_CENTER
    result["day_centered"] = result["day_c"]
    if set(result["day_c"].unique()) != {-2.0, -1.0, 0.0, 1.0, 2.0}:
        raise ValueError("Figure-8 source was not rebuilt on the D1-D5 center")
    return result

def _validate_exact_cohort(
    table: pd.DataFrame,
    cohort: Mapping[str, Sequence[str]],
    *,
    label: str,
) -> None:
    _require_columns(table, {"group", "mouse"}, label)
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
    unexpected = sorted(set(table["group"].astype(str)).difference(cohort))
    if unexpected:
        raise ValueError(f"{label} contains unexpected groups: {unexpected}")

def _validate_complete_support(
    table: pd.DataFrame,
    *,
    keys: Sequence[str],
    expected_rows: int,
    label: str,
) -> None:
    _validate_exact_cohort(table, COHORT, label=label)
    if len(table) != int(expected_rows):
        raise ValueError(f"{label} row count mismatch: {len(table)} != {expected_rows}")
    _assert_unique(table, keys, label)
    if not np.isfinite(pd.to_numeric(table["value"], errors="coerce")).all():
        raise ValueError(f"{label} contains nonfinite values")

def _build_support(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in table.groupby(
        ["metric", "section", "group"], observed=True, sort=True
    ):
        rows.append(
            {
                "metric": str(key[0]),
                "section": str(key[1]),
                "group": str(key[2]),
                "n_rows": int(len(frame)),
                "n_mice": int(frame["mouse"].nunique()),
                "n_days": int(frame["day"].nunique()),
                "mice": ";".join(sorted(frame["mouse"].astype(str).unique())),
                "days": ";".join(
                    str(day)
                    for day in sorted(
                        pd.to_numeric(frame["day"], errors="raise").astype(int).unique()
                    )
                ),
            }
        )
    return pd.DataFrame(rows)

def _require_columns(
    table: pd.DataFrame,
    columns: Sequence[str] | set[str],
    label: str,
) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")

def _assert_unique(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicate = table.duplicated(list(columns), keep=False)
    if duplicate.any():
        examples = table.loc[duplicate, list(columns)].head(10).to_dict("records")
        raise ValueError(f"{label} contains duplicate sampling units: {examples}")
