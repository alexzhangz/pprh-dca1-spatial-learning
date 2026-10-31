"""Manuscript definitions collected from figs4_ad_tmaze_pv.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any, Mapping

import numpy as np

import pandas as pd

from .fig7_s4_tmaze_pprh import (
    COVARIATE_COLUMNS,
    DAY_CENTER,
    DAYS,
    GROUP_ORDER,
    PLANNED_ESTIMANDS,
    RAW_COVARIATE_COLUMNS,
    SECTIONS,
    PprhCohort,
    prepare_cp_covariates,
    raw_p_to_symbol,
)

from .manuscript_lmm import LongitudinalLMMResult, fit_adjusted_unadjusted_lmm

CONTRASTS = ("sample_left_vs_right", "choice_left_vs_right")

CONTRAST_PANEL = {
    "sample_left_vs_right": {"heatmap": "S4A", "curves": "S4B"},
    "choice_left_vs_right": {"heatmap": "S4C", "curves": "S4D"},
}

CONTRAST_PHASE = {
    "sample_left_vs_right": "sample",
    "choice_left_vs_right": "choice",
}

PV_CACHE_NAME = "tmaze_pv_shuffle_50all_zscore.npy"

MODEL_FORMULA_CONTRACT = (
    "value ~ (day - 3) * group + "
    "route_relative_lateral_deviation_z + (1 | mouse)"
)

@dataclass(frozen=True)
class FigS4ADSourceBundle:
    """Normalized locked sources for Fig. S4A-D."""

    pv_bin: pd.DataFrame
    pv_section: pd.DataFrame
    support_qc: pd.DataFrame
    source_qc: pd.DataFrame
    covariates: pd.DataFrame
    covariate_qc: pd.DataFrame
    behavior_qc: pd.DataFrame

@dataclass
class FigS4ADStats:
    """Adjusted/unadjusted longitudinal LMM outputs."""

    contrasts: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    support: pd.DataFrame
    fit_results: dict[
        tuple[str, str], dict[str, LongitudinalLMMResult]
    ] = field(default_factory=dict, repr=False)

    @property
    def manuscript_contrasts(self) -> pd.DataFrame:
        return self.contrasts.loc[
            self.contrasts["model_role"].eq("adjusted")
        ].copy()

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()

def extract_high_dimensional_pv_sources(
    aligned_root: str | Path,
    cohort: PprhCohort,
    *,
    cache_name: str = PV_CACHE_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract both within-phase route contrasts from locked C/P caches."""

    aligned_root = Path(aligned_root)
    bin_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for group, mice in cohort.as_group_mapping().items():
        for mouse in mice:
            path = aligned_root / mouse / cache_name
            cache = np.load(path, allow_pickle=True).item()
            if str(cache.get("group")) != group or str(cache.get("mouse_name")) != mouse:
                raise ValueError(f"PV cache identity mismatch: {path}")
            for key, expected in (
                ("target_neuron_count", 50),
                ("n_repeat", 100),
                ("n_shuffle", 100),
                ("min_trials_per_condition", 3),
            ):
                if int(cache.get(key, -1)) != expected:
                    raise ValueError(
                        f"PV cache {key} mismatch for {path}: "
                        f"expected={expected}, observed={cache.get(key)!r}"
                    )
            metric_map = _unwrap_object_dict(
                cache["per_bin_separation_z_by_metric_contrast"]
            )
            spearman = _unwrap_object_dict(metric_map["spearman"])
            count_map = {
                "sample_left_vs_right": (
                    np.asarray(cache["n_sl"], dtype=float),
                    np.asarray(cache["n_sr"], dtype=float),
                ),
                "choice_left_vs_right": (
                    np.asarray(cache["n_cl"], dtype=float),
                    np.asarray(cache["n_cr"], dtype=float),
                ),
            }
            for contrast in CONTRASTS:
                if contrast not in spearman:
                    raise ValueError(f"PV cache lacks {contrast}: {path}")
                values = np.asarray(spearman[contrast], dtype=float)
                if values.ndim != 2 or values.shape[0] < 5 or values.shape[1] != 29:
                    raise ValueError(
                        f"unexpected {contrast} PV shape {values.shape}: {path}"
                    )
                left_count, right_count = count_map[contrast]
                if left_count.size < 5 or right_count.size < 5:
                    raise ValueError(f"trial-count arrays do not cover D1-D5: {path}")
                for day_index, day in enumerate(DAYS):
                    day_values = values[day_index]
                    finite_count = int(np.isfinite(day_values).sum())
                    expected_valid = bool(
                        left_count[day_index] >= 3 and right_count[day_index] >= 3
                    )
                    if expected_valid and finite_count != 29:
                        raise ValueError(
                            f"expected-valid {mouse}/{day}/{contrast} has "
                            f"{finite_count}/29 finite bins"
                        )
                    if not expected_valid and finite_count != 0:
                        raise ValueError(
                            f"invalid-support {mouse}/{day}/{contrast} has "
                            f"{finite_count}/29 finite bins"
                        )
                    status = (
                        "complete"
                        if finite_count == 29
                        else "expected_missing_insufficient_trials"
                    )
                    support_rows.append(
                        {
                            "group": group,
                            "mouse": mouse,
                            "day": day,
                            "contrast": contrast,
                            "left_condition_trials": int(left_count[day_index]),
                            "right_condition_trials": int(right_count[day_index]),
                            "minimum_trials_required": 3,
                            "expected_valid_from_trials": expected_valid,
                            "n_finite_spatial_bin": finite_count,
                            "status": status,
                            "source_path": str(path.resolve()),
                        }
                    )
                    for spatial_bin, value in enumerate(day_values):
                        bin_rows.append(
                            {
                                "project": "DNMP T-maze",
                                "metric": "high_dimensional_pv_separation",
                                "source_metric": "spearman_separation_z",
                                "contrast": contrast,
                                "phase": CONTRAST_PHASE[contrast],
                                "group": group,
                                "mouse": mouse,
                                "day": day,
                                "day_idx": int(day - 1),
                                "day_centered": float(day - DAY_CENTER),
                                "day_c": float(day - DAY_CENTER),
                                "spatial_bin": int(spatial_bin),
                                "value": float(value),
                                "source_path": str(path.resolve()),
                            }
                        )
            source_rows.append(
                {
                    "source_family": "high_dimensional_pv",
                    "group": group,
                    "mouse": mouse,
                    "source_path": str(path.resolve()),
                    "source_sha256": sha256_file(path),
                    "target_neuron_count": int(cache["target_neuron_count"]),
                    "n_repeat": int(cache["n_repeat"]),
                    "n_shuffle": int(cache["n_shuffle"]),
                    "minimum_trials_per_condition": int(
                        cache["min_trials_per_condition"]
                    ),
                    "metric": "spearman_separation_z",
                    "population_space": "original_high_dimensional",
                    "manifold_covariance": "not_applicable",
                    "status": "ok",
                }
            )

    bin_table = pd.DataFrame(bin_rows).sort_values(
        ["contrast", "group", "mouse", "day", "spatial_bin"],
        kind="mergesort",
    ).reset_index(drop=True)
    _validate_bin_grid(bin_table, cohort)
    section_rows: list[dict[str, Any]] = []
    key_columns = [
        "project",
        "metric",
        "source_metric",
        "contrast",
        "phase",
        "group",
        "mouse",
        "day",
        "day_idx",
        "day_centered",
        "day_c",
    ]
    for key, subset in bin_table.groupby(key_columns, observed=True, sort=True):
        for source_section, (section, bins) in SECTIONS.items():
            raw_values = pd.to_numeric(
                subset.loc[subset["spatial_bin"].isin(bins), "value"],
                errors="coerce",
            ).to_numpy(float)
            finite = raw_values[np.isfinite(raw_values)]
            row = dict(zip(key_columns, key, strict=True))
            row.update(
                {
                    "section": section,
                    "source_section": source_section,
                    "n_section_bin": int(len(bins)),
                    "n_finite_bin": int(finite.size),
                    "value": float(finite.mean()) if finite.size else np.nan,
                }
            )
            section_rows.append(row)
    section_table = pd.DataFrame(section_rows).sort_values(
        ["contrast", "section", "group", "mouse", "day"],
        kind="mergesort",
    ).reset_index(drop=True)
    _assert_unique(
        section_table,
        ("contrast", "section", "group", "mouse", "day"),
        "Fig. S4A-D section table",
    )
    return (
        bin_table,
        section_table,
        pd.DataFrame(support_rows).sort_values(
            ["contrast", "group", "mouse", "day"], kind="mergesort"
        ).reset_index(drop=True),
        pd.DataFrame(source_rows).sort_values(
            ["group", "mouse"], kind="mergesort"
        ).reset_index(drop=True),
    )

def build_source_bundle(
    *,
    data_root: str | Path,
    aligned_root: str | Path,
    cohort: PprhCohort,
) -> FigS4ADSourceBundle:
    """Build high-dimensional PV and covariate sources under one contract."""

    covariates, covariate_qc, behavior_qc = prepare_cp_covariates(
        data_root, cohort, strict=True
    )
    pv_bin, pv_section, support_qc, source_qc = (
        extract_high_dimensional_pv_sources(aligned_root, cohort)
    )
    return FigS4ADSourceBundle(
        pv_bin=pv_bin,
        pv_section=pv_section,
        support_qc=support_qc,
        source_qc=source_qc,
        covariates=covariates,
        covariate_qc=covariate_qc,
        behavior_qc=behavior_qc,
    )

def build_longitudinal_table(bundle: FigS4ADSourceBundle) -> pd.DataFrame:
    """Attach locked C/P covariates to one mouse-day-section value."""

    keys = ["group", "mouse", "day"]
    table = bundle.pv_section.merge(
        bundle.covariates[
            ["group", "mouse", "day", *RAW_COVARIATE_COLUMNS, *COVARIATE_COLUMNS]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if table[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("Fig. S4A-D covariate merge produced unmatched mouse-days")
    _assert_unique(
        table,
        ("contrast", "section", "group", "mouse", "day"),
        "Fig. S4A-D LMM input",
    )
    table["day_idx"] = pd.to_numeric(table["day"], errors="raise").astype(int) - 1
    table["day_centered"] = pd.to_numeric(table["day"], errors="raise") - DAY_CENTER
    table["day_c"] = table["day_centered"]
    return table.sort_values(
        ["contrast", "section", "group", "mouse", "day"], kind="mergesort"
    ).reset_index(drop=True)

def build_group_day_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return mouse-level means and SEMs for S4B/D."""

    finite = table.loc[np.isfinite(pd.to_numeric(table["value"], errors="coerce"))]
    grouped = finite.groupby(
        ["metric", "contrast", "section", "group", "day"],
        observed=True,
        sort=True,
    )["value"]
    summary = grouped.agg(mean="mean", std="std", n="size").reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    return summary

def fit_longitudinal_models(
    table: pd.DataFrame,
    *,
    maxiter: int = 1000,
) -> FigS4ADStats:
    """Fit adjusted and common-sample unadjusted mouse-level LMMs."""

    _assert_unique(
        table,
        ("contrast", "section", "group", "mouse", "day"),
        "Fig. S4A-D LMM input",
    )
    contrast_blocks: list[pd.DataFrame] = []
    coefficient_blocks: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    fit_results: dict[
        tuple[str, str], dict[str, LongitudinalLMMResult]
    ] = {}
    for raw_key, subset in table.groupby(
        ["contrast", "section"], observed=True, sort=True
    ):
        contrast, section = (str(value) for value in raw_key)
        for group in GROUP_ORDER:
            finite_group = subset.loc[
                subset["group"].eq(group)
                & np.isfinite(pd.to_numeric(subset["value"], errors="coerce"))
            ]
            if set(finite_group["day"].astype(int)) != set(DAYS):
                raise ValueError(
                    f"{contrast}/{section}/{group} lacks finite D1-D5 group support"
                )
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
            maxiter=maxiter,
        )
        fit_results[(contrast, section)] = paired
        if paired["adjusted"].diagnostics.get("nobs") != paired["unadjusted"].diagnostics.get("nobs"):
            raise RuntimeError(f"paired LMM sample mismatch: {contrast}/{section}")
        for model_role, result in paired.items():
            if not result.succeeded:
                raise RuntimeError(
                    f"LMM failed for {contrast}/{section}/{model_role}: "
                    f"{result.diagnostics}"
                )
            decorated = result.contrasts.rename(
                columns={"contrast": "estimand_contrast"}
            ).copy()
            if set(decorated["estimand_contrast"]) != set(PLANNED_ESTIMANDS):
                raise RuntimeError(f"unexpected estimands for {contrast}/{section}")
            decorated.insert(0, "model_role", model_role)
            decorated.insert(0, "section", section)
            decorated.insert(0, "contrast", contrast)
            decorated.insert(0, "metric", "high_dimensional_pv_separation")
            decorated.insert(0, "panel", CONTRAST_PANEL[contrast]["curves"])
            decorated["reported_probability"] = "P"
            decorated["reported_probability_value"] = pd.to_numeric(
                decorated["p_value"], errors="coerce"
            )
            decorated["multiple_comparison_correction"] = "none"
            decorated["manuscript_facing"] = model_role == "adjusted"
            decorated["sampling_unit"] = "mouse/day/section"
            decorated["biological_replicate"] = "mouse"
            decorated["model_formula_contract"] = MODEL_FORMULA_CONTRACT
            contrast_blocks.append(decorated)

            coefficients = result.coefficients.copy()
            coefficients.insert(0, "model_role", model_role)
            coefficients.insert(0, "section", section)
            coefficients.insert(0, "contrast", contrast)
            coefficients.insert(0, "metric", "high_dimensional_pv_separation")
            coefficients.insert(0, "panel", CONTRAST_PANEL[contrast]["curves"])
            coefficient_blocks.append(coefficients)
            diagnostics = result.diagnostics
            diagnostic_rows.append(
                {
                    "panel": CONTRAST_PANEL[contrast]["curves"],
                    "metric": "high_dimensional_pv_separation",
                    "contrast": contrast,
                    "section": section,
                    "model_role": model_role,
                    "model_status": diagnostics.get("model_status"),
                    "optimizer": diagnostics.get("optimizer"),
                    "converged": diagnostics.get("converged"),
                    "covariance_finite": diagnostics.get("covariance_finite"),
                    "fallback_used": diagnostics.get("fallback_used"),
                    "boundary_random_intercept": diagnostics.get(
                        "boundary_random_intercept"
                    ),
                    "nobs": diagnostics.get("nobs"),
                    "n_mice": diagnostics.get("n_mice"),
                    "n_mice_C": diagnostics.get("n_mice_reference"),
                    "n_mice_P": diagnostics.get("n_mice_comparison"),
                    "reference_group": diagnostics.get("reference_group"),
                    "comparison_group": diagnostics.get("comparison_group"),
                    "continuous_covariates": tuple(
                        diagnostics.get("continuous_covariates", ())
                    ),
                    "formula": diagnostics.get("formula"),
                    "common_complete_case": diagnostics.get(
                        "common_complete_case"
                    ),
                }
            )
        finite = subset.loc[
            np.isfinite(pd.to_numeric(subset["value"], errors="coerce"))
        ]
        support_rows.append(
            {
                "panel": CONTRAST_PANEL[contrast]["curves"],
                "contrast": contrast,
                "section": section,
                "nominal_rows": int(len(subset)),
                "finite_rows": int(len(finite)),
                "missing_rows": int(len(subset) - len(finite)),
                "nominal_mice_C": int(len(set(subset.loc[subset["group"].eq("C"), "mouse"]))),
                "nominal_mice_P": int(len(set(subset.loc[subset["group"].eq("P"), "mouse"]))),
                "finite_mice_C": int(finite.loc[finite["group"].eq("C"), "mouse"].nunique()),
                "finite_mice_P": int(finite.loc[finite["group"].eq("P"), "mouse"].nunique()),
                "finite_days_C": ";".join(
                    str(day)
                    for day in sorted(set(finite.loc[finite["group"].eq("C"), "day"]))
                ),
                "finite_days_P": ";".join(
                    str(day)
                    for day in sorted(set(finite.loc[finite["group"].eq("P"), "day"]))
                ),
            }
        )
    contrasts = pd.concat(contrast_blocks, ignore_index=True)
    if any(column.startswith("q") or "bh_" in column.lower() for column in contrasts):
        raise RuntimeError("Fig. S4A-D release must not contain corrected P columns")
    return FigS4ADStats(
        contrasts=contrasts,
        coefficients=pd.concat(coefficient_blocks, ignore_index=True),
        diagnostics=pd.DataFrame(diagnostic_rows),
        support=pd.DataFrame(support_rows),
        fit_results=fit_results,
    )

def build_annotation_payloads(
    contrasts: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build Fig. 3/4-style star annotations from raw adjusted P values."""

    selected = contrasts.loc[contrasts["model_role"].eq("adjusted")].copy()
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_key, subset in selected.groupby(
        ["contrast", "section"], observed=True, sort=True
    ):
        indexed = subset.set_index("estimand_contrast")
        missing = sorted(set(PLANNED_ESTIMANDS).difference(indexed.index))
        if missing:
            raise ValueError(f"annotation stratum {raw_key} lacks {missing}")
        payloads[tuple(str(value) for value in raw_key)] = {
            "overall": _raw_p_to_display_symbol(
                indexed.loc["centered_overall_group_effect", "p_value"]
            ),
            "slopes": {
                "C": _raw_p_to_display_symbol(
                    indexed.loc["reference_group_day_slope", "p_value"]
                ),
                "P": _raw_p_to_display_symbol(
                    indexed.loc["comparison_group_day_slope", "p_value"]
                ),
            },
            "interaction": _raw_p_to_display_symbol(
                indexed.loc["group_by_day_interaction", "p_value"]
            ),
            "reported_probability": "P",
            "multiple_comparison_correction": "none",
        }
    return payloads

def _raw_p_to_display_symbol(value: Any) -> str:
    """Map finite raw P values to stars and invalid Wald tests to ``n.e.``."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "n.e."
    if not np.isfinite(probability):
        return "n.e."
    return raw_p_to_symbol(probability)

def _validate_bin_grid(table: pd.DataFrame, cohort: PprhCohort) -> None:
    _assert_unique(
        table,
        ("contrast", "group", "mouse", "day", "spatial_bin"),
        "Fig. S4A-D bin table",
    )
    expected = {
        (contrast, group, mouse, day, spatial_bin)
        for contrast in CONTRASTS
        for group, mice in cohort.as_group_mapping().items()
        for mouse in mice
        for day in DAYS
        for spatial_bin in range(29)
    }
    observed = set(
        table[
            ["contrast", "group", "mouse", "day", "spatial_bin"]
        ].itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(
            "Fig. S4A-D PV bin grid mismatch; "
            f"missing={len(expected-observed)}, extra={len(observed-expected)}"
        )

def _assert_unique(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    if table.duplicated(list(columns)).any():
        examples = table.loc[
            table.duplicated(list(columns), keep=False), list(columns)
        ].head().to_dict("records")
        raise ValueError(f"{label} has duplicated keys: {examples}")

def _unwrap_object_dict(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value
