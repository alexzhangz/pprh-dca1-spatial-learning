"""Manuscript definitions collected from fig7_s4_tmaze_pprh.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import hashlib

import json

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from .behavior_covariates import pooled_z_standardize_covariates

from .fig3_tmaze_behavior import build_fig3h_tmaze_behavior

from .manuscript_lmm import LongitudinalLMMResult, fit_adjusted_unadjusted_lmm

DAYS = (1, 2, 3, 4, 5)

GROUP_ORDER = ("C", "P")

DAY_CENTER = 3.0

CONTRASTS = (
    "sample_vs_choice",
    "sample_left_vs_right",
    "choice_left_vs_right",
)

MANIFOLD_SOURCE_CONTRASTS = (
    "sample_vs_choice_left",
    "sample_vs_choice_right",
    "sample_left_vs_right",
    "choice_left_vs_right",
    "sample_vs_choice",
)

SECTIONS = {
    "Stem": ("stem", np.arange(0, 13, dtype=int)),
    "Choice arm": ("choice_arm", np.arange(13, 29, dtype=int)),
}

COVARIATE_COLUMNS = (
    "route_relative_lateral_deviation_z",
)

RAW_COVARIATE_COLUMNS = (
    "mean_speed_cm_s",
    "route_relative_lateral_deviation_cm",
)

MODEL_FORMULA_CONTRACT = (
    "value ~ (day - 3) * group + "
    "route_relative_lateral_deviation_z + (1 | mouse)"
)

PLANNED_ESTIMANDS = (
    "reference_group_day_slope",
    "comparison_group_day_slope",
    "group_by_day_interaction",
    "centered_overall_group_effect",
    "between_group_D1",
    "between_group_D5",
)

PPRH_SUPPORT_STATUSES = (
    "complete",
    "expected_missing_insufficient_trials",
)

PV_CACHE_NAME = "tmaze_pv_shuffle_50all_zscore.npy"

_PANEL_ROWS = (
    (
        "7F",
        "sample_vs_choice",
        "high_dimensional_pv_separation",
        "spatial_bin_heatmap",
        "regenerate",
    ),
    (
        "7G",
        "sample_vs_choice",
        "high_dimensional_pv_separation",
        "section_curves",
        "refit_and_redraw",
    ),
    (
        "7H",
        "sample_vs_choice",
        "representative_neural_manifold",
        "static_representative",
        "retain_no_inference",
    ),
    (
        "7I",
        "sample_vs_choice",
        "neural_manifold_separation",
        "section_curves",
        "refit_and_redraw",
    ),
    (
        "S4E",
        "sample_left_vs_right",
        "neural_manifold_separation",
        "section_curves",
        "refit_and_redraw",
    ),
    (
        "S4F",
        "choice_left_vs_right",
        "neural_manifold_separation",
        "section_curves",
        "refit_and_redraw",
    ),
)

@dataclass(frozen=True)
class PprhCohort:
    """Exact C/P cohort used by the T-maze recording comparison."""

    control: tuple[str, ...]
    pprh: tuple[str, ...]

    def as_group_mapping(self) -> dict[str, tuple[str, ...]]:
        return {"C": self.control, "P": self.pprh}

@dataclass(frozen=True)
class PprhSourceBundle:
    """Normalized source tables entering the Fig. 7/S4 rebuild."""

    pv_bin: pd.DataFrame
    pv_section: pd.DataFrame
    manifold_section: pd.DataFrame
    manifold_support_qc: pd.DataFrame
    covariates: pd.DataFrame
    covariate_qc: pd.DataFrame
    source_qc: pd.DataFrame

@dataclass
class PprhLongitudinalStats:
    """Decorated adjusted/unadjusted C-versus-P longitudinal fits."""

    contrasts: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    support: pd.DataFrame
    fit_results: dict[
        tuple[str, str, str], dict[str, LongitudinalLMMResult]
    ] = field(default_factory=dict, repr=False)

    @property
    def manuscript_contrasts(self) -> pd.DataFrame:
        return self.contrasts.loc[
            self.contrasts["model_role"].eq("adjusted")
        ].copy()

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()

def panel_map() -> pd.DataFrame:
    """Return the accepted, version-locked Fig. 7F-I/S4E-F mapping."""

    return pd.DataFrame(
        _PANEL_ROWS,
        columns=("panel", "contrast", "metric", "display", "action"),
    )

def prepare_cp_covariates(
    data_root: str | Path,
    cohort: PprhCohort,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and pooled-z-standardize C/P mouse-day covariates."""

    atomic = build_fig3h_tmaze_behavior(
        Path(data_root),
        group_mice=cohort.as_group_mapping(),
        days=DAYS,
        strict=strict,
    )
    mouse_day = atomic.mouse_day.copy()
    standardized, standardization_qc = standardize_cp_covariate_table(
        mouse_day, cohort
    )
    return standardized, standardization_qc, atomic.qc

def standardize_cp_covariate_table(
    mouse_day: pd.DataFrame,
    cohort: PprhCohort,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the C/P grid and standardize the lateral-only adjustment."""

    required = {"group", "mouse", "day", *RAW_COVARIATE_COLUMNS}
    _require_columns(mouse_day, required, "C/P covariate table")
    table = mouse_day.copy()
    table["group"] = table["group"].astype(str)
    table["mouse"] = table["mouse"].astype(str)
    table["day"] = pd.to_numeric(table["day"], errors="raise").astype(int)
    allowed = {
        (group, mouse, day)
        for group, mice in cohort.as_group_mapping().items()
        for mouse in mice
        for day in DAYS
    }
    observed = set(
        table[["group", "mouse", "day"]].itertuples(index=False, name=None)
    )
    if observed != allowed:
        raise ValueError(
            "C/P covariates do not match the locked mouse-day grid; "
            f"missing={sorted(allowed-observed)}, extra={sorted(observed-allowed)}"
        )
    _assert_unique(table, ("group", "mouse", "day"), "C/P covariates")
    standardized, qc = pooled_z_standardize_covariates(
        table,
        columns=("route_relative_lateral_deviation_cm",),
        output_columns={
            "route_relative_lateral_deviation_cm": (
                "route_relative_lateral_deviation_z"
            ),
        },
        ddof=1,
    )
    if not qc["status"].eq("ok").all():
        raise ValueError(f"C/P covariate standardization failed: {qc.to_dict('records')}")
    for column in (*RAW_COVARIATE_COLUMNS, *COVARIATE_COLUMNS):
        values = pd.to_numeric(standardized[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"C/P covariate {column!r} contains nonfinite values")
    keep = [
        "group",
        "mouse",
        "day",
        *RAW_COVARIATE_COLUMNS,
        *COVARIATE_COLUMNS,
    ]
    optional = [
        column
        for column in (
            "covariate_raw_frame_count",
            "covariate_running_frame_count",
            "covariate_valid_frame_count",
            "covariate_status",
        )
        if column in standardized
    ]
    standardized = standardized.loc[:, keep + optional].sort_values(
        ["group", "mouse", "day"], kind="mergesort"
    ).reset_index(drop=True)
    qc = qc.assign(
        cohort="C7_P5",
        groups="C;P",
        days="D1-D5",
        standardization_scope="locked_C7_P5_mouse_day_pool",
    )
    return standardized, qc

def extract_high_dimensional_pv_sources(
    aligned_root: str | Path,
    cohort: PprhCohort,
    *,
    cache_name: str = PV_CACHE_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract Fig. 7F/G sample-choice Spearman separation from C/P caches."""

    aligned_root = Path(aligned_root)
    bin_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
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
            values = np.asarray(spearman["sample_vs_choice"], dtype=float)
            if values.ndim != 2 or values.shape[0] < 5 or values.shape[1] != 29:
                raise ValueError(f"unexpected sample-choice PV shape {values.shape}: {path}")
            for day_index, day in enumerate(DAYS):
                for spatial_bin, value in enumerate(values[day_index]):
                    bin_rows.append(
                        {
                            "project": "DNMP T-maze",
                            "metric": "high_dimensional_pv_separation",
                            "source_metric": "spearman_separation_z",
                            "contrast": "sample_vs_choice",
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
            qc_rows.append(
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
                    "dprime_covariance": "not_applicable_for_spearman",
                    "status": "ok",
                }
            )
    bin_table = pd.DataFrame(bin_rows).sort_values(
        ["group", "mouse", "day", "spatial_bin"], kind="mergesort"
    ).reset_index(drop=True)
    _validate_pv_bin_table(bin_table, cohort)
    section_rows: list[dict[str, Any]] = []
    key_columns = [
        "project",
        "metric",
        "source_metric",
        "contrast",
        "group",
        "mouse",
        "day",
        "day_idx",
        "day_centered",
        "day_c",
    ]
    for key, subset in bin_table.groupby(key_columns, observed=True, sort=True):
        for source_section, (section, bins) in SECTIONS.items():
            selected = subset.loc[subset["spatial_bin"].isin(bins), "value"]
            finite = pd.to_numeric(selected, errors="coerce").to_numpy(float)
            finite = finite[np.isfinite(finite)]
            row = dict(zip(key_columns, key, strict=True))
            row.update(
                {
                    "section": section,
                    "source_section": source_section,
                    "n_section_bin": int(len(bins)),
                    "n_finite_bin": int(finite.size),
                    "value": float(np.mean(finite)) if finite.size else np.nan,
                }
            )
            section_rows.append(row)
    section_table = pd.DataFrame(section_rows).sort_values(
        ["group", "mouse", "day", "section"], kind="mergesort"
    ).reset_index(drop=True)
    _assert_unique(
        section_table,
        ("metric", "contrast", "section", "group", "mouse", "day"),
        "high-dimensional PV section table",
    )
    return bin_table, section_table, pd.DataFrame(qc_rows)

def load_legacy_pooled_manifold_section(
    control_path: str | Path,
    pprh_path: str | Path,
    pprh_qc_path: str | Path,
    cohort: PprhCohort,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load C and P pooled-covariance manifold section tables.

    The control source is the historical pooled snapshot.  The P source must
    be a separately generated pooled table accompanied by a mouse-level QC
    table.  A within-class QC value is a blocking error.
    """

    control_path = Path(control_path)
    pprh_path = Path(pprh_path)
    pprh_qc_path = Path(pprh_qc_path)
    control = pd.read_csv(control_path)
    pprh = pd.read_csv(pprh_path)
    pprh_qc = pd.read_csv(pprh_qc_path)

    normalized = pd.concat(
        [
            _normalize_manifold_section_source(
                control,
                expected_group="C",
                expected_mice=cohort.control,
                source_path=control_path,
            ),
            _normalize_manifold_section_source(
                pprh,
                expected_group="P",
                expected_mice=cohort.pprh,
                source_path=pprh_path,
            ),
        ],
        ignore_index=True,
    )
    _validate_pprh_manifold_qc(pprh_qc, cohort.pprh)
    p_qc = pprh_qc.copy()
    p_qc["source_family"] = "neural_manifold"
    p_qc["source_path"] = str(pprh_path.resolve())
    p_qc["source_sha256"] = sha256_file(pprh_path)
    p_qc["qc_path"] = str(pprh_qc_path.resolve())
    p_qc["qc_sha256"] = sha256_file(pprh_qc_path)
    p_qc["status"] = "ok"
    c_qc = pd.DataFrame(
        [
            {
                "source_family": "neural_manifold",
                "group": "C",
                "mouse": mouse,
                "reducer": "spectral",
                "embedding_dim": 15,
                "target_neuron_count": 50,
                "n_repeat": 100,
                "n_shuffle": 100,
                "dprime_covariance": "legacy_pooled",
                "source_path": str(control_path.resolve()),
                "source_sha256": sha256_file(control_path),
                "qc_path": "tracked analysis contract",
                "qc_sha256": "",
                "status": "ok",
            }
            for mouse in cohort.control
        ]
    )
    qc_columns = sorted(set(c_qc.columns).union(p_qc.columns))
    source_qc = pd.concat(
        [c_qc.reindex(columns=qc_columns), p_qc.reindex(columns=qc_columns)],
        ignore_index=True,
    )
    _validate_manifold_section_support(normalized, cohort)
    return normalized.sort_values(
        ["contrast", "section", "group", "mouse", "day"], kind="mergesort"
    ).reset_index(drop=True), source_qc

def load_pprh_manifold_support_qc(
    path: str | Path,
    cohort: PprhCohort,
) -> pd.DataFrame:
    """Load and fail-closed validate the P1-P5 day/contrast support audit.

    The pooled-manifold bin/section tables intentionally omit contrasts for
    which every spatial bin is unavailable because the relevant trial class
    has fewer than three trials.  This audit distinguishes those prespecified
    expected-missing cells from compute failures and prevents silent row loss.
    """

    path = Path(path)
    table = pd.read_csv(path)
    required = {
        "group",
        "mouse",
        "day",
        "day_idx",
        "day_centered",
        "contrast",
        "is_primary_contrast",
        "valid_subject_day",
        "minimum_condition_count",
        "min_trials_required",
        "expected_valid_from_trials",
        "n_bin_expected",
        "finite_dprime_bin_count",
        "finite_shuffle_z_bin_count",
        "finite_union_bin_count",
        "support_status",
    }
    _require_columns(table, required, "pPRH pooled-manifold support QC")
    table = table.copy()
    table["group"] = table["group"].astype(str)
    table["mouse"] = table["mouse"].astype(str)
    table["contrast"] = table["contrast"].astype(str)
    table["support_status"] = table["support_status"].astype(str)
    table["day"] = pd.to_numeric(table["day"], errors="raise").astype(int)
    if set(table["group"]) != {"P"}:
        raise ValueError("pPRH manifold support QC must contain only group P")
    if set(table["mouse"]) != set(cohort.pprh):
        raise ValueError(
            "pPRH manifold support QC cohort mismatch: "
            f"expected={sorted(cohort.pprh)}, observed={sorted(table['mouse'])}"
        )
    if set(table["day"]) != set(DAYS):
        raise ValueError("pPRH manifold support QC must cover group-level D1-D5")
    if set(table["contrast"]) != set(MANIFOLD_SOURCE_CONTRASTS):
        raise ValueError("pPRH manifold support QC contrast set mismatch")
    _assert_unique(
        table,
        ("group", "mouse", "day", "contrast"),
        "pPRH pooled-manifold support QC",
    )
    expected_rows = len(cohort.pprh) * len(DAYS) * len(MANIFOLD_SOURCE_CONTRASTS)
    if len(table) != expected_rows:
        raise ValueError(
            "pPRH manifold support QC must contain one row per mouse/day/contrast; "
            f"expected={expected_rows}, observed={len(table)}"
        )

    primary = _coerce_bool_series(
        table["is_primary_contrast"], "is_primary_contrast"
    )
    valid_subject_day = _coerce_bool_series(
        table["valid_subject_day"], "valid_subject_day"
    )
    expected_valid = _coerce_bool_series(
        table["expected_valid_from_trials"], "expected_valid_from_trials"
    )
    expected_primary = table["contrast"].isin(CONTRASTS)
    if not primary.eq(expected_primary).all():
        raise ValueError("pPRH manifold support QC primary-contrast flag mismatch")
    if not valid_subject_day.all():
        raise ValueError("pPRH manifold support QC contains an invalid subject-day")
    if not table["support_status"].isin(PPRH_SUPPORT_STATUSES).all():
        raise ValueError("pPRH manifold support QC contains an unaccepted status")

    numeric_columns = (
        "minimum_condition_count",
        "min_trials_required",
        "n_bin_expected",
        "finite_dprime_bin_count",
        "finite_shuffle_z_bin_count",
        "finite_union_bin_count",
    )
    numeric = {
        column: pd.to_numeric(table[column], errors="raise")
        for column in numeric_columns
    }
    if not numeric["min_trials_required"].eq(3).all():
        raise ValueError("pPRH manifold support QC requires three trials per condition")
    if not numeric["n_bin_expected"].eq(29).all():
        raise ValueError("pPRH manifold support QC requires 29 spatial bins")

    complete = table["support_status"].eq("complete")
    missing = table["support_status"].eq(
        "expected_missing_insufficient_trials"
    )
    if not expected_valid.eq(complete).all():
        raise ValueError("pPRH manifold support status disagrees with trial support")
    for column in (
        "finite_dprime_bin_count",
        "finite_shuffle_z_bin_count",
        "finite_union_bin_count",
    ):
        if not numeric[column].loc[complete].eq(29).all():
            raise ValueError(f"complete pPRH support rows require 29 finite {column}")
        if not numeric[column].loc[missing].eq(0).all():
            raise ValueError(f"expected-missing pPRH rows require zero finite {column}")
    if not (
        numeric["minimum_condition_count"].loc[missing]
        < numeric["min_trials_required"].loc[missing]
    ).all():
        raise ValueError("expected-missing pPRH rows are not explained by trial counts")
    if int(missing.sum()) != 12 or int((missing & primary).sum()) != 9:
        raise ValueError(
            "unexpected P1-P5 support pattern; required 12 total and 9 primary "
            f"expected-missing cells, observed={int(missing.sum())}/"
            f"{int((missing & primary).sum())}"
        )

    # Source indexing fields are not trusted.  Rebuild them from D1-D5 just
    # as for every model-facing source table.
    table["day_idx"] = table["day"] - 1
    table["day_centered"] = table["day"].astype(float) - DAY_CENTER
    table["day_c"] = table["day"].astype(float) - DAY_CENTER
    table["is_primary_contrast"] = primary
    table["valid_subject_day"] = valid_subject_day
    table["expected_valid_from_trials"] = expected_valid
    table["source_path"] = str(path.resolve())
    table["source_sha256"] = sha256_file(path)
    return table.sort_values(
        ["mouse", "day", "contrast"], kind="mergesort"
    ).reset_index(drop=True)

def build_source_bundle(
    *,
    data_root: str | Path,
    aligned_root: str | Path,
    control_manifold_section: str | Path,
    pprh_manifold_section: str | Path,
    pprh_manifold_qc: str | Path,
    pprh_manifold_support_qc: str | Path,
    cohort: PprhCohort,
) -> PprhSourceBundle:
    """Build all normalized C/P sources under one locked method contract."""

    covariates, covariate_qc, behavior_qc = prepare_cp_covariates(
        data_root, cohort, strict=True
    )
    pv_bin, pv_section, pv_qc = extract_high_dimensional_pv_sources(
        aligned_root, cohort
    )
    manifold_section, manifold_qc = load_legacy_pooled_manifold_section(
        control_manifold_section,
        pprh_manifold_section,
        pprh_manifold_qc,
        cohort,
    )
    manifold_support_qc = load_pprh_manifold_support_qc(
        pprh_manifold_support_qc,
        cohort,
    )
    source_qc = pd.concat([pv_qc, manifold_qc], ignore_index=True, sort=False)
    behavior_qc = behavior_qc.copy()
    behavior_qc["source_family"] = "behavior_covariates"
    primary_complete = (
        behavior_qc["primary_complete"].astype(bool)
        if "primary_complete" in behavior_qc
        else pd.Series(False, index=behavior_qc.index)
    )
    behavior_qc["status"] = np.where(
        primary_complete, "ok", "failed"
    )
    source_qc = pd.concat(
        [source_qc, behavior_qc], ignore_index=True, sort=False
    )
    return PprhSourceBundle(
        pv_bin=pv_bin,
        pv_section=pv_section,
        manifold_section=manifold_section,
        manifold_support_qc=manifold_support_qc,
        covariates=covariates,
        covariate_qc=covariate_qc,
        source_qc=source_qc,
    )

def build_longitudinal_table(bundle: PprhSourceBundle) -> pd.DataFrame:
    """Attach C/P covariates and return one canonical section-level table."""

    sources = pd.concat(
        [bundle.pv_section, bundle.manifold_section],
        ignore_index=True,
        sort=False,
    )
    keys = ["group", "mouse", "day"]
    overlap = sorted(
        set(bundle.covariates.columns).intersection(sources.columns).difference(keys)
    )
    if overlap:
        raise ValueError(f"source table already contains covariates: {overlap}")
    table = sources.merge(
        bundle.covariates[
            ["group", "mouse", "day", *RAW_COVARIATE_COLUMNS, *COVARIATE_COLUMNS]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if table[list(COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("covariate merge produced unmatched C/P mouse-days")
    required = {
        "project",
        "metric",
        "contrast",
        "section",
        "group",
        "mouse",
        "day",
        "value",
        *COVARIATE_COLUMNS,
    }
    _require_columns(table, required, "C/P longitudinal table")
    _assert_unique(
        table,
        ("metric", "contrast", "section", "group", "mouse", "day"),
        "C/P longitudinal table",
    )
    table["day_c"] = pd.to_numeric(table["day"], errors="raise") - DAY_CENTER
    table["day_idx"] = pd.to_numeric(table["day"], errors="raise").astype(int) - 1
    table["day_centered"] = table["day_c"]
    return table.sort_values(
        ["metric", "contrast", "section", "group", "mouse", "day"],
        kind="mergesort",
    ).reset_index(drop=True)

def build_group_day_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return mouse-level mean, SEM and support for each displayed curve."""

    finite = table.loc[np.isfinite(pd.to_numeric(table["value"], errors="coerce"))]
    grouped = finite.groupby(
        ["metric", "contrast", "section", "group", "day"],
        observed=True,
        sort=True,
    )["value"]
    summary = grouped.agg(mean="mean", std="std", n="size").reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    return summary

def fit_cp_longitudinal_models(
    table: pd.DataFrame,
    *,
    maxiter: int = 1000,
) -> PprhLongitudinalStats:
    """Fit adjusted and shared-sample unadjusted C/P mouse-level LMMs."""

    required = {
        "metric",
        "contrast",
        "section",
        "group",
        "mouse",
        "day",
        "value",
        *COVARIATE_COLUMNS,
    }
    _require_columns(table, required, "C/P longitudinal table")
    unexpected_groups = sorted(set(table["group"].astype(str)).difference(GROUP_ORDER))
    if unexpected_groups:
        raise ValueError(f"unexpected C/P groups: {unexpected_groups}")
    _assert_unique(
        table,
        ("metric", "contrast", "section", "group", "mouse", "day"),
        "C/P longitudinal table",
    )

    contrast_blocks: list[pd.DataFrame] = []
    coefficient_blocks: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    fit_results: dict[
        tuple[str, str, str], dict[str, LongitudinalLMMResult]
    ] = {}
    strata = ("metric", "contrast", "section")
    for raw_key, subset in table.groupby(list(strata), observed=True, sort=True):
        key = tuple(str(value) for value in raw_key)
        for group in GROUP_ORDER:
            present_days = set(
                pd.to_numeric(
                    subset.loc[subset["group"].eq(group), "day"], errors="coerce"
                ).dropna().astype(int)
            )
            if present_days != set(DAYS):
                raise ValueError(
                    f"stratum {key}, group {group} lacks D1-D5 group support: "
                    f"{sorted(present_days)}"
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
        fit_results[key] = paired
        adjusted_nobs = paired["adjusted"].diagnostics.get("nobs")
        unadjusted_nobs = paired["unadjusted"].diagnostics.get("nobs")
        if adjusted_nobs != unadjusted_nobs:
            raise RuntimeError(f"paired C/P LMM samples differ for stratum {key}")
        for model_role, result in paired.items():
            if not result.succeeded:
                raise RuntimeError(
                    f"C/P LMM failed for {key}/{model_role}: {result.diagnostics}"
                )
            decorated = result.contrasts.rename(
                columns={"contrast": "estimand_contrast"}
            ).copy()
            if set(decorated["estimand_contrast"]) != set(PLANNED_ESTIMANDS):
                raise RuntimeError(f"unexpected C/P estimands for {key}")
            decorated.insert(0, "model_role", model_role)
            decorated.insert(0, "section", key[2])
            decorated.insert(0, "contrast", key[1])
            decorated.insert(0, "metric", key[0])
            decorated["reported_probability"] = "P"
            decorated["reported_probability_value"] = pd.to_numeric(
                decorated["p_value"], errors="coerce"
            )
            decorated["multiple_comparison_correction"] = "none"
            decorated["manuscript_facing"] = model_role == "adjusted"
            decorated["sampling_unit"] = "mouse/day"
            decorated["biological_replicate"] = "mouse"
            decorated["model_formula_contract"] = MODEL_FORMULA_CONTRACT
            decorated["panel"] = decorated.apply(
                lambda row: _panel_for_metric_contrast(key[0], key[1]), axis=1
            )
            contrast_blocks.append(decorated)

            coefficients = result.coefficients.copy()
            coefficients.insert(0, "model_role", model_role)
            coefficients.insert(0, "section", key[2])
            coefficients.insert(0, "contrast", key[1])
            coefficients.insert(0, "metric", key[0])
            coefficient_blocks.append(coefficients)
            diagnostics = result.diagnostics
            diagnostic_rows.append(
                {
                    "metric": key[0],
                    "contrast": key[1],
                    "section": key[2],
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
                    "sampling_unit_cols": tuple(
                        diagnostics.get("sampling_unit_cols", ())
                    ),
                    "formula": diagnostics.get("formula"),
                    "common_complete_case": diagnostics.get(
                        "common_complete_case"
                    ),
                    "optimizer_attempts": json.dumps(
                        diagnostics.get("optimizer_attempts", ()),
                        sort_keys=True,
                        default=str,
                    ),
                }
            )
    contrasts = pd.concat(contrast_blocks, ignore_index=True)
    if any(column.startswith("q") or "bh_" in column for column in contrasts):
        raise RuntimeError("C/P release must not contain multiplicity-corrected columns")
    coefficients = pd.concat(coefficient_blocks, ignore_index=True)
    support = _build_support(table)
    return PprhLongitudinalStats(
        contrasts=contrasts,
        coefficients=coefficients,
        diagnostics=pd.DataFrame(diagnostic_rows),
        support=support,
        fit_results=fit_results,
    )

def build_annotation_payloads(
    contrasts: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Build Fig. 3/4-style raw-P annotations for each C/P curve stratum."""

    selected = contrasts.loc[contrasts["model_role"].eq("adjusted")].copy()
    payloads: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_key, subset in selected.groupby(
        ["metric", "contrast", "section"], observed=True, sort=True
    ):
        indexed = subset.set_index("estimand_contrast")
        missing = sorted(set(PLANNED_ESTIMANDS).difference(indexed.index))
        if missing:
            raise ValueError(f"annotation stratum {raw_key} lacks {missing}")
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
    """Map one raw two-sided P value to the locked manuscript star grammar."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "n.e."
    if not np.isfinite(probability):
        return "n.e."
    if probability < 0.001:
        return "***"
    if probability < 0.01:
        return "**"
    if probability < 0.05:
        return "*"
    return "n.s."

def _normalize_manifold_section_source(
    raw: pd.DataFrame,
    *,
    expected_group: str,
    expected_mice: Sequence[str],
    source_path: Path,
) -> pd.DataFrame:
    required = {
        "group",
        "mouse",
        "day",
        "contrast",
        "section",
        "n_section_bin",
        "shuffle_z",
    }
    _require_columns(raw, required, f"manifold section source {source_path}")
    table = raw.copy()
    table["group"] = table["group"].astype(str)
    table["mouse"] = table["mouse"].astype(str)
    table["day"] = pd.to_numeric(table["day"], errors="raise").astype(int)
    table = table.loc[
        table["group"].eq(expected_group)
        & table["mouse"].isin(tuple(expected_mice))
        & table["day"].isin(DAYS)
        & table["contrast"].isin(CONTRASTS)
        & table["section"].isin(SECTIONS)
    ].copy()
    observed_mice = set(table["mouse"])
    if observed_mice != set(expected_mice):
        raise ValueError(
            f"manifold {expected_group} cohort mismatch in {source_path}; "
            f"expected={sorted(expected_mice)}, observed={sorted(observed_mice)}"
        )
    table["project"] = "DNMP T-maze"
    table["metric"] = "neural_manifold_separation"
    table["source_metric"] = "shuffle_z"
    table["source_section"] = table["section"].astype(str)
    table["section"] = table["source_section"].map(
        {source: normalized for source, (normalized, _bins) in SECTIONS.items()}
    )
    table["value"] = pd.to_numeric(table["shuffle_z"], errors="coerce")
    # Source snapshots used incompatible centers (historical C: D1=-3;
    # locked P run: D1=-2).  Never propagate either source field.  Rebuild
    # both indices from the biological D1-D5 label after filtering.
    table["day_idx"] = table["day"].astype(int) - 1
    table["day_centered"] = table["day"].astype(float) - DAY_CENTER
    table["day_c"] = table["day"].astype(float) - DAY_CENTER
    table["source_path"] = str(source_path.resolve())
    table["n_section_bin"] = pd.to_numeric(
        table["n_section_bin"], errors="raise"
    ).astype(int)
    expected_bins = table["source_section"].map(
        {source: len(bins) for source, (_section, bins) in SECTIONS.items()}
    )
    if not table["n_section_bin"].eq(expected_bins).all():
        raise ValueError(f"manifold section bin count mismatch: {source_path}")
    _assert_unique(
        table,
        ("contrast", "section", "group", "mouse", "day"),
        f"manifold section source {source_path}",
    )
    keep = [
        "project",
        "metric",
        "source_metric",
        "contrast",
        "group",
        "mouse",
        "day",
        "day_idx",
        "day_centered",
        "day_c",
        "section",
        "source_section",
        "n_section_bin",
        "value",
        "source_path",
    ]
    return table.loc[:, keep]

def _validate_pprh_manifold_qc(qc: pd.DataFrame, expected_mice: Sequence[str]) -> None:
    required = {
        "group",
        "mouse",
        "reducer",
        "embedding_dim",
        "target_neuron_count",
        "selected_neuron_count",
        "n_repeat",
        "n_shuffle",
        "valid_days",
        "required_days",
        "dprime_covariance",
    }
    _require_columns(qc, required, "pPRH pooled-manifold QC")
    table = qc.copy()
    table["group"] = table["group"].astype(str)
    table["mouse"] = table["mouse"].astype(str)
    if set(table["group"]) != {"P"}:
        raise ValueError("pPRH manifold QC must contain only group P")
    if set(table["mouse"]) != set(expected_mice):
        raise ValueError(
            "pPRH manifold QC cohort mismatch: "
            f"expected={sorted(expected_mice)}, observed={sorted(table['mouse'])}"
        )
    _assert_unique(table, ("group", "mouse"), "pPRH manifold QC")
    checks = {
        "reducer": table["reducer"].astype(str).str.lower().eq("spectral"),
        "embedding_dim": pd.to_numeric(table["embedding_dim"], errors="coerce").eq(15),
        "target_neuron_count": pd.to_numeric(
            table["target_neuron_count"], errors="coerce"
        ).eq(50),
        "selected_neuron_count": pd.to_numeric(
            table["selected_neuron_count"], errors="coerce"
        ).eq(50),
        "n_repeat": pd.to_numeric(table["n_repeat"], errors="coerce").eq(100),
        "n_shuffle": pd.to_numeric(table["n_shuffle"], errors="coerce").eq(100),
        "valid_days": pd.to_numeric(table["valid_days"], errors="coerce").eq(5),
        "required_days": pd.to_numeric(table["required_days"], errors="coerce").eq(5),
        "dprime_covariance": table["dprime_covariance"].astype(str).eq(
            "legacy_pooled"
        ),
    }
    failed = [key for key, condition in checks.items() if not condition.all()]
    if failed:
        raise ValueError(f"pPRH pooled-manifold QC failed fields: {failed}")

def _validate_manifold_section_support(
    table: pd.DataFrame,
    cohort: PprhCohort,
) -> None:
    _assert_unique(
        table,
        ("contrast", "section", "group", "mouse", "day"),
        "combined pooled-manifold table",
    )
    for contrast in CONTRASTS:
        subset = table.loc[table["contrast"].eq(contrast)]
        for group, mice in cohort.as_group_mapping().items():
            group_table = subset.loc[subset["group"].eq(group)]
            if set(group_table["mouse"]) != set(mice):
                raise ValueError(f"manifold {contrast}/{group} nominal cohort mismatch")
            if set(group_table["day"]) != set(DAYS):
                raise ValueError(f"manifold {contrast}/{group} lacks D1-D5 group support")
            counts = group_table.groupby(
                ["mouse", "day"], observed=True
            )["section"].nunique()
            if not counts.isin((2,)).all():
                raise ValueError(
                    f"manifold {contrast}/{group} has partial section support"
                )

def _validate_pv_bin_table(table: pd.DataFrame, cohort: PprhCohort) -> None:
    _assert_unique(
        table,
        ("group", "mouse", "day", "spatial_bin"),
        "high-dimensional PV bin table",
    )
    expected = {
        (group, mouse, day, spatial_bin)
        for group, mice in cohort.as_group_mapping().items()
        for mouse in mice
        for day in DAYS
        for spatial_bin in range(29)
    }
    observed = set(
        table[["group", "mouse", "day", "spatial_bin"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected:
        raise ValueError(
            "high-dimensional PV bin grid mismatch; "
            f"missing={len(expected-observed)}, extra={len(observed-expected)}"
        )

def _build_support(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, subset in table.groupby(
        ["metric", "contrast", "section", "group", "day"],
        observed=True,
        sort=True,
    ):
        value = pd.to_numeric(subset["value"], errors="coerce").to_numpy(float)
        covariates = np.column_stack(
            [
                pd.to_numeric(subset[column], errors="coerce").to_numpy(float)
                for column in COVARIATE_COLUMNS
            ]
        )
        complete = np.isfinite(value) & np.isfinite(covariates).all(axis=1)
        row = dict(
            zip(
                ("metric", "contrast", "section", "group", "day"),
                key,
                strict=True,
            )
        )
        row.update(
            {
                "nominal_n": int(subset["mouse"].nunique()),
                "finite_value_n": int(np.isfinite(value).sum()),
                "adjusted_complete_n": int(complete.sum()),
                "finite_mice": ";".join(
                    sorted(subset.loc[np.isfinite(value), "mouse"].astype(str).unique())
                ),
                "adjusted_complete_mice": ";".join(
                    sorted(subset.loc[complete, "mouse"].astype(str).unique())
                ),
                "sampling_unit": "mouse/day",
                "biological_replicate": "mouse",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

def _panel_for_metric_contrast(metric: str, contrast: str) -> str:
    mapping = panel_map()
    matches = mapping.loc[
        mapping["metric"].eq(metric)
        & mapping["contrast"].eq(contrast)
        & mapping["display"].eq("section_curves"),
        "panel",
    ]
    if len(matches) != 1:
        raise ValueError(
            f"metric/contrast has no unique accepted panel: {metric}/{contrast}"
        )
    return str(matches.iloc[0])

def _unwrap_object_dict(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value

def _coerce_bool_series(series: pd.Series, label: str) -> pd.Series:
    """Parse a CSV boolean column without accepting arbitrary truthy text."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        invalid = sorted(normalized.loc[mapped.isna()].unique())
        raise ValueError(f"{label} contains invalid boolean values: {invalid}")
    return mapped.astype(bool)

def _require_columns(table: pd.DataFrame, columns: Sequence[str] | set[str], label: str) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")

def _assert_unique(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicated = table.duplicated(list(columns), keep=False)
    if duplicated.any():
        examples = (
            table.loc[duplicated, list(columns)]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"{label} has duplicate keys {tuple(columns)}: {examples}")
