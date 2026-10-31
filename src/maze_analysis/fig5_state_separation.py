"""Manuscript definitions collected from fig5_state_separation.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from .manuscript_cohort import (
    DEFAULT_MANUSCRIPT_COHORT_PATH,
    group_mice as cohort_group_mice,
    load_manuscript_cohort,
)

DAYS = (1, 2, 3, 4, 5)

DAY_CENTER = 3.0

PROJECT_ORDER = ("Figure-8", "DNMP T-maze")

GROUP_ORDER = ("C", "I")

def _group_mice_from_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, tuple[str, ...]]]:
    """Translate the tracked manuscript contract to Fig. 5 project labels."""

    return {
        "Figure-8": cohort_group_mice(contract, "figure_8_maze"),
        "DNMP T-maze": cohort_group_mice(contract, "t_maze"),
    }

ACTIVE_MANUSCRIPT_COHORT = load_manuscript_cohort(DEFAULT_MANUSCRIPT_COHORT_PATH)

LOCKED_GROUP_MICE = _group_mice_from_contract(ACTIVE_MANUSCRIPT_COHORT)

def configure_manuscript_cohort(
    path: str | Path = DEFAULT_MANUSCRIPT_COHORT_PATH,
) -> dict[str, Any]:
    """Load one tracked cohort contract and apply it to every Fig. 5 loader.

    ``LOCKED_GROUP_MICE`` is updated in place so callers that imported the
    mapping before configuration still observe the same authoritative cohort.
    """

    contract = load_manuscript_cohort(path)
    groups = _group_mice_from_contract(contract)
    ACTIVE_MANUSCRIPT_COHORT.clear()
    ACTIVE_MANUSCRIPT_COHORT.update(contract)
    LOCKED_GROUP_MICE.clear()
    LOCKED_GROUP_MICE.update(groups)
    return contract

FIGURE8_MANIFOLD_EXPECTED_MISSING = frozenset({("I", "B7", 4)})

TMAZE_EXPECTED_MISSING = frozenset(
    {
        ("C", "C7", 1),
        ("C", "C8", 1),
        ("I", "I3", 5),
        ("I", "I7", 1),
        ("I", "I7", 5),
        ("I", "I8", 3),
        ("I", "I9", 4),
        ("I", "I9", 5),
    }
)

FIGURE8_CONTRAST = "left_vs_right_correct"

TMAZE_MAIN_CONTRASTS = (
    "sample_left_vs_right",
    "choice_left_vs_right",
    "sample_vs_choice",
)

FIGURE8_PV_SECTION_MAP = {
    "Start arm": "start_arm",
    "Side arm": "side_arm",
    "Reward arm": "reward_arm",
    "End arm": "end_arm",
    "Reward location": "reward_location",
}

FIGURE8_MANIFOLD_SECTION_MAP = {
    "Bins 0-13": "start_arm",
    "Bins 14-27": "side_arm",
    "Bins 28-41": "reward_arm",
    "Bins 42-55": "end_arm",
}

TMAZE_SECTION_MAP = {
    "Stem": "stem",
    "Choice arm": "choice_arm",
}

SHARED_COVARIATE_COLUMNS = (
    "mean_speed_cm_s",
    "route_relative_lateral_deviation_cm",
    "mean_speed_z",
    "route_relative_lateral_deviation_z",
)

@dataclass(frozen=True)
class Fig5SourcePaths:
    """Filesystem locations for all locked Fig. 5 source artifacts."""

    figure8_pv_bin: Path
    figure8_pv_section: Path
    figure8_manifold_bin: Path
    figure8_manifold_section: Path
    figure8_manifold_qc: Path
    tmaze_pv_bin: Path
    tmaze_pv_section: Path
    tmaze_pv_qc: Path
    tmaze_manifold_bin: Path
    tmaze_manifold_section: Path
    tmaze_manifold_qc: Path
    shared_covariates: Path

    def as_dict(self) -> dict[str, Path]:
        """Return a defensive field-to-path mapping."""

        return {field.name: Path(getattr(self, field.name)) for field in fields(self)}

@dataclass(frozen=True)
class Fig5SourceBundle:
    """Normalized, covariate-attached Fig. 5 source tables and config QC."""

    figure8_pv_bin: pd.DataFrame
    figure8_pv_section: pd.DataFrame
    figure8_manifold_bin: pd.DataFrame
    figure8_manifold_section: pd.DataFrame
    tmaze_pv_bin: pd.DataFrame
    tmaze_pv_section: pd.DataFrame
    tmaze_manifold_bin: pd.DataFrame
    tmaze_manifold_section: pd.DataFrame
    config_qc: pd.DataFrame

    def source_tables(self, *, copy: bool = True) -> dict[str, pd.DataFrame]:
        """Return the eight normalized quantitative source tables."""

        names = (
            "figure8_pv_bin",
            "figure8_pv_section",
            "figure8_manifold_bin",
            "figure8_manifold_section",
            "tmaze_pv_bin",
            "tmaze_pv_section",
            "tmaze_manifold_bin",
            "tmaze_manifold_section",
        )
        return {
            name: getattr(self, name).copy() if copy else getattr(self, name)
            for name in names
        }

    def source_table(self, key: str, *, copy: bool = True) -> pd.DataFrame:
        """Return one normalized table by its generic source key."""

        tables = self.source_tables(copy=copy)
        if key not in tables:
            raise KeyError(f"unknown Fig. 5 source key {key!r}")
        return tables[key]

@dataclass(frozen=True)
class _SourceContract:
    key: str
    project: str
    level: str
    metric: str
    source_metric: str
    contrasts: tuple[str, ...]
    n_bin: int | None
    source_sections: tuple[str, ...]
    section_map: Mapping[str, str]
    expected_missing: frozenset[tuple[str, str, int]]

_SOURCE_CONTRACTS = {
    "figure8_pv_bin": _SourceContract(
        key="figure8_pv_bin",
        project="Figure-8",
        level="spatial_bin",
        metric="high_dimensional_pv_separation",
        source_metric="pv_separation_z",
        contrasts=(FIGURE8_CONTRAST,),
        n_bin=56,
        source_sections=(),
        section_map={},
        expected_missing=frozenset(),
    ),
    "figure8_pv_section": _SourceContract(
        key="figure8_pv_section",
        project="Figure-8",
        level="spatial_section",
        metric="high_dimensional_pv_separation",
        source_metric="pv_separation_z",
        contrasts=(FIGURE8_CONTRAST,),
        n_bin=None,
        source_sections=tuple(FIGURE8_PV_SECTION_MAP),
        section_map=FIGURE8_PV_SECTION_MAP,
        expected_missing=frozenset(),
    ),
    "figure8_manifold_bin": _SourceContract(
        key="figure8_manifold_bin",
        project="Figure-8",
        level="spatial_bin",
        metric="neural_manifold_separation",
        source_metric="shuffle_z",
        contrasts=(FIGURE8_CONTRAST,),
        n_bin=56,
        source_sections=(),
        section_map={},
        expected_missing=FIGURE8_MANIFOLD_EXPECTED_MISSING,
    ),
    "figure8_manifold_section": _SourceContract(
        key="figure8_manifold_section",
        project="Figure-8",
        level="spatial_section",
        metric="neural_manifold_separation",
        source_metric="shuffle_z",
        contrasts=(FIGURE8_CONTRAST,),
        n_bin=None,
        source_sections=tuple(FIGURE8_MANIFOLD_SECTION_MAP),
        section_map=FIGURE8_MANIFOLD_SECTION_MAP,
        expected_missing=FIGURE8_MANIFOLD_EXPECTED_MISSING,
    ),
    "tmaze_pv_bin": _SourceContract(
        key="tmaze_pv_bin",
        project="DNMP T-maze",
        level="spatial_bin",
        metric="high_dimensional_pv_separation",
        source_metric="spearman_separation_z",
        contrasts=TMAZE_MAIN_CONTRASTS,
        n_bin=29,
        source_sections=(),
        section_map={},
        expected_missing=TMAZE_EXPECTED_MISSING,
    ),
    "tmaze_pv_section": _SourceContract(
        key="tmaze_pv_section",
        project="DNMP T-maze",
        level="spatial_section",
        metric="high_dimensional_pv_separation",
        source_metric="spearman_separation_z",
        contrasts=TMAZE_MAIN_CONTRASTS,
        n_bin=None,
        source_sections=tuple(TMAZE_SECTION_MAP),
        section_map=TMAZE_SECTION_MAP,
        expected_missing=TMAZE_EXPECTED_MISSING,
    ),
    "tmaze_manifold_bin": _SourceContract(
        key="tmaze_manifold_bin",
        project="DNMP T-maze",
        level="spatial_bin",
        metric="neural_manifold_separation",
        source_metric="shuffle_z",
        contrasts=TMAZE_MAIN_CONTRASTS,
        n_bin=29,
        source_sections=(),
        section_map={},
        expected_missing=TMAZE_EXPECTED_MISSING,
    ),
    "tmaze_manifold_section": _SourceContract(
        key="tmaze_manifold_section",
        project="DNMP T-maze",
        level="spatial_section",
        metric="neural_manifold_separation",
        source_metric="shuffle_z",
        contrasts=TMAZE_MAIN_CONTRASTS,
        n_bin=None,
        source_sections=tuple(TMAZE_SECTION_MAP),
        section_map=TMAZE_SECTION_MAP,
        expected_missing=TMAZE_EXPECTED_MISSING,
    ),
}

_PANEL_ROWS = (
    ("A", "Figure-8", "task comparison schematic", None, None, None, (), False, False, "keep_static"),
    ("B", "Figure-8", "linearized trajectory schematic", None, None, None, (), False, False, "keep_static"),
    ("C", "Figure-8", "representative neural manifold", None, None, None, (), False, True, "keep_user_lock"),
    ("D", "Figure-8", "manifold section summaries", "figure8_manifold_section", "spatial_section", FIGURE8_CONTRAST, ("start_arm", "side_arm", "reward_arm", "end_arm"), True, False, "redraw_same_manifold_settings"),
    ("E", "Figure-8", "high-dimensional PV heatmaps", "figure8_pv_bin", "spatial_bin", FIGURE8_CONTRAST, (), True, False, "reanalyze_and_redraw"),
    ("F", "Figure-8", "high-dimensional PV section summaries", "figure8_pv_section", "spatial_section", FIGURE8_CONTRAST, ("start_arm", "side_arm", "reward_arm", "end_arm"), True, False, "reanalyze_and_redraw"),
    ("G", "DNMP T-maze", "contrast schematic", None, None, None, (), False, False, "keep_static"),
    ("H", "DNMP T-maze", "linearized trajectory schematic", None, None, None, (), False, False, "keep_static"),
    ("I", "DNMP T-maze", "representative neural manifold", None, None, None, (), False, False, "keep_representative"),
    ("J", "DNMP T-maze", "sample-route manifold heatmaps", "tmaze_manifold_bin", "spatial_bin", "sample_left_vs_right", (), True, False, "redraw_same_manifold_settings"),
    ("K", "DNMP T-maze", "choice-route manifold heatmaps", "tmaze_manifold_bin", "spatial_bin", "choice_left_vs_right", (), True, False, "redraw_same_manifold_settings"),
    ("L", "DNMP T-maze", "sample-choice manifold heatmaps", "tmaze_manifold_bin", "spatial_bin", "sample_vs_choice", (), True, False, "redraw_same_manifold_settings"),
    ("M", "DNMP T-maze", "sample-route manifold summaries", "tmaze_manifold_section", "spatial_section", "sample_left_vs_right", ("stem", "choice_arm"), True, False, "redraw_same_manifold_settings"),
    ("N", "DNMP T-maze", "choice-route manifold summaries", "tmaze_manifold_section", "spatial_section", "choice_left_vs_right", ("stem", "choice_arm"), True, False, "redraw_same_manifold_settings"),
    ("O", "DNMP T-maze", "sample-choice manifold summaries", "tmaze_manifold_section", "spatial_section", "sample_vs_choice", ("stem", "choice_arm"), True, False, "redraw_same_manifold_settings"),
    ("P", "DNMP T-maze", "sample-route high-dimensional PV summaries", "tmaze_pv_section", "spatial_section", "sample_left_vs_right", ("stem", "choice_arm"), True, False, "reanalyze_and_redraw"),
    ("Q", "DNMP T-maze", "choice-route high-dimensional PV summaries", "tmaze_pv_section", "spatial_section", "choice_left_vs_right", ("stem", "choice_arm"), True, False, "reanalyze_and_redraw"),
    ("R", "DNMP T-maze", "sample-choice high-dimensional PV summaries", "tmaze_pv_section", "spatial_section", "sample_vs_choice", ("stem", "choice_arm"), True, False, "reanalyze_and_redraw"),
)

def default_fig5_source_paths(
    repository_root: str | Path | None = None,
    *,
    tmaze_root: str | Path | None = None,
) -> Fig5SourcePaths:
    """Return the current locked source paths without reading or writing them."""

    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[2]
    root = Path(repository_root)
    troot = root if tmaze_root is None else Path(tmaze_root)
    f8_outputs = root / "notebook-8-maze" / "analysis_outputs"
    tm_outputs = troot / "notebook-8maze-migration"
    # Author-selected formal source: the historical D1-D5 pooled-covariance
    # snapshot.  Its dedicated QC file records the actual legacy algorithm;
    # never pair these tables with the later within-class cache QC.
    tm_manifold = (
        tm_outputs
        / "04_neural_manifold"
        / "analysis_outputs"
        / "manifold_quantification_d1_d5_exclude_I1"
        / "tables"
    )
    tm_manifold_qc = (
        root
        / "notebook-8-maze"
        / "config"
        / "tmaze_manifold_legacy_pooled_snapshot_qc.csv"
    )
    tm_pv = (
        tm_outputs
        / "05_pv_shuffle"
        / "analysis_outputs"
        / "pv_shuffle_control"
    )
    shared = (
        f8_outputs
        / "manuscript_d1_d5_cohort_v3_drop_A9_A11_C5_C6_20260804"
        / "fig3_behavior_reset_20260804"
        / "source_data"
        / "00_Shared_behavior_covariates_mouse_day.csv"
    )
    return Fig5SourcePaths(
        figure8_pv_bin=f8_outputs / "pv_shuffle_control" / "pv_bin_long_CL_CR_C-I.csv",
        figure8_pv_section=f8_outputs
        / "pv_shuffle_control"
        / "pv_region_long_CL_CR_C-I.csv",
        figure8_manifold_bin=f8_outputs
        / "manifold_quantification"
        / "manifold_spatial_bin_mouse_day_table_shuffle_z_left_vs_right_current_C-I.csv",
        figure8_manifold_section=f8_outputs
        / "manifold_quantification"
        / "manifold_spatial_section_mouse_day_table_shuffle_z_left_vs_right_current_C-I.csv",
        figure8_manifold_qc=f8_outputs
        / "manifold_quantification"
        / "manifold_quantification_cache_qc_C-I.csv",
        tmaze_pv_bin=tm_pv / "pv_shuffle_spatial_bin_mouse_day_table.csv",
        tmaze_pv_section=tm_pv / "pv_shuffle_spatial_section_mouse_day_table.csv",
        tmaze_pv_qc=tm_pv / "pv_shuffle_plot_cache_qc.csv",
        tmaze_manifold_bin=tm_manifold
        / "manifold_spatial_bin_mouse_day_table_d1_d5_exclude_I1.csv",
        tmaze_manifold_section=tm_manifold
        / "manifold_spatial_section_mouse_day_table_d1_d5_exclude_I1.csv",
        tmaze_manifold_qc=tm_manifold_qc,
        shared_covariates=shared,
    )

def fig5_panel_map() -> pd.DataFrame:
    """Return the authoritative accepted A--R panel mapping.

    Panel C remains the accepted representative visualization. Quantitative
    panels D, L and M are cohort-dependent and are therefore redraw targets in
    the v3 release, while retaining the accepted A--R arrangement.
    """

    columns = (
        "panel",
        "project",
        "display_role",
        "source_key",
        "level",
        "contrast",
        "sections",
        "quantitative",
        "literal_user_lock",
        "action",
    )
    frame = pd.DataFrame(_PANEL_ROWS, columns=columns)
    frame["sections"] = frame["sections"].map(tuple)
    return frame

def fig5_config_metadata(
    *, tmaze_manifold_covariance: str = "within_class_ledoit_wolf"
) -> pd.DataFrame:
    """Return the frozen metric/configuration contract for the four families."""

    if tmaze_manifold_covariance not in {
        "within_class_ledoit_wolf",
        "legacy_pooled",
    }:
        raise ValueError(
            "unsupported T-maze manifold covariance: "
            f"{tmaze_manifold_covariance!r}"
        )

    rows = [
        {
            "source_family": "figure8_pv",
            "project": "Figure-8",
            "metric": "high_dimensional_pv_separation",
            "source_metric": "pv_separation_z",
            "contrast": FIGURE8_CONTRAST,
            "days": "D1-D5",
            "n_bin": 56,
            "target_neuron_count": 50,
            "embedding_reducer": "",
            "embedding_dim": np.nan,
            "n_repeat": 100,
            "n_shuffle": 100,
            "dprime_covariance": "not_applicable",
            "shuffle_formula": "(mean_shuffle-observed_spearman)/sd_shuffle",
            "correct_trials_only": True,
            "minimum_trials_per_condition": 3,
            "trial_support_rule": "all_eligible_correct_CL_CR",
            "sample_vs_choice_sampling_rule": "not_applicable",
            "sample_vs_choice_shuffle_rule": "not_applicable",
            "population_space": (
                "original_high_dimensional_population_space_after_fixed_50_subsampling"
            ),
            "covariate_scope": (
                "mouse_day_broad_movement_summary_not_phase_specific_matching"
            ),
            "manifold_settings_frozen": False,
        },
        {
            "source_family": "figure8_manifold",
            "project": "Figure-8",
            "metric": "neural_manifold_separation",
            "source_metric": "shuffle_z",
            "contrast": FIGURE8_CONTRAST,
            "days": "D1-D5",
            "n_bin": 56,
            "target_neuron_count": 50,
            "embedding_reducer": "spectral",
            "embedding_dim": 15,
            "n_repeat": 100,
            "n_shuffle": 100,
            "dprime_covariance": "legacy_pooled",
            "shuffle_formula": "(observed_dprime-mean_shuffle)/sd_shuffle",
            "correct_trials_only": True,
            "minimum_trials_per_condition": 3,
            "trial_support_rule": "correct_CL_CR_with_valid_preceding_trial",
            "sample_vs_choice_sampling_rule": "not_applicable",
            "sample_vs_choice_shuffle_rule": "not_applicable",
            "population_space": "15_D_SpectralEmbedding",
            "covariate_scope": (
                "mouse_day_broad_movement_summary_not_phase_specific_matching"
            ),
            "manifold_settings_frozen": True,
        },
        {
            "source_family": "tmaze_pv",
            "project": "DNMP T-maze",
            "metric": "high_dimensional_pv_separation",
            "source_metric": "spearman_separation_z",
            "contrast": ";".join(TMAZE_MAIN_CONTRASTS),
            "days": "D1-D5",
            "n_bin": 29,
            "target_neuron_count": 50,
            "embedding_reducer": "",
            "embedding_dim": np.nan,
            "n_repeat": 100,
            "n_shuffle": 100,
            "dprime_covariance": "not_applicable_for_spearman",
            "shuffle_formula": "(mean_shuffle-observed_spearman)/sd_shuffle",
            "correct_trials_only": True,
            "minimum_trials_per_condition": 3,
            "trial_support_rule": "eligible_correct_trials_for_each_contrast",
            "sample_vs_choice_sampling_rule": (
                "route_stratified_balanced_undersampling"
            ),
            "sample_vs_choice_shuffle_rule": (
                "phase_labels_separately_within_left_and_right_route"
            ),
            "population_space": (
                "original_high_dimensional_population_space_after_fixed_50_subsampling"
            ),
            "covariate_scope": (
                "mouse_day_broad_movement_summary_not_phase_specific_matching"
            ),
            "manifold_settings_frozen": False,
        },
        {
            "source_family": "tmaze_manifold",
            "project": "DNMP T-maze",
            "metric": "neural_manifold_separation",
            "source_metric": "shuffle_z",
            "contrast": ";".join(TMAZE_MAIN_CONTRASTS),
            "days": "D1-D5",
            "n_bin": 29,
            "target_neuron_count": 50,
            "embedding_reducer": "spectral",
            "embedding_dim": 15,
            "n_repeat": 100,
            "n_shuffle": 100,
            "dprime_covariance": tmaze_manifold_covariance,
            "shuffle_formula": "(observed_dprime-mean_shuffle)/sd_shuffle",
            "correct_trials_only": True,
            "minimum_trials_per_condition": 3,
            "trial_support_rule": "eligible_correct_trials_for_each_contrast",
            "sample_vs_choice_sampling_rule": (
                "route_stratified_balanced_undersampling"
            ),
            "sample_vs_choice_shuffle_rule": (
                "phase_labels_separately_within_left_and_right_route"
            ),
            "population_space": "15_D_SpectralEmbedding",
            "covariate_scope": (
                "mouse_day_broad_movement_summary_not_phase_specific_matching"
            ),
            "manifold_settings_frozen": True,
        },
    ]
    return pd.DataFrame(rows)

def load_fig5_sources(paths: Fig5SourcePaths) -> Fig5SourceBundle:
    """Load, normalize, validate, and covariate-attach all Fig. 5 sources."""

    path_map = paths.as_dict()
    raw = {name: _read_csv(path, name) for name, path in path_map.items()}
    covariates = prepare_shared_covariates(raw["shared_covariates"])

    tables = {
        "figure8_pv_bin": _normalize_figure8_pv(
            raw["figure8_pv_bin"],
            _SOURCE_CONTRACTS["figure8_pv_bin"],
            path_map["figure8_pv_bin"],
        ),
        "figure8_pv_section": _normalize_figure8_pv(
            raw["figure8_pv_section"],
            _SOURCE_CONTRACTS["figure8_pv_section"],
            path_map["figure8_pv_section"],
        ),
        "figure8_manifold_bin": _normalize_manifold(
            raw["figure8_manifold_bin"],
            _SOURCE_CONTRACTS["figure8_manifold_bin"],
            path_map["figure8_manifold_bin"],
            source_contrast_map={"left_vs_right_current": FIGURE8_CONTRAST},
        ),
        "figure8_manifold_section": _normalize_manifold(
            raw["figure8_manifold_section"],
            _SOURCE_CONTRACTS["figure8_manifold_section"],
            path_map["figure8_manifold_section"],
            source_contrast_map={"left_vs_right_current": FIGURE8_CONTRAST},
        ),
        "tmaze_pv_bin": _normalize_tmaze_pv(
            raw["tmaze_pv_bin"],
            _SOURCE_CONTRACTS["tmaze_pv_bin"],
            path_map["tmaze_pv_bin"],
        ),
        "tmaze_pv_section": _normalize_tmaze_pv(
            raw["tmaze_pv_section"],
            _SOURCE_CONTRACTS["tmaze_pv_section"],
            path_map["tmaze_pv_section"],
        ),
        "tmaze_manifold_bin": _normalize_manifold(
            raw["tmaze_manifold_bin"],
            _SOURCE_CONTRACTS["tmaze_manifold_bin"],
            path_map["tmaze_manifold_bin"],
            source_contrast_map={name: name for name in TMAZE_MAIN_CONTRASTS},
        ),
        "tmaze_manifold_section": _normalize_manifold(
            raw["tmaze_manifold_section"],
            _SOURCE_CONTRACTS["tmaze_manifold_section"],
            path_map["tmaze_manifold_section"],
            source_contrast_map={name: name for name in TMAZE_MAIN_CONTRASTS},
        ),
    }

    for key, table in tables.items():
        _validate_source_table(table, _SOURCE_CONTRACTS[key])
        tables[key] = attach_shared_covariates(table, covariates)

    _validate_matching_support(
        tables["figure8_pv_bin"],
        tables["figure8_pv_section"],
        "Figure-8 PV",
    )
    _validate_matching_support(
        tables["figure8_manifold_bin"],
        tables["figure8_manifold_section"],
        "Figure-8 manifold",
    )
    _validate_matching_support(
        tables["tmaze_pv_bin"], tables["tmaze_pv_section"], "T-maze PV"
    )
    _validate_matching_support(
        tables["tmaze_manifold_bin"],
        tables["tmaze_manifold_section"],
        "T-maze manifold",
    )

    config_qc = _load_config_qc(raw)
    return Fig5SourceBundle(**tables, config_qc=config_qc)

def prepare_shared_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    """Filter and validate the active contract's exact D1--D5 mouse-day grids."""

    required = {"project", "group", "mouse", "day", *SHARED_COVARIATE_COLUMNS}
    _require_columns(covariates, required, "shared Fig. 5 covariates")
    table = covariates.copy()
    table = table.loc[
        table["project"].astype(str).isin(PROJECT_ORDER)
        & table["group"].astype(str).isin(GROUP_ORDER)
    ].copy()
    allowed = {
        (project, group, mouse)
        for project in PROJECT_ORDER
        for group in GROUP_ORDER
        for mouse in LOCKED_GROUP_MICE[project][group]
    }
    table = table.loc[
        [
            (str(project), str(group), str(mouse)) in allowed
            for project, group, mouse in table[["project", "group", "mouse"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    table["day"] = pd.to_numeric(table["day"], errors="raise").astype(int)
    table = table.loc[table["day"].isin(DAYS)].copy()
    _assert_unique(table, ["project", "group", "mouse", "day"], "shared covariates")

    expected = {
        (project, group, mouse, day)
        for project in PROJECT_ORDER
        for group in GROUP_ORDER
        for mouse in LOCKED_GROUP_MICE[project][group]
        for day in DAYS
    }
    observed = set(
        table[["project", "group", "mouse", "day"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected:
        raise ValueError(
            "shared covariates do not match the locked Fig. 5 grids; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for column in SHARED_COVARIATE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        if not np.isfinite(table[column].to_numpy(dtype=float)).all():
            raise ValueError(f"shared covariate {column!r} contains nonfinite values")

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
    keep = ["project", "group", "mouse", "day", *SHARED_COVARIATE_COLUMNS]
    return table[keep + optional].sort_values(keep[:4]).reset_index(drop=True)

def attach_shared_covariates(
    source: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach formal mouse-day covariates to a normalized Fig. 5 source."""

    keys = ["project", "group", "mouse", "day"]
    _require_columns(source, set(keys), "Fig. 5 source")
    prepared = prepare_shared_covariates(covariates)
    overlap = sorted(
        set(prepared.columns).intersection(source.columns).difference(keys)
    )
    if overlap:
        raise ValueError(f"Fig. 5 source already contains covariate columns: {overlap}")
    result = source.merge(prepared, on=keys, how="left", validate="many_to_one")
    if result[list(SHARED_COVARIATE_COLUMNS)].isna().any().any():
        raise ValueError("Fig. 5 covariate merge produced unmatched rows")
    return result

def build_fig5_support_qc(bundle: Fig5SourceBundle) -> pd.DataFrame:
    """Return nominal/effective support and unit-completeness rows."""

    rows: list[dict[str, Any]] = []
    for key, table in bundle.source_tables(copy=False).items():
        contract = _SOURCE_CONTRACTS[key]
        unit_column = "spatial_bin" if contract.level == "spatial_bin" else "section"
        unit_counts = table.groupby(
            ["group", "mouse", "day", "contrast"], observed=True
        )[unit_column].nunique()
        expected_unit_size = (
            int(contract.n_bin)
            if contract.level == "spatial_bin"
            else len(contract.source_sections)
        )
        for contrast in contract.contrasts:
            subset = table.loc[table["contrast"].eq(contrast)]
            for group in GROUP_ORDER:
                nominal_mice = set(LOCKED_GROUP_MICE[contract.project][group])
                for day in DAYS:
                    observed_mice = set(
                        subset.loc[
                            subset["group"].eq(group) & subset["day"].eq(day),
                            "mouse",
                        ].astype(str)
                    )
                    expected_mice = {
                        mouse
                        for mouse in nominal_mice
                        if (group, mouse, day) not in contract.expected_missing
                    }
                    selected_counts = unit_counts.loc[
                        [
                            index
                            for index in unit_counts.index
                            if index[0] == group
                            and index[2] == day
                            and index[3] == contrast
                        ]
                    ]
                    rows.append(
                        {
                            "source_key": key,
                            "project": contract.project,
                            "level": contract.level,
                            "metric": contract.metric,
                            "source_metric": contract.source_metric,
                            "contrast": contrast,
                            "group": group,
                            "day": day,
                            "day_c": float(day - DAY_CENTER),
                            "nominal_n": len(nominal_mice),
                            "expected_finite_n": len(expected_mice),
                            "finite_n": len(observed_mice),
                            "missing_ids": ";".join(sorted(nominal_mice - observed_mice)),
                            "expected_unit_size": expected_unit_size,
                            "min_unit_size": (
                                int(selected_counts.min()) if len(selected_counts) else 0
                            ),
                            "max_unit_size": (
                                int(selected_counts.max()) if len(selected_counts) else 0
                            ),
                            "status": (
                                "ok"
                                if observed_mice == expected_mice
                                and len(selected_counts)
                                and bool((selected_counts == expected_unit_size).all())
                                else "mismatch"
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["project", "metric", "source_key", "contrast", "group", "day"]
    ).reset_index(drop=True)

def build_fig5_section_summaries(bundle: Fig5SourceBundle) -> pd.DataFrame:
    """Return group/day mean, SD, SEM, and mouse n for every section source."""

    tables = [
        bundle.figure8_pv_section,
        bundle.figure8_manifold_section,
        bundle.tmaze_pv_section,
        bundle.tmaze_manifold_section,
    ]
    source = pd.concat(tables, ignore_index=True)
    unit = [
        "source_key",
        "project",
        "metric",
        "source_metric",
        "contrast",
        "source_contrast",
        "section",
        "source_section",
        "section_idx",
        "group",
        "day",
        "day_c",
    ]
    _assert_unique(
        source,
        [
            "source_key",
            "group",
            "mouse",
            "day",
            "contrast",
            "section",
        ],
        "combined Fig. 5 section sources",
    )
    summary = (
        source.groupby(unit, observed=True)["value"]
        .agg(mean="mean", sd="std", n_mouse="count")
        .reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_mouse"])
    return summary.sort_values(
        ["project", "metric", "contrast", "section_idx", "group", "day"]
    ).reset_index(drop=True)

def select_panel_source(bundle: Fig5SourceBundle, panel: str) -> pd.DataFrame:
    """Select a quantitative source using the accepted A--R panel contract.

    Static and representative-only panels do not have a normalized
    quantitative table and therefore raise ``ValueError``. Cohort-dependent
    quantitative panels D, L and M remain selectable and are regenerated from
    the stored metric tables without recomputing SpectralEmbedding.
    """

    panel = str(panel).upper()
    mapping = fig5_panel_map().set_index("panel", verify_integrity=True)
    if panel not in mapping.index:
        raise KeyError(f"unknown accepted Fig. 5 panel {panel!r}")
    row = mapping.loc[panel]
    source_key = row["source_key"]
    if pd.isna(source_key) or source_key is None:
        raise ValueError(f"Fig. 5{panel} has no normalized quantitative source")
    table = bundle.source_table(str(source_key))
    contrast = row["contrast"]
    if not pd.isna(contrast):
        table = table.loc[table["contrast"].eq(str(contrast))].copy()
    sections = tuple(row["sections"])
    if sections:
        table = table.loc[table["section"].isin(sections)].copy()
    return table.reset_index(drop=True)

def _normalize_figure8_pv(
    frame: pd.DataFrame,
    contract: _SourceContract,
    source_path: Path,
) -> pd.DataFrame:
    required = {"group", "mouse", "day", "metric", "value"}
    required.add("bin" if contract.level == "spatial_bin" else "section")
    _require_columns(frame, required, contract.key)
    table = frame.loc[
        frame["metric"].astype(str).eq(contract.source_metric)
        & frame["group"].astype(str).isin(GROUP_ORDER)
    ].copy()
    table["source_contrast"] = "CL_CR"
    table["contrast"] = FIGURE8_CONTRAST
    return _normalize_core(table, contract, source_path, value_column="value")

def _normalize_tmaze_pv(
    frame: pd.DataFrame,
    contract: _SourceContract,
    source_path: Path,
) -> pd.DataFrame:
    required = {
        "group",
        "mouse",
        "day",
        "contrast",
        contract.source_metric,
    }
    required.add("bin" if contract.level == "spatial_bin" else "section")
    _require_columns(frame, required, contract.key)
    table = frame.loc[
        frame["group"].astype(str).isin(GROUP_ORDER)
        & frame["contrast"].astype(str).isin(contract.contrasts)
    ].copy()
    table["source_contrast"] = table["contrast"].astype(str)
    table["contrast"] = table["source_contrast"]
    return _normalize_core(
        table, contract, source_path, value_column=contract.source_metric
    )

def _normalize_manifold(
    frame: pd.DataFrame,
    contract: _SourceContract,
    source_path: Path,
    *,
    source_contrast_map: Mapping[str, str],
) -> pd.DataFrame:
    required = {"group", "mouse", "day", "contrast", "shuffle_z"}
    required.add("bin" if contract.level == "spatial_bin" else "section")
    _require_columns(frame, required, contract.key)
    table = frame.loc[
        frame["group"].astype(str).isin(GROUP_ORDER)
        & frame["contrast"].astype(str).isin(source_contrast_map)
    ].copy()
    table["source_contrast"] = table["contrast"].astype(str)
    table["contrast"] = table["source_contrast"].map(source_contrast_map)
    return _normalize_core(table, contract, source_path, value_column="shuffle_z")

def _normalize_core(
    table: pd.DataFrame,
    contract: _SourceContract,
    source_path: Path,
    *,
    value_column: str,
) -> pd.DataFrame:
    allowed = {
        (group, mouse)
        for group in GROUP_ORDER
        for mouse in LOCKED_GROUP_MICE[contract.project][group]
    }
    table = table.loc[
        [
            (str(group), str(mouse)) in allowed
            for group, mouse in table[["group", "mouse"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    table["day"] = pd.to_numeric(table["day"], errors="raise").astype(int)
    table = table.loc[table["day"].isin(DAYS)].copy()
    common = pd.DataFrame(
        {
            "source_key": contract.key,
            "project": contract.project,
            "level": contract.level,
            "metric": contract.metric,
            "source_metric": contract.source_metric,
            "contrast": table["contrast"].astype(str),
            "source_contrast": table["source_contrast"].astype(str),
            "group": table["group"].astype(str),
            "mouse": table["mouse"].astype(str),
            "day": table["day"].astype(int),
            "day_c": table["day"].astype(float) - DAY_CENTER,
            "value": pd.to_numeric(table[value_column], errors="coerce"),
            "source_path": str(Path(source_path)),
        }
    )
    if contract.level == "spatial_bin":
        common["spatial_bin"] = pd.to_numeric(table["bin"], errors="raise").astype(int)
    else:
        common["source_section"] = table["section"].astype(str)
        common["section"] = common["source_section"].map(contract.section_map)
        if common["section"].isna().any():
            unknown = sorted(common.loc[common["section"].isna(), "source_section"].unique())
            raise ValueError(f"{contract.key} contains unknown sections: {unknown}")
        expected_index = {
            section: index for index, section in enumerate(contract.source_sections)
        }
        common["section_idx"] = common["source_section"].map(expected_index).astype(int)
        if "n_section_bin" in table:
            common["n_section_bin"] = pd.to_numeric(
                table["n_section_bin"], errors="raise"
            ).astype(int)
        else:
            common["n_section_bin"] = common["source_section"].map(
                _section_bin_counts(contract)
            ).astype(int)
    return common.reset_index(drop=True)

def _validate_source_table(table: pd.DataFrame, contract: _SourceContract) -> None:
    base = ["group", "mouse", "day", "contrast"]
    unit = "spatial_bin" if contract.level == "spatial_bin" else "section"
    _assert_unique(table, [*base, unit], contract.key)
    if table.empty:
        raise ValueError(f"{contract.key} is empty after locked filtering")
    if not np.isfinite(table["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{contract.key} contains nonfinite target values")
    if not table["day_c"].equals(table["day"].astype(float) - DAY_CENTER):
        raise ValueError(f"{contract.key} day_c was not rebuilt as day - 3")

    expected_support = _expected_support(contract)
    observed_support = set(
        table[base].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_support != expected_support:
        raise ValueError(
            f"{contract.key} support mismatch; "
            f"missing={sorted(expected_support - observed_support)}, "
            f"extra={sorted(observed_support - expected_support)}"
        )

    counts = table.groupby(base, observed=True)[unit].nunique()
    if contract.level == "spatial_bin":
        expected_values = set(range(int(contract.n_bin)))
        if not bool((counts == int(contract.n_bin)).all()):
            raise ValueError(f"{contract.key} has incomplete spatial-bin units")
        for _, values in table.groupby(base, observed=True)[unit]:
            if set(values.astype(int)) != expected_values:
                raise ValueError(f"{contract.key} has a noncanonical spatial-bin grid")
    else:
        expected_values = set(contract.section_map.values())
        if not bool((counts == len(expected_values)).all()):
            raise ValueError(f"{contract.key} has incomplete spatial-section units")
        for _, values in table.groupby(base, observed=True)[unit]:
            if set(values.astype(str)) != expected_values:
                raise ValueError(f"{contract.key} has a noncanonical section grid")
        expected_bins = _section_bin_counts(contract)
        observed_bins = (
            table[["source_section", "n_section_bin"]]
            .drop_duplicates()
            .set_index("source_section")["n_section_bin"]
            .to_dict()
        )
        if observed_bins != expected_bins:
            raise ValueError(
                f"{contract.key} section-bin counts mismatch; "
                f"expected={expected_bins}, observed={observed_bins}"
            )

def _expected_support(contract: _SourceContract) -> set[tuple[str, str, int, str]]:
    return {
        (group, mouse, day, contrast)
        for group in GROUP_ORDER
        for mouse in LOCKED_GROUP_MICE[contract.project][group]
        for day in DAYS
        if (group, mouse, day) not in contract.expected_missing
        for contrast in contract.contrasts
    }

def _section_bin_counts(contract: _SourceContract) -> dict[str, int]:
    if contract.key == "figure8_pv_section":
        return {
            "Start arm": 14,
            "Side arm": 14,
            "Reward arm": 14,
            "End arm": 14,
            "Reward location": 3,
        }
    if contract.key == "figure8_manifold_section":
        return {section: 14 for section in contract.source_sections}
    if contract.project == "DNMP T-maze":
        return {"Stem": 13, "Choice arm": 16}
    raise ValueError(f"no section-bin contract for {contract.key}")

def _validate_matching_support(
    bin_table: pd.DataFrame,
    section_table: pd.DataFrame,
    label: str,
) -> None:
    keys = ["group", "mouse", "day", "contrast"]
    bin_support = set(
        bin_table[keys].drop_duplicates().itertuples(index=False, name=None)
    )
    section_support = set(
        section_table[keys].drop_duplicates().itertuples(index=False, name=None)
    )
    if bin_support != section_support:
        raise ValueError(
            f"{label} bin/section support mismatch; "
            f"bin_only={sorted(bin_support - section_support)}, "
            f"section_only={sorted(section_support - bin_support)}"
        )

def _load_config_qc(raw: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    tmaze_manifold_covariance = _resolve_tmaze_manifold_covariance(
        raw["tmaze_manifold_qc"]
    )
    rows.append(
        _validate_config_qc_table(
            raw["figure8_manifold_qc"],
            project="Figure-8",
            source_family="figure8_manifold",
            expected={
                "reducer": "spectral",
                "n_repeat": 100,
                "n_shuffle": 100,
                "embedding_dim": 15,
                "target_neuron_count": 50,
                "dprime_covariance": "legacy_pooled",
            },
            require_params_ok=True,
        )
    )
    rows.append(
        _validate_config_qc_table(
            raw["tmaze_pv_qc"],
            project="DNMP T-maze",
            source_family="tmaze_pv",
            expected={
                "n_repeat": 100,
                "n_shuffle": 100,
                "target_neuron_count": 50,
            },
            require_params_ok=False,
        )
    )
    rows.append(
        _validate_config_qc_table(
            raw["tmaze_manifold_qc"],
            project="DNMP T-maze",
            source_family="tmaze_manifold",
            expected={
                "reducer": "spectral",
                "n_repeat": 100,
                "n_shuffle": 100,
                "embedding_dim": 15,
                "target_neuron_count": 50,
                "dprime_covariance": tmaze_manifold_covariance,
            },
            require_params_ok=False,
        )
    )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["project", "source_family", "group", "mouse"]
    ).reset_index(drop=True)

def _resolve_tmaze_manifold_covariance(frame: pd.DataFrame) -> str:
    _require_columns(
        frame,
        {"group", "mouse", "dprime_covariance"},
        "tmaze_manifold config QC",
    )
    locked = {
        (group, mouse)
        for group in GROUP_ORDER
        for mouse in LOCKED_GROUP_MICE["DNMP T-maze"][group]
    }
    selected = frame.loc[
        [
            (str(group), str(mouse)) in locked
            for group, mouse in frame[["group", "mouse"]].itertuples(
                index=False, name=None
            )
        ],
        "dprime_covariance",
    ].astype(str)
    observed = tuple(sorted(selected.unique()))
    allowed = {"within_class_ledoit_wolf", "legacy_pooled"}
    if len(observed) != 1 or observed[0] not in allowed:
        raise ValueError(
            "T-maze manifold QC must specify one supported covariance for the "
            f"locked cohort; observed={observed}"
        )
    return observed[0]

def _validate_config_qc_table(
    frame: pd.DataFrame,
    *,
    project: str,
    source_family: str,
    expected: Mapping[str, Any],
    require_params_ok: bool,
) -> pd.DataFrame:
    required = {"group", "mouse", *expected}
    if require_params_ok:
        required.add("params_ok")
    _require_columns(frame, required, f"{source_family} config QC")
    locked = {
        (group, mouse)
        for group in GROUP_ORDER
        for mouse in LOCKED_GROUP_MICE[project][group]
    }
    table = frame.loc[
        [
            (str(group), str(mouse)) in locked
            for group, mouse in frame[["group", "mouse"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    table["group"] = table["group"].astype(str)
    table["mouse"] = table["mouse"].astype(str)
    _assert_unique(table, ["group", "mouse"], f"{source_family} config QC")
    observed = set(table[["group", "mouse"]].itertuples(index=False, name=None))
    if observed != locked:
        raise ValueError(
            f"{source_family} config QC cohort mismatch; "
            f"missing={sorted(locked - observed)}, extra={sorted(observed - locked)}"
        )
    for column, expected_value in expected.items():
        if isinstance(expected_value, str):
            valid = table[column].astype(str).eq(expected_value)
        else:
            valid = pd.to_numeric(table[column], errors="coerce").eq(expected_value)
        if not bool(valid.all()):
            bad = table.loc[~valid, ["group", "mouse", column]].to_dict("records")
            raise ValueError(
                f"{source_family} violates frozen {column}={expected_value!r}: {bad}"
            )
    if require_params_ok and not bool(table["params_ok"].map(_as_bool).all()):
        raise ValueError(f"{source_family} config QC contains params_ok=False")
    table.insert(0, "source_family", source_family)
    table.insert(0, "project", project)
    return table

def _read_csv(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing Fig. 5 {label} source: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"unreadable Fig. 5 {label} source: {path}") from error

def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

def _assert_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicated = frame.duplicated(list(columns), keep=False)
    if bool(duplicated.any()):
        examples = frame.loc[duplicated, list(columns)].head(8).to_dict("records")
        raise ValueError(f"{label} contains duplicate units {list(columns)}: {examples}")

def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
