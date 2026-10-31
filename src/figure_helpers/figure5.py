"""Manuscript definitions collected from build_manuscript_fig5_state_separation.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from maze_analysis.fig5_state_separation import (  # noqa: E402
    DAYS,
    FIGURE8_CONTRAST,
    GROUP_ORDER,
    LOCKED_GROUP_MICE,
    TMAZE_MAIN_CONTRASTS,
    Fig5SourceBundle,
    Fig5SourcePaths,
    build_fig5_section_summaries,
    build_fig5_support_qc,
    configure_manuscript_cohort,
    default_fig5_source_paths,
    fig5_config_metadata,
    fig5_panel_map,
    load_fig5_sources,
    select_panel_source,
)

from maze_analysis.fig5_state_separation_plotting import (  # noqa: E402
    FIG5_LOCKED_PANELS,
    FIG5_REGENERATION_PANELS,
    plot_figure8_manifold_section_curves,
    plot_figure8_pv_heatmap,
    plot_figure8_section_curves,
    plot_tmaze_manifold_heatmap,
    plot_tmaze_section_curves,
    save_panel_bundle,
    validate_regeneration_bundle,
)

from maze_analysis.fig5_state_separation_stats import (  # noqa: E402
    CORE_ANNOTATION_ESTIMANDS,
    MODEL_FORMULA_CONTRACT,
    OUTCOME_TRANSFORM,
    PLANNED_ESTIMANDS,
    RAW_P_REPORTING_BASIS,
    STRATUM_COLUMNS,
    Fig5LongitudinalStatsResult,
    build_fig5_annotation_payloads,
    fit_fig5_longitudinal_models,
    validate_fig5_longitudinal_table,
)

GENERATED_PANELS = ("D", "E", "F", "J", "K", "L", "M", "N", "O", "P", "Q", "R")

PRIMARY_ROLE = "primary_regenerated_panel"

CONSISTENCY_AUDIT_ROLE = "consistency_audit_locked_panel_no_redraw"

SECONDARY_ROLE = "secondary_reward_location_not_plotted"

DISPLAY_SECTION = {
    "start_arm": "Start arm",
    "side_arm": "Side arm",
    "reward_arm": "Reward arm",
    "end_arm": "End arm",
    "reward_location": "Reward location",
    "stem": "Start arm",
    "choice_arm": "Choice arm",
}

@dataclass(frozen=True)
class Fig5ReleaseStatistics:
    """Role-decorated statistical artifacts for all 21 section strata."""

    model_table: pd.DataFrame
    contrasts: pd.DataFrame
    primary: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    support: pd.DataFrame
    annotation_audit: pd.DataFrame
    fit_result: Fig5LongitudinalStatsResult

def build_section_stratum_map() -> pd.DataFrame:
    """Return the exact 21 section-level model strata and reporting roles."""

    rows: list[dict[str, Any]] = []

    def add(
        project: str,
        metric: str,
        contrast: str,
        sections: Sequence[str],
        panel: str,
        role: str,
        *,
        plotted: bool,
    ) -> None:
        for section in sections:
            rows.append(
                {
                    "project": project,
                    "metric": metric,
                    "contrast": contrast,
                    "section": section,
                    "panel": panel,
                    "statistical_role": role,
                    "panel_generated": bool(panel in GENERATED_PANELS),
                    "section_plotted": bool(plotted),
                    "consistency_audit_only": role == CONSISTENCY_AUDIT_ROLE,
                    "secondary_estimand": role == SECONDARY_ROLE,
                }
            )

    add(
        "Figure-8",
        "neural_manifold_separation",
        FIGURE8_CONTRAST,
        ("start_arm", "side_arm", "reward_arm", "end_arm"),
        "D",
        PRIMARY_ROLE,
        plotted=True,
    )
    add(
        "Figure-8",
        "high_dimensional_pv_separation",
        FIGURE8_CONTRAST,
        ("start_arm", "side_arm", "reward_arm", "end_arm"),
        "F",
        PRIMARY_ROLE,
        plotted=True,
    )
    add(
        "Figure-8",
        "high_dimensional_pv_separation",
        FIGURE8_CONTRAST,
        ("reward_location",),
        "F",
        SECONDARY_ROLE,
        plotted=False,
    )
    for metric, panel_by_contrast in (
        (
            "neural_manifold_separation",
            {
                "sample_left_vs_right": ("M", PRIMARY_ROLE),
                "choice_left_vs_right": ("N", PRIMARY_ROLE),
                "sample_vs_choice": ("O", PRIMARY_ROLE),
            },
        ),
        (
            "high_dimensional_pv_separation",
            {
                "sample_left_vs_right": ("P", PRIMARY_ROLE),
                "choice_left_vs_right": ("Q", PRIMARY_ROLE),
                "sample_vs_choice": ("R", PRIMARY_ROLE),
            },
        ),
    ):
        for contrast in TMAZE_MAIN_CONTRASTS:
            panel, role = panel_by_contrast[contrast]
            add(
                "DNMP T-maze",
                metric,
                contrast,
                ("stem", "choice_arm"),
                panel,
                role,
                plotted=role == PRIMARY_ROLE,
            )

    table = pd.DataFrame(rows)
    if len(table) != 21 or table.duplicated(list(STRATUM_COLUMNS)).any():
        raise AssertionError("Fig. 5 section-stratum contract must contain 21 unique rows")
    if bool(table["consistency_audit_only"].any()):
        raise AssertionError("v3 regenerates every quantitative section panel")
    return table.sort_values([*STRATUM_COLUMNS]).reset_index(drop=True)

def build_section_model_table(bundle: Fig5SourceBundle) -> pd.DataFrame:
    """Combine the four normalized section sources into 21 model strata."""

    source = pd.concat(
        [
            bundle.figure8_manifold_section,
            bundle.figure8_pv_section,
            bundle.tmaze_manifold_section,
            bundle.tmaze_pv_section,
        ],
        ignore_index=True,
    )
    required = [
        *STRATUM_COLUMNS,
        "group",
        "mouse",
        "day",
        "value",
        "mean_speed_z",
        "route_relative_lateral_deviation_z",
    ]
    missing = sorted(set(required).difference(source.columns))
    if missing:
        raise ValueError(f"combined Fig. 5 section sources are missing columns: {missing}")
    model_table = validate_fig5_longitudinal_table(source.loc[:, required])
    expected = set(
        build_section_stratum_map()[list(STRATUM_COLUMNS)].itertuples(
            index=False, name=None
        )
    )
    observed = set(
        model_table[list(STRATUM_COLUMNS)]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(
            "Fig. 5 model strata mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return model_table

def _attach_stratum_roles(table: pd.DataFrame) -> pd.DataFrame:
    """Attach one declared panel/reporting role to every section stratum."""

    role_map = build_section_stratum_map()
    result = table.merge(
        role_map,
        on=list(STRATUM_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    if result["statistical_role"].isna().any():
        missing = result.loc[
            result["statistical_role"].isna(), list(STRATUM_COLUMNS)
        ].drop_duplicates()
        raise ValueError(
            "statistical output contains unmapped Fig. 5 strata: "
            f"{missing.to_dict('records')}"
        )
    return result

def fit_all_section_models(
    model_table: pd.DataFrame,
    *,
    maxiter: int = 1000,
) -> Fig5ReleaseStatistics:
    """Fit all 21 adjusted/unadjusted models and decorate release roles."""

    if int(maxiter) <= 0:
        raise ValueError("maxiter must be positive")
    fitted = fit_fig5_longitudinal_models(
        model_table,
        include_bh_audit=True,
        maxiter=int(maxiter),
    )
    contrasts = _attach_stratum_roles(fitted.contrasts)
    coefficients = _attach_stratum_roles(fitted.coefficients)
    diagnostics = _attach_stratum_roles(fitted.diagnostics)
    support = _attach_stratum_roles(fitted.support)

    contrasts["adjusted_model"] = contrasts["model_role"].eq("adjusted")
    contrasts["manuscript_facing"] = (
        contrasts["adjusted_model"]
        & contrasts["statistical_role"].eq(PRIMARY_ROLE)
    )
    contrasts["reporting_role"] = np.select(
        [
            contrasts["statistical_role"].eq(CONSISTENCY_AUDIT_ROLE),
            contrasts["statistical_role"].eq(SECONDARY_ROLE),
            contrasts["manuscript_facing"],
        ],
        [
            "adjusted/unadjusted consistency audit for locked image; no redraw",
            "secondary reward-location estimand; not plotted in Fig. 5F",
            "primary adjusted LMM for regenerated panel",
        ],
        default="shared-complete-case unadjusted sensitivity",
    )
    contrasts["primary_multiplicity_adjustment"] = "none"
    contrasts["bh_role"] = "audit_only_not_manuscript_facing"
    primary = contrasts.loc[contrasts["manuscript_facing"]].copy()
    annotation_audit = build_annotation_audit(contrasts)
    return Fig5ReleaseStatistics(
        model_table=model_table.copy(),
        contrasts=contrasts,
        primary=primary,
        coefficients=coefficients,
        diagnostics=diagnostics,
        support=support,
        annotation_audit=annotation_audit,
        fit_result=fitted,
    )

def build_annotation_audit(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Return one raw-P star audit row for each of the 21 strata."""

    selected = contrasts.loc[
        contrasts["model_role"].eq("adjusted")
        & contrasts["estimand_contrast"].isin(CORE_ANNOTATION_ESTIMANDS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for key, subset in selected.groupby(list(STRATUM_COLUMNS), observed=True, sort=True):
        indexed = subset.set_index("estimand_contrast", verify_integrity=True)
        missing = sorted(set(CORE_ANNOTATION_ESTIMANDS).difference(indexed.index))
        if missing:
            raise ValueError(f"annotation audit stratum {key} is missing {missing}")
        first = indexed.iloc[0]
        row = dict(zip(STRATUM_COLUMNS, key, strict=True))
        row.update(
            {
                "panel": first["panel"],
                "statistical_role": first["statistical_role"],
                "panel_generated": bool(first["panel_generated"]),
                "section_plotted": bool(first["section_plotted"]),
                "reported_probability": "P",
                "primary_inference_basis": RAW_P_REPORTING_BASIS,
            }
        )
        for label, estimand in (
            ("control_slope", "reference_group_day_slope"),
            ("inhibition_slope", "comparison_group_day_slope"),
            ("group_by_day", "group_by_day_interaction"),
            ("centered_I_minus_C", "centered_overall_group_effect"),
        ):
            effect = indexed.loc[estimand]
            row[f"{label}_beta"] = effect["beta"]
            row[f"{label}_ci95_low"] = effect["ci95_low"]
            row[f"{label}_ci95_high"] = effect["ci95_high"]
            row[f"{label}_raw_P"] = effect["p_value"]
            row[f"{label}_panel_symbol"] = effect["panel_symbol"]
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 21:
        raise AssertionError("Fig. 5 annotation audit must contain 21 strata")
    return result

def _display_sections(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if "section" in result:
        result["section"] = result["section"].map(DISPLAY_SECTION)
        if result["section"].isna().any():
            raise ValueError("Fig. 5 source contains an unmapped display section")
    return result

def _panel_summary(
    summaries: pd.DataFrame,
    *,
    source_key: str,
    contrast: str,
    sections: Sequence[str],
) -> pd.DataFrame:
    table = summaries.loc[
        summaries["source_key"].eq(source_key)
        & summaries["contrast"].eq(contrast)
        & summaries["section"].isin(tuple(sections))
    ].copy()
    if table.empty:
        raise ValueError(
            f"missing Fig. 5 summary for {source_key}/{contrast}/{tuple(sections)}"
        )
    return _display_sections(table)

def _plot_annotations(
    contrasts: pd.DataFrame,
    observation_table: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Return plotting-safe raw-P payloads while retaining full audit elsewhere."""

    payloads = build_fig5_annotation_payloads(contrasts, model_role="adjusted")
    keys = observation_table[list(STRATUM_COLUMNS)].drop_duplicates()
    output: dict[str, dict[str, Any]] = {}
    for key in keys.itertuples(index=False, name=None):
        payload = payloads[tuple(str(value) for value in key)]
        display_section = DISPLAY_SECTION[str(key[3])]
        output[display_section] = {
            "overall": payload["overall"],
            "slopes": dict(payload["slopes"]),
            "interaction": payload["interaction"],
        }
    return output
