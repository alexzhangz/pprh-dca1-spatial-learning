"""Manuscript definitions collected from build_manuscript_fig4_stability.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from maze_analysis.fig4_stability import (  # noqa: E402
    ADJACENT_DAY_PAIRS,
    ANALYSIS_CONFIGS,
    ANCHOR_DAY_PAIRS,
    DAYS,
    GROUP_ORDER,
    LOCKED_GROUP_MICE,
    SPLIT_STRATEGY,
    HeatmapPairData,
    attach_interday_pair_covariates,
    attach_within_day_covariates,
    build_interday_pair_covariates,
    build_interday_pv_bin_source,
    build_interday_pv_mouse_pair_source,
    build_interday_single_neuron_source,
    build_within_pv_bin_source,
    build_within_pv_mouse_day_source,
    build_within_single_neuron_source,
    configure_manuscript_cohort,
    load_locked_subject_caches,
    load_registered_left_maps,
    prepare_adjacent_left_heatmaps,
    prepare_shared_within_day_covariates,
    prepare_within_left_heatmaps,
)

from maze_analysis.fig4_stability_plotting import (  # noqa: E402
    FIG4_ADJACENT_PAIR_ORDER,
    plot_correlation_curve_panel,
    plot_fig4d_activity_heatmap,
    plot_fig4h_adjacent_activity_maps,
    plot_spatial_bin_heatmap,
    save_panel_bundle,
)

from maze_analysis.manuscript_lmm import (  # noqa: E402
    LongitudinalLMMResult,
    apply_bh_correction,
    fit_adjusted_unadjusted_lmm,
)

from maze_analysis.manuscript_heatmap_style import (  # noqa: E402
    HEATMAP_STYLE_VERSION,
    PV_HEATMAP_CMAP,
    heatmap_panel_spec,
)

COVARIATES = ("mean_speed_z", "route_relative_lateral_deviation_z")

CONDITION_DISPLAY = {"CL": "Left", "CR": "Right"}

PLOT_LIMITS = {
    "E": (
        heatmap_panel_spec("Fig4E", expected_family="PV").vmin,
        heatmap_panel_spec("Fig4E", expected_family="PV").vmax,
    ),
    "I": (
        heatmap_panel_spec("Fig4I", expected_family="PV").vmin,
        heatmap_panel_spec("Fig4I", expected_family="PV").vmax,
    ),
    "F": (0.30, 0.68),
    # PV panels show every mouse as a scatter point.  The v3 C5/I7 cohort
    # extends beyond the legacy display windows, so keep all observations
    # inside the axes rather than clipping influential mice at the bounds.
    "G": (0.30, 0.90),
    "J": (0.05, 0.65),
    "K": (0.00, 0.78),
    "L": (0.04, 0.31),
    "M": (-0.05, 0.72),
}

BH_FAMILY_COLUMNS = (
    "analysis",
    "metric",
    "model_role",
    "condition",
    "estimand_family",
)

PRIMARY_REPORTING_BASIS = "planned contrasts; unadjusted two-sided P values"

PANEL_ANNOTATION_BASIS = {
    "reference_group_day_slope": PRIMARY_REPORTING_BASIS,
    "comparison_group_day_slope": PRIMARY_REPORTING_BASIS,
    "group_by_day_interaction": PRIMARY_REPORTING_BASIS,
    "centered_overall_group_effect": PRIMARY_REPORTING_BASIS,
}

ENDPOINT_REPORTING_BASIS = PRIMARY_REPORTING_BASIS

BH_AUDIT_ROLE = "condition-specific BH-FDR sensitivity/audit only; not manuscript-facing"

SINGLE_NEURON_CLUSTER_COL = "neuron_unit"

SINGLE_NEURON_PSEUDOREPLICATION_WARNING = (
    "Author-requested primary single-neuron model omits mouse and mouse-by-time "
    "random effects; neurons from the same mouse are therefore treated as "
    "independent between-neuron sampling units for group inference."
)

POSTHOC_COHORT_WARNING = (
    "The A9/A11 whole-mouse exclusions were selected after inspection of "
    "statistical outcomes and are applied task-wide under the manuscript v3 contract."
)

def load_sources(
    cache_root: Path,
    aligned_root: Path,
    covariate_path: Path,
) -> dict[str, Any]:
    covariates_raw = pd.read_csv(covariate_path)
    covariates = prepare_shared_within_day_covariates(covariates_raw)
    records14 = load_locked_subject_caches(cache_root, "14")
    records16 = load_locked_subject_caches(cache_root, "16")
    records17 = load_locked_subject_caches(cache_root, "17")

    within_pv_bin = build_within_pv_bin_source(records14)
    within_pv = attach_within_day_covariates(
        build_within_pv_mouse_day_source(within_pv_bin), covariates
    )
    within_single, within_qc = build_within_single_neuron_source(records16)
    within_single = attach_within_day_covariates(within_single, covariates)

    pair_covariates = build_interday_pair_covariates(covariates)
    inter_pv_bin = build_interday_pv_bin_source(records17)
    inter_pv = attach_interday_pair_covariates(
        build_interday_pv_mouse_pair_source(inter_pv_bin), pair_covariates
    )
    inter_single = attach_interday_pair_covariates(
        build_interday_single_neuron_source(records17), pair_covariates
    )
    inter_pv["time_index"] = inter_pv["pair_index"] + 1
    inter_single["time_index"] = inter_single["pair_index"] + 1
    inter_pv_bin["time_index"] = inter_pv_bin["pair_index"] + 1
    inter_pv_bin["day_pair_display"] = inter_pv_bin.apply(
        lambda row: f"D{int(row.day_a)}\N{EN DASH}D{int(row.day_b)}", axis=1
    )

    within_heatmaps = prepare_within_left_heatmaps(records16)
    registered_maps = load_registered_left_maps(records17, aligned_root)
    adjacent_heatmaps = prepare_adjacent_left_heatmaps(registered_maps)
    return {
        "records14": records14,
        "records16": records16,
        "records17": records17,
        "covariates": covariates,
        "pair_covariates": pair_covariates,
        "within_pv_bin": within_pv_bin,
        "within_pv": within_pv,
        "within_single": within_single,
        "within_qc": within_qc,
        "inter_pv_bin": inter_pv_bin,
        "inter_pv": inter_pv,
        "inter_single": inter_single,
        "within_heatmaps": within_heatmaps,
        "adjacent_heatmaps": adjacent_heatmaps,
    }

def descriptive_summary(
    table: pd.DataFrame,
    *,
    time_col: str,
    sampling_unit: str,
) -> pd.DataFrame:
    result = (
        table.groupby(["group", "condition", time_col], observed=True)["raw_r"]
        .agg(mean="mean", sem="sem", n_finite="count")
        .reset_index()
    )
    result["sampling_unit"] = sampling_unit
    return result

def build_descriptive_tables(sources: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    tables = {
        "within_single": descriptive_summary(
            sources["within_single"], time_col="day", sampling_unit="neuron"
        ),
        "within_pv": descriptive_summary(
            sources["within_pv"], time_col="day", sampling_unit="mouse"
        ),
    }
    for mode in ("adjacent", "anchor"):
        tables[f"{mode}_single"] = descriptive_summary(
            sources["inter_single"].loc[
                sources["inter_single"]["pair_mode"].eq(mode)
            ],
            time_col="time_index",
            sampling_unit="neuron",
        )
        tables[f"{mode}_pv"] = descriptive_summary(
            sources["inter_pv"].loc[sources["inter_pv"]["pair_mode"].eq(mode)],
            time_col="time_index",
            sampling_unit="mouse",
        )
    return tables

def _estimand_family(contrast: str) -> str:
    if contrast in {"reference_group_day_slope", "comparison_group_day_slope"}:
        return "within_group_time_slopes"
    if contrast == "group_by_day_interaction":
        return "group_by_time_interactions"
    if contrast == "centered_overall_group_effect":
        return "centered_overall_group_effects"
    if contrast.startswith("between_group_"):
        return "endpoint_group_contrasts_secondary"
    raise ValueError(f"unrecognized contrast {contrast!r}")

def apply_fig4_bh_correction(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Compute a sensitivity-only BH audit without changing raw model results.

    CL and CR are distinct audit families. The resulting q values must not
    drive panels, primary tables, legends, or manuscript prose: all planned
    Fig. 4 contrasts are reported with their unadjusted two-sided P values.
    """

    required = {
        "analysis",
        "metric",
        "model_role",
        "condition",
        "contrast",
        "p_value",
    }
    missing = sorted(required.difference(contrasts.columns))
    if missing:
        raise ValueError(f"Fig. 4 contrast table is missing columns: {missing}")
    result = contrasts.copy()
    result["estimand_family"] = result["contrast"].map(_estimand_family)
    if result["estimand_family"].isna().any():
        raise ValueError("all Fig. 4 contrasts must map to an estimand family")
    result["correction_family"] = result.apply(
        lambda row: "|".join(
            [
                "Figure8",
                "Fig4",
                str(row.analysis),
                str(row.metric),
                str(row.model_role),
                str(row.condition),
                str(row.estimand_family),
            ]
        ),
        axis=1,
    )
    # Permit safe re-correction of an existing release from its unchanged raw P
    # values.  The previous q/significance columns never enter the calculation.
    result = result.drop(
        columns=[
            "q_value",
            "bh_q_audit_only",
            "significant",
            "bh_significant_audit_only",
            "correction_family_size",
        ],
        errors="ignore",
    )
    result = apply_bh_correction(
        result,
        p_col="p_value",
        family_cols=BH_FAMILY_COLUMNS,
    )
    result = result.rename(
        columns={
            "q_value": "bh_q_audit_only",
            "significant": "bh_significant_audit_only",
        }
    )
    family_sizes = (
        result.groupby("correction_family", observed=True)
        .size()
        .rename("correction_family_size")
    )
    result = result.join(family_sizes, on="correction_family")
    result["bh_adjustment_role"] = BH_AUDIT_ROLE
    result["reported_probability"] = "P"
    result["reported_probability_value"] = pd.to_numeric(
        result["p_value"], errors="coerce"
    )
    result["significant"] = result["reported_probability_value"] < 0.05
    result["primary_inference_basis"] = PRIMARY_REPORTING_BASIS

    expected_sizes = {
        "within_group_time_slopes": 2,
        "group_by_time_interactions": 1,
        "centered_overall_group_effects": 1,
        "endpoint_group_contrasts_secondary": 2,
    }
    observed_sizes = (
        result[["estimand_family", "correction_family_size"]]
        .drop_duplicates()
        .groupby("estimand_family", observed=True)["correction_family_size"]
        .agg(lambda values: tuple(sorted(set(int(value) for value in values))))
        .to_dict()
    )
    expected_observed = {
        family: (size,) for family, size in expected_sizes.items()
    }
    if observed_sizes != expected_observed:
        raise AssertionError(
            "unexpected condition-specific Fig. 4 BH family sizes; "
            f"expected={expected_observed}, observed={observed_sizes}"
        )
    return result

def _fit_posthoc_diagnostics(result: LongitudinalLMMResult) -> dict[str, Any]:
    fit = result.fit_result
    output = dict(result.diagnostics)
    if fit is None:
        output["posthoc_fixed_covariance_positive_definite"] = False
        output["posthoc_fixed_covariance_min_eigenvalue"] = np.nan
        return output
    names = list(fit.fe_params.index)
    covariance = fit.cov_params().loc[names, names].to_numpy(dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    output["posthoc_fixed_covariance_positive_definite"] = bool(
        eigenvalues.size
        and np.isfinite(eigenvalues).all()
        and float(eigenvalues.min()) > 0
    )
    output["posthoc_fixed_covariance_min_eigenvalue"] = (
        float(eigenvalues.min()) if eigenvalues.size else np.nan
    )
    output["posthoc_design_full_rank"] = bool(
        np.linalg.matrix_rank(np.asarray(fit.model.exog, dtype=float))
        == np.asarray(fit.model.exog).shape[1]
    )
    return output

def prepare_globally_unique_neuron_units(table: pd.DataFrame) -> pd.DataFrame:
    """Create the sole random-intercept level for requested neuron-only models."""

    required = {"mouse", "neuron"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"single-neuron source is missing columns: {missing}")
    result = table.loc[table["mouse"].notna() & table["neuron"].notna()].copy()
    neuron_numeric = pd.to_numeric(result["neuron"], errors="raise")
    if not np.isfinite(neuron_numeric.to_numpy(dtype=float)).all():
        raise ValueError("single-neuron source contains nonfinite neuron identifiers")
    neuron_label = neuron_numeric.map(
        lambda value: str(int(value)) if float(value).is_integer() else str(float(value))
    )
    result[SINGLE_NEURON_CLUSTER_COL] = (
        result["mouse"].astype(str) + "::" + neuron_label
    )
    identity = result[[SINGLE_NEURON_CLUSTER_COL, "mouse", "neuron"]].drop_duplicates()
    if identity[SINGLE_NEURON_CLUSTER_COL].duplicated(keep=False).any():
        raise ValueError("globally unique neuron identifier maps to multiple source neurons")
    return result

def _complete_case_support(
    table: pd.DataFrame,
    *,
    time_col: str,
    cluster_col: str,
) -> dict[str, Any]:
    """Summarize the exact adjusted-model complete-case support."""

    required = [
        "fisher_z",
        time_col,
        "group",
        "mouse",
        cluster_col,
        *COVARIATES,
    ]
    complete = table.dropna(subset=required).copy()
    finite = np.ones(len(complete), dtype=bool)
    for column in ("fisher_z", time_col, *COVARIATES):
        finite &= np.isfinite(
            pd.to_numeric(complete[column], errors="coerce").to_numpy(dtype=float)
        )
    complete = complete.loc[
        finite & complete["group"].astype(str).isin(GROUP_ORDER)
    ].copy()
    mice = {
        group: tuple(
            sorted(complete.loc[complete["group"].eq(group), "mouse"].astype(str).unique())
        )
        for group in GROUP_ORDER
    }
    clusters = {
        group: int(
            complete.loc[complete["group"].eq(group), cluster_col].astype(str).nunique()
        )
        for group in GROUP_ORDER
    }
    return {
        "nobs": int(len(complete)),
        "mice": mice,
        "n_mice": {group: len(mice[group]) for group in GROUP_ORDER},
        "n_clusters": clusters,
    }

def fit_requested_single_neuron_models(
    table: pd.DataFrame,
    *,
    time_col: str,
    first_time: float,
    last_time: float,
    maxiter: int,
) -> tuple[dict[str, LongitudinalLMMResult], dict[str, Any]]:
    """Fit adjusted/unadjusted models with only ``mouse::neuron`` clustering."""

    model_table = prepare_globally_unique_neuron_units(table)
    support = _complete_case_support(
        model_table,
        time_col=time_col,
        cluster_col=SINGLE_NEURON_CLUSTER_COL,
    )
    fits = fit_adjusted_unadjusted_lmm(
        model_table,
        continuous_covariates=COVARIATES,
        value_col="fisher_z",
        time_col=time_col,
        group_col="group",
        mouse_col=SINGLE_NEURON_CLUSTER_COL,
        reference_group="C",
        comparison_group="I",
        sampling_unit_cols=(SINGLE_NEURON_CLUSTER_COL, time_col),
        first_time=first_time,
        last_time=last_time,
        maxiter=int(maxiter),
    )
    return fits, support

def fit_all_models(
    sources: Mapping[str, Any],
    *,
    maxiter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = [
        {
            "analysis": "within",
            "metric": "single_neuron",
            "table": sources["within_single"],
            "time_col": "day",
            "first": 1,
            "last": 5,
        },
        {
            "analysis": "within",
            "metric": "pv",
            "table": sources["within_pv"],
            "time_col": "day",
            "first": 1,
            "last": 5,
        },
    ]
    for analysis in ("adjacent", "anchor"):
        for metric, source_name in (("single_neuron", "inter_single"), ("pv", "inter_pv")):
            jobs.append(
                {
                    "analysis": analysis,
                    "metric": metric,
                    "table": sources[source_name].loc[
                        sources[source_name]["pair_mode"].eq(analysis)
                    ].copy(),
                    "time_col": "time_index",
                    "first": 1,
                    "last": 4,
                }
            )

    contrast_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for job in jobs:
        for condition in ("CL", "CR"):
            table = job["table"].loc[job["table"]["condition"].eq(condition)].copy()
            label = f"{job['analysis']}/{job['metric']}/{condition}"
            print(f"Fitting {label} ...", flush=True)
            common_kwargs = {
                "continuous_covariates": COVARIATES,
                "value_col": "fisher_z",
                "time_col": job["time_col"],
                "group_col": "group",
                "reference_group": "C",
                "comparison_group": "I",
                "first_time": job["first"],
                "last_time": job["last"],
                "maxiter": int(maxiter),
            }
            if job["metric"] == "single_neuron":
                fits, support = fit_requested_single_neuron_models(
                    table,
                    time_col=job["time_col"],
                    first_time=job["first"],
                    last_time=job["last"],
                    maxiter=int(maxiter),
                )
                cluster_definition = "globally unique mouse::neuron"
                pseudoreplication_warning = SINGLE_NEURON_PSEUDOREPLICATION_WARNING
            else:
                model_table = table
                support = _complete_case_support(
                    model_table,
                    time_col=job["time_col"],
                    cluster_col="mouse",
                )
                fits = fit_adjusted_unadjusted_lmm(
                    model_table,
                    mouse_col="mouse",
                    **common_kwargs,
                )
                cluster_definition = "mouse"
                pseudoreplication_warning = None
            for role, result in fits.items():
                fixed_terms = (
                    "time_c*group + mean_speed_z + "
                    "route_relative_lateral_deviation_z"
                    if role == "adjusted"
                    else "time_c*group"
                )
                random_term = (
                    "(1|mouse::neuron)"
                    if job["metric"] == "single_neuron"
                    else "(1|mouse)"
                )
                model_formula_with_random_effect = (
                    f"fisher_z ~ {fixed_terms} + {random_term}"
                )
                diagnostic = _fit_posthoc_diagnostics(result)
                n_model_clusters = int(diagnostic["n_mice"])
                diagnostic.update(
                    {
                        "model": (
                            "MixedLM random intercept by globally unique mouse::neuron"
                            if job["metric"] == "single_neuron"
                            else "MixedLM random intercept by mouse"
                        ),
                        "model_formula_with_random_effect": model_formula_with_random_effect,
                        "analysis": job["analysis"],
                        "metric": job["metric"],
                        "condition": condition,
                        "model_role": role,
                        "outcome_scale": "Fisher z of correlation",
                        "plotted_scale": "raw correlation r",
                        "model_cluster_definition": cluster_definition,
                        "n_model_clusters": n_model_clusters,
                        "n_biological_mice": int(sum(support["n_mice"].values())),
                        "n_biological_mice_C": int(support["n_mice"]["C"]),
                        "n_biological_mice_I": int(support["n_mice"]["I"]),
                        "biological_mice": support["mice"],
                        "n_clusters_C": int(support["n_clusters"]["C"]),
                        "n_clusters_I": int(support["n_clusters"]["I"]),
                        "mouse_random_effect_included": job["metric"] == "pv",
                        "mouse_by_time_random_effect_included": False,
                        "pseudoreplication_warning": pseudoreplication_warning,
                        "posthoc_cohort_warning": POSTHOC_COHORT_WARNING,
                    }
                )
                # The generic fitter names its clustering level ``n_mice``.
                # Replace that public diagnostic field with the biological
                # mouse count and retain the actual cluster count explicitly.
                diagnostic["n_mice"] = int(sum(support["n_mice"].values()))
                diagnostic["n_mice_reference"] = int(support["n_mice"]["C"])
                diagnostic["n_mice_comparison"] = int(support["n_mice"]["I"])
                diagnostic["mice"] = tuple(
                    mouse for group in GROUP_ORDER for mouse in support["mice"][group]
                )
                diagnostics.append(diagnostic)
                if diagnostic.get("model_status") != "ok":
                    raise RuntimeError(f"primary model failed: {label}/{role}")
                if not diagnostic.get("posthoc_fixed_covariance_positive_definite"):
                    raise RuntimeError(f"non-positive-definite fixed covariance: {label}/{role}")

                contrasts = result.contrasts.copy()
                contrasts.insert(0, "model_role", role)
                contrasts.insert(0, "condition", condition)
                contrasts.insert(0, "metric", job["metric"])
                contrasts.insert(0, "analysis", job["analysis"])
                contrasts["n_model_clusters"] = contrasts["n_mice"].astype(int)
                contrasts["n_mice"] = int(sum(support["n_mice"].values()))
                contrasts["n_mice_C"] = int(support["n_mice"]["C"])
                contrasts["n_mice_I"] = int(support["n_mice"]["I"])
                contrasts["n_clusters_C"] = int(support["n_clusters"]["C"])
                contrasts["n_clusters_I"] = int(support["n_clusters"]["I"])
                contrasts["model_cluster_definition"] = cluster_definition
                contrasts[
                    "model_formula_with_random_effect"
                ] = model_formula_with_random_effect
                contrasts["mouse_random_effect_included"] = job["metric"] == "pv"
                contrasts["mouse_by_time_random_effect_included"] = False
                contrasts["pseudoreplication_warning"] = pseudoreplication_warning
                contrasts["posthoc_cohort_warning"] = POSTHOC_COHORT_WARNING
                contrast_frames.append(contrasts)
                coefficients = result.coefficients.copy()
                coefficients.insert(0, "model_role", role)
                coefficients.insert(0, "condition", condition)
                coefficients.insert(0, "metric", job["metric"])
                coefficients.insert(0, "analysis", job["analysis"])
                coefficients["model_cluster_definition"] = cluster_definition
                coefficients[
                    "model_formula_with_random_effect"
                ] = model_formula_with_random_effect
                coefficients["n_biological_mice"] = int(
                    sum(support["n_mice"].values())
                )
                coefficients["n_model_clusters"] = n_model_clusters
                coefficients["pseudoreplication_warning"] = pseudoreplication_warning
                coefficient_frames.append(coefficients)

    contrasts = pd.concat(contrast_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    contrasts = apply_fig4_bh_correction(contrasts)
    contrasts["project"] = "CA Figure-8 maze"
    contrasts["figure"] = "Fig. 4"
    contrasts["outcome_scale"] = "Fisher z of correlation"
    contrasts["plotted_scale"] = "raw correlation r"
    contrasts["time_definition"] = contrasts["analysis"].map(
        {
            "within": "D1-D5; day_c=day-3",
            "adjacent": "D1-D2 through D4-D5; pair_progress_c=pair_progress-2.5",
            "anchor": "D1-D2 through D1-D5; lag_c=lag-2.5",
        }
    )
    return contrasts, coefficients, diagnostics

def p_star_annotation(value: Any) -> str:
    """Return the Fig. 3-compatible panel symbol for a raw two-sided P value."""

    value = float(value)
    if not np.isfinite(value):
        return "n.s."
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."

def reported_probability(row: pd.Series) -> tuple[float, str]:
    """Return the manuscript-reporting value and its explicit statistical basis."""

    contrast = str(row.get("contrast", row.name))
    if contrast in {
        "reference_group_day_slope",
        "comparison_group_day_slope",
        "group_by_day_interaction",
        "centered_overall_group_effect",
    }:
        return float(row["p_value"]), PANEL_ANNOTATION_BASIS[contrast]
    if contrast.startswith("between_group_"):
        return float(row["p_value"]), ENDPOINT_REPORTING_BASIS
    raise ValueError(f"unrecognized reporting contrast {contrast!r}")

def build_annotations(
    contrasts: pd.DataFrame,
    *,
    analysis: str,
    metric: str,
) -> dict[str, dict[str, Any]]:
    selected = contrasts.loc[
        contrasts["analysis"].eq(analysis)
        & contrasts["metric"].eq(metric)
        & contrasts["model_role"].eq("adjusted")
    ]
    output: dict[str, dict[str, Any]] = {}
    for condition_key, condition_display in CONDITION_DISPLAY.items():
        table = selected.loc[selected["condition"].eq(condition_key)].set_index(
            "contrast"
        )
        overall_value, overall_basis = reported_probability(
            table.loc["centered_overall_group_effect"]
        )
        control_value, control_basis = reported_probability(
            table.loc["reference_group_day_slope"]
        )
        inhibition_value, inhibition_basis = reported_probability(
            table.loc["comparison_group_day_slope"]
        )
        interaction_value, interaction_basis = reported_probability(
            table.loc["group_by_day_interaction"]
        )
        output[condition_display] = {
            "overall": p_star_annotation(overall_value),
            "slopes": {
                "C": p_star_annotation(control_value),
                "I": p_star_annotation(inhibition_value),
            },
            "interaction": p_star_annotation(interaction_value),
            "basis": {
                "overall": overall_basis,
                "slopes": {"C": control_basis, "I": inhibition_basis},
                "interaction": interaction_basis,
            },
        }
    return output

def _display_condition(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["condition"] = result["condition"].map(CONDITION_DISPLAY)
    if result["condition"].isna().any():
        raise ValueError("unexpected condition while preparing plot")
    return result
