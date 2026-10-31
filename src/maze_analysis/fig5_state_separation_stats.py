"""Manuscript definitions collected from fig5_state_separation_stats.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, Mapping

import numpy as np

import pandas as pd

from .manuscript_lmm import (
    LongitudinalLMMResult,
    apply_bh_correction,
    fit_adjusted_unadjusted_lmm,
)

STRATUM_COLUMNS = ("project", "metric", "contrast", "section")

GROUP_COLUMN = "group"

MOUSE_COLUMN = "mouse"

DAY_COLUMN = "day"

VALUE_COLUMN = "value"

COVARIATE_COLUMNS = (
    "mean_speed_z",
    "route_relative_lateral_deviation_z",
)

CANONICAL_COLUMNS = (
    *STRATUM_COLUMNS,
    GROUP_COLUMN,
    MOUSE_COLUMN,
    DAY_COLUMN,
    VALUE_COLUMN,
    *COVARIATE_COLUMNS,
)

SAMPLING_UNIT_COLUMNS = (MOUSE_COLUMN, DAY_COLUMN)

CANONICAL_KEY_COLUMNS = (
    *STRATUM_COLUMNS,
    GROUP_COLUMN,
    MOUSE_COLUMN,
    DAY_COLUMN,
)

EXPECTED_GROUPS = ("C", "I")

EXPECTED_DAYS = (1, 2, 3, 4, 5)

DAY_CENTER = 3.0

PLANNED_ESTIMANDS = (
    "reference_group_day_slope",
    "comparison_group_day_slope",
    "group_by_day_interaction",
    "centered_overall_group_effect",
    "between_group_D1",
    "between_group_D5",
)

CORE_ANNOTATION_ESTIMANDS = (
    "reference_group_day_slope",
    "comparison_group_day_slope",
    "group_by_day_interaction",
    "centered_overall_group_effect",
)

RAW_P_REPORTING_BASIS = "planned contrasts; raw two-sided P values"

OUTCOME_TRANSFORM = "none_shuffle_z_direct"

OUTCOME_SCALE = "shuffle-standardized state-separation z score"

MODEL_FORMULA_CONTRACT = (
    "value ~ day_c * group + mean_speed_z + "
    "route_relative_lateral_deviation_z + (1 | mouse)"
)

@dataclass
class Fig5LongitudinalStatsResult:
    """Tables and in-memory fits from one generic Fig. 5 model run."""

    contrasts: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    support: pd.DataFrame
    fit_results: dict[
        tuple[str, str, str, str], dict[str, LongitudinalLMMResult]
    ] = field(default_factory=dict, repr=False)

    @property
    def manuscript_contrasts(self) -> pd.DataFrame:
        """Return the adjusted planned contrasts used for manuscript claims."""

        if "manuscript_facing" not in self.contrasts.columns:
            return self.contrasts.iloc[0:0].copy()
        return self.contrasts.loc[self.contrasts["manuscript_facing"]].copy()

def validate_fig5_longitudinal_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize one Fig. 5 mouse-day source table.

    Missing ``value`` or covariate cells are retained so that support and the
    shared complete-case sample remain auditable.  Infinite values and
    nonnumeric tokens are rejected.  Every analysis stratum and group must
    retain at least one row on each of D1-D5; individual mouse-days may be
    absent when the upstream contrast is not estimable.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    missing = sorted(set(CANONICAL_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(f"table is missing canonical Fig. 5 columns: {missing}")

    result = table.copy()
    identity_columns = (*STRATUM_COLUMNS, GROUP_COLUMN, MOUSE_COLUMN)
    for column in identity_columns:
        if result[column].isna().any():
            raise ValueError(f"identity column {column!r} contains missing values")
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"identity column {column!r} contains blank values")

    unexpected_groups = sorted(set(result[GROUP_COLUMN]).difference(EXPECTED_GROUPS))
    if unexpected_groups:
        raise ValueError(
            f"group must contain only {EXPECTED_GROUPS}; found {unexpected_groups}"
        )

    original_day = result[DAY_COLUMN]
    numeric_day = pd.to_numeric(original_day, errors="coerce")
    invalid_day_token = original_day.notna() & numeric_day.isna()
    if original_day.isna().any() or invalid_day_token.any():
        raise ValueError("day contains missing or nonnumeric values")
    day_values = numeric_day.to_numpy(dtype=float)
    if not np.isfinite(day_values).all():
        raise ValueError("day must be finite")
    if not np.equal(day_values, np.floor(day_values)).all():
        raise ValueError("day must contain integer biological day labels")
    result[DAY_COLUMN] = numeric_day.astype(int)
    unexpected_days = sorted(set(result[DAY_COLUMN]).difference(EXPECTED_DAYS))
    if unexpected_days:
        raise ValueError(
            f"day must be restricted to D1-D5; found {unexpected_days}"
        )

    for column in (VALUE_COLUMN, *COVARIATE_COLUMNS):
        original = result[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_token = original.notna() & numeric.isna()
        if invalid_token.any():
            examples = original.loc[invalid_token].astype(str).unique()[:5].tolist()
            raise ValueError(f"column {column!r} contains nonnumeric values: {examples}")
        numeric_values = numeric.to_numpy(dtype=float)
        if np.isinf(numeric_values).any():
            raise ValueError(f"column {column!r} contains infinite values")
        result[column] = numeric.astype(float)

    duplicate = result.duplicated(CANONICAL_KEY_COLUMNS, keep=False)
    if duplicate.any():
        examples = (
            result.loc[duplicate, list(CANONICAL_KEY_COLUMNS)]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "duplicate Fig. 5 mouse-day sampling units for canonical key "
            f"{CANONICAL_KEY_COLUMNS}; examples={examples}"
        )

    mouse_groups = result.groupby(
        ["project", MOUSE_COLUMN], observed=True
    )[GROUP_COLUMN].nunique()
    if (mouse_groups > 1).any():
        collided = [tuple(value) for value in mouse_groups.index[mouse_groups > 1]]
        raise ValueError(f"mouse identifiers occur in both groups: {collided}")

    for stratum, subset in result.groupby(
        list(STRATUM_COLUMNS), observed=True, sort=False
    ):
        present_groups = set(subset[GROUP_COLUMN])
        if present_groups != set(EXPECTED_GROUPS):
            raise ValueError(
                f"stratum {tuple(stratum)} must contain both C and I; "
                f"found {sorted(present_groups)}"
            )
        for group, group_table in subset.groupby(
            GROUP_COLUMN, observed=True, sort=False
        ):
            present_days = set(group_table[DAY_COLUMN])
            if present_days != set(EXPECTED_DAYS):
                raise ValueError(
                    f"stratum {tuple(stratum)}, group {group!r} must cover D1-D5; "
                    f"found {sorted(present_days)}"
                )

    result["day_c"] = result[DAY_COLUMN].astype(float) - DAY_CENTER
    return result.sort_values(
        list(CANONICAL_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)

def _build_support_table(validated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = (*STRATUM_COLUMNS, GROUP_COLUMN, DAY_COLUMN)
    for key, subset in validated.groupby(
        list(group_columns), observed=True, sort=True
    ):
        value_finite = np.isfinite(subset[VALUE_COLUMN].to_numpy(dtype=float))
        speed_finite = np.isfinite(
            subset[COVARIATE_COLUMNS[0]].to_numpy(dtype=float)
        )
        lateral_finite = np.isfinite(
            subset[COVARIATE_COLUMNS[1]].to_numpy(dtype=float)
        )
        covariates_complete = speed_finite & lateral_finite
        adjusted_complete = value_finite & covariates_complete

        def mice(mask: np.ndarray) -> tuple[str, ...]:
            return tuple(sorted(subset.loc[mask, MOUSE_COLUMN].astype(str).unique()))

        all_mice = mice(np.ones(subset.shape[0], dtype=bool))
        adjusted_mice = mice(adjusted_complete)
        row = dict(zip(group_columns, key, strict=True))
        row.update(
            {
                "n_rows": int(subset.shape[0]),
                "n_mice_all": len(all_mice),
                "n_value_finite": int(value_finite.sum()),
                "n_covariates_complete": int(covariates_complete.sum()),
                "n_adjusted_complete": int(adjusted_complete.sum()),
                "n_missing_value": int((~value_finite).sum()),
                "n_missing_mean_speed_z": int((~speed_finite).sum()),
                "n_missing_route_relative_lateral_deviation_z": int(
                    (~lateral_finite).sum()
                ),
                "mice_all": all_mice,
                "mice_value_finite": mice(value_finite),
                "mice_adjusted_complete": adjusted_mice,
                "mice_excluded_from_adjusted": tuple(
                    mouse for mouse in all_mice if mouse not in adjusted_mice
                ),
                "sampling_unit": "mouse/day",
                "biological_replicate": "mouse",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

def fit_fig5_longitudinal_models(
    table: pd.DataFrame,
    *,
    include_bh_audit: bool = False,
    bh_alpha: float = 0.05,
    boundary_tolerance: float = 1e-8,
    maxiter: int = 1000,
) -> Fig5LongitudinalStatsResult:
    """Fit adjusted and shared-sample unadjusted LMMs for every stratum.

    The adjusted model is manuscript-facing.  The unadjusted model is an audit
    sensitivity fitted to the exact same covariate-complete observations.
    Model convergence failures remain explicit failed MixedLM results; this
    wrapper never selects an OLS fallback or a model based on significance.
    """

    validated = validate_fig5_longitudinal_table(table)
    support = _build_support_table(validated)
    fit_results: dict[
        tuple[str, str, str, str], dict[str, LongitudinalLMMResult]
    ] = {}
    contrast_blocks: list[pd.DataFrame] = []
    coefficient_blocks: list[pd.DataFrame] = []

    for raw_key, subset in validated.groupby(
        list(STRATUM_COLUMNS), observed=True, sort=True
    ):
        key = tuple(str(value) for value in raw_key)
        paired = fit_adjusted_unadjusted_lmm(
            subset,
            continuous_covariates=COVARIATE_COLUMNS,
            value_col=VALUE_COLUMN,
            time_col=DAY_COLUMN,
            group_col=GROUP_COLUMN,
            mouse_col=MOUSE_COLUMN,
            reference_group="C",
            comparison_group="I",
            sampling_unit_cols=SAMPLING_UNIT_COLUMNS,
            first_time=EXPECTED_DAYS[0],
            last_time=EXPECTED_DAYS[-1],
            boundary_tolerance=boundary_tolerance,
            maxiter=maxiter,
        )
        fit_results[key] = paired
        for model_role in ("adjusted", "unadjusted"):
            model = paired[model_role]
            contrast_blocks.append(
                _decorate_contrasts(model.contrasts, key=key, model_role=model_role)
            )
            coefficient_blocks.append(
                _decorate_coefficients(
                    model.coefficients, key=key, model_role=model_role
                )
            )

    contrasts = pd.concat(contrast_blocks, ignore_index=True)
    if include_bh_audit:
        contrasts = add_bh_audit_columns(contrasts, alpha=bh_alpha)
    coefficients = pd.concat(coefficient_blocks, ignore_index=True)
    diagnostics = build_fig5_diagnostics_table(fit_results)
    return Fig5LongitudinalStatsResult(
        contrasts=contrasts,
        coefficients=coefficients,
        diagnostics=diagnostics,
        support=support,
        fit_results=fit_results,
    )

def _decorate_contrasts(
    table: pd.DataFrame,
    *,
    key: tuple[str, str, str, str],
    model_role: str,
) -> pd.DataFrame:
    result = table.rename(columns={"contrast": "estimand_contrast"}).copy()
    unexpected = sorted(set(result["estimand_contrast"]).difference(PLANNED_ESTIMANDS))
    missing = sorted(set(PLANNED_ESTIMANDS).difference(result["estimand_contrast"]))
    if unexpected or missing:
        raise RuntimeError(
            "manuscript_lmm returned an unexpected planned-estimand set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for column, value in zip(STRATUM_COLUMNS, key, strict=True):
        result[column] = value
    result["model_role"] = str(model_role)
    result["reported_probability"] = "P"
    result["reported_probability_value"] = pd.to_numeric(
        result["p_value"], errors="coerce"
    )
    finite_p = np.isfinite(result["reported_probability_value"].to_numpy(float))
    result["raw_p_significant"] = finite_p & (
        result["reported_probability_value"].to_numpy(float) < 0.05
    )
    result["panel_symbol"] = result["reported_probability_value"].map(
        raw_p_to_symbol
    )
    result["primary_inference_basis"] = RAW_P_REPORTING_BASIS
    result["manuscript_facing"] = model_role == "adjusted"
    result["reporting_role"] = np.where(
        result["manuscript_facing"],
        "primary adjusted LMM",
        "shared-complete-case unadjusted sensitivity",
    )
    result["estimand_family_audit_only"] = result["estimand_contrast"].map(
        _estimand_family
    )
    result["outcome_scale"] = OUTCOME_SCALE
    result["outcome_transform"] = OUTCOME_TRANSFORM
    result["fisher_transform_applied"] = False
    result["sampling_unit"] = "mouse/day"
    result["biological_replicate"] = "mouse"
    result["day_center"] = DAY_CENTER
    result["model_formula_contract"] = MODEL_FORMULA_CONTRACT
    leading = [*STRATUM_COLUMNS, "model_role", "estimand_contrast"]
    return result.loc[:, leading + [c for c in result.columns if c not in leading]]

def _decorate_coefficients(
    table: pd.DataFrame,
    *,
    key: tuple[str, str, str, str],
    model_role: str,
) -> pd.DataFrame:
    result = table.copy()
    for column, value in zip(STRATUM_COLUMNS, key, strict=True):
        result[column] = value
    result["model_role"] = str(model_role)
    result["outcome_transform"] = OUTCOME_TRANSFORM
    result["fisher_transform_applied"] = False
    leading = [*STRATUM_COLUMNS, "model_role", "term"]
    return result.loc[:, leading + [c for c in result.columns if c not in leading]]

def build_fig5_diagnostics_table(
    fit_results: Mapping[
        tuple[str, str, str, str], Mapping[str, LongitudinalLMMResult]
    ],
) -> pd.DataFrame:
    """Flatten paired LMM diagnostics and expose blocking/nonblocking QC."""

    rows: list[dict[str, Any]] = []
    for raw_key, paired in fit_results.items():
        key = tuple(str(value) for value in raw_key)
        if len(key) != len(STRATUM_COLUMNS):
            raise ValueError(
                f"fit-results key must match {STRATUM_COLUMNS}; received {raw_key!r}"
            )
        if set(paired) != {"adjusted", "unadjusted"}:
            raise ValueError(
                "each fit-results entry must contain adjusted and unadjusted models"
            )
        paired_nobs_match = (
            paired["adjusted"].diagnostics.get("nobs")
            == paired["unadjusted"].diagnostics.get("nobs")
        )
        for model_role in ("adjusted", "unadjusted"):
            model = paired[model_role]
            diagnostics = model.diagnostics
            expected_covariates = COVARIATE_COLUMNS if model_role == "adjusted" else ()
            covariates_match = tuple(
                diagnostics.get("continuous_covariates", ())
            ) == tuple(expected_covariates)
            sampling_unit_match = tuple(
                diagnostics.get("sampling_unit_cols") or ()
            ) == SAMPLING_UNIT_COLUMNS
            center_match = np.isclose(
                float(diagnostics.get("time_center", np.nan)), DAY_CENTER
            )
            no_fallback = diagnostics.get("fallback_used") is False
            shared_complete_case = bool(
                diagnostics.get("common_complete_case", False)
            )
            model_qc_pass = bool(
                model.succeeded
                and diagnostics.get("converged") is True
                and diagnostics.get("covariance_finite") is True
                and no_fallback
                and shared_complete_case
                and paired_nobs_match
                and sampling_unit_match
                and center_match
                and covariates_match
            )
            row = dict(zip(STRATUM_COLUMNS, key, strict=True))
            row.update(
                {
                    "model_role": model_role,
                    "model_status": diagnostics.get("model_status"),
                    "succeeded": bool(model.succeeded),
                    "model_qc_pass": model_qc_pass,
                    "optimizer": diagnostics.get("optimizer"),
                    "converged": diagnostics.get("converged"),
                    "covariance_finite": diagnostics.get("covariance_finite"),
                    "boundary_random_intercept": diagnostics.get(
                        "boundary_random_intercept"
                    ),
                    "boundary_is_nonblocking_flag": True,
                    "random_intercept_variance": diagnostics.get(
                        "random_intercept_variance"
                    ),
                    "fallback_used": diagnostics.get("fallback_used"),
                    "failure_reason": diagnostics.get("failure_reason"),
                    "nobs": diagnostics.get("nobs"),
                    "n_candidate_rows": diagnostics.get("n_candidate_rows"),
                    "n_dropped_rows": diagnostics.get("n_dropped_rows"),
                    "n_mice": diagnostics.get("n_mice"),
                    "n_mice_C": diagnostics.get("n_mice_reference"),
                    "n_mice_I": diagnostics.get("n_mice_comparison"),
                    "mice": tuple(diagnostics.get("mice", ())),
                    "common_complete_case_n_candidate": diagnostics.get(
                        "common_complete_case_n_candidate"
                    ),
                    "common_complete_case_nobs": diagnostics.get(
                        "common_complete_case_nobs"
                    ),
                    "common_complete_case_n_dropped": diagnostics.get(
                        "common_complete_case_n_dropped"
                    ),
                    "paired_nobs_match": bool(paired_nobs_match),
                    "shared_complete_case_verified": bool(
                        shared_complete_case and paired_nobs_match
                    ),
                    "sampling_unit_cols": tuple(
                        diagnostics.get("sampling_unit_cols") or ()
                    ),
                    "sampling_unit_contract_match": bool(sampling_unit_match),
                    "continuous_covariates": tuple(
                        diagnostics.get("continuous_covariates", ())
                    ),
                    "covariate_contract_match": bool(covariates_match),
                    "time_center": diagnostics.get("time_center"),
                    "day_center_contract_match": bool(center_match),
                    "formula": diagnostics.get("formula"),
                    "model_formula_contract": MODEL_FORMULA_CONTRACT,
                    "outcome_scale": OUTCOME_SCALE,
                    "outcome_transform": OUTCOME_TRANSFORM,
                    "fisher_transform_applied": False,
                    "optimizer_attempts": diagnostics.get("optimizer_attempts"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [*STRATUM_COLUMNS, "model_role"], kind="mergesort"
    ).reset_index(drop=True)

def add_bh_audit_columns(
    contrasts: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add BH results under names that explicitly mark them audit-only.

    Families are separate for every project/metric/condition contrast/section
    and model role.  The two within-group slopes form one family, D1/D5 form a
    second family, and the interaction and centered group effect are singleton
    audit families.  Raw P remains the only manuscript-facing probability.
    """

    required = {
        *STRATUM_COLUMNS,
        "model_role",
        "estimand_contrast",
        "p_value",
    }
    missing = sorted(required.difference(contrasts.columns))
    if missing:
        raise ValueError(f"contrasts table is missing BH audit columns: {missing}")
    stale = sorted(
        set(contrasts.columns).intersection(
            {"q_value", "significant", "bh_q_audit_only", "bh_significant_audit_only"}
        )
    )
    if stale:
        raise ValueError(f"contrasts table already contains correction columns: {stale}")

    result = contrasts.copy()
    result["estimand_family_audit_only"] = result["estimand_contrast"].map(
        _estimand_family
    )
    family_columns = (
        *STRATUM_COLUMNS,
        "model_role",
        "estimand_family_audit_only",
    )
    result["bh_family_id_audit_only"] = result.loc[:, family_columns].astype(str).agg(
        "|".join, axis=1
    )
    corrected = apply_bh_correction(
        result,
        p_col="p_value",
        family_cols="bh_family_id_audit_only",
        alpha=alpha,
    ).rename(
        columns={
            "q_value": "bh_q_audit_only",
            "significant": "bh_significant_audit_only",
        }
    )
    corrected["bh_family_size_audit_only"] = corrected.groupby(
        "bh_family_id_audit_only", observed=True
    )["estimand_contrast"].transform("size")
    corrected["bh_alpha_audit_only"] = float(alpha)
    corrected["bh_is_manuscript_facing"] = False
    return corrected

def build_fig5_annotation_payloads(
    contrasts: pd.DataFrame,
    *,
    model_role: str = "adjusted",
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Build Fig. 3/4-style raw-P annotation payloads for each stratum."""

    required = {
        *STRATUM_COLUMNS,
        "model_role",
        "estimand_contrast",
        "p_value",
    }
    missing = sorted(required.difference(contrasts.columns))
    if missing:
        raise ValueError(f"contrasts table is missing annotation columns: {missing}")
    selected = contrasts.loc[contrasts["model_role"].eq(model_role)].copy()
    if selected.empty:
        raise ValueError(f"no contrasts found for model_role={model_role!r}")

    payloads: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for raw_key, subset in selected.groupby(
        list(STRATUM_COLUMNS), observed=True, sort=True
    ):
        duplicated = subset["estimand_contrast"].duplicated(keep=False)
        if duplicated.any():
            names = sorted(subset.loc[duplicated, "estimand_contrast"].unique())
            raise ValueError(f"duplicate annotation estimands in {raw_key}: {names}")
        indexed = subset.set_index("estimand_contrast")
        missing_estimands = sorted(
            set(CORE_ANNOTATION_ESTIMANDS).difference(indexed.index)
        )
        if missing_estimands:
            raise ValueError(
                f"stratum {raw_key} is missing annotation estimands: {missing_estimands}"
            )

        def symbol(estimand: str) -> str:
            return raw_p_to_symbol(indexed.loc[estimand, "p_value"])

        payloads[tuple(str(value) for value in raw_key)] = {
            "overall": symbol("centered_overall_group_effect"),
            "slopes": {
                "C": symbol("reference_group_day_slope"),
                "I": symbol("comparison_group_day_slope"),
            },
            "interaction": symbol("group_by_day_interaction"),
            "basis": {
                "overall": RAW_P_REPORTING_BASIS,
                "slopes": {
                    "C": RAW_P_REPORTING_BASIS,
                    "I": RAW_P_REPORTING_BASIS,
                },
                "interaction": RAW_P_REPORTING_BASIS,
            },
            "reported_probability": "P",
        }
    return payloads

def raw_p_to_symbol(value: Any) -> str:
    """Return the released Fig. 3/4 strict raw-P star grammar."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "n.s."
    if not np.isfinite(probability):
        return "n.s."
    if probability < 0.001:
        return "***"
    if probability < 0.01:
        return "**"
    if probability < 0.05:
        return "*"
    return "n.s."

def _estimand_family(estimand: str) -> str:
    if estimand in {"reference_group_day_slope", "comparison_group_day_slope"}:
        return "within_group_day_slopes"
    if estimand == "group_by_day_interaction":
        return "group_by_day_interaction"
    if estimand == "centered_overall_group_effect":
        return "centered_overall_group_effect"
    if estimand in {"between_group_D1", "between_group_D5"}:
        return "between_group_endpoints"
    raise ValueError(f"unrecognized planned estimand {estimand!r}")
