"""Manuscript definitions collected from manuscript_validation.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from itertools import combinations, product

from math import comb

from typing import Any, Sequence

import numpy as np

import pandas as pd

from scipy import stats

from statsmodels.genmod.cov_struct import Exchangeable

from statsmodels.genmod.families import Binomial

from statsmodels.genmod.generalized_estimating_equations import GEE

from .manuscript_lmm import apply_bh_correction, fit_longitudinal_lmm

_NORMAL_975 = float(stats.norm.ppf(0.975))

@dataclass(frozen=True)
class GEESensitivityResult:
    """Trial-level binomial GEE sensitivity output."""

    contrasts: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: dict[str, Any]

def build_group_day_summary(
    table: pd.DataFrame,
    *,
    value_col: str,
    group_col: str = "group",
    day_col: str = "day",
    mouse_col: str = "mouse",
) -> pd.DataFrame:
    """Return mean, SD, SEM and exact biological n for each group/day."""

    required = {value_col, group_col, day_col, mouse_col}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (group, day), subset in table.groupby(
        [group_col, day_col], observed=True, sort=True
    ):
        numeric = pd.to_numeric(subset[value_col], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        values = numeric.loc[finite].to_numpy(dtype=float)
        mice = subset.loc[finite, mouse_col].astype(str)
        n_mouse = int(mice.nunique())
        if values.size > 1:
            sd = float(np.std(values, ddof=1))
            sem = float(sd / np.sqrt(values.size))
        else:
            sd = np.nan
            sem = np.nan
        rows.append(
            {
                "group": str(group),
                "day": int(day),
                "mean": float(np.mean(values)) if values.size else np.nan,
                "sd": sd,
                "sem": sem,
                "n_mouse": n_mouse,
                "n_observation": int(values.size),
            }
        )
    return pd.DataFrame(rows)

def manuscript_family_id(contrast: str) -> str:
    """Map the six locked longitudinal contrasts to prespecified BH families."""

    if contrast in {"reference_group_day_slope", "comparison_group_day_slope"}:
        return "within_group_day_slopes"
    if contrast == "group_by_day_interaction":
        return "group_by_day_interaction"
    if contrast == "centered_overall_group_effect":
        return "centered_overall_group_effect"
    if str(contrast).startswith("between_group_D"):
        return "between_group_endpoints"
    raise ValueError(f"unrecognized manuscript contrast: {contrast!r}")

def finalize_manuscript_contrasts(
    contrasts: pd.DataFrame,
    *,
    panel: str,
    model_variant: str,
) -> pd.DataFrame:
    """Attach explicit correction-family IDs and BH q values."""

    if "contrast" not in contrasts.columns:
        raise ValueError("contrasts table is missing 'contrast'")
    result = contrasts.copy()
    result.insert(0, "panel", str(panel))
    result.insert(1, "model_variant", str(model_variant))
    result["family"] = result["contrast"].map(manuscript_family_id)
    result["family_id"] = (
        result["panel"].astype(str)
        + "|"
        + result["model_variant"].astype(str)
        + "|"
        + result["family"].astype(str)
    )
    return apply_bh_correction(result, family_cols="family_id")

def mouse_linear_slopes(
    table: pd.DataFrame,
    *,
    value_col: str,
    group_col: str = "group",
    day_col: str = "day",
    mouse_col: str = "mouse",
) -> pd.DataFrame:
    """Fit one raw ordinary least-squares day slope per mouse."""

    required = {value_col, group_col, day_col, mouse_col}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    rows = []
    for (group, mouse), subset in table.groupby(
        [group_col, mouse_col], observed=True, sort=True
    ):
        x = pd.to_numeric(subset[day_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(subset[value_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size >= 2 and np.unique(x).size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope = intercept = np.nan
        rows.append(
            {
                "group": str(group),
                "mouse": str(mouse),
                "slope": float(slope),
                "intercept": float(intercept),
                "n_day": int(x.size),
                "first_day": float(np.min(x)) if x.size else np.nan,
                "last_day": float(np.max(x)) if x.size else np.nan,
            }
        )
    return pd.DataFrame(rows)

def exact_sign_flip_test(values: Sequence[float]) -> dict[str, float | int]:
    """Two-sided exact sign-flip test of the sample mean against zero."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": np.nan, "p_value": np.nan, "n": 0, "n_permutation": 0}
    if values.size > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 observations")
    observed = float(np.mean(values))
    null = np.fromiter(
        (
            float(np.mean(values * np.asarray(signs, dtype=float)))
            for signs in product((-1.0, 1.0), repeat=int(values.size))
        ),
        dtype=float,
        count=2 ** int(values.size),
    )
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    return {
        "estimate": observed,
        "p_value": p_value,
        "n": int(values.size),
        "n_permutation": int(null.size),
    }

def exact_label_permutation_test(
    values: Sequence[float],
    labels: Sequence[str],
    *,
    reference_group: str = "C",
    comparison_group: str = "I",
) -> dict[str, float | int]:
    """Two-sided exact test of comparison-minus-reference mean difference."""

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=str)
    finite = np.isfinite(values) & np.isin(labels, [reference_group, comparison_group])
    values = values[finite]
    labels = labels[finite]
    n_comparison = int(np.sum(labels == comparison_group))
    n_reference = int(np.sum(labels == reference_group))
    if not n_comparison or not n_reference:
        return {
            "estimate": np.nan,
            "p_value": np.nan,
            "n_reference": n_reference,
            "n_comparison": n_comparison,
            "n_permutation": 0,
        }
    n_total = int(values.size)
    n_permutation = comb(n_total, n_comparison)
    if n_permutation > 2_000_000:
        raise ValueError("exact label enumeration exceeds 2,000,000 assignments")
    observed = float(
        np.mean(values[labels == comparison_group])
        - np.mean(values[labels == reference_group])
    )
    all_indices = np.arange(n_total)
    count_extreme = 0
    for comparison_tuple in combinations(range(n_total), n_comparison):
        comparison_indices = np.fromiter(comparison_tuple, dtype=int)
        mask = np.ones(n_total, dtype=bool)
        mask[comparison_indices] = False
        delta = float(np.mean(values[comparison_indices]) - np.mean(values[all_indices[mask]]))
        count_extreme += int(abs(delta) >= abs(observed) - 1e-15)
    return {
        "estimate": observed,
        "p_value": float(count_extreme / n_permutation),
        "n_reference": n_reference,
        "n_comparison": n_comparison,
        "n_permutation": int(n_permutation),
    }

def exact_mouse_level_sensitivities(
    table: pd.DataFrame,
    *,
    value_col: str,
    panel: str,
    first_day: int = 1,
    last_day: int = 5,
    group_col: str = "group",
    day_col: str = "day",
    mouse_col: str = "mouse",
    reference_group: str = "C",
    comparison_group: str = "I",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exact raw mouse-summary checks matched to the six LMM estimands."""

    slopes = mouse_linear_slopes(
        table,
        value_col=value_col,
        group_col=group_col,
        day_col=day_col,
        mouse_col=mouse_col,
    )
    rows: list[dict[str, Any]] = []
    for group, contrast in (
        (reference_group, "reference_group_day_slope"),
        (comparison_group, "comparison_group_day_slope"),
    ):
        test = exact_sign_flip_test(slopes.loc[slopes["group"] == group, "slope"])
        rows.append(
            {
                "contrast": contrast,
                "estimand": f"mean raw mouse day slope within {group}",
                "test": "exact two-sided sign-flip test",
                **test,
            }
        )
    group_slope = exact_label_permutation_test(
        slopes["slope"],
        slopes["group"],
        reference_group=reference_group,
        comparison_group=comparison_group,
    )
    rows.append(
        {
            "contrast": "group_by_day_interaction",
            "estimand": f"{comparison_group} minus {reference_group} raw mouse day slope",
            "test": "exact two-sided label permutation",
            **group_slope,
        }
    )

    mouse_means = (
        table.groupby([group_col, mouse_col], as_index=False, observed=True)[value_col]
        .mean()
        .rename(columns={value_col: "mouse_mean"})
    )
    overall = exact_label_permutation_test(
        mouse_means["mouse_mean"],
        mouse_means[group_col],
        reference_group=reference_group,
        comparison_group=comparison_group,
    )
    rows.append(
        {
            "contrast": "centered_overall_group_effect",
            "estimand": f"{comparison_group} minus {reference_group} raw five-day mouse mean",
            "test": "exact two-sided label permutation",
            **overall,
        }
    )
    for day in (first_day, last_day):
        endpoint = table.loc[pd.to_numeric(table[day_col], errors="coerce") == day]
        test = exact_label_permutation_test(
            endpoint[value_col],
            endpoint[group_col],
            reference_group=reference_group,
            comparison_group=comparison_group,
        )
        rows.append(
            {
                "contrast": f"between_group_D{day}",
                "estimand": f"{comparison_group} minus {reference_group} raw mouse mean at D{day}",
                "test": "exact two-sided label permutation",
                **test,
            }
        )
    result = pd.DataFrame(rows)
    result = finalize_manuscript_contrasts(
        result,
        panel=panel,
        model_variant="exact_raw_mouse_summary_sensitivity",
    )
    return slopes, result

def leave_one_mouse_out_lmm(
    table: pd.DataFrame,
    *,
    value_col: str,
    continuous_covariates: Sequence[str],
    panel: str,
    time_col: str = "day",
    group_col: str = "group",
    mouse_col: str = "mouse",
    reference_group: str = "C",
    comparison_group: str = "I",
    first_time: float = 1,
    last_time: float = 5,
) -> pd.DataFrame:
    """Refit the adjusted LMM after omitting each biological replicate once."""

    rows = []
    mice = sorted(table[mouse_col].dropna().astype(str).unique())
    for omitted_mouse in mice:
        subset = table.loc[table[mouse_col].astype(str) != omitted_mouse].copy()
        result = fit_longitudinal_lmm(
            subset,
            value_col=value_col,
            time_col=time_col,
            group_col=group_col,
            mouse_col=mouse_col,
            reference_group=reference_group,
            comparison_group=comparison_group,
            continuous_covariates=tuple(continuous_covariates),
            first_time=first_time,
            last_time=last_time,
        )
        for _, contrast in result.contrasts.iterrows():
            rows.append(
                {
                    "panel": str(panel),
                    "omitted_mouse": omitted_mouse,
                    "contrast": contrast["contrast"],
                    "beta": contrast["beta"],
                    "standard_error": contrast["standard_error"],
                    "ci95_low": contrast["ci95_low"],
                    "ci95_high": contrast["ci95_high"],
                    "p_value": contrast["p_value"],
                    "model_status": result.diagnostics["model_status"],
                    "optimizer": result.diagnostics["optimizer"],
                    "boundary_random_intercept": result.diagnostics[
                        "boundary_random_intercept"
                    ],
                    "nobs": result.diagnostics["nobs"],
                    "n_mice": result.diagnostics["n_mice"],
                }
            )
    return pd.DataFrame(rows)

def _wald_from_vector(
    params: np.ndarray,
    covariance: np.ndarray,
    vector: np.ndarray,
) -> dict[str, float]:
    beta = float(vector @ params)
    variance = float(vector @ covariance @ vector)
    if variance < 0 and abs(variance) <= 1e-12:
        variance = 0.0
    se = float(np.sqrt(variance)) if variance >= 0 else np.nan
    z_value = float(beta / se) if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        "beta": beta,
        "standard_error": se,
        "z_value": z_value,
        "ci95_low": float(beta - _NORMAL_975 * se),
        "ci95_high": float(beta + _NORMAL_975 * se),
        "p_value": p_value,
        "odds_ratio": float(np.exp(beta)),
        "odds_ratio_ci95_low": float(np.exp(beta - _NORMAL_975 * se)),
        "odds_ratio_ci95_high": float(np.exp(beta + _NORMAL_975 * se)),
    }

def fit_trial_binomial_gee(
    table: pd.DataFrame,
    *,
    outcome_col: str,
    continuous_covariates: Sequence[str],
    panel: str,
    time_col: str = "day",
    group_col: str = "group",
    mouse_col: str = "mouse",
    reference_group: str = "C",
    comparison_group: str = "I",
    first_time: float = 1,
    last_time: float = 5,
) -> GEESensitivityResult:
    """Fit a trial-level binomial GEE clustered by mouse as sensitivity only."""

    required = {
        outcome_col,
        time_col,
        group_col,
        mouse_col,
        *tuple(continuous_covariates),
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    work = table.loc[
        table[group_col].astype(str).isin([reference_group, comparison_group])
    ].copy()
    center = (float(first_time) + float(last_time)) / 2.0
    work["_time"] = pd.to_numeric(work[time_col], errors="coerce") - center
    work["_group"] = (work[group_col].astype(str) == comparison_group).astype(float)
    work["_interaction"] = work["_time"] * work["_group"]
    work["_outcome"] = pd.to_numeric(work[outcome_col], errors="coerce")
    design_columns = ["_intercept", "_time", "_group", "_interaction"]
    work["_intercept"] = 1.0
    for index, column in enumerate(continuous_covariates):
        alias = f"_covariate_{index}"
        work[alias] = pd.to_numeric(work[column], errors="coerce")
        design_columns.append(alias)
    finite = work[mouse_col].notna()
    finite &= np.isin(work["_outcome"].to_numpy(dtype=float), [0.0, 1.0])
    for column in design_columns:
        finite &= np.isfinite(work[column].to_numpy(dtype=float))
    work = work.loc[finite].copy()
    if work[mouse_col].nunique() < 4:
        raise ValueError("binomial GEE requires at least four mouse clusters")
    model = GEE(
        work["_outcome"].to_numpy(dtype=float),
        work[design_columns].to_numpy(dtype=float),
        groups=work[mouse_col].astype(str).to_numpy(),
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    fit = model.fit(maxiter=200)
    params = np.asarray(fit.params, dtype=float)
    covariance = np.asarray(fit.cov_params(), dtype=float)

    def vector(time_weight=0.0, group_weight=0.0, interaction_weight=0.0):
        result = np.zeros(len(design_columns), dtype=float)
        result[1] = float(time_weight)
        result[2] = float(group_weight)
        result[3] = float(interaction_weight)
        return result

    first_delta = float(first_time) - center
    last_delta = float(last_time) - center
    definitions = [
        ("reference_group_day_slope", vector(time_weight=1)),
        (
            "comparison_group_day_slope",
            vector(time_weight=1, interaction_weight=1),
        ),
        ("group_by_day_interaction", vector(interaction_weight=1)),
        ("centered_overall_group_effect", vector(group_weight=1)),
        (
            f"between_group_D{int(first_time)}",
            vector(group_weight=1, interaction_weight=first_delta),
        ),
        (
            f"between_group_D{int(last_time)}",
            vector(group_weight=1, interaction_weight=last_delta),
        ),
    ]
    contrast_rows = []
    for contrast, contrast_vector in definitions:
        contrast_rows.append(
            {
                "contrast": contrast,
                "test": "trial-level binomial GEE robust Wald z-test",
                "scale": "log odds; odds_ratio=exp(beta)",
                **_wald_from_vector(params, covariance, contrast_vector),
            }
        )
    contrasts = finalize_manuscript_contrasts(
        pd.DataFrame(contrast_rows),
        panel=panel,
        model_variant="trial_binomial_GEE_sensitivity",
    )
    coefficient_rows = []
    for index, term in enumerate(design_columns):
        coefficient_rows.append(
            {"term": term, **_wald_from_vector(params, covariance, np.eye(len(params))[index])}
        )
    diagnostics = {
        "model": "trial-level binomial GEE with exchangeable working correlation",
        "role": "distributional sensitivity; does not replace the primary LMM",
        "n_trial": int(work.shape[0]),
        "n_mice": int(work[mouse_col].nunique()),
        "reference_group": reference_group,
        "comparison_group": comparison_group,
        "time_center": center,
        "continuous_covariates": tuple(continuous_covariates),
        "converged": bool(getattr(fit, "converged", True)),
        "working_correlation": np.asarray(fit.cov_struct.dep_params).tolist(),
    }
    return GEESensitivityResult(
        contrasts=contrasts,
        coefficients=pd.DataFrame(coefficient_rows),
        diagnostics=diagnostics,
    )
