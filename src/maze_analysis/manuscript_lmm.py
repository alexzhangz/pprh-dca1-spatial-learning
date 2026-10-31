"""Manuscript definitions collected from manuscript_lmm.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import json

import math

import warnings

from dataclasses import dataclass, field

from typing import Any, Sequence

import numpy as np

import pandas as pd

from scipy import stats

import statsmodels.formula.api as smf

from .stats import benjamini_hochberg

OPTIMIZER_POLICY = (
    ("Powell", "powell"),
    ("BFGS", "bfgs"),
    ("L-BFGS", "lbfgs"),
)

_VALUE = "_lmm_value"

_TIME = "_lmm_time_centered"

_GROUP = "_lmm_group_comparison"

_MOUSE = "_lmm_mouse"

_INTERACTION = f"{_TIME}:{_GROUP}"

_NORMAL_975 = float(stats.norm.ppf(0.975))

class _DefaultSamplingUnitColumns:
    """Sentinel resolving to the active mouse/time column names."""

_DEFAULT_SAMPLING_UNIT_COLS = _DefaultSamplingUnitColumns()

@dataclass
class LongitudinalLMMResult:
    """Container for one longitudinal random-intercept LMM fit."""

    contrasts: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: dict[str, Any]
    fit_result: Any = field(default=None, repr=False)

    @property
    def succeeded(self) -> bool:
        """Whether an acceptable MixedLM fit was obtained."""

        return self.fit_result is not None and self.diagnostics.get("model_status") == "ok"

def _as_tuple(columns: Sequence[str] | str | None) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    return tuple(columns)

def _resolve_sampling_unit_cols(
    sampling_unit_cols: Sequence[str] | str | None | _DefaultSamplingUnitColumns,
    *,
    mouse_col: str,
    time_col: str,
) -> tuple[str, ...] | None:
    if sampling_unit_cols is _DEFAULT_SAMPLING_UNIT_COLS:
        return (mouse_col, time_col)
    if sampling_unit_cols is None:
        return None
    resolved = _as_tuple(sampling_unit_cols)
    if not resolved:
        raise ValueError(
            "sampling_unit_cols must contain at least one column; use None to "
            "disable the uniqueness check explicitly"
        )
    duplicates = sorted(
        {column for column in resolved if resolved.count(column) > 1}
    )
    if duplicates:
        raise ValueError(f"sampling_unit_cols contains duplicates: {duplicates}")
    return resolved

def _quoted_column(column: str) -> str:
    return f"Q({json.dumps(str(column), ensure_ascii=True)})"

def _display_formula(
    *,
    value_col: str,
    time_col: str,
    group_col: str,
    reference_group: str,
    time_center: float,
    continuous_covariates: Sequence[str],
) -> str:
    time_term = f"I({_quoted_column(time_col)} - {time_center:.12g})"
    group_term = (
        f"C({_quoted_column(group_col)}, "
        f"Treatment(reference={json.dumps(str(reference_group))}))"
    )
    formula = f"{_quoted_column(value_col)} ~ {time_term} * {group_term}"
    if continuous_covariates:
        formula += " + " + " + ".join(
            _quoted_column(column) for column in continuous_covariates
        )
    return formula

def _internal_formula(n_covariates: int) -> str:
    formula = f"{_VALUE} ~ {_TIME} * {_GROUP}"
    if n_covariates:
        formula += " + " + " + ".join(
            f"_lmm_covariate_{index}" for index in range(n_covariates)
        )
    return formula

def _validate_column_contract(
    table: pd.DataFrame,
    *,
    value_col: str,
    time_col: str,
    group_col: str,
    mouse_col: str,
    continuous_covariates: Sequence[str],
) -> None:
    named_columns = (value_col, time_col, group_col, mouse_col) + tuple(
        continuous_covariates
    )
    missing = sorted(set(named_columns).difference(table.columns))
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    duplicate_covariates = sorted(
        {
            column
            for column in continuous_covariates
            if continuous_covariates.count(column) > 1
        }
    )
    if duplicate_covariates:
        raise ValueError(
            f"continuous_covariates contains duplicates: {duplicate_covariates}"
        )
    reserved_collisions = sorted(
        set(continuous_covariates).intersection(
            {value_col, time_col, group_col, mouse_col}
        )
    )
    if reserved_collisions:
        raise ValueError(
            "continuous covariates must differ from the outcome, time, group, "
            f"and mouse columns: {reserved_collisions}"
        )

def _prepare_model_table(
    table: pd.DataFrame,
    *,
    value_col: str,
    time_col: str,
    group_col: str,
    mouse_col: str,
    reference_group: str,
    comparison_group: str,
    continuous_covariates: Sequence[str],
    time_center: float,
) -> tuple[pd.DataFrame, int]:
    model_table = table.loc[
        table[group_col].astype(str).isin(
            [str(reference_group), str(comparison_group)]
        )
    ].copy()
    n_candidate = int(model_table.shape[0])
    model_table[_VALUE] = pd.to_numeric(model_table[value_col], errors="coerce")
    numeric_columns = [_VALUE]
    time_numeric = pd.to_numeric(model_table[time_col], errors="coerce")
    model_table[_TIME] = time_numeric - float(time_center)
    numeric_columns.append(_TIME)
    for index, column in enumerate(continuous_covariates):
        alias = f"_lmm_covariate_{index}"
        model_table[alias] = pd.to_numeric(model_table[column], errors="coerce")
        numeric_columns.append(alias)

    model_table[_MOUSE] = model_table[mouse_col]
    model_table[_GROUP] = (
        model_table[group_col].astype(str) == str(comparison_group)
    ).astype(float)
    complete = model_table[_MOUSE].notna()
    for column in numeric_columns:
        complete &= np.isfinite(model_table[column].to_numpy(dtype=float))
    model_table = model_table.loc[complete].copy()
    model_table["_lmm_original_order"] = np.arange(model_table.shape[0])
    model_table = model_table.sort_values(
        [_MOUSE, _TIME, "_lmm_original_order"], kind="mergesort"
    ).reset_index(drop=True)

    groups_per_mouse = model_table.groupby(_MOUSE, observed=True)[_GROUP].nunique()
    if (groups_per_mouse > 1).any():
        collided = groups_per_mouse.index[groups_per_mouse > 1].tolist()
        raise ValueError(f"mouse identifiers occur in both groups: {collided}")
    available = set(model_table[_GROUP].unique())
    if not {0.0, 1.0}.issubset(available):
        raise ValueError(
            "both reference and comparison groups require at least one complete case"
        )
    if model_table[_MOUSE].nunique() < 2:
        raise ValueError("MixedLM requires at least two mice")
    return model_table, n_candidate

def _validate_sampling_unit_uniqueness(
    model_table: pd.DataFrame,
    *,
    sampling_unit_cols: Sequence[str] | None,
    mouse_col: str,
    time_col: str,
    group_col: str,
) -> None:
    if sampling_unit_cols is None:
        return
    missing = sorted(set(sampling_unit_cols).difference(model_table.columns))
    if missing:
        raise ValueError(f"table is missing sampling-unit columns: {missing}")

    normalized_columns = []
    for column in sampling_unit_cols:
        if column == mouse_col:
            normalized_columns.append(_MOUSE)
        elif column == time_col:
            normalized_columns.append(_TIME)
        elif column == group_col:
            normalized_columns.append(_GROUP)
        else:
            normalized_columns.append(column)
    null_columns = [
        original
        for original, normalized in zip(sampling_unit_cols, normalized_columns)
        if model_table[normalized].isna().any()
    ]
    if null_columns:
        raise ValueError(
            f"sampling-unit columns contain missing values: {null_columns}"
        )
    duplicate = model_table.duplicated(normalized_columns, keep=False)
    if duplicate.any():
        examples = (
            model_table.loc[duplicate, list(sampling_unit_cols)]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "duplicate sampling units for columns "
            f"{tuple(sampling_unit_cols)}; examples={examples}"
        )

def _common_complete_case_table(
    table: pd.DataFrame,
    *,
    value_col: str,
    time_col: str,
    group_col: str,
    mouse_col: str,
    reference_group: str,
    comparison_group: str,
    continuous_covariates: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_column_contract(
        table,
        value_col=value_col,
        time_col=time_col,
        group_col=group_col,
        mouse_col=mouse_col,
        continuous_covariates=continuous_covariates,
    )
    selected_group = table[group_col].astype(str).isin(
        [str(reference_group), str(comparison_group)]
    )
    candidate = table.loc[selected_group].copy()
    complete = candidate[group_col].notna() & candidate[mouse_col].notna()
    numeric_columns = (value_col, time_col) + tuple(continuous_covariates)
    for column in numeric_columns:
        numeric = pd.to_numeric(candidate[column], errors="coerce")
        complete &= np.isfinite(numeric.to_numpy(dtype=float))
    common = candidate.loc[complete].copy()
    metadata = {
        "common_complete_case": True,
        "common_complete_case_columns": (
            value_col,
            time_col,
            group_col,
            mouse_col,
        )
        + tuple(continuous_covariates),
        "common_complete_case_n_candidate": int(candidate.shape[0]),
        "common_complete_case_nobs": int(common.shape[0]),
        "common_complete_case_n_dropped": int(candidate.shape[0] - common.shape[0]),
    }
    return common, metadata

def _fit_covariances(fit: Any) -> tuple[np.ndarray, np.ndarray]:
    fe_names = list(fit.fe_params.index)
    fixed_covariance = (
        fit.cov_params().loc[fe_names, fe_names].to_numpy(dtype=float)
    )
    random_covariance = np.asarray(fit.cov_re, dtype=float)
    return fixed_covariance, random_covariance

def _fit_is_acceptable(fit: Any) -> tuple[bool, bool]:
    converged = bool(getattr(fit, "converged", False))
    try:
        fixed_covariance, random_covariance = _fit_covariances(fit)
        params_finite = np.isfinite(
            np.asarray(fit.fe_params, dtype=float)
        ).all()
        covariance_finite = bool(
            fixed_covariance.size
            and random_covariance.size
            and np.isfinite(fixed_covariance).all()
            and np.isfinite(random_covariance).all()
            and params_finite
        )
    except Exception:
        covariance_finite = False
    return converged, covariance_finite

def _fit_one_optimizer(
    model_table: pd.DataFrame,
    formula: str,
    *,
    method: str,
    maxiter: int,
) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            formula,
            data=model_table,
            groups=model_table[_MOUSE],
            re_formula="1",
        )
        fit = model.fit(
            reml=True,
            method=method,
            maxiter=int(maxiter),
            disp=False,
        )
    return fit, [str(item.message) for item in caught]

def _fit_with_optimizer_policy(
    model_table: pd.DataFrame,
    formula: str,
    *,
    maxiter: int,
) -> tuple[Any | None, str | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for optimizer, method in OPTIMIZER_POLICY:
        try:
            fit, warning_messages = _fit_one_optimizer(
                model_table,
                formula,
                method=method,
                maxiter=maxiter,
            )
            converged, covariance_finite = _fit_is_acceptable(fit)
            accepted = bool(converged and covariance_finite)
            attempts.append(
                {
                    "optimizer": optimizer,
                    "method": method,
                    "converged": converged,
                    "covariance_finite": covariance_finite,
                    "accepted": accepted,
                    "warnings": warning_messages,
                    "error": None,
                }
            )
            if accepted:
                return fit, optimizer, attempts
        except Exception as error:
            attempts.append(
                {
                    "optimizer": optimizer,
                    "method": method,
                    "converged": False,
                    "covariance_finite": False,
                    "accepted": False,
                    "warnings": [],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return None, None, attempts

def _wald_values(
    params: np.ndarray,
    covariance: np.ndarray,
    vector: np.ndarray,
) -> dict[str, float]:
    beta = float(vector @ params)
    variance = float(vector @ covariance @ vector)
    if variance < 0 and abs(variance) <= 1e-12:
        variance = 0.0
    standard_error = math.sqrt(variance) if variance >= 0 else np.nan
    z_value = beta / standard_error if standard_error > 0 else np.nan
    p_value = (
        float(2.0 * stats.norm.sf(abs(z_value)))
        if np.isfinite(z_value)
        else np.nan
    )
    return {
        "beta": beta,
        "standard_error": float(standard_error),
        "z_value": float(z_value),
        "ci95_low": float(beta - _NORMAL_975 * standard_error),
        "ci95_high": float(beta + _NORMAL_975 * standard_error),
        "p_value": p_value,
    }

def _nan_wald_values() -> dict[str, float]:
    return {
        "beta": np.nan,
        "standard_error": np.nan,
        "z_value": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "p_value": np.nan,
    }

def _time_label(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return f"D{int(value)}"
    return f"D{value:g}"

def _contrast_definitions(
    *,
    fe_names: Sequence[str],
    reference_group: str,
    comparison_group: str,
    first_time: float,
    last_time: float,
    time_center: float,
) -> list[dict[str, Any]]:
    index = {name: position for position, name in enumerate(fe_names)}
    required = {_TIME, _GROUP, _INTERACTION}
    missing = sorted(required.difference(index))
    if missing:
        raise RuntimeError(f"MixedLM is missing required fixed effects: {missing}")

    def vector(**weights: float) -> np.ndarray:
        result = np.zeros(len(fe_names), dtype=float)
        for term, weight in weights.items():
            result[index[term]] = float(weight)
        return result

    first_delta = float(first_time) - float(time_center)
    last_delta = float(last_time) - float(time_center)
    return [
        {
            "contrast": "reference_group_day_slope",
            "estimand": f"day slope within {reference_group}",
            "group": reference_group,
            "time": np.nan,
            "vector": vector(**{_TIME: 1.0}),
        },
        {
            "contrast": "comparison_group_day_slope",
            "estimand": f"day slope within {comparison_group}",
            "group": comparison_group,
            "time": np.nan,
            "vector": vector(**{_TIME: 1.0, _INTERACTION: 1.0}),
        },
        {
            "contrast": "group_by_day_interaction",
            "estimand": f"{comparison_group} minus {reference_group} day slope",
            "group": np.nan,
            "time": np.nan,
            "vector": vector(**{_INTERACTION: 1.0}),
        },
        {
            "contrast": "centered_overall_group_effect",
            "estimand": (
                f"{comparison_group} minus {reference_group} at centered day "
                f"{time_center:g}"
            ),
            "group": np.nan,
            "time": float(time_center),
            "vector": vector(**{_GROUP: 1.0}),
        },
        {
            "contrast": f"between_group_{_time_label(first_time)}",
            "estimand": (
                f"{comparison_group} minus {reference_group} at "
                f"{_time_label(first_time)}"
            ),
            "group": np.nan,
            "time": float(first_time),
            "vector": vector(
                **{_GROUP: 1.0, _INTERACTION: first_delta}
            ),
        },
        {
            "contrast": f"between_group_{_time_label(last_time)}",
            "estimand": (
                f"{comparison_group} minus {reference_group} at "
                f"{_time_label(last_time)}"
            ),
            "group": np.nan,
            "time": float(last_time),
            "vector": vector(
                **{_GROUP: 1.0, _INTERACTION: last_delta}
            ),
        },
    ]

def _build_contrast_table(
    *,
    fit: Any | None,
    optimizer: str | None,
    model_status: str,
    display_formula: str,
    adjusted: bool,
    continuous_covariates: Sequence[str],
    reference_group: str,
    comparison_group: str,
    first_time: float,
    last_time: float,
    time_center: float,
    nobs: int,
    n_mice: int,
) -> pd.DataFrame:
    if fit is not None:
        fe_names = list(fit.fe_params.index)
    else:
        fe_names = ["Intercept", _TIME, _GROUP, _INTERACTION]
    definitions = _contrast_definitions(
        fe_names=fe_names,
        reference_group=reference_group,
        comparison_group=comparison_group,
        first_time=first_time,
        last_time=last_time,
        time_center=time_center,
    )

    params = covariance = None
    if fit is not None:
        params = fit.fe_params.to_numpy(dtype=float)
        covariance, _ = _fit_covariances(fit)
    rows = []
    for definition in definitions:
        inference = (
            _wald_values(params, covariance, definition["vector"])
            if fit is not None
            else _nan_wald_values()
        )
        rows.append(
            {
                "contrast": definition["contrast"],
                "estimand": definition["estimand"],
                "reference_group": reference_group,
                "comparison_group": comparison_group,
                "group": definition["group"],
                "time": definition["time"],
                **inference,
                "test": "MixedLM Wald z-test",
                "adjusted": bool(adjusted),
                "continuous_covariates": tuple(continuous_covariates),
                "model_formula": display_formula,
                "model_status": model_status,
                "optimizer": optimizer,
                "nobs": int(nobs),
                "n_mice": int(n_mice),
                "fallback_used": False,
            }
        )
    return pd.DataFrame(rows)

def _build_coefficient_table(
    fit: Any | None,
    *,
    continuous_covariates: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "term",
        "internal_term",
        "beta",
        "standard_error",
        "z_value",
        "ci95_low",
        "ci95_high",
        "p_value",
    ]
    if fit is None:
        return pd.DataFrame(columns=columns)
    fe_names = list(fit.fe_params.index)
    params = fit.fe_params.to_numpy(dtype=float)
    covariance, _ = _fit_covariances(fit)
    labels = {
        "Intercept": "intercept_reference_at_centered_time",
        _TIME: "reference_group_day_slope",
        _GROUP: "centered_overall_group_effect",
        _INTERACTION: "group_by_day_interaction",
    }
    labels.update(
        {
            f"_lmm_covariate_{index}": f"covariate::{column}"
            for index, column in enumerate(continuous_covariates)
        }
    )
    rows = []
    for index, internal_term in enumerate(fe_names):
        vector = np.zeros(len(fe_names), dtype=float)
        vector[index] = 1.0
        rows.append(
            {
                "term": labels.get(internal_term, internal_term),
                "internal_term": internal_term,
                **_wald_values(params, covariance, vector),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def fit_longitudinal_lmm(
    table: pd.DataFrame,
    *,
    value_col: str = "value",
    time_col: str = "day",
    group_col: str = "group",
    mouse_col: str = "mouse",
    reference_group: str = "C",
    comparison_group: str = "I",
    continuous_covariates: Sequence[str] | str | None = None,
    sampling_unit_cols: (
        Sequence[str] | str | None | _DefaultSamplingUnitColumns
    ) = _DEFAULT_SAMPLING_UNIT_COLS,
    first_time: float | None = 1,
    last_time: float | None = 5,
    boundary_tolerance: float = 1e-8,
    maxiter: int = 1000,
) -> LongitudinalLMMResult:
    """Fit the locked longitudinal random-intercept LMM.

    The fitted model is ``value ~ centered_time * group + covariates`` with
    ``reference_group`` coded as zero and a random intercept for ``mouse``.
    All six manuscript estimands are Wald contrasts from this same fit.

    Continuous covariates are used as supplied; scaling or standardization is
    intentionally the caller's responsibility and must therefore be explicit
    in the source-data column names.  By default, one observation is required
    per ``(mouse_col, time_col)`` sampling unit.  Callers must explicitly pass
    ``sampling_unit_cols=None`` for data whose declared unit is not mouse-day.
    """

    continuous_covariates = _as_tuple(continuous_covariates)
    resolved_sampling_unit_cols = _resolve_sampling_unit_cols(
        sampling_unit_cols,
        mouse_col=mouse_col,
        time_col=time_col,
    )
    _validate_column_contract(
        table,
        value_col=value_col,
        time_col=time_col,
        group_col=group_col,
        mouse_col=mouse_col,
        continuous_covariates=continuous_covariates,
    )
    if str(reference_group) == str(comparison_group):
        raise ValueError("reference_group and comparison_group must differ")
    if first_time is None:
        first_time = float(pd.to_numeric(table[time_col], errors="coerce").min())
    if last_time is None:
        last_time = float(pd.to_numeric(table[time_col], errors="coerce").max())
    first_time = float(first_time)
    last_time = float(last_time)
    if not np.isfinite(first_time) or not np.isfinite(last_time):
        raise ValueError("first_time and last_time must be finite")
    if first_time >= last_time:
        raise ValueError("first_time must be smaller than last_time")
    time_center = (first_time + last_time) / 2.0

    model_table, n_candidate = _prepare_model_table(
        table,
        value_col=value_col,
        time_col=time_col,
        group_col=group_col,
        mouse_col=mouse_col,
        reference_group=reference_group,
        comparison_group=comparison_group,
        continuous_covariates=continuous_covariates,
        time_center=time_center,
    )
    _validate_sampling_unit_uniqueness(
        model_table,
        sampling_unit_cols=resolved_sampling_unit_cols,
        mouse_col=mouse_col,
        time_col=time_col,
        group_col=group_col,
    )
    internal_formula = _internal_formula(len(continuous_covariates))
    display_formula = _display_formula(
        value_col=value_col,
        time_col=time_col,
        group_col=group_col,
        reference_group=reference_group,
        time_center=time_center,
        continuous_covariates=continuous_covariates,
    )
    fit, optimizer, attempts = _fit_with_optimizer_policy(
        model_table,
        internal_formula,
        maxiter=maxiter,
    )

    nobs = int(model_table.shape[0])
    n_mice = int(model_table[_MOUSE].nunique())
    reference_mice = int(
        model_table.loc[model_table[_GROUP] == 0.0, _MOUSE].nunique()
    )
    comparison_mice = int(
        model_table.loc[model_table[_GROUP] == 1.0, _MOUSE].nunique()
    )
    selected_attempt = attempts[-1] if attempts else {}
    if fit is not None:
        selected_attempt = next(
            attempt for attempt in attempts if attempt["accepted"]
        )
        random_intercept_variance = float(np.asarray(fit.cov_re)[0, 0])
        selected_warnings = selected_attempt.get("warnings", [])
        boundary = bool(
            random_intercept_variance <= float(boundary_tolerance)
            or any("boundary" in message.lower() for message in selected_warnings)
        )
        covariance_finite = True
        converged = True
        model_status = "ok"
        failure_reason = None
    else:
        random_intercept_variance = np.nan
        boundary = False
        covariance_finite = bool(selected_attempt.get("covariance_finite", False))
        converged = bool(selected_attempt.get("converged", False))
        model_status = "failed"
        failure_reason = "all MixedLM optimizers failed convergence/covariance gate"

    diagnostics = {
        "model": "MixedLM random intercept by mouse",
        "model_status": model_status,
        "optimizer": optimizer,
        "optimizer_policy": tuple(name for name, _ in OPTIMIZER_POLICY),
        "optimizer_attempts": attempts,
        "converged": converged,
        "covariance_finite": covariance_finite,
        "random_intercept_variance": random_intercept_variance,
        "boundary_random_intercept": boundary,
        "boundary_tolerance": float(boundary_tolerance),
        "formula": display_formula,
        "internal_formula": internal_formula,
        "reml": True,
        "nobs": nobs,
        "n_candidate_rows": n_candidate,
        "n_dropped_rows": n_candidate - nobs,
        "n_mice": n_mice,
        "n_mice_reference": reference_mice,
        "n_mice_comparison": comparison_mice,
        "mice": tuple(sorted(model_table[_MOUSE].astype(str).unique())),
        "reference_group": str(reference_group),
        "comparison_group": str(comparison_group),
        "time_center": time_center,
        "first_time": first_time,
        "last_time": last_time,
        "continuous_covariates": tuple(continuous_covariates),
        "adjusted": bool(continuous_covariates),
        "sampling_unit_cols": resolved_sampling_unit_cols,
        "sampling_unit_uniqueness_checked": resolved_sampling_unit_cols is not None,
        "fallback_used": False,
        "failure_reason": failure_reason,
    }
    contrasts = _build_contrast_table(
        fit=fit,
        optimizer=optimizer,
        model_status=model_status,
        display_formula=display_formula,
        adjusted=bool(continuous_covariates),
        continuous_covariates=continuous_covariates,
        reference_group=str(reference_group),
        comparison_group=str(comparison_group),
        first_time=first_time,
        last_time=last_time,
        time_center=time_center,
        nobs=nobs,
        n_mice=n_mice,
    )
    coefficients = _build_coefficient_table(
        fit,
        continuous_covariates=continuous_covariates,
    )
    return LongitudinalLMMResult(
        contrasts=contrasts,
        coefficients=coefficients,
        diagnostics=diagnostics,
        fit_result=fit,
    )

def fit_adjusted_unadjusted_lmm(
    table: pd.DataFrame,
    *,
    continuous_covariates: Sequence[str] | str,
    **kwargs: Any,
) -> dict[str, LongitudinalLMMResult]:
    """Fit adjusted and unadjusted LMMs on one common complete-case sample."""

    covariates = _as_tuple(continuous_covariates)
    if not covariates:
        raise ValueError("continuous_covariates must be non-empty for paired fits")
    value_col = kwargs.get("value_col", "value")
    time_col = kwargs.get("time_col", "day")
    group_col = kwargs.get("group_col", "group")
    mouse_col = kwargs.get("mouse_col", "mouse")
    reference_group = kwargs.get("reference_group", "C")
    comparison_group = kwargs.get("comparison_group", "I")
    common_table, common_metadata = _common_complete_case_table(
        table,
        value_col=value_col,
        time_col=time_col,
        group_col=group_col,
        mouse_col=mouse_col,
        reference_group=reference_group,
        comparison_group=comparison_group,
        continuous_covariates=covariates,
    )
    adjusted = fit_longitudinal_lmm(
        common_table,
        continuous_covariates=covariates,
        **kwargs,
    )
    unadjusted = fit_longitudinal_lmm(
        common_table,
        continuous_covariates=(),
        **kwargs,
    )
    for role, result in (("adjusted", adjusted), ("unadjusted", unadjusted)):
        result.diagnostics.update(common_metadata)
        result.diagnostics["paired_fit_role"] = role
    return {"adjusted": adjusted, "unadjusted": unadjusted}

def apply_bh_correction(
    table: pd.DataFrame,
    *,
    p_col: str = "p_value",
    family_cols: Sequence[str] | str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply the project's existing Benjamini-Hochberg helper by family."""

    if p_col not in table.columns:
        raise ValueError(f"table is missing p-value column {p_col!r}")
    family_cols = _as_tuple(family_cols)
    missing = sorted(set(family_cols).difference(table.columns))
    if missing:
        raise ValueError(f"table is missing BH family columns: {missing}")
    result = table.copy()
    result["q_value"] = np.nan
    result["significant"] = False
    if result.empty:
        return result

    if family_cols:
        grouper = family_cols[0] if len(family_cols) == 1 else list(family_cols)
        grouped_indices = (
            group.index
            for _, group in result.groupby(
                grouper,
                observed=True,
                dropna=False,
                sort=False,
            )
        )
    else:
        grouped_indices = (result.index,)
    for index in grouped_indices:
        q_values, significant = benjamini_hochberg(
            result.loc[index, p_col].to_numpy(dtype=float),
            alpha=alpha,
        )
        result.loc[index, "q_value"] = q_values
        result.loc[index, "significant"] = significant
    return result
