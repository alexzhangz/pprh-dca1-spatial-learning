"""Manuscript definitions collected from finalize_fig3e_gee_diagnostics.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

from scipy import stats

from statsmodels.genmod.cov_struct import Exchangeable, Independence

from statsmodels.genmod.families import Binomial

from statsmodels.genmod.generalized_estimating_equations import GEE

TERMS = ("intercept", "trial_c10", "group_I", "trial_c10_x_group_I")

def prepare_data(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    required = {"group", "mouse", "day", "cumulative_trial", "correct"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    work = raw.loc[
        raw["group"].isin(["C", "I"])
        & pd.to_numeric(raw["cumulative_trial"], errors="coerce").between(1, 90)
    ].copy()
    for column in ("day", "cumulative_trial", "correct"):
        work[column] = pd.to_numeric(work[column], errors="raise")
    if work[["group", "mouse", "cumulative_trial", "correct"]].isna().any().any():
        raise ValueError("missing values in required analysis columns")
    work["cumulative_trial"] = work["cumulative_trial"].astype(int)
    work["correct"] = work["correct"].astype(int)
    if not set(work["correct"].unique()).issubset({0, 1}):
        raise ValueError("correct is not binary")
    if work.duplicated(["mouse", "cumulative_trial"]).any():
        raise ValueError("duplicate mouse-by-cumulative-trial rows")
    work = work.sort_values(["mouse", "cumulative_trial"], kind="stable").reset_index(drop=True)

    support = (
        work.groupby(["group", "mouse"], observed=True)["cumulative_trial"]
        .agg(n="size", unique="nunique", first="min", last="max")
        .reset_index()
    )
    group_counts = support.groupby("group", observed=True)["mouse"].nunique().to_dict()
    support_ok = bool(
        work.shape[0] == 1080
        and support.shape[0] == 12
        and group_counts == {"C": 5, "I": 7}
        and (
            support["n"].eq(90)
            & support["unique"].eq(90)
            & support["first"].eq(1)
            & support["last"].eq(90)
        ).all()
    )
    if not support_ok:
        raise ValueError(f"locked first-90-trial support failed: {support.to_dict('records')}")

    work["trial_c10"] = (work["cumulative_trial"] - 45.5) / 10.0
    work["group_I"] = work["group"].eq("I").astype(float)
    work["trial_c10_x_group_I"] = work["trial_c10"] * work["group_I"]
    work["intercept"] = 1.0
    contract = {
        "passed": support_ok,
        "n_binary_trials": int(work.shape[0]),
        "n_mice": int(work["mouse"].nunique()),
        "n_control_mice": int(work.loc[work["group"].eq("C"), "mouse"].nunique()),
        "n_inhibition_mice": int(work.loc[work["group"].eq("I"), "mouse"].nunique()),
        "trials_per_mouse": sorted(work.groupby("mouse").size().unique().astype(int).tolist()),
        "trial_range": [int(work["cumulative_trial"].min()), int(work["cumulative_trial"].max())],
        "missing_required_values": 0,
        "duplicate_mouse_trial_rows": 0,
    }
    return work, contract

def fit_gee(
    data: pd.DataFrame,
    *,
    covariance_type: str = "bias_reduced",
    working: str = "exchangeable",
) -> tuple[Any, np.ndarray, np.ndarray]:
    design = data.loc[:, TERMS].to_numpy(dtype=float)
    covariance_structure = Exchangeable() if working == "exchangeable" else Independence()
    model = GEE(
        data["correct"].to_numpy(dtype=float),
        design,
        groups=data["mouse"].astype(str).to_numpy(),
        family=Binomial(),
        cov_struct=covariance_structure,
    )
    fit = model.fit(maxiter=200, ctol=1e-10, cov_type=covariance_type)
    return fit, np.asarray(fit.params, dtype=float), np.asarray(fit.cov_params(), dtype=float)

def contrast_result(
    label: str,
    vector: np.ndarray,
    params: np.ndarray,
    covariance: np.ndarray,
    *,
    df: int,
) -> dict[str, Any]:
    beta = float(vector @ params)
    variance = float(vector @ covariance @ vector)
    if variance < 0 and abs(variance) < 1e-12:
        variance = 0.0
    se = float(np.sqrt(variance)) if variance >= 0 else float("nan")
    t_value = float(beta / se) if se > 0 else float("nan")
    critical = float(stats.t.ppf(0.975, df=df))
    return {
        "contrast": label,
        "beta": beta,
        "standard_error": se,
        "t_value": t_value,
        "df": int(df),
        "p_value": float(2 * stats.t.sf(abs(t_value), df=df)),
        "ci95_low": float(beta - critical * se),
        "ci95_high": float(beta + critical * se),
        "odds_ratio": float(np.exp(beta)),
        "odds_ratio_ci95_low": float(np.exp(beta - critical * se)),
        "odds_ratio_ci95_high": float(np.exp(beta + critical * se)),
    }

def all_estimands(params: np.ndarray, covariance: np.ndarray, *, df: int) -> pd.DataFrame:
    first_delta = (1.0 - 45.5) / 10.0
    last_delta = (90.0 - 45.5) / 10.0
    definitions = (
        ("control_trial_slope_per_10_trials", np.array([0.0, 1.0, 0.0, 0.0])),
        ("inhibition_trial_slope_per_10_trials", np.array([0.0, 1.0, 0.0, 1.0])),
        ("group_by_trial_interaction_I_minus_C", np.array([0.0, 0.0, 0.0, 1.0])),
        ("inhibition_minus_control_at_trial_45_5", np.array([0.0, 0.0, 1.0, 0.0])),
        ("inhibition_minus_control_at_trial_1", np.array([0.0, 0.0, 1.0, first_delta])),
        ("inhibition_minus_control_at_trial_90", np.array([0.0, 0.0, 1.0, last_delta])),
    )
    return pd.DataFrame(
        [contrast_result(label, vector, params, covariance, df=df) for label, vector in definitions]
    )

def covariance_diagnostics(covariance: np.ndarray) -> dict[str, Any]:
    symmetric = bool(np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12))
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    finite = bool(np.isfinite(covariance).all())
    positive_definite = bool(finite and eigenvalues.min() > 0)
    return {
        "finite": finite,
        "symmetric": symmetric,
        "positive_definite": positive_definite,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(np.linalg.cond(covariance)),
    }
