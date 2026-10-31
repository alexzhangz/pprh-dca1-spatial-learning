"""Manuscript definitions collected from build_manuscript_fig3_behavior.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from maze_analysis.behavior_covariates import (  # noqa: E402
    CANONICAL_COVARIATE_COLUMNS,
    pooled_z_standardize_covariates,
)

from maze_analysis.fig3_8maze_behavior import (  # noqa: E402
    build_fig3_8maze_behavior_tables,
)

from maze_analysis.fig3_behavior_plotting import (  # noqa: E402
    plot_fig3d,
    plot_fig3e,
    plot_fig3h,
    save_panel_bundle,
)

from maze_analysis.fig3_tmaze_behavior import (  # noqa: E402
    build_fig3h_tmaze_behavior,
)

from maze_analysis.manuscript_cohort import (  # noqa: E402
    DEFAULT_MANUSCRIPT_COHORT_PATH,
    cohort_count_label,
    group_mice,
    load_manuscript_cohort,
)

DAYS = (1, 2, 3, 4, 5)

GROUP_ORDER = ("C", "I")

COVARIATE_Z_COLUMNS = (
    "mean_speed_z",
    "route_relative_lateral_deviation_z",
)

COVARIATE_OUTPUT_NAMES = {
    "mean_speed_cm_s": COVARIATE_Z_COLUMNS[0],
    "route_relative_lateral_deviation_cm": COVARIATE_Z_COLUMNS[1],
}

def validate_mouse_day_table(
    table: pd.DataFrame,
    *,
    group_mice: Mapping[str, Sequence[str]],
    value_col: str,
    label: str,
) -> None:
    expected = {
        (str(group), str(mouse), int(day))
        for group, mice in group_mice.items()
        for mouse in mice
        for day in DAYS
    }
    observed = set(
        table.loc[:, ["group", "mouse", "day"]]
        .assign(
            group=lambda x: x["group"].astype(str),
            mouse=lambda x: x["mouse"].astype(str),
            day=lambda x: pd.to_numeric(x["day"], errors="raise").astype(int),
        )
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise RuntimeError(
            f"{label}: locked mouse-day set mismatch; "
            f"missing={sorted(expected-observed)[:10]}, extra={sorted(observed-expected)[:10]}"
        )
    duplicate = table.duplicated(["group", "mouse", "day"], keep=False)
    if duplicate.any():
        raise RuntimeError(f"{label}: duplicate mouse-day rows detected")
    required_finite = [
        value_col,
        "mean_speed_cm_s",
        "route_relative_lateral_deviation_cm",
    ]
    for column in required_finite:
        if not np.isfinite(pd.to_numeric(table[column], errors="coerce")).all():
            bad = table.loc[
                ~np.isfinite(pd.to_numeric(table[column], errors="coerce")),
                ["group", "mouse", "day"],
            ]
            raise RuntimeError(f"{label}: nonfinite {column}: {bad.to_dict('records')}")

def validate_trial_table(
    trial: pd.DataFrame,
    mouse_day: pd.DataFrame,
    *,
    group_mice: Mapping[str, Sequence[str]],
    label: str,
) -> None:
    required = {"group", "mouse", "day", "correct"}
    missing = sorted(required.difference(trial.columns))
    if missing:
        raise RuntimeError(f"{label}: trial table is missing {missing}")
    if trial.empty or not set(trial["correct"].dropna().astype(int).unique()).issubset({0, 1}):
        raise RuntimeError(f"{label}: trial outcomes are not nonempty binary values")
    expected_pairs = {
        (str(group), str(mouse))
        for group, mice in group_mice.items()
        for mouse in mice
    }
    observed_pairs = set(
        trial[["group", "mouse"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            f"{label}: trial mouse set mismatch; "
            f"missing={sorted(expected_pairs-observed_pairs)}, "
            f"extra={sorted(observed_pairs-expected_pairs)}"
        )
    if not set(pd.to_numeric(trial["day"], errors="raise").astype(int)).issubset(DAYS):
        raise RuntimeError(f"{label}: trial table contains days outside D1-D5")
    regrouped = (
        trial.groupby(["group", "mouse", "day"], observed=True)["correct"]
        .agg(n_valid="size", n_correct="sum", success_rate="mean")
        .reset_index()
    )
    compare = mouse_day.merge(
        regrouped,
        on=["group", "mouse", "day"],
        how="outer",
        validate="one_to_one",
        suffixes=("_mouse_day", "_trial"),
        indicator=True,
    )
    if not compare["_merge"].eq("both").all():
        raise RuntimeError(f"{label}: trial-to-mouse-day support mismatch")
    if not np.allclose(
        compare["success_rate_mouse_day"],
        compare["success_rate_trial"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{label}: trial aggregation does not reproduce success rate")

def standardize_project_covariates(
    table: pd.DataFrame,
    *,
    project: str,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standardized, qc = pooled_z_standardize_covariates(
        table,
        columns=CANONICAL_COVARIATE_COLUMNS,
        output_columns=COVARIATE_OUTPUT_NAMES,
        ddof=1,
    )
    qc.insert(0, "project", project)
    qc.insert(1, "cohort", cohort)
    if not qc["status"].eq("ok").all():
        raise RuntimeError(f"{project}/{cohort}: covariate z-score reference failed")
    for column in COVARIATE_Z_COLUMNS:
        if not np.isfinite(standardized[column]).all():
            raise RuntimeError(f"{project}/{cohort}: nonfinite standardized {column}")
    return standardized, qc

def build_figure8_tables(
    aligned_root: Path,
    group_mice: Mapping[str, Sequence[str]],
) -> dict[str, pd.DataFrame]:
    atomic = build_fig3_8maze_behavior_tables(
        aligned_root,
        group_mice=group_mice,
        days=DAYS,
    )
    covariate_columns = [
        "group",
        "mouse",
        "day",
        "mean_speed_cm_s",
        "route_relative_lateral_deviation_cm",
        "covariate_raw_frame_count",
        "covariate_running_frame_count",
        "covariate_valid_frame_count",
        "covariate_speed_frame_count",
        "covariate_lateral_frame_count",
        "covariate_status",
        "missing_reason",
    ]
    covariates = atomic["covariates_mouse_day"].loc[:, covariate_columns].rename(
        columns={"missing_reason": "covariate_missing_reason"}
    )
    mouse_day = atomic["success_mouse_day"].rename(
        columns={"missing_reason": "behavior_missing_reason"}
    ).merge(
        covariates,
        on=["group", "mouse", "day"],
        how="left",
        validate="one_to_one",
    )
    mouse_day["day_c"] = pd.to_numeric(mouse_day["day"], errors="raise") - 3.0
    validate_mouse_day_table(
        mouse_day,
        group_mice=group_mice,
        value_col="success_rate",
        label="Figure-8 Fig3D",
    )
    cohort = f"primary_{cohort_count_label(group_mice)}"
    mouse_day, z_qc = standardize_project_covariates(
        mouse_day, project="Figure-8", cohort=cohort
    )
    trial = atomic["success_mouse_trial"].merge(
        mouse_day[
            ["group", "mouse", "day", *COVARIATE_Z_COLUMNS]
        ],
        on=["group", "mouse", "day"],
        how="left",
        validate="many_to_one",
    )
    validate_trial_table(
        trial,
        mouse_day,
        group_mice=group_mice,
        label="Figure-8 Fig3D/E",
    )
    return {
        **atomic,
        "mouse_day": mouse_day,
        "trial": trial,
        "z_qc": z_qc,
    }

def build_tmaze_tables(
    data_root: Path,
    group_mice: Mapping[str, Sequence[str]],
    *,
    project: str,
    cohort: str,
) -> dict[str, pd.DataFrame]:
    atomic = build_fig3h_tmaze_behavior(
        data_root,
        group_mice=group_mice,
        days=DAYS,
        strict=True,
    )
    mouse_day = atomic.mouse_day.copy()
    validate_mouse_day_table(
        mouse_day,
        group_mice=group_mice,
        value_col="success_rate",
        label=f"T-maze {cohort}",
    )
    mouse_day, z_qc = standardize_project_covariates(
        mouse_day, project=project, cohort=cohort
    )
    trial = atomic.trials.merge(
        mouse_day[["group", "mouse", "day", *COVARIATE_Z_COLUMNS]],
        on=["group", "mouse", "day"],
        how="left",
        validate="many_to_one",
    )
    trial["correct"] = trial["correct"].astype(int)
    validate_trial_table(
        trial,
        mouse_day,
        group_mice=group_mice,
        label=f"T-maze {cohort}",
    )
    return {
        "mouse_day": mouse_day,
        "trial": trial,
        "qc": atomic.qc,
        "z_qc": z_qc,
    }

def strict_common_support_max(
    summary: pd.DataFrame,
    group_mice: Mapping[str, Sequence[str]],
) -> int:
    expected_n = {group: len(tuple(mice)) for group, mice in group_mice.items()}
    pivot = summary.pivot(
        index="cumulative_trial", columns="group", values="effective_mouse_n"
    ).sort_index()
    support = 0
    for cumulative_trial, row in pivot.iterrows():
        if int(cumulative_trial) != support + 1:
            break
        if all(
            group in row.index
            and np.isfinite(row[group])
            and int(row[group]) == expected_n[group]
            for group in GROUP_ORDER
        ):
            support = int(cumulative_trial)
        else:
            break
    if support < 1:
        raise RuntimeError("Panel E has no strict full-cohort common support")
    return support
