"""Manuscript definitions collected from fig3_tmaze_behavior.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from typing import Mapping, Sequence

import numpy as np

import pandas as pd

from .behavior_covariates import (
    empty_frame_weighted_covariates,
    framewise_speed_cm_s,
    point_to_polyline_distance,
    pool_finite_frame_values,
    summarize_frame_weighted_covariates,
    sustained_running_mask,
)

FS = 25.0

SPEED_UNIT_DIVISOR = 10.0

SPEED_THRESHOLD_CM_S = 1.0

MIN_RUN_FRAMES = 15

PROJECTION_THRESHOLD_RAW = 50.0

DAY_CENTER = 3.0

FIG3_TMAZE_DAYS = (1, 2, 3, 4, 5)

FIG3_TMAZE_GROUP_MICE = {
    "C": ("C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"),
    "I": ("I2", "I3", "I4", "I5", "I7", "I8", "I9"),
}

REQUIRED_CHOICE_COLUMNS = ("Day", "Trail", "Choice")

@dataclass(frozen=True)
class Fig3TmazeBehaviorTables:
    """Source tables used to redraw and analyse the T-maze Fig. 3H panel."""

    trials: pd.DataFrame
    mouse_day: pd.DataFrame
    qc: pd.DataFrame

def _empty_trial_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "group",
            "mouse",
            "day",
            "day_c",
            "trial",
            "choice_value",
            "correct",
            "direction",
            "source_row",
        ]
    )

def _strip_choice_headers(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = [str(column).strip() for column in table.columns]
    # Most workbooks use the historical typo ``Trail``; I7 uses ``Trial``.
    if "Trail" not in table.columns and "Trial" in table.columns:
        table = table.rename(columns={"Trial": "Trail"})
    return table

def _normalize_day(raw_day: pd.Series) -> pd.Series:
    """Convert zero-based legacy days to manuscript days while preserving NaN."""
    numeric = pd.to_numeric(raw_day, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty and float(finite.min()) == 0.0:
        numeric = numeric + 1.0
    return numeric

def normalize_choice_sheet(
    table: pd.DataFrame,
    *,
    mouse: str,
    group: str,
    days: Sequence[int] = FIG3_TMAZE_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid trial rows and per-day missingness from a Sheet2 table.

    A valid behavioral observation has finite ``Day``, ``Trail``, and ``Choice``
    fields, with ``Choice`` restricted to 0/1.  The legacy code defines a
    correct trial as ``Choice == 0``.
    """
    table = _strip_choice_headers(table)
    missing = sorted(set(REQUIRED_CHOICE_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(f"choice Sheet2 is missing required columns: {missing}")

    working = table.copy()
    working["source_row"] = np.arange(len(working), dtype=int) + 2
    working["day"] = _normalize_day(working["Day"])
    working["trial"] = pd.to_numeric(working["Trail"], errors="coerce")
    working["choice_value"] = pd.to_numeric(working["Choice"], errors="coerce")
    if "Direction" in working:
        working["direction"] = pd.to_numeric(working["Direction"], errors="coerce")
    else:
        working["direction"] = np.nan

    requested_days = tuple(int(day) for day in days)
    day_selected = working["day"].isin(requested_days)
    required_valid = (
        np.isfinite(working["day"])
        & np.isfinite(working["trial"])
        & working["choice_value"].isin((0.0, 1.0))
    )
    working["behavior_valid"] = required_valid
    selected = working.loc[day_selected & required_valid].copy()
    selected["group"] = str(group)
    selected["mouse"] = str(mouse)
    selected["day"] = selected["day"].astype(int)
    selected["day_c"] = selected["day"].astype(float) - DAY_CENTER
    selected["trial"] = selected["trial"].astype(int)
    selected["correct"] = selected["choice_value"].eq(0.0)
    selected = selected[
        [
            "group",
            "mouse",
            "day",
            "day_c",
            "trial",
            "choice_value",
            "correct",
            "direction",
            "source_row",
        ]
    ].sort_values(["day", "trial", "source_row"], ignore_index=True)

    qc_rows = []
    for day in requested_days:
        day_rows = working.loc[working["day"].eq(day)]
        valid_rows = day_rows.loc[day_rows["behavior_valid"]]
        duplicate_count = int(valid_rows.duplicated(["day", "trial"], keep=False).sum())
        qc_rows.append(
            {
                "group": str(group),
                "mouse": str(mouse),
                "day": int(day),
                "behavior_row_count": int(len(day_rows)),
                "behavior_valid_trial_count": int(len(valid_rows)),
                "behavior_invalid_trial_count": int((~day_rows["behavior_valid"]).sum()),
                "behavior_duplicate_row_count": duplicate_count,
                "behavior_status": (
                    "ok"
                    if len(valid_rows) > 0
                    and duplicate_count == 0
                    and bool(day_rows["behavior_valid"].all())
                    else "partial"
                    if len(valid_rows) > 0
                    else "no_valid_trials"
                ),
            }
        )
    return selected, pd.DataFrame(qc_rows)

def read_choice_sheet2(
    path: Path,
    *,
    mouse: str,
    group: str,
    days: Sequence[int] = FIG3_TMAZE_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and normalize the ``Sheet2`` behavioral source for one mouse."""
    table = pd.read_excel(Path(path), sheet_name="Sheet2")
    return normalize_choice_sheet(table, mouse=mouse, group=group, days=days)

def calculate_speed(
    xy: np.ndarray,
    *,
    fs: float = FS,
    unit_divisor: float = SPEED_UNIT_DIVISOR,
) -> np.ndarray:
    """Compatibility wrapper for legacy raw-coordinate speed in cm/s."""

    return framewise_speed_cm_s(
        xy,
        sampling_rate_hz=fs,
        coordinate_units_per_cm=unit_divisor,
        short_trace_value=0.0,
    )

def keep_long_speed_runs(
    speed: np.ndarray,
    *,
    threshold: float = SPEED_THRESHOLD_CM_S,
    n: int = MIN_RUN_FRAMES,
) -> np.ndarray:
    """Compatibility wrapper around the shared sustained-running primitive."""

    return sustained_running_mask(
        speed,
        threshold_cm_s=threshold,
        min_consecutive_frames=n,
    )

def estimate_day_translation(xy_day_parts: Sequence[np.ndarray]) -> tuple[float, float]:
    """Reproduce the day-level stem/side median translation used for map QC."""
    parts = [np.asarray(part, dtype=float) for part in xy_day_parts if np.asarray(part).size]
    if not parts:
        return 0.0, 0.0
    xy = np.vstack(parts)
    x = xy[:, 0]
    y = xy[:, 1]
    stem = (np.abs(x) <= 120) & (y >= 75) & (y <= 375)
    side = (np.abs(y) <= 120) & (np.abs(x) >= 75) & (np.abs(x) <= 400)
    dx = float(np.nanmedian(x[stem])) if np.sum(stem) >= 50 else 0.0
    dy = float(np.nanmedian(y[side])) if np.sum(side) >= 50 else 0.0
    return dx, dy

def path_domain_distance(
    xy: np.ndarray,
    *,
    is_left: bool,
    stem_top_raw: float = 387.5,
    side_low_raw: float = -12.5,
    side_high_raw: float = 412.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return route-relative distance, domain gate, and stem-frame indicator."""
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        empty_float = np.empty(0, dtype=float)
        empty_bool = np.empty(0, dtype=bool)
        return empty_float, empty_bool, empty_bool
    x = xy[:, 0]
    y = xy[:, 1]
    stem_frames = y > 50.0
    side_frames = ~stem_frames
    distance = np.full(x.shape, np.nan, dtype=float)
    in_domain = np.zeros(x.shape, dtype=bool)

    distance[stem_frames] = np.abs(x[stem_frames])
    in_domain[stem_frames] = stem_frames[stem_frames] & (y[stem_frames] <= stem_top_raw)

    side_progress = -x if bool(is_left) else x
    side_in_domain = side_frames & (side_progress >= side_low_raw) & (side_progress <= side_high_raw)
    distance[side_frames] = np.abs(y[side_frames])
    in_domain[side_frames] = side_in_domain[side_frames]

    # Recompute eligible distances through the shared orthogonal-distance
    # primitive.  The segment endpoints match the legacy domain gates exactly,
    # so this preserves |x| on the stem and |y| on the selected choice arm.
    eligible_stem = stem_frames & in_domain
    if np.any(eligible_stem):
        stem_vertices = np.asarray([[0.0, 50.0], [0.0, float(stem_top_raw)]])
        distance[eligible_stem] = point_to_polyline_distance(
            xy[eligible_stem], stem_vertices
        )
    eligible_side = side_frames & in_domain
    if np.any(eligible_side):
        if bool(is_left):
            side_vertices = np.asarray(
                [[-float(side_low_raw), 0.0], [-float(side_high_raw), 0.0]]
            )
        else:
            side_vertices = np.asarray(
                [[float(side_low_raw), 0.0], [float(side_high_raw), 0.0]]
            )
        distance[eligible_side] = point_to_polyline_distance(
            xy[eligible_side], side_vertices
        )
    return distance, in_domain, stem_frames

def choice_run_is_left(index_all: np.ndarray) -> np.ndarray:
    """Return the actual C-run route branch from reward-side and correctness labels."""
    index_all = np.asarray(index_all)
    if index_all.ndim != 3 or index_all.shape[0] < 2:
        raise ValueError(f"index_all must have shape (>=2, day, trial), got {index_all.shape}")
    reward_left = np.asarray(index_all[0], dtype=bool)
    correct = np.asarray(index_all[1], dtype=bool)
    return correct == reward_left

def compute_choice_run_covariates(
    extracted_xy: Mapping[str, np.ndarray],
    index_all: np.ndarray,
    *,
    mouse: str,
    group: str,
    days: Sequence[int] = FIG3_TMAZE_DAYS,
    fs: float = FS,
    speed_unit_divisor: float = SPEED_UNIT_DIVISOR,
    speed_threshold: float = SPEED_THRESHOLD_CM_S,
    min_run_frames: int = MIN_RUN_FRAMES,
    projection_threshold_raw: float = PROJECTION_THRESHOLD_RAW,
) -> pd.DataFrame:
    """Compute pooled C-run speed and lateral deviation for each mouse-day.

    Both means use the same frames: C-run frames belonging to a >=15-frame
    run at >=1 cm/s and passing the actual-route domain and 50-raw-unit
    projection gate.  Lateral deviation is ``|x|`` on the stem and ``|y|`` on
    the choice arm after day translation, converted to cm by division by 10.
    """
    if "C_runout" not in extracted_xy:
        raise ValueError("extracted_xy is missing C_runout")
    c_xy = np.asarray(extracted_xy["C_runout"], dtype=object)
    if c_xy.ndim != 2:
        raise ValueError(f"C_runout must have shape (day, trial), got {c_xy.shape}")
    index_all = np.asarray(index_all)
    if index_all.ndim != 3 or index_all.shape[1:] != c_xy.shape:
        raise ValueError(
            f"index_all day/trial shape {index_all.shape[1:]} does not match C_runout {c_xy.shape}"
        )
    c_is_left = choice_run_is_left(index_all)
    requested_days = tuple(int(day) for day in days)
    num_day, num_trial = c_xy.shape
    rows = []

    for day in requested_days:
        day_idx = day - 1
        base = {
            "group": str(group),
            "mouse": str(mouse),
            "day": int(day),
            "day_c": float(day) - DAY_CENTER,
        }
        if day_idx < 0 or day_idx >= num_day:
            rows.append(
                {
                    **base,
                    **empty_frame_weighted_covariates("day_out_of_range"),
                    "mean_lateral_deviation_cm": np.nan,
                    "mean_stem_lateral_deviation_cm": np.nan,
                    "mean_choice_arm_lateral_deviation_cm": np.nan,
                    "translation_dx_raw": np.nan,
                    "translation_dy_raw": np.nan,
                    "translation_dx_cm": np.nan,
                    "translation_dy_cm": np.nan,
                    "choice_run_trial_count": 0,
                    "choice_run_valid_trial_count": 0,
                    "choice_run_left_trial_count": 0,
                    "choice_run_right_trial_count": 0,
                    "choice_run_raw_frame_count": 0,
                    "choice_run_motion_frame_count": 0,
                    "choice_run_domain_frame_count": 0,
                    "choice_run_valid_frame_count": 0,
                    "choice_run_valid_stem_frame_count": 0,
                    "choice_run_valid_choice_arm_frame_count": 0,
                }
            )
            continue

        day_parts = []
        for source_key in ("C_runout", "S_runout"):
            source = extracted_xy.get(source_key)
            if source is None:
                continue
            source = np.asarray(source, dtype=object)
            if source.ndim != 2 or day_idx >= source.shape[0]:
                continue
            for trial_idx in range(source.shape[1]):
                part = np.asarray(source[day_idx, trial_idx], dtype=float)
                if part.ndim == 2 and part.shape[1] == 2 and part.size:
                    day_parts.append(part)
        dx, dy = estimate_day_translation(day_parts)

        speed_values = []
        lateral_values = []
        stem_lateral_values = []
        arm_lateral_values = []
        raw_frame_count = 0
        motion_frame_count = 0
        domain_frame_count = 0
        valid_frame_count = 0
        valid_stem_frame_count = 0
        valid_arm_frame_count = 0
        valid_trial_count = 0
        left_trial_count = 0
        right_trial_count = 0

        for trial_idx in range(num_trial):
            xy = np.asarray(c_xy[day_idx, trial_idx], dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2 or not xy.size:
                continue
            is_left = bool(c_is_left[day_idx, trial_idx])
            left_trial_count += int(is_left)
            right_trial_count += int(not is_left)
            xy = xy.copy()
            xy[:, 0] -= dx
            xy[:, 1] -= dy
            raw_frame_count += int(xy.shape[0])
            speed = calculate_speed(xy, fs=fs, unit_divisor=speed_unit_divisor)
            motion = keep_long_speed_runs(speed, threshold=speed_threshold, n=min_run_frames)
            motion_frame_count += int(motion.sum())
            if not np.any(motion):
                continue
            xy_motion = xy[motion]
            speed_motion = speed[motion]
            distance, in_domain, stem_frames = path_domain_distance(
                xy_motion,
                is_left=is_left,
            )
            domain_frame_count += int(in_domain.sum())
            valid = (
                in_domain
                & np.isfinite(distance)
                & (distance <= float(projection_threshold_raw))
                & np.isfinite(speed_motion)
            )
            if not np.any(valid):
                continue
            lateral_cm = distance[valid] / float(speed_unit_divisor)
            finite = np.isfinite(lateral_cm)
            if not np.any(finite):
                continue
            valid_indices = np.flatnonzero(valid)[finite]
            valid_stem = stem_frames[valid_indices]
            trial_speed = speed_motion[valid_indices]
            trial_lateral = lateral_cm[finite]
            speed_values.append(trial_speed)
            lateral_values.append(trial_lateral)
            if np.any(valid_stem):
                stem_lateral_values.append(trial_lateral[valid_stem])
            if np.any(~valid_stem):
                arm_lateral_values.append(trial_lateral[~valid_stem])
            trial_valid_count = int(valid_indices.size)
            valid_frame_count += trial_valid_count
            valid_stem_frame_count += int(valid_stem.sum())
            valid_arm_frame_count += int((~valid_stem).sum())
            valid_trial_count += 1

        shared_summary = summarize_frame_weighted_covariates(
            speed_values,
            lateral_values,
            raw_frame_count=raw_frame_count,
            running_frame_count=motion_frame_count,
            valid_frame_count=valid_frame_count,
            empty_status="no_valid_frames",
        )
        pooled_stem = pool_finite_frame_values(stem_lateral_values)
        pooled_arm = pool_finite_frame_values(arm_lateral_values)
        rows.append(
            {
                **base,
                **shared_summary,
                # Backward-compatible alias retained for existing Fig. 3H code.
                "mean_lateral_deviation_cm": shared_summary[
                    "route_relative_lateral_deviation_cm"
                ],
                "mean_stem_lateral_deviation_cm": (
                    float(np.mean(pooled_stem)) if pooled_stem.size else np.nan
                ),
                "mean_choice_arm_lateral_deviation_cm": (
                    float(np.mean(pooled_arm)) if pooled_arm.size else np.nan
                ),
                "translation_dx_raw": float(dx),
                "translation_dy_raw": float(dy),
                "translation_dx_cm": float(dx) / float(speed_unit_divisor),
                "translation_dy_cm": float(dy) / float(speed_unit_divisor),
                "choice_run_trial_count": int(num_trial),
                "choice_run_valid_trial_count": int(valid_trial_count),
                "choice_run_left_trial_count": int(left_trial_count),
                "choice_run_right_trial_count": int(right_trial_count),
                "choice_run_raw_frame_count": int(raw_frame_count),
                "choice_run_motion_frame_count": int(motion_frame_count),
                "choice_run_domain_frame_count": int(domain_frame_count),
                "choice_run_valid_frame_count": int(valid_frame_count),
                "choice_run_valid_stem_frame_count": int(valid_stem_frame_count),
                "choice_run_valid_choice_arm_frame_count": int(valid_arm_frame_count),
            }
        )
    return pd.DataFrame(rows)

def _expected_mouse_days(
    group_mice: Mapping[str, Sequence[str]],
    days: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for group, mice in group_mice.items():
        for mouse in mice:
            for day in days:
                rows.append(
                    {
                        "group": str(group),
                        "mouse": str(mouse),
                        "day": int(day),
                        "day_c": float(day) - DAY_CENTER,
                    }
                )
    return pd.DataFrame(rows)

def build_fig3h_tmaze_behavior(
    data_root: Path,
    *,
    group_mice: Mapping[str, Sequence[str]] = FIG3_TMAZE_GROUP_MICE,
    days: Sequence[int] = FIG3_TMAZE_DAYS,
    strict: bool = True,
) -> Fig3TmazeBehaviorTables:
    """Load trial-level behavior and build the complete Fig. 3H mouse-day table."""
    data_root = Path(data_root)
    days = tuple(int(day) for day in days)
    expected = _expected_mouse_days(group_mice, days)
    trial_tables = []
    behavior_qc_tables = []
    covariate_tables = []

    for group, mice in group_mice.items():
        for mouse in mice:
            choice_path = data_root / "Behavior_data" / mouse / f"choice_{mouse}.xlsx"
            try:
                trials, behavior_qc = read_choice_sheet2(
                    choice_path,
                    mouse=mouse,
                    group=group,
                    days=days,
                )
            except (FileNotFoundError, ValueError, KeyError) as exc:
                trials = _empty_trial_table()
                behavior_qc = expected.loc[
                    (expected["group"] == group) & (expected["mouse"] == mouse),
                    ["group", "mouse", "day"],
                ].copy()
                behavior_qc["behavior_row_count"] = 0
                behavior_qc["behavior_valid_trial_count"] = 0
                behavior_qc["behavior_invalid_trial_count"] = 0
                behavior_qc["behavior_duplicate_row_count"] = 0
                behavior_qc["behavior_status"] = f"error:{type(exc).__name__}"
            trial_tables.append(trials)
            behavior_qc_tables.append(behavior_qc)

            aligned_dir = data_root / "Aligened_data" / mouse
            try:
                extracted_xy = np.load(
                    aligned_dir / "extracted_xy.npy", allow_pickle=True
                ).item()
                index_all = np.load(aligned_dir / "index_all.npy", allow_pickle=True)
                covariates = compute_choice_run_covariates(
                    extracted_xy,
                    index_all,
                    mouse=mouse,
                    group=group,
                    days=days,
                )
            except (FileNotFoundError, ValueError, KeyError, AttributeError) as exc:
                covariates = expected.loc[
                    (expected["group"] == group) & (expected["mouse"] == mouse)
                ].copy()
                empty_covariates = empty_frame_weighted_covariates(
                    f"error:{type(exc).__name__}"
                )
                for column, value in empty_covariates.items():
                    covariates[column] = value
                covariates["mean_lateral_deviation_cm"] = np.nan
            covariate_tables.append(covariates)

    nonempty_trials = [table for table in trial_tables if not table.empty]
    trials = (
        pd.concat(nonempty_trials, ignore_index=True)
        if nonempty_trials
        else _empty_trial_table()
    )
    if not trials.empty:
        trials["day"] = pd.to_numeric(trials["day"], errors="raise").astype(int)
        trials["day_c"] = pd.to_numeric(trials["day_c"], errors="raise").astype(float)
        trials["trial"] = pd.to_numeric(trials["trial"], errors="raise").astype(int)
        trials["choice_value"] = pd.to_numeric(
            trials["choice_value"], errors="raise"
        ).astype(float)
        trials["correct"] = trials["correct"].astype(bool)
    behavior_qc = pd.concat(behavior_qc_tables, ignore_index=True)
    covariates = pd.concat(covariate_tables, ignore_index=True)

    behavior_summary = (
        trials.groupby(["group", "mouse", "day", "day_c"], as_index=False, observed=True)
        .agg(
            n_behavior_trials=("correct", "size"),
            n_correct=("correct", "sum"),
            success_rate=("correct", "mean"),
        )
    )
    mouse_day = expected.merge(
        behavior_summary,
        on=["group", "mouse", "day", "day_c"],
        how="left",
        validate="one_to_one",
    ).merge(
        covariates,
        on=["group", "mouse", "day", "day_c"],
        how="left",
        validate="one_to_one",
    )
    mouse_day["day"] = pd.to_numeric(mouse_day["day"], errors="raise").astype(int)
    mouse_day["day_c"] = pd.to_numeric(mouse_day["day_c"], errors="raise").astype(float)
    mouse_day["success_rate"] = pd.to_numeric(
        mouse_day["success_rate"], errors="coerce"
    )
    for column in ("n_behavior_trials", "n_correct"):
        mouse_day[column] = pd.to_numeric(mouse_day[column], errors="coerce").fillna(0).astype(int)

    qc = expected.merge(
        behavior_qc,
        on=["group", "mouse", "day"],
        how="left",
        validate="one_to_one",
    ).merge(
        covariates,
        on=["group", "mouse", "day", "day_c"],
        how="left",
        validate="one_to_one",
    )
    completeness = mouse_day[["group", "mouse", "day"]].copy()
    completeness["primary_complete"] = (
        np.isfinite(pd.to_numeric(mouse_day["success_rate"], errors="coerce"))
        & np.isfinite(pd.to_numeric(mouse_day["mean_speed_cm_s"], errors="coerce"))
        & np.isfinite(
            pd.to_numeric(mouse_day["mean_lateral_deviation_cm"], errors="coerce")
        )
    )
    qc = qc.merge(
        completeness,
        on=["group", "mouse", "day"],
        how="left",
        validate="one_to_one",
    )

    # Cross-check the independent behavioral and aligned-data correctness labels.
    mismatch_rows = []
    for (group, mouse, day), day_trials in trials.groupby(
        ["group", "mouse", "day"], observed=True, sort=False
    ):
        aligned_dir = data_root / "Aligened_data" / mouse
        mismatch = np.nan
        compared = 0
        try:
            index_all = np.load(aligned_dir / "index_all.npy", allow_pickle=True)
            correct_index = np.asarray(index_all[1, int(day) - 1], dtype=bool)
            indices = day_trials["trial"].to_numpy(dtype=int) - 1
            in_range = (indices >= 0) & (indices < correct_index.size)
            if np.any(in_range):
                observed = day_trials.loc[in_range, "correct"].to_numpy(dtype=bool)
                expected_correct = correct_index[indices[in_range]]
                compared = int(in_range.sum())
                mismatch = int(np.sum(observed != expected_correct))
        except (FileNotFoundError, IndexError, ValueError):
            pass
        mismatch_rows.append(
            {
                "group": group,
                "mouse": mouse,
                "day": int(day),
                "behavior_index_compared_trial_count": compared,
                "behavior_index_correct_mismatch_count": mismatch,
            }
        )
    mismatch_table = pd.DataFrame(mismatch_rows)
    if mismatch_table.empty:
        mismatch_table = expected[["group", "mouse", "day"]].copy()
        mismatch_table["behavior_index_compared_trial_count"] = 0
        mismatch_table["behavior_index_correct_mismatch_count"] = np.nan
    qc = qc.merge(
        mismatch_table,
        on=["group", "mouse", "day"],
        how="left",
        validate="one_to_one",
    )
    qc["behavior_index_compared_trial_count"] = qc[
        "behavior_index_compared_trial_count"
    ].fillna(0).astype(int)

    mouse_day = mouse_day.sort_values(["group", "mouse", "day"], ignore_index=True)
    qc = qc.sort_values(["group", "mouse", "day"], ignore_index=True)
    trials = trials.sort_values(["group", "mouse", "day", "trial"], ignore_index=True)

    if strict:
        problems = qc.loc[
            ~qc["primary_complete"]
            | ~qc["behavior_status"].eq("ok")
            | qc["behavior_index_correct_mismatch_count"].fillna(1).ne(0)
        ]
        if not problems.empty:
            keys = problems[["group", "mouse", "day"]].to_dict("records")
            raise ValueError(f"Fig. 3H source/QC validation failed for {keys}")

    return Fig3TmazeBehaviorTables(trials=trials, mouse_day=mouse_day, qc=qc)
