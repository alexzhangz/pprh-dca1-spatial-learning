"""Manuscript definitions collected from fig3_8maze_behavior.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from pathlib import Path

from typing import Iterable, Mapping, Sequence

import numpy as np

import pandas as pd

from .behavior_covariates import (
    empty_frame_weighted_covariates,
    framewise_speed_cm_s as _shared_framewise_speed_cm_s,
    point_to_polyline_distance,
    summarize_frame_weighted_covariates,
    sustained_running_mask,
)

from .spatial import (
    LEFT_REF_X,
    LEFT_REF_Y,
    RIGHT_REF_X,
    RIGHT_REF_Y,
)

from .subjects import PV_PC_GROUP_MICE

FIG3_8MAZE_GROUP_MICE = {
    "C": tuple(PV_PC_GROUP_MICE["C"]),
    "I": tuple(PV_PC_GROUP_MICE["I"]),
}

FIG3_8MAZE_DAYS = (1, 2, 3, 4, 5)

SAMPLING_RATE_HZ = 25.0

RUN_SPEED_THRESHOLD_CM_S = 1.0

RUN_MIN_CONSECUTIVE_FRAMES = 15

MOVING_WINDOW_HALF_WIDTH = 10

def valid_trial_indices(day_labels, *, n_xy_trial: int | None = None) -> np.ndarray:
    """Return original trial-column indices under the locked valid-trial rule.

    Column zero is skipped unconditionally.  All-false columns are then
    excluded.  When ``n_xy_trial`` is supplied, only label columns with a
    corresponding trajectory are considered.
    """

    labels = _validated_day_labels(day_labels)
    stop = labels.shape[1]
    if n_xy_trial is not None:
        stop = min(stop, max(int(n_xy_trial), 0))
    if stop <= 1:
        return np.empty(0, dtype=int)
    candidate = np.arange(1, stop, dtype=int)
    valid = np.any(labels[:3, candidate].astype(bool), axis=0)
    return candidate[valid]

def finite_trial_outcomes(day_labels) -> np.ndarray:
    """Return 0/1 outcomes for valid trials after the placeholder column."""

    labels = _validated_day_labels(day_labels)
    indices = valid_trial_indices(labels)
    return labels[2, indices].astype(float)

def summarize_success_day(day_labels) -> dict:
    """Summarize correct/valid counts and success rate for one mouse-day."""

    outcomes = finite_trial_outcomes(day_labels)
    n_valid = int(outcomes.size)
    n_correct = int(np.sum(outcomes))
    return {
        "n_valid_trial": n_valid,
        "n_correct_trial": n_correct,
        "success_rate": float(outcomes.mean()) if n_valid else np.nan,
        "missing_reason": None if n_valid else "no_valid_trials",
    }

def success_trial_rows_for_mouse(
    trial_labels_all,
    *,
    mouse: str,
    group: str,
    days: Sequence[int] = FIG3_8MAZE_DAYS,
) -> pd.DataFrame:
    """Return one atomic 0/1 correctness row per valid Figure-8 trial."""

    trial_labels_all = _as_day_sequence(trial_labels_all)
    rows = []
    cumulative_trial = 0
    for day in days:
        day_idx = int(day) - 1
        if day_idx < 0:
            raise ValueError("days must use one-based positive indices")
        if day_idx >= len(trial_labels_all):
            continue
        labels = _validated_day_labels(trial_labels_all[day_idx])
        indices = valid_trial_indices(labels)
        for valid_ordinal, source_trial_index in enumerate(indices, start=1):
            cumulative_trial += 1
            rows.append(
                {
                    "group": str(group),
                    "mouse": str(mouse),
                    "day": int(day),
                    "day_c": float(day) - 3.0,
                    "trial_in_day": int(valid_ordinal),
                    "cumulative_trial": int(cumulative_trial),
                    "source_trial_index": int(source_trial_index),
                    "actual_left": bool(labels[0, source_trial_index]),
                    "rewarded_left": bool(labels[1, source_trial_index]),
                    "correct": int(bool(labels[2, source_trial_index])),
                }
            )
    return pd.DataFrame(rows, columns=_SUCCESS_TRIAL_COLUMNS)

def moving_success_rows_for_mouse(
    trial_labels_all,
    *,
    mouse: str,
    group: str,
    days: Sequence[int] = FIG3_8MAZE_DAYS,
    window_half: int = MOVING_WINDOW_HALF_WIDTH,
) -> pd.DataFrame:
    """Build the legacy centered moving-success trace for one mouse.

    The centered window is computed separately within each day, so it never
    crosses a day boundary.  ``cumulative_trial`` is the sole quantity that
    continues across days.
    """

    if int(window_half) < 0:
        raise ValueError("window_half must be >= 0")
    trial_labels_all = _as_day_sequence(trial_labels_all)
    rows = []
    cumulative_trial = 0
    for day in days:
        day_idx = int(day) - 1
        if day_idx < 0:
            raise ValueError("days must use one-based positive indices")
        if day_idx >= len(trial_labels_all):
            continue
        outcomes = finite_trial_outcomes(trial_labels_all[day_idx])
        for trial_idx in range(outcomes.size):
            start = max(0, trial_idx - int(window_half))
            end = min(outcomes.size, trial_idx + int(window_half) + 1)
            cumulative_trial += 1
            rows.append(
                {
                    "group": str(group),
                    "mouse": str(mouse),
                    "day": int(day),
                    "trial_in_day": int(trial_idx + 1),
                    "cumulative_trial": int(cumulative_trial),
                    "moving_success_rate": float(outcomes[start:end].mean()),
                    "window_n_trial": int(end - start),
                    "window_start_trial_in_day": int(start + 1),
                    "window_end_trial_in_day": int(end),
                }
            )
    return pd.DataFrame(rows, columns=_MOVING_LONG_COLUMNS)

def summarize_moving_success_rate(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize moving-success traces with the effective mouse n at each x."""

    missing = set(_MOVING_LONG_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"moving-success table is missing columns: {sorted(missing)}")
    rows = []
    grouped = table.groupby(["group", "cumulative_trial"], observed=True, sort=True)
    for (group, cumulative_trial), trace in grouped:
        finite = trace.loc[
            np.isfinite(pd.to_numeric(trace["moving_success_rate"], errors="coerce"))
        ].copy()
        values = finite["moving_success_rate"].to_numpy(dtype=float)
        n_mouse = int(finite["mouse"].nunique())
        if values.size > 1:
            sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
        else:
            sem = np.nan
        rows.append(
            {
                "group": str(group),
                "cumulative_trial": int(cumulative_trial),
                "mean": float(values.mean()) if values.size else np.nan,
                "sem": sem,
                "effective_mouse_n": n_mouse,
                # Retained for direct compatibility with the legacy table.
                "n_observation": int(values.size),
            }
        )
    return pd.DataFrame(rows, columns=_MOVING_SUMMARY_COLUMNS)

def framewise_speed_cm_s(xy, *, fs: float = SAMPLING_RATE_HZ) -> np.ndarray:
    """Compatibility wrapper around the shared gradient-speed primitive."""

    return _shared_framewise_speed_cm_s(
        xy,
        sampling_rate_hz=fs,
        coordinate_units_per_cm=1.0,
        short_trace_value=np.nan,
    )

def running_frame_mask(
    speed,
    *,
    threshold_cm_s: float = RUN_SPEED_THRESHOLD_CM_S,
    min_consecutive_frames: int = RUN_MIN_CONSECUTIVE_FRAMES,
) -> np.ndarray:
    """Compatibility wrapper around the shared sustained-running primitive."""

    return sustained_running_mask(
        speed,
        threshold_cm_s=threshold_cm_s,
        min_consecutive_frames=min_consecutive_frames,
    )

def route_centerline_vertices(actual_left: bool) -> np.ndarray:
    """Return the ideal centerline polyline for the actually traversed route.

    Maze extents are derived from the shared spatial reference paths.  The
    returned path follows the start arm, top arm, outer side arm, and end arm.
    """

    ref_x = LEFT_REF_X if bool(actual_left) else RIGHT_REF_X
    ref_y = LEFT_REF_Y if bool(actual_left) else RIGHT_REF_Y
    outer_x = float(np.nanmin(ref_x) if bool(actual_left) else np.nanmax(ref_x))
    top_y = float(np.nanmax(ref_y))
    bottom_y = float(np.nanmin(ref_y))
    return np.asarray(
        [
            (0.0, bottom_y),
            (0.0, top_y),
            (outer_x, top_y),
            (outer_x, bottom_y),
            (0.0, bottom_y),
        ],
        dtype=float,
    )

def route_relative_lateral_deviation_cm(xy, *, actual_left: bool) -> np.ndarray:
    """Return absolute orthogonal distance from each point to route centerline."""

    return point_to_polyline_distance(
        xy,
        route_centerline_vertices(bool(actual_left)),
    )

def compute_mouse_day_covariates(
    xy_day,
    labels_day,
    *,
    fs: float = SAMPLING_RATE_HZ,
    speed_threshold_cm_s: float = RUN_SPEED_THRESHOLD_CM_S,
    min_consecutive_frames: int = RUN_MIN_CONSECUTIVE_FRAMES,
) -> dict:
    """Compute frame-weighted behavioral covariates for one mouse-day."""

    labels = _validated_day_labels(labels_day)
    xy_trials = _as_trial_sequence(xy_day)
    label_valid_indices = valid_trial_indices(labels)
    available_indices = valid_trial_indices(labels, n_xy_trial=len(xy_trials))

    speed_parts = []
    deviation_parts = []
    n_total_frame = 0
    n_finite_xy_frame = 0
    n_running_frame = 0
    n_lateral_frame = 0
    n_trial_with_trajectory = 0
    n_trial_with_running_frames = 0
    n_invalid_xy_trial = 0

    for trial_idx in available_indices:
        try:
            xy = _validated_xy(xy_trials[int(trial_idx)])
        except (TypeError, ValueError):
            n_invalid_xy_trial += 1
            continue
        n_trial_with_trajectory += 1
        n_total_frame += int(xy.shape[0])
        finite_xy = np.isfinite(xy).all(axis=1)
        n_finite_xy_frame += int(np.sum(finite_xy))
        if xy.shape[0] < 2:
            continue

        speed = framewise_speed_cm_s(xy, fs=fs)
        running = running_frame_mask(
            speed,
            threshold_cm_s=speed_threshold_cm_s,
            min_consecutive_frames=min_consecutive_frames,
        )
        running &= finite_xy & np.isfinite(speed)
        if not np.any(running):
            continue
        n_trial_with_running_frames += 1
        n_running_frame += int(np.sum(running))
        speed_parts.append(speed[running])

        deviation = route_relative_lateral_deviation_cm(
            xy,
            actual_left=bool(labels[0, int(trial_idx)]),
        )
        lateral_valid = running & np.isfinite(deviation)
        n_lateral_frame += int(np.sum(lateral_valid))
        if np.any(lateral_valid):
            deviation_parts.append(deviation[lateral_valid])

    missing_reason = _covariate_missing_reason(
        n_valid_trial=int(label_valid_indices.size),
        n_trial_with_trajectory=n_trial_with_trajectory,
        n_finite_xy_frame=n_finite_xy_frame,
        n_running_frame=n_running_frame,
        n_lateral_frame=n_lateral_frame,
    )
    shared_summary = summarize_frame_weighted_covariates(
        speed_parts,
        deviation_parts,
        raw_frame_count=n_total_frame,
        running_frame_count=n_running_frame,
        valid_frame_count=n_lateral_frame,
        empty_status=missing_reason or "no_valid_frames",
    )
    return {
        **shared_summary,
        "n_valid_trial": int(label_valid_indices.size),
        "n_valid_trial_with_xy": int(available_indices.size),
        "n_trial_missing_xy": int(label_valid_indices.size - available_indices.size),
        "n_trial_with_trajectory": int(n_trial_with_trajectory),
        "n_trial_with_running_frames": int(n_trial_with_running_frames),
        "n_invalid_xy_trial": int(n_invalid_xy_trial),
        "n_total_frame": int(n_total_frame),
        "n_finite_xy_frame": int(n_finite_xy_frame),
        "n_running_frame": int(n_running_frame),
        "n_lateral_frame": int(n_lateral_frame),
        "missing_reason": missing_reason,
    }

def build_fig3_8maze_behavior_tables(
    aligned_root,
    *,
    group_mice: Mapping[str, Iterable[str]] = FIG3_8MAZE_GROUP_MICE,
    days: Sequence[int] = FIG3_8MAZE_DAYS,
    window_half: int = MOVING_WINDOW_HALF_WIDTH,
    fs: float = SAMPLING_RATE_HZ,
    speed_threshold_cm_s: float = RUN_SPEED_THRESHOLD_CM_S,
    min_consecutive_frames: int = RUN_MIN_CONSECUTIVE_FRAMES,
) -> dict[str, pd.DataFrame]:
    """Build all atomic Figure 3D/E five-day behavior source tables."""

    aligned_root = Path(aligned_root)
    success_rows = []
    success_trial_parts = []
    covariate_rows = []
    moving_parts = []

    for group, mice in group_mice.items():
        for mouse in mice:
            subject_dir = aligned_root / str(mouse)
            labels_path = subject_dir / "trial_labels.npy"
            xy_path = subject_dir / "xy_trial_all.npy"
            labels_all = _load_optional_npy(labels_path)
            xy_all = _load_optional_npy(xy_path)

            if labels_all is not None:
                success_trial_parts.append(
                    success_trial_rows_for_mouse(
                        labels_all,
                        mouse=str(mouse),
                        group=str(group),
                        days=days,
                    )
                )
                moving_parts.append(
                    moving_success_rows_for_mouse(
                        labels_all,
                        mouse=str(mouse),
                        group=str(group),
                        days=days,
                        window_half=window_half,
                    )
                )

            for day in days:
                day_idx = int(day) - 1
                identity = {"group": str(group), "mouse": str(mouse), "day": int(day)}
                if labels_all is None:
                    success_rows.append(
                        {
                            **identity,
                            "n_valid_trial": 0,
                            "n_correct_trial": 0,
                            "success_rate": np.nan,
                            "missing_reason": "missing_trial_labels_file",
                        }
                    )
                    covariate_rows.append(
                        _empty_covariate_row(identity, "missing_trial_labels_file")
                    )
                    continue
                if day_idx < 0 or day_idx >= len(labels_all):
                    success_rows.append(
                        {
                            **identity,
                            "n_valid_trial": 0,
                            "n_correct_trial": 0,
                            "success_rate": np.nan,
                            "missing_reason": "missing_day",
                        }
                    )
                    covariate_rows.append(_empty_covariate_row(identity, "missing_day"))
                    continue

                success_rows.append({**identity, **summarize_success_day(labels_all[day_idx])})
                if xy_all is None:
                    covariate_rows.append(
                        _empty_covariate_row(identity, "missing_xy_trial_file")
                    )
                elif day_idx >= len(xy_all):
                    covariate_rows.append(_empty_covariate_row(identity, "missing_xy_day"))
                else:
                    covariate_rows.append(
                        {
                            **identity,
                            **compute_mouse_day_covariates(
                                xy_all[day_idx],
                                labels_all[day_idx],
                                fs=fs,
                                speed_threshold_cm_s=speed_threshold_cm_s,
                                min_consecutive_frames=min_consecutive_frames,
                            ),
                        }
                    )

    moving_long = (
        pd.concat(moving_parts, ignore_index=True)
        if moving_parts
        else pd.DataFrame(columns=_MOVING_LONG_COLUMNS)
    )
    success_trial = (
        pd.concat(success_trial_parts, ignore_index=True)
        if success_trial_parts
        else pd.DataFrame(columns=_SUCCESS_TRIAL_COLUMNS)
    )
    return {
        "success_mouse_day": pd.DataFrame(success_rows),
        "success_mouse_trial": success_trial,
        "moving_success_mouse_trial": moving_long,
        "moving_success_group_summary": summarize_moving_success_rate(moving_long),
        "covariates_mouse_day": pd.DataFrame(covariate_rows),
    }

def _validated_day_labels(day_labels) -> np.ndarray:
    labels = np.asarray(day_labels)
    if labels.ndim != 2 or labels.shape[0] < 3:
        raise ValueError(f"day_labels must have shape (>=3, n_trials), got {labels.shape}")
    return labels

def _validated_xy(xy) -> np.ndarray:
    result = np.asarray(xy, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"xy must have shape (n_frame, 2), got {result.shape}")
    return result

def _as_day_sequence(values):
    if isinstance(values, np.ndarray) and values.ndim == 0:
        raise ValueError("day-level input must contain a day axis")
    return values

def _as_trial_sequence(values):
    if isinstance(values, np.ndarray) and values.ndim == 0:
        raise ValueError("xy_day must contain a trial axis")
    return values

def _load_optional_npy(path: Path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)

def _covariate_missing_reason(
    *,
    n_valid_trial: int,
    n_trial_with_trajectory: int,
    n_finite_xy_frame: int,
    n_running_frame: int,
    n_lateral_frame: int,
) -> str | None:
    if n_valid_trial == 0:
        return "no_valid_trials"
    if n_trial_with_trajectory == 0:
        return "no_valid_trajectory"
    if n_finite_xy_frame == 0:
        return "no_finite_xy_frames"
    if n_running_frame == 0:
        return "no_running_frames"
    if n_lateral_frame == 0:
        return "no_lateral_frames"
    return None

def _empty_covariate_row(identity: Mapping[str, object], reason: str) -> dict:
    return {
        **identity,
        **empty_frame_weighted_covariates(reason),
        "n_valid_trial": 0,
        "n_valid_trial_with_xy": 0,
        "n_trial_missing_xy": 0,
        "n_trial_with_trajectory": 0,
        "n_trial_with_running_frames": 0,
        "n_invalid_xy_trial": 0,
        "n_total_frame": 0,
        "n_finite_xy_frame": 0,
        "n_running_frame": 0,
        "n_lateral_frame": 0,
        "missing_reason": str(reason),
    }

_MOVING_LONG_COLUMNS = [
    "group",
    "mouse",
    "day",
    "trial_in_day",
    "cumulative_trial",
    "moving_success_rate",
    "window_n_trial",
    "window_start_trial_in_day",
    "window_end_trial_in_day",
]

_SUCCESS_TRIAL_COLUMNS = [
    "group",
    "mouse",
    "day",
    "day_c",
    "trial_in_day",
    "cumulative_trial",
    "source_trial_index",
    "actual_left",
    "rewarded_left",
    "correct",
]

_MOVING_SUMMARY_COLUMNS = [
    "group",
    "cumulative_trial",
    "mean",
    "sem",
    "effective_mouse_n",
    "n_observation",
]
