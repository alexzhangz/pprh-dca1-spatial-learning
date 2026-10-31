"""Manuscript definitions collected from behavior_covariates.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

import pandas as pd

DEFAULT_SAMPLING_RATE_HZ = 25.0

DEFAULT_SPEED_THRESHOLD_CM_S = 1.0

DEFAULT_MIN_CONSECUTIVE_FRAMES = 15

CANONICAL_COVARIATE_COLUMNS = (
    "mean_speed_cm_s",
    "route_relative_lateral_deviation_cm",
)

CANONICAL_Z_COLUMNS = {
    "mean_speed_cm_s": "speed_z",
    "route_relative_lateral_deviation_cm": "lateral_deviation_z",
}

def framewise_speed_cm_s(
    xy,
    *,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    coordinate_units_per_cm: float = 1.0,
    short_trace_value: float = np.nan,
) -> np.ndarray:
    """Return gradient-based framewise Euclidean speed in cm/s.

    ``coordinate_units_per_cm`` allows a task adapter to retain raw coordinate
    units while exposing the same cm/s output.  For example, T-maze coordinates
    use 10 raw units per cm whereas the Figure-8 coordinates are already in cm.
    """

    points = validate_xy(xy)
    sampling_rate_hz = float(sampling_rate_hz)
    coordinate_units_per_cm = float(coordinate_units_per_cm)
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and > 0")
    if not np.isfinite(coordinate_units_per_cm) or coordinate_units_per_cm <= 0:
        raise ValueError("coordinate_units_per_cm must be finite and > 0")
    if points.shape[0] < 2:
        return np.full(points.shape[0], float(short_trace_value), dtype=float)
    vx = np.gradient(points[:, 0]) * sampling_rate_hz / coordinate_units_per_cm
    vy = np.gradient(points[:, 1]) * sampling_rate_hz / coordinate_units_per_cm
    return np.hypot(vx, vy)

def sustained_running_mask(
    speed,
    *,
    threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    min_consecutive_frames: int = DEFAULT_MIN_CONSECUTIVE_FRAMES,
) -> np.ndarray:
    """Keep every frame in complete above-threshold runs of sufficient length."""

    threshold_cm_s = float(threshold_cm_s)
    min_consecutive_frames = int(min_consecutive_frames)
    if not np.isfinite(threshold_cm_s):
        raise ValueError("threshold_cm_s must be finite")
    if min_consecutive_frames < 1:
        raise ValueError("min_consecutive_frames must be >= 1")

    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 1:
        raise ValueError(f"speed must be one-dimensional, got {speed.shape}")
    selected = np.isfinite(speed) & (speed >= threshold_cm_s)
    keep = np.zeros(selected.shape, dtype=bool)
    run_start = None
    for index, is_selected in enumerate(selected):
        if is_selected and run_start is None:
            run_start = index
        elif not is_selected and run_start is not None:
            if index - run_start >= min_consecutive_frames:
                keep[run_start:index] = True
            run_start = None
    if run_start is not None and selected.size - run_start >= min_consecutive_frames:
        keep[run_start:] = True
    return keep

def point_to_polyline_distance(points, vertices) -> np.ndarray:
    """Return minimum absolute orthogonal distance to a finite polyline.

    Projection onto each line segment is clamped to the segment endpoints.
    Non-finite points return NaN; vertices must be finite.  Degenerate segments
    are ignored as long as at least one non-degenerate segment remains.
    """

    points = validate_xy(points)
    vertices = validate_xy(vertices, name="vertices")
    if vertices.shape[0] < 2:
        raise ValueError("vertices must contain at least two points")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)

    starts = vertices[:-1]
    vectors = vertices[1:] - starts
    squared_lengths = np.sum(vectors**2, axis=1)
    nondegenerate = squared_lengths > 0
    if not np.any(nondegenerate):
        raise ValueError("polyline must contain at least one non-degenerate segment")
    starts = starts[nondegenerate]
    vectors = vectors[nondegenerate]
    squared_lengths = squared_lengths[nondegenerate]

    result = np.full(points.shape[0], np.nan, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return result
    selected = points[finite]
    delta = selected[:, np.newaxis, :] - starts[np.newaxis, :, :]
    projection = np.sum(delta * vectors[np.newaxis, :, :], axis=2)
    projection = projection / squared_lengths[np.newaxis, :]
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[np.newaxis, :, :] + projection[:, :, np.newaxis] * vectors
    distances = np.linalg.norm(selected[:, np.newaxis, :] - closest, axis=2)
    result[finite] = np.min(distances, axis=1)
    return result

def pool_finite_frame_values(parts: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate finite frame values without averaging within trials first."""

    finite_parts = []
    for part in parts:
        values = np.asarray(part, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            finite_parts.append(values)
    return np.concatenate(finite_parts) if finite_parts else np.empty(0, dtype=float)

def summarize_frame_weighted_covariates(
    speed_parts: Iterable[np.ndarray],
    lateral_deviation_parts: Iterable[np.ndarray],
    *,
    raw_frame_count: int,
    running_frame_count: int,
    valid_frame_count: int | None = None,
    empty_status: str = "no_valid_frames",
) -> dict:
    """Pool framewise values and emit the canonical value/QC contract.

    Speed and lateral values are pooled independently, preserving the source
    task's eligibility rules.  Consequently this function never introduces a
    mean-of-trial-means weighting artifact.
    """

    raw_frame_count = _nonnegative_count(raw_frame_count, "raw_frame_count")
    running_frame_count = _nonnegative_count(running_frame_count, "running_frame_count")
    if running_frame_count > raw_frame_count:
        raise ValueError("running_frame_count cannot exceed raw_frame_count")

    speed = pool_finite_frame_values(speed_parts)
    lateral = pool_finite_frame_values(lateral_deviation_parts)
    if valid_frame_count is None:
        valid_frame_count = min(speed.size, lateral.size)
    valid_frame_count = _nonnegative_count(valid_frame_count, "valid_frame_count")
    if valid_frame_count > running_frame_count:
        raise ValueError("valid_frame_count cannot exceed running_frame_count")

    if speed.size and lateral.size:
        status = "ok"
    elif not speed.size and not lateral.size:
        status = str(empty_status)
    elif not speed.size:
        status = "no_speed_frames"
    else:
        status = "no_lateral_frames"

    return {
        "mean_speed_cm_s": float(np.mean(speed)) if speed.size else np.nan,
        "route_relative_lateral_deviation_cm": (
            float(np.mean(lateral)) if lateral.size else np.nan
        ),
        "covariate_raw_frame_count": raw_frame_count,
        "covariate_running_frame_count": running_frame_count,
        "covariate_valid_frame_count": valid_frame_count,
        "covariate_speed_frame_count": int(speed.size),
        "covariate_lateral_frame_count": int(lateral.size),
        "covariate_status": status,
    }

def empty_frame_weighted_covariates(status: str) -> dict:
    """Return an empty row following the canonical value/QC contract."""

    return {
        "mean_speed_cm_s": np.nan,
        "route_relative_lateral_deviation_cm": np.nan,
        "covariate_raw_frame_count": 0,
        "covariate_running_frame_count": 0,
        "covariate_valid_frame_count": 0,
        "covariate_speed_frame_count": 0,
        "covariate_lateral_frame_count": 0,
        "covariate_status": str(status),
    }

def pooled_z_standardize_covariates(
    table: pd.DataFrame,
    *,
    columns: Sequence[str] = CANONICAL_COVARIATE_COLUMNS,
    output_columns: Mapping[str, str] = CANONICAL_Z_COLUMNS,
    reference_mask=None,
    ddof: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-standardize covariates against one pooled joint-complete reference.

    The default reference is every row finite for *all* requested covariates.
    A caller can supply ``reference_mask`` to restrict that pool to the exact
    cohort/day/contrast rows entering a model.  The returned QC table records
    the reference distribution and formula inputs for Methods provenance.
    """

    columns = tuple(str(column) for column in columns)
    if not columns:
        raise ValueError("columns must not be empty")
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"covariate table is missing columns: {missing}")
    missing_outputs = [column for column in columns if column not in output_columns]
    if missing_outputs:
        raise ValueError(f"output_columns has no names for: {missing_outputs}")
    ddof = int(ddof)
    if ddof < 0:
        raise ValueError("ddof must be >= 0")

    result = table.copy()
    numeric = {
        column: pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        for column in columns
    }
    joint_finite = np.logical_and.reduce([np.isfinite(numeric[column]) for column in columns])
    if reference_mask is None:
        requested_reference = np.ones(len(result), dtype=bool)
    else:
        requested_reference = np.asarray(reference_mask, dtype=bool)
        if requested_reference.ndim != 1 or requested_reference.shape[0] != len(result):
            raise ValueError("reference_mask must be one-dimensional with one value per row")
    reference = requested_reference & joint_finite
    reference_n = int(np.sum(reference))

    qc_rows = []
    for column in columns:
        output_column = str(output_columns[column])
        values = numeric[column]
        z = np.full(values.shape, np.nan, dtype=float)
        if reference_n <= ddof:
            mean = np.nan
            scale = np.nan
            status = "insufficient_reference"
        else:
            reference_values = values[reference]
            mean = float(np.mean(reference_values))
            scale = float(np.std(reference_values, ddof=ddof))
            if not np.isfinite(scale) or scale <= 0:
                status = "zero_variance"
            else:
                finite = np.isfinite(values)
                z[finite] = (values[finite] - mean) / scale
                status = "ok"
        result[output_column] = z
        qc_rows.append(
            {
                "source_column": column,
                "z_column": output_column,
                "reference_scope": "pooled_joint_complete_case",
                "reference_n": reference_n,
                "reference_mean": mean,
                "reference_sd": scale,
                "ddof": ddof,
                "reference_requested_n": int(np.sum(requested_reference)),
                "reference_excluded_nonfinite_n": int(
                    np.sum(requested_reference & ~joint_finite)
                ),
                "transformed_finite_n": int(np.sum(np.isfinite(z))),
                "status": status,
                "formula": "z=(x-reference_mean)/reference_sd",
            }
        )
    return result, pd.DataFrame(qc_rows)

def validate_xy(xy, *, name: str = "xy") -> np.ndarray:
    """Validate and return an ``(n_frame, 2)`` floating-point array."""

    result = np.asarray(xy, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_frame, 2), got {result.shape}")
    return result

def _nonnegative_count(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value
