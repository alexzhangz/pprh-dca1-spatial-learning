"""Manuscript definitions collected from plot_tmaze_sample_choice_manifold_trajectory.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import math

import warnings

from pathlib import Path

from typing import Mapping

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.collections import LineCollection

from sklearn.manifold import SpectralEmbedding

from t_maze.manifold import (  # noqa: E402
    CONDITION_ORDER,
    build_tmaze_manifold_condition_trials,
)

CONDITIONS = ("sample", "choice")

CONDITION_CMAPS = {"sample": "inferno", "choice": "magma"}

PANEL_SIZE = 300.0

SEED = 2026

MIN_FINITE_FRACTION = 0.80

def load_condition_trials(tmaze_root: str | Path, mouse: str, day: int) -> dict[str, np.ndarray]:
    subject_dir = Path(tmaze_root) / "Data" / "Aligened_data" / str(mouse)
    maps = np.load(subject_dir / "Snarm_trial_v2.npy", allow_pickle=True).item()
    index_all = np.load(subject_dir / "index_all.npy", allow_pickle=True)
    day_idx = int(day) - 1
    return build_tmaze_manifold_condition_trials(
        maps["S"][day_idx],
        maps["C"][day_idx],
        index_all,
        day_idx=day_idx,
    )

def balance_sample_choice_trials(
    condition_trials: Mapping[str, np.ndarray], *, seed: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Select equal counts of SL/SR/CL/CR without replacement."""
    missing = [label for label in CONDITION_ORDER if label not in condition_trials]
    if missing:
        raise KeyError(f"missing condition arrays: {missing}")
    arrays = {label: np.asarray(condition_trials[label], dtype=float) for label in CONDITION_ORDER}
    shapes = {values.shape[1:] for values in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"condition arrays must share bin/neuron axes, got {shapes}")
    balance_count = min(values.shape[0] for values in arrays.values())
    if balance_count < 1:
        raise ValueError("all SL/SR/CL/CR conditions require at least one correct trial")
    rng = np.random.default_rng(int(seed))
    selected = {}
    rows = []
    for label in CONDITION_ORDER:
        values = arrays[label]
        if values.shape[0] == balance_count:
            indices = np.arange(balance_count, dtype=int)
        else:
            indices = np.sort(rng.choice(values.shape[0], size=balance_count, replace=False))
        selected[label] = values[indices]
        phase = "sample" if label.startswith("S") else "choice"
        route = "left" if label.endswith("L") else "right"
        rows.extend(
            {
                "condition_label": label,
                "phase": phase,
                "route": route,
                "source_trial_index": int(index),
            }
            for index in indices
        )
    return selected, pd.DataFrame(rows)

def _zscore_columns(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mean = np.nanmean(points, axis=0, keepdims=True)
    std = np.nanstd(points, axis=0, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return np.nan_to_num((points - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

def _spectral_embedding_3d(
    points: np.ndarray, *, seed: int, n_neighbors: int = 0
) -> tuple[np.ndarray, int]:
    if int(n_neighbors) <= 0:
        n_neighbors = int(np.ceil(np.sqrt(points.shape[0])))
    n_neighbors = min(int(n_neighbors), max(int(points.shape[0]) - 1, 1))
    model = SpectralEmbedding(
        n_components=3,
        n_neighbors=n_neighbors,
        n_jobs=1,
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Graph is not fully connected.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*", category=UserWarning)
        return np.asarray(model.fit_transform(points), dtype=float), int(n_neighbors)

def fit_sample_choice_embedding(
    selected_by_route: Mapping[str, np.ndarray],
    *,
    target_neuron_count: int = 0,
    min_finite_fraction: float = MIN_FINITE_FRACTION,
    seed: int = SEED,
    n_neighbors: int = 0,
) -> dict:
    """Jointly embed route-balanced sample and choice trial-bin samples."""
    arrays = {label: np.asarray(selected_by_route[label], dtype=float) for label in CONDITION_ORDER}
    pooled_routes = np.concatenate([arrays[label] for label in CONDITION_ORDER], axis=0)
    n_trial, n_bin, _ = pooled_routes.shape
    finite_fraction = np.isfinite(pooled_routes).sum(axis=(0, 1)) / float(n_trial * n_bin)
    candidates = np.where(finite_fraction >= float(min_finite_fraction))[0]
    if candidates.size < 3:
        raise ValueError(f"too few finite neurons for 3D embedding: {candidates.size}")
    rng = np.random.default_rng(int(seed))
    target = int(target_neuron_count)
    if target > 0 and candidates.size > target:
        selected_neurons = np.sort(rng.choice(candidates, size=target, replace=False))
    else:
        selected_neurons = candidates

    phase_arrays = {
        "sample": np.concatenate([arrays["SL"], arrays["SR"]], axis=0),
        "choice": np.concatenate([arrays["CL"], arrays["CR"]], axis=0),
    }
    pooled_phases = np.concatenate([phase_arrays[name] for name in CONDITIONS], axis=0)
    flat = pooled_phases[:, :, selected_neurons].reshape(-1, selected_neurons.size)
    embedding, effective_neighbors = _spectral_embedding_3d(
        _zscore_columns(flat), seed=int(seed), n_neighbors=int(n_neighbors)
    )

    conditions = {}
    offset = 0
    point_rows = []
    for condition in CONDITIONS:
        values = phase_arrays[condition]
        n_condition_trials = int(values.shape[0])
        n_points = n_condition_trials * n_bin
        points = embedding[offset : offset + n_points]
        bins = np.tile(np.arange(n_bin, dtype=int), n_condition_trials)
        trial_ids = np.repeat(np.arange(n_condition_trials, dtype=int), n_bin)
        conditions[condition] = {"points": points, "bins": bins, "trial_ids": trial_ids}
        point_rows.extend(
            {
                "condition": condition,
                "phase_trial_index": int(trial_id),
                "spatial_bin": int(bin_value),
                "spectral_1": float(point[0]),
                "spectral_2": float(point[1]),
                "spectral_3": float(point[2]),
            }
            for point, bin_value, trial_id in zip(points, bins, trial_ids)
        )
        offset += n_points
    return {
        "conditions": conditions,
        "points_table": pd.DataFrame(point_rows),
        "candidate_neuron_count": int(candidates.size),
        "selected_neuron_count": int(selected_neurons.size),
        "selected_neuron_indices": selected_neurons,
        "n_bins": int(n_bin),
        "n_neighbors": int(effective_neighbors),
    }

def mean_trajectories(conditions: Mapping[str, Mapping]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    trajectories = {}
    rows = []
    for condition in CONDITIONS:
        points = np.asarray(conditions[condition]["points"], dtype=float)
        bins = np.asarray(conditions[condition]["bins"], dtype=int)
        unique_bins = np.unique(bins)
        means = np.vstack([np.mean(points[bins == value], axis=0) for value in unique_bins])
        trajectories[condition] = means
        rows.extend(
            {
                "condition": condition,
                "spatial_bin": int(bin_value),
                "mean_spectral_1": float(point[0]),
                "mean_spectral_2": float(point[1]),
                "mean_spectral_3": float(point[2]),
                "n_points": int(np.sum(bins == bin_value)),
            }
            for bin_value, point in zip(unique_bins, means)
        )
    return trajectories, pd.DataFrame(rows)

def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if np.isfinite(norm) and norm > 1e-12 else np.array([1.0, 0.0, 0.0])

def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])

def _axis_angle(axis, angle: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, dtype=float))
    half = angle / 2.0
    return _normalize(np.array([math.cos(half), *(axis * math.sin(half))]))

def default_rotation() -> np.ndarray:
    return _normalize(_quat_multiply(_axis_angle([1, 0, 0], 0.72), _axis_angle([0, 0, 1], -0.78)))

def _rotate(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    qv = quaternion[1:]
    uv = np.cross(qv, vector)
    uuv = np.cross(qv, uv)
    return vector + 2.0 * (quaternion[0] * uv + uuv)

def _bounds(conditions: Mapping[str, Mapping]) -> tuple[np.ndarray, float]:
    points = np.vstack([np.asarray(conditions[name]["points"], dtype=float) for name in CONDITIONS])
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return (lower + upper) / 2.0, max(float(np.max(upper - lower)), 1e-9)

def _raw_project(points: np.ndarray, bounds: tuple[np.ndarray, float]) -> np.ndarray:
    center, span = bounds
    quaternion = default_rotation()
    rotated = np.vstack([_rotate((point - center) / span, quaternion) for point in points])
    xy = rotated[:, :2] * (PANEL_SIZE * 0.78)
    xy[:, 0] += PANEL_SIZE / 2.0
    xy[:, 1] = PANEL_SIZE / 2.0 - xy[:, 1]
    return np.column_stack([xy, rotated[:, 2]])

def _projection_contract(conditions: Mapping[str, Mapping]) -> dict:
    bounds = _bounds(conditions)
    all_points = np.vstack([np.asarray(conditions[name]["points"], dtype=float) for name in CONDITIONS])
    projected = _raw_project(all_points, bounds)
    xy = projected[:, :2]
    xy_center = np.array(
        [(xy[:, 0].max() + xy[:, 0].min()) / 2, (xy[:, 1].max() + xy[:, 1].min()) / 2]
    )
    scale = PANEL_SIZE * 0.90 / max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)
    return {"bounds": bounds, "xy_center": xy_center, "scale": scale}

def project_latent(points: np.ndarray, *, projection: Mapping) -> np.ndarray:
    raw = _raw_project(np.asarray(points, dtype=float), projection["bounds"])
    raw[:, :2] = PANEL_SIZE / 2.0 + (raw[:, :2] - projection["xy_center"]) * projection["scale"]
    return raw

def _cmap(condition: str):
    return plt.get_cmap(CONDITION_CMAPS[condition])

def _style_axis(ax) -> None:
    ax.set_xlim(0, PANEL_SIZE)
    ax.set_ylim(PANEL_SIZE, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

def draw_point_cloud(ax, conditions: Mapping[str, Mapping], max_bin: int) -> None:
    projection = _projection_contract(conditions)
    records = []
    for condition in CONDITIONS:
        points = np.asarray(conditions[condition]["points"], dtype=float)
        projected = project_latent(points, projection=projection)
        bins = np.asarray(conditions[condition]["bins"], dtype=int)
        records.extend(zip(projected, bins, [condition] * len(bins)))
    for condition in CONDITIONS:
        subset = [(point, bin_value) for point, bin_value, name in records if name == condition]
        subset.sort(key=lambda item: item[0][2])
        ax.scatter(
            [item[0][0] for item in subset],
            [item[0][1] for item in subset],
            c=[item[1] / max_bin for item in subset],
            cmap=_cmap(condition),
            vmin=0,
            vmax=1,
            s=3.5,
            alpha=0.55,
            linewidths=0,
            rasterized=False,
        )
    _style_axis(ax)

def draw_mean_trajectories(
    ax,
    conditions: Mapping[str, Mapping],
    trajectories: Mapping[str, np.ndarray],
    max_bin: int,
) -> None:
    projection = _projection_contract(conditions)
    for condition in CONDITIONS:
        projected = project_latent(np.asarray(trajectories[condition]), projection=projection)
        bins = np.arange(projected.shape[0], dtype=int)
        segments = [[projected[i, :2], projected[i + 1, :2]] for i in range(len(projected) - 1)]
        colors = [_cmap(condition)(((bins[i] + bins[i + 1]) / 2) / max_bin) for i in range(len(segments))]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.4, alpha=0.92))
        ax.scatter(*projected[0, :2], marker="s", s=18, c=[_cmap(condition)(0.0)], linewidths=0)
        ax.scatter(*projected[-1, :2], marker="x", s=24, c=[_cmap(condition)(1.0)], linewidths=1.0)
        start = max(0, int((len(projected) - 1) * 0.45))
        end = min(len(projected) - 1, start + max(1, len(projected) // 8))
        color = _cmap(condition)(((bins[start] + bins[end]) / 2) / max_bin)
        ax.annotate(
            "",
            xy=projected[end, :2],
            xytext=projected[start, :2],
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.0, "mutation_scale": 7},
        )
    _style_axis(ax)
