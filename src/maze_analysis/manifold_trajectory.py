"""Manuscript definitions collected from manifold_trajectory.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .plotting import DEFAULT_FIGURE_FORMATS, add_project_colorbar, configure_matplotlib, save_figure

CORRECT_LEFT_RIGHT_CONDITIONS = ("correct_left", "correct_right")

CORRECT_LEFT_RIGHT_LINESTYLES = {
    "correct_left": "solid",
    "correct_right": "dashed",
}

CORRECT_LEFT_RIGHT_LABELS = {
    "correct_left": "Correct left",
    "correct_right": "Correct right",
}

DEFAULT_TRAJECTORY_CMAP = "viridis"

DEFAULT_LEFT_POINT_CMAP = "viridis"

DEFAULT_RIGHT_POINT_CMAP = "plasma"

DEFAULT_TRAJECTORY_VIEW_ELEV = 40.0

DEFAULT_TRAJECTORY_VIEW_AZIM = -75.0

MAZE_LIKE_TOPDOWN_ELEV_THRESHOLD = 80.0

def compute_correct_left_right_trajectories(embedding, labels, *, n_bins=None, min_count=1):
    """Average saved embedding points into correct-left/right spatial-bin trajectories."""
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError(f"embedding must have shape (n_point, n_dim), got {embedding.shape}")

    position_bin = np.asarray(labels["position_bin"], dtype=int)
    actual_left = np.asarray(labels["actual_left"], dtype=bool)
    outcome = np.asarray(labels["outcome"], dtype=bool)
    n_point = embedding.shape[0]
    for name, values in {
        "position_bin": position_bin,
        "actual_left": actual_left,
        "outcome": outcome,
    }.items():
        if values.shape[:1] != (n_point,):
            raise ValueError(
                f"labels[{name!r}] must have one value per embedding point, "
                f"got {values.shape[:1]} for {n_point} points"
            )

    if n_bins is None:
        finite_bins = position_bin[np.isfinite(position_bin)]
        n_bins = int(np.max(finite_bins)) + 1 if finite_bins.size else 0
    n_bins = int(n_bins)
    min_count = int(min_count)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    conditions = {
        "correct_left": outcome & actual_left,
        "correct_right": outcome & ~actual_left,
    }
    return {
        name: _average_by_position_bin(
            embedding,
            position_bin,
            mask,
            n_bins=n_bins,
            min_count=min_count,
        )
        for name, mask in conditions.items()
    }

def compute_correct_left_right_points(embedding, labels):
    """Return individual saved embedding points for correct-left/right trials."""
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError(f"embedding must have shape (n_point, n_dim), got {embedding.shape}")

    position_bin = np.asarray(labels["position_bin"], dtype=int)
    actual_left = np.asarray(labels["actual_left"], dtype=bool)
    outcome = np.asarray(labels["outcome"], dtype=bool)
    n_point = embedding.shape[0]
    for name, values in {
        "position_bin": position_bin,
        "actual_left": actual_left,
        "outcome": outcome,
    }.items():
        if values.shape[:1] != (n_point,):
            raise ValueError(
                f"labels[{name!r}] must have one value per embedding point, "
                f"got {values.shape[:1]} for {n_point} points"
            )

    conditions = {
        "correct_left": outcome & actual_left,
        "correct_right": outcome & ~actual_left,
    }
    return {
        name: {
            "points": embedding[mask],
            "position_bin": position_bin[mask],
        }
        for name, mask in conditions.items()
    }

def _average_by_position_bin(embedding, position_bin, mask, *, n_bins, min_count):
    n_dim = int(embedding.shape[1])
    points = np.full((n_bins, n_dim), np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for bin_idx in range(n_bins):
        bin_mask = mask & (position_bin == bin_idx)
        counts[bin_idx] = int(np.sum(bin_mask))
        if counts[bin_idx] >= min_count:
            points[bin_idx] = np.nanmean(embedding[bin_mask], axis=0)
    return {"points": points, "counts": counts}

def load_correct_left_right_trajectory_session(session_dir, *, n_bins=None, min_count=1):
    """Load one saved embedding session and return correct-left/right trajectories."""
    session_dir = Path(session_dir)
    embedding = np.load(session_dir / "embedding.npy")
    with np.load(session_dir / "labels.npz") as labels_npz:
        labels = {name: labels_npz[name] for name in labels_npz.files}
    return compute_correct_left_right_trajectories(
        embedding,
        labels,
        n_bins=n_bins,
        min_count=min_count,
    )

def load_correct_left_right_point_session(session_dir):
    """Load one saved embedding session and return correct-left/right individual points."""
    session_dir = Path(session_dir)
    embedding = np.load(session_dir / "embedding.npy")
    with np.load(session_dir / "labels.npz") as labels_npz:
        labels = {name: labels_npz[name] for name in labels_npz.files}
    return compute_correct_left_right_points(embedding, labels)

def compute_shared_embedding_axis_limits(
    trajectories_collection,
    *,
    is_3d,
    lower_percentile=2.0,
    upper_percentile=98.0,
    padding_fraction=0.08,
):
    """Return robust shared axis limits for multiple trajectory dictionaries."""
    n_dim = 3 if is_3d else 2
    points = []
    for trajectories in trajectories_collection:
        for info in trajectories.values():
            values = np.asarray(info["points"], dtype=float)[:, :n_dim]
            finite = np.isfinite(values).all(axis=1)
            if finite.any():
                points.append(values[finite])
    if not points:
        return None
    stacked = np.vstack(points)
    limits = []
    for dim_idx in range(n_dim):
        dim_values = stacked[:, dim_idx]
        low = float(np.nanpercentile(dim_values, lower_percentile))
        high = float(np.nanpercentile(dim_values, upper_percentile))
        if np.isclose(low, high):
            pad = max(abs(low) * 0.05, 1e-6)
        else:
            pad = (high - low) * float(padding_fraction)
        limits.append((low - pad, high + pad))
    return tuple(limits)

def _spatial_bin_norm(collections):
    from matplotlib.colors import Normalize

    bins = []
    for point_sets in collections:
        for info in point_sets.values():
            if "position_bin" in info:
                values = np.asarray(info["position_bin"], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    bins.append(finite)
    if not bins:
        return Normalize(vmin=0, vmax=1)
    stacked = np.concatenate(bins)
    max_bin = max(float(np.nanmax(stacked)), 1.0)
    return Normalize(vmin=0, vmax=max_bin)

def _point_condition_cmaps(plt):
    return {
        "correct_left": plt.get_cmap(DEFAULT_LEFT_POINT_CMAP),
        "correct_right": plt.get_cmap(DEFAULT_RIGHT_POINT_CMAP),
    }

def plot_correct_left_right_trajectories(
    trajectories,
    *,
    title=None,
    view_elev=DEFAULT_TRAJECTORY_VIEW_ELEV,
    view_azim=DEFAULT_TRAJECTORY_VIEW_AZIM,
    trajectory_cmap=DEFAULT_TRAJECTORY_CMAP,
    show_colorbar=True,
):
    """Plot correct-left/right mean trajectories in a saved embedding space."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    configure_matplotlib(plt)
    first_points = next(iter(trajectories.values()))["points"]
    is_3d = np.asarray(first_points).shape[1] >= 3
    subplot_kw = {"projection": "3d"} if is_3d else None
    fig_size = (4.0, 3.0) if is_3d and show_colorbar else (3.2, 2.8)
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True, subplot_kw=subplot_kw)
    cmap = plt.get_cmap(trajectory_cmap)
    max_bin = max(
        int(np.asarray(info["points"]).shape[0]) - 1
        for info in trajectories.values()
    )
    norm = Normalize(vmin=0, vmax=max(max_bin, 1))
    legend_handles = []
    _plot_trajectories_on_axis(ax, trajectories, cmap=cmap, norm=norm, is_3d=is_3d)
    for condition in CORRECT_LEFT_RIGHT_CONDITIONS:
        label = CORRECT_LEFT_RIGHT_LABELS[condition]
        linestyle = CORRECT_LEFT_RIGHT_LINESTYLES[condition]
        legend_handles.append(Line2D([0], [0], color="0.25", linestyle=linestyle, linewidth=1.0, label=label))
    if title:
        ax.set_title(title)
    _set_embedding_axis_limits(ax, trajectories, is_3d=is_3d)
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    if is_3d:
        ax.set_zlabel("Embedding 3", labelpad=8)
        _apply_3d_camera(ax, view_elev=view_elev, view_azim=view_azim)
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="best")
    if show_colorbar:
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        add_project_colorbar(
            fig,
            mappable,
            ax,
            label="Spatial bin",
            pad=0.10 if is_3d else 0.02,
            fraction=0.05 if is_3d else 0.04,
        )
    return fig, ax

def plot_correct_left_right_points(
    point_sets,
    *,
    title=None,
    view_elev=DEFAULT_TRAJECTORY_VIEW_ELEV,
    view_azim=DEFAULT_TRAJECTORY_VIEW_AZIM,
    trajectory_cmap=None,
    show_colorbar=True,
):
    """Plot correct-left/right individual embedding points without averaging."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    configure_matplotlib(plt)
    first_points = next(iter(point_sets.values()))["points"]
    is_3d = np.asarray(first_points).shape[1] >= 3
    subplot_kw = {"projection": "3d"} if is_3d else None
    fig_size = (4.0, 3.0) if is_3d and show_colorbar else (3.2, 2.8)
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True, subplot_kw=subplot_kw)
    point_cmaps = _point_condition_cmaps(plt)
    norm = _spatial_bin_norm((point_sets,))
    _plot_points_on_axis(ax, point_sets, point_cmaps=point_cmaps, norm=norm, is_3d=is_3d)
    legend_handles = [
        Line2D([0], [0], color=point_cmaps["correct_left"](0.70), marker="o", linestyle="None", markersize=4, label="Correct left"),
        Line2D([0], [0], color=point_cmaps["correct_right"](0.70), marker="o", linestyle="None", markersize=4, label="Correct right"),
    ]
    if title:
        ax.set_title(title)
    axis_limits = compute_shared_embedding_axis_limits((point_sets,), is_3d=is_3d)
    _apply_axis_limits(ax, axis_limits, is_3d=is_3d)
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    if is_3d:
        ax.set_zlabel("Embedding 3", labelpad=8)
        _apply_3d_camera(ax, view_elev=view_elev, view_azim=view_azim)
    ax.legend(handles=legend_handles, frameon=False, loc="best")
    if show_colorbar:
        for idx, condition in enumerate(CORRECT_LEFT_RIGHT_CONDITIONS):
            mappable = ScalarMappable(norm=norm, cmap=point_cmaps[condition])
            mappable.set_array([])
            add_project_colorbar(
                fig,
                mappable,
                ax,
                label="Left spatial bin" if condition == "correct_left" else "Right spatial bin",
                pad=(0.10 if is_3d else 0.02) + idx * 0.08,
                fraction=0.035 if is_3d else 0.03,
            )
    return fig, ax

def _plot_trajectories_on_axis(ax, trajectories, *, cmap, norm, is_3d):
    for condition in CORRECT_LEFT_RIGHT_CONDITIONS:
        info = trajectories[condition]
        points = np.asarray(info["points"], dtype=float)
        finite = np.isfinite(points[:, :3 if is_3d else 2]).all(axis=1)
        if not finite.any():
            continue
        _plot_condition_trajectory(
            ax,
            points,
            finite,
            cmap=cmap,
            norm=norm,
            linestyle=CORRECT_LEFT_RIGHT_LINESTYLES[condition],
            is_3d=is_3d,
        )

def _plot_points_on_axis(ax, point_sets, *, point_cmaps, norm, is_3d):
    for condition in CORRECT_LEFT_RIGHT_CONDITIONS:
        info = point_sets[condition]
        points = np.asarray(info["points"], dtype=float)
        position_bin = np.asarray(info["position_bin"], dtype=float)
        finite = np.isfinite(points[:, :3 if is_3d else 2]).all(axis=1) & np.isfinite(position_bin)
        if not finite.any():
            continue
        marker = "o"
        cmap = point_cmaps[condition]
        if is_3d:
            ax.scatter(
                points[finite, 0],
                points[finite, 1],
                points[finite, 2],
                c=position_bin[finite],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=6,
                alpha=0.45,
                linewidths=0,
                depthshade=False,
            )
            ax.auto_scale_xyz(points[finite, 0], points[finite, 1], points[finite, 2])
        else:
            ax.scatter(
                points[finite, 0],
                points[finite, 1],
                c=position_bin[finite],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=8,
                alpha=0.45,
                linewidths=0,
            )

def _set_embedding_axis_limits(ax, trajectories, *, is_3d):
    points = []
    n_dim = 3 if is_3d else 2
    for info in trajectories.values():
        values = np.asarray(info["points"], dtype=float)[:, :n_dim]
        finite = np.isfinite(values).all(axis=1)
        if finite.any():
            points.append(values[finite])
    if not points:
        return
    stacked = np.vstack(points)
    limits = []
    for dim_idx in range(n_dim):
        dim_values = stacked[:, dim_idx]
        low = float(np.nanmin(dim_values))
        high = float(np.nanmax(dim_values))
        if np.isclose(low, high):
            pad = max(abs(low) * 0.05, 1e-6)
        else:
            pad = (high - low) * 0.08
        limits.append((low - pad, high + pad))
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    if is_3d:
        ax.set_zlim(*limits[2])

def _apply_axis_limits(ax, axis_limits, *, is_3d):
    if axis_limits is None:
        return
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    if is_3d:
        ax.set_zlim(*axis_limits[2])

def _apply_3d_camera(ax, *, view_elev, view_azim):
    ax.set_proj_type("ortho")
    ax.view_init(elev=view_elev, azim=view_azim)
    if float(view_elev) >= MAZE_LIKE_TOPDOWN_ELEV_THRESHOLD:
        ax.set_zlabel("")
        ax.set_zticks([])

def _plot_condition_trajectory(ax, points, finite, *, cmap, norm, linestyle, is_3d):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    if is_3d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segments = _finite_segments(finite)
    for segment in segments:
        segment_points = points[segment]
        if segment_points.shape[0] < 2:
            continue
        bin_indices = np.arange(segment.start, segment.stop, dtype=float)
        line_segments = np.stack([segment_points[:-1], segment_points[1:]], axis=1)
        segment_colors = bin_indices[:-1] + 0.5
        if is_3d:
            collection = Line3DCollection(
                line_segments[:, :, :3],
                cmap=cmap,
                norm=norm,
                linewidths=1.0,
                linestyles=linestyle,
            )
            collection.set_array(segment_colors)
            ax.add_collection(collection)
            ax.auto_scale_xyz(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2])
            ax.scatter(
                segment_points[0, 0],
                segment_points[0, 1],
                segment_points[0, 2],
                color=cmap(norm(bin_indices[0])),
                marker="s",
                s=18,
                linewidths=0,
            )
            ax.scatter(
                segment_points[-1, 0],
                segment_points[-1, 1],
                segment_points[-1, 2],
                color=cmap(norm(bin_indices[-1])),
                marker="x",
                s=22,
                linewidths=0.8,
            )
            _add_direction_arrow_3d(ax, segment_points, bin_indices, cmap=cmap, norm=norm)
        else:
            collection = LineCollection(
                line_segments[:, :, :2],
                cmap=cmap,
                norm=norm,
                linewidths=1.0,
                linestyles=linestyle,
            )
            collection.set_array(segment_colors)
            ax.add_collection(collection)
            ax.update_datalim(segment_points[:, :2])
            ax.autoscale_view()
            ax.plot(
                segment_points[0, 0],
                segment_points[0, 1],
                marker="s",
                linestyle="None",
                color=cmap(norm(bin_indices[0])),
                markersize=4,
                markeredgewidth=0,
            )
            ax.plot(
                segment_points[-1, 0],
                segment_points[-1, 1],
                marker="x",
                linestyle="None",
                color=cmap(norm(bin_indices[-1])),
                markersize=5,
                markeredgewidth=0.8,
            )
            arrow_start, arrow_end, arrow_color = _direction_arrow_points(segment_points, bin_indices, cmap, norm)
            ax.add_patch(
                FancyArrowPatch(
                    arrow_start[:2],
                    arrow_end[:2],
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.8,
                    color=arrow_color,
                    shrinkA=0,
                    shrinkB=0,
                )
            )

def _direction_arrow_points(segment_points, bin_indices, cmap, norm):
    arrow_idx = max(0, int((segment_points.shape[0] - 1) * 0.45))
    arrow_step = max(1, segment_points.shape[0] // 8)
    arrow_end_idx = min(segment_points.shape[0] - 1, arrow_idx + arrow_step)
    if arrow_end_idx == arrow_idx and arrow_idx > 0:
        arrow_idx -= 1
    arrow_start = segment_points[arrow_idx]
    arrow_end = segment_points[arrow_end_idx]
    arrow_color = cmap(norm((bin_indices[arrow_idx] + bin_indices[arrow_end_idx]) / 2))
    return arrow_start, arrow_end, arrow_color

def _add_direction_arrow_3d(ax, segment_points, bin_indices, *, cmap, norm):
    arrow_start, arrow_end, arrow_color = _direction_arrow_points(segment_points, bin_indices, cmap, norm)
    delta = arrow_end - arrow_start
    length = _direction_arrow_length(segment_points)
    ax.quiver(
        arrow_start[0],
        arrow_start[1],
        arrow_start[2],
        delta[0],
        delta[1],
        delta[2],
        color=arrow_color,
        arrow_length_ratio=0.35,
        length=length,
        linewidth=0.8,
        normalize=True,
    )

def _direction_arrow_length(segment_points):
    span = np.nanmax(segment_points[:, :3], axis=0) - np.nanmin(segment_points[:, :3], axis=0)
    scale = float(np.linalg.norm(span))
    return max(scale * 0.08, 1e-12)

def _finite_segments(finite):
    finite = np.asarray(finite, dtype=bool)
    segments = []
    start = None
    for idx, value in enumerate(finite):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append(slice(start, idx))
            start = None
    if start is not None:
        segments.append(slice(start, finite.size))
    return segments
