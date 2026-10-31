"""Manuscript definitions collected from pprh_manifold_heatmap_plotting.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.image import AxesImage

import numpy as np

import pandas as pd

from .fig7_figure8_pprh_plotting import GROUP_COLORS, MPL_STYLE

from .manuscript_heatmap_style import NEURAL_MANIFOLD_HEATMAP_CMAP

DAYS = (1, 2, 3, 4, 5)

PPRH_GROUP = "P"

PPRH_LABEL = "pPRH"

MANIFOLD_COLORBAR_LABEL = "z-scored d\u2032"

def build_pprh_group_mean_matrix(
    table: pd.DataFrame,
    *,
    n_bins: int,
    days: Sequence[int] = DAYS,
    group_col: str = "group",
    mouse_col: str = "mouse",
    day_col: str = "day",
    bin_col: str = "spatial_bin",
    value_col: str = "value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the displayed bin-by-day mean matrix and per-cell mouse support."""

    required = {group_col, mouse_col, day_col, bin_col, value_col}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"pPRH manifold table is missing columns: {missing}")
    days = tuple(int(day) for day in days)
    if days != DAYS:
        raise ValueError("pPRH manuscript heatmaps must display D1-D5 in order")
    if int(n_bins) <= 1:
        raise ValueError("n_bins must be greater than one")

    selected = table.loc[
        table[group_col].astype(str).eq(PPRH_GROUP)
        & pd.to_numeric(table[day_col], errors="coerce").isin(days)
    ].copy()
    selected[day_col] = pd.to_numeric(
        selected[day_col], errors="raise"
    ).astype(int)
    selected[bin_col] = pd.to_numeric(
        selected[bin_col], errors="raise"
    ).astype(int)
    selected[value_col] = pd.to_numeric(selected[value_col], errors="coerce")
    if selected.empty:
        raise ValueError("pPRH manifold table contains no requested observations")
    if selected.duplicated([mouse_col, day_col, bin_col]).any():
        raise ValueError("pPRH manifold table has duplicate mouse/day/bin rows")
    if not np.isfinite(selected[value_col]).all():
        raise ValueError("pPRH manifold table contains nonfinite displayed values")
    expected_bins = tuple(range(int(n_bins)))
    if tuple(sorted(selected[bin_col].unique())) != expected_bins:
        raise ValueError("pPRH manifold table has unexpected spatial-bin support")

    grouped = selected.groupby(
        [bin_col, day_col], observed=False, sort=True
    )[value_col]
    matrix = (
        grouped.mean()
        .unstack(day_col)
        .reindex(index=expected_bins, columns=days)
    )
    support = (
        grouped.count()
        .unstack(day_col)
        .reindex(index=expected_bins, columns=days)
    )
    if matrix.shape != (int(n_bins), len(days)) or matrix.isna().any().any():
        raise ValueError("pPRH manifold displayed matrix is incomplete")
    if support.shape != matrix.shape or support.lt(1).any().any():
        raise ValueError("pPRH manifold displayed support is incomplete")
    return matrix, support

def plot_pprh_manifold_heatmap(
    table: pd.DataFrame,
    *,
    n_bins: int,
    boundaries: Sequence[int],
    vmin: float,
    vmax: float,
    figsize: tuple[float, float] = (2.15, 2.10),
):
    """Draw one pPRH-only D1-D5 neural-manifold heatmap.

    The dense matrix is a self-contained raster layer in SVG/PDF, while text,
    axes and maze boundaries remain vector objects.  This matches the accepted
    Fig. 5/Fig. 7 hybrid-layered contract and avoids vector polygon seams.
    """

    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
        raise ValueError("heatmap endpoints must be finite with vmin < vmax")
    if not float(vmin) < 0.0 < float(vmax):
        raise ValueError("neural-manifold heatmap range must include zero")
    boundaries = tuple(int(boundary) for boundary in boundaries)
    if any(boundary <= 0 or boundary >= int(n_bins) for boundary in boundaries):
        raise ValueError("heatmap boundaries must fall inside the spatial-bin range")

    matrix, _support = build_pprh_group_mean_matrix(table, n_bins=int(n_bins))
    values = matrix.to_numpy(dtype=float)
    norm = mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax))

    with mpl.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.23, right=0.78, bottom=0.22, top=0.87)
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=NEURAL_MANIFOLD_HEATMAP_CMAP,
            norm=norm,
            rasterized=True,
        )
        image.set_gid("pprh_neural_manifold_heatmap_matrix")
        ax.set_gid("pprh_neural_manifold_heatmap_axis")
        for index, boundary in enumerate(boundaries, start=1):
            line = ax.axhline(
                float(boundary) - 0.5,
                color="black",
                linewidth=0.5,
                linestyle="--",
                alpha=0.85,
            )
            line.set_gid(f"maze_section_boundary_{index}")
        ax.set_xlim(-0.5, len(DAYS) - 0.5)
        ax.set_ylim(-0.5, int(n_bins) - 0.5)
        ax.set_xticks(np.arange(len(DAYS)))
        ax.set_xticklabels([f"D{day}" for day in DAYS])
        ax.set_yticks([0, int(n_bins) - 1])
        ax.set_yticklabels(["0", "1"])
        ax.set_xlabel("Day")
        ax.set_ylabel("Spatial bin")
        ax.set_title(PPRH_LABEL, color=GROUP_COLORS[PPRH_GROUP], pad=3)
        ax.tick_params(axis="both", direction="out", length=2, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(GROUP_COLORS[PPRH_GROUP])

        colorbar = fig.colorbar(image, ax=ax, fraction=0.09, pad=0.07)
        colorbar.set_ticks((float(vmin), 0.0, float(vmax)))
        colorbar.set_label(MANIFOLD_COLORBAR_LABEL)
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(width=0.5, length=2, direction="out")
        colorbar.ax.set_gid("neural_manifold_colorbar")
    return fig, ax, colorbar

def heatmap_image(axis) -> AxesImage:
    """Return the unique hybrid-layer heatmap image from a plotted data axis."""

    images = [artist for artist in axis.images if isinstance(artist, AxesImage)]
    if len(images) != 1:
        raise ValueError(f"expected one heatmap image, observed {len(images)}")
    return images[0]
