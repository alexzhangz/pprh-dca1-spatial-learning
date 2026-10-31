"""Manuscript definitions collected from build_fig6j_adjacent_interday.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import matplotlib as mpl

from matplotlib import colors as mpl_colors

import numpy as np

import pandas as pd

from maze_analysis.manuscript_heatmap_style import (  # noqa: E402
    HEATMAP_STYLE_VERSION,
    PV_HEATMAP_CMAP,
    heatmap_panel_spec,
)

from figure_helpers.figure6 import plot_pv_heatmap

MPL_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

ADJACENT_PAIRS = (
    (1, 1, 2, "D1-D2"),
    (2, 2, 3, "D2-D3"),
    (3, 3, 4, "D3-D4"),
    (4, 4, 5, "D4-D5"),
)

SPATIAL_BINS = tuple(range(56))

FIG4I_SPEC = heatmap_panel_spec("Fig4I", expected_family="PV")

VMIN = float(FIG4I_SPEC.vmin)

VMAX = float(FIG4I_SPEC.vmax)

def build_displayed_cells(source: pd.DataFrame) -> pd.DataFrame:
    cells = (
        source.groupby(
            ["time_index", "day_a", "day_b", "day_pair_label", "spatial_bin"],
            as_index=False,
            observed=True,
        )["raw_r"]
        .mean()
        .rename(columns={"raw_r": "displayed_mean_raw_r"})
        .sort_values(["time_index", "spatial_bin"], kind="stable")
        .reset_index(drop=True)
    )
    expected_cells = len(ADJACENT_PAIRS) * len(SPATIAL_BINS)
    if len(cells) != expected_cells:
        raise ValueError(
            f"adjacent Fig. 6J displayed-cell count drifted: "
            f"{len(cells)} != {expected_cells}"
        )
    values = cells["displayed_mean_raw_r"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("adjacent Fig. 6J displayed matrix contains missing cells")
    if float(values.min()) < VMIN or float(values.max()) > VMAX:
        raise ValueError(
            "Fig. 4I-matched limits would clip adjacent Fig. 6J values: "
            f"observed [{values.min()}, {values.max()}], limits [{VMIN}, {VMAX}]"
        )
    return cells

def plot_adjacent_fig6j(source: pd.DataFrame):
    with mpl.rc_context(MPL_STYLE):
        figure = plot_pv_heatmap(
            source,
            groups=("P",),
            condition="CL",
            mode="adjacent",
            panel_letter="J",
            vmin=VMIN,
            vmax=VMAX,
        )
    images = [image for axis in figure.axes for image in axis.images]
    if len(images) != 1:
        raise AssertionError(f"expected one Fig. 6J image, observed {len(images)}")
    image = images[0]
    if image.get_cmap().name != PV_HEATMAP_CMAP:
        raise AssertionError("Fig. 6J PV colormap drifted")
    if not isinstance(image.norm, mpl_colors.Normalize) or hasattr(
        image.norm, "vcenter"
    ):
        raise AssertionError("Fig. 6J must use ordinary linear normalization")
    if (float(image.norm.vmin), float(image.norm.vmax)) != (VMIN, VMAX):
        raise AssertionError("Fig. 6J color limits drifted")
    return figure
