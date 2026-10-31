"""Manuscript definitions collected from build_manuscript_fig6_figs3_stability.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from maze_analysis.manuscript_heatmap_style import (  # noqa: E402
    HEATMAP_STYLE_VERSION,
    PV_HEATMAP_CMAP,
    heatmap_panel_spec,
)

GROUP_COLORS = {
    "C": "#7ED3FC",
    "I": "#B39DCA",
    "P": "#9BCD88",
}

FIG6_LABELS = {"C": "dCA1", "P": "pPRH"}

FIGS3_LABELS = {
    "C": r"pPRH$^{\mathrm{dCA1}}$-Con",
    "I": r"pPRH$^{\mathrm{dCA1}}$-In",
}

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

def plot_pv_heatmap(
    table: pd.DataFrame,
    *,
    groups: Sequence[str],
    condition: str,
    mode: str,
    panel_letter: str,
    vmin: float,
    vmax: float,
):
    table = table.loc[table["condition"].astype(str).eq(str(condition))].copy()
    time_col = "day" if mode == "within" else "time_index"
    x_values = (1, 2, 3, 4, 5) if mode == "within" else (1, 2, 3, 4)
    fig, axes = plt.subplots(len(groups), 1, figsize=(3.2, 2.15 * len(groups)), squeeze=False)
    axes = axes[:, 0]
    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
        raise ValueError("PV heatmap requires explicit finite vmin < vmax")
    if mode == "within":
        labels = [str(value) for value in x_values]
        xlabel = "Day"
    elif mode == "adjacent":
        labels = ["D1-D2", "D2-D3", "D3-D4", "D4-D5"]
        xlabel = "Adjacent-day pair"
    else:
        labels = ["D1-D2", "D1-D3", "D1-D4", "D1-D5"]
        xlabel = "Day-1-anchored pair"
    image = None
    for axis_index, (ax, group) in enumerate(zip(axes, groups, strict=True)):
        subset = table.loc[table["group"].eq(group)].copy()
        summary = subset.groupby(["spatial_bin", time_col], observed=True)["raw_r"].mean().unstack(time_col)
        summary = summary.reindex(index=range(int(subset["spatial_bin"].max()) + 1), columns=x_values)
        image = ax.imshow(
            summary.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=PV_HEATMAP_CMAP,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels(labels, rotation=0)
        # In vertically stacked heatmaps the lower axis already carries the
        # shared x label.  Omitting the duplicate upper label prevents it from
        # colliding with the lower heatmap while retaining all pair tick labels.
        ax.set_xlabel(xlabel if axis_index == len(axes) - 1 else "")
        ax.set_ylabel(f"{FIG6_LABELS.get(group, FIGS3_LABELS.get(group, group))}\nSpatial bin", color=GROUP_COLORS[group])
        if int(summary.shape[0]) == 56:
            for boundary in (13.5, 27.5, 41.5):
                ax.axhline(boundary, color="black", linewidth=0.5, linestyle="--", alpha=0.7)
        elif int(summary.shape[0]) == 29:
            ax.axhline(12.5, color="black", linewidth=0.5, linestyle="--", alpha=0.7)
        _box_axis(ax)
    if len(axes) > 1 and (
        any(axis.get_xlabel() for axis in axes[:-1])
        or axes[-1].get_xlabel() != xlabel
    ):
        raise ValueError(
            "stacked PV heatmaps require one shared x label on the bottom axis"
        )
    colorbar = fig.colorbar(image, ax=axes.tolist(), fraction=0.04, pad=0.04)
    colorbar.set_label("PV correlation (r)")
    colorbar.outline.set_linewidth(0.5)
    fig.text(0.01, 0.98, panel_letter, fontsize=10, fontweight="bold", va="top")
    return fig

def _box_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
