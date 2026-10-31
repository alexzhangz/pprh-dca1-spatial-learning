"""Manuscript definitions collected from fig7_s4_tmaze_pprh_plotting.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from collections.abc import Mapping

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from .manuscript_heatmap_style import PV_HEATMAP_CMAP, heatmap_panel_spec

from .subjects import DEFAULT_GROUP_COLORS

GROUP_ORDER = ("C", "P")

GROUP_COLORS = {group: DEFAULT_GROUP_COLORS[group] for group in GROUP_ORDER}

GROUP_LABELS = {"C": "dCA1", "P": "pPRH"}

DAY_ORDER = (1, 2, 3, 4, 5)

SECTION_ORDER = ("stem", "choice_arm")

SECTION_LABELS = {"stem": "Start arm", "choice_arm": "Choice arm"}

MPL_STYLE = {
    "font.family": "Arial",
    "font.style": "normal",
    "font.weight": "normal",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.labelsize": 8,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#FBF2E2",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

def plot_fig7f_pv_heatmap(
    table: pd.DataFrame,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = PV_HEATMAP_CMAP,
):
    """Draw the accepted C/P sample-choice high-dimensional PV heatmap."""

    required = {
        "metric",
        "contrast",
        "group",
        "mouse",
        "day",
        "spatial_bin",
        "value",
    }
    _require_columns(table, required, "Fig. 7F source")
    display = table.loc[
        table["metric"].eq("high_dimensional_pv_separation")
        & table["contrast"].eq("sample_vs_choice")
        & table["group"].isin(GROUP_ORDER)
        & table["day"].isin(DAY_ORDER)
    ].copy()
    if set(display["group"]) != set(GROUP_ORDER):
        raise ValueError("Fig. 7F requires both C and P")
    spec = heatmap_panel_spec("Fig7F", expected_family="PV")
    if vmin is None:
        vmin = spec.vmin
    if vmax is None:
        vmax = spec.vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= 0 or vmax <= 0:
        raise ValueError("Fig. 7F color limits must straddle zero")

    with mpl.rc_context(MPL_STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(3.45, 2.25),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.15, right=0.86, bottom=0.20, top=0.88, wspace=0.12)
        mappable = None
        for axis, group in zip(axes, GROUP_ORDER, strict=True):
            group_table = display.loc[display["group"].eq(group)]
            aggregate = (
                group_table.groupby(["spatial_bin", "day"], observed=True)["value"]
                .mean()
                .unstack("day")
                .reindex(index=np.arange(29), columns=DAY_ORDER)
            )
            if aggregate.shape != (29, 5):
                raise ValueError(f"Fig. 7F {group} heatmap is not 29 x 5")
            mappable = axis.imshow(
                aggregate.to_numpy(dtype=float),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=float(vmin),
                vmax=float(vmax),
                extent=(0.5, 5.5, -0.5, 28.5),
            )
            axis.axhline(12.5, color="#555555", linewidth=0.5, linestyle="--")
            axis.set_title(GROUP_LABELS[group], color=GROUP_COLORS[group], pad=3)
            axis.set_xticks(DAY_ORDER)
            axis.set_xticklabels([f"D{day}" for day in DAY_ORDER])
            axis.set_xlabel("Day")
            # Bins 12 and 13 straddle the section boundary but are only one
            # row apart; labelling both causes unavoidable glyph overlap at
            # the manuscript panel size.  The dashed line marks the 12/13
            # boundary, while ticks retain the start-arm endpoint and maze
            # endpoints without collision.
            axis.set_yticks((0, 12, 28))
            _style_heatmap_axis(axis, group)
        axes[0].set_ylabel("Spatial bin")
        colorbar = fig.colorbar(mappable, ax=axes, pad=0.035, fraction=0.055)
        colorbar.set_label("z-scored PV separation")
        colorbar.set_ticks((vmin, 0.0, vmax))
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(length=2, width=0.5)
    return fig, axes, colorbar

def plot_cp_section_curves(
    observations: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    metric: str,
    contrast: str,
    panel: str,
    annotations: Mapping[tuple[str, str, str], Mapping] | None = None,
):
    """Draw one accepted C/P two-section longitudinal curve panel."""

    required_observations = {
        "metric",
        "contrast",
        "section",
        "group",
        "mouse",
        "day",
        "value",
    }
    required_summaries = {
        "metric",
        "contrast",
        "section",
        "group",
        "day",
        "mean",
        "sem",
    }
    _require_columns(observations, required_observations, f"{panel} observations")
    _require_columns(summaries, required_summaries, f"{panel} summaries")
    accepted = {
        ("7G", "high_dimensional_pv_separation", "sample_vs_choice"),
        ("7I", "neural_manifold_separation", "sample_vs_choice"),
        ("S4E", "neural_manifold_separation", "sample_left_vs_right"),
        ("S4F", "neural_manifold_separation", "choice_left_vs_right"),
    }
    if (panel, metric, contrast) not in accepted:
        raise ValueError(f"unaccepted panel/metric/contrast: {panel}/{metric}/{contrast}")

    display = observations.loc[
        observations["metric"].eq(metric)
        & observations["contrast"].eq(contrast)
        & observations["section"].isin(SECTION_ORDER)
        & observations["group"].isin(GROUP_ORDER)
        & observations["day"].isin(DAY_ORDER)
    ].copy()
    summary = summaries.loc[
        summaries["metric"].eq(metric)
        & summaries["contrast"].eq(contrast)
        & summaries["section"].isin(SECTION_ORDER)
        & summaries["group"].isin(GROUP_ORDER)
        & summaries["day"].isin(DAY_ORDER)
    ].copy()
    if set(display["section"]) != set(SECTION_ORDER):
        raise ValueError(f"{panel} requires stem and choice-arm sources")

    ylabel = (
        "z-scored PV separation"
        if metric == "high_dimensional_pv_separation"
        else "z-scored d′"
    )
    with mpl.rc_context(MPL_STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(3.75, 2.25),
            sharex=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.16, right=0.95, bottom=0.22, top=0.88, wspace=0.38)
        for index, (axis, section) in enumerate(zip(axes, SECTION_ORDER, strict=True)):
            section_observations = display.loc[display["section"].eq(section)]
            section_summary = summary.loc[summary["section"].eq(section)]
            for group in GROUP_ORDER:
                color = GROUP_COLORS[group]
                points = section_observations.loc[
                    section_observations["group"].eq(group)
                ].copy()
                points["day"] = pd.to_numeric(points["day"], errors="coerce")
                points["value"] = pd.to_numeric(points["value"], errors="coerce")
                points = points.loc[
                    np.isfinite(points["day"]) & np.isfinite(points["value"])
                ].sort_values(["day", "mouse"])
                axis.scatter(
                    points["day"],
                    points["value"],
                    s=9,
                    color=color,
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.50,
                    zorder=1,
                    label=f"individual_{group}",
                )

                curve = section_summary.loc[
                    section_summary["group"].eq(group)
                ].sort_values("day")
                day = pd.to_numeric(curve["day"], errors="coerce").to_numpy(float)
                mean = pd.to_numeric(curve["mean"], errors="coerce").to_numpy(float)
                sem = pd.to_numeric(curve["sem"], errors="coerce").fillna(0).to_numpy(float)
                finite = np.isfinite(day) & np.isfinite(mean) & np.isfinite(sem)
                day, mean, sem = day[finite], mean[finite], sem[finite]
                axis.fill_between(
                    day,
                    mean - sem,
                    mean + sem,
                    color=color,
                    alpha=0.30,
                    edgecolor="none",
                    linewidth=0,
                    zorder=2,
                )
                axis.plot(
                    day,
                    mean,
                    color=color,
                    linewidth=1.0,
                    marker="o",
                    markersize=3,
                    markeredgewidth=0,
                    zorder=3,
                    label=f"group_summary_{group}",
                )
                annotation = (
                    annotations.get((metric, contrast, section), {})
                    if annotations
                    else {}
                )
                slope = annotation.get("slopes", {}).get(group)
                if slope and day.size:
                    axis.annotate(
                        str(slope),
                        xy=(float(day[-1]), float(mean[-1])),
                        xytext=(5, 5 if group == "C" else -5),
                        textcoords="offset points",
                        color="black",
                        fontsize=8,
                        fontweight="bold",
                        ha="left",
                        va="center",
                        clip_on=False,
                        annotation_clip=False,
                        zorder=5,
                    )
            annotation = (
                annotations.get((metric, contrast, section), {})
                if annotations
                else {}
            )
            _draw_region_annotations(axis, annotation)
            axis.axhline(0.0, color="#777777", linewidth=0.5, linestyle="--", zorder=0)
            axis.set_title(SECTION_LABELS[section], y=1.18)
            axis.set_xlim(0.65, 5.45)
            axis.set_xticks(DAY_ORDER)
            axis.set_xticklabels([f"D{day}" for day in DAY_ORDER])
            axis.set_xlabel("Day")
            _style_curve_axis(axis)
            if index == 0:
                axis.set_ylabel(ylabel)
    return fig, axes

def save_panel_bundle(
    fig,
    output_dir: str | Path,
    filename_stem: str,
    *,
    raster_dpi: int = 600,
) -> tuple[Path, ...]:
    """Export editable vector files plus publication raster previews."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(raster_dpi) <= 0:
        raise ValueError("raster_dpi must be positive")
    if not filename_stem or Path(filename_stem).name != filename_stem:
        raise ValueError("filename_stem must be a plain non-empty filename stem")
    saved: list[Path] = []
    with mpl.rc_context(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "fig7_s4_tmaze_pprh_legacy_pooled_v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        for extension in ("svg", "pdf", "png", "tiff"):
            path = output_dir / f"{filename_stem}.{extension}"
            kwargs = {
                "format": extension,
                "facecolor": "white",
                "edgecolor": "none",
                "transparent": False,
                "bbox_inches": "tight",
            }
            if extension == "svg":
                kwargs["metadata"] = {"Date": None, "Creator": "matplotlib"}
            elif extension == "pdf":
                kwargs["metadata"] = {
                    "CreationDate": None,
                    "ModDate": None,
                    "Creator": "matplotlib",
                    "Producer": "matplotlib",
                }
            elif extension == "png":
                kwargs["metadata"] = {"Software": "matplotlib"}
            if extension in {"png", "tiff"}:
                kwargs["dpi"] = int(raster_dpi)
            if extension == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(path, **kwargs)
            saved.append(path)
    return tuple(saved)

def _draw_region_annotations(axis, annotation: Mapping) -> None:
    # Match the released Fig. 3/4 annotation grammar: group labels at the
    # upper left, a compact overall-effect bracket between them, and the
    # interaction term above the upper-right edge of the plotting rectangle.
    positions = {"C": 0.965, "P": 0.885}
    for group in GROUP_ORDER:
        axis.text(
            0.02,
            positions[group],
            GROUP_LABELS[group],
            transform=axis.transAxes,
            color=GROUP_COLORS[group],
            fontsize=8,
            ha="left",
            va="top",
            clip_on=False,
            zorder=6,
        )
    overall = annotation.get("overall") if annotation else None
    if overall:
        x_left, x_right = 0.52, 0.57
        y_top, y_bottom = 0.957, 0.875
        axis.plot(
            [x_left, x_right, x_right, x_left],
            [y_top, y_top, y_bottom, y_bottom],
            transform=axis.transAxes,
            color="black",
            linewidth=0.5,
            clip_on=False,
            zorder=6,
        )
        axis.text(
            x_right + 0.025,
            (y_top + y_bottom) / 2,
            str(overall),
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
            clip_on=False,
            zorder=6,
        )
    interaction = annotation.get("interaction") if annotation else None
    if interaction:
        axis.text(
            0.98,
            1.025,
            f"G × Day  {interaction}",
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="bottom",
            clip_on=False,
            zorder=6,
        )

def _style_heatmap_axis(axis, group: str) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(GROUP_COLORS[group])
    axis.tick_params(direction="out", length=2, width=0.5)

def _style_curve_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.5)
    axis.spines["bottom"].set_linewidth(0.5)
    axis.tick_params(direction="out", length=2, width=0.5)

def _require_columns(table: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
