"""Manuscript definitions collected from figs4_ad_tmaze_pv_plotting.py.
See provenance/source_files.csv for original source-file hashes.
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

PHASE_LABELS = {
    "sample_left_vs_right": "Sample phase",
    "choice_left_vs_right": "Choice phase",
}

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

def plot_pprh_spatial_heatmap(
    table: pd.DataFrame,
    *,
    contrast: str,
    panel: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = PV_HEATMAP_CMAP,
):
    """Draw the pPRH D1-D5 spatial-bin heatmap for S4A or S4C."""

    accepted = {
        ("sample_left_vs_right", "S4A"),
        ("choice_left_vs_right", "S4C"),
    }
    if (contrast, panel) not in accepted:
        raise ValueError(f"unaccepted heatmap panel/contrast: {panel}/{contrast}")
    _require_columns(
        table,
        {
            "metric",
            "contrast",
            "group",
            "mouse",
            "day",
            "spatial_bin",
            "value",
        },
        f"Fig. {panel} source",
    )
    display = table.loc[
        table["metric"].eq("high_dimensional_pv_separation")
        & table["contrast"].eq(contrast)
        & table["group"].eq("P")
        & table["day"].isin(DAY_ORDER)
    ].copy()
    if set(display["day"]) != set(DAY_ORDER):
        raise ValueError(f"Fig. {panel} lacks D1-D5 group support")
    spec = heatmap_panel_spec(f"Fig{panel}", expected_family="PV")
    if vmin is None:
        vmin = spec.vmin
    if vmax is None:
        vmax = spec.vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= 0 or vmax <= 0:
        raise ValueError("heatmap color limits must straddle zero")
    aggregate = (
        display.groupby(["spatial_bin", "day"], observed=True)["value"]
        .mean()
        .unstack("day")
        .reindex(index=np.arange(29), columns=DAY_ORDER)
    )
    if aggregate.shape != (29, 5):
        raise ValueError(f"Fig. {panel} heatmap is not 29 x 5")

    with mpl.rc_context(MPL_STYLE):
        fig, axis = plt.subplots(figsize=(1.85, 2.25), constrained_layout=False)
        fig.subplots_adjust(left=0.24, right=0.76, bottom=0.20, top=0.84)
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
        axis.set_title(f"{PHASE_LABELS[contrast]} (pPRH)", pad=4)
        axis.set_xticks(DAY_ORDER)
        axis.set_xticklabels([f"D{day}" for day in DAY_ORDER])
        axis.set_xlabel("Day")
        axis.set_yticks((0, 12, 28))
        axis.set_ylabel("Spatial bin")
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(GROUP_COLORS["P"])
        axis.tick_params(direction="out", length=2, width=0.5)
        colorbar = fig.colorbar(mappable, ax=axis, pad=0.07, fraction=0.10)
        colorbar.set_label("z-scored PV separation")
        colorbar.set_ticks((vmin, 0.0, vmax))
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(length=2, width=0.5)
    return fig, axis, colorbar

def plot_cp_section_curves(
    observations: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    contrast: str,
    panel: str,
    annotations: Mapping[tuple[str, str], Mapping] | None = None,
):
    """Draw the C-versus-pPRH section curves for S4B or S4D."""

    accepted = {
        ("sample_left_vs_right", "S4B"),
        ("choice_left_vs_right", "S4D"),
    }
    if (contrast, panel) not in accepted:
        raise ValueError(f"unaccepted curve panel/contrast: {panel}/{contrast}")
    _require_columns(
        observations,
        {"metric", "contrast", "section", "group", "mouse", "day", "value"},
        f"Fig. {panel} observations",
    )
    _require_columns(
        summaries,
        {"metric", "contrast", "section", "group", "day", "mean", "sem"},
        f"Fig. {panel} summaries",
    )
    display = observations.loc[
        observations["metric"].eq("high_dimensional_pv_separation")
        & observations["contrast"].eq(contrast)
        & observations["section"].isin(SECTION_ORDER)
        & observations["group"].isin(GROUP_ORDER)
        & observations["day"].isin(DAY_ORDER)
    ].copy()
    summary = summaries.loc[
        summaries["metric"].eq("high_dimensional_pv_separation")
        & summaries["contrast"].eq(contrast)
        & summaries["section"].isin(SECTION_ORDER)
        & summaries["group"].isin(GROUP_ORDER)
        & summaries["day"].isin(DAY_ORDER)
    ].copy()
    if set(display["section"]) != set(SECTION_ORDER):
        raise ValueError(f"Fig. {panel} requires stem and choice-arm sources")

    with mpl.rc_context(MPL_STYLE):
        fig, axes = plt.subplots(
            1, 2, figsize=(3.75, 2.25), sharex=True, constrained_layout=False
        )
        fig.subplots_adjust(left=0.16, right=0.95, bottom=0.22, top=0.88, wspace=0.38)
        for index, (axis, section) in enumerate(
            zip(axes, SECTION_ORDER, strict=True)
        ):
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
                sem = (
                    pd.to_numeric(curve["sem"], errors="coerce")
                    .fillna(0)
                    .to_numpy(float)
                )
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
                annotation = annotations.get((contrast, section), {}) if annotations else {}
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
            annotation = annotations.get((contrast, section), {}) if annotations else {}
            _draw_group_annotations(axis, annotation)
            axis.axhline(0.0, color="#777777", linewidth=0.5, linestyle="--", zorder=0)
            axis.set_title(SECTION_LABELS[section], y=1.18)
            axis.set_xlim(0.65, 5.45)
            axis.set_xticks(DAY_ORDER)
            axis.set_xticklabels([f"D{day}" for day in DAY_ORDER])
            axis.set_xlabel("Day")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_linewidth(0.5)
            axis.spines["bottom"].set_linewidth(0.5)
            axis.tick_params(direction="out", length=2, width=0.5)
            if index == 0:
                axis.set_ylabel("z-scored PV separation")
    return fig, axes

def save_panel_bundle(
    fig,
    output_dir: str | Path,
    filename_stem: str,
    *,
    raster_dpi: int = 600,
) -> tuple[Path, ...]:
    """Export deterministic editable vector and publication raster files."""

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
            "svg.hashsalt": "figs4_ad_tmaze_highdim_pv_v1",
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

def _draw_group_annotations(axis, annotation: Mapping) -> None:
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

def _require_columns(table: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
