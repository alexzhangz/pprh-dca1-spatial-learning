"""Manuscript definitions collected from fig7_figure8_pprh_plotting.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from .manuscript_heatmap_style import PV_HEATMAP_CMAP, heatmap_panel_spec

GROUP_ORDER = ("C", "P")

GROUP_COLORS = {"C": "#7ED3FC", "P": "#9BCD88"}

DIRECT_GROUP_LABELS = {"C": "dCA1", "P": "pPRH"}

DAYS = (1, 2, 3, 4, 5)

PLOTTED_SECTIONS = ("Start arm", "Side arm", "Reward arm", "End arm")

AXES_FACE = "#FBF2E2"

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
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

def plot_pprh_pv_heatmap(
    table: pd.DataFrame,
    *,
    cmap: str = PV_HEATMAP_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (2.15, 2.10),
):
    """Plot the accepted pPRH-only D1-D5 high-dimensional PV heatmap."""

    _require_columns(
        table,
        ("group", "mouse", "day", "spatial_bin", "value"),
        "Fig7A heatmap source",
    )
    selected = table.loc[
        table["group"].astype(str).eq("P")
        & pd.to_numeric(table["day"], errors="coerce").isin(DAYS)
    ].copy()
    selected["spatial_bin"] = pd.to_numeric(
        selected["spatial_bin"], errors="raise"
    ).astype(int)
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    if set(selected["mouse"].astype(str)) != {"H1", "H2", "H3", "H5", "H6"}:
        raise ValueError("Fig7A heatmap does not contain the locked pPRH P5 cohort")
    grouped = (
        selected.groupby(["spatial_bin", "day"], observed=True, sort=True)["value"]
        .mean()
        .unstack("day")
        .reindex(index=range(56), columns=DAYS)
    )
    if grouped.isna().any().any():
        raise ValueError("Fig7A heatmap lacks the complete 56-bin by D1-D5 grid")
    values = grouped.to_numpy(dtype=float)
    spec = heatmap_panel_spec("Fig7A", expected_family="PV")
    if vmin is None:
        vmin = spec.vmin
    if vmax is None:
        vmax = spec.vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Fig7A heatmap color limits are invalid")

    with mpl.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.23, right=0.78, bottom=0.22, top=0.87)
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        for boundary in (13.5, 27.5, 41.5):
            ax.axhline(
                boundary,
                color="black",
                linewidth=0.5,
                linestyle="--",
                alpha=0.85,
            )
        ax.set_xticks(np.arange(len(DAYS)))
        ax.set_xticklabels([f"D{day}" for day in DAYS])
        ax.set_yticks([0, 55])
        ax.set_yticklabels(["0", "1"])
        ax.set_xlabel("Day")
        ax.set_ylabel("Spatial bin")
        ax.set_title("pPRH", color=GROUP_COLORS["P"], pad=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(GROUP_COLORS["P"])
        colorbar = fig.colorbar(image, ax=ax, fraction=0.09, pad=0.07)
        colorbar.set_label("z-scored PV separation")
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(width=0.5, length=2)
    return fig, ax

def plot_section_curves(
    observations: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    stat_annotations: Mapping[tuple[str, str], Mapping] | None,
    section_order: Sequence[str] = PLOTTED_SECTIONS,
    figsize: tuple[float, float] = (7.3, 2.3),
):
    """Draw Fig. 7B/E with scatter-only mice and Fig. 3/4-style statistics."""

    _require_columns(
        observations,
        ("metric", "group", "mouse", "day", "section", "value"),
        "Fig7B/E observations",
    )
    _require_columns(
        summaries,
        ("metric", "group", "day", "section", "mean", "sem"),
        "Fig7B/E summaries",
    )
    sections = tuple(str(section) for section in section_order)
    display = observations.loc[observations["metric"].astype(str).eq(metric)].copy()
    summary = summaries.loc[summaries["metric"].astype(str).eq(metric)].copy()
    if set(display["section"].astype(str)).intersection(sections) != set(sections):
        raise ValueError(f"{metric} does not contain all plotted sections")
    x_lookup = {day: index for index, day in enumerate(DAYS)}

    with mpl.rc_context(MPL_STYLE):
        fig, axes = plt.subplots(1, len(sections), figsize=figsize, squeeze=False)
        axes = axes[0]
        fig.subplots_adjust(left=0.10, right=0.975, bottom=0.25, top=0.79, wspace=0.34)
        for ax, section in zip(axes, sections, strict=True):
            ax.set_facecolor(AXES_FACE)
            ax.axhline(
                0.0,
                color="#666666",
                linestyle="--",
                linewidth=0.5,
                alpha=0.8,
                zorder=0,
                label="zero_baseline",
            )
            annotation = (
                stat_annotations.get((metric, section))
                if stat_annotations is not None
                else None
            )
            for group in GROUP_ORDER:
                color = GROUP_COLORS[group]
                points = display.loc[
                    display["group"].astype(str).eq(group)
                    & display["section"].astype(str).eq(section)
                    & pd.to_numeric(display["day"], errors="coerce").isin(DAYS)
                    & np.isfinite(pd.to_numeric(display["value"], errors="coerce"))
                ].copy()
                points["__x"] = pd.to_numeric(points["day"], errors="raise").map(x_lookup)
                ax.scatter(
                    points["__x"].to_numpy(dtype=float),
                    pd.to_numeric(points["value"], errors="raise").to_numpy(dtype=float),
                    s=9,
                    color=color,
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.50,
                    zorder=1,
                    label=f"individual_scatter_{group}",
                )
                group_summary = summary.loc[
                    summary["group"].astype(str).eq(group)
                    & summary["section"].astype(str).eq(section)
                    & pd.to_numeric(summary["day"], errors="coerce").isin(DAYS)
                ].copy()
                group_summary["__x"] = pd.to_numeric(
                    group_summary["day"], errors="raise"
                ).map(x_lookup)
                group_summary = group_summary.sort_values("__x")
                x = group_summary["__x"].to_numpy(dtype=float)
                mean = pd.to_numeric(group_summary["mean"], errors="raise").to_numpy(
                    dtype=float
                )
                sem = pd.to_numeric(group_summary["sem"], errors="coerce").fillna(0).to_numpy(
                    dtype=float
                )
                ax.fill_between(
                    x,
                    mean - sem,
                    mean + sem,
                    color=color,
                    alpha=0.30,
                    edgecolor="none",
                    linewidth=0,
                    zorder=2,
                )
                ax.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=1.0,
                    marker="o",
                    markersize=3,
                    markeredgewidth=0,
                    zorder=3,
                    label=f"group_summary_{group}",
                )
                slopes = annotation.get("slopes") if annotation else None
                if slopes and slopes.get(group):
                    ax.annotate(
                        str(slopes[group]),
                        xy=(float(x[-1]), float(mean[-1])),
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
            ax.set_title(section, y=1.18)
            ax.set_xticks(np.arange(len(DAYS)))
            ax.set_xticklabels([f"D{day}" for day in DAYS])
            ax.set_xlim(-0.45, len(DAYS) - 0.55)
            ax.set_xlabel("Day")
            _style_axis(ax)
            _draw_group_bracket(ax, annotation)
            _draw_interaction(ax, annotation)
        axes[0].set_ylabel(ylabel)
    return fig, axes

def save_panel_bundle(
    fig,
    output_dir: str | Path,
    stem: str,
    *,
    formats: Sequence[str] = ("svg", "pdf", "png", "tiff"),
    dpi: int = 600,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with mpl.rc_context(
        {
            "svg.hashsalt": "fig7_figure8_pprh_v1",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        for raw_format in formats:
            fmt = str(raw_format).lower().lstrip(".")
            path = output / f"{stem}.{fmt}"
            kwargs: dict[str, object] = {
                "format": fmt,
                "dpi": int(dpi),
                "bbox_inches": "tight",
                "pad_inches": 0.02,
                "facecolor": "white",
            }
            if fmt == "svg":
                kwargs["metadata"] = {"Date": None, "Creator": "matplotlib"}
            elif fmt == "pdf":
                kwargs["metadata"] = {
                    "CreationDate": None,
                    "ModDate": None,
                    "Creator": "matplotlib",
                    "Producer": "matplotlib",
                }
            elif fmt in {"tif", "tiff"}:
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            elif fmt == "png":
                kwargs["metadata"] = {"Software": "matplotlib"}
            fig.savefig(path, **kwargs)
            saved.append(path)
    return saved

def _draw_interaction(ax, annotation: Mapping | None) -> None:
    if not annotation or not annotation.get("interaction"):
        return
    ax.text(
        0.98,
        1.04,
        f"G × Day  {annotation['interaction']}",
        transform=ax.transAxes,
        color="black",
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
        zorder=6,
    )

def _draw_group_bracket(ax, annotation: Mapping | None) -> None:
    y_positions = {"C": 0.965, "P": 0.885}
    for group in GROUP_ORDER:
        ax.text(
            0.02,
            y_positions[group],
            DIRECT_GROUP_LABELS[group],
            transform=ax.transAxes,
            color=GROUP_COLORS[group],
            fontsize=8,
            ha="left",
            va="top",
            clip_on=False,
            zorder=6,
        )
    if not annotation or not annotation.get("overall"):
        return
    x_left, x_right = 0.57, 0.62
    y_top, y_bottom = 0.957, 0.875
    ax.plot(
        [x_left, x_right, x_right, x_left],
        [y_top, y_top, y_bottom, y_bottom],
        transform=ax.transAxes,
        color="black",
        linewidth=0.5,
        clip_on=False,
        zorder=6,
        label="overall_group_bracket",
    )
    ax.text(
        x_right + 0.025,
        (y_top + y_bottom) / 2,
        str(annotation["overall"]),
        transform=ax.transAxes,
        color="black",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
        zorder=6,
    )

def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in (ax.spines["left"], ax.spines["bottom"]):
        spine.set_linewidth(0.5)
    ax.tick_params(direction="out", width=0.5, length=2)

def _require_columns(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
