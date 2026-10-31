"""Manuscript definitions collected from fig5_state_separation_plotting.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pathlib import Path

import re

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from .manuscript_heatmap_style import (
    NEURAL_MANIFOLD_HEATMAP_CMAP,
    PV_HEATMAP_CMAP,
    displayed_group_mean_limits,
    heatmap_panel_spec,
    outward_rounded_limits,
)

FIG5_GROUP_ORDER = ("C", "I")

FIG5_GROUP_COLORS = {"C": "#7ED3FC", "I": "#B39DCA"}

FIG5_GROUP_LABELS = {
    "C": r"pPRH$^{\mathrm{dCA1}}$-Con",
    "I": r"pPRH$^{\mathrm{dCA1}}$-In",
}

FIG5_DIRECT_GROUP_LABELS = {"C": "Con", "I": "In"}

FIG5_DAY_ORDER = (1, 2, 3, 4, 5)

FIG5_FIGURE8_SECTIONS = ("Start arm", "Side arm", "Reward arm", "End arm")

FIG5_TMAZE_SECTIONS = ("Start arm", "Choice arm")

FIG5_LOCKED_PANELS = frozenset({"C"})

FIG5_REGENERATION_PANELS = frozenset(
    {"D", "E", "F", "J", "K", "L", "M", "N", "O", "P", "Q", "R"}
)

FIG5_MANIFOLD_HEATMAP_PANELS = frozenset({"J", "K", "L"})

FIG5_MANIFOLD_CURVE_PANELS = frozenset({"M", "N", "O"})

FIG5_PV_CURVE_PANELS = frozenset({"P", "Q", "R"})

FIG5_PV_LABEL = "z-scored PV separation"

FIG5_MANIFOLD_LABEL = "z-scored d′"

FIG5_CURVE_AXES_FACE = "#FBF2E2"

FIG5_MPL_STYLE = {
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

def validate_regeneration_bundle(panels: Sequence[str]) -> tuple[str, ...]:
    """Validate an ordered Fig. 5 quantitative regeneration bundle.

    C remains author-locked and raises a dedicated error. Other static A--R
    panels are rejected because this module only regenerates cohort-dependent
    quantitative panels D/E/F/J/K/L/M/N/O/P/Q/R.
    """

    normalized = tuple(str(panel).strip().upper() for panel in panels)
    if not normalized:
        raise ValueError("at least one Fig. 5 panel is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Fig. 5 regeneration bundle contains duplicate panels")
    locked = sorted(FIG5_LOCKED_PANELS.intersection(normalized))
    if locked:
        raise ValueError(
            "author-locked Fig. 5 panels cannot be regenerated: " + ", ".join(locked)
        )
    unsupported = sorted(set(normalized).difference(FIG5_REGENERATION_PANELS))
    if unsupported:
        raise ValueError(
            "this presentation module does not regenerate Fig. 5 panels: "
            + ", ".join(unsupported)
        )
    return normalized

def plot_figure8_pv_heatmap(
    table: pd.DataFrame,
    *,
    panel: str = "E",
    group_col: str = "group",
    day_col: str = "day",
    bin_col: str = "spatial_bin",
    value_col: str = "value",
    days: Sequence = FIG5_DAY_ORDER,
    boundaries: Sequence[float] = (14, 28, 42),
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (3.45, 2.15),
):
    """Draw accepted Fig. 5E: C/I Figure-8 PV bin-by-day heatmaps.

    Replicate rows are descriptively averaged at group/day/bin coordinates.
    No binwise inferential mark is calculated or displayed.
    """

    _validate_panel(panel, allowed={"E"})
    if vmin is None or vmax is None:
        automatic_vmin, automatic_vmax, _ = displayed_group_mean_limits(
            table,
            group_col=group_col,
            column_col=day_col,
            row_col=bin_col,
            value_col=value_col,
            groups=FIG5_GROUP_ORDER,
            columns=days,
            fixed_vmin=None,
        )
        automatic_vmin, automatic_vmax = outward_rounded_limits(
            automatic_vmin, automatic_vmax, step=0.5
        )
        if vmin is None:
            vmin = automatic_vmin
        if vmax is None:
            vmax = automatic_vmax
    return _plot_group_heatmaps(
        table,
        panel=panel,
        group_col=group_col,
        day_col=day_col,
        bin_col=bin_col,
        value_col=value_col,
        days=days,
        boundaries=boundaries,
        cmap=PV_HEATMAP_CMAP,
        vmin=vmin,
        vmax=vmax,
        vcenter=None,
        colorbar_label=FIG5_PV_LABEL,
        figsize=figsize,
    )

def plot_figure8_manifold_heatmap(
    table: pd.DataFrame,
    *,
    panel: str = "D",
    group_col: str = "group",
    day_col: str = "day",
    bin_col: str = "spatial_bin",
    value_col: str = "value",
    days: Sequence = FIG5_DAY_ORDER,
    boundaries: Sequence[float] = (14, 28, 42),
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (3.45, 2.15),
):
    """Draw the complementary Figure-8 neural-manifold heatmap pair.

    This is a descriptive group/day/bin view of the same Figure-8 manifold
    source used for the accepted Fig. 5D section curves.  It does not refit the
    embedding, recompute d-prime or add binwise inference.
    """

    _validate_panel(panel, allowed={"D"})
    if vmin is None or vmax is None:
        automatic_vmin, automatic_vmax, _ = displayed_group_mean_limits(
            table,
            group_col=group_col,
            column_col=day_col,
            row_col=bin_col,
            value_col=value_col,
            groups=FIG5_GROUP_ORDER,
            columns=days,
            fixed_vmin=None,
        )
        automatic_vmin, automatic_vmax = outward_rounded_limits(
            automatic_vmin, automatic_vmax, step=0.5
        )
        if vmin is None:
            vmin = automatic_vmin
        if vmax is None:
            vmax = automatic_vmax
    if (
        not np.isfinite(vmin)
        or not np.isfinite(vmax)
        or float(vmin) >= 0.0
        or float(vmax) <= 0.0
        or float(vmax) <= float(vmin)
    ):
        raise ValueError(
            "manifold heatmap endpoints must be finite, ordered, and straddle zero"
        )
    return _plot_group_heatmaps(
        table,
        panel=panel,
        group_col=group_col,
        day_col=day_col,
        bin_col=bin_col,
        value_col=value_col,
        days=days,
        boundaries=boundaries,
        cmap=NEURAL_MANIFOLD_HEATMAP_CMAP,
        vmin=float(vmin),
        vmax=float(vmax),
        vcenter=None,
        colorbar_label=FIG5_MANIFOLD_LABEL,
        figsize=figsize,
    )

def plot_figure8_section_curves(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    panel: str = "F",
    section_order: Sequence[str] = FIG5_FIGURE8_SECTIONS,
    stat_annotations: Mapping | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (7.20, 2.15),
    **column_overrides,
):
    """Draw accepted Fig. 5F: four Figure-8 PV-separation day curves."""

    _validate_panel(panel, allowed={"F"})
    return _plot_section_curves(
        observation_table,
        summary_table,
        panel=panel,
        section_order=section_order,
        ylabel=FIG5_PV_LABEL,
        stat_annotations=stat_annotations,
        ylim=ylim,
        figsize=figsize,
        **column_overrides,
    )

def plot_figure8_manifold_section_curves(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    panel: str = "D",
    section_order: Sequence[str] = FIG5_FIGURE8_SECTIONS,
    stat_annotations: Mapping | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (7.20, 2.15),
    **column_overrides,
):
    """Draw cohort-dependent Fig. 5D neural-manifold section curves."""

    _validate_panel(panel, allowed={"D"})
    return _plot_section_curves(
        observation_table,
        summary_table,
        panel=panel,
        section_order=section_order,
        ylabel=FIG5_MANIFOLD_LABEL,
        stat_annotations=stat_annotations,
        ylim=ylim,
        figsize=figsize,
        **column_overrides,
    )

def plot_tmaze_manifold_heatmap(
    table: pd.DataFrame,
    *,
    panel: str,
    group_col: str = "group",
    day_col: str = "day",
    bin_col: str = "spatial_bin",
    value_col: str = "value",
    days: Sequence = FIG5_DAY_ORDER,
    boundaries: Sequence[float] = (13,),
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (3.45, 2.15),
):
    """Draw one accepted T-maze manifold heatmap pair (J, K or L).

    The caller supplies data for the already selected contrast; manifold
    computation and settings remain outside this presentation module.
    """

    _validate_panel(panel, allowed=FIG5_MANIFOLD_HEATMAP_PANELS)
    spec = heatmap_panel_spec(f"Fig5{panel}", expected_family="neural_manifold")
    if vmin is None:
        vmin = spec.vmin
    if vmax is None:
        vmax = spec.vmax
    if (
        not np.isfinite(vmin)
        or not np.isfinite(vmax)
        or float(vmin) >= 0.0
        or float(vmax) <= 0.0
        or float(vmax) <= float(vmin)
    ):
        raise ValueError(
            "manifold heatmap endpoints must be finite, ordered, and straddle zero"
        )
    return _plot_group_heatmaps(
        table,
        panel=panel,
        group_col=group_col,
        day_col=day_col,
        bin_col=bin_col,
        value_col=value_col,
        days=days,
        boundaries=boundaries,
        cmap=NEURAL_MANIFOLD_HEATMAP_CMAP,
        vmin=float(vmin),
        vmax=float(vmax),
        vcenter=None,
        colorbar_label=FIG5_MANIFOLD_LABEL,
        figsize=figsize,
    )

def plot_tmaze_section_curves(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    panel: str,
    section_order: Sequence[str] = FIG5_TMAZE_SECTIONS,
    stat_annotations: Mapping | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (4.30, 2.15),
    **column_overrides,
):
    """Draw accepted T-maze two-section curves (M/N/O or P/Q/R).

    M/N/O display neural-manifold d-prime; P/Q/R display high-dimensional PV
    separation.
    """

    panel = _normalize_panel(panel)
    _validate_panel(panel, allowed=FIG5_MANIFOLD_CURVE_PANELS | FIG5_PV_CURVE_PANELS)
    ylabel = FIG5_MANIFOLD_LABEL if panel in FIG5_MANIFOLD_CURVE_PANELS else FIG5_PV_LABEL
    return _plot_section_curves(
        observation_table,
        summary_table,
        panel=panel,
        section_order=section_order,
        ylabel=ylabel,
        stat_annotations=stat_annotations,
        ylim=ylim,
        figsize=figsize,
        **column_overrides,
    )

def save_panel_bundle(
    fig,
    output_dir,
    filename_stem: str,
    *,
    raster_dpi: int = 600,
) -> tuple[Path, ...]:
    """Deterministically export editable SVG/PDF and PNG/TIFF rasters."""

    if int(raster_dpi) <= 0:
        raise ValueError("raster_dpi must be a positive integer")
    stem = str(filename_stem).strip()
    if not stem or Path(stem).name != stem:
        raise ValueError("filename_stem must be a non-empty filename stem")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with mpl.rc_context(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "fig5_state_separation_plotting_v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        for extension in ("svg", "pdf", "png", "tiff"):
            path = output_dir / f"{stem}.{extension}"
            kwargs = {
                "format": extension,
                "facecolor": "white",
                "edgecolor": "none",
                "transparent": False,
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

def _plot_group_heatmaps(
    table: pd.DataFrame,
    *,
    panel: str,
    group_col: str,
    day_col: str,
    bin_col: str,
    value_col: str,
    days: Sequence,
    boundaries: Sequence[float],
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    vcenter: float | None,
    colorbar_label: str,
    figsize: tuple[float, float],
):
    _require_columns(table, (group_col, day_col, bin_col, value_col), "heatmap table")
    days = tuple(days)
    if days != FIG5_DAY_ORDER:
        raise ValueError(f"Fig. 5{panel} must display D1–D5 in order")

    display = table.copy()
    display[bin_col] = pd.to_numeric(display[bin_col], errors="coerce")
    display[value_col] = pd.to_numeric(display[value_col], errors="coerce")
    display = display.loc[
        display[group_col].isin(FIG5_GROUP_ORDER)
        & display[day_col].isin(days)
        & np.isfinite(display[bin_col])
        & np.isfinite(display[value_col])
    ].copy()
    if display.empty:
        raise ValueError("heatmap table contains no finite requested observations")
    missing_groups = [group for group in FIG5_GROUP_ORDER if not display[group_col].eq(group).any()]
    if missing_groups:
        raise ValueError(f"heatmap table is missing groups: {missing_groups}")
    bin_order = tuple(sorted(display[bin_col].unique()))

    finite_values = display[value_col].to_numpy(dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(finite_values))
    if vmax is None:
        vmax = float(np.nanmax(finite_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
        raise ValueError("heatmap color limits must be finite with vmax > vmin")
    if vcenter is not None and not (
        float(vmin) < float(vcenter) < float(vmax)
    ):
        raise ValueError("heatmap vcenter must lie strictly between vmin and vmax")
    norm = (
        mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax))
        if vcenter is None
        else mpl.colors.TwoSlopeNorm(
            vmin=float(vmin), vcenter=float(vcenter), vmax=float(vmax)
        )
    )

    with mpl.rc_context(FIG5_MPL_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False, sharey=True)
        axes = axes[0]
        fig.subplots_adjust(left=0.16, right=0.86, bottom=0.22, top=0.84, wspace=0.10)
        mappable = None
        for ax, group in zip(axes, FIG5_GROUP_ORDER, strict=True):
            pivot = (
                display.loc[display[group_col].eq(group)]
                .pivot_table(
                    index=bin_col,
                    columns=day_col,
                    values=value_col,
                    aggfunc="mean",
                    observed=False,
                )
                .reindex(index=bin_order, columns=days)
            )
            mappable = ax.imshow(
                pivot.to_numpy(dtype=float),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            for boundary in boundaries:
                position = int(np.searchsorted(np.asarray(bin_order, dtype=float), boundary)) - 0.5
                if -0.5 < position < len(bin_order) - 0.5:
                    ax.axhline(
                        position,
                        color="black",
                        linestyle="--",
                        linewidth=0.5,
                        alpha=0.75,
                        label="section_boundary",
                    )
            ax.set_title(FIG5_GROUP_LABELS[group], color=FIG5_GROUP_COLORS[group], pad=3)
            ax.set_xticks(np.arange(len(days)))
            ax.set_xticklabels([f"D{int(day)}" for day in days])
            ax.set_xlabel("Day")
            ax.set_yticks(_compact_bin_ticks(bin_order))
            ax.set_yticklabels(_compact_bin_ticklabels(bin_order))
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(FIG5_GROUP_COLORS[group])
                spine.set_linewidth(1.5)
            ax.tick_params(axis="both", direction="out", length=2, width=0.5)
        axes[0].set_ylabel("Spatial bin")

        colorbar = fig.colorbar(mappable, ax=axes.tolist(), pad=0.025, fraction=0.060)
        if float(vmin) < 0.0 < float(vmax):
            colorbar.set_ticks((float(vmin), 0.0, float(vmax)))
        else:
            colorbar.set_ticks((float(vmin), float(vmax)))
        colorbar.set_label(colorbar_label)
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(direction="out", length=2, width=0.5)
    return fig, axes, colorbar

def _plot_section_curves(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    panel: str,
    section_order: Sequence[str],
    ylabel: str,
    stat_annotations: Mapping | None,
    ylim: tuple[float, float] | None,
    figsize: tuple[float, float],
    group_col: str = "group",
    mouse_col: str = "mouse",
    day_col: str = "day",
    section_col: str = "section",
    value_col: str = "value",
    mean_col: str = "mean",
    sem_col: str = "sem",
    axes_facecolor: str = FIG5_CURVE_AXES_FACE,
    show_zero_baseline: bool = True,
):
    _require_columns(
        observation_table,
        (group_col, mouse_col, day_col, section_col, value_col),
        "observation table",
    )
    _require_columns(
        summary_table,
        (group_col, day_col, section_col, mean_col, sem_col),
        "summary table",
    )
    section_order = tuple(str(section) for section in section_order)
    if not section_order:
        raise ValueError("section_order must not be empty")
    _validate_stat_annotations(stat_annotations, section_order=section_order)

    observations = observation_table.copy()
    observations[value_col] = pd.to_numeric(observations[value_col], errors="coerce")
    summaries = summary_table.copy()
    summaries[mean_col] = pd.to_numeric(summaries[mean_col], errors="coerce")
    summaries[sem_col] = pd.to_numeric(summaries[sem_col], errors="coerce")
    x_positions = {day: index for index, day in enumerate(FIG5_DAY_ORDER)}

    with mpl.rc_context(FIG5_MPL_STYLE):
        fig, axes = plt.subplots(
            1,
            len(section_order),
            figsize=figsize,
            squeeze=False,
            sharey=False,
        )
        axes = axes[0]
        left = 0.11 if len(section_order) == 4 else 0.16
        fig.subplots_adjust(left=left, right=0.94, bottom=0.24, top=0.80, wspace=0.34)

        for ax, section in zip(axes, section_order, strict=True):
            ax.set_facecolor(axes_facecolor)
            slope_endpoints: dict[str, tuple[float, float, str]] = {}
            if show_zero_baseline:
                ax.axhline(
                    0.0,
                    color="#666666",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.8,
                    zorder=0,
                    label="zero_baseline",
                )

            annotation = _lookup_annotation(stat_annotations, section)
            for group in FIG5_GROUP_ORDER:
                color = FIG5_GROUP_COLORS[group]
                subset = observations.loc[
                    observations[group_col].eq(group)
                    & observations[section_col].astype(str).eq(section)
                    & observations[day_col].isin(FIG5_DAY_ORDER)
                    & np.isfinite(observations[value_col])
                ].copy()
                subset["__x"] = subset[day_col].map(x_positions)
                ax.scatter(
                    subset["__x"].to_numpy(dtype=float),
                    subset[value_col].to_numpy(dtype=float),
                    s=9,
                    color=color,
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.50,
                    zorder=1,
                )

                summary = summaries.loc[
                    summaries[group_col].eq(group)
                    & summaries[section_col].astype(str).eq(section)
                    & summaries[day_col].isin(FIG5_DAY_ORDER)
                    & np.isfinite(summaries[mean_col])
                ].copy()
                summary["__x"] = summary[day_col].map(x_positions)
                summary = summary.sort_values("__x")
                if summary.empty:
                    continue
                x = summary["__x"].to_numpy(dtype=float)
                mean = summary[mean_col].to_numpy(dtype=float)
                sem = summary[sem_col].fillna(0.0).to_numpy(dtype=float)
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
                    markersize=3.0,
                    markeredgewidth=0,
                    label=f"group_summary_{group}",
                    zorder=3,
                )

                slopes = annotation.get("slopes") if annotation else None
                if slopes and slopes.get(group):
                    endpoint = summary.iloc[-1]
                    slope_endpoints[group] = (
                        float(endpoint["__x"]),
                        float(endpoint[mean_col]),
                        str(slopes[group]),
                    )

            # Move both slope labels away from one another according to their
            # actual endpoint ordering.  A fixed C-up/I-down rule made labels
            # collide whenever the inhibition endpoint was higher (Fig. 5F
            # Start arm and Fig. 5N Start arm).
            slope_offsets = _outward_slope_label_offsets(slope_endpoints)
            for group in FIG5_GROUP_ORDER:
                if group not in slope_endpoints:
                    continue
                x, y, label = slope_endpoints[group]
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(5, slope_offsets[group]),
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

            # Keep the group-by-day statistic in a dedicated subtitle row
            # above the data rectangle.  The earlier bottom-left placement
            # obscured zero-proximal observations in several T-maze panels.
            # Leave a small safety margin above the section heading in all
            # raster/vector exports; y=1.22 put the 8-pt glyph box against
            # the canvas edge in the two-section T-maze panels.
            ax.set_title(section, y=1.18)
            ax.set_xticks(np.arange(len(FIG5_DAY_ORDER)))
            ax.set_xticklabels([f"D{day}" for day in FIG5_DAY_ORDER])
            ax.set_xlim(-0.45, len(FIG5_DAY_ORDER) - 0.55)
            ax.set_xlabel("Day")
            if ylim is not None:
                ax.set_ylim(*ylim)
            _style_curve_axis(ax)
            _draw_fig3_style_group_bracket(
                ax,
                annotation,
                group_labels=FIG5_DIRECT_GROUP_LABELS,
            )
            _draw_group_day_annotation(ax, annotation)
        axes[0].set_ylabel(ylabel)
    return fig, axes

def _outward_slope_label_offsets(
    endpoints: Mapping[str, tuple[float, float, str]],
) -> dict[str, float]:
    """Return endpoint-label offsets that increase, rather than reduce, separation."""

    present = [group for group in FIG5_GROUP_ORDER if group in endpoints]
    if len(present) < 2:
        return {group: 5.0 for group in present}

    first, second = present[:2]
    first_y = float(endpoints[first][1])
    second_y = float(endpoints[second][1])
    if first_y >= second_y:
        return {first: 7.0, second: -7.0}
    return {first: -7.0, second: 7.0}

def _draw_group_day_annotation(ax, annotation: Mapping | None) -> None:
    """Draw the raw-P interaction symbol using the Fig. 3 annotation grammar."""

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

def _draw_fig3_style_group_bracket(
    ax,
    annotation: Mapping | None,
    *,
    group_labels: Mapping[str, str],
) -> None:
    y_positions = {"C": 0.965, "I": 0.885}
    for group in FIG5_GROUP_ORDER:
        ax.text(
            0.02,
            y_positions[group],
            str(group_labels.get(group, group)),
            transform=ax.transAxes,
            color=FIG5_GROUP_COLORS[group],
            fontsize=8,
            ha="left",
            va="top",
            clip_on=False,
            zorder=6,
        )
    if not annotation or not annotation.get("overall"):
        return

    x_left = 0.52
    x_right = 0.57
    y_top = 0.957
    y_bottom = 0.875
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

def _validate_stat_annotations(stat_annotations: Mapping | None, *, section_order) -> None:
    if not stat_annotations:
        return
    unknown_sections = sorted(set(stat_annotations).difference(section_order))
    if unknown_sections:
        raise ValueError(f"stat annotations contain unknown sections: {unknown_sections}")
    for section, annotation in stat_annotations.items():
        if not isinstance(annotation, Mapping):
            raise ValueError(f"stat annotation for {section!r} must be a mapping")
        unknown_keys = sorted(
            set(annotation).difference({"overall", "slopes", "interaction"})
        )
        if unknown_keys:
            raise ValueError(
                "Fig. 5 panels permit only overall, interaction and C/I slope stars; "
                f"unsupported keys for {section!r}: {unknown_keys}"
            )
        if annotation.get("overall") is not None:
            _validate_effect_label(annotation["overall"], context=f"{section} overall")
        slopes = annotation.get("slopes") or {}
        if not isinstance(slopes, Mapping):
            raise ValueError(f"slopes for {section!r} must be a mapping")
        unknown_groups = sorted(set(slopes).difference(FIG5_GROUP_ORDER))
        if unknown_groups:
            raise ValueError(f"unknown slope groups for {section!r}: {unknown_groups}")
        for group, label in slopes.items():
            _validate_effect_label(label, context=f"{section} {group} slope")
        if annotation.get("interaction") is not None:
            _validate_effect_label(
                annotation["interaction"], context=f"{section} group-by-day"
            )

def _validate_effect_label(label: object, *, context: str) -> None:
    text = str(label).strip()
    if not re.fullmatch(r"(?:\*{1,4}|n\.?s\.?)", text, flags=re.IGNORECASE):
        raise ValueError(
            f"{context} must be stars or n.s.; do not place P or q values inside Fig. 5 panels"
        )

def _validate_panel(panel: str, *, allowed: set[str] | frozenset[str]) -> str:
    panel = _normalize_panel(panel)
    if panel in FIG5_LOCKED_PANELS:
        raise ValueError(f"Fig. 5{panel} is author-locked and cannot be regenerated")
    if panel not in allowed:
        raise ValueError(f"panel must be one of {sorted(allowed)}")
    return panel

def _normalize_panel(panel: str) -> str:
    normalized = str(panel).strip().upper()
    if len(normalized) != 1 or not ("A" <= normalized <= "R"):
        raise ValueError("Fig. 5 panel must be a single letter A–R")
    return normalized

def _lookup_annotation(stat_annotations: Mapping | None, section: str) -> Mapping | None:
    if not stat_annotations:
        return None
    if section in stat_annotations:
        return stat_annotations[section]
    section_lower = str(section).lower()
    for key, value in stat_annotations.items():
        if str(key).lower() == section_lower:
            return value
    return None

def _style_curve_axis(ax) -> None:
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=2, width=0.5)

def _compact_bin_ticks(bin_order: Sequence[float]) -> np.ndarray:
    if len(bin_order) == 1:
        return np.asarray([0.0])
    return np.asarray([0.0, float(len(bin_order) - 1)])

def _compact_bin_ticklabels(bin_order: Sequence[float]) -> tuple[str, ...]:
    # The accepted A–R composite displays normalized spatial coordinates while
    # retaining the real-bin row order and boundary locations in the matrix.
    return ("0", "1") if len(bin_order) > 1 else ("0",)

def _require_columns(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
