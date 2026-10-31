"""Manuscript definitions collected from fig4_stability_plotting.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pathlib import Path

import re

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from .manuscript_heatmap_style import PV_HEATMAP_CMAP

FIG4_GROUP_ORDER = ("C", "I")

FIG4_GROUP_COLORS = {"C": "#7ED3FC", "I": "#B39DCA"}

FIG4_GROUP_LABELS = {"C": "pPRH-dCA1-Con", "I": "pPRH-dCA1-In"}

FIG4_ACTIVITY_CMAPS = {"C": "Blues", "I": "Purples"}

FIG4_DAY_ORDER = (1, 2, 3, 4, 5)

FIG4_ADJACENT_PAIR_ORDER = ("D1–D2", "D2–D3", "D3–D4", "D4–D5")

FIG4_MPL_STYLE = {
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

def plot_fig4d_activity_heatmap(
    matrices: Mapping[tuple[str, int], object],
    *,
    days: Sequence[int] = FIG4_DAY_ORDER,
    group_cmaps: Mapping[str, str] = FIG4_ACTIVITY_CMAPS,
    figsize: tuple[float, float] = (3.5, 4.4),
):
    """Plot five-day correct-left even/odd activity heatmaps (Fig. 4D).

    ``matrices[(group, day)]`` must contain the two already row-normalized
    neuron-by-spatial-bin matrices.  The value may be a mapping with ``"Odd"``
    and ``"Even"`` keys, or a two-item sequence ordered as odd then even.
    Neuron order is accepted as supplied and is never recomputed here.

    Columns are fixed to C Odd, C Even, I Odd, and I Even; rows are D1--D5.
    All images share a 0--1 color scale.
    """

    days = tuple(int(day) for day in days)
    if not days or tuple(sorted(set(days))) != days or any(day not in FIG4_DAY_ORDER for day in days):
        raise ValueError("Display days must be an ordered subset of D1–D5")

    column_spec = (("C", "Odd"), ("C", "Even"), ("I", "Odd"), ("I", "Even"))
    resolved: dict[tuple[str, int, str], np.ndarray] = {}
    for group, split in column_spec:
        for day in days:
            key = (group, day)
            if key not in matrices:
                raise ValueError(f"missing Fig. 4D matrices for {key}")
            odd, even = _resolve_odd_even(matrices[key], label=f"matrices[{key!r}]")
            if odd.shape != even.shape:
                raise ValueError(f"Odd/Even matrices for {key} must have identical shape")
            resolved[(group, day, "Odd")] = odd
            resolved[(group, day, "Even")] = even

    with mpl.rc_context(FIG4_MPL_STYLE):
        fig, axes = plt.subplots(
            len(days),
            len(column_spec),
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"wspace": 0.06, "hspace": 0.10},
        )
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.90)
        group_mappables = {}
        for row, day in enumerate(days):
            for column, (group, split) in enumerate(column_spec):
                ax = axes[row, column]
                matrix = resolved[(group, day, split)]
                mappable = ax.imshow(
                    matrix,
                    aspect="auto",
                    interpolation="nearest",
                    origin="upper",
                    cmap=group_cmaps[group],
                    vmin=0.0,
                    vmax=1.0,
                    rasterized=True,
                )
                group_mappables[group] = mappable
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                if row == 0:
                    ax.set_title(
                        split,
                        color=FIG4_GROUP_COLORS[group],
                        pad=2,
                    )
                if column == 0:
                    ax.set_ylabel(f"D{day}", rotation=0, ha="right", va="center")
        for group, columns in (("C", (0, 1)), ("I", (2, 3))):
            left = axes[0, columns[0]].get_position().x0
            right = axes[0, columns[1]].get_position().x1
            fig.text(
                (left + right) / 2,
                0.965,
                FIG4_GROUP_LABELS[group],
                color=FIG4_GROUP_COLORS[group],
                ha="center",
                va="top",
            )
            fig.text(
                (left + right) / 2,
                0.105,
                "Spatial bin",
                ha="center",
                va="center",
            )

        colorbars = {}
        for group, bounds in (
            # Keep the complete horizontal-colorbar tick-label bounding boxes
            # inside the fixed-size export canvas.  This is presentation-only:
            # the axes, matrices, colour limits, ticks and labels are unchanged.
            ("C", (0.10, 0.055, 0.42, 0.012)),
            ("I", (0.56, 0.055, 0.42, 0.012)),
        ):
            colorbar_axis = fig.add_axes(bounds)
            colorbar = fig.colorbar(
                group_mappables[group],
                cax=colorbar_axis,
                orientation="horizontal",
            )
            colorbar.set_ticks((0.0, 1.0))
            _style_colorbar(colorbar, "Row-normalized activity")
            colorbar.ax.xaxis.set_label_position("top")
            colorbar.ax.xaxis.labelpad = 1
            colorbars[group] = colorbar
    return fig, axes, colorbars

def plot_spatial_bin_heatmap(
    table: pd.DataFrame,
    *,
    panel: str,
    time_col: str | None = None,
    time_order: Sequence | None = None,
    time_labels: Sequence[str] | None = None,
    group_col: str = "group",
    bin_col: str = "spatial_bin",
    value_col: str = "raw_r",
    condition_col: str | None = None,
    condition_value: object = "CL",
    metric_label: str | None = None,
    boundaries: Sequence[int] = (14, 28, 42),
    cmap: str = PV_HEATMAP_CMAP,
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] = (1.85, 2.75),
):
    """Plot C/I spatial-bin PV heatmaps for Fig. 4E or Fig. 4I.

    The input is a long table of raw correlation coefficients.  Fig. 4E uses
    D1--D5 columns; Fig. 4I uses all four adjacent-day pairs.  Replicate rows
    at a group/bin/time coordinate are descriptively averaged for display.
    Pass ``condition_col`` to restrict the table to correct-left (``CL``)
    observations inside this helper.
    """

    panel = str(panel).upper()
    if panel not in {"E", "I"}:
        raise ValueError("panel must be 'E' or 'I'")
    if time_col is None:
        time_col = "day" if panel == "E" else "day_pair"
    if time_order is None:
        time_order = FIG4_DAY_ORDER if panel == "E" else FIG4_ADJACENT_PAIR_ORDER
    time_order = tuple(time_order)
    expected_count = 5 if panel == "E" else 4
    if len(time_order) != expected_count:
        raise ValueError(f"Fig. 4{panel} requires {expected_count} time columns")
    if time_labels is None:
        time_labels = (
            tuple(f"D{int(day)}" for day in time_order)
            if panel == "E"
            else tuple(str(pair) for pair in time_order)
        )
    time_labels = tuple(str(label) for label in time_labels)
    if len(time_labels) != len(time_order):
        raise ValueError("time_labels and time_order must have equal length")
    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmin) >= float(vmax):
        raise ValueError("vmin and vmax must be finite with vmin < vmax")

    required = [group_col, bin_col, time_col, value_col]
    if condition_col is not None:
        required.append(condition_col)
    _require_columns(table, required, "spatial-bin table")
    display = table.copy()
    if condition_col is not None:
        display = display.loc[display[condition_col].eq(condition_value)].copy()
        if display.empty:
            raise ValueError(
                f"no observations match {condition_col}={condition_value!r}"
            )
    display[value_col] = pd.to_numeric(display[value_col], errors="coerce")
    display[bin_col] = pd.to_numeric(display[bin_col], errors="coerce")
    display = display.loc[
        display[group_col].isin(FIG4_GROUP_ORDER)
        & display[time_col].isin(time_order)
        & np.isfinite(display[bin_col])
        & np.isfinite(display[value_col])
    ].copy()
    if display.empty:
        raise ValueError("spatial-bin table contains no finite requested observations")
    bin_order = tuple(sorted(display[bin_col].unique()))

    if metric_label is None:
        metric_label = (
            "Within-session PV reliability (r)"
            if panel == "E"
            else "Adjacent-day PV stability (r)"
        )
    _reject_zscore_label(metric_label)

    with mpl.rc_context(FIG4_MPL_STYLE):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            squeeze=False,
            sharex=True,
            gridspec_kw={"hspace": 0.12},
        )
        axes = axes[:, 0]
        fig.subplots_adjust(left=0.23, right=0.96, bottom=0.38, top=0.96)
        mappable = None
        for ax, group in zip(axes, FIG4_GROUP_ORDER, strict=True):
            subset = display.loc[display[group_col].eq(group)]
            pivot = subset.pivot_table(
                index=bin_col,
                columns=time_col,
                values=value_col,
                aggfunc="mean",
                observed=False,
            ).reindex(index=bin_order, columns=time_order)
            mappable = ax.imshow(
                pivot.to_numpy(dtype=float),
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap=cmap,
                vmin=float(vmin),
                vmax=float(vmax),
                rasterized=True,
            )
            for boundary in boundaries:
                position = int(np.searchsorted(np.asarray(bin_order, dtype=float), boundary)) - 0.5
                if -0.5 < position < len(bin_order) - 0.5:
                    ax.axhline(position, color="white", linestyle="--", linewidth=0.5)
            ax.set_yticks(_compact_bin_ticks(bin_order))
            ax.set_yticklabels(_compact_bin_ticklabels(bin_order))
            ax.set_ylabel("Spatial bin")
            ax.text(
                0.01,
                0.98,
                FIG4_GROUP_LABELS[group],
                transform=ax.transAxes,
                color=FIG4_GROUP_COLORS[group],
                ha="left",
                va="top",
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        axes[-1].set_xticks(np.arange(len(time_order)))
        axes[-1].set_xticklabels(time_labels, rotation=0)
        axes[-1].set_xlabel("Day" if panel == "E" else "Adjacent-day pair")
        colorbar_axis = fig.add_axes((0.29, 0.150, 0.58, 0.018))
        colorbar = fig.colorbar(
            mappable,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_ticks((float(vmin), float(vmax)))
        _style_colorbar(colorbar, metric_label)
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.ax.xaxis.labelpad = 2
    return fig, axes, colorbar

def plot_correlation_curve_panel(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    x_col: str,
    x_order: Sequence,
    panel_kind: str,
    condition_col: str = "condition",
    condition_order: Sequence = ("Left", "Right"),
    group_col: str = "group",
    unit_col: str = "mouse",
    value_col: str = "raw_r",
    mean_col: str = "mean",
    sem_col: str = "sem",
    x_labels: Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    stat_annotations: Mapping | None = None,
    show_interaction: bool = False,
    ylim: tuple[float, float] | None = None,
    show_legend: bool = False,
    group_labels: Mapping[str, str] = FIG4_GROUP_LABELS,
    axes_facecolor: str = "#FBF2E2",
    figsize: tuple[float, float] = (5.20, 2.20),
):
    """Draw a generic Left/Right raw-correlation curve pair (F/G/J/K/L/M).

    ``panel_kind='neuron'`` suppresses all individual observations.
    ``panel_kind='pv'`` draws individual mice as faint scatter only (never as
    cross-day connecting lines).  Both modes draw caller-supplied group means
    and SEM bands.  ``stat_annotations`` is keyed by condition and may contain
    ``overall``, ``slopes={'C': ..., 'I': ...}``, and ``interaction`` entries.

    Statistical annotations deliberately reproduce the released Fig. 3D/H
    visual grammar: colored group labels in the upper left, a right-opening
    bracket plus the overall-group effect label, and each group slope label at
    the endpoint of its mean curve.  When requested, the group-by-time symbol
    is drawn in the lower left using the same ``G × Day`` grammar as Fig. 3.
    """

    panel_kind = str(panel_kind).lower()
    if panel_kind not in {"neuron", "pv"}:
        raise ValueError("panel_kind must be 'neuron' or 'pv'")
    x_order = tuple(x_order)
    if not x_order:
        raise ValueError("x_order must not be empty")
    condition_order = tuple(condition_order)
    if len(condition_order) != 2:
        raise ValueError("condition_order must contain exactly Left and Right conditions")
    if x_labels is None:
        x_labels = tuple(str(value) for value in x_order)
    x_labels = tuple(str(label) for label in x_labels)
    if len(x_labels) != len(x_order):
        raise ValueError("x_labels and x_order must have equal length")

    _require_columns(
        observation_table,
        (group_col, condition_col, x_col, unit_col, value_col),
        "observation table",
    )
    _require_columns(
        summary_table,
        (group_col, condition_col, x_col, mean_col, sem_col),
        "summary table",
    )
    if ylabel is None:
        ylabel = (
            "Single-neuron spatial activity correlation (r)"
            if panel_kind == "neuron"
            else "PV correlation (r)"
        )
    _reject_zscore_label(ylabel)

    observations = observation_table.copy()
    observations[value_col] = pd.to_numeric(observations[value_col], errors="coerce")
    summaries = summary_table.copy()
    summaries[mean_col] = pd.to_numeric(summaries[mean_col], errors="coerce")
    summaries[sem_col] = pd.to_numeric(summaries[sem_col], errors="coerce")
    x_positions = {value: index for index, value in enumerate(x_order)}

    with mpl.rc_context(FIG4_MPL_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False, sharey=True)
        axes = axes[0]
        fig.subplots_adjust(left=0.14, right=0.92, bottom=0.24, top=0.83, wspace=0.30)

        for ax, condition in zip(axes, condition_order, strict=True):
            ax.set_facecolor(axes_facecolor)
            for group in FIG4_GROUP_ORDER:
                color = FIG4_GROUP_COLORS[group]
                if panel_kind == "pv":
                    subset = observations.loc[
                        observations[group_col].eq(group)
                        & observations[condition_col].eq(condition)
                        & observations[x_col].isin(x_order)
                        & np.isfinite(observations[value_col])
                    ].copy()
                    subset["__x"] = subset[x_col].map(x_positions)
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
                    & summaries[condition_col].eq(condition)
                    & summaries[x_col].isin(x_order)
                    & np.isfinite(summaries[mean_col])
                ].copy()
                summary["__x"] = summary[x_col].map(x_positions)
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
                    label=FIG4_GROUP_LABELS[group],
                    zorder=3,
                )

                annotation = _lookup_annotation(stat_annotations, condition)
                slopes = annotation.get("slopes") if annotation else None
                if slopes and slopes.get(group):
                    endpoint = summary.iloc[-1]
                    ax.annotate(
                        str(slopes[group]),
                        xy=(float(endpoint["__x"]), float(endpoint[mean_col])),
                        xytext=(5, 0),
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

            # Reserve a compact subtitle line above the axes for the
            # group-by-day result.  Keeping this statistic outside the data
            # rectangle prevents it from covering low-valued PV observations
            # in panels K and M while retaining one aligned visual grammar.
            # The previous 1.18 offset put the title bounding box about 3 pt
            # beyond the fixed export canvas.  Lowering it by ~5.6 pt retains
            # the same facet-title grammar while preventing source-level crop.
            ax.set_title(str(condition), y=1.12)
            ax.set_xticks(np.arange(len(x_order)))
            ax.set_xticklabels(x_labels)
            ax.set_xlim(-0.45, len(x_order) - 0.55)
            ax.set_xlabel(xlabel or x_col.replace("_", " ").title())
            if ylim is not None:
                ax.set_ylim(*ylim)
            _style_axis(ax)
            _draw_fig3_style_group_bracket(
                ax,
                _lookup_annotation(stat_annotations, condition),
                group_labels=group_labels,
                show_interaction=show_interaction,
            )

        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        if show_legend and handles:
            axes[0].legend(handles[:2], labels[:2], loc="lower right", handlelength=1.4)
    return fig, axes

def plot_fig4h_adjacent_activity_maps(
    matrices: Mapping[tuple[str, object], object],
    *,
    pair_order: Sequence = FIG4_ADJACENT_PAIR_ORDER,
    group_cmaps: Mapping[str, str] = FIG4_ACTIVITY_CMAPS,
    figsize: tuple[float, float] = (6.5, 2.25),
):
    """Plot left-turn adjacent-day activity maps with later days above earlier.

    ``matrices[(group, pair)]`` may be a mapping with ``"earlier"`` and
    ``"later"`` matrices or a two-item sequence ordered earlier then later.
    Matrices must already use earlier-day peak sorting.  Four columns show all
    D1--D2 through D4--D5 comparisons; C and I occupy separate two-row blocks.
    """

    pair_order = tuple(pair_order)
    if len(pair_order) != 4:
        raise ValueError("Fig. 4H requires all four adjacent-day pairs")

    resolved: dict[tuple[str, object, str], np.ndarray] = {}
    for group in FIG4_GROUP_ORDER:
        for pair in pair_order:
            key = (group, pair)
            if key not in matrices:
                raise ValueError(f"missing Fig. 4H matrices for {key}")
            earlier, later = _resolve_earlier_later(
                matrices[key], label=f"matrices[{key!r}]"
            )
            if earlier.shape != later.shape:
                raise ValueError(f"Earlier/later matrices for {key} must have identical shape")
            resolved[(group, pair, "earlier")] = earlier
            resolved[(group, pair, "later")] = later

    with mpl.rc_context(FIG4_MPL_STYLE):
        fig, axes = plt.subplots(
            2,
            8,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"wspace": 0.07, "hspace": 0.10},
        )
        fig.subplots_adjust(left=0.07, right=0.93, bottom=0.18, top=0.90)
        # Reserve a slightly wider visual break between the C and I blocks.
        for ax in axes[:, :4].ravel():
            position = ax.get_position()
            ax.set_position(
                [position.x0 - 0.008, position.y0, position.width, position.height]
            )
        for ax in axes[:, 4:].ravel():
            position = ax.get_position()
            ax.set_position(
                [position.x0 + 0.008, position.y0, position.width, position.height]
            )
        group_mappables = {}
        for group_index, group in enumerate(FIG4_GROUP_ORDER):
            later_row = 0
            earlier_row = 1
            column_offset = group_index * 4
            for column, pair in enumerate(pair_order):
                display_column = column_offset + column
                earlier_day, later_day = _parse_pair_days(pair)
                for row, timing, day_number in (
                    (later_row, "later", later_day),
                    (earlier_row, "earlier", earlier_day),
                ):
                    ax = axes[row, display_column]
                    mappable = ax.imshow(
                        resolved[(group, pair, timing)],
                        aspect="auto",
                        interpolation="nearest",
                        origin="upper",
                        cmap=group_cmaps[group],
                        vmin=0.0,
                        vmax=1.0,
                        rasterized=True,
                    )
                    group_mappables[group] = mappable
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(
                        0.02,
                        0.98,
                        f"D{day_number}",
                        transform=ax.transAxes,
                        color="white",
                        ha="left",
                        va="top",
                    )
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.5)
                    if row == later_row:
                        ax.set_title(str(pair), pad=2)
                    if row == earlier_row:
                        ax.set_xlabel("Spatial bin")
            fig.text(
                0.018 if group == "C" else 0.505,
                0.50,
                FIG4_GROUP_LABELS[group],
                color=FIG4_GROUP_COLORS[group],
                rotation=90,
                ha="center",
                va="center",
            )

        colorbars = {}
        for group, bounds in (
            ("C", (0.945, 0.58, 0.008, 0.28)),
            ("I", (0.945, 0.18, 0.008, 0.28)),
        ):
            colorbar_axis = fig.add_axes(bounds)
            colorbar = fig.colorbar(
                group_mappables[group],
                cax=colorbar_axis,
            )
            colorbar.set_ticks((0.0, 1.0))
            _style_colorbar(colorbar, "")
            colorbars[group] = colorbar
        fig.text(
            0.985,
            0.50,
            "Row-normalized activity",
            rotation=90,
            ha="center",
            va="center",
        )
    return fig, axes, colorbars

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
            "svg.hashsalt": "fig4_stability_plotting_v1",
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

def _resolve_odd_even(value: object, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, Mapping):
        lowered = {str(key).lower(): item for key, item in value.items()}
        if "odd" not in lowered or "even" not in lowered:
            raise ValueError(f"{label} must contain Odd and Even matrices")
        odd, even = lowered["odd"], lowered["even"]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        odd, even = value
    else:
        raise ValueError(f"{label} must be an Odd/Even mapping or two-item sequence")
    return _validate_activity_matrix(odd, f"{label} Odd"), _validate_activity_matrix(
        even, f"{label} Even"
    )

def _resolve_earlier_later(value: object, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, Mapping):
        lowered = {str(key).lower(): item for key, item in value.items()}
        if "earlier" not in lowered or "later" not in lowered:
            raise ValueError(f"{label} must contain earlier and later matrices")
        earlier, later = lowered["earlier"], lowered["later"]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        earlier, later = value
    else:
        raise ValueError(f"{label} must be an earlier/later mapping or two-item sequence")
    return _validate_activity_matrix(
        earlier, f"{label} earlier"
    ), _validate_activity_matrix(later, f"{label} later")

def _validate_activity_matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{label} must be a non-empty 2D matrix")
    finite = matrix[np.isfinite(matrix)]
    if finite.size and (finite.min() < -1e-9 or finite.max() > 1.0 + 1e-9):
        raise ValueError(f"{label} must be row-normalized to the 0–1 range")
    return matrix

def _draw_fig3_style_group_bracket(
    ax,
    annotation: Mapping | None,
    *,
    group_labels: Mapping[str, str],
    show_interaction: bool,
) -> None:
    """Reuse the released Fig. 3D/H group-label and bracket geometry."""

    y_positions = {"C": 0.965, "I": 0.885}
    for group in FIG4_GROUP_ORDER:
        ax.text(
            0.02,
            y_positions[group],
            str(group_labels.get(group, group)),
            transform=ax.transAxes,
            color=FIG4_GROUP_COLORS[group],
            fontsize=8,
            ha="left",
            va="top",
            clip_on=False,
            zorder=6,
        )

    if not annotation:
        return
    overall = annotation.get("overall")
    if overall:
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
        )
        ax.text(
            x_right + 0.025,
            (y_top + y_bottom) / 2,
            str(overall),
            transform=ax.transAxes,
            color="black",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
            clip_on=False,
            zorder=6,
        )
    interaction = annotation.get("interaction")
    if show_interaction and interaction:
        ax.text(
            0.98,
            1.025,
            f"G × Day  {interaction}",
            transform=ax.transAxes,
            color="black",
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="bottom",
            clip_on=False,
            zorder=6,
        )

def _lookup_annotation(stat_annotations: Mapping | None, condition: object) -> Mapping | None:
    if not stat_annotations:
        return None
    if condition in stat_annotations:
        return stat_annotations[condition]
    condition_lower = str(condition).lower()
    for key, value in stat_annotations.items():
        if str(key).lower() == condition_lower:
            return value
    return None

def _parse_pair_days(pair: object) -> tuple[int, int]:
    days = [int(value) for value in re.findall(r"\d+", str(pair))]
    if len(days) != 2 or days[1] != days[0] + 1:
        raise ValueError(f"adjacent-day pair label must contain consecutive days: {pair!r}")
    return days[0], days[1]

def _compact_bin_ticks(bin_order: Sequence[float]) -> np.ndarray:
    if len(bin_order) == 1:
        return np.asarray([0.0])
    return np.asarray([0.0, float(len(bin_order) - 1)])

def _compact_bin_ticklabels(bin_order: Sequence[float]) -> tuple[str, ...]:
    first = float(bin_order[0])
    last = float(bin_order[-1])
    return (_format_number(first), _format_number(last)) if len(bin_order) > 1 else (_format_number(first),)

def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"

def _reject_zscore_label(label: object) -> None:
    normalized = re.sub(r"[^a-z]", "", str(label).lower())
    if "zscore" in normalized or "zscored" in normalized:
        raise ValueError("Fig. 4 displays raw correlations; z-score labels are not allowed")

def _style_axis(ax) -> None:
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=2, width=0.5)

def _style_colorbar(colorbar, label: str) -> None:
    _reject_zscore_label(label)
    colorbar.set_label(str(label))
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(direction="out", length=2, width=0.5)

def _require_columns(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
