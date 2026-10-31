"""Manuscript definitions collected from fig3_behavior_plotting.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from collections.abc import Mapping

from pathlib import Path

import re

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

FIG3_GROUP_ORDER = ("C", "I")

FIG3_GROUP_COLORS = {"C": "#7ED3FC", "I": "#B39DCA"}

FIG3_GROUP_LABELS = {"C": "pPRH-dCA1-Con", "I": "pPRH-dCA1-In"}

FIG3_PANEL_SIZES_IN = {
    "D": (3.0, 2.2),
    "E": (5.5, 2.2),
    "H": (3.2, 2.4),
}

FIG3_MPL_STYLE = {
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

def exact_overlap_jitter(
    table: pd.DataFrame,
    *,
    group_col: str = "group",
    mouse_col: str = "mouse",
    day_col: str = "day",
    value_col: str = "success_rate",
    output_col: str = "plot_day",
    step: float = 0.03,
) -> pd.DataFrame:
    """Return a copy with minimal deterministic jitter for exact overlaps.

    Only observations sharing the same ``group``, ``day``, and *exact* finite
    value are offset.  Distinct values remain centered on the integer day.
    Within an overlap set, natural mouse-name ordering determines symmetric
    offsets, making the result independent of input row order.
    """

    _require_columns(table, (group_col, mouse_col, day_col, value_col), "observation table")
    if not np.isfinite(step) or float(step) < 0:
        raise ValueError("step must be a finite non-negative number")

    result = table.copy()
    result[output_col] = pd.to_numeric(result[day_col], errors="coerce").astype(float)
    numeric_value = pd.to_numeric(result[value_col], errors="coerce")
    numeric_day = pd.to_numeric(result[day_col], errors="coerce")
    finite = np.isfinite(numeric_value) & np.isfinite(numeric_day)

    candidates = result.loc[finite, [group_col, mouse_col, day_col, value_col]].copy()
    candidates["__row_index"] = candidates.index
    for _, overlap in candidates.groupby(
        [group_col, day_col, value_col], sort=False, dropna=False
    ):
        if len(overlap) < 2:
            continue
        ordered_indices = sorted(
            overlap["__row_index"].tolist(),
            key=lambda index: (_natural_key(result.at[index, mouse_col]), str(index)),
        )
        offsets = (
            np.arange(len(ordered_indices), dtype=float) - (len(ordered_indices) - 1) / 2.0
        ) * float(step)
        for index, offset in zip(ordered_indices, offsets, strict=True):
            result.at[index, output_col] = float(numeric_day.at[index]) + float(offset)
    return result

def plot_fig3d(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    overall_annotation: str | None,
    slope_annotations: Mapping[str, str] | None = None,
    interaction_annotation: str | None = None,
    group_labels: Mapping[str, str] = FIG3_GROUP_LABELS,
    ylabel: str = "Success rate",
    jitter_step: float = 0.03,
):
    """Draw the five-day Figure-8 acquisition panel (Fig. 3D)."""

    return _plot_day_success_panel(
        observation_table,
        summary_table,
        panel="D",
        overall_annotation=overall_annotation,
        slope_annotations=slope_annotations,
        interaction_annotation=interaction_annotation,
        group_labels=group_labels,
        ylabel=ylabel,
        jitter_step=jitter_step,
    )

def plot_fig3h(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    overall_annotation: str | None,
    slope_annotations: Mapping[str, str] | None,
    interaction_annotation: str | None = None,
    group_labels: Mapping[str, str] = FIG3_GROUP_LABELS,
    ylabel: str = "Success rate",
    jitter_step: float = 0.03,
):
    """Draw the five-day DNMP T-maze acquisition panel (Fig. 3H)."""

    return _plot_day_success_panel(
        observation_table,
        summary_table,
        panel="H",
        overall_annotation=overall_annotation,
        slope_annotations=slope_annotations,
        interaction_annotation=interaction_annotation,
        group_labels=group_labels,
        ylabel=ylabel,
        jitter_step=jitter_step,
    )

def plot_fig3e(
    individual_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    common_support_max: int,
    ylabel: str = "Success rate",
):
    """Draw marker-free individual traces plus group mean and SEM.

    Each individual mouse is shown as a faint connecting line without point
    markers. Group mean markers and SEM are retained. Every displayed layer is
    restricted to cumulative trials not exceeding ``common_support_max``.
    """

    _require_columns(
        individual_table,
        ("group", "mouse", "cumulative_trial", "moving_success_rate"),
        "individual table",
    )
    _require_columns(
        summary_table,
        ("group", "cumulative_trial", "mean", "sem"),
        "summary table",
    )
    if int(common_support_max) < 1:
        raise ValueError("common_support_max must be >= 1")

    display_individual = individual_table.copy()
    display_individual["cumulative_trial"] = pd.to_numeric(
        display_individual["cumulative_trial"], errors="coerce"
    )
    display_individual["moving_success_rate"] = pd.to_numeric(
        display_individual["moving_success_rate"], errors="coerce"
    )
    display_individual = display_individual.loc[
        np.isfinite(display_individual["cumulative_trial"])
        & np.isfinite(display_individual["moving_success_rate"])
        & display_individual["cumulative_trial"].le(int(common_support_max))
    ].copy()
    if display_individual.empty:
        raise ValueError("individual table contains no finite observations within common support")

    display_summary = summary_table.copy()
    for column in ("cumulative_trial", "mean", "sem"):
        display_summary[column] = pd.to_numeric(display_summary[column], errors="coerce")
    display_summary = display_summary.loc[
        np.isfinite(display_summary["cumulative_trial"])
        & np.isfinite(display_summary["mean"])
        & display_summary["cumulative_trial"].le(int(common_support_max))
    ].copy()
    if display_summary.empty:
        raise ValueError("summary table contains no finite observations within common support")

    with mpl.rc_context(FIG3_MPL_STYLE):
        fig, ax = plt.subplots(figsize=FIG3_PANEL_SIZES_IN["E"])
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.23, top=0.96)

        for group in FIG3_GROUP_ORDER:
            color = FIG3_GROUP_COLORS[group]
            observations = display_individual.loc[
                display_individual["group"].eq(group)
            ].copy()
            for _, trace in observations.groupby("mouse", sort=True):
                trace = trace.sort_values("cumulative_trial")
                ax.plot(
                    trace["cumulative_trial"].to_numpy(dtype=float),
                    trace["moving_success_rate"].to_numpy(dtype=float),
                    color=color,
                    linewidth=0.35,
                    marker=None,
                    alpha=0.30,
                    zorder=1,
                )

            summary = display_summary.loc[
                display_summary["group"].eq(group)
            ].sort_values("cumulative_trial")
            if summary.empty:
                continue
            x = summary["cumulative_trial"].to_numpy(dtype=float)
            mean = summary["mean"].to_numpy(dtype=float)
            sem = summary["sem"].fillna(0.0).to_numpy(dtype=float)
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
                zorder=3,
                label=f"group_summary_{group}",
            )

        finite_x = display_individual["cumulative_trial"]
        x_min = float(finite_x.min())
        x_max = float(finite_x.max())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks(np.arange(0.2, 1.01, 0.2))
        ax.set_xticks(_cumulative_trial_ticks(x_min, x_max))
        ax.set_xlabel("Trials")
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    return fig, ax

def save_panel_bundle(
    fig,
    output_dir,
    filename_stem: str,
    *,
    raster_dpi: int = 600,
) -> tuple[Path, ...]:
    """Export editable vector and 600-dpi raster versions of one panel."""

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
            "svg.hashsalt": "fig3_behavior_plotting_v1",
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

def _plot_day_success_panel(
    observation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    panel: str,
    overall_annotation: str | None,
    slope_annotations: Mapping[str, str] | None,
    interaction_annotation: str | None,
    group_labels: Mapping[str, str],
    ylabel: str,
    jitter_step: float,
):
    _require_columns(
        observation_table,
        ("group", "mouse", "day", "success_rate"),
        "observation table",
    )
    _require_columns(summary_table, ("group", "day", "mean", "sem"), "summary table")
    if panel not in {"D", "H"}:
        raise ValueError("day-success panel must be 'D' or 'H'")

    jittered = exact_overlap_jitter(observation_table, step=jitter_step)
    with mpl.rc_context(FIG3_MPL_STYLE):
        fig, ax = plt.subplots(figsize=FIG3_PANEL_SIZES_IN[panel])
        right = 0.93 if slope_annotations else (0.93 if panel == "H" else 0.97)
        fig.subplots_adjust(left=0.22, right=right, bottom=0.23, top=0.96)

        for group in FIG3_GROUP_ORDER:
            color = FIG3_GROUP_COLORS[group]
            observations = jittered.loc[jittered["group"].eq(group)].copy()
            observations["plot_day"] = pd.to_numeric(
                observations["plot_day"], errors="coerce"
            )
            observations["success_rate"] = pd.to_numeric(
                observations["success_rate"], errors="coerce"
            )
            observations = observations.loc[
                np.isfinite(observations["plot_day"])
                & np.isfinite(observations["success_rate"])
            ].sort_values(["day", "mouse"])
            ax.scatter(
                observations["plot_day"].to_numpy(dtype=float),
                observations["success_rate"].to_numpy(dtype=float),
                s=9,
                color=color,
                edgecolors="none",
                linewidths=0,
                alpha=0.50,
                zorder=1,
            )

            summary = summary_table.loc[summary_table["group"].eq(group)].copy()
            for column in ("day", "mean", "sem"):
                summary[column] = pd.to_numeric(summary[column], errors="coerce")
            summary = summary.loc[
                np.isfinite(summary["day"]) & np.isfinite(summary["mean"])
            ].sort_values("day")
            if summary.empty:
                continue
            day = summary["day"].to_numpy(dtype=float)
            mean = summary["mean"].to_numpy(dtype=float)
            sem = summary["sem"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(
                day,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.30,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                day,
                mean,
                color=color,
                linewidth=1.0,
                marker="o",
                markersize=3.0,
                markeredgewidth=0,
                zorder=3,
                label=f"group_summary_{group}",
            )

            if slope_annotations and slope_annotations.get(group):
                endpoint = summary.iloc[-1]
                ax.annotate(
                    str(slope_annotations[group]),
                    xy=(float(endpoint["day"]), float(endpoint["mean"])),
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

        _draw_direct_labels_and_overall_bracket(
            ax,
            group_labels=group_labels,
            overall_annotation=overall_annotation,
            interaction_annotation=interaction_annotation,
        )
        ax.set_xlim(0.55, 5.45 if slope_annotations or panel == "H" else 5.15)
        ax.set_ylim(0.2, 1.0)
        ax.set_xticks(np.arange(1, 6))
        ax.set_yticks(np.arange(0.2, 1.01, 0.2))
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    return fig, ax

def _draw_direct_labels_and_overall_bracket(
    ax,
    *,
    group_labels: Mapping[str, str],
    overall_annotation: str | None,
    interaction_annotation: str | None,
) -> None:
    y_positions = {"C": 0.965, "I": 0.885}
    for group in FIG3_GROUP_ORDER:
        ax.text(
            0.02,
            y_positions[group],
            str(group_labels.get(group, group)),
            transform=ax.transAxes,
            color=FIG3_GROUP_COLORS[group],
            fontsize=8,
            ha="left",
            va="top",
            clip_on=False,
            zorder=6,
        )
    if overall_annotation:
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
            str(overall_annotation),
            transform=ax.transAxes,
            color="black",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
            clip_on=False,
            zorder=6,
        )
    if interaction_annotation:
        ax.text(
            0.02,
            0.04,
            str(interaction_annotation),
            transform=ax.transAxes,
            color="black",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
            zorder=6,
        )

def _style_axis(ax) -> None:
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=2, width=0.5)

def _cumulative_trial_ticks(x_min: float, x_max: float) -> np.ndarray:
    if x_max <= 100.0:
        midpoint = 50.0 if x_min < 50.0 < x_max else (x_min + x_max) / 2.0
        return np.unique(np.asarray([x_min, midpoint, x_max], dtype=float))
    ticks = [x_min]
    first_hundred = int(np.ceil(max(x_min, 1.0) / 100.0) * 100)
    ticks.extend(np.arange(first_hundred, x_max + 1e-9, 100.0).tolist())
    unique = []
    for value in ticks:
        if not unique or not np.isclose(value, unique[-1]):
            unique.append(value)
    return np.asarray(unique, dtype=float)

def _natural_key(value) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)

def _require_columns(table: pd.DataFrame, columns, label: str) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
