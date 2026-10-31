"""Manuscript definitions collected from plotting.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from collections.abc import Mapping

from pathlib import Path

import re

from .subjects import DEFAULT_GROUP_COLORS, GROUP_ORDER

GROUP_COLORS = dict(DEFAULT_GROUP_COLORS)

DEFAULT_FIGURE_FORMATS = ("svg",)

DEFAULT_PNG_DPI = 300

DEFAULT_DAY_ERROR_STYLE = "errorbar"

DEFAULT_INDIVIDUAL_DISPLAY = "line"

DEFAULT_SEM_BAND_ALPHA = 0.3

VALID_FIGURE_FORMATS = ("svg", "png")

VALID_DAY_ERROR_STYLES = ("errorbar", "band")

VALID_INDIVIDUAL_DISPLAYS = ("line", "scatter", "none")

MATPLOTLIB_STYLE = {
    "font.family": "Arial",
    "font.style": "normal",
    "font.weight": "normal",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.labelsize": 8,
    "axes.labelweight": "normal",
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "figure.dpi": 120,
    "savefig.dpi": 300,
}

def validate_groups(groups, *, group_order=GROUP_ORDER):
    """Return groups in project order and reject unknown group labels."""
    requested = tuple(groups)
    unknown = [group for group in requested if group not in group_order]
    if unknown:
        raise ValueError(f"unknown group labels: {unknown}; expected one of {list(group_order)}")
    return [group for group in group_order if group in requested]

def validate_figure_formats(figure_formats):
    """Normalize and validate one or more output figure formats."""
    if isinstance(figure_formats, str):
        requested = [figure_formats]
    else:
        requested = list(figure_formats)
    normalized = []
    seen = set()
    for figure_format in requested:
        normalized_format = str(figure_format).lower()
        if normalized_format not in VALID_FIGURE_FORMATS:
            raise ValueError(
                f"unknown figure format: {figure_format!r}; expected one of {list(VALID_FIGURE_FORMATS)}"
            )
        if normalized_format not in seen:
            normalized.append(normalized_format)
            seen.add(normalized_format)
    if not normalized:
        raise ValueError("at least one figure format is required")
    return tuple(normalized)

def validate_day_plot_options(error_style, individual_display):
    """Validate day-level summary and individual display modes."""
    if error_style not in VALID_DAY_ERROR_STYLES:
        raise ValueError(
            f"unknown day error style: {error_style!r}; expected one of {list(VALID_DAY_ERROR_STYLES)}"
        )
    if individual_display not in VALID_INDIVIDUAL_DISPLAYS:
        raise ValueError(
            "unknown individual display: "
            f"{individual_display!r}; expected one of {list(VALID_INDIVIDUAL_DISPLAYS)}"
        )
    return error_style, individual_display

def slug_join(values):
    """Return a filesystem-safe slug for an ordered value collection."""
    if isinstance(values, str):
        values = [values]
    slugs = []
    for value in values:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
        slug = slug.strip("-")
        slugs.append(slug or "none")
    return "-".join(slugs)

def output_config_tag(
    *,
    plot_groups,
    sampling_unit,
    stat_groups=None,
    plot_metrics=None,
    stat_metrics=None,
    view=None,
    contrast=None,
    extra=None,
):
    """Build the compact config tag appended to saved output stems."""
    if sampling_unit is None:
        raise ValueError("sampling_unit must be specified")
    return slug_join(plot_groups)

def output_filename_stem(stem, *, metric=None, contrast=None, extra=None):
    """Return a compact output stem with necessary metric/contrast context."""
    stem = str(stem)
    if metric is None and isinstance(extra, Mapping):
        metric = extra.get("metric")

    parts = [stem]
    for value in (metric, contrast):
        if value is None:
            continue
        slug = slug_join(value)
        if slug and slug not in stem and slug not in parts:
            parts.append(slug)
    return "_".join(parts)

def tagged_output_name(stem, tag, *, metric=None, contrast=None, extra=None):
    """Append a compact group tag to an output filename stem."""
    stem = output_filename_stem(stem, metric=metric, contrast=contrast, extra=extra)
    return f"{stem}_{tag}" if tag else stem

def configure_matplotlib(plt, style=None):
    """Apply the shared notebook plotting style to matplotlib."""
    plt.rcParams.update(MATPLOTLIB_STYLE if style is None else style)

def style_axis(ax):
    """Apply the shared minimal axis style used across project notebooks."""
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2, width=0.5)

def group_errorbar_kwargs(group, *, linewidth=1, marker_size=3):
    """Return standard group errorbar styling for day-level curves."""
    return {
        "fmt": "o-",
        "ms": marker_size,
        "mfc": GROUP_COLORS[group],
        "mec": "none",
        "mew": 0,
        "linewidth": linewidth,
        "capsize": 2,
        "elinewidth": 0.5,
        "capthick": 0.5,
        "color": GROUP_COLORS[group],
        "ecolor": GROUP_COLORS[group],
    }

def group_scatter_kwargs(group, *, size=9, alpha=0.85):
    """Return standard group scatter styling for mouse-level points."""
    return {
        "color": GROUP_COLORS[group],
        "edgecolors": "none",
        "linewidths": 0,
        "alpha": alpha,
        "s": size,
    }

def add_project_colorbar(fig, mappable, ax, *, label, ticks=None, pad=0.02, fraction=0.04):
    """Add a project-styled colorbar for a matplotlib mappable."""
    colorbar = fig.colorbar(mappable, ax=ax, ticks=ticks, pad=pad, fraction=fraction)
    colorbar.set_label(label, fontfamily="Arial", fontstyle="normal", fontweight="normal", fontsize=8)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(labelsize=8, width=0.5, length=2)
    return colorbar

def save_figure(
    fig,
    output_dir,
    filename_stem,
    *,
    enabled=True,
    figure_formats=None,
    png_dpi=DEFAULT_PNG_DPI,
):
    """Save a figure to one or more project-approved formats when enabled."""
    if not enabled:
        return ()
    normalized_formats = validate_figure_formats(
        DEFAULT_FIGURE_FORMATS if figure_formats is None else figure_formats
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for normalized_format in normalized_formats:
        output_path = output_dir / f"{filename_stem}.{normalized_format}"
        savefig_kwargs = {"bbox_inches": "tight"}
        if normalized_format == "png":
            savefig_kwargs["dpi"] = png_dpi
        fig.savefig(output_path, **savefig_kwargs)
        saved_paths.append(output_path)
    return tuple(saved_paths)

def plot_day_level_group(
    ax,
    summary_table,
    observation_table,
    *,
    group,
    unit_col,
    group_col="group",
    day_col="day",
    mean_col="mean",
    sem_col="sem",
    value_col="value",
    error_style=DEFAULT_DAY_ERROR_STYLE,
    individual_display=DEFAULT_INDIVIDUAL_DISPLAY,
    band_alpha=DEFAULT_SEM_BAND_ALPHA,
):
    """Plot project-standard day-level summaries and individual observations for one group."""
    validate_groups([group])
    error_style, individual_display = validate_day_plot_options(error_style, individual_display)

    group_summary = summary_table.loc[summary_table[group_col] == group].sort_values(day_col)
    group_observations = observation_table.loc[observation_table[group_col] == group].sort_values(
        [unit_col, day_col]
    )

    individual_artists = []
    for _, unit_table in group_observations.groupby(unit_col, sort=True):
        days = unit_table[day_col].to_numpy()
        values = unit_table[value_col].to_numpy()
        if individual_display == "line":
            artist, = ax.plot(
                days,
                values,
                color=GROUP_COLORS[group],
                linewidth=0.5,
                marker="o",
                markersize=3,
                markeredgewidth=0,
                alpha=0.3,
                zorder=1,
            )
            individual_artists.append(artist)
        elif individual_display == "scatter":
            artist = ax.scatter(
                days,
                values,
                zorder=1,
                **group_scatter_kwargs(group, size=9, alpha=0.5),
            )
            individual_artists.append(artist)

    summary_line = None
    errorbar = None
    band = None
    if not group_summary.empty:
        days = group_summary[day_col].to_numpy()
        means = group_summary[mean_col].to_numpy()
        sems = group_summary[sem_col].to_numpy()
        if error_style == "errorbar":
            errorbar = ax.errorbar(days, means, yerr=sems, zorder=3, **group_errorbar_kwargs(group))
            summary_line = errorbar.lines[0]
            summary_line.set_zorder(3)
            for capline in errorbar.lines[1]:
                capline.set_linewidth(0.5)
            for barlinecol in errorbar.lines[2]:
                barlinecol.set_linewidth(0.5)
        else:
            summary_line, = ax.plot(
                days,
                means,
                color=GROUP_COLORS[group],
                linewidth=1,
                marker="None",
                zorder=3,
            )
            band = ax.fill_between(
                days,
                means - sems,
                means + sems,
                color=GROUP_COLORS[group],
                alpha=band_alpha,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )

    return {
        "summary_line": summary_line,
        "errorbar": errorbar,
        "band": band,
        "individual": individual_artists,
    }
