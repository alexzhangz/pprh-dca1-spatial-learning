"""Manuscript definitions collected from build_manuscript_fig6_figs3_stability.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import argparse

import sys

from pathlib import Path

from typing import Any, Mapping, Sequence

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.collections import PathCollection

import numpy as np

import pandas as pd

from maze_analysis.fig6_figs3_stability import (  # noqa: E402
    COVARIATE_COLUMNS,
    DAYS,
    FIG6_COHORT,
    FIG6_PPRH_COVARIATE_COLUMNS,
    FIGS3_COHORT,
    FIGS3_PATHWAY_COVARIATE_COLUMNS,
    INTERDAY_PAIRS,
    SINGLE_NEURON_PSEUDOREPLICATION_WARNING,
    attach_pair_covariates,
    attach_within_covariates,
    build_8maze_interday_sources,
    build_8maze_within_sources,
    build_figure8_cp_covariates,
    build_pair_covariates,
    build_raw_summary,
    build_tmaze_interday_sources,
    fit_stability_jobs,
    load_8maze_cache_records,
    prepare_tmaze_ci_covariates,
    sha256_file,
    validate_exact_cohort,
)

from maze_analysis.manuscript_heatmap_style import (  # noqa: E402
    HEATMAP_STYLE_VERSION,
    PV_HEATMAP_CMAP,
    heatmap_panel_spec,
)

from maze_analysis.spatial import normalize_spatial_rows, spatial_peak_sort_index  # noqa: E402

AXES_FACE = "#FBF2E2"

FIG6K_RIGHT_GROUP_LABEL_POSITIONS = ((0.02, 0.79), (0.02, 0.71))

GROUP_COLORS = {
    "C": "#7ED3FC",
    "I": "#B39DCA",
    "P": "#9BCD88",
}

GROUP_CMAPS = {"C": "Blues", "I": "Purples", "P": "Greens"}

FIG6_LABELS = {"C": "dCA1", "P": "pPRH"}

FIGS3_LABELS = {
    "C": r"pPRH$^{\mathrm{dCA1}}$-Con",
    "I": r"pPRH$^{\mathrm{dCA1}}$-In",
}

def p_star_annotation(value: Any) -> str:
    """Return the locked Fig. 3/4 panel symbol for a raw two-sided P value."""

    value = float(value)
    if not np.isfinite(value):
        return "n.s."
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."

def load_sources(args: argparse.Namespace) -> dict[str, Any]:
    records14, cache14 = load_8maze_cache_records(args.cache_root, "14")
    records16, cache16 = load_8maze_cache_records(args.cache_root, "16")
    records17, cache17 = load_8maze_cache_records(args.cache_root, "17")
    within = build_8maze_within_sources(records14, records16)
    interday = build_8maze_interday_sources(records17)

    fig6_covariates, fig6_covariate_qc = build_figure8_cp_covariates(
        args.aligned_root
    )
    fig6_pair_covariates = build_pair_covariates(
        fig6_covariates, cohort=FIG6_COHORT
    )
    within["pv_mouse_day"] = attach_within_covariates(
        within["pv_mouse_day"], fig6_covariates
    )
    within["single_neuron"] = attach_within_covariates(
        within["single_neuron"], fig6_covariates
    )
    interday["pv_mouse_pair"] = attach_pair_covariates(
        interday["pv_mouse_pair"], fig6_pair_covariates
    )
    interday["single_neuron"] = attach_pair_covariates(
        interday["single_neuron"], fig6_pair_covariates
    )

    tmaze = build_tmaze_interday_sources(args.tmaze_root)
    shared_covariates = pd.read_csv(args.shared_covariates)
    figs3_covariates, figs3_covariate_qc = prepare_tmaze_ci_covariates(
        shared_covariates
    )
    figs3_pair_covariates = build_pair_covariates(
        figs3_covariates, cohort=FIGS3_COHORT
    )
    tmaze["pv_mouse_pair"] = attach_pair_covariates(
        tmaze["pv_mouse_pair"], figs3_pair_covariates
    )
    tmaze["single_neuron"] = attach_pair_covariates(
        tmaze["single_neuron"], figs3_pair_covariates
    )

    return {
        "records14": records14,
        "records16": records16,
        "records17": records17,
        "cache_manifest": pd.concat([cache14, cache16, cache17], ignore_index=True),
        "fig6_within": within,
        "fig6_interday": interday,
        "fig6_covariates": fig6_covariates,
        "fig6_covariate_qc": fig6_covariate_qc,
        "fig6_pair_covariates": fig6_pair_covariates,
        "figs3": tmaze,
        "figs3_covariates": figs3_covariates,
        "figs3_covariate_qc": figs3_covariate_qc,
        "figs3_pair_covariates": figs3_pair_covariates,
    }

def fit_models(
    sources: Mapping[str, Any], *, maxiter: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    within = sources["fig6_within"]
    interday = sources["fig6_interday"]
    fig6_jobs = [
        {
            "analysis": "within",
            "metric": "single_neuron",
            "table": within["single_neuron"],
            "time_col": "day",
            "first_time": 1,
            "last_time": 5,
            "conditions": ("CL", "CR"),
            "continuous_covariates": FIG6_PPRH_COVARIATE_COLUMNS,
        },
        {
            "analysis": "within",
            "metric": "pv",
            "table": within["pv_mouse_day"],
            "time_col": "day",
            "first_time": 1,
            "last_time": 5,
            "conditions": ("CL", "CR"),
            "continuous_covariates": FIG6_PPRH_COVARIATE_COLUMNS,
        },
    ]
    for mode in ("adjacent", "anchor"):
        fig6_jobs.extend(
            [
                {
                    "analysis": mode,
                    "metric": "single_neuron",
                    "table": interday["single_neuron"].loc[
                        interday["single_neuron"]["pair_mode"].eq(mode)
                    ].copy(),
                    "time_col": "time_index",
                    "first_time": 1,
                    "last_time": 4,
                    "conditions": ("CL", "CR"),
                    "continuous_covariates": FIG6_PPRH_COVARIATE_COLUMNS,
                },
                {
                    "analysis": mode,
                    "metric": "pv",
                    "table": interday["pv_mouse_pair"].loc[
                        interday["pv_mouse_pair"]["pair_mode"].eq(mode)
                    ].copy(),
                    "time_col": "time_index",
                    "first_time": 1,
                    "last_time": 4,
                    "conditions": ("CL", "CR"),
                    "continuous_covariates": FIG6_PPRH_COVARIATE_COLUMNS,
                },
            ]
        )

    fig6_contrasts, fig6_coefficients, fig6_diagnostics = fit_stability_jobs(
        fig6_jobs,
        reference_group="C",
        comparison_group="P",
        project="CA Figure-8 maze",
        figure="Fig. 6",
        maxiter=maxiter,
    )

    figs3 = sources["figs3"]
    figs3_jobs = []
    for mode in ("adjacent", "anchor"):
        figs3_jobs.extend(
            [
                {
                    "analysis": mode,
                    "metric": "single_neuron",
                    "table": figs3["single_neuron"].loc[
                        figs3["single_neuron"]["pair_mode"].eq(mode)
                    ].copy(),
                    "time_col": "time_index",
                    "first_time": 1,
                    "last_time": 4,
                    "conditions": ("averaged",),
                    "continuous_covariates": FIGS3_PATHWAY_COVARIATE_COLUMNS,
                },
                {
                    "analysis": mode,
                    "metric": "pv",
                    "table": figs3["pv_mouse_pair"].loc[
                        figs3["pv_mouse_pair"]["pair_mode"].eq(mode)
                    ].copy(),
                    "time_col": "time_index",
                    "first_time": 1,
                    "last_time": 4,
                    "conditions": ("averaged",),
                    "continuous_covariates": FIGS3_PATHWAY_COVARIATE_COLUMNS,
                },
            ]
        )
    figs3_contrasts, figs3_coefficients, figs3_diagnostics = fit_stability_jobs(
        figs3_jobs,
        reference_group="C",
        comparison_group="I",
        project="DNMP T-maze",
        figure="Fig. S3",
        maxiter=maxiter,
    )
    contrasts = pd.concat([fig6_contrasts, figs3_contrasts], ignore_index=True)
    coefficients = pd.concat([fig6_coefficients, figs3_coefficients], ignore_index=True)
    diagnostics = [*fig6_diagnostics, *figs3_diagnostics]
    return contrasts, coefficients, diagnostics

def build_heatmap_payloads(
    sources: Mapping[str, Any], args: argparse.Namespace, directory: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    payloads: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []

    fig6_within = collect_8maze_within_activity(
        sources["records16"], group="P", condition="CL"
    )
    payloads["Fig6C"] = fig6_within
    for day, pair in fig6_within.items():
        manifest_rows.extend(
            save_heatmap_pair(
                pair,
                directory,
                panel="Fig6C",
                key=f"P_CL_D{day}_odd_even",
                map_a_label="Odd",
                map_b_label="Even",
            )
        )

    fig6_adjacent = collect_8maze_interday_activity(
        args.aligned_root,
        cohort=FIG6_COHORT,
        group="P",
        condition="CL",
        pair_mode="adjacent",
    )
    payloads["Fig6G"] = fig6_adjacent
    for pair_label, pair in fig6_adjacent.items():
        manifest_rows.extend(
            save_heatmap_pair(
                pair,
                directory,
                panel="Fig6G",
                key=f"P_CL_{pair_label}",
                map_a_label="Earlier",
                map_b_label="Later",
            )
        )

    figs3_activity = {}
    for group in ("C", "I"):
        group_payload = collect_tmaze_interday_activity(
            args.tmaze_root,
            cohort=FIGS3_COHORT,
            group=group,
            condition="CLC",
            pair_mode="adjacent",
        )
        figs3_activity[group] = group_payload
        for pair_label, pair in group_payload.items():
            manifest_rows.extend(
                save_heatmap_pair(
                    pair,
                    directory,
                    panel="FigS3B",
                    key=f"{group}_CLC_{pair_label}",
                    map_a_label="Earlier",
                    map_b_label="Later",
                )
            )
    payloads["FigS3B"] = figs3_activity
    manifest = pd.DataFrame(manifest_rows)
    write_csv(manifest, directory / "heatmap_matrix_manifest.csv")
    return payloads, manifest

def collect_8maze_within_activity(
    records16: Sequence[Mapping[str, Any]], *, group: str, condition: str
) -> dict[int, dict[str, Any]]:
    result = {}
    for day in DAYS:
        odd_parts = []
        even_parts = []
        label_parts = []
        for record in records16:
            if str(record["group"]) != str(group):
                continue
            mouse = str(record["mouse"])
            entry = record["payload"]["map_store"][(int(day), str(condition))]
            odd = np.vstack([np.asarray(item, dtype=float) for item in entry["odd"]])
            even = np.vstack([np.asarray(item, dtype=float) for item in entry["even"]])
            if odd.shape != even.shape:
                raise ValueError(f"odd/even map shape mismatch for {mouse}/D{day}/{condition}")
            finite = np.isfinite(odd).any(axis=1) & np.isfinite(even).any(axis=1)
            odd_parts.append(odd[finite])
            even_parts.append(even[finite])
            neurons = np.asarray(entry["neuron"])[finite]
            label_parts.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "mouse": mouse,
                        "neuron": neurons,
                        "day": int(day),
                        "condition": condition,
                    }
                )
            )
        odd = np.vstack(odd_parts)
        even = np.vstack(even_parts)
        labels = pd.concat(label_parts, ignore_index=True)
        order = spatial_peak_sort_index(odd)
        result[int(day)] = {
            "map_a_raw": odd[order],
            "map_b_raw": even[order],
            "map_a_display": normalize_spatial_rows(odd)[order],
            "map_b_display": normalize_spatial_rows(even)[order],
            "row_labels": labels.iloc[order].reset_index(drop=True),
        }
    return result

def collect_8maze_interday_activity(
    aligned_root: Path,
    *,
    cohort: Mapping[str, Sequence[str]],
    group: str,
    condition: str,
    pair_mode: str,
) -> dict[str, dict[str, Any]]:
    maps_by_mouse = {}
    for mouse in cohort[group]:
        raw = np.load(Path(aligned_root) / mouse / "snarm_all.npy", allow_pickle=True).item()
        maps_by_mouse[mouse] = np.asarray(raw[condition], dtype=float)
    result = {}
    for day_a, day_b in INTERDAY_PAIRS[pair_mode]:
        a_parts, b_parts, labels = [], [], []
        for mouse, maps in maps_by_mouse.items():
            a = maps[day_a - 1]
            b = maps[day_b - 1]
            finite = np.isfinite(a).any(axis=1) & np.isfinite(b).any(axis=1)
            a_parts.append(a[finite])
            b_parts.append(b[finite])
            labels.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "mouse": mouse,
                        "neuron": np.flatnonzero(finite),
                        "day_a": day_a,
                        "day_b": day_b,
                        "condition": condition,
                    }
                )
            )
        a = np.vstack(a_parts)
        b = np.vstack(b_parts)
        row_labels = pd.concat(labels, ignore_index=True)
        order = spatial_peak_sort_index(a)
        result[f"D{day_a}-D{day_b}"] = {
            "map_a_raw": a[order],
            "map_b_raw": b[order],
            "map_a_display": normalize_spatial_rows(a)[order],
            "map_b_display": normalize_spatial_rows(b)[order],
            "row_labels": row_labels.iloc[order].reset_index(drop=True),
        }
    return result

def collect_tmaze_interday_activity(
    tmaze_root: Path,
    *,
    cohort: Mapping[str, Sequence[str]],
    group: str,
    condition: str,
    pair_mode: str,
) -> dict[str, dict[str, Any]]:
    root = Path(tmaze_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from t_maze.stability import load_subject_condition_maps  # type: ignore

    aligned = root / "Data" / "Aligened_data"
    maps_by_mouse = {
        mouse: load_subject_condition_maps(
            aligned / mouse, condition_keys=(condition,)
        )[condition]
        for mouse in cohort[group]
    }
    result = {}
    for day_a, day_b in INTERDAY_PAIRS[pair_mode]:
        a_parts, b_parts, labels = [], [], []
        for mouse, maps in maps_by_mouse.items():
            a = np.asarray(maps[day_a - 1], dtype=float)
            b = np.asarray(maps[day_b - 1], dtype=float)
            finite = np.isfinite(a).any(axis=1) & np.isfinite(b).any(axis=1)
            a_parts.append(a[finite])
            b_parts.append(b[finite])
            labels.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "mouse": mouse,
                        "neuron": np.flatnonzero(finite),
                        "day_a": day_a,
                        "day_b": day_b,
                        "condition": condition,
                    }
                )
            )
        a = np.vstack(a_parts)
        b = np.vstack(b_parts)
        row_labels = pd.concat(labels, ignore_index=True)
        order = spatial_peak_sort_index(a)
        result[f"D{day_a}-D{day_b}"] = {
            "map_a_raw": a[order],
            "map_b_raw": b[order],
            "map_a_display": normalize_spatial_rows(a)[order],
            "map_b_display": normalize_spatial_rows(b)[order],
            "row_labels": row_labels.iloc[order].reset_index(drop=True),
        }
    return result

def save_heatmap_pair(
    pair: Mapping[str, Any],
    directory: Path,
    *,
    panel: str,
    key: str,
    map_a_label: str,
    map_b_label: str,
) -> list[dict[str, Any]]:
    safe = key.replace("/", "_").replace(" ", "_")
    rows = []
    for name in ("map_a_raw", "map_b_raw", "map_a_display", "map_b_display"):
        path = directory / f"{panel}_{safe}_{name}.npy"
        np.save(path, np.asarray(pair[name], dtype=float), allow_pickle=False)
        rows.append(
            {
                "panel": panel,
                "key": key,
                "artifact": name,
                "map_a_label": map_a_label,
                "map_b_label": map_b_label,
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": int(np.asarray(pair[name]).shape[0]),
                "columns": int(np.asarray(pair[name]).shape[1]),
            }
        )
    labels_path = directory / f"{panel}_{safe}_row_labels.csv"
    write_csv(pair["row_labels"], labels_path)
    rows.append(
        {
            "panel": panel,
            "key": key,
            "artifact": "row_labels",
            "map_a_label": map_a_label,
            "map_b_label": map_b_label,
            "path": labels_path.name,
            "sha256": sha256_file(labels_path),
            "rows": int(len(pair["row_labels"])),
            "columns": int(len(pair["row_labels"].columns)),
        }
    )
    return rows

def plot_within_activity_grid(payload: Mapping[int, Mapping[str, Any]]):
    days = tuple(sorted(payload))
    fig, axes = plt.subplots(2, len(days), figsize=(7.1, 2.8), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.955, bottom=0.16, top=0.90, wspace=0.06, hspace=0.08)
    for column, day in enumerate(days):
        pair = payload[day]
        # Odd trials are the sorting reference in collect_8maze_within_activity;
        # show that reference on top and retain its row order for the Even map.
        for row, (key, label) in enumerate((("map_a_display", "Odd"), ("map_b_display", "Even"))):
            ax = axes[row, column]
            ax.imshow(pair[key], aspect="auto", interpolation="nearest", origin="lower", cmap="Greens", vmin=0, vmax=1, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"D{day}")
            if column == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Spatial bin")
            _box_axis(ax)
    fig.text(0.01, 0.98, "C", fontsize=10, fontweight="bold", va="top")
    fig.text(0.985, 0.50, "Row-normalized activity", rotation=90, va="center", ha="center")
    return fig

def plot_interday_activity_grid(payload, *, groups: Sequence[str], panel_letter: str):
    if len(groups) == 1:
        group_payloads = {groups[0]: payload}
    else:
        group_payloads = payload
    pair_labels = tuple(next(iter(group_payloads.values())).keys())
    fig, axes = plt.subplots(2 * len(groups), len(pair_labels), figsize=(7.2, 1.65 * len(groups) + 1.2), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.955, bottom=0.17, top=0.90, wspace=0.06, hspace=0.08)
    for group_index, group in enumerate(groups):
        for column, pair_label in enumerate(pair_labels):
            pair = group_payloads[group][pair_label]
            day_a, day_b = [int(value[1:]) for value in pair_label.split("-")]
            for local_row, (key, day) in enumerate((("map_b_display", day_b), ("map_a_display", day_a))):
                row = group_index * 2 + local_row
                ax = axes[row, column]
                ax.imshow(pair[key], aspect="auto", interpolation="nearest", origin="lower", cmap=GROUP_CMAPS[group], vmin=0, vmax=1, rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])
                if group_index == 0 and local_row == 0:
                    ax.set_title(pair_label)
                if column == 0:
                    label = f"{FIG6_LABELS.get(group, FIGS3_LABELS.get(group, group))}\nD{day}"
                    ax.set_ylabel(label, color=GROUP_COLORS[group])
                if row == len(groups) * 2 - 1:
                    ax.set_xlabel("Spatial bin")
                _box_axis(ax)
    fig.text(0.01, 0.98, panel_letter, fontsize=10, fontweight="bold", va="top")
    fig.text(0.985, 0.50, "Row-normalized activity", rotation=90, va="center", ha="center")
    return fig

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

def plot_condition_curve_panel(
    table: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    figure: str,
    analysis: str,
    metric: str,
    time_col: str,
    x_values: Sequence[int],
    x_labels: Sequence[str],
    groups: Sequence[str],
    group_labels: Mapping[str, str],
    conditions: Sequence[str],
    condition_labels: Sequence[str],
    ylabel: str,
    xlabel: str,
    panel_letter: str,
    individual_scatter: bool,
):
    fig, axes = plt.subplots(1, len(conditions), figsize=(5.2, 2.2), squeeze=False, sharey=True)
    axes = axes[0]
    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.24, top=0.83, wspace=0.30)
    for ax, condition, title in zip(axes, conditions, condition_labels, strict=True):
        condition_table = table.loc[table["condition"].astype(str).eq(str(condition))].copy()
        group_label_positions = (
            FIG6K_RIGHT_GROUP_LABEL_POSITIONS
            if (
                figure == "Fig. 6"
                and analysis == "anchor"
                and metric == "single_neuron"
                and str(condition) == "CR"
            )
            else None
        )
        _plot_timecourse_axis(
            ax,
            condition_table,
            stats,
            figure=figure,
            analysis=analysis,
            metric=metric,
            condition=str(condition),
            time_col=time_col,
            x_values=x_values,
            x_labels=x_labels,
            groups=groups,
            group_labels=group_labels,
            ylabel=ylabel,
            xlabel=xlabel,
            individual_scatter=individual_scatter,
            group_label_positions=group_label_positions,
        )
        ax.set_title(title, y=1.18)
    axes[0].set_ylabel(ylabel)
    fig.text(0.01, 0.98, panel_letter, fontsize=10, fontweight="bold", va="top")
    validate_curve_panel_figure(
        fig, individual_mouse_scatter=(individual_scatter,) * len(axes)
    )
    return fig

def plot_figs3_metric_pair(
    sources: Mapping[str, pd.DataFrame],
    stats: pd.DataFrame,
    *,
    mode: str,
    panel_letter: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), squeeze=False)
    axes = axes[0]
    # The left-axis endpoint symbols and the right-axis y label both occupy the
    # inter-panel gutter.  A deliberately wider gutter keeps those statistical
    # annotations legible without moving them inside the data region.
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.25, top=0.80, wspace=0.58)
    specs = (
        ("single_neuron", sources["single_neuron"], "Single-neuron spatial activity correlation (r)", False),
        ("pv", sources["pv_mouse_pair"], "PV correlation (r)", True),
    )
    x_labels = (
        ("D1-D2", "D2-D3", "D3-D4", "D4-D5")
        if mode == "adjacent"
        else ("D1-D2", "D1-D3", "D1-D4", "D1-D5")
    )
    for ax, (metric, table, ylabel, scatter) in zip(axes, specs, strict=True):
        subset = table.loc[
            table["pair_mode"].eq(mode) & table["condition"].eq("averaged")
        ].copy()
        _plot_timecourse_axis(
            ax,
            subset,
            stats,
            figure="Fig. S3",
            analysis=mode,
            metric=metric,
            condition="averaged",
            time_col="time_index",
            x_values=(1, 2, 3, 4),
            x_labels=x_labels,
            groups=("C", "I"),
            group_labels=FIGS3_LABELS,
            ylabel=ylabel,
            xlabel=("Adjacent-day pair" if mode == "adjacent" else "Day-1-anchored pair"),
            individual_scatter=scatter,
        )
        ax.set_ylabel(ylabel)
    fig.text(0.01, 0.98, panel_letter, fontsize=10, fontweight="bold", va="top")
    validate_curve_panel_figure(fig, individual_mouse_scatter=(False, True))
    return fig

def validate_curve_panel_figure(
    fig,
    *,
    individual_mouse_scatter: Sequence[bool],
) -> None:
    """Fail closed on the manuscript-wide Fig. 3/4 curve-panel grammar."""

    axes = tuple(fig.axes)
    if len(axes) != len(tuple(individual_mouse_scatter)):
        raise ValueError("curve-panel axis/scatter contract length mismatch")
    expected_face = np.asarray(mpl.colors.to_rgba(AXES_FACE), dtype=float)
    symbol_pattern = {"***", "**", "*", "n.s."}
    for axis, expect_scatter in zip(
        axes, tuple(individual_mouse_scatter), strict=True
    ):
        if not np.allclose(np.asarray(axis.get_facecolor()), expected_face):
            raise ValueError("curve-panel axes must use the locked beige face colour")
        visible = [str(text.get_text()) for text in axis.texts]
        if any("P=" in text or "P<" in text for text in visible):
            raise ValueError("curve-panel text must not expose exact P labels")
        interaction = [text for text in visible if text.startswith("G × Day")]
        if len(interaction) != 1 or "\n" in interaction[0]:
            raise ValueError("curve panel requires one single-line G × Day symbol")
        if not interaction[0].split("  ")[-1] in symbol_pattern:
            raise ValueError("curve-panel interaction must use stars or n.s.")
        if len([text for text in visible if text in symbol_pattern]) < 3:
            raise ValueError("curve panel is missing slope/overall symbols")
        scatter_count = sum(
            isinstance(collection, PathCollection)
            for collection in axis.collections
        )
        if expect_scatter and scatter_count < 2:
            raise ValueError("PV curve panel requires individual-mouse scatter")
        if not expect_scatter and scatter_count != 0:
            raise ValueError("single-neuron curve panel must omit individual points")
        if len(axis.lines) != 3:
            raise ValueError(
                "curve panel must contain two group summaries and one overall bracket; "
                "individual connecting lines are forbidden"
            )

def _plot_timecourse_axis(
    ax,
    table: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    figure: str,
    analysis: str,
    metric: str,
    condition: str,
    time_col: str,
    x_values: Sequence[int],
    x_labels: Sequence[str],
    groups: Sequence[str],
    group_labels: Mapping[str, str],
    ylabel: str,
    xlabel: str,
    individual_scatter: bool,
    group_label_positions: Sequence[tuple[float, float]] | None = None,
):
    ax.set_facecolor(AXES_FACE)
    x_map = {value: index for index, value in enumerate(x_values)}
    summary = build_raw_summary(table, time_col=time_col)
    annotations = _annotation_rows(
        stats,
        figure=figure,
        analysis=analysis,
        metric=metric,
        condition=condition,
    )
    endpoint_annotations = []
    for group in groups:
        color = GROUP_COLORS[group]
        if individual_scatter:
            observations = table.loc[
                table["group"].eq(group)
                & table[time_col].isin(x_values)
                & np.isfinite(pd.to_numeric(table["raw_r"], errors="coerce"))
            ].copy()
            x = observations[time_col].map(x_map).to_numpy(dtype=float)
            ax.scatter(
                x,
                observations["raw_r"].to_numpy(dtype=float),
                s=9,
                color=color,
                edgecolors="none",
                linewidths=0,
                alpha=0.50,
                zorder=1,
            )
        group_summary = summary.loc[
            summary["group"].eq(group)
            & summary["condition"].astype(str).eq(str(condition))
            & summary[time_col].isin(x_values)
        ].copy()
        group_summary["__x"] = group_summary[time_col].map(x_map)
        group_summary = group_summary.sort_values("__x")
        x = group_summary["__x"].to_numpy(dtype=float)
        mean = group_summary["mean_raw_r"].to_numpy(dtype=float)
        sem = group_summary["sem_raw_r"].fillna(0).to_numpy(dtype=float)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.30, edgecolor="none", linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=1.0, marker="o", markersize=3, markeredgewidth=0, zorder=3)
        endpoint = (float(x[-1]), float(mean[-1]))
        endpoint_annotation = ax.annotate(
            p_star_annotation(annotations[f"{group}_slope"]),
            xy=endpoint,
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
            clip_on=False,
            annotation_clip=False,
            zorder=5,
        )
        endpoint_annotations.append((endpoint_annotation, endpoint[1]))
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels(x_labels, rotation=25 if len("".join(x_labels)) > 12 else 0, ha="right" if len("".join(x_labels)) > 12 else "center")
    ax.set_xlabel(xlabel)
    _style_axis(ax)
    group_label_artists = _draw_group_bracket(
        ax,
        groups=groups,
        group_labels=group_labels,
        overall_p=annotations["overall"],
        interaction_p=annotations["interaction"],
        interaction_label="G × Day",
        group_label_positions=group_label_positions,
    )
    if group_label_positions is not None:
        _validate_group_label_clearance(
            ax,
            group_label_artists=group_label_artists,
            group_lines=tuple(ax.lines[: len(groups)]),
        )
    _separate_overlapping_endpoint_annotations(ax, endpoint_annotations)

def _separate_overlapping_endpoint_annotations(ax, annotations) -> None:
    """Retain the Fig. 3/4 endpoint offset unless two slope symbols overlap."""

    if len(annotations) != 2:
        return
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    first_box = annotations[0][0].get_window_extent(renderer=renderer)
    second_box = annotations[1][0].get_window_extent(renderer=renderer)
    if not first_box.overlaps(second_box):
        return
    ordered = sorted(annotations, key=lambda item: float(item[1]))
    ordered[0][0].set_position((5, -5))
    ordered[1][0].set_position((5, 5))
    ax.figure.canvas.draw()
    first_box = ordered[0][0].get_window_extent(renderer=renderer)
    second_box = ordered[1][0].get_window_extent(renderer=renderer)
    if first_box.overlaps(second_box):
        raise ValueError("endpoint slope symbols remain overlapped after collision avoidance")

def _annotation_rows(
    stats: pd.DataFrame,
    *,
    figure: str,
    analysis: str,
    metric: str,
    condition: str,
) -> dict[str, float]:
    table = stats.loc[
        stats["figure"].eq(figure)
        & stats["analysis"].eq(analysis)
        & stats["metric"].eq(metric)
        & stats["condition"].astype(str).eq(str(condition))
        & stats["model_role"].eq("adjusted")
    ]
    mapping = {
        "C_slope": "reference_group_day_slope",
        "P_slope": "comparison_group_day_slope",
        "I_slope": "comparison_group_day_slope",
        "overall": "centered_overall_group_effect",
        "interaction": "group_by_day_interaction",
    }
    result = {}
    for key, contrast in mapping.items():
        rows = table.loc[table["contrast"].eq(contrast)]
        if len(rows) != 1:
            raise ValueError(
                f"expected one annotation row for {figure}/{analysis}/{metric}/{condition}/{contrast}; got {len(rows)}"
            )
        result[key] = float(rows.iloc[0]["p_value"])
    return result

def _draw_group_bracket(
    ax,
    *,
    groups: Sequence[str],
    group_labels: Mapping[str, str],
    overall_p: float,
    interaction_p: float,
    interaction_label: str,
    group_label_positions: Sequence[tuple[float, float]] | None = None,
):
    positions = (
        tuple(group_label_positions)
        if group_label_positions is not None
        else ((0.02, 0.965), (0.02, 0.885))
    )
    if len(positions) != len(groups):
        raise ValueError("group-label position count must match group count")
    group_label_artists = []
    for group, (x, y) in zip(groups, positions, strict=True):
        artist = ax.text(
            x,
            y,
            group_labels[group],
            transform=ax.transAxes,
            color=GROUP_COLORS[group],
            fontsize=8,
            ha="left",
            va="top",
            clip_on=False,
            zorder=6,
        )
        group_label_artists.append(artist)
    x_left, x_right, y_top, y_bottom = 0.52, 0.57, 0.957, 0.875
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
        p_star_annotation(overall_p),
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
        zorder=6,
    )
    ax.text(
        0.98,
        1.025,
        f"{interaction_label}  {p_star_annotation(interaction_p)}",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
        zorder=6,
    )
    return tuple(group_label_artists)

def _validate_group_label_clearance(
    ax,
    *,
    group_label_artists: Sequence[Any],
    group_lines: Sequence[Any],
) -> None:
    """Fail closed if a relocated direct group label is crossed by a curve."""

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for label in group_label_artists:
        label_box = label.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
        for line in group_lines:
            display_path = line.get_path().transformed(line.get_transform())
            if display_path.intersects_bbox(label_box, filled=False):
                raise ValueError(
                    f"group label {label.get_text()!r} intersects a summary curve"
                )

def write_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")
    return path

def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(direction="out", length=2, width=0.5)

def _box_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
