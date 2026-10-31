"""Manuscript definitions collected from manuscript_heatmap_style.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dataclasses import dataclass

from types import MappingProxyType

import numpy as np

import pandas as pd

HEATMAP_STYLE_VERSION = (
    "manuscript_heatmap_v5_unified_plasma_linear_panel_ranges_20260805"
)

PV_HEATMAP_CMAP = "plasma"

NEURAL_MANIFOLD_HEATMAP_CMAP = "plasma"

@dataclass(frozen=True)
class HeatmapPanelSpec:
    """One auditable palette-and-range contract for a manuscript heatmap."""

    family: str
    cmap: str
    vmin: float
    vmax: float
    vcenter: float | None = None

    def __post_init__(self) -> None:
        if self.family not in {"PV", "neural_manifold"}:
            raise ValueError(f"unsupported heatmap family: {self.family}")
        if not np.isfinite(self.vmin) or not np.isfinite(self.vmax):
            raise ValueError("heatmap endpoints must be finite")
        if float(self.vmax) <= float(self.vmin):
            raise ValueError("heatmap endpoints require vmin < vmax")
        if self.vcenter is not None and not (
            float(self.vmin) < float(self.vcenter) < float(self.vmax)
        ):
            raise ValueError("vcenter must lie strictly between vmin and vmax")

HEATMAP_PANEL_SPECS = MappingProxyType(
    {
        "Fig4E": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, 0.45, 0.90),
        "Fig4I": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, 0.30, 0.75),
        "Fig5E": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, -1.5, 16.0),
        "Fig5J": HeatmapPanelSpec(
            "neural_manifold", NEURAL_MANIFOLD_HEATMAP_CMAP, -0.2, 1.4
        ),
        "Fig5K": HeatmapPanelSpec(
            "neural_manifold", NEURAL_MANIFOLD_HEATMAP_CMAP, -0.2, 1.5
        ),
        "Fig5L": HeatmapPanelSpec(
            "neural_manifold", NEURAL_MANIFOLD_HEATMAP_CMAP, -0.5, 2.6
        ),
        "Fig6D": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, 0.35, 0.85),
        "Fig6J": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, 0.30, 0.60),
        "Fig7A": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, -0.5, 10.5),
        "Fig7F": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, -0.5, 2.5),
        "FigS3C": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, 0.05, 0.60),
        "FigS4A": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, -1.0, 1.5),
        "FigS4C": HeatmapPanelSpec("PV", PV_HEATMAP_CMAP, -1.0, 1.5),
    }
)

def heatmap_panel_spec(
    panel: str,
    *,
    expected_family: str | None = None,
) -> HeatmapPanelSpec:
    """Return one locked panel spec and reject representation-family mixups."""

    key = str(panel).replace(".", "").replace(" ", "")
    if key not in HEATMAP_PANEL_SPECS:
        raise KeyError(f"unknown quantitative manuscript heatmap: {panel}")
    spec = HEATMAP_PANEL_SPECS[key]
    if expected_family is not None and spec.family != str(expected_family):
        raise ValueError(
            f"{key} is a {spec.family} heatmap, not {expected_family}"
        )
    return spec

def displayed_group_mean_limits(
    table: pd.DataFrame,
    *,
    group_col: str,
    column_col: str,
    row_col: str,
    value_col: str,
    groups: Iterable[object] | None = None,
    columns: Sequence[object] | None = None,
    fixed_vmin: float | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """Return limits from the cells that a group-mean heatmap displays.

    The displayed matrix is a mean over biological-replicate rows at each
    group × column × row coordinate.  Deriving an automatic maximum from these
    cells avoids letting a hidden individual-replicate extreme flatten the
    rendered comparison.  No clipping or percentile cap is applied.
    """

    required = {group_col, column_col, row_col, value_col}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"heatmap source is missing columns: {missing}")

    display = table.loc[:, [group_col, column_col, row_col, value_col]].copy()
    if groups is not None:
        group_values = tuple(groups)
        display = display.loc[display[group_col].isin(group_values)].copy()
    if columns is not None:
        column_values = tuple(columns)
        display = display.loc[display[column_col].isin(column_values)].copy()
    display[value_col] = pd.to_numeric(display[value_col], errors="coerce")
    display[row_col] = pd.to_numeric(display[row_col], errors="coerce")
    display = display.loc[
        np.isfinite(display[value_col]) & np.isfinite(display[row_col])
    ].copy()
    if display.empty:
        raise ValueError("heatmap source contains no finite requested observations")

    cells = (
        display.groupby(
            [group_col, column_col, row_col], observed=False, as_index=False
        )
        .agg(value=(value_col, "mean"))
        .sort_values([group_col, column_col, row_col], kind="stable")
        .reset_index(drop=True)
    )
    values = cells["value"].to_numpy(dtype=float)
    vmin = float(np.min(values)) if fixed_vmin is None else float(fixed_vmin)
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("displayed heatmap limits must be finite with vmax > vmin")
    return vmin, vmax, cells

def outward_rounded_limits(
    vmin: float,
    vmax: float,
    *,
    step: float = 0.5,
) -> tuple[float, float]:
    """Round finite limits outward without clipping any displayed value."""

    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
        raise ValueError("limits must be finite with vmax > vmin")
    if not np.isfinite(step) or float(step) <= 0:
        raise ValueError("step must be finite and positive")
    lower = float(np.floor(float(vmin) / float(step)) * float(step))
    upper = float(np.ceil(float(vmax) / float(step)) * float(step))
    if upper <= lower:
        upper = lower + float(step)
    return lower, upper
