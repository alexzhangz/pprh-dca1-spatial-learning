"""Manuscript definitions collected from build_fig5_manifold_heatmap_completion.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from maze_analysis.fig5_state_separation_plotting import (  # noqa: E402
    FIG5_MANIFOLD_LABEL,
    plot_figure8_manifold_heatmap,
    plot_tmaze_manifold_heatmap,
)

@dataclass(frozen=True)
class PanelSpec:
    order: int
    key: str
    project_key: str
    source_family: str
    contrast: str
    n_bins: int
    boundaries: tuple[int, ...]
    vmin: float
    vmax: float
    filename_stem: str
    accepted_tmaze_panel: str | None = None

PANEL_SPECS = (
    PanelSpec(
        1,
        "Figure8_route_left_vs_right",
        "figure_8_maze",
        "figure8",
        "left_vs_right_correct",
        56,
        (14, 28, 42),
        -0.5,
        6.0,
        "01_Figure8_neural_manifold_left_vs_right_heatmap_D1-D5",
    ),
    PanelSpec(
        2,
        "Tmaze_sample_left_vs_right",
        "t_maze",
        "tmaze",
        "sample_left_vs_right",
        29,
        (13,),
        -0.2,
        1.4,
        "02_Tmaze_neural_manifold_sample_left_vs_right_heatmap_D1-D5",
        "J",
    ),
    PanelSpec(
        3,
        "Tmaze_choice_left_vs_right",
        "t_maze",
        "tmaze",
        "choice_left_vs_right",
        29,
        (13,),
        -0.2,
        1.5,
        "03_Tmaze_neural_manifold_choice_left_vs_right_heatmap_D1-D5",
        "K",
    ),
    PanelSpec(
        4,
        "Tmaze_sample_vs_choice",
        "t_maze",
        "tmaze",
        "sample_vs_choice",
        29,
        (13,),
        -0.5,
        2.6,
        "04_Tmaze_neural_manifold_sample_vs_choice_heatmap_D1-D5",
        "L",
    ),
)

def render_panel(table: pd.DataFrame, spec: PanelSpec):
    if spec.source_family == "figure8":
        return plot_figure8_manifold_heatmap(
            table,
            panel="D",
            boundaries=spec.boundaries,
            vmin=spec.vmin,
            vmax=spec.vmax,
        )[0]
    return plot_tmaze_manifold_heatmap(
        table,
        panel=str(spec.accepted_tmaze_panel),
        boundaries=spec.boundaries,
        vmin=spec.vmin,
        vmax=spec.vmax,
    )[0]
