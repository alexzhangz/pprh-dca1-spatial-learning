"""Manuscript definitions collected from build_fig7_pprh_manifold_heatmap_completion.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from maze_analysis.pprh_manifold_heatmap_plotting import (  # noqa: E402
    DAYS,
    MANIFOLD_COLORBAR_LABEL,
    PPRH_GROUP,
    PPRH_LABEL,
    build_pprh_group_mean_matrix,
    heatmap_image,
    plot_pprh_manifold_heatmap,
)

FIGURE8_MICE = ("H1", "H2", "H3", "H5", "H6")

TMAZE_MICE = ("P1", "P2", "P3", "P4", "P5")

TMAZE_EXPECTED_MISSING = (("P1", 1), ("P2", 1), ("P2", 2))

@dataclass(frozen=True)
class PanelSpec:
    order: int
    key: str
    project: str
    source_family: str
    source_contrast: str
    display_contrast: str
    n_bins: int
    boundaries: tuple[int, ...]
    expected_mice: tuple[str, ...]
    expected_missing_mouse_days: tuple[tuple[str, int], ...]
    vmin: float
    vmax: float
    filename_stem: str

PANEL_SPECS = (
    PanelSpec(
        1,
        "Figure8_pPRH_route_left_vs_right",
        "CA Figure-8 maze",
        "figure8",
        "left_vs_right_current",
        "correct_left_vs_right_route",
        56,
        (14, 28, 42),
        FIGURE8_MICE,
        (),
        -0.4,
        1.9,
        "01_Figure8_pPRH_neural_manifold_correct_left_vs_right_heatmap_D1-D5",
    ),
    PanelSpec(
        2,
        "Tmaze_pPRH_sample_left_vs_right",
        "DNMP T-maze",
        "tmaze",
        "sample_left_vs_right",
        "sample_left_vs_right",
        29,
        (13,),
        TMAZE_MICE,
        TMAZE_EXPECTED_MISSING,
        -0.5,
        0.3,
        "02_Tmaze_pPRH_neural_manifold_sample_left_vs_right_heatmap_D1-D5",
    ),
    PanelSpec(
        3,
        "Tmaze_pPRH_choice_left_vs_right",
        "DNMP T-maze",
        "tmaze",
        "choice_left_vs_right",
        "choice_left_vs_right",
        29,
        (13,),
        TMAZE_MICE,
        TMAZE_EXPECTED_MISSING,
        -0.2,
        0.3,
        "03_Tmaze_pPRH_neural_manifold_choice_left_vs_right_heatmap_D1-D5",
    ),
    PanelSpec(
        4,
        "Tmaze_pPRH_sample_vs_choice",
        "DNMP T-maze",
        "tmaze",
        "sample_vs_choice",
        "sample_vs_choice",
        29,
        (13,),
        TMAZE_MICE,
        TMAZE_EXPECTED_MISSING,
        -0.5,
        1.9,
        "04_Tmaze_pPRH_neural_manifold_sample_vs_choice_heatmap_D1-D5",
    ),
)

def render_panel(table: pd.DataFrame, spec: PanelSpec):
    return plot_pprh_manifold_heatmap(
        table,
        n_bins=spec.n_bins,
        boundaries=spec.boundaries,
        vmin=spec.vmin,
        vmax=spec.vmax,
    )
