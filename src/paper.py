"""Figure-ordered workflows for the accompanying manuscript.

Shared scientific functions retain their original implementations. This module
replaces historic release-directory and Word-document dependencies with explicit
source-data paths and the panel numbering of the submitted manuscript.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_io import read_table, save_panel, write_tables, p_symbol, write_run_record


def _annotations(frame):
    if "model_variant" in frame:
        frame = frame[frame.model_variant.eq("primary_adjusted_LMM")]
    if "model_role" in frame:
        frame = frame[frame.model_role.eq("adjusted")]
    def symbol(name):
        part = frame[frame.contrast.eq(name)]
        return p_symbol(part.iloc[0].p_value) if len(part) == 1 else None
    return dict(overall_annotation=symbol("centered_overall_group_effect"), slope_annotations={"C": symbol("reference_group_day_slope"), "I": symbol("comparison_group_day_slope")}, interaction_annotation=symbol("group_by_day_interaction"))


def figure3(data_root, output, *, refit=False, formats=("svg", "png"), dpi=300):
    from figure_helpers import figure3 as h
    from figure_helpers import figure3_trial_gee as gee
    from maze_analysis.manuscript_lmm import fit_adjusted_unadjusted_lmm
    exports = []
    for panel, prefix, plotter in (("d", "01_Fig3D_Figure8", h.plot_fig3d), ("h", "03_Fig3H_Tmaze", h.plot_fig3h)):
        observations = read_table(data_root, f"fig3/source_data/{prefix}_success_mouse_day.csv")
        summary = read_table(data_root, f"fig3/source_data/{prefix}_success_group_summary.csv")
        statistics = read_table(data_root, f"fig3/stats/{prefix}_success_LMM_estimands.csv")
        if refit:
            fit = fit_adjusted_unadjusted_lmm(observations, value_col="success_rate", time_col="day", group_col="group", mouse_col="mouse", reference_group="C", comparison_group="I", continuous_covariates=("mean_speed_z", "route_relative_lateral_deviation_z"), first_time=1, last_time=5)["adjusted"]
            if not fit.succeeded:
                raise RuntimeError(fit.diagnostics)
            statistics = fit.contrasts
            write_tables({f"Fig3{panel}_refitted_estimands": statistics}, Path(output) / "statistics")
        fig = plotter(observations, summary, **_annotations(statistics))[0]
        exports += save_panel(fig, output, f"Fig3_{panel}", "learning_curve", formats=formats, dpi=dpi)
    individual = read_table(data_root, "fig3/source_data/02_Fig3E_Figure8_moving_success_mouse_trial_display_common_support.csv")
    summary = read_table(data_root, "fig3/source_data/02_Fig3E_Figure8_moving_success_group_summary_common_support.csv")
    fig = h.plot_fig3e(individual, summary, common_support_max=90)[0]
    exports += save_panel(fig, output, "Fig3_e", "trial_resolved_learning", formats=formats, dpi=dpi)
    if refit:
        trials, _ = gee.prepare_data(Path(data_root) / "fig3_trial/atomic.csv")
        _, params, covariance = gee.fit_gee(trials)
        estimates = gee.all_estimands(params, covariance, df=trials.mouse.nunique() - 4)
        write_tables({"Fig3e_refitted_GEE": estimates}, Path(output) / "statistics")
    return exports


def _separation_bundle(data_root):
    from maze_analysis.fig5_state_separation import Fig5SourceBundle
    names = {
        "figure8_pv_bin": "01_Fig5_Figure8_high_dimensional_PV_spatial_bin_D1-D5.csv",
        "figure8_pv_section": "02_Fig5_Figure8_high_dimensional_PV_sections_D1-D5.csv",
        "figure8_manifold_bin": "03_Fig5_Figure8_neural_manifold_spatial_bin_D1-D5.csv",
        "figure8_manifold_section": "04_Fig5_Figure8_neural_manifold_sections_D1-D5.csv",
        "tmaze_pv_bin": "05_Fig5_Tmaze_high_dimensional_PV_spatial_bin_D1-D5.csv",
        "tmaze_pv_section": "06_Fig5_Tmaze_high_dimensional_PV_sections_D1-D5.csv",
        "tmaze_manifold_bin": "07_Fig5_Tmaze_neural_manifold_spatial_bin_D1-D5.csv",
        "tmaze_manifold_section": "08_Fig5_Tmaze_neural_manifold_sections_D1-D5.csv",
    }
    return Fig5SourceBundle(**{k: read_table(data_root, "fig5/source_data/" + v) for k, v in names.items()}, config_qc=pd.DataFrame())


def figure4(data_root, output, *, refit=False, formats=("svg", "png"), dpi=300):
    from figure_helpers import figure5 as h
    from figure_helpers import figure5_heatmaps as hm
    from maze_analysis.fig5_state_separation import build_fig5_section_summaries, select_panel_source
    bundle = _separation_bundle(data_root)
    summaries = build_fig5_section_summaries(bundle)
    statistics = read_table(data_root, "fig5/stats/Fig5_LMM_all_contrasts.csv")
    if refit:
        result = h.fit_all_section_models(h.build_section_model_table(bundle), maxiter=1000)
        statistics = result.contrasts
        write_tables({"state_separation_refitted_estimands": statistics}, Path(output) / "statistics")
    exports = []
    # Main Fig.4f and supplementary Fig.S5a/d are the accepted manifold heatmaps.
    for spec, panel, table in ((hm.PANEL_SPECS[0], "Fig4_f", bundle.figure8_manifold_bin), (hm.PANEL_SPECS[1], "FigS5_a", bundle.tmaze_manifold_bin), (hm.PANEL_SPECS[2], "FigS5_d", bundle.tmaze_manifold_bin), (hm.PANEL_SPECS[3], "Fig4_l", bundle.tmaze_manifold_bin)):
        selected = table[table.contrast.eq(spec.contrast)].copy()
        fig = hm.render_panel(selected, spec)
        exports += save_panel(fig, output, panel, "manifold_heatmap", old_label="D" if spec.source_family == "figure8" else str(spec.accepted_tmaze_panel), formats=formats, dpi=dpi)
    specs = [
        ("D", "Fig4_g", "figure8_manifold_section", "left_vs_right_correct", ("start_arm", "side_arm", "reward_arm", "end_arm"), h.plot_figure8_manifold_section_curves),
        ("F", "Fig4_h", "figure8_pv_section", "left_vs_right_correct", ("start_arm", "side_arm", "reward_arm", "end_arm"), h.plot_figure8_section_curves),
        ("O", "Fig4_m", "tmaze_manifold_section", "sample_vs_choice", ("stem", "choice_arm"), h.plot_tmaze_section_curves),
        ("R", "Fig4_n", "tmaze_pv_section", "sample_vs_choice", ("stem", "choice_arm"), h.plot_tmaze_section_curves),
        ("M", "FigS5_b", "tmaze_manifold_section", "sample_left_vs_right", ("stem", "choice_arm"), h.plot_tmaze_section_curves),
        ("P", "FigS5_c", "tmaze_pv_section", "sample_left_vs_right", ("stem", "choice_arm"), h.plot_tmaze_section_curves),
        ("N", "FigS5_e", "tmaze_manifold_section", "choice_left_vs_right", ("stem", "choice_arm"), h.plot_tmaze_section_curves),
        ("Q", "FigS5_f", "tmaze_pv_section", "choice_left_vs_right", ("stem", "choice_arm"), h.plot_tmaze_section_curves),
    ]
    for old, panel, source_key, contrast, sections, plotter in specs:
        observations = select_panel_source(bundle, old)
        if source_key.startswith("figure8"):
            observations = observations[observations.section.isin(sections)].copy()
            contrast = h.FIGURE8_CONTRAST
        summary = h._panel_summary(summaries, source_key=source_key, contrast=contrast, sections=sections)
        fig = plotter(h._display_sections(observations), summary, panel=old, stat_annotations=h._plot_annotations(statistics, observations))[0]
        exports += save_panel(fig, output, panel, "state_separation", old_label=old, formats=formats, dpi=dpi)
    from maze_analysis import manifold_trajectory as trajectory
    from figure_helpers import tmaze_trajectory as phase_trajectory
    session = Path(data_root) / "representatives/figure8_A4_D7"
    points = trajectory.load_correct_left_right_point_session(session)
    means = trajectory.load_correct_left_right_trajectory_session(session, n_bins=56)
    fig = trajectory.plot_correct_left_right_points(points, view_elev=75, view_azim=-90)
    exports += save_panel(fig, output, "Fig4_d", "Figure8_latent_points_A4_D7", formats=formats, dpi=dpi)
    fig = trajectory.plot_correct_left_right_trajectories(means, view_elev=75, view_azim=-90)
    exports += save_panel(fig, output, "Fig4_d", "Figure8_mean_trajectories_A4_D7", formats=formats, dpi=dpi)
    latent = read_table(data_root, "representatives/tmaze_phase_latent_points.csv")
    conditions = {phase: {"points": rows[["spectral_1", "spectral_2", "spectral_3"]].to_numpy(), "bins": rows.spatial_bin.to_numpy(), "trial_ids": rows.phase_trial_index.to_numpy()} for phase, rows in latent.groupby("condition", sort=False)}
    means, _ = phase_trajectory.mean_trajectories(conditions)
    fig, axes = plt.subplots(2, 1, figsize=(2.05, 3.45), gridspec_kw={"height_ratios": [1, .78], "hspace": .03})
    max_bin = int(latent.spatial_bin.max())
    phase_trajectory.draw_point_cloud(axes[0], conditions, max_bin)
    phase_trajectory.draw_mean_trajectories(axes[1], conditions, means, max_bin)
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    exports += save_panel(fig, output, "Fig4_j", "Tmaze_phase_latent_trajectories", formats=formats, dpi=dpi)
    return exports


def _stability_ci_sources(data_root):
    names = {"within_pv_bin": "Fig4_within_pv_bin.csv", "within_pv": "Fig4_within_pv.csv", "within_single": "Fig4_within_single.csv", "inter_pv_bin": "Fig4_inter_pv_bin.csv", "inter_pv": "Fig4_inter_pv.csv", "inter_single": "Fig4_inter_single.csv"}
    return {k: read_table(data_root, "fig4/source_data/" + v) for k, v in names.items()}


def figure5(data_root, output, *, refit=False, formats=("svg", "png"), dpi=300):
    from figure_helpers import figure4 as h
    sources = _stability_ci_sources(data_root)
    statistics = read_table(data_root, "fig4/stats/Fig4_LMM_all_contrasts.csv")
    if refit:
        statistics, coefficients, _ = h.fit_all_models(sources, maxiter=1000)
        write_tables({"stability_ci_refitted_estimands": statistics, "stability_ci_refitted_coefficients": coefficients}, Path(output) / "statistics")
    descriptives = h.build_descriptive_tables(sources)
    exports = []
    within, adjacent = {}, {}
    for group in ("C", "I"):
        for day in (1, 3, 5):
            path = Path(data_root) / f"fig4/source_data/heatmaps/Fig4D_{group}_D{day}_CL_matrices.npz"
            with np.load(path) as z:
                within[(group, day)] = {"Odd": z["map_a_display"], "Even": z["map_b_display"]}
        for a, b in ((1, 2), (2, 3), (3, 4), (4, 5)):
            path = Path(data_root) / f"fig4/source_data/heatmaps/Fig4H_{group}_D{a}-D{b}_CL_matrices.npz"
            with np.load(path) as z:
                adjacent[(group, f"D{a}\N{EN DASH}D{b}")] = {"earlier": z["map_a_display"], "later": z["map_b_display"]}
    fig = h.plot_fig4d_activity_heatmap(within, days=(1, 3, 5))[0]
    exports += save_panel(fig, output, "Fig5_b", "odd_even_activity_maps", old_label="D", formats=formats, dpi=dpi)
    fig = h.plot_fig4h_adjacent_activity_maps(adjacent)[0]
    exports += save_panel(fig, output, "Fig5_g", "adjacent_day_activity_maps", old_label="H", formats=formats, dpi=dpi)
    for old, panel, table, time_col, time_order, label in (("E", "Fig5_c", sources["within_pv_bin"], "day", (1, 2, 3, 4, 5), "PV reliability (r)"), ("I", "Fig5_h", sources["inter_pv_bin"][sources["inter_pv_bin"].pair_mode.eq("adjacent")], "day_pair_display", ("D1–D2", "D2–D3", "D3–D4", "D4–D5"), "PV stability (r)")):
        fig = h.plot_spatial_bin_heatmap(table, panel=old, time_col=time_col, time_order=time_order, condition_col="condition", condition_value="CL", metric_label=label, vmin=h.PLOT_LIMITS[old][0], vmax=h.PLOT_LIMITS[old][1])[0]
        exports += save_panel(fig, output, panel, "PV_heatmap", old_label=old, formats=formats, dpi=dpi)
    for old, panel, analysis, metric in (("F", "Fig5_d", "within", "single_neuron"), ("G", "Fig5_e", "within", "pv"), ("J", "Fig5_i", "adjacent", "single_neuron"), ("K", "Fig5_j", "adjacent", "pv"), ("L", "Fig5_k", "anchor", "single_neuron"), ("M", "Fig5_l", "anchor", "pv")):
        key = ("within_single" if metric == "single_neuron" else "within_pv") if analysis == "within" else ("inter_single" if metric == "single_neuron" else "inter_pv")
        obs = sources[key]
        if analysis != "within":
            obs = obs[obs.pair_mode.eq(analysis)]
        summary_name = f"{analysis}_{'single' if metric == 'single_neuron' else 'pv'}"
        fig = h.plot_correlation_curve_panel(h._display_condition(obs), h._display_condition(descriptives[summary_name]), x_col="day" if analysis == "within" else "time_index", x_order=(1, 2, 3, 4, 5) if analysis == "within" else (1, 2, 3, 4), panel_kind="neuron" if metric == "single_neuron" else "pv", xlabel="Day" if analysis == "within" else "Day pair", ylabel="Spatial activity correlation (r)" if metric == "single_neuron" else "PV correlation (r)", stat_annotations=h.build_annotations(statistics, analysis=analysis, metric=metric), show_interaction=True, ylim=h.PLOT_LIMITS[old])[0]
        exports += save_panel(fig, output, panel, "spatial_stability", formats=formats, dpi=dpi)
    return exports


def _stability_cp_sources(data_root):
    def read(name):
        return read_table(data_root, "fig6/source_data/" + name + ".csv")
    return {
        "fig6_within": {"pv_bin": read("Fig6_within_pv_bin"), "pv_mouse_day": read("Fig6_within_pv_mouse_day"), "single_neuron": read("Fig6_within_single_neuron")},
        "fig6_interday": {"pv_bin": read("Fig6_interday_pv_bin"), "pv_mouse_pair": read("Fig6_interday_pv_mouse_pair"), "single_neuron": read("Fig6_interday_single_neuron")},
        "figs3": {"pv_bin": read("FigS3_interday_pv_bin"), "pv_mouse_pair": read("FigS3_interday_pv_mouse_pair"), "single_neuron": read("FigS3_interday_single_neuron")},
    }


def _activity_pair(data_root, stem):
    folder = Path(data_root) / "fig6/source_data/heatmap_matrices"
    return {name: np.load(folder / f"{stem}_{name}.npy") for name in ("map_a_display", "map_b_display")}


def figure6(data_root, output, *, refit=False, formats=("svg", "png"), dpi=300):
    from figure_helpers import figure6 as h
    from figure_helpers import figure6_adjacent as adj
    sources = _stability_cp_sources(data_root)
    statistics = read_table(data_root, "fig6/stats/Fig6_FigS3_primary_adjusted_estimands_raw_P.csv")
    if refit:
        contrasts, coefficients, _ = h.fit_models(sources, maxiter=1000)
        statistics = contrasts[contrasts.model_role.eq("adjusted")]
        write_tables({"stability_cp_refitted_estimands": contrasts, "stability_cp_refitted_coefficients": coefficients}, Path(output) / "statistics")
    exports = []
    payload = {day: _activity_pair(data_root, f"Fig6C_P_CL_D{day}_odd_even") for day in (1, 3, 5)}
    fig = h.plot_within_activity_grid(payload)
    exports += save_panel(fig, output, "Fig6_d", "odd_even_activity_maps", old_label="C", formats=formats, dpi=dpi)
    payload = {f"D{a}-D{b}": _activity_pair(data_root, f"Fig6G_P_CL_D{a}-D{b}") for a, b in ((1, 2), (4, 5))}
    fig = h.plot_interday_activity_grid(payload, groups=("P",), panel_letter="I")
    exports += save_panel(fig, output, "Fig6_i", "adjacent_day_activity_maps", formats=formats, dpi=dpi)
    fig = h.plot_pv_heatmap(sources["fig6_within"]["pv_bin"], groups=("P",), condition="CL", mode="within", panel_letter="E", vmin=.35, vmax=.85)
    exports += save_panel(fig, output, "Fig6_e", "within_PV_heatmap", formats=formats, dpi=dpi)
    adjacent_rows = read_table(data_root, "fig6j/source_data/Fig6J_adjacent_interday_locked_mouse_bin_rows.csv")
    fig = adj.plot_adjacent_fig6j(adjacent_rows)
    exports += save_panel(fig, output, "Fig6_j", "adjacent_PV_heatmap", formats=formats, dpi=dpi)
    for panel, analysis, metric in (("f", "within", "single_neuron"), ("g", "within", "pv"), ("k", "adjacent", "single_neuron"), ("l", "adjacent", "pv"), ("m", "anchor", "single_neuron"), ("n", "anchor", "pv")):
        key = "fig6_within" if analysis == "within" else "fig6_interday"
        obs = sources[key]["single_neuron" if metric == "single_neuron" else "pv_mouse_day" if analysis == "within" else "pv_mouse_pair"]
        if analysis != "within":
            obs = obs[obs.pair_mode.eq(analysis)]
        fig = h.plot_condition_curve_panel(obs, statistics, figure="Fig. 6", analysis=analysis, metric=metric, time_col="day" if analysis == "within" else "time_index", x_values=(1, 2, 3, 4, 5) if analysis == "within" else (1, 2, 3, 4), x_labels=("1", "2", "3", "4", "5") if analysis == "within" else ("D1-D2", "D2-D3", "D3-D4", "D4-D5") if analysis == "adjacent" else ("D1-D2", "D1-D3", "D1-D4", "D1-D5"), groups=("C", "P"), group_labels=h.FIG6_LABELS, conditions=("CL", "CR"), condition_labels=("Left", "Right"), ylabel="Single-neuron spatial activity correlation (r)" if metric == "single_neuron" else "PV correlation (r)", xlabel="Day" if analysis == "within" else "Day pair", panel_letter=panel.upper(), individual_scatter=metric == "pv")
        exports += save_panel(fig, output, f"Fig6_{panel}", "spatial_stability", formats=formats, dpi=dpi)
    # Old Fig.S3 is Fig.S6 in the current manuscript; Fig.S3 is excluded here.
    payload = {g: {f"D{a}-D{b}": _activity_pair(data_root, f"FigS3B_{g}_CLC_D{a}-D{b}") for a, b in ((1, 2), (2, 3), (3, 4), (4, 5))} for g in ("C", "I")}
    fig = h.plot_interday_activity_grid(payload, groups=("C", "I"), panel_letter="C")
    exports += save_panel(fig, output, "FigS6_c", "adjacent_activity_maps", formats=formats, dpi=dpi)
    obs = sources["figs3"]["pv_bin"]
    fig = h.plot_pv_heatmap(obs[obs.pair_mode.eq("adjacent")], groups=("C", "I"), condition="left", mode="adjacent", panel_letter="D", vmin=.05, vmax=.6)
    exports += save_panel(fig, output, "FigS6_d", "adjacent_PV_heatmap", formats=formats, dpi=dpi)
    for panel, mode in (("e", "adjacent"), ("f", "anchor")):
        fig = h.plot_figs3_metric_pair(sources["figs3"], statistics, mode=mode, panel_letter=panel.upper())
        exports += save_panel(fig, output, f"FigS6_{panel}", "spatial_stability", formats=formats, dpi=dpi)
    return exports


def figure7(data_root, output, *, refit=False, formats=("svg", "png"), dpi=300):
    from maze_analysis import fig7_figure8_pprh as f8, fig7_figure8_pprh_plotting as p8
    from maze_analysis import fig7_s4_tmaze_pprh as tm, fig7_s4_tmaze_pprh_plotting as pt
    from maze_analysis import figs4_ad_tmaze_pv as pv, figs4_ad_tmaze_pv_plotting as ppv
    from figure_helpers import manifold_heatmaps as hm
    exports = []
    table8 = read_table(data_root, "fig7_figure8/source_data/05_Fig7BE_longitudinal_model_table_D1-D5.csv")
    stats8 = read_table(data_root, "fig7_figure8/statistics/01_all_lmm_contrasts.csv")
    if refit:
        result = f8.fit_figure8_pprh_models(table8)
        stats8 = result.contrasts
        write_tables({"figure8_pprh_refitted_estimands": stats8}, Path(output) / "statistics")
    annotations = f8.build_annotation_payloads(stats8)
    bins = read_table(data_root, "fig7_figure8/source_data/01_Fig7A_pPRH_high_dimensional_PV_bins_D1-D5.csv")
    fig = p8.plot_pprh_pv_heatmap(bins, vmin=-.5, vmax=10.5)[0]
    exports += save_panel(fig, output, "Fig7_b", "PV_route_heatmap", formats=formats, dpi=dpi)
    for panel, metric, label in (("c", "high_dimensional_pv", "z-scored PV separation"), ("e", "neural_manifold", "z-scored d′")):
        fig = p8.plot_section_curves(table8, f8.build_group_day_summary(table8), metric=metric, ylabel=label, stat_annotations=annotations)[0]
        exports += save_panel(fig, output, f"Fig7_{panel}", "route_state_separation", formats=formats, dpi=dpi)
    ttable = read_table(data_root, "fig7_tmaze/source_data/08_CP_longitudinal_LMM_input.csv")
    tstats = read_table(data_root, "fig7_tmaze/statistics/Fig7_S4_LMM_all_contrasts.csv")
    if refit:
        result = tm.fit_cp_longitudinal_models(ttable)
        tstats = result.contrasts
        write_tables({"tmaze_pprh_refitted_estimands": tstats}, Path(output) / "statistics")
    tann = tm.build_annotation_payloads(tstats)
    bins = read_table(data_root, "fig7_tmaze/source_data/05_Fig7F_high_dimensional_PV_spatial_bin.csv")
    fig = pt.plot_fig7f_pv_heatmap(bins, vmin=-.5, vmax=2.5)[0]
    exports += save_panel(fig, output, "Fig7_g", "PV_phase_heatmap", old_label="F", formats=formats, dpi=dpi)
    for panel, old, metric, contrast in (("Fig7_h", "7G", "high_dimensional_pv_separation", "sample_vs_choice"), ("Fig7_j", "7I", "neural_manifold_separation", "sample_vs_choice"), ("FigS8_d", "S4E", "neural_manifold_separation", "sample_left_vs_right"), ("FigS8_h", "S4F", "neural_manifold_separation", "choice_left_vs_right")):
        fig = pt.plot_cp_section_curves(ttable, tm.build_group_day_summary(ttable), metric=metric, contrast=contrast, panel=old, annotations=tann)[0]
        exports += save_panel(fig, output, panel, "task_state_separation", old_label=old, formats=formats, dpi=dpi)
    raw = read_table(data_root, "pprh_manifold_heatmaps/source_data/01_pPRH_neural_manifold_mouse_day_bin_rows.csv")
    for spec, panel in zip(hm.PANEL_SPECS, ("Fig7_d", "Fig7_i")):
        selected = raw[raw.panel_key.eq(spec.key)].copy()
        fig = hm.render_panel(selected, spec)
        exports += save_panel(fig, output, panel, "manifold_heatmap", formats=formats, dpi=dpi)
    ptable = read_table(data_root, "supp_pv/source_data/08_FigS4BD_longitudinal_LMM_input.csv")
    pstats = read_table(data_root, "supp_pv/statistics/FigS4AD_LMM_all_contrasts.csv")
    if refit:
        pstats = pv.fit_longitudinal_models(ptable).contrasts
        write_tables({"route_pv_refitted_estimands": pstats}, Path(output) / "statistics")
    pann = pv.build_annotation_payloads(pstats)
    pbins = read_table(data_root, "supp_pv/source_data/06_FigS4AC_highdim_PV_spatial_bin.csv")
    for phase, heatpanel, curvepanel, oldh, oldc in (("sample_left_vs_right", "b", "c", "S4A", "S4B"), ("choice_left_vs_right", "f", "g", "S4C", "S4D")):
        fig = ppv.plot_pprh_spatial_heatmap(pbins, contrast=phase, panel=oldh, vmin=-1, vmax=1.5)[0]
        exports += save_panel(fig, output, f"FigS8_{heatpanel}", "route_PV_heatmap", old_label=oldh, formats=formats, dpi=dpi)
        fig = ppv.plot_cp_section_curves(ptable, pv.build_group_day_summary(ptable), contrast=phase, panel=oldc, annotations=pann)[0]
        exports += save_panel(fig, output, f"FigS8_{curvepanel}", "route_PV_separation", old_label=oldc, formats=formats, dpi=dpi)
    return exports


def decoding(data_root, output, *, refit=False, formats=("svg", "png"), dpi=300, comparisons=("CI", "CP")):
    from figure_helpers import bayesian_plots as h
    from maze_analysis.stats import build_group_day_summary
    from maze_analysis.plotting import add_project_colorbar
    from maze_analysis.position_decoding_summary import fit_position_inference
    exports = []
    for comparison, panel in (("CI", "FigS4"), ("CP", "FigS7")):
        if comparison not in comparisons:
            continue
        with np.load(Path(data_root) / f"decode_{comparison}/matrices.npz") as z:
            groups, days, matrices = z["groups"].astype(str), z["days"], z["mean_posterior"]
        fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.5), layout="constrained")
        vmax = h.ceiling(matrices, .45)
        for i, group in enumerate(groups):
            for j, day in enumerate(days):
                image = h.heatmap(axes[i, j], matrices[i, j], f"{group} D{day}", vmax, "Candidate position bin")
        add_project_colorbar(fig, image, axes, label="Mean posterior probability (linear)", ticks=np.linspace(0, vmax, 6))
        exports += save_panel(fig, output, f"{panel}_b", "mean_posterior", formats=formats, dpi=dpi)
        daily = read_table(data_root, f"decode_{comparison}/mouse_day.csv")
        daily = daily[daily.model.isin(("Bayes", "Shift control"))].copy()
        long = daily.melt(id_vars=["group", "mouse", "day", "model"], value_vars=["euclidean_mean_cm"], var_name="metric", value_name="value")
        summary = build_group_day_summary(long, "value", by=["model", "metric"])
        h.PLOT_GROUPS = tuple(groups)
        fig, ax = plt.subplots(figsize=(4.2, 3.5), layout="constrained")
        h.performance(ax, "euclidean_mean_cm", long, summary, days)
        ax.set_ylabel("Decoding error (cm)")
        exports += save_panel(fig, output, f"{panel}_c", "Euclidean_decoding_error", formats=formats, dpi=dpi)
        if refit:
            pattern = f"model_input*{'C-I' if comparison == 'CI' else 'C-P'}.csv"
            paths = list((Path(data_root) / "decode_statistics").glob(pattern))
            if len(paths) != 1:
                raise ValueError("Expected one decoding model-input table")
            inputs = pd.read_csv(paths[0])
            inputs["metric"] = "distance_cm"
            details = fit_position_inference(inputs, stat_groups=tuple(groups), stat_metrics=("distance_cm",), return_details=True)
            write_tables(details, Path(output) / "statistics" / panel)
    return exports


WORKFLOWS = {"3": figure3, "4": figure4, "5": figure5, "6": figure6, "7": figure7, "S4": partial(decoding, comparisons=("CI",)), "S7": partial(decoding, comparisons=("CP",))}


def run_figure(figure, data_root, output_root, *, refit=False, formats=("svg", "png"), dpi=300):
    if str(figure) not in WORKFLOWS and str(figure) != "S4-S7":
        raise ValueError(f"Choose one of {list(WORKFLOWS)}")
    out = Path(output_root) / ("Fig" + str(figure))
    out.mkdir(parents=True, exist_ok=True)
    workflow = decoding if str(figure) == "S4-S7" else WORKFLOWS[str(figure)]
    exports = workflow(Path(data_root), out, refit=refit, formats=tuple(formats), dpi=dpi)
    write_run_record(out / "run.json", figure=str(figure), refit=refit, exports=exports)
    return exports
