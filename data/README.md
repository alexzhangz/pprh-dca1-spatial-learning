# Manuscript source data

This repository contains code, parameters and file manifests. Neural, behavioral and source-table data are separate inputs and are not bundled here. There is currently no dataset download link in this repository. The synthetic example and scientific unit tests can run without those data; the manuscript figure workflows require them.

Place the matching source-data directory at `data/input/`, supply `--data-root`, or set `PPRH_DATA_DIR` for the notebooks. Check file identity with `python main.py --check-data --data-root /path/to/manuscript-source-data`. The MIT code licence does not determine the licence of the separate data.

`required_files.csv` lists the 99 frozen inputs needed by the figure workflows and estimate checks, with relative paths, sizes and SHA-256 values. This list was checked against actual file reads during execution of all figure notebooks: 95 directly accessed files plus four frozen statistical reference tables. Unused output tables and alternative heatmap arrays from historical release folders are omitted. Raw aligned inputs for notebook 00 are additional data, described below rather than included in this source-table inventory. The top-level directory contains:

```text
fig3/                       Behavior mouse-day/trial tables and frozen LMMs
fig3_trial/                 Atomic trials 1–90 and final GEE reference estimates
fig4/                       Historical Fig.4 stability source tables and map arrays
fig5/                       Historical Fig.5 manifold/PV source tables and LMMs
fig6/                       pPRH and T-maze stability source tables and map arrays
fig6j/                      Accepted adjacent-day PV source for current Fig.6j
fig7_figure8/               Figure-8 C/P separation tables and model estimates
fig7_tmaze/                 T-maze C/P separation tables and model estimates
supp_pv/                    T-maze C/P within-phase PV separation tables
pprh_manifold_heatmaps/     Mouse/day/bin rows used by pPRH manifold heatmaps
decode_CI/                  Mouse/day decoder errors and mean posterior matrices
decode_CP/                  Matching C/P decoder data
decode_statistics/          Decoder model-input and reference-estimate tables
representatives/            Frozen representative embedding coordinates
```

Historical folder labels describe the original source releases, not the current manuscript figure numbering. Numeric source values are preserved. The figure workflows use only the required source columns; original provenance columns may retain earlier local path strings.

CSV tables preserve mouse, group, recording day, route/phase contrast and spatial bin or section, where applicable. Missing contrast/day combinations remain missing when minimum trial support is not met. Correlation tables contain raw `r` for display and Fisher `z` for inference. Separation `value` fields are the released shuffle-standardized separation, with positive values indicating stronger separation. Decoder errors are in cm; the original shared inference helper reports native effect/SE/CI columns in mm, while z/P are scale-invariant. This unit convention is retained during refitting.

The source-table dataset supports the figure workflows and statistical refits. Earlier aligned data are additionally needed to recompute neural metrics:

- Figure-8: per-subject `xy_trial_all.npy`, `d_dff_trial_all.npy`, `trial_labels.npy`, `snarm_1d.npy`, and the available trial context/segment metadata. Neural arrays and trial-label rows must retain their original contracts. The manuscript quantitative map has 56 path bins; Bayesian decoding uses a separate 98-bin physical geometry.
- T-maze: per-subject `Snarm_trial_v2.npy`, `Snarm_1d_v2.npy`, `index_all.npy`, and original aligned trajectories/traces for the analyses using them. Preserve sample and choice phases. The current path maps have 29 bins.
- Reproducing the exact Bayesian training run also requires the frozen per-mouse neuron indices, trial-fold assignments and shuffle seeds from its session records. Use those records when requesting exact numerical parity; newly drawing neurons/folds changes the stochastic computation.

Object-valued NumPy inputs are loaded with `allow_pickle=True` only in the original aligned-data loaders. Supply files from the authors' corresponding dataset. The source-table figure workflows load ordinary CSV and numeric NPZ/NPY arrays.
