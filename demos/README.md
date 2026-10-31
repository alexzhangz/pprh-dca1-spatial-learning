# Notebook guide

These notebooks expose analysis steps for the 30 October 2026 manuscript. Run them from top to bottom with the installed environment and the separate manuscript inputs. Each figure notebook contains notebook-local helpers, readable source-data paths, parameter choices, statistical-model calls, inspection tables and separate panel-plotting cells. Shared numerical estimators and plotting primitives are implemented in `../src/`.

| Notebook | Content |
|---|---|
| [00_recompute_metrics](00_recompute_metrics.ipynb) | Explicit functions and opt-in aligned-data calls for manifold separation, odd/even and interday stability, PV shuffle separation and the Gaussian Bayesian decoder |
| [Figure 3](figure_3.ipynb) | Figure-8/T-maze daily success, trial-level binomial GEE and panels 3d/e/h |
| [Figure 4](figure_4.ipynb) | dCA1 state separation, representative manifolds and S5 |
| [Figure 5](figure_5.ipynb) | dCA1 odd/even, adjacent-day and D1-anchor stability |
| [Figure 6](figure_6.ipynb) | pPRH stability, the accepted adjacent-day 6j heatmap and T-maze S6 |
| [Figure 7](figure_7.ipynb) | pPRH route/phase separation and within-phase S8 |
| [Figure S4](figure_S4.ipynb) | C/I Bayesian posterior, mouse-day decoding errors and inference |
| [Figure S7](figure_S7.ipynb) | C/P Bayesian posterior, mouse-day decoding errors and inference |

The figure notebooks use analysis source tables and frozen numeric maps/embeddings. Set `PPRH_DATA_DIR` or edit `DATA_ROOT` in the first code cell. Set `REFIT_STATISTICS=False` to use frozen estimates; the default refits the manuscript models. `FORMATS` and `DPI` control export. `PPRH_OUTPUT_DIR` or `OUTPUT_ROOT` chooses a separate output directory. The default notebook location is `outputs/notebooks/` with a subdirectory per figure.

Notebook 00 needs additional aligned trial data and original stochastic-session choices. Its expensive switches default to `False`. It preserves mouse/day/phase/route dimensions and writes new outputs, without changing the released source data. Defining these functions or running the disabled cells does not constitute a raw-data rerun. The one-session decoder example does not recreate the entire original cohort batch or the published Euclidean-error table automatically.

Historical filenames are retained: old Fig.4 stability becomes current Fig.5; old Fig.5 separation becomes current Fig.4; historical stability S3 becomes current S6; historical separation S4 becomes current S8. Use [the current panel map](../figure_panel_map.csv) for attribution.

All eight notebooks have been executed in isolated kernels. Figure notebooks generated 60 PNG exports and refitted 882 contrast records matching the frozen manuscript estimates. Raw switches remained disabled; the notebook decoder function also passed a separate synthetic-session check. Execution records and limits are in [the validation documentation](../docs/REPRODUCIBILITY.md). Public notebook files have clean outputs and empty execution counts.
