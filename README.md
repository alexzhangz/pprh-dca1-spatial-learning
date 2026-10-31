# pPRH-dCA1 spatial-learning analysis

Analysis code accompanying **Direct posterior perirhinal input guides dorsal CA1 task-state representations during learning**, Xu et al., manuscript version dated 30 October 2026.

**Version:** 0.1.0 · **Python:** 3.12 · **Licence:** [MIT](LICENSE)

[Figure-to-code map](figure_panel_map.csv) · [Input data](data/README.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Citation](CITATION.cff)

This code collection follows **Xu et al., manuscript dated 30 October 2026** (`Xu et_al_manuscript_20261030.docx`). It contains the computational analyses for Figures 3–7 and their associated analysis supplements, Figures S4–S8. The manuscript file is identified by SHA-256 in `provenance/manuscript.json`; the document itself is not included.

The organization follows a main script, shared functions and figure-specific Jupyter notebooks. Scientific functions were collected from the Figure-8 and DNMP T-maze projects by tracing the selected functions and their imports. Historical exploratory notebooks, CEBRA analyses, CV-PCA manifold controls, KNN decoding results and unrelated experiments are not included.

## Layout

```text
main.py                       Run one figure or all included figures
environment.yml               Validation environment
requirements.txt              The same Python package versions
demos/
  00_recompute_metrics.ipynb   Neural-computation methods and aligned-data interfaces
  figure_3.ipynb               Learning curves and trial-level GEE
  figure_4.ipynb               dCA1 task-state separation and Fig. S5
  figure_5.ipynb               dCA1 spatial stability
  figure_6.ipynb               pPRH spatial stability and Fig. S6
  figure_7.ipynb               pPRH task-state separation and Fig. S8
  figure_S4.ipynb              Control/inhibition Bayesian position decoding
  figure_S7.ipynb              dCA1/pPRH Bayesian position decoding
src/
  paper.py                    Figure workflows using current panel numbers
  paper_io.py                 Explicit data loading and figure export
  figure_helpers/             Collected figure and visualization functions
  maze_analysis/              Shared Figure-8 and manuscript functions
  t_maze/                     Shared T-maze functions
  config/                     Original cohort and method contracts
data/
  README.md                   Input contracts and separate data preparation
  required_files.csv          Source-data file paths, sizes and SHA-256 values
provenance/                   Source code, parameters and manuscript identity
tests/                        Tests of the included Bayesian and posterior methods
```

## Installation and execution

Run commands from the repository root. Install the pinned environment with Conda:

```bash
conda env create -f environment.yml
conda activate pprh-manuscript
```

Alternatively, with Python 3.12 in your own virtual environment:

```bash
python -m pip install -r requirements.txt
```

Start with the checks and a synthetic example, which require no manuscript data:

```bash
python scripts/check_repository.py
python -m pytest -q
python scripts/run_demo.py --output-root outputs/demo
```

The example runs the included Gaussian Bayesian decoder and writes a posterior plot, numeric posterior matrix, predictions and a run record. Every output is explicitly labelled as synthetic; it is an installation example rather than a reproduction of a manuscript result.

To reproduce manuscript panels, obtain the separate source-data files listed in [data/required_files.csv](data/required_files.csv), then run:

```bash
python main.py --check-data --data-root /path/to/manuscript-source-data
python main.py --figure 4 --data-root /path/to/manuscript-source-data
python main.py --figure all --data-root /path/to/manuscript-source-data --refit
```

The [notebook guide](demos/README.md) links to the expanded figure workflows. Each notebook shows local functions, source-data loading, analysis parameters, statistical fitting and separate panel-plotting cells. They do not delegate the figure to `run_figure()`. Shared numerical estimators remain in `src/`.

Run Jupyter from this directory. Set `PPRH_DATA_DIR` to the separate source-data directory, or change `DATA_ROOT` in the notebook. Notebooks default to statistical refitting; set `REFIT_STATISTICS=False` to use frozen estimates. Notebook outputs go under `outputs/notebooks/` by default, or the directory specified by `PPRH_OUTPUT_DIR`. CLI outputs go under `outputs/` unless `--output-root` is supplied.

Without `--refit`, the program renders the provided source data using frozen manuscript statistical estimates. With `--refit`, it refits the original manuscript models from the source tables before rendering. This option refits statistical models; it does not rerun the neural sampling, embeddings or decoder training. The corresponding original neural-computation functions and their aligned-data contracts are provided in `00_recompute_metrics.ipynb` and `src/`. Inputs and outputs are separate: original input data are never replaced by the figure workflows.

## Figure coverage

| Notebook | Current computational panels | Historical source numbering |
|---|---|---|
| Figure 3 | 3d, 3e, 3h | 3D/E/H; 3e uses the final trial-level GEE |
| Figure 4 | 4d, 4f–h, 4j, 4l–n; S5a–f | Earlier Fig.5 separation analyses, plus subsequent manifold heatmaps and phase visualization |
| Figure 5 | 5b–e, 5g–l | Earlier Fig.4 stability analyses |
| Figure 6 | 6d–g, 6i–n; S6c–f | Earlier Fig.6 and Fig.S3 stability analyses; 6j uses the adjacent-day replacement |
| Figure 7 | 7b–e, 7g–j; S8b–d, S8f–h | Earlier Fig.7 and Fig.S4 separation analyses, plus pPRH manifold heatmaps |
| Figure S4 | S4b–c | 98-bin Gaussian Bayesian position decoding, C/I |
| Figure S7 | S7b–c | The same Bayesian position decoder, C/P |

Static diagrams, histology, manipulation validation and independent behavioral experiments are outside this collection. In particular, post-training behavior in Fig.3i and Figs.S1–S3/S9–S10 are excluded from this preparation scope. The scripts export computational panels for assembly; they do not rebuild the entire manuscript page layout. Figure-8 manifold point clouds and mean trajectories are exported separately for the representative panel. Source-table filenames retain historical labels to preserve traceability; use `figure_panel_map.csv` for current panel identifiers.

## Frozen analysis choices

- All main learning analyses retain D1–D5. Interday stability retains each adjacent or Day-1-anchored day pair. No across-day mouse averaging is added by this collection.
- Quantitative manifold analyses use 50 neurons, 15-dimensional SpectralEmbedding, 100 repeats, 100 label shuffles and **`legacy_pooled` covariance**. The three-dimensional representative visualizations are distinct from the quantified 15-dimensional analysis.
- Figure-8 within-day reliability uses condition-ordinal odd/even correct-trial maps. Single-neuron and PV correlations are Pearson and Spearman correlations, respectively. Correlations are displayed as raw `r` and modelled on the Fisher `z` scale.
- Control/inhibition population models retain mean-speed and lateral-deviation adjustment. Direct dCA1/pPRH recording comparisons retain lateral-deviation adjustment. The Bayesian decoding models have no movement covariates.
- Fig.3e uses binary outcomes from cumulative trials 1–90, a mouse-clustered binomial GEE, exchangeable working correlation, bias-reduced covariance and Student-t inference with 8 degrees of freedom. The moving windows are for display only.
- Bayesian decoding uses 98 physical spatial bins, five complete-trial folds, 50 fixed neurons per mouse across days and 100 within-trial circular label shifts. Posterior matrices average within true bin per mouse before averaging mice equally. Shuffle errors are averaged within mouse/day, not treated as biological replicates.
- Frozen model choices, cohort exclusions and their recorded limitations are retained. In particular, outcome-informed exclusions remain identified as such, and the original single-neuron clustering choice is not silently replaced. The collection reproduces the manuscript analyses rather than changing its inference strategy.

Parameters and input contracts are in `src/config/` and `provenance/neural_metric_parameters.csv`. Original contract paths have been converted to logical source-project paths for provenance. Executable figure workflows use the explicitly supplied data directory; the historical project layout is not required for these workflows.

## Validation

The collected modules were imported from this directory and the figure workflows were executed against the manuscript source data. Original Bayesian/posterior unit tests were retained. Statistical models were refitted and their coefficients, standard errors and probabilities compared with the corresponding frozen result tables. See `provenance/validation_summary.json` for the completed checks and their numerical tolerances.

The validation environment is recorded separately from the historical analysis records. This collection has not been tested in a freshly installed environment on another computer. GitHub Actions is configured to check imports, notebooks, scientific unit tests and the synthetic example on Ubuntu and Windows after upload. It does not run the manuscript analyses without their separate input data. See [the validation details](docs/REPRODUCIBILITY.md) for the precise boundary.

## Data availability

Manuscript input data are separate from this code repository. The code includes an exact input inventory and a synthetic example; it does not currently include a public download link for the manuscript dataset. The figure workflows require the matching input files. See [data/README.md](data/README.md) for file layout and raw aligned-data contracts.

## Citation and licence

Please cite the accompanying manuscript when using these analyses. [CITATION.cff](CITATION.cff) contains the author list and manuscript title. Publication details and a DOI have not been assigned in this version.

Code is distributed under the [MIT licence](LICENSE). The code licence does not assign a licence to separately distributed manuscript data. For reproducibility issues, see [CONTRIBUTING.md](CONTRIBUTING.md).
