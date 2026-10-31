# pPRH–dCA1 spatial-learning analysis

Code accompanying **Direct posterior perirhinal input guides dorsal CA1 task-state representations during learning**, Xu et al.

The repository contains the Figure-8 and DNMP T-maze computational analyses for Figures 3–7 and supplementary Figures S4–S8.

## Notebooks

| Notebook | Analysis |
|---|---|
| [00_recompute_metrics](demos/00_recompute_metrics.ipynb) | Original neural-computation functions and opt-in aligned-data examples |
| [Figure 3](demos/figure_3.ipynb) | Learning curves and trial-level GEE; panels 3d/e/h |
| [Figure 4](demos/figure_4.ipynb) | dCA1 task-state separation, representative manifolds and S5 |
| [Figure 5](demos/figure_5.ipynb) | dCA1 within-day and interday spatial stability |
| [Figure 6](demos/figure_6.ipynb) | pPRH spatial stability and T-maze S6 |
| [Figure 7](demos/figure_7.ipynb) | pPRH route/phase separation and S8 |
| [Figure S4](demos/figure_S4.ipynb) | Control/inhibition Bayesian position decoding |
| [Figure S7](demos/figure_S7.ipynb) | dCA1/pPRH Bayesian position decoding |

Each notebook shows data loading, parameters, statistical fitting and panel plotting. Shared methods are in `src/`. The [panel map](figure_panel_map.csv) identifies the current manuscript panels and their source modules.

## Installation

Use Python 3.12. Create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate pprh-manuscript
```

Alternatively, install into an existing Python 3.12 virtual environment with `python -m pip install -r requirements.txt`.

## Run

Run commands from the repository root, supplying the separate manuscript source-data directory:

```bash
python main.py --check-data --data-root /path/to/source-data
python main.py --figure 4 --data-root /path/to/source-data --refit
python main.py --figure all --data-root /path/to/source-data --refit
```

Without `--refit`, the CLI uses frozen statistical estimates. `--refit` refits models from the source tables; it does not rerun neural sampling, embeddings or decoder training. CLI figures are written to `outputs/`.

For notebooks, set `PPRH_DATA_DIR` or edit `DATA_ROOT`, then run cells in order. Statistical refitting is enabled by default. Notebook outputs go to `outputs/notebooks/`. Raw neural computations in notebook 00 require additional aligned data and original session choices; their expensive switches default to disabled.

To check installation without manuscript data:

```bash
python scripts/check_repository.py
python -m pytest -q
python scripts/run_demo.py
```

The demo uses explicitly labelled synthetic data. See [reproducibility details](docs/REPRODUCIBILITY.md) for validation and execution limits.

## Data

Manuscript data are separate from this code repository; a dataset download link is not included here. The [input guide](data/README.md) describes the data layout, and [required_files.csv](data/required_files.csv) lists the figure inputs and their hashes. The [cohort definition](src/config/manuscript_cohort_v3_20260804.json) and [neural-metric parameters](provenance/neural_metric_parameters.csv) specify the analysis choices.

## Citation and licence

Please cite the accompanying manuscript when using these analyses. Citation metadata are in [CITATION.cff](CITATION.cff). Code is released under the [MIT licence](LICENSE); separately distributed data have their own terms.
