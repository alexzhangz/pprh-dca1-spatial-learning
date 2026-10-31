# Reproducibility and validation

## Levels of execution

1. **Code checks and synthetic example.** Install `requirements.txt`, run `scripts/check_repository.py`, `pytest`, and `scripts/run_demo.py`. No manuscript data are required. The example uses the included Gaussian Bayesian decoder with deterministic synthetic neural activity; its output is not a manuscript result.
2. **Manuscript panels.** Supply the separate input files listed in `data/required_files.csv`. `main.py` renders the source tables with frozen statistical estimates. `--refit` refits the original statistical models from those tables.
3. **Original neural computations.** `demos/00_recompute_metrics.ipynb` exposes the included original neural-computation APIs and aligned-data contracts. Full-cohort neural sampling, embedding, shuffling and decoder training are separate computations and are not performed by `--refit`.

## Completed manuscript checks

The code was validated with Python 3.12 using the pinned package versions in `requirements.txt` and `environment.yml`.

- All retained analysis modules import from the standalone repository. `provenance/source_files.csv` identifies their original sources and current file hashes.
- All 99 source-data files in the current inventory matched their SHA-256 values. The inventory contains 95 files read by the figure notebooks and four frozen statistical reference tables.
- The 13 retained Bayesian decoder and posterior tests passed.
- All eight expanded notebooks passed structure and syntax checks and were executed cell by cell in isolated Python 3.12 kernels. All seven figure notebooks ran with statistical refitting enabled; expensive raw-data switches in notebook 00 remained disabled.
- The figure workflows produced 60 PNG exports. Sampled stability, separation and Bayesian-error panels were inspected visually.
- Statistical refits matched 882 contrast records across 11 analysis families, comparing effects/coefficient estimates, standard errors and P values with relative tolerance `1e-6` and absolute tolerance `1e-8`. The largest absolute P-value difference was approximately `2.71e-8`.

These checks do not establish pixel equality with the manually assembled manuscript figures. Full-cohort raw neural computations have not been rerun. GitHub Actions separately installs the pinned environment and runs the repository checks, 13 scientific tests and synthetic decoder example on fresh Ubuntu and Windows runners.

The expanded notebook cells produced 60 PNG exports and refitted the same 882 contrast records. Notebook 00 contains explicit per-subject computation functions and calls for manifold, stability, PV shuffles and a one-session decoder; execution with raw-data switches disabled checks definitions/imports rather than the expensive raw computations. Its decoder function was separately exercised with 10 synthetic trials, 50 neurons, five trial folds and 100 within-trial shifts. That synthetic check is not a full-cohort manuscript rerun.

## Repository checks

`scripts/check_repository.py` checks required files, source integrity, Python imports, notebook structure and syntax, portable paths, and input-manifest paths. It also requires notebook-local functions and rejects delegation to `run_figure()`. GitHub Actions runs it, the scientific tests and the synthetic example on Ubuntu and Windows using Python 3.12. These automated jobs do not have manuscript data and therefore do not claim to reproduce manuscript statistics. A workflow result exists only after the repository is uploaded and the job completes.

The workflow follows the [official GitHub Python workflow documentation](https://docs.github.com/en/actions/tutorials/build-and-test-code/python). Citation metadata follows the [Citation File Format](https://citation-file-format.github.io/).

## Provenance and adapted paths

Original source-file hashes and current module hashes are recorded in `provenance/source_files.csv`; frozen input hashes are in `data/required_files.csv`. Historical source filenames remain traceable through `figure_panel_map.csv`. The manuscript cohort and numerical settings remain in the cohort JSON, metric-parameter CSV and executable notebook cells.

Odd/even maps display D1/D3/D5 while longitudinal analyses retain D1–D5. The adjacent-day 6j heatmap uses its accepted adjacent-day source table and the shared Fig.6 plotting implementation.
