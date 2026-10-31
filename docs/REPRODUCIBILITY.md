# Reproducibility and validation

## Levels of execution

1. **Code checks and synthetic example.** Install `requirements.txt`, run `scripts/check_repository.py`, `pytest`, and `scripts/run_demo.py`. No manuscript data are required. The example uses the included Gaussian Bayesian decoder with deterministic synthetic neural activity; its output is not a manuscript result.
2. **Manuscript panels.** Supply the separate input files listed in `data/required_files.csv`. `main.py` renders the source tables with frozen statistical estimates. `--refit` refits the original statistical models from those tables.
3. **Original neural computations.** `demos/00_recompute_metrics.ipynb` exposes the included original neural-computation APIs and aligned-data contracts. Full-cohort neural sampling, embedding, shuffling and decoder training are separate computations and are not performed by `--refit`.

## Completed manuscript checks

The collection was validated with Python 3.12 using the package versions in `provenance/tested_environment.json`.

- All 52 collected original source modules imported from an extracted standalone archive.
- All 304 source-data files matched their SHA-256 inventory.
- The 13 retained Bayesian decoder and posterior tests passed.
- Eight notebooks passed structure and syntax checks. The Figure 3 notebook was executed; all seven figure workflows were executed through their common Python API.
- The figure workflows produced 60 PNG exports. Sampled stability, separation and Bayesian-error panels were inspected visually.
- Statistical refits matched 882 contrast records across 11 analysis families, comparing effects/coefficient estimates, standard errors and P values with relative tolerance `1e-6` and absolute tolerance `1e-8`. The largest absolute P-value difference was approximately `2.71e-8`.

Numerical details and the validation boundary are in `provenance/validation_summary.json`. This does not establish pixel equality with the manually assembled manuscript figures. Full raw neural computations have not been rerun for this collection, and a fresh installation on a second computer has not been verified locally.

## Repository checks

`scripts/check_repository.py` checks required files, source integrity, Python imports, notebook structure and syntax, portable paths, and input-manifest paths. GitHub Actions runs it, the scientific tests and the synthetic example on Ubuntu and Windows using Python 3.12. These automated jobs do not have manuscript data and therefore do not claim to reproduce manuscript statistics. A workflow result exists only after the repository is uploaded and the job completes.

The workflow follows the [official GitHub Python workflow documentation](https://docs.github.com/en/actions/tutorials/build-and-test-code/python). Citation metadata follows the [Citation File Format](https://citation-file-format.github.io/).

## Provenance and adapted paths

Original source-file and definition hashes remain in `provenance/source_files.csv` and `provenance/source_definitions.csv`. Historical filenames and analysis-version identifiers are retained. `provenance/release_adjustments.json` records changes needed to collect the code and make paths portable. Hashes inside historical contracts refer to their historical source records; `provenance/release_files.sha256` identifies the actual files in this upload snapshot. No manuscript DOI, publication status or repository URL has been invented.
