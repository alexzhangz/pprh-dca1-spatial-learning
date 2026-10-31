# Contributing

For a reproducibility issue, describe the figure, command, Python version, dependency versions and error. Include a small input example if it can be shared. Do not attach unpublished manuscript data in a public issue.

Run the following checks from the repository root:

```bash
python scripts/check_repository.py
python -m pytest -q
python scripts/run_demo.py --output-root outputs/demo
```

Changes to scientific calculations should identify their effect on manuscript results. Preserve mouse, day, phase, route and trial dimensions and the frozen cohort and analysis contracts. Changing a cohort, covariance estimator, day contrast or stochastic seed is an analysis change, even if the API is unchanged. Use a new output directory for revised results.

The hashes in `provenance/source_definitions.csv` identify original source segments. `provenance/source_files.csv` records the collected release modules. The initial upload snapshot is described by `provenance/release_files.sha256`; update that snapshot deliberately when preparing a new release. Statistical refits require the separate manuscript inputs; the synthetic example checks execution only.
