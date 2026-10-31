"""Manuscript definitions collected from stats.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import warnings

import numpy as np

import pandas as pd

from scipy.stats import sem

def _as_column_tuple(columns):
    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    return tuple(columns)

def sem_no_warning(values, axis=None):
    """Return scipy SEM with NaN omission while suppressing empty-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sem(values, axis=axis, nan_policy="omit")

def benjamini_hochberg(p_values, alpha=0.05):
    """Return Benjamini-Hochberg q-values and significance mask."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    significant = np.zeros(p_values.shape, dtype=bool)
    finite = np.isfinite(p_values)
    if not np.any(finite):
        return q_values, significant

    finite_p = p_values[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n_test = ranked.size
    adjusted = ranked * n_test / np.arange(1, n_test + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q_finite = np.empty_like(adjusted)
    q_finite[order] = adjusted
    q_values[finite] = q_finite
    significant[finite] = q_finite <= alpha
    return q_values, significant

def build_group_day_summary(
    table,
    value_col,
    by=(),
    group_col="group",
    day_col="day",
):
    """Summarize finite observations by locator, group, and day."""
    keys = []
    for column in _as_column_tuple(by) + (group_col, day_col):
        if column not in keys:
            keys.append(column)
    required = set(keys + [value_col])
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"table is missing columns: {missing}")
    rows = []
    grouper = keys[0] if len(keys) == 1 else keys
    for key, group_table in table.groupby(
        grouper, observed=True, dropna=False, sort=False
    ):
        key_values = (key,) if len(keys) == 1 else tuple(key)
        values = pd.to_numeric(group_table[value_col], errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        row = dict(zip(keys, key_values))
        row["mean"] = float(values.mean()) if values.size else np.nan
        row["sem"] = float(sem_no_warning(values)) if values.size else np.nan
        row["n_observation"] = int(values.size)
        rows.append(row)
    return pd.DataFrame(rows, columns=keys + ["mean", "sem", "n_observation"])
