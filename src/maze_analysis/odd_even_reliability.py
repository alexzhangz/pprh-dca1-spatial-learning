"""Manuscript definitions collected from odd_even_reliability.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import numpy as np

import pandas as pd

SPLIT_STRATEGY = "condition_ordinal_odd_even_v1"

PART_TO_CONDITION = {"left": "CL", "right": "CR"}

def fisher_z(values):
    """Return finite Fisher-z values while preserving NaNs."""
    values = np.asarray(values, dtype=float)
    bound = np.nextafter(1.0, 0.0)
    clipped = np.clip(values, -bound, bound)
    return np.arctanh(clipped)

def build_pv_bin_source(records, *, num_day=5, num_bin=56):
    """Build a complete mouse/day/condition/bin table from analysis-14 payloads."""
    rows = []
    for record in records:
        group = str(record["group"])
        mouse = str(record["mouse"])
        payload = record["payload"]
        if payload.get("split_strategy") != SPLIT_STRATEGY:
            raise ValueError(f"unexpected split strategy for {mouse}")
        for part, condition in PART_TO_CONDITION.items():
            values = np.asarray(payload[part], dtype=float)
            if values.shape != (num_day, num_bin):
                raise ValueError(
                    f"analysis-14 {mouse}/{part} shape {values.shape}, expected {(num_day, num_bin)}"
                )
            for day_idx in range(num_day):
                valid_day = bool(payload[f"valid_day_{part}"][day_idx])
                trial_count = int(payload[f"trial_count_{part}"][day_idx])
                odd_count = int(payload[f"odd_trial_count_{part}"][day_idx])
                even_count = int(payload[f"even_trial_count_{part}"][day_idx])
                for bin_idx, value in enumerate(values[day_idx]):
                    rows.append(
                        {
                            "group": group,
                            "mouse": mouse,
                            "day": day_idx + 1,
                            "day_c": float(day_idx - 2),
                            "condition": condition,
                            "spatial_bin": int(bin_idx),
                            "raw_r": float(value) if np.isfinite(value) else np.nan,
                            "valid_day": valid_day,
                            "trial_count": trial_count,
                            "odd_trial_count": odd_count,
                            "even_trial_count": even_count,
                            "candidate_neuron_count": int(
                                payload["candidate_neuron_count"][day_idx]
                            ),
                            "fixed_neuron_target": 50,
                            "neuron_subsample_iterations": 100,
                            "pv_method": str(payload.get("pv_method", "spearman")),
                            "split_strategy": SPLIT_STRATEGY,
                        }
                    )
    return pd.DataFrame(rows)

def build_pv_mouse_day_table(pv_bin_source):
    """Reduce spatially resolved PV correlations to one value per mouse-day-condition."""
    required = {
        "group",
        "mouse",
        "day",
        "day_c",
        "condition",
        "raw_r",
        "trial_count",
        "odd_trial_count",
        "even_trial_count",
        "candidate_neuron_count",
    }
    missing = sorted(required.difference(pv_bin_source.columns))
    if missing:
        raise ValueError(f"PV bin source is missing columns: {missing}")

    rows = []
    keys = ["group", "mouse", "day", "day_c", "condition"]
    for key, table in pv_bin_source.groupby(keys, observed=True, sort=False):
        raw = pd.to_numeric(table["raw_r"], errors="coerce").to_numpy(dtype=float)
        finite = raw[np.isfinite(raw)]
        raw_mean = float(np.mean(finite)) if finite.size else np.nan
        rows.append(
            {
                **dict(zip(keys, key)),
                "metric": "odd_even_pv_correlation",
                "section": "All bins",
                "raw_r": raw_mean,
                "fisher_z": float(fisher_z(raw_mean)) if np.isfinite(raw_mean) else np.nan,
                "n_bin_finite": int(finite.size),
                "trial_count": int(table["trial_count"].iloc[0]),
                "odd_trial_count": int(table["odd_trial_count"].iloc[0]),
                "even_trial_count": int(table["even_trial_count"].iloc[0]),
                "candidate_neuron_count": int(table["candidate_neuron_count"].iloc[0]),
                "split_strategy": SPLIT_STRATEGY,
            }
        )
    return pd.DataFrame(rows)

def build_single_neuron_source(records, *, num_day=5):
    """Build neuron-level raw and Fisher-z source data from analysis-16 payloads."""
    frames = []
    qc_frames = []
    for record in records:
        group = str(record["group"])
        mouse = str(record["mouse"])
        payload = record["payload"]
        if payload.get("split_strategy") != SPLIT_STRATEGY:
            raise ValueError(f"unexpected split strategy for {mouse}")
        table = payload["long_table"].copy()
        table = table.loc[
            table["day"].between(1, num_day)
            & table["condition"].isin(PART_TO_CONDITION.values())
        ].copy()
        table["group"] = group
        table["mouse"] = mouse
        table["day_c"] = pd.to_numeric(table["day"], errors="coerce") - 3.0
        table = table.rename(columns={"pc_within_corr": "raw_r"})
        table["fisher_z"] = fisher_z(table["raw_r"].to_numpy(dtype=float))
        table["split_strategy"] = SPLIT_STRATEGY
        frames.append(
            table[
                [
                    "group",
                    "mouse",
                    "day",
                    "day_c",
                    "condition",
                    "neuron",
                    "raw_r",
                    "fisher_z",
                    "split_strategy",
                ]
            ]
        )

        qc = payload["qc_table"].copy()
        qc = qc.loc[
            qc["day"].between(1, num_day)
            & qc["condition"].isin(PART_TO_CONDITION.values())
        ].copy()
        qc["group"] = group
        qc["mouse"] = mouse
        qc_frames.append(qc)

    source = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    qc = pd.concat(qc_frames, ignore_index=True) if qc_frames else pd.DataFrame()
    identity = ["group", "mouse", "day", "condition", "neuron"]
    if not source.empty and source.duplicated(identity, keep=False).any():
        raise ValueError("duplicate single-neuron source rows")
    return source, qc
