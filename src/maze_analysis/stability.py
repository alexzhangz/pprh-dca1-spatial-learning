"""Manuscript definitions collected from stability.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import warnings

import numpy as np

from .correlation import pc_corr, rowwise_corr, subsampled_pair_pv_corr, subsampled_pv_corr

from .trials import eligible_previous_current_indices, valid_labeled_trials

PV_GAP_COLUMNS = [
    "pv_pair",
    "condition",
    "trial_i",
    "trial_j",
    "trial_gap",
    "bin",
    "pv_corr",
    "n_neuron",
    "selected_neuron_count",
]

INTERDAY_PC_COLUMNS = [
    "group",
    "mouse",
    "pair_mode",
    "pair_index",
    "day_a",
    "day_b",
    "day_pair_label",
    "condition",
    "condition_key",
    "neuron",
    "pc_interday_corr",
]

def trial_spatial_corr(prev_maps, curr_maps, method="pearson"):
    """Return per-pair, per-neuron spatial correlations across bins."""
    prev_maps = np.asarray(prev_maps, dtype=float)
    curr_maps = np.asarray(curr_maps, dtype=float)
    if prev_maps.shape != curr_maps.shape:
        raise ValueError(f"Input shapes must match, got {prev_maps.shape} and {curr_maps.shape}")
    if prev_maps.ndim != 3:
        raise ValueError(f"Expected arrays with shape (pair, neuron, bin), got {prev_maps.shape}")

    n_pair, n_neuron, n_bin = prev_maps.shape
    corr = rowwise_corr(
        prev_maps.reshape(n_pair * n_neuron, n_bin),
        curr_maps.reshape(n_pair * n_neuron, n_bin),
        method=method,
    )
    return corr.reshape(n_pair, n_neuron)

def interday_day_pairs(num_day, pair_mode):
    """Return zero-based day-pair records for adjacent or D1-anchor comparisons."""
    num_day = int(num_day)
    if num_day < 2:
        return []
    mode = str(pair_mode).lower().replace("-", "_").replace(" ", "_")
    if mode not in {"adjacent", "anchor"}:
        raise ValueError("pair_mode must be 'adjacent' or 'anchor'")

    rows = []
    if mode == "adjacent":
        pairs = [(day_idx, day_idx + 1) for day_idx in range(num_day - 1)]
    else:
        pairs = [(0, day_idx) for day_idx in range(1, num_day)]
    for pair_index, (day_a, day_b) in enumerate(pairs):
        rows.append(
            {
                "pair_mode": mode,
                "pair_index": int(pair_index),
                "day_a": int(day_a),
                "day_b": int(day_b),
                "day_pair_label": f"D{day_a + 1}-D{day_b + 1}",
            }
        )
    return rows

def resolve_interday_condition_parts(pv_pair):
    """Return left/right condition keys for supported 56-bin interday pairs."""
    pair_key = _normalize_pv_pair(pv_pair)
    if pair_key == "CL_CR":
        return {"left": "CL", "right": "CR"}
    return {"left": "PL", "right": "PR"}

def compute_interday_pv_stability(
    snarm_all,
    condition_parts,
    day_pairs,
    neuron_mask=None,
    selected_neuron_count=50,
    n_subsample_iter=100,
    seed=2026,
    method="spearman",
    bin_indices=None,
    valid_day_mask=None,
):
    """Compute same-condition interday binwise PV stability for one subject."""
    condition_parts = dict(condition_parts)
    if set(condition_parts) != {"left", "right"}:
        raise ValueError("condition_parts must contain 'left' and 'right'")
    day_pairs = list(day_pairs)
    if not day_pairs:
        return _empty_interday_pv_result(condition_parts, 0, 0)

    first_key = condition_parts["left"]
    if first_key not in snarm_all:
        raise ValueError(f"snarm_all is missing condition {first_key!r}")
    first_map = np.asarray(snarm_all[first_key], dtype=float)
    if first_map.ndim != 3:
        raise ValueError(f"snarm_all[{first_key!r}] must have shape (day, neuron, bin)")
    num_day, num_neuron, num_bin = first_map.shape
    if bin_indices is None:
        bin_indices = np.arange(num_bin, dtype=int)
    else:
        bin_indices = np.asarray(bin_indices, dtype=int).ravel()
    if np.any(bin_indices < 0) or np.any(bin_indices >= num_bin):
        raise ValueError(f"bin_indices must index bins 0..{num_bin - 1}")

    neuron_mask = _align_neuron_mask_to_count(neuron_mask, num_neuron)
    valid_day_mask = _align_valid_day_mask(valid_day_mask, num_day)
    result = _empty_interday_pv_result(condition_parts, len(day_pairs), bin_indices.size)
    result["n_neuron"] = int(np.sum(neuron_mask))

    for pair_idx, pair in enumerate(day_pairs):
        day_a = int(pair["day_a"])
        day_b = int(pair["day_b"])
        if (
            day_a < 0
            or day_b < 0
            or day_a >= num_day
            or day_b >= num_day
            or not valid_day_mask[day_a]
            or not valid_day_mask[day_b]
        ):
            continue
        for part, condition_key in condition_parts.items():
            if condition_key not in snarm_all:
                continue
            condition_maps = np.asarray(snarm_all[condition_key], dtype=float)
            if condition_maps.ndim != 3 or condition_maps.shape[0] <= max(day_a, day_b):
                continue
            if condition_maps.shape[1] != num_neuron:
                raise ValueError("all interday condition maps must share neuron count")
            map_a = condition_maps[day_a][neuron_mask, :][:, bin_indices]
            map_b = condition_maps[day_b][neuron_mask, :][:, bin_indices]
            result[part][pair_idx, :] = _interday_binwise_pv_corr(
                map_a,
                map_b,
                selected_neuron_count=selected_neuron_count,
                n_subsample_iter=n_subsample_iter,
                seed=int(seed) + pair_idx * 10_000 + _condition_seed_offset(condition_key),
                method=method,
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result["averaged"] = np.nanmean(np.stack([result["left"], result["right"]]), axis=0)
    return result

def compute_interday_pc_long_table(
    snarm_all,
    condition_parts,
    day_pairs,
    group,
    mouse,
    bin_indices=None,
    valid_day_mask=None,
    method="pearson",
):
    """Return all-neuron row interday PC spatial correlations for one subject."""
    import pandas as pd

    condition_parts = dict(condition_parts)
    if set(condition_parts) != {"left", "right"}:
        raise ValueError("condition_parts must contain 'left' and 'right'")
    day_pairs = list(day_pairs)
    if not day_pairs:
        return pd.DataFrame(columns=INTERDAY_PC_COLUMNS)

    available_key = next((key for key in condition_parts.values() if key in snarm_all), None)
    if available_key is None:
        return pd.DataFrame(columns=INTERDAY_PC_COLUMNS)
    available_map = np.asarray(snarm_all[available_key], dtype=float)
    if available_map.ndim != 3:
        raise ValueError(f"snarm_all[{available_key!r}] must have shape (day, neuron, bin)")
    num_day, num_neuron, num_bin = available_map.shape
    if bin_indices is None:
        bin_indices = np.arange(num_bin, dtype=int)
    else:
        bin_indices = np.asarray(bin_indices, dtype=int).ravel()
    if np.any(bin_indices < 0) or np.any(bin_indices >= num_bin):
        raise ValueError(f"bin_indices must index bins 0..{num_bin - 1}")
    valid_day_mask = _align_valid_day_mask(valid_day_mask, num_day)

    rows = []
    condition_values_by_pair = {}
    for pair in day_pairs:
        day_a = int(pair["day_a"])
        day_b = int(pair["day_b"])
        if (
            day_a < 0
            or day_b < 0
            or day_a >= num_day
            or day_b >= num_day
            or not valid_day_mask[day_a]
            or not valid_day_mask[day_b]
        ):
            continue

        per_part_values = {}
        for part, condition_key in condition_parts.items():
            if condition_key not in snarm_all:
                continue
            condition_maps = np.asarray(snarm_all[condition_key], dtype=float)
            if condition_maps.ndim != 3 or condition_maps.shape[0] <= max(day_a, day_b):
                continue
            if condition_maps.shape[1] != num_neuron:
                raise ValueError("all interday condition maps must share neuron count")
            map_a = condition_maps[day_a, :, :][:, bin_indices]
            map_b = condition_maps[day_b, :, :][:, bin_indices]
            values = np.asarray(
                [
                    pc_corr(map_a[neuron_idx], map_b[neuron_idx], method=method)
                    for neuron_idx in range(num_neuron)
                ],
                dtype=float,
            )
            per_part_values[part] = values
            rows.extend(
                _interday_pc_rows(
                    values,
                    pair,
                    group=group,
                    mouse=mouse,
                    condition=part,
                    condition_key=condition_key,
                )
            )

        if "left" in per_part_values and "right" in per_part_values:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                averaged = np.nanmean(
                    np.vstack([per_part_values["left"], per_part_values["right"]]),
                    axis=0,
                )
            condition_values_by_pair[(pair["pair_mode"], pair["pair_index"])] = averaged
            rows.extend(
                _interday_pc_rows(
                    averaged,
                    pair,
                    group=group,
                    mouse=mouse,
                    condition="averaged",
                    condition_key="averaged",
                )
            )

    return pd.DataFrame(rows, columns=INTERDAY_PC_COLUMNS)

def trial_pair_pv_corr_by_gap(
    snarm_day,
    labels_day,
    neuron_mask,
    pv_pair="CL_CR",
    bin_indices=None,
    is_context_trial=None,
    segment_ids=None,
    selected_neuron_count=50,
    n_subsample_iter=100,
    seed=2026,
    method="spearman",
):
    """Return same-condition trial-pair PV correlations labeled by trial-index gap.

    `snarm_day` is expected to use the same trial alignment produced by
    `extract_subject_spatial_maps`: row 0 corresponds to `labels_day[:, 1]`.
    `pv_pair="CL_CR"` compares current correct-left and correct-right trials
    within condition. `pv_pair="PL_PR"` compares previous trials grouped by the
    following current correct side, matching notebook 11's PL/PR convention.
    """
    import pandas as pd

    snarm_day = np.asarray(snarm_day, dtype=float)
    labels = np.asarray(labels_day)
    if snarm_day.ndim != 3:
        raise ValueError(f"snarm_day must have shape (trial, neuron, bin), got {snarm_day.shape}")
    if labels.ndim != 2 or labels.shape[0] < 3:
        raise ValueError(f"labels_day must have shape (3, n_trials), got {labels.shape}")

    n_trial, n_neuron, n_bin = snarm_day.shape
    neuron_mask = _align_neuron_mask_to_count(neuron_mask, n_neuron)
    if bin_indices is None:
        bin_indices = np.arange(n_bin, dtype=int)
    else:
        bin_indices = np.asarray(bin_indices, dtype=int).ravel()
    if np.any(bin_indices < 0) or np.any(bin_indices >= n_bin):
        raise ValueError(f"bin_indices must index snarm_day bins 0..{n_bin - 1}")

    pair_key = _normalize_pv_pair(pv_pair)
    condition_indices = _condition_trial_indices(
        labels,
        n_trial,
        pair_key,
        is_context_trial=is_context_trial,
        segment_ids=segment_ids,
    )
    if not condition_indices:
        return pd.DataFrame(columns=PV_GAP_COLUMNS)

    result_frames = []
    selected_count_value = "all" if selected_neuron_count is None else int(selected_neuron_count)
    n_selected = int(np.sum(neuron_mask))
    for condition, trial_indices in condition_indices.items():
        trial_indices = np.asarray(trial_indices, dtype=int)
        if trial_indices.size < 2:
            continue

        pair_i, pair_j = np.triu_indices(trial_indices.size, k=1)
        trial_i = trial_indices[pair_i]
        trial_j = trial_indices[pair_j]
        prev_maps = snarm_day[trial_i][:, neuron_mask, :][:, :, bin_indices]
        curr_maps = snarm_day[trial_j][:, neuron_mask, :][:, :, bin_indices]

        pv_corr = _trial_pair_binwise_pv_corr(
            prev_maps,
            curr_maps,
            selected_neuron_count=selected_neuron_count,
            n_subsample_iter=n_subsample_iter,
            seed=int(seed) + _condition_seed_offset(condition),
            method=method,
        )
        n_pair, n_bin_selected = pv_corr.shape
        result_frames.append(
            pd.DataFrame(
                {
                    "pv_pair": pair_key,
                    "condition": condition,
                    "trial_i": np.repeat(trial_i, n_bin_selected),
                    "trial_j": np.repeat(trial_j, n_bin_selected),
                    "trial_gap": np.repeat(trial_j - trial_i, n_bin_selected),
                    "bin": np.tile(bin_indices, n_pair),
                    "pv_corr": pv_corr.reshape(-1),
                    "n_neuron": n_selected,
                    "selected_neuron_count": selected_count_value,
                }
            )
        )

    if not result_frames:
        return pd.DataFrame(columns=PV_GAP_COLUMNS)
    return pd.concat(result_frames, ignore_index=True)

def _empty_interday_pv_result(condition_parts, n_pair, n_bin):
    return {
        "left": np.full((int(n_pair), int(n_bin)), np.nan, dtype=float),
        "right": np.full((int(n_pair), int(n_bin)), np.nan, dtype=float),
        "averaged": np.full((int(n_pair), int(n_bin)), np.nan, dtype=float),
        "condition_keys": np.array(dict(condition_parts), dtype=object),
        "n_neuron": 0,
    }

def _align_valid_day_mask(valid_day_mask, num_day):
    if valid_day_mask is None:
        return np.ones(int(num_day), dtype=bool)
    mask = np.asarray(valid_day_mask, dtype=bool).ravel()
    aligned = np.zeros(int(num_day), dtype=bool)
    n_copy = min(mask.size, int(num_day))
    aligned[:n_copy] = mask[:n_copy]
    return aligned

def _normalize_pv_pair(pv_pair):
    key = str(pv_pair).upper().replace("-", "_").replace("/", "_").replace(" ", "_")
    if key not in {"CL_CR", "PL_PR"}:
        raise ValueError("pv_pair must be 'CL_CR' or 'PL_PR'")
    return key

def _align_neuron_mask_to_count(neuron_mask, n_neuron):
    if neuron_mask is None:
        return np.ones(int(n_neuron), dtype=bool)
    mask = np.asarray(neuron_mask, dtype=bool).ravel()
    aligned = np.zeros(int(n_neuron), dtype=bool)
    n_copy = min(mask.size, int(n_neuron))
    aligned[:n_copy] = mask[:n_copy]
    return aligned

def _align_optional_trial_metadata(values, n_aligned, dtype=None):
    if values is None:
        return None
    values = np.asarray(values, dtype=dtype).ravel()
    if values.shape[0] >= n_aligned + 1:
        return values[1 : n_aligned + 1]
    if values.shape[0] == n_aligned:
        return values
    raise ValueError("trial metadata must align with labels_day trial columns")

def _condition_trial_indices(
    labels_day,
    n_trial,
    pv_pair,
    is_context_trial=None,
    segment_ids=None,
):
    labels_aligned = np.asarray(labels_day)[:, 1:]
    n_aligned = min(int(n_trial), labels_aligned.shape[1])
    labels_aligned = labels_aligned[:, :n_aligned]
    valid = valid_labeled_trials(labels_aligned)
    if is_context_trial is not None:
        context = _align_optional_trial_metadata(is_context_trial, n_aligned, dtype=bool)
        valid &= ~context

    actual_left = labels_aligned[0].astype(bool)
    correct = labels_aligned[2].astype(bool)
    if pv_pair == "CL_CR":
        return {
            "CL": np.flatnonzero(valid & actual_left & correct),
            "CR": np.flatnonzero(valid & ~actual_left & correct),
        }

    prev_idx, curr_idx = eligible_previous_current_indices(
        labels_day,
        n_aligned,
        segment_ids=segment_ids,
        is_context_trial=is_context_trial,
    )
    if prev_idx.size == 0:
        return {"PL": np.array([], dtype=int), "PR": np.array([], dtype=int)}

    current_left = labels_aligned[0, curr_idx].astype(bool)
    previous_left = labels_aligned[0, prev_idx].astype(bool)
    return {
        "PL": prev_idx[current_left & ~previous_left],
        "PR": prev_idx[~current_left & previous_left],
    }

def _condition_seed_offset(condition):
    return sum((idx + 1) * ord(char) for idx, char in enumerate(str(condition)))

def _interday_binwise_pv_corr(
    map_a,
    map_b,
    selected_neuron_count,
    n_subsample_iter,
    seed,
    method,
):
    map_a = np.asarray(map_a, dtype=float)
    map_b = np.asarray(map_b, dtype=float)
    if map_a.shape != map_b.shape:
        raise ValueError(f"Input shapes must match, got {map_a.shape} and {map_b.shape}")
    if map_a.ndim != 2:
        raise ValueError(f"Expected arrays with shape (neuron, bin), got {map_a.shape}")

    if selected_neuron_count is None:
        return rowwise_corr(map_a.T, map_b.T, method=method)

    return subsampled_pv_corr(
        map_a,
        map_b,
        int(selected_neuron_count),
        n_iter=int(n_subsample_iter),
        seed=int(seed),
        method=method,
    )

def _interday_pc_rows(values, pair, group, mouse, condition, condition_key):
    values = np.asarray(values, dtype=float).ravel()
    return [
        {
            "group": group,
            "mouse": mouse,
            "pair_mode": pair["pair_mode"],
            "pair_index": int(pair["pair_index"]),
            "day_a": int(pair["day_a"]) + 1,
            "day_b": int(pair["day_b"]) + 1,
            "day_pair_label": pair["day_pair_label"],
            "condition": condition,
            "condition_key": condition_key,
            "neuron": int(neuron_idx),
            "pc_interday_corr": float(value),
        }
        for neuron_idx, value in enumerate(values)
    ]

def _trial_pair_binwise_pv_corr(
    prev_maps,
    curr_maps,
    selected_neuron_count,
    n_subsample_iter,
    seed,
    method,
):
    prev_maps = np.asarray(prev_maps, dtype=float)
    curr_maps = np.asarray(curr_maps, dtype=float)
    if prev_maps.shape != curr_maps.shape:
        raise ValueError(f"Input shapes must match, got {prev_maps.shape} and {curr_maps.shape}")
    if prev_maps.ndim != 3:
        raise ValueError(f"Expected arrays with shape (pair, neuron, bin), got {prev_maps.shape}")

    if selected_neuron_count is None:
        n_pair, n_neuron, n_bin = prev_maps.shape
        corr = rowwise_corr(
            prev_maps.transpose(0, 2, 1).reshape(n_pair * n_bin, n_neuron),
            curr_maps.transpose(0, 2, 1).reshape(n_pair * n_bin, n_neuron),
            method=method,
        )
        return corr.reshape(n_pair, n_bin)

    return subsampled_pair_pv_corr(
        prev_maps,
        curr_maps,
        int(selected_neuron_count),
        n_iter=int(n_subsample_iter),
        seed=int(seed),
        method=method,
    )
