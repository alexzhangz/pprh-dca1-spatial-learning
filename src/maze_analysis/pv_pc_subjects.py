"""Manuscript definitions collected from pv_pc_subjects.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import warnings

from collections import defaultdict

from pathlib import Path

import numpy as np

import pandas as pd

from .correlation import pc_corr, subsampled_pair_pv_corr

from .pv_shuffle import _align_neuron_mask, _compute_iterated_pv, resolve_pv_pair_config

from .spatial import compute_condition_odd_even_maps

from .stability import (
    compute_interday_pc_long_table,
    compute_interday_pv_stability,
    interday_day_pairs,
    resolve_interday_condition_parts,
    trial_pair_pv_corr_by_gap,
    trial_spatial_corr,
)

from .trials import (
    align_neuron_mask,
    eligible_previous_current_indices,
    should_skip_mouse_day,
    shuffle_previous_indices_within_current_side,
)

from .subjects import GROUP_ORDER, PV_PC_GROUP_MICE

def load_optional_array(load_path, candidates):
    load_path = Path(load_path)
    for filename in candidates:
        path = load_path / filename
        if path.exists():
            return np.load(path, allow_pickle=True)
    return None

def day_metadata(metadata, day_idx):
    if metadata is None or day_idx >= len(metadata):
        return None
    return metadata[day_idx]

def compute_pv_within_reliability_subject(load_path, mouse, group, config):
    """Compute condition-specific odd/even PV reliability for one subject."""
    if config.get("split_strategy") != "condition_ordinal_odd_even_v1":
        raise ValueError("analysis 14 requires condition_ordinal_odd_even_v1 splitting")
    if config.get("pv_method", "spearman") != "spearman":
        raise ValueError("analysis 14 currently supports Spearman PV correlation only")
    load_path = Path(load_path)
    xy_all = np.load(load_path / "xy_trial_all.npy", allow_pickle=True)
    dff_all = np.load(load_path / "d_dff_trial_all.npy", allow_pickle=True)
    labels_all = np.load(load_path / "trial_labels.npy", allow_pickle=True)
    segment_ids_all = load_optional_array(load_path, ["trial_segment_ids.npy", "trial_segment_ids_auto.npy"])
    context_all = load_optional_array(load_path, ["is_context_trial.npy", "is_context_trial_auto.npy"])
    pv_config = resolve_pv_pair_config(config["pv_pair"], config["pv_neuron_mode"])
    selected_trials = pv_config["selected_trials"]
    left_key, right_key = selected_trials
    num_day = int(config["num_day"])
    num_bin = 56
    target = int(pv_config["target_neuron_count"])
    payload = {
        "left": np.full((num_day, num_bin), np.nan),
        "right": np.full((num_day, num_bin), np.nan),
        "averaged": np.full((num_day, num_bin), np.nan),
        "candidate_neuron_count": np.full(num_day, np.nan),
        "valid_day_left": np.zeros(num_day, dtype=bool),
        "valid_day_right": np.zeros(num_day, dtype=bool),
        "valid_day_averaged": np.zeros(num_day, dtype=bool),
        "trial_count_left": np.zeros(num_day, dtype=int),
        "trial_count_right": np.zeros(num_day, dtype=int),
        "odd_trial_count_left": np.zeros(num_day, dtype=int),
        "odd_trial_count_right": np.zeros(num_day, dtype=int),
        "even_trial_count_left": np.zeros(num_day, dtype=int),
        "even_trial_count_right": np.zeros(num_day, dtype=int),
        "split_strategy": "condition_ordinal_odd_even_v1",
        "pv_method": "spearman",
    }
    pc_index = (
        np.load(load_path / "pc_index_p01_z2.npy", allow_pickle=True)
        if pv_config["neuron_selection"] == "pc"
        else None
    )
    for day_idx in range(num_day):
        if should_skip_mouse_day(day_idx, mouse):
            continue
        if day_idx >= len(xy_all) or day_idx >= len(dff_all) or day_idx >= len(labels_all):
            continue
        group_idx, mouse_idx = _cohort_indices(group, mouse)
        base_seed = 2026 + day_idx * 1000 + mouse_idx * 17 + group_idx * 10000
        split_maps = compute_condition_odd_even_maps(
            xy_all[day_idx],
            dff_all[day_idx],
            labels_all[day_idx],
            condition_keys=selected_trials,
            trial_segment_ids=day_metadata(segment_ids_all, day_idx),
            is_context_trial=day_metadata(context_all, day_idx),
            distance_threshold=float(config["distance_threshold"]),
            smooth=bool(config["smooth"]),
            transition_penalty=float(config["transition_penalty"]),
            min_split_trials=int(config["min_split_trials"]),
        )
        if left_key not in split_maps or right_key not in split_maps:
            continue
        for part, condition_key in (("left", left_key), ("right", right_key)):
            payload[f"trial_count_{part}"][day_idx] = split_maps[condition_key]["trial_count"]
            payload[f"odd_trial_count_{part}"][day_idx] = split_maps[condition_key]["odd_trial_count"]
            payload[f"even_trial_count_{part}"][day_idx] = split_maps[condition_key]["even_trial_count"]
        n_neuron = split_maps[left_key]["odd_trials"].shape[0]
        if pc_index is None:
            mask = np.ones(n_neuron, dtype=bool)
        elif day_idx < pc_index.shape[0]:
            mask = _align_neuron_mask(pc_index[day_idx], n_neuron)
        else:
            mask = np.zeros(n_neuron, dtype=bool)
        payload["candidate_neuron_count"][day_idx] = int(np.sum(mask))
        if np.sum(mask) < target:
            continue
        left = _condition_odd_even_summary(
            split_maps[left_key], mask, target, int(config["n_subsample_iter"]), 1000000 + base_seed
        )
        right = _condition_odd_even_summary(
            split_maps[right_key], mask, target, int(config["n_subsample_iter"]), 1010000 + base_seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            averaged = np.nanmean(np.stack([left, right]), axis=0)
        for part, values in (("left", left), ("right", right), ("averaged", averaged)):
            payload[part][day_idx] = values
            payload[f"valid_day_{part}"][day_idx] = np.isfinite(values).any()
    return payload

def compute_intraday_pc_within_subject(load_path, mouse, group, config):
    """Compute row-level condition-specific odd/even spatial correlations."""
    if config.get("split_strategy") != "condition_ordinal_odd_even_v1":
        raise ValueError("analysis 16 requires condition_ordinal_odd_even_v1 splitting")
    load_path = Path(load_path)
    xy_all = np.load(load_path / "xy_trial_all.npy", allow_pickle=True)
    dff_all = np.load(load_path / "d_dff_trial_all.npy", allow_pickle=True)
    labels_all = np.load(load_path / "trial_labels.npy", allow_pickle=True)
    segment_ids_all = load_optional_array(load_path, ["trial_segment_ids.npy", "trial_segment_ids_auto.npy"])
    context_all = load_optional_array(load_path, ["is_context_trial.npy", "is_context_trial_auto.npy"])
    frames = []
    qc_rows = []
    map_store = defaultdict(lambda: {"odd": [], "even": [], "mouse": [], "neuron": []})
    condition_keys = tuple(config["condition_keys"])
    for day_idx in range(int(config["num_day"])):
        if should_skip_mouse_day(day_idx, mouse):
            continue
        if day_idx >= len(xy_all) or day_idx >= len(dff_all) or day_idx >= len(labels_all):
            continue
        split_maps = compute_condition_odd_even_maps(
            xy_all[day_idx],
            dff_all[day_idx],
            labels_all[day_idx],
            condition_keys=condition_keys,
            trial_segment_ids=day_metadata(segment_ids_all, day_idx),
            is_context_trial=day_metadata(context_all, day_idx),
            distance_threshold=float(config["distance_threshold"]),
            smooth=bool(config["smooth"]),
            transition_penalty=float(config["transition_penalty"]),
            min_split_trials=int(config["min_split_trials"]),
        )
        for condition in condition_keys:
            odd = np.asarray(split_maps[condition]["odd_trials"], dtype=float)[:, :56]
            even = np.asarray(split_maps[condition]["even_trials"], dtype=float)[:, :56]
            values = np.asarray([pc_corr(odd[idx], even[idx], method=config["pc_method"]) for idx in range(odd.shape[0])])
            neuron_ids = np.arange(odd.shape[0], dtype=int)
            frames.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "mouse": mouse,
                        "day": day_idx + 1,
                        "day_centered": float((day_idx + 1) - ((1 + int(config["num_day"])) / 2.0)),
                        "condition": condition,
                        "neuron": neuron_ids,
                        "pc_within_corr": values,
                    }
                )
            )
            keep = np.isfinite(odd).any(axis=1) | np.isfinite(even).any(axis=1)
            if np.any(keep):
                entry = map_store[(day_idx + 1, condition)]
                entry["odd"].append(odd[keep])
                entry["even"].append(even[keep])
                entry["mouse"].extend([mouse] * int(np.sum(keep)))
                entry["neuron"].extend(neuron_ids[keep].tolist())
            qc_rows.append(
                {
                    "group": group,
                    "mouse": mouse,
                    "day": day_idx + 1,
                    "condition": condition,
                    "n_neuron": int(odd.shape[0]),
                    "trial_count": int(split_maps[condition]["trial_count"]),
                    "odd_trial_count": int(split_maps[condition]["odd_trial_count"]),
                    "even_trial_count": int(split_maps[condition]["even_trial_count"]),
                    "finite_pc_corr_count": int(np.isfinite(values).sum()),
                }
            )
    long_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not long_table.empty:
        averaged = (
            long_table.pivot_table(
                index=["group", "mouse", "day", "day_centered", "neuron"],
                columns="condition",
                values="pc_within_corr",
                aggfunc="mean",
                observed=True,
            )
            .mean(axis=1, skipna=True)
            .reset_index(name="pc_within_corr")
        )
        averaged["condition"] = "averaged"
        long_table = pd.concat([long_table, averaged[long_table.columns]], ignore_index=True)
    return {
        "long_table": long_table,
        "qc_table": pd.DataFrame(qc_rows),
        "map_store": dict(map_store),
        "split_strategy": "condition_ordinal_odd_even_v1",
    }

def compute_interday_stability_subject(load_path, mouse, group, config):
    """Compute notebook-17 adjacent/anchor interday PV and PC metrics."""
    load_path = Path(load_path)
    snarm_all = np.load(load_path / "snarm_all.npy", allow_pickle=True).item()
    pv_config = resolve_pv_pair_config(config["pv_pair"], config["pv_neuron_mode"])
    condition_parts = resolve_interday_condition_parts(config["pv_pair"])
    first_key = next(key for key in condition_parts.values() if key in snarm_all)
    n_day, n_neuron, _ = np.asarray(snarm_all[first_key]).shape
    valid_days = np.array([not should_skip_mouse_day(idx, mouse) for idx in range(n_day)], dtype=bool)
    payload = {"pv_arrays": {}, "candidate_neuron_count": {}, "pc_long": {}}
    for mode in config["pair_modes"]:
        pairs = interday_day_pairs(int(config["num_day"]), mode)
        pc_table = compute_interday_pc_long_table(
            snarm_all,
            condition_parts,
            pairs,
            group=group,
            mouse=mouse,
            bin_indices=np.arange(56),
            valid_day_mask=valid_days,
            method=config["pc_method"],
        )
        candidate = np.full(len(pairs), np.nan)
        arrays = {part: np.full((len(pairs), 56), np.nan) for part in ("left", "right", "averaged")}
        for pair_idx, pair in enumerate(pairs):
            mask = _interday_neuron_mask(load_path, pair["day_a"], pair["day_b"], n_neuron, pv_config)
            candidate[pair_idx] = int(np.sum(mask))
            result = compute_interday_pv_stability(
                snarm_all,
                condition_parts,
                [pair],
                neuron_mask=mask,
                selected_neuron_count=pv_config["target_neuron_count"],
                n_subsample_iter=int(config["n_subsample_iter"]),
                seed=int(config["random_seed"]) + _cohort_indices(group, mouse)[1] * 1000 + pair_idx * 100 + len(group),
                method=config["pv_method"],
                bin_indices=np.arange(56),
                valid_day_mask=valid_days,
            )
            for part in arrays:
                arrays[part][pair_idx] = result[part][0]
        payload["pv_arrays"][mode] = arrays
        payload["candidate_neuron_count"][mode] = candidate
        payload["pc_long"][mode] = pc_table
    return payload

def _condition_odd_even_summary(split, mask, target, n_iter, seed):
    values = _compute_iterated_pv(
        np.asarray(split["odd_trials"], dtype=float)[None, :, :],
        np.asarray(split["even_trials"], dtype=float)[None, :, :],
        selected_index=mask,
        target_n=target,
        n_subsample_iter=n_iter,
        seed_base=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=0)

def _interday_neuron_mask(load_path, day_a, day_b, n_neuron, pv_config):
    if pv_config["neuron_selection"] == "all":
        return np.ones(n_neuron, dtype=bool)
    pc_path = Path(load_path) / "pc_index_p01_z2.npy"
    if not pc_path.exists():
        return np.zeros(n_neuron, dtype=bool)
    pc_index = np.asarray(np.load(pc_path, allow_pickle=True), dtype=bool)
    if pc_index.ndim != 2 or pc_index.shape[0] <= max(day_a, day_b):
        return np.zeros(n_neuron, dtype=bool)
    return align_neuron_mask(pc_index[day_a] | pc_index[day_b], n_neuron)

def _cohort_indices(group, mouse):
    return list(GROUP_ORDER).index(group), list(PV_PC_GROUP_MICE[group]).index(mouse)
