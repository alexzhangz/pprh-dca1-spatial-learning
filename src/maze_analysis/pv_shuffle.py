"""Manuscript definitions collected from pv_shuffle.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import os

import warnings

import numpy as np

from .correlation import subsampled_pv_corr

from .spatial import (
    compute_condition_shuffle_maps,
    compute_condition_split_half_maps,
)

from .trials import should_skip_mouse_day

PV_PAIR_CONFIGS = {
    "PL_PR": {
        "selected_trials": ("PL", "PR"),
        "trial_pair_label": "PL vs PR",
        "file_prefix": "pv_pl_pr",
    },
    "CL_CR": {
        "selected_trials": ("CL", "CR"),
        "trial_pair_label": "CL vs CR",
        "file_prefix": "pv_cl_cr",
    },
}

PV_NEURON_MODE_CONFIGS = {
    "pc50": {
        "neuron_mode": "pc50",
        "neuron_selection": "pc",
        "target_neuron_count": 50,
        "cache_suffix": "50pc",
        "stats_suffix": "50pc",
    },
    "all50": {
        "neuron_mode": "all50",
        "neuron_selection": "all",
        "target_neuron_count": 50,
        "cache_suffix": "50all",
        "stats_suffix": "50all",
    },
    "all100": {
        "neuron_mode": "all100",
        "neuron_selection": "all",
        "target_neuron_count": 100,
        "cache_suffix": "100all",
        "stats_suffix": "100all",
    },
}

PV_SHUFFLE_CACHE_METRICS = [
    "pv_real",
    "pv_shuffle_mean",
    "pv_shuffle_std",
    "pv_shuffle_delta",
    "pv_shuffle_z",
    "pv_within_reliability",
]

PV_SHUFFLE_RESULT_METRICS = [
    *PV_SHUFFLE_CACHE_METRICS,
    "pv_separation_z",
]

def _pv_cache_metric(cache, metric):
    """Return cached PV metric values, including derived compatibility metrics."""
    if metric == "pv_separation_z":
        if "pv_separation_z" in cache:
            return cache["pv_separation_z"]
        return -np.asarray(cache["pv_shuffle_z"], dtype=float)
    return cache[metric]

def build_pv_bin_inference_table(
    metric_data,
    *,
    mouse_names,
    groups,
    spatial_bins,
    metric,
    skip_mouse_day=None,
    first_day=1,
    last_day=None,
):
    """Return one PV metric observation per mouse-day-spatial bin."""
    import pandas as pd

    spatial_bins = np.asarray(spatial_bins, dtype=int).ravel()
    if spatial_bins.size == 0:
        raise ValueError("spatial_bins must not be empty")
    rows = []
    for group in groups:
        if group not in metric_data or group not in mouse_names:
            raise ValueError(f"metric_data and mouse_names must contain group {group!r}")
        values = np.asarray(metric_data[group], dtype=float)
        if values.ndim != 3:
            raise ValueError(
                f"metric_data[{group!r}] must have shape (day, mouse, bin), got {values.shape}"
            )
        mice = list(mouse_names[group])
        if values.shape[1] != len(mice):
            raise ValueError(
                f"mouse axis mismatch for {group!r}: expected {len(mice)}, got {values.shape[1]}"
            )
        if values.shape[2] != spatial_bins.size:
            raise ValueError(
                f"bin axis mismatch for {group!r}: expected {spatial_bins.size}, got {values.shape[2]}"
            )
        group_last_day = int(first_day) + values.shape[0] - 1
        if last_day is not None and group_last_day != int(last_day):
            raise ValueError(
                f"day axis mismatch for {group!r}: expected {int(last_day) - int(first_day) + 1}, "
                f"got {values.shape[0]}"
            )
        center = (float(first_day) + float(group_last_day)) / 2.0
        for day_idx in range(values.shape[0]):
            day = int(first_day) + day_idx
            for mouse_idx, mouse in enumerate(mice):
                if skip_mouse_day is not None and skip_mouse_day(day_idx, mouse):
                    continue
                for local_bin, spatial_bin in enumerate(spatial_bins):
                    value = float(values[day_idx, mouse_idx, local_bin])
                    if np.isfinite(value):
                        rows.append(
                            {
                                "group": group,
                                "mouse": mouse,
                                "day": day,
                                "day_centered": day - center,
                                "bin": int(spatial_bin),
                                metric: value,
                            }
                        )
    columns = ["group", "mouse", "day", "day_centered", "bin", metric]
    return pd.DataFrame(rows, columns=columns)

def build_pv_region_inference_table(
    metric_data,
    *,
    mouse_names,
    groups,
    sections,
    metric,
    skip_mouse_day=None,
    first_day=1,
    last_day=None,
):
    """Return one PV metric observation per mouse-day-spatial section."""
    import pandas as pd

    rows = []
    for group in groups:
        if group not in metric_data or group not in mouse_names:
            raise ValueError(f"metric_data and mouse_names must contain group {group!r}")
        values = np.asarray(metric_data[group], dtype=float)
        if values.ndim != 3:
            raise ValueError(
                f"metric_data[{group!r}] must have shape (day, mouse, bin), got {values.shape}"
            )
        mice = list(mouse_names[group])
        if values.shape[1] != len(mice):
            raise ValueError(
                f"mouse axis mismatch for {group!r}: expected {len(mice)}, got {values.shape[1]}"
            )
        group_last_day = int(first_day) + values.shape[0] - 1
        if last_day is not None and group_last_day != int(last_day):
            raise ValueError(
                f"day axis mismatch for {group!r}: expected {int(last_day) - int(first_day) + 1}, "
                f"got {values.shape[0]}"
            )
        center = (float(first_day) + float(group_last_day)) / 2.0
        for section_idx, (section, bins) in enumerate(sections.items()):
            bins = np.asarray(bins, dtype=int).ravel()
            if bins.size == 0:
                raise ValueError(f"section {section!r} has no bins")
            if np.any(bins < 0) or np.any(bins >= values.shape[2]):
                raise ValueError(
                    f"section {section!r} contains bins outside 0-{values.shape[2] - 1}"
                )
            for day_idx in range(values.shape[0]):
                day = int(first_day) + day_idx
                for mouse_idx, mouse in enumerate(mice):
                    if skip_mouse_day is not None and skip_mouse_day(day_idx, mouse):
                        continue
                    section_values = values[day_idx, mouse_idx, bins]
                    if np.isfinite(section_values).any():
                        rows.append(
                            {
                                "group": group,
                                "mouse": mouse,
                                "day": day,
                                "day_centered": day - center,
                                "section": section,
                                "section_idx": int(section_idx),
                                metric: float(np.nanmean(section_values)),
                            }
                        )
    columns = [
        "group",
        "mouse",
        "day",
        "day_centered",
        "section",
        "section_idx",
        metric,
    ]
    return pd.DataFrame(rows, columns=columns)

def resolve_pv_neuron_mode_config(pv_neuron_mode="pc50"):
    """Return canonical neuron sampling settings for a supported PV analysis mode."""
    key = str(pv_neuron_mode).lower().replace("-", "").replace("_", "")
    aliases = {
        "50pc": "pc50",
        "pc": "pc50",
        "pc50": "pc50",
        "50all": "all50",
        "all50": "all50",
        "100all": "all100",
        "all100": "all100",
    }
    key = aliases.get(key, key)
    if key not in PV_NEURON_MODE_CONFIGS:
        raise ValueError(
            f"Unsupported pv_neuron_mode {pv_neuron_mode!r}. Expected one of: "
            f"{', '.join(PV_NEURON_MODE_CONFIGS)}"
        )
    return dict(PV_NEURON_MODE_CONFIGS[key])

def resolve_pv_pair_config(pv_pair, pv_neuron_mode="pc50"):
    """Return canonical file/label settings for a supported PV pair and neuron mode."""
    key = str(pv_pair).upper().replace("-", "_").replace("/", "_").replace(" ", "_")
    if key == "BOTH":
        raise ValueError("'both' is only valid for the compute CLI, not one pair config")
    if key not in PV_PAIR_CONFIGS:
        raise ValueError(
            f"Unsupported pv_pair {pv_pair!r}. Expected one of: "
            f"{', '.join(PV_PAIR_CONFIGS)}"
        )
    config = dict(PV_PAIR_CONFIGS[key])
    neuron_config = resolve_pv_neuron_mode_config(pv_neuron_mode)
    config["pv_pair"] = key
    config["selected_trials"] = list(config["selected_trials"])
    config.update(neuron_config)
    config["cache_filename"] = (
        f"{config['file_prefix']}_shuffle_control_{config['cache_suffix']}.npy"
    )
    config["stats_prefix"] = f"{config['file_prefix']}_{config['stats_suffix']}"
    return config

def validate_pv_cache_pair(cache, selected_trials, cache_path=None):
    """Ensure a loaded cache was computed for the requested condition pair."""
    expected = list(selected_trials)
    observed = list(np.asarray(cache.get("selected_trials", []), dtype=object))
    if observed != expected:
        location = f" in {cache_path}" if cache_path is not None else ""
        raise ValueError(
            f"PV cache pair mismatch{location}: expected {expected}, found {observed}. "
            "Regenerate the cache with scripts/compute_pv_shuffle_control.py for this pair."
        )

def validate_pv_cache_config(
    cache,
    selected_trials,
    neuron_mode="pc50",
    target_neuron_count=50,
    cache_path=None,
):
    """Ensure a loaded cache matches both requested condition pair and neuron mode."""
    validate_pv_cache_pair(cache, selected_trials, cache_path=cache_path)
    expected_mode = resolve_pv_neuron_mode_config(neuron_mode)["neuron_mode"]
    observed_mode = cache.get("neuron_mode", None)
    if observed_mode is None:
        if expected_mode != "pc50":
            location = f" in {cache_path}" if cache_path is not None else ""
            raise ValueError(
                f"PV cache neuron_mode mismatch{location}: expected {expected_mode}, "
                "found legacy pc50 cache without neuron_mode metadata. Regenerate this cache."
            )
    elif str(observed_mode) != expected_mode:
        location = f" in {cache_path}" if cache_path is not None else ""
        raise ValueError(
            f"PV cache neuron_mode mismatch{location}: expected {expected_mode}, "
            f"found {observed_mode}."
        )

    observed_target = cache.get("target_neuron_count", cache.get("fixed_pc_target", None))
    if observed_target is None:
        return
    observed_target_array = np.asarray(observed_target, dtype=float).ravel()
    observed_target_scalar = observed_target_array[0] if observed_target_array.size else np.nan
    if np.isfinite(observed_target_scalar) and int(observed_target_scalar) != int(target_neuron_count):
        location = f" in {cache_path}" if cache_path is not None else ""
        raise ValueError(
            f"PV cache target_neuron_count mismatch{location}: expected "
            f"{target_neuron_count}, found {observed_target_scalar:g}."
        )

def load_pv_shuffle_results_from_cache(
    mouse_names,
    num_day,
    pv_pair,
    pv_neuron_mode="pc50",
    base_path="../8-maze/Aligned_data",
):
    """Load per-mouse PV shuffle cache files into group arrays."""
    config = resolve_pv_pair_config(pv_pair, pv_neuron_mode)
    selected_trials = config["selected_trials"]
    cache_filename = config["cache_filename"]
    metric_names = PV_SHUFFLE_RESULT_METRICS
    results = {
        metric: {
            group: np.full((num_day, len(mice), 56), np.nan)
            for group, mice in mouse_names.items()
        }
        for metric in metric_names
    }
    results["pc_count"] = {
        group: np.full((num_day, len(mice)), np.nan)
        for group, mice in mouse_names.items()
    }
    results["candidate_neuron_count"] = {
        group: np.full((num_day, len(mice)), np.nan)
        for group, mice in mouse_names.items()
    }
    results["target_neuron_count"] = {
        group: np.full((num_day, len(mice)), np.nan)
        for group, mice in mouse_names.items()
    }
    results["valid_day"] = {
        group: np.zeros((num_day, len(mice)), dtype=bool)
        for group, mice in mouse_names.items()
    }
    results["pair_config"] = config

    missing = []
    for group, mice in mouse_names.items():
        for mouse_idx, mouse_name in enumerate(mice):
            cache_path = os.path.join(base_path, mouse_name, cache_filename)
            if not os.path.exists(cache_path):
                missing.append(cache_path)
                continue
            cache = np.load(cache_path, allow_pickle=True).item()
            validate_pv_cache_config(
                cache,
                selected_trials,
                neuron_mode=config["neuron_mode"],
                target_neuron_count=config["target_neuron_count"],
                cache_path=cache_path,
            )
            for metric in metric_names:
                results[metric][group][:, mouse_idx, :] = _pv_cache_metric(cache, metric)
            results["pc_count"][group][:, mouse_idx] = cache.get("pc_count", np.nan)
            results["candidate_neuron_count"][group][:, mouse_idx] = cache.get(
                "candidate_neuron_count", cache.get("pc_count", np.nan)
            )
            target_count = cache.get("target_neuron_count", cache.get("fixed_pc_target", np.nan))
            target_count_array = np.asarray(target_count, dtype=float)
            if target_count_array.ndim == 0:
                results["target_neuron_count"][group][:, mouse_idx] = float(target_count_array)
            else:
                results["target_neuron_count"][group][:, mouse_idx] = target_count_array[:num_day]
            results["valid_day"][group][:, mouse_idx] = cache["valid_day"]
    if missing:
        shown = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {config['trial_pair_label']} PV shuffle cache files. "
            f"Run: conda run -n cebra python notebook-8-maze\\scripts\\compute_pv_shuffle_control.py "
            f"--pair {config['pv_pair']}\n{shown}"
        )
    return results

def compute_pv_shuffle_control_for_subject(
    load_path,
    mouse_name,
    group_idx,
    mouse_idx,
    num_day,
    selected_trials,
    selected_control,
    pc_target_by_day,
    n_subsample_iter,
    n_condition_shuffle_iter,
    n_within_condition_split_iter,
    fixed_pc_target,
    pv_neuron_mode="pc50",
):
    """Compute compact PV shuffle-control cache for one mouse."""
    snarm_path = os.path.join(load_path, "snarm_all.npy")
    pc_path = os.path.join(load_path, "pc_index_p01_z2.npy")
    neuron_config = resolve_pv_neuron_mode_config(pv_neuron_mode)
    needs_pc_index = neuron_config["neuron_selection"] == "pc"
    if not os.path.exists(snarm_path) or (needs_pc_index and not os.path.exists(pc_path)):
        return None

    snarm_all = np.load(snarm_path, allow_pickle=True).item()
    if not all(key in snarm_all for key in selected_trials):
        return None
    pc_index = np.load(pc_path, allow_pickle=True) if os.path.exists(pc_path) else None
    num_bin = 56
    subject_cache = {
        "mouse_name": mouse_name,
        "selected_trials": np.array(selected_trials, dtype=object),
        "trial_pair_label": f"{selected_trials[0]} vs {selected_trials[1]}",
        "neuron_mode": neuron_config["neuron_mode"],
        "neuron_selection": neuron_config["neuron_selection"],
        "target_neuron_count": int(fixed_pc_target),
        "loc_range": np.arange(num_bin),
        "fixed_pc_target": fixed_pc_target,
        "n_subsample_iter": n_subsample_iter,
        "n_condition_shuffle_iter": n_condition_shuffle_iter,
        "n_within_condition_split_iter": n_within_condition_split_iter,
        "pv_real": np.full((num_day, num_bin), np.nan),
        "pv_shuffle_mean": np.full((num_day, num_bin), np.nan),
        "pv_shuffle_std": np.full((num_day, num_bin), np.nan),
        "pv_shuffle_delta": np.full((num_day, num_bin), np.nan),
        "pv_shuffle_z": np.full((num_day, num_bin), np.nan),
        "pv_separation_z": np.full((num_day, num_bin), np.nan),
        "pv_within_reliability": np.full((num_day, num_bin), np.nan),
        "pc_count": np.full(num_day, np.nan),
        "candidate_neuron_count": np.full(num_day, np.nan),
        "valid_day": np.zeros(num_day, dtype=bool),
    }

    xy_path = os.path.join(load_path, "xy_trial_all.npy")
    dff_path = os.path.join(load_path, "d_dff_trial_all.npy")
    labels_path = os.path.join(load_path, "trial_labels.npy")
    can_compute_shuffle = all(os.path.exists(path) for path in [xy_path, dff_path, labels_path])
    if can_compute_shuffle:
        xy_all = np.load(xy_path, allow_pickle=True)
        dff_all = np.load(dff_path, allow_pickle=True)
        labels_all = np.load(labels_path, allow_pickle=True)
        segment_ids_all = _load_optional_array(
            load_path, ["trial_segment_ids.npy", "trial_segment_ids_auto.npy"]
        )
        context_all = _load_optional_array(
            load_path, ["is_context_trial.npy", "is_context_trial_auto.npy"]
        )
    else:
        xy_all = dff_all = labels_all = segment_ids_all = context_all = None

    for day_idx in range(num_day):
        if should_skip_mouse_day(day_idx, mouse_name):
            continue
        if day_idx >= snarm_all[selected_trials[0]].shape[0]:
            continue
        if needs_pc_index and day_idx >= pc_index.shape[0]:
            continue
        n_neuron = snarm_all[selected_trials[0]][day_idx].shape[0]
        if pc_index is not None and day_idx < pc_index.shape[0]:
            pc_mask = _align_neuron_mask(pc_index[day_idx], n_neuron)
            subject_cache["pc_count"][day_idx] = np.sum(pc_mask)
        else:
            pc_mask = np.zeros(n_neuron, dtype=bool)
        selected_index = pc_mask if needs_pc_index else np.ones(n_neuron, dtype=bool)
        subject_cache["candidate_neuron_count"][day_idx] = np.sum(selected_index)
        target_n = pc_target_by_day[day_idx]
        if not selected_control or not np.isfinite(target_n):
            continue

        snarm_a = snarm_all[selected_trials[0]][day_idx][selected_index, :]
        snarm_b = snarm_all[selected_trials[1]][day_idx][selected_index, :]
        if snarm_a.shape[0] < target_n or snarm_b.shape[0] < target_n:
            continue

        base_seed = 2026 + day_idx * 1000 + mouse_idx * 17 + group_idx * 10000
        real_pv = subsampled_pv_corr(
            snarm_a,
            snarm_b,
            target_n,
            n_iter=n_subsample_iter,
            seed=base_seed,
            method="spearman",
        )
        subject_cache["pv_real"][day_idx, :] = real_pv
        subject_cache["valid_day"][day_idx] = np.isfinite(real_pv).any()

        if not (
            can_compute_shuffle
            and day_idx < len(xy_all)
            and day_idx < len(dff_all)
            and day_idx < len(labels_all)
        ):
            continue

        shuffle_maps = compute_condition_shuffle_maps(
            xy_all[day_idx],
            dff_all[day_idx],
            labels_all[day_idx],
            condition_keys=selected_trials,
            trial_segment_ids=_day_metadata(segment_ids_all, day_idx),
            is_context_trial=_day_metadata(context_all, day_idx),
            n_shuffle=n_condition_shuffle_iter,
            seed=700000 + base_seed,
            distance_threshold=6,
            smooth=True,
            transition_penalty=0.25,
        )
        shuffle_mean, shuffle_std = _summarize_shuffle_maps(
            shuffle_maps,
            selected_trials=selected_trials,
            selected_index=selected_index,
            target_n=target_n,
            n_subsample_iter=n_subsample_iter,
            seed_base=800000 + base_seed,
        )
        subject_cache["pv_shuffle_mean"][day_idx, :] = shuffle_mean
        subject_cache["pv_shuffle_std"][day_idx, :] = shuffle_std
        delta = real_pv - shuffle_mean
        subject_cache["pv_shuffle_delta"][day_idx, :] = delta
        shuffle_z = np.divide(
            delta,
            shuffle_std,
            out=np.full_like(delta, np.nan, dtype=float),
            where=(shuffle_std > 0) & np.isfinite(shuffle_std),
        )
        subject_cache["pv_shuffle_z"][day_idx, :] = shuffle_z
        subject_cache["pv_separation_z"][day_idx, :] = -shuffle_z

        split_maps = compute_condition_split_half_maps(
            xy_all[day_idx],
            dff_all[day_idx],
            labels_all[day_idx],
            condition_keys=selected_trials,
            trial_segment_ids=_day_metadata(segment_ids_all, day_idx),
            is_context_trial=_day_metadata(context_all, day_idx),
            n_split=n_within_condition_split_iter,
            seed=900000 + base_seed,
            distance_threshold=6,
            smooth=True,
            transition_penalty=0.25,
        )
        subject_cache["pv_within_reliability"][day_idx, :] = _summarize_split_half_maps(
            split_maps,
            selected_trials=selected_trials,
            selected_index=selected_index,
            target_n=target_n,
            n_subsample_iter=n_subsample_iter,
            seed_base=1000000 + base_seed,
        )
    return subject_cache

def _summarize_shuffle_maps(
    shuffle_maps,
    selected_trials,
    selected_index,
    target_n,
    n_subsample_iter,
    seed_base,
):
    shuffle_iter = _compute_iterated_pv(
        shuffle_maps[selected_trials[0]],
        shuffle_maps[selected_trials[1]],
        selected_index=selected_index,
        target_n=target_n,
        n_subsample_iter=n_subsample_iter,
        seed_base=seed_base,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(shuffle_iter, axis=0), np.nanstd(shuffle_iter, axis=0, ddof=1)

def _summarize_split_half_maps(
    split_maps,
    selected_trials,
    selected_index,
    target_n,
    n_subsample_iter,
    seed_base,
):
    reliability = []
    for cond_idx, condition_key in enumerate(selected_trials):
        if condition_key not in split_maps:
            continue
        reliability.append(
            _compute_iterated_pv(
                split_maps[condition_key]["half_a"],
                split_maps[condition_key]["half_b"],
                selected_index=selected_index,
                target_n=target_n,
                n_subsample_iter=n_subsample_iter,
                seed_base=seed_base + cond_idx * 10000,
            )
        )
    if not reliability:
        return np.full(56, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(reliability, axis=0), axis=(0, 1))

def _compute_iterated_pv(
    map_a_iter,
    map_b_iter,
    selected_index,
    target_n,
    n_subsample_iter,
    seed_base,
):
    out = np.full((map_a_iter.shape[0], map_a_iter.shape[2]), np.nan)
    for iter_idx in range(map_a_iter.shape[0]):
        map_a = map_a_iter[iter_idx][selected_index, :]
        map_b = map_b_iter[iter_idx][selected_index, :]
        if map_a.shape[0] < target_n or map_b.shape[0] < target_n:
            continue
        out[iter_idx, :] = subsampled_pv_corr(
            map_a,
            map_b,
            target_n,
            n_iter=n_subsample_iter,
            seed=seed_base + iter_idx,
            method="spearman",
        )
    return out

def _align_neuron_mask(mask, n_neuron):
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == n_neuron:
        return mask
    aligned = np.zeros(int(n_neuron), dtype=bool)
    n_copy = min(mask.size, aligned.size)
    aligned[:n_copy] = mask[:n_copy]
    return aligned

def _load_optional_array(load_path, candidates):
    for filename in candidates:
        candidate_path = os.path.join(load_path, filename)
        if os.path.exists(candidate_path):
            return np.load(candidate_path, allow_pickle=True)
    return None

def _day_metadata(metadata, day_idx):
    if metadata is None or day_idx >= len(metadata):
        return None
    return metadata[day_idx]
