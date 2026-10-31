"""Manuscript definitions collected from spatial.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import numpy as np

from scipy.ndimage import gaussian_filter1d

MAIN_REF_X = np.asarray([0] * len(np.arange(-28, 28, 4)))

MAIN_REF_Y = np.arange(-28, 28, 4) + 2

LEFT_REF_X = np.concatenate(
    (
        MAIN_REF_X,
        np.arange(-56, 0, 4)[::-1] + 2,
        np.asarray([-56] * len(np.arange(-28, 28, 4))),
        np.arange(-56, 0, 4) + 2,
    )
)

LEFT_REF_Y = np.concatenate(
    (
        MAIN_REF_Y,
        np.asarray([28] * len(np.arange(-56, 0, 4))),
        np.arange(-28, 28, 4)[::-1] + 2,
        np.asarray([-28] * len(np.arange(-56, 0, 4))),
    )
)

RIGHT_REF_X = np.concatenate(
    (
        MAIN_REF_X,
        np.arange(0, 56, 4) + 2,
        np.asarray([56] * len(np.arange(-28, 28, 4))),
        np.arange(0, 56, 4)[::-1] + 2,
    )
)

RIGHT_REF_Y = np.concatenate(
    (
        MAIN_REF_Y,
        np.asarray([28] * len(np.arange(0, 56, 4))),
        np.arange(-28, 28, 4)[::-1] + 2,
        np.asarray([-28] * len(np.arange(0, 56, 4))),
    )
)

def _trial_ref(is_left_choice):
    if bool(is_left_choice):
        return LEFT_REF_X, LEFT_REF_Y
    return RIGHT_REF_X, RIGHT_REF_Y

def _nan_map(num_neuron, num_bin=56):
    return np.full((int(num_neuron), int(num_bin)), np.nan, dtype=float)

def compute_spatial_activity_map(
    xy,
    dff,
    is_left_choice,
    distance_threshold=6,
    smooth=True,
    transition_penalty=0.25,
    return_projection=False,
):
    """Project one trial's activity onto the traversed 56-bin maze arm."""
    xy = np.asarray(xy, dtype=float)
    dff = np.asarray(dff, dtype=float)
    if dff.ndim != 2:
        raise ValueError(f"dff must have shape (n_neuron, n_frame), got {dff.shape}")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (n_frame, 2), got {xy.shape}")
    if xy.shape[0] != dff.shape[1]:
        raise ValueError(
            f"xy frame count ({xy.shape[0]}) must match dff frame count ({dff.shape[1]})"
        )

    num_neuron = dff.shape[0]
    ref_x, ref_y = _trial_ref(is_left_choice)
    if xy.shape[0] == 0:
        snarm = _nan_map(num_neuron, len(ref_x))
        if return_projection:
            return snarm, _empty_projection(len(ref_x))
        return snarm

    distance_matrix = np.sqrt(
        (xy[:, 0][:, np.newaxis] - ref_x[np.newaxis, :]) ** 2
        + (xy[:, 1][:, np.newaxis] - ref_y[np.newaxis, :]) ** 2
    )
    distance_matrix = np.nan_to_num(
        distance_matrix,
        nan=1_000_000.0,
        posinf=1_000_000.0,
        neginf=1_000_000.0,
    )
    assigned_bin = _ordered_bin_assignment(
        distance_matrix,
        transition_penalty=transition_penalty,
    )
    assigned_distance = distance_matrix[np.arange(distance_matrix.shape[0]), assigned_bin]
    valid_indices = assigned_distance < distance_threshold
    if not np.any(valid_indices):
        snarm = _nan_map(num_neuron, len(ref_x))
        if return_projection:
            return snarm, {
                "assigned_bin": assigned_bin,
                "assigned_distance": assigned_distance,
                "valid_indices": valid_indices,
            }
        return snarm

    nearest_bin = assigned_bin[valid_indices]
    dff_valid = dff[:, valid_indices]
    occupied_bins = np.unique(nearest_bin)
    if occupied_bins.size == 0:
        snarm = _nan_map(num_neuron, len(ref_x))
        if return_projection:
            return snarm, {
                "assigned_bin": assigned_bin,
                "assigned_distance": assigned_distance,
                "valid_indices": valid_indices,
            }
        return snarm

    snarm = _nan_map(num_neuron, len(ref_x))
    for bin_idx in occupied_bins:
        snarm[:, bin_idx] = np.nanmean(dff_valid[:, nearest_bin == bin_idx], axis=1)

    full_bins = np.arange(len(ref_x))
    snarm_interp = _nan_map(num_neuron, len(ref_x))
    for neuron_idx in range(num_neuron):
        observed_bins = np.where(~np.isnan(snarm[neuron_idx, :]))[0]
        if observed_bins.size == 0:
            continue
        snarm_interp[neuron_idx, :] = np.interp(
            full_bins,
            observed_bins,
            snarm[neuron_idx, observed_bins],
        )

    if smooth:
        snarm_interp = gaussian_filter1d(snarm_interp, sigma=1, axis=1)
    if return_projection:
        return snarm_interp, {
            "assigned_bin": assigned_bin,
            "assigned_distance": assigned_distance,
            "valid_indices": valid_indices,
        }
    return snarm_interp

def compute_condition_shuffle_maps(
    xy_day,
    dff_day,
    labels_day,
    condition_keys=("PL", "PR"),
    trial_segment_ids=None,
    is_context_trial=None,
    n_shuffle=100,
    seed=2026,
    distance_threshold=6,
    smooth=True,
    transition_penalty=0.25,
):
    """Shuffle selected 56-bin condition labels within one mouse-day.

    The union of the requested real conditions is shuffled while preserving the
    real trial count for each condition. This supports both previous-trial
    conditions (PL/PR) and current correct-trial conditions (CL/CR).
    """
    labels_day = np.asarray(labels_day, dtype=bool)
    xy_day, dff_day = _align_single_day_trial_arrays(labels_day, xy_day, dff_day)
    _validate_snarm_all_day_inputs(0, labels_day, xy_day, dff_day)

    condition_keys = tuple(condition_keys)
    n_shuffle = int(n_shuffle)
    if n_shuffle < 0:
        raise ValueError("n_shuffle must be non-negative")

    labels_aligned = labels_day[:3, 1:]
    valid_labels = np.any(labels_aligned, axis=0)
    actual_left = labels_aligned[0].astype(bool)
    num_neuron = _infer_num_neuron([dff_day])
    projected_trials = _project_day_trials(
        xy_day,
        dff_day,
        actual_left,
        valid_labels,
        num_neuron,
        distance_threshold=distance_threshold,
        transition_penalty=transition_penalty,
    )
    segment_ids_day = _align_single_day_snarm_metadata(
        trial_segment_ids, labels_aligned.shape[1]
    )
    context_day = _align_single_day_snarm_metadata(
        is_context_trial, labels_aligned.shape[1], dtype=bool
    )
    real_masks = _snarm_all_56_condition_masks(
        labels_aligned,
        condition_keys,
        segment_ids=segment_ids_day,
        is_context_trial=context_day,
    )
    condition_counts = {key: int(np.sum(real_masks[key])) for key in condition_keys}
    eligible_mask = np.zeros(labels_aligned.shape[1], dtype=bool)
    for key in condition_keys:
        eligible_mask |= real_masks[key]
    eligible_indices = np.flatnonzero(eligible_mask)

    shuffle_maps = {
        key: np.full((n_shuffle, num_neuron, len(LEFT_REF_X)), np.nan)
        for key in condition_keys
    }
    if n_shuffle == 0 or any(count == 0 for count in condition_counts.values()):
        return shuffle_maps

    rng = np.random.default_rng(seed)
    for shuffle_idx in range(n_shuffle):
        shuffled_indices = rng.permutation(eligible_indices)
        masks = {key: np.zeros(labels_aligned.shape[1], dtype=bool) for key in condition_keys}
        start = 0
        for key in condition_keys:
            stop = start + condition_counts[key]
            masks[key][shuffled_indices[start:stop]] = True
            start = stop
        for key in condition_keys:
            map_day, _ = _aggregate_56_bin_map(
                projected_trials,
                masks[key],
                num_neuron,
                smooth=smooth,
                min_trials=1,
            )
            shuffle_maps[key][shuffle_idx, :, :] = map_day
    return shuffle_maps

def compute_condition_odd_even_maps(
    xy_day,
    dff_day,
    labels_day,
    condition_keys=("PL", "PR"),
    trial_segment_ids=None,
    is_context_trial=None,
    distance_threshold=6,
    smooth=True,
    transition_penalty=0.25,
    min_split_trials=4,
):
    """Return condition-specific odd/even 56-bin maps for one mouse-day.

    Valid trials are kept in their chronological order separately within each
    requested condition. The first, third, fifth, ... occurrences form the odd
    subset and the second, fourth, sixth, ... occurrences form the even subset.
    When the trial count is odd, the additional trial remains in the odd subset.
    At least two trials are required in each subset.
    """
    labels_day = np.asarray(labels_day, dtype=bool)
    xy_day, dff_day = _align_single_day_trial_arrays(labels_day, xy_day, dff_day)
    _validate_snarm_all_day_inputs(0, labels_day, xy_day, dff_day)

    condition_keys = tuple(condition_keys)
    min_split_trials = int(min_split_trials)
    if min_split_trials < 1:
        raise ValueError("min_split_trials must be positive")

    labels_aligned = labels_day[:3, 1:]
    valid_labels = np.any(labels_aligned, axis=0)
    actual_left = labels_aligned[0].astype(bool)
    num_neuron = _infer_num_neuron([dff_day])
    projected_trials = _project_day_trials(
        xy_day,
        dff_day,
        actual_left,
        valid_labels,
        num_neuron,
        distance_threshold=distance_threshold,
        transition_penalty=transition_penalty,
    )
    segment_ids_day = _align_single_day_snarm_metadata(
        trial_segment_ids, labels_aligned.shape[1]
    )
    context_day = _align_single_day_snarm_metadata(
        is_context_trial, labels_aligned.shape[1], dtype=bool
    )
    real_masks = _snarm_all_56_condition_masks(
        labels_aligned,
        condition_keys,
        segment_ids=segment_ids_day,
        is_context_trial=context_day,
    )
    split_maps = {
        key: {
            "odd_trials": _nan_map(num_neuron, len(LEFT_REF_X)),
            "even_trials": _nan_map(num_neuron, len(LEFT_REF_X)),
            "trial_count": 0,
            "odd_trial_count": 0,
            "even_trial_count": 0,
        }
        for key in condition_keys
    }

    for key in condition_keys:
        condition_indices = np.flatnonzero(real_masks[key])
        odd_indices = condition_indices[0::2]
        even_indices = condition_indices[1::2]
        split_maps[key]["trial_count"] = int(condition_indices.size)
        split_maps[key]["odd_trial_count"] = int(odd_indices.size)
        split_maps[key]["even_trial_count"] = int(even_indices.size)
        if condition_indices.size < min_split_trials:
            continue
        if odd_indices.size < 2 or even_indices.size < 2:
            continue

        for subset_key, trial_indices in (
            ("odd_trials", odd_indices),
            ("even_trials", even_indices),
        ):
            subset_mask = np.zeros(labels_aligned.shape[1], dtype=bool)
            subset_mask[trial_indices] = True
            map_day, _ = _aggregate_56_bin_map(
                projected_trials,
                subset_mask,
                num_neuron,
                smooth=smooth,
                min_trials=1,
            )
            split_maps[key][subset_key] = map_day
    return split_maps

def compute_condition_split_half_maps(
    xy_day,
    dff_day,
    labels_day,
    condition_keys=("PL", "PR"),
    trial_segment_ids=None,
    is_context_trial=None,
    n_split=100,
    seed=2026,
    distance_threshold=6,
    smooth=True,
    transition_penalty=0.25,
    min_split_trials=4,
):
    """Return within-condition split-half maps for selected 56-bin conditions."""
    labels_day = np.asarray(labels_day, dtype=bool)
    xy_day, dff_day = _align_single_day_trial_arrays(labels_day, xy_day, dff_day)
    _validate_snarm_all_day_inputs(0, labels_day, xy_day, dff_day)

    condition_keys = tuple(condition_keys)
    n_split = int(n_split)
    if n_split < 0:
        raise ValueError("n_split must be non-negative")

    labels_aligned = labels_day[:3, 1:]
    valid_labels = np.any(labels_aligned, axis=0)
    actual_left = labels_aligned[0].astype(bool)
    num_neuron = _infer_num_neuron([dff_day])
    projected_trials = _project_day_trials(
        xy_day,
        dff_day,
        actual_left,
        valid_labels,
        num_neuron,
        distance_threshold=distance_threshold,
        transition_penalty=transition_penalty,
    )
    segment_ids_day = _align_single_day_snarm_metadata(
        trial_segment_ids, labels_aligned.shape[1]
    )
    context_day = _align_single_day_snarm_metadata(
        is_context_trial, labels_aligned.shape[1], dtype=bool
    )
    real_masks = _snarm_all_56_condition_masks(
        labels_aligned,
        condition_keys,
        segment_ids=segment_ids_day,
        is_context_trial=context_day,
    )
    split_maps = {
        key: {
            "half_a": np.full((n_split, num_neuron, len(LEFT_REF_X)), np.nan),
            "half_b": np.full((n_split, num_neuron, len(LEFT_REF_X)), np.nan),
        }
        for key in condition_keys
    }
    if n_split == 0:
        return split_maps

    rng = np.random.default_rng(seed)
    for key in condition_keys:
        condition_indices = np.flatnonzero(real_masks[key])
        if condition_indices.size < min_split_trials:
            continue
        split_n = condition_indices.size // 2
        for split_idx in range(n_split):
            shuffled_indices = rng.permutation(condition_indices)
            masks = {
                "half_a": np.zeros(labels_aligned.shape[1], dtype=bool),
                "half_b": np.zeros(labels_aligned.shape[1], dtype=bool),
            }
            masks["half_a"][shuffled_indices[:split_n]] = True
            masks["half_b"][shuffled_indices[split_n:]] = True
            for half_key, half_mask in masks.items():
                map_day, _ = _aggregate_56_bin_map(
                    projected_trials,
                    half_mask,
                    num_neuron,
                    smooth=smooth,
                    min_trials=1,
                )
                split_maps[key][half_key][split_idx, :, :] = map_day
    return split_maps

def normalize_spatial_rows(snarm_map):
    """Return row-wise min-max normalized copy for heatmap display."""
    snarm_map = np.asarray(snarm_map, dtype=float)
    if snarm_map.ndim != 2:
        raise ValueError(f"snarm_map must be 2D, got {snarm_map.shape}")

    normalized = np.full(snarm_map.shape, np.nan, dtype=float)
    for row_idx, row in enumerate(snarm_map):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        row_min = np.nanmin(row[finite])
        row_max = np.nanmax(row[finite])
        if row_max > row_min:
            normalized[row_idx, finite] = (row[finite] - row_min) / (row_max - row_min)
        else:
            normalized[row_idx, finite] = 0.0
    return normalized

def spatial_peak_sort_index(snarm_map):
    """Return neuron indices ordered by spatial peak bin, with all-NaN rows last."""
    normalized = normalize_spatial_rows(snarm_map)
    num_neuron = normalized.shape[0]
    peak_bins = np.full(num_neuron, normalized.shape[1], dtype=int)
    for neuron_idx, row in enumerate(normalized):
        finite = np.isfinite(row)
        if np.any(finite):
            peak_bins[neuron_idx] = int(np.nanargmax(row))
    return np.lexsort((np.arange(num_neuron), peak_bins))

def _validate_snarm_all_day_inputs(day_idx, labels_day, xy_day, dff_day):
    if labels_day.ndim != 2 or labels_day.shape[0] < 3:
        raise ValueError(
            f"trial_labels[{day_idx}] must have shape (3, n_trials), got {labels_day.shape}"
        )
    expected_trials = labels_day.shape[1] - 1
    if expected_trials < 0:
        raise ValueError(f"trial_labels[{day_idx}] has no trial axis")
    if len(xy_day) != expected_trials:
        raise ValueError(
            f"Day {day_idx + 1}: expected {expected_trials} extracted xy trials, got {len(xy_day)}"
        )
    if len(dff_day) != expected_trials:
        raise ValueError(
            f"Day {day_idx + 1}: expected {expected_trials} extracted dff trials, got {len(dff_day)}"
        )

def _align_single_day_trial_arrays(labels_day, xy_day, dff_day):
    expected_trials = labels_day.shape[1] - 1
    xy_values = np.asarray(xy_day, dtype=object)
    dff_values = np.asarray(dff_day, dtype=object)
    if len(xy_values) == expected_trials and len(dff_values) == expected_trials:
        return xy_day, dff_day
    if len(xy_values) == labels_day.shape[1] and len(dff_values) == labels_day.shape[1]:
        return xy_values[1:], dff_values[1:]
    return xy_day, dff_day

def _project_day_trials(
    xy_day,
    dff_day,
    actual_left,
    valid_labels,
    num_neuron,
    distance_threshold,
    transition_penalty,
):
    projected = []
    for trial_idx in range(len(xy_day)):
        if not valid_labels[trial_idx]:
            projected.append(None)
            continue
        xy = np.asarray(xy_day[trial_idx], dtype=float)
        dff = np.asarray(dff_day[trial_idx], dtype=float)
        if dff.ndim != 2 or dff.shape[0] != num_neuron or dff.shape[1] == 0:
            projected.append(None)
            continue
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != dff.shape[1]:
            raise ValueError(
                f"Trial {trial_idx} has incompatible xy/dff shapes: {xy.shape} vs {dff.shape}"
            )
        bins, dff_valid = _project_trial_frames(
            xy,
            dff,
            actual_left[trial_idx],
            distance_threshold=distance_threshold,
            transition_penalty=transition_penalty,
        )
        projected.append(
            {
                "actual_left": bool(actual_left[trial_idx]),
                "bins": bins,
                "dff": dff_valid,
                "valid_frame_count": int(bins.size),
            }
        )
    return projected

def _project_trial_frames(
    xy,
    dff,
    is_left_choice,
    distance_threshold,
    transition_penalty,
):
    ref_x, ref_y = _trial_ref(is_left_choice)
    distance_matrix = np.sqrt(
        (xy[:, 0][:, np.newaxis] - ref_x[np.newaxis, :]) ** 2
        + (xy[:, 1][:, np.newaxis] - ref_y[np.newaxis, :]) ** 2
    )
    distance_matrix = np.nan_to_num(
        distance_matrix,
        nan=1_000_000.0,
        posinf=1_000_000.0,
        neginf=1_000_000.0,
    )
    assigned_bin = _ordered_bin_assignment(
        distance_matrix,
        transition_penalty=transition_penalty,
    )
    assigned_distance = distance_matrix[np.arange(distance_matrix.shape[0]), assigned_bin]
    valid = assigned_distance < distance_threshold
    return assigned_bin[valid], dff[:, valid]

def _snarm_all_condition_masks(labels_aligned, is_context_trial=None):
    valid = np.any(labels_aligned[:3], axis=0)
    if is_context_trial is not None:
        valid &= ~np.asarray(is_context_trial, dtype=bool)
    actual_left = labels_aligned[0].astype(bool)
    reward_left = labels_aligned[1].astype(bool)
    correct = labels_aligned[2].astype(bool)
    return {
        "All": valid,
        "BL": valid & actual_left,
        "BR": valid & ~actual_left,
        "TL": valid & reward_left,
        "TR": valid & ~reward_left,
        "CA": valid & correct,
        "IA": valid & ~correct,
        "CL": valid & actual_left & correct,
        "CR": valid & ~actual_left & correct,
        "IL": valid & reward_left & ~correct,
        "IR": valid & ~reward_left & ~correct,
        "C": valid & correct,
        "I": valid & ~correct,
    }

def _align_single_day_snarm_metadata(metadata, n_aligned, dtype=None):
    if metadata is None:
        return None
    values = np.asarray(metadata, dtype=dtype).ravel()
    if values.shape[0] >= n_aligned + 1:
        return values[1 : n_aligned + 1]
    if values.shape[0] == n_aligned:
        return values
    raise ValueError("trial metadata must align with trial_labels day columns")

def _snarm_all_previous_condition_masks(labels_aligned, segment_ids=None, is_context_trial=None):
    valid = np.any(labels_aligned[:3], axis=0)
    if is_context_trial is not None:
        valid &= ~np.asarray(is_context_trial, dtype=bool)
    actual_left = labels_aligned[0].astype(bool)
    correct = labels_aligned[2].astype(bool)
    current_correct_left = valid & actual_left & correct
    current_correct_right = valid & ~actual_left & correct

    previous_to_correct_left = np.zeros(valid.shape, dtype=bool)
    previous_to_correct_right = np.zeros(valid.shape, dtype=bool)
    if valid.size > 1:
        same_segment = np.ones(valid.size - 1, dtype=bool)
        if segment_ids is not None:
            segment_ids = np.asarray(segment_ids).ravel()
            if segment_ids.shape[0] != valid.shape[0]:
                raise ValueError("segment_ids must align with trial_labels[:, 1:] columns")
            same_segment = segment_ids[:-1] == segment_ids[1:]
        previous_to_correct_left[:-1] = (
            current_correct_left[1:] & valid[:-1] & ~actual_left[:-1] & same_segment
        )
        previous_to_correct_right[:-1] = (
            current_correct_right[1:] & valid[:-1] & actual_left[:-1] & same_segment
        )
    return {
        "PL": previous_to_correct_left,
        "PR": previous_to_correct_right,
    }

def _snarm_all_56_condition_masks(
    labels_aligned,
    condition_keys,
    segment_ids=None,
    is_context_trial=None,
):
    current_masks = _snarm_all_condition_masks(
        labels_aligned,
        is_context_trial=is_context_trial,
    )
    previous_masks = _snarm_all_previous_condition_masks(
        labels_aligned,
        segment_ids=segment_ids,
        is_context_trial=is_context_trial,
    )
    available_masks = {**current_masks, **previous_masks}
    unknown = [key for key in condition_keys if key not in available_masks]
    if unknown:
        raise ValueError(f"Unknown 56-bin condition key(s): {unknown}")
    return {key: available_masks[key] for key in condition_keys}

def _aggregate_56_bin_map(projected_trials, mask, num_neuron, smooth, min_trials=1):
    trial_count = int(np.sum(mask))
    sums, counts = _empty_accumulator(num_neuron, len(LEFT_REF_X))
    valid_frame_count = 0
    if trial_count >= min_trials:
        for trial in _selected_projected_trials(projected_trials, mask):
            bins = trial["bins"]
            dff = trial["dff"]
            _accumulate_binned_frames(sums, counts, bins, dff)
            valid_frame_count += int(bins.size)
    snarm_map = _finalize_accumulator(sums, counts, smooth=smooth)
    return snarm_map, {
        "trial_count": trial_count,
        "left_trial_count": np.nan,
        "right_trial_count": np.nan,
        "valid_frame_count": valid_frame_count,
        "left_side_nan": False,
        "right_side_nan": False,
    }

def _selected_projected_trials(projected_trials, mask):
    for selected, trial in zip(mask, projected_trials):
        if selected and trial is not None and trial["valid_frame_count"] > 0:
            yield trial

def _trial_bins_to_full_bins(bins, actual_left):
    bins = np.asarray(bins, dtype=int)
    full_bins = bins.copy()
    side = bins >= len(MAIN_REF_X)
    if bool(actual_left):
        full_bins[side] = bins[side] + (len(RIGHT_REF_X) - len(MAIN_REF_X))
    return full_bins

def _empty_accumulator(num_neuron, num_bin):
    return (
        np.zeros((int(num_neuron), int(num_bin)), dtype=float),
        np.zeros((int(num_neuron), int(num_bin)), dtype=int),
    )

def _accumulate_binned_frames(sums, counts, bins, dff):
    if bins.size == 0:
        return
    for bin_idx in np.unique(bins):
        frame_mask = bins == bin_idx
        values = dff[:, frame_mask]
        finite = np.isfinite(values)
        sums[:, bin_idx] += np.nansum(values, axis=1)
        counts[:, bin_idx] += np.sum(finite, axis=1)

def _finalize_accumulator(sums, counts, smooth):
    raw = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=raw, where=counts > 0)
    snarm_map = _interpolate_spatial_rows(raw)
    if smooth:
        snarm_map = gaussian_filter1d(snarm_map, sigma=1, axis=1)
    return snarm_map

def _interpolate_spatial_rows(snarm_map):
    full_bins = np.arange(snarm_map.shape[1])
    interpolated = np.full(snarm_map.shape, np.nan, dtype=float)
    for neuron_idx, row in enumerate(snarm_map):
        observed = np.where(np.isfinite(row))[0]
        if observed.size == 0:
            continue
        interpolated[neuron_idx, :] = np.interp(full_bins, observed, row[observed])
    return interpolated

def _ordered_bin_assignment(distance_matrix, transition_penalty=0.25):
    """Assign frames to monotonically nondecreasing behavior bins."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must have shape (n_frame, n_bin)")
    num_frame, num_bin = distance_matrix.shape
    if num_frame == 0:
        return np.empty(0, dtype=int)
    if num_bin == 0:
        raise ValueError("distance_matrix must contain at least one bin")

    bins = np.arange(num_bin, dtype=float)
    observation_cost = distance_matrix**2
    dp_prev = observation_cost[0, :] + float(transition_penalty) * bins
    backpointers = np.zeros((num_frame, num_bin), dtype=np.int16)

    for frame_idx in range(1, num_frame):
        transition_base = dp_prev - float(transition_penalty) * bins
        cumulative_min = np.empty(num_bin, dtype=float)
        cumulative_argmin = np.empty(num_bin, dtype=np.int16)
        best_cost = np.inf
        best_bin = 0
        for bin_idx in range(num_bin):
            if transition_base[bin_idx] < best_cost:
                best_cost = transition_base[bin_idx]
                best_bin = bin_idx
            cumulative_min[bin_idx] = best_cost
            cumulative_argmin[bin_idx] = best_bin

        dp_prev = observation_cost[frame_idx, :] + float(transition_penalty) * bins + cumulative_min
        backpointers[frame_idx, :] = cumulative_argmin

    assigned = np.empty(num_frame, dtype=int)
    assigned[-1] = int(np.argmin(dp_prev))
    for frame_idx in range(num_frame - 1, 0, -1):
        assigned[frame_idx - 1] = int(backpointers[frame_idx, assigned[frame_idx]])
    return assigned

def _empty_projection(num_bin):
    return {
        "assigned_bin": np.empty(0, dtype=int),
        "assigned_distance": np.empty(0, dtype=float),
        "valid_indices": np.empty(0, dtype=bool),
        "num_bin": int(num_bin),
    }

def _infer_num_neuron(d_dff_all):
    for day in d_dff_all:
        for trial in day:
            trial_arr = np.asarray(trial)
            if trial_arr.ndim == 2:
                return int(trial_arr.shape[0])
    raise ValueError("Could not infer neuron count from d_dff_all")
