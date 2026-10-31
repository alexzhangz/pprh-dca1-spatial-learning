"""Manuscript definitions collected from trials.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

import numpy as np

def should_skip_mouse_day(day_idx, mouse_name):
    """Return True for project-specific mouse/day exclusions."""
    return (day_idx == 6 and mouse_name == "A9") or (day_idx == 5 and mouse_name == "A7")

def valid_labeled_trials(labels_day):
    """Return a trial mask that excludes columns where all three labels are false."""
    labels = np.asarray(labels_day)
    if labels.ndim != 2 or labels.shape[0] < 3:
        raise ValueError(f"labels_day must have shape (3, n_trials), got {labels.shape}")
    return np.any(labels[:3].astype(bool), axis=0)

def eligible_previous_current_indices(labels_day, n_trial, segment_ids=None, is_context_trial=None):
    """Return previous/current trial indices for valid current-correct adjacent pairs.

    The notebook 11b alignment drops the first label column before matching labels
    to `snarm_1d_trial`. Both previous and current trials must be labeled, and
    the current trial must be correct. Optional trial metadata is aligned the same
    way, so returned indices still refer to the `labels_day[:, 1:]` / snarm space.
    """
    labels = np.asarray(labels_day)
    if labels.ndim != 2 or labels.shape[0] < 3:
        raise ValueError(f"labels_day must have shape (3, n_trials), got {labels.shape}")

    n_trial = int(n_trial)
    labels_aligned = labels[:, 1:]
    n_aligned = min(n_trial, labels_aligned.shape[1])
    offset = 1

    if n_aligned < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    labels_aligned = labels_aligned[:, :n_aligned]
    valid = valid_labeled_trials(labels_aligned)
    if is_context_trial is not None:
        context = _align_trial_metadata(is_context_trial, n_aligned, offset, dtype=bool)
        valid &= ~context

    current_correct = labels_aligned[2].astype(bool)

    current_idx = np.where(current_correct & valid)[0]
    current_idx = current_idx[current_idx > 0]
    previous_idx = current_idx - 1

    pair_valid = valid[previous_idx] & valid[current_idx]
    if segment_ids is not None and previous_idx.size:
        segments = _align_trial_metadata(segment_ids, n_aligned, offset)
        pair_valid &= segments[previous_idx] == segments[current_idx]
    return previous_idx[pair_valid].astype(int), current_idx[pair_valid].astype(int)

def shuffle_previous_indices_within_current_side(
    labels_day,
    previous_idx,
    current_idx,
    n_iter=200,
    seed=2026,
    side_row=0,
    max_attempts=100,
):
    """Shuffle previous-trial indices within current-trial left/right strata.

    Returned indices refer to the same `labels_day[:, 1:]` / snarm space as
    `eligible_previous_current_indices`. Current-trial indices are never
    shuffled. Strata with fewer than two pairs are marked invalid.
    """
    labels = np.asarray(labels_day)
    if labels.ndim != 2 or labels.shape[0] < 3:
        raise ValueError(f"labels_day must have shape (3, n_trials), got {labels.shape}")

    side_row = int(side_row)
    if side_row < 0 or side_row >= labels.shape[0]:
        raise ValueError(f"side_row must index labels_day rows, got {side_row}")

    previous_idx = np.asarray(previous_idx, dtype=int).ravel()
    current_idx = np.asarray(current_idx, dtype=int).ravel()
    if previous_idx.shape != current_idx.shape:
        raise ValueError(
            f"previous_idx and current_idx must have the same shape, got {previous_idx.shape} and {current_idx.shape}"
        )

    n_iter = int(n_iter)
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")

    n_pair = previous_idx.size
    shuffled = np.full((n_iter, n_pair), -1, dtype=int)
    valid = np.zeros((n_iter, n_pair), dtype=bool)
    if n_iter == 0 or n_pair == 0:
        return shuffled, valid

    labels_aligned = labels[:, 1:]
    if np.any(current_idx < 0) or np.any(current_idx >= labels_aligned.shape[1]):
        raise ValueError("current_idx must be aligned to labels_day[:, 1:] columns")

    current_side = labels_aligned[side_row, current_idx].astype(bool)
    rng = np.random.default_rng(seed)

    for side_value in (False, True):
        positions = np.where(current_side == side_value)[0]
        if positions.size < 2:
            continue

        previous_values = previous_idx[positions]
        current_values = current_idx[positions]
        for iter_idx in range(n_iter):
            candidate = rng.permutation(previous_values)
            candidate_valid = candidate != current_values
            attempts = 0
            while not np.all(candidate_valid) and attempts < max_attempts:
                candidate = rng.permutation(previous_values)
                candidate_valid = candidate != current_values
                attempts += 1

            shuffled[iter_idx, positions] = candidate
            valid[iter_idx, positions] = candidate_valid

    return shuffled, valid

def align_neuron_mask(mask, n_neuron):
    """Pad or truncate a boolean neuron mask to match the requested neuron count."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.shape[0] == n_neuron:
        return mask

    aligned = np.zeros(int(n_neuron), dtype=bool)
    n_copy = min(mask.shape[0], int(n_neuron))
    aligned[:n_copy] = mask[:n_copy]
    return aligned

def _align_trial_metadata(values, n_aligned, offset, dtype=None):
    values = np.asarray(values, dtype=dtype).ravel()
    if values.shape[0] >= offset + n_aligned:
        return values[offset : offset + n_aligned]
    if values.shape[0] == n_aligned:
        return values
    raise ValueError("trial metadata must align with labels_day trial columns")
