"""Manuscript definitions collected from bayesian_position.py.
See provenance/source_files.csv for original hashes and definition ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.special import logsumexp

from scipy.sparse.csgraph import shortest_path

from .spatial import (
    MAIN_REF_X, MAIN_REF_Y, LEFT_REF_X, LEFT_REF_Y, RIGHT_REF_X, RIGHT_REF_Y,
    compute_spatial_activity_map, _trial_bins_to_full_bins,
)

from .trials import valid_labeled_trials

def bin_geometry():
    """Return full-map centers and true path distances via explicit junctions."""
    xy = np.column_stack((
        np.r_[MAIN_REF_X, RIGHT_REF_X[14:], LEFT_REF_X[14:]],
        np.r_[MAIN_REF_Y, RIGHT_REF_Y[14:], LEFT_REF_Y[14:]],
    )).astype(float)
    corners = np.array([[0, 28], [0, -28], [-56, 28], [-56, -28], [56, 28], [56, -28]])
    nodes = np.vstack((xy, corners))
    graph = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(graph, 0)
    for axis, value in [(0, 0), (0, -56), (0, 56), (1, -28), (1, 28)]:
        ids = np.flatnonzero(nodes[:, axis] == value)
        ids = ids[np.argsort(nodes[ids, 1-axis])]
        for a, b in zip(ids[:-1], ids[1:]):
            graph[a, b] = graph[b, a] = np.linalg.norm(nodes[a] - nodes[b])
    return xy, shortest_path(graph, directed=False)[:98, :98]

@dataclass
class BinnedTrial:
    trial_id: int
    actual_left: bool
    correct: bool
    neural: np.ndarray
    bins: np.ndarray
    xy: np.ndarray
    frame_ids: np.ndarray

def prepare_binned_trials(xy_trials, neural_trials, labels, neurons, *, context=None):
    """Reuse spatial-map projection, retaining original frame and trial IDs.

    Project the full behavior sequence before filtering finite neural samples.
    No neural temporal or spatial smoothing is performed here.
    """
    labels = np.asarray(labels, dtype=bool)
    valid = valid_labeled_trials(labels)
    if len(xy_trials) != len(neural_trials) or len(valid) != len(xy_trials):
        raise ValueError('trial/label alignment mismatch')
    if context is not None:
        context = np.asarray(context, dtype=bool)
        if context.shape != valid.shape:
            raise ValueError('context flags must exactly match trials')
        valid &= ~context
    neurons = np.asarray(neurons, dtype=int)
    if neurons.ndim != 1 or not len(neurons) or len(np.unique(neurons)) != len(neurons):
        raise ValueError('neurons must be unique indices')
    trials, qc = [], []
    for idx, (positions, activity) in enumerate(zip(xy_trials, neural_trials)):
        xy = np.asarray(positions, dtype=float)
        neural = np.asarray(activity, dtype=float)
        if neural.ndim != 2 or xy.shape != (neural.shape[1], 2):
            raise ValueError(f'trial {idx}: position/neural alignment mismatch')
        row = dict(trial=idx+1, input_frames=len(xy), retained_frames=0,
                   actual_left=bool(labels[0, idx]), correct=bool(labels[2, idx]))
        if not valid[idx]:
            qc.append(dict(row, status='invalid_label_or_context'))
            continue
        _, projection = compute_spatial_activity_map(
            xy, np.zeros((1, len(xy))), is_left_choice=labels[0, idx],
            smooth=False, return_projection=True,
        )
        finite = np.isfinite(xy).all(axis=1) & np.isfinite(neural[neurons]).all(axis=0)
        keep = projection['valid_indices'] & finite
        bins = _trial_bins_to_full_bins(projection['assigned_bin'][keep], labels[0, idx])
        row.update(retained_frames=int(keep.sum()),
                   excluded_nonfinite=int((~finite).sum()),
                   excluded_projection=int((finite & ~projection['valid_indices']).sum()))
        if keep.sum() < 2:
            qc.append(dict(row, status='too_few_frames'))
            continue
        trials.append(BinnedTrial(idx, bool(labels[0, idx]), bool(labels[2, idx]),
                                  neural[neurons][:, keep].T, bins, xy[keep], np.flatnonzero(keep)))
        qc.append(dict(row, status='ok'))
    if len(trials) < 3:
        raise ValueError('at least three retained trials required')
    return trials, qc

@dataclass
class GaussianBinDecoder:
    means: np.ndarray
    variances: np.ndarray
    counts: np.ndarray
    center: np.ndarray
    active: np.ndarray

    @classmethod
    def fit(cls, neural, bins, *, shrinkage=0.1):
        """Train-only moments; variance shrunk 10% toward each neuron's global variance.

        Flat prior over observed bins. Unobserved bins have zero posterior and
        are reported by evaluation, not filled using held-out activity.
        """
        x, y = np.asarray(neural, float), np.asarray(bins, int)
        if x.ndim != 2 or len(x) != len(y) or not len(x) or not np.isfinite(x).all():
            raise ValueError('finite nonempty neural matrix and matching bins required')
        if np.any((y < 0) | (y >= 98)) or not 0 < shrinkage <= 1:
            raise ValueError('invalid bins or variance shrinkage')
        center = x.mean(axis=0)
        x = x - center
        counts = np.bincount(y, minlength=98)
        sums, squares = np.zeros((98, x.shape[1])), np.zeros((98, x.shape[1]))
        np.add.at(sums, y, x)
        np.add.at(squares, y, x*x)
        means = sums / np.maximum(counts[:, None], 1)
        var = np.maximum(squares / np.maximum(counts[:, None], 1) - means*means, 0)
        global_var = x.var(axis=0)
        var = (1-shrinkage)*var + shrinkage*global_var
        var = np.maximum(var, np.maximum(global_var*1e-6, 1e-12))
        return cls(means + center, var, counts, center, global_var > 0)

    def predict_proba(self, neural):
        x = np.asarray(neural, float)
        if x.ndim != 2 or x.shape[1] != self.means.shape[1] or not np.isfinite(x).all():
            raise ValueError('invalid prediction matrix')
        # Training centering avoids cancellation from large constant baselines.
        # Constant training neurons contain no class information and contribute
        # only a common likelihood term, which cancels from the posterior.
        x = (x - self.center)[:, self.active]
        means = (self.means - self.center)[:, self.active]
        variances = self.variances[:, self.active]
        inv = 1 / variances
        scores = -0.5 * (x*x @ inv.T - 2*x @ (means*inv).T
                         + (means*means*inv + np.log(variances)).sum(axis=1))
        scores[:, self.counts == 0] = -np.inf
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

def shift_bins_within_trials(trials, rng):
    """Shift at least 20% of each trial, preserving within-trial label sequence.

    A conditional temporal null: preserves trial route identity and occupancy,
    and therefore is not a test of route/choice information alone.
    """
    shifted, shifts = [], []
    for t in trials:
        margin = max(1, int(np.ceil(len(t.bins)*0.2)))
        shift = int(rng.integers(margin, len(t.bins)-margin+1))
        shifted.append(np.roll(t.bins, shift))
        shifts.append(shift)
    return shifted, shifts

def decoding_metrics(truth, prediction, path_distances):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    distance = path_distances[truth, prediction]
    correct = truth == prediction
    balanced = np.mean([correct[truth == b].mean() for b in np.unique(truth)])
    return dict(accuracy=float(correct.mean()), balanced_accuracy=float(balanced),
                path_mean_cm=float(distance.mean()), path_median_cm=float(np.median(distance)),
                within_8cm=float((distance <= 8+1e-9).mean()))
