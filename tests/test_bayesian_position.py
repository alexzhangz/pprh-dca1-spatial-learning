import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from maze_analysis.bayesian_position import (
    bin_geometry, GaussianBinDecoder, prepare_binned_trials, shift_bins_within_trials,
)
from maze_analysis.spatial import LEFT_REF_X, LEFT_REF_Y, RIGHT_REF_X, RIGHT_REF_Y


def test_full_maze_geometry_and_junction_distances():
    xy, distances = bin_geometry()
    assert xy.shape == (98, 2)
    assert len(np.unique(xy, axis=0)) == 98
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0)
    assert distances[14, 56] == 4  # top junction, not via a duplicate stem bin
    assert distances[55, 97] == 4  # bottom junction
    assert distances[27, 28] == 4  # right upper corner
    assert distances[0, 13] == 52
    assert np.isfinite(distances).all()


def test_gaussian_posterior_matches_direct_density_and_excludes_unobserved():
    x = np.array([[0., 2], [1, 2], [8, 2], [9, 2]])
    y = np.array([0, 0, 97, 97])
    model = GaussianBinDecoder.fit(x, y)
    p = model.predict_proba(x)
    score = -0.5*((x[:, None, :]-model.means[None])**2/model.variances[None]
                  + np.log(model.variances[None])).sum(axis=2)
    score[:, model.counts == 0] = -np.inf
    direct = np.exp(score-score.max(axis=1, keepdims=True))
    direct /= direct.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p, direct, atol=1e-12)
    np.testing.assert_array_equal(p.argmax(axis=1), y)
    assert (p[:, 1:97] == 0).all()
    original = model.means.copy()
    model.predict_proba(np.full((1, 2), 10000.))
    np.testing.assert_array_equal(model.means, original)


def test_projection_uses_actual_choice_including_incorrect_trials():
    left = np.column_stack([LEFT_REF_X, LEFT_REF_Y])
    right = np.column_stack([RIGHT_REF_X, RIGHT_REF_Y])
    labels = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]], bool)
    trials, qc = prepare_binned_trials([left, right, left, right],
                                      [np.ones((2, 56))]*4, labels, [0, 1])
    np.testing.assert_array_equal(trials[0].bins, np.r_[np.arange(14), np.arange(56, 98)])
    np.testing.assert_array_equal(trials[1].bins, np.arange(56))
    assert not trials[0].correct and not trials[1].correct
    assert qc[-1]['status'] == 'invalid_label_or_context'
    shifted, offsets = shift_bins_within_trials(trials, np.random.default_rng(0))
    for t, s, offset in zip(trials, shifted, offsets):
        np.testing.assert_array_equal(np.sort(s), np.sort(t.bins))
        assert 0.2*len(s) <= offset <= 0.8*len(s)


def test_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match='alignment'):
        prepare_binned_trials([], [], np.ones((3, 2)), [0])


def test_posterior_invariant_to_offset_and_constant_training_features():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(30, 3))
    y = np.repeat([0, 40, 90], 10)
    x[:, 0] += y / 20
    test = rng.normal(size=(5, 3))
    baseline = GaussianBinDecoder.fit(x, y).predict_proba(test)
    offset = np.array([10000., -20000., 30000.])
    augmented = np.column_stack((x+offset, np.full(30, 7.)))
    query = np.column_stack((test+offset, np.full(5, 10000.)))
    shifted = GaussianBinDecoder.fit(augmented, y).predict_proba(query)
    np.testing.assert_allclose(baseline, shifted, atol=1e-10)
    constant = GaussianBinDecoder.fit(np.ones((4, 2)), [0, 0, 1, 1])
    np.testing.assert_allclose(constant.predict_proba([[100., -100.]])[0, :2], [.5, .5])
