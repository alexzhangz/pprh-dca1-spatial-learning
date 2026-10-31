import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from maze_analysis.position_posterior import mean_posterior_matrix, average_mouse_matrices


def test_preserves_soft_probability_instead_of_argmax():
    p = np.array([[.6,.3,.1], [.8,.1,.1], [.2,.7,.1]])
    matrix, counts = mean_posterior_matrix(np.array([0,0,1]), p)
    np.testing.assert_allclose(matrix[:2], [[.7,.2,.1], [.2,.7,.1]])
    np.testing.assert_array_equal(counts, [2,1,0])
    assert np.isnan(matrix[2]).all()
    assert matrix[0,0] != 1  # Both frames have argmax 0, but soft mean is .7.


def test_mouse_equal_weighting_and_missing_rows():
    a,_ = mean_posterior_matrix(np.zeros(100,dtype=int), np.tile([.9,.1],(100,1)))
    b,_ = mean_posterior_matrix(np.array([0,1]), np.array([[.1,.9],[.3,.7]]))
    average,counts = average_mouse_matrices([a,b])
    np.testing.assert_allclose(average, [[.5,.5],[.3,.7]])
    np.testing.assert_array_equal(counts,[2,1])
    np.testing.assert_allclose(average.sum(axis=1),1)


@pytest.mark.parametrize('p', [np.array([[.2,.2]]),np.array([[np.nan,.5]]),np.array([[-.1,1.1]])])
def test_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError): mean_posterior_matrix(np.array([0]),p)
