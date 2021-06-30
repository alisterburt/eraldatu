import einops
import numpy as np
import pytest

m = 5  # number of transformations
n = 200 # number of poses


@pytest.fixture
def positions():
    return np.random.randint(low=0, high=200, size=(n, 3))


@pytest.fixture
def shifts():
    return np.random.normal(size=(m, 3))


@pytest.fixture
def identity_matrices():
    return einops.repeat(np.eye(3), 'i j -> new_axis i j', new_axis=n)


@pytest.fixture
def rotz90_matrices():
    cos90 = 0
    sin90 = 1
    matrix = np.array(
        [[cos90, -sin90, 0],
         [sin90, cos90, 0],
         [0, 0, 1]], dtype=float
    )
    return einops.repeat(matrix, 'i j -> new_axis i j', new_axis=m)
