import numpy as np
import pytest

from ..operators import C, D
from ...components.data_model import Transform

@pytest.fixture
def c3():
    return C(3)


@pytest.fixture
def d5():
    return D(5)


@pytest.fixture
def identity_basis():
    return np.eye(3)


@pytest.fixture
def simple_transform():
    shifts = np.random.random(3)
    rotation = np.eye(3)
    return Transform(shifts=shifts, rotations=rotation)
