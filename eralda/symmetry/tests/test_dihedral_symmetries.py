import numpy as np
import pytest

from ..dihedral import derive_rotation_matrices


@pytest.mark.parametrize('order', [i for i in range(1, 5)])
def test_cyclic_symmetry(order):
    rotation_matrices = derive_rotation_matrices(order)
    assert rotation_matrices.shape == (2 * order, 3, 3)
    assert np.allclose(np.linalg.det(rotation_matrices), 1)
