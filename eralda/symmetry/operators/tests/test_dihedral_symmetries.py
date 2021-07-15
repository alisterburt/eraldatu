import numpy as np
import pytest

from ..dihedral import derive_rotation_matrices, Dihedral


@pytest.mark.parametrize('order', [i for i in range(1, 5)])
def test_dihedral_symmetry(order):
    rotation_matrices = derive_rotation_matrices(order)
    assert rotation_matrices.shape == (2 * order, 3, 3)
    assert np.allclose(np.linalg.det(rotation_matrices), 1)


@pytest.mark.parametrize('order', [i for i in range(1, 5)])
def test_dihedral_operator_instantiation_valid(order):
    operator = Dihedral(order)
    assert operator.matrices.shape == (2 * order, 3, 3)
    assert np.allclose(np.linalg.det(operator.matrices), 1)


@pytest.mark.parametrize('order', (0, -1))
def test_cyclic_operator_instantiation_invalid(order):
    with pytest.raises(ValueError):
        operator = Dihedral(order)
