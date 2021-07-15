from functools import cached_property
from dataclasses import dataclass

import einops
import numpy as np

from .operator import SymmetryOperator

def derive_rotation_matrices(order: int) -> np.ndarray:
    """Calculate a set of rotation matrices for a cyclic symmetry.

    Axis of rotation is the Z axis.
    Matrices are defined
    Rz = [[cos(t), -sin(t),    0],
          [sin(t),  cos(t),    0],
          [     0,       0,    1]]


    Parameters
    ----------
    order : int
        symmetry order

    Returns
    -------
    rotation_matrices: (order, 3, 3) np.ndarray
        matrices defining rotations around the Z axis
    """
    theta = np.linspace(0, 2 * np.pi, num=order, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrices = einops.repeat(
        np.eye(3),
        'i j -> new_axis i j',
        new_axis=order
    )
    rotation_matrices[:, 0, 0] = cos_theta
    rotation_matrices[:, 0, 1] = -sin_theta
    rotation_matrices[:, 1, 0] = sin_theta
    rotation_matrices[:, 1, 1] = cos_theta
    return rotation_matrices


@dataclass
class Cyclic(SymmetryOperator):
    order: int

    def __post_init__(self):
        self.order = int(self.order)

        if self.order < 1:
            raise ValueError('`order` must be 1 order greater')

    @cached_property
    def matrices(self):
        return derive_rotation_matrices(self.order)
