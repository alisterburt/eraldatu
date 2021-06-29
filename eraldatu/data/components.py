import einops
import numpy as np
from pydantic import BaseModel

from ..utils.typing import Array


class Pose(BaseModel):
    """Pose object modelling a set of poses in 3D

    Attributes
    ----------
    positions : (n, 3, 1) np.ndarray
        Positions in 3D represented as column vectors
    orientations : (n, 3, 3) np.ndarray
        Orientations in 3D described as rotation matrices which premultiply
        column vectors (v -> v' == Rv = v')
    """
    positions: Array[float, (-1, 3, 1)]
    orientations: Array[float, (-1, 3, 3)]

    @property
    def count(self):
        return self.positions.shape[0]


class Transform(BaseModel):
    """Transform object modelling a set of transforms in 3D

    Attributes
    ----------
    shifts : (n, 3, 1) np.ndarray
        Shifts in 3D
    rotations : (n, 3, 3) np.ndarray
        Rotations in 3D described as rotation matrices which premultiply
        column vectors ( v -> v' | Rv == v' )
    """
    shifts: Array[float, (-1, 3, 1)]
    rotations: Array[float, (-1, 3, 3)]

    @property
    def count(self):
        return self.shifts.shape[0]

    def apply(self, pose: Pose) -> tuple[Array, Array]:
        """Apply transformations on a set of poses

        Parameters
        ----------
        pose: Pose
            A set of poses on which transforms should be applied

        Returns
        -------
        transformed_poses: (transformed_positions, transformed_orientations)
            Transformed poses as a tuple of (m, n, 3) positions and
            (m, n, 3, 3) orientations where n is the number of poses and m is
            the number of transforms
        """
        # transformation rotations     (m, 3, 3)
        # broadcastable             (m, 1, 3, 3)
        # pose orientations            (n, 3, 3)
        # final rotations           (m, n, 3, 3)
        broadcastable_rotations = einops.rearrange(
            self.rotations, 'm i j -> m 1 i j'
        )
        final_rotations = broadcastable_rotations @ pose.orientations

        # shifts                       (m, 3, 1)
        # broadcastable             (m, 1, 3, 1)
        # pose orientations            (n, 3, 3)
        # oriented                  (m, n, 3, 1)
        # pose positions               (n, 3, 1)
        # final positions           (m, n, 3, 1)
        broadcastable_shifts = einops.rearrange(
            self.shifts, 'm i j -> m 1 i j'
        )
        oriented_shifts = pose.orientations @ broadcastable_shifts
        final_positions = pose.positions + oriented_shifts

        return final_positions.squeeze(), final_rotations.squeeze()

