import einops
import numpy as np
from pydantic import BaseModel

from ..utils.typing import Array


class Pose(BaseModel):
    """Pose object modelling a set of poses in 3D

    Attributes
    ----------
    positions : (n, 3) np.ndarray
        Positions in 3D
    orientations : (n, 3, 3) np.ndarray
        Orientations in 3D described as rotation matrices which premultiply
        column vectors (v -> v' == Rv = v')
    """
    positions: Array[float, (-1, 3)]
    orientations: Array[float, (-1, 3, 3)]

    @property
    def count(self):
        return self.positions.shape[0]

    @property
    def positions_homogeneous(self):
        """Positions expressed in homogeneous coordinates"""
        target_shape = (self.count, 4, 1)
        result = np.ones(shape=target_shape, dtype=float)
        result[:, :-1, :] = self.positions
        return result

    @property
    def matrix(self):
        """Pose expressed as an (n, 4, 4) set of matrices.
        """
        result = einops.repeat(
            np.eye(4, dtype=float), 'i j -> new_axis i j', new_axis=self.count
        )
        result[:, :3, :3] = self.orientations
        result[:, :3, 3] = self.positions
        return result


class Transform(BaseModel):
    """Transform object modelling a set of transforms in 3D

    Attributes
    ----------
    shifts : (n, 3) np.ndarray
        Shifts in 3D
    rotations : (n, 3, 3) np.ndarray
        Rotations in 3D described as rotation matrices which premultiply
        column vectors ( v -> v' | Rv == v' )
    """
    shifts: Array[float, (-1, 3)]
    rotations: Array[float, (-1, 3, 3)]

    @property
    def count(self):
        return self.shifts.shape[0]

    @property
    def matrix(self):
        """Transformations expressed as an (m, 4, 4) set of matrices
        """
        result = einops.repeat(
            np.eye(4, dtype=float), 'i j -> new_axis i j', new_axis=self.count
        )
        result[:, :3, :3] = self.rotations
        result[:, :3, 3] = self.shifts
        return result

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
        # rearrange transformation matrix for desired broadcasting
        # pose matrices (n, 4, 4)
        # transformation matrices (m, 4, 4)
        # transformed matrices (m, n, 4, 4)
        transformation_matrix = einops.rearrange(
            self.matrix, 'm i j -> m 1 i j'
        )

        # apply transformation
        transformed_matrix = transformation_matrix @ pose.matrix

        # decompose transformed matrix into final positions and orientations
        transformed_positions = transformed_matrix[:, :, :3, 3]
        transformed_orientations = transformed_matrix[:, :, :3, :3]

        return transformed_positions, transformed_orientations

