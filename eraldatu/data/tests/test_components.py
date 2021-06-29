import einops
import numpy as np
from numpy.testing import assert_array_equal

from ..components import Pose, Transform


def test_pose_instantiation(positions, identity_matrices):
    p = Pose(positions=positions, orientations=identity_matrices)
    assert isinstance(p, Pose)


def test_transform_instantiation(positions, identity_matrices):
    t = Transform(shifts=positions, rotations=identity_matrices)
    assert isinstance(t, Transform)


def test_transform_application(positions,
                               identity_matrices,
                               shifts,
                               rotz90_matrices):
    pose = Pose(positions=positions, orientations=identity_matrices)
    transform = Transform(shifts=shifts, rotations=rotz90_matrices)
    transformed_positions, transformed_orientations = transform.apply(pose=pose)

    # check orientations have been applied
    expected_orientations = np.broadcast_to(
        einops.rearrange(rotz90_matrices, 'm i j ->m 1 i j'), (5, 200, 3, 3)
    )
    assert_array_equal(transformed_orientations, expected_orientations)

    # check shifts have been applied
    broadcastable_shifts = einops.rearrange(shifts, 'n i -> n 1 i')
    expected_positions = positions + broadcastable_shifts
    assert_array_equal(transformed_positions, expected_positions)
