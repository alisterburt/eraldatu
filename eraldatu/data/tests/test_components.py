import numpy as np
import einops
from numpy.testing import assert_array_equal

from ..components import Pose, Transform


def test_pose_instantiation(positions, identity_matrices):
    p = Pose(positions=positions, orientations=identity_matrices)
    assert isinstance(p, Pose)


def test_pose_homogeneous_coordinates(positions, identity_matrices):
    p = Pose(positions=positions, orientations=identity_matrices)
    assert_array_equal(p.positions_homogeneous[:, 0:3, :], p.positions)
    assert_array_equal(
        p.positions_homogeneous[:, -1, :], np.ones((p.count, 1), dtype=float)
    )


def test_transform_instantiation(positions, identity_matrices):
    t = Transform(shifts=positions, rotations=identity_matrices)
    assert isinstance(t, Transform)


def test_transform_matrix(positions, rotz90_matrices):
    t = Transform(shifts=positions, rotations=rotz90_matrices)
    assert t.matrix.shape == (t.count, 4, 4)
    assert_array_equal(t.matrix[:, :3, :3], t.rotations)
    assert_array_equal(t.matrix[:, :3, 3], t.shifts)
    expected_final_row = np.broadcast_to(
        np.array([0, 0, 0, 1], dtype=float), (t.count, 4)
    )
    assert_array_equal(t.matrix[:, -1, :], expected_final_row)


def test_transform_application(positions,
                               identity_matrices,
                               shifts,
                               rotz90_matrices):
    pose = Pose(positions=positions, orientations=identity_matrices)
    transform = Transform(shifts=shifts, rotations=rotz90_matrices)
    transformed_positions, transformed_orientations = transform.apply(pose=pose)

    expected_orientations = np.broadcast_to(
        einops.rearrange(rotz90_matrices, 'm i j ->m 1 i j'), (5, 200, 3, 3)
    )
    assert_array_equal(transformed_orientations, expected_orientations)

    broadcastable_shifts = einops.rearrange(shifts, 'n i -> n 1 i')
    expected_positions = positions + broadcastable_shifts
    assert_array_equal(transformed_positions, expected_positions)

