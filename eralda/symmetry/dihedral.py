import numpy as np
import einops
from .cyclic import derive_rotation_matrices as derive_cyclic_rotation_matrices


def derive_rotation_matrices(order: int) -> np.ndarray:
    cyclic_matrices = derive_cyclic_rotation_matrices(order)
    rotation_matrices = einops.repeat(
        np.eye(3),
        'i j -> 2 new_axis i j',
        new_axis=order
    )
    rotation_matrices[..., :, :] = cyclic_matrices
    rotation_matrices[1, :, :, 1:] *= -1

    rotation_matrices = einops.rearrange(
        rotation_matrices,
        'd n i j -> (d n) i j'
    )
    return rotation_matrices