from __future__ import annotations

import einops
import numpy as np
from pydantic import BaseModel, validator, ValidationError

from .operators import SymmetryOperator
from ..components.data_model import Transform
from ..utils.typing import Array


class Symmetry(BaseModel):
    operator: SymmetryOperator
    basis: Array[float, (3, 3)]

    class Config:
        arbitrary_types_allowed = True

    @validator('basis')
    def is_unimodular(cls, value):
        determinant = np.linalg.det(value)
        if determinant != 1:
            raise ValidationError('basis is not unimodular')
        return value

    @property
    def oriented_operator_matrices(self):
        return self.basis @ self.operator.matrices

    def apply_on_transform(self, transform: Transform) -> Transform:
        """Apply symmetry on a transform to generate symmetrized transforms
        """
        broadcastable_operator_matrices = einops.rearrange(
            self.oriented_operator_matrices, 'm i j -> m 1 i j',
        )
        symmetry_expanded_transform_shifts = (
                broadcastable_operator_matrices @ transform.shifts
        )
        symmetry_expanded_transform_rotations = (
                broadcastable_operator_matrices @ transform.rotations
        )
        return Transform(shifts=symmetry_expanded_transform_shifts,
                         rotations=symmetry_expanded_transform_rotations)
