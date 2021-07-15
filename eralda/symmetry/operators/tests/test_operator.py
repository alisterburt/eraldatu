from ..operator import SymmetryOperator
import pytest


def test_symmetry_operator_instantiation():
    """"Check that direct instantiation of SymmetryOperator fails.
    """
    with pytest.raises(TypeError):
        s = SymmetryOperator()


def test_symmetry_operator_subclass_instantiation():
    """Check that subclasses of Symmetry operator which implement
    SymmetryOperator.matrices instantiate correctly
    """
    class MyOperator(SymmetryOperator):
        def __init__(self):
            super().__init__()

        @property
        def matrices(self):
            pass

    operator = MyOperator()
    assert isinstance(operator, SymmetryOperator)