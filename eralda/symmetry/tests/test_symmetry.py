from ..symmetry import Symmetry


def test_symmetry_instantiation(c3, identity_basis):
    sym = Symmetry(operator=c3, basis=identity_basis)
    assert isinstance(sym, Symmetry)


def test_symmetry_application(c3, identity_basis, simple_transform):
    sym = Symmetry(operator=c3, basis=identity_basis)
    symmetry_expanded = sym.apply_on_transform(simple_transform)
    4