import re

cyclic_symmetry_regex = re.compile(r'^c\d+$', re.IGNORECASE)
dihedral_symmetry_regex = re.compile(r'^d\d+$', re.IGNORECASE)
symmetric_symmetry_regex = re.compile(r'^s\d+$', re.IGNORECASE)
icosahedral_symmetry_regex = re.compile(r'^ico$', re.IGNORECASE)