from ..regex import cyclic_symmetry_regex
import pytest
import re


@pytest.mark.parametrize('text_to_match, expected',
                         [('c3', re.Match),
                          ('C3', re.Match),
                          ('c232149786', re.Match),
                          ('C21837431', re.Match),
                          ('c', None),
                          ('s', None),
                          ('c3c', None),
                          ('dc3', None)
                          ])
def test_cyclic_symmetry_regex_matches(text_to_match: str, expected: type):
    match = cyclic_symmetry_regex.match(text_to_match)
    if match is not None:
        assert isinstance(match, expected)
    else:
        assert match is None