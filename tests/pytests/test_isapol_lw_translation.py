"""LW translation substitution: literal fixtures from our permitted LGPL kernel.

No old repository or reference executable is accessed by these tests. The numeric
matrix fixture was produced during development from our own old C++ code, not
ORIENT. Production continues using only isa_multipole_translation.
"""
import json
from pathlib import Path
import numpy as np
from psi4 import core


def test_lw_old_cpp_translation_matrix_equivalence():
    fixture = json.loads((Path(__file__).parent/'data_isapol/lw_translation_reference.json').read_text())
    assert len(fixture['cases']) >= 10
    for case in fixture['cases']:
        actual = core.isa_multipole_translation(3, case['displacement']).to_array()
        np.testing.assert_allclose(actual, case['expected'], atol=5e-12, rtol=0)


def test_lw_rank3_translation_matches_arbitrary_displacement_fixtures():
    displacement = [0.2, -0.3, 0.4]
    expected_scalar = [2.0, 0.8, 0.4, -0.6, 0.19, 0.277128129211, -0.415692193817,
        -0.0866025403784, -0.207846096908, -0.028, 0.124923976882, -0.187385965323,
        -0.0774596669241, -0.185903200618, -0.0727323861839, -0.0142302494708]
    expected_dense = [0.1, 0.24, 0.32, 0.37, 0.7295, 0.890984535672, 0.852420471066,
        1.10743901834, 0.872287187079, 1.88348457268, 2.29457302188, 1.30348504869,
        3.00906108622, 2.04427885923, 2.75223325142, 1.30215605855]
    matrix = core.isa_multipole_translation(3, displacement).to_array()
    np.testing.assert_allclose(matrix @ np.array([2.0]+[0.]*15), expected_scalar, atol=5e-12, rtol=0)
    np.testing.assert_allclose(matrix @ np.array([.1*i for i in range(1,17)]), expected_dense, atol=5e-11, rtol=0)
