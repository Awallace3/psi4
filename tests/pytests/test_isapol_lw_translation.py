"""LW translation substitution: literal fixtures from our permitted LGPL kernel.

No old repository or reference executable is accessed by these tests. The numeric
matrix fixture was produced during development from our own old C++ code, not
ORIENT. Production continues using only isa_multipole_translation.
"""
import numpy as np
from psi4 import core

#: Worst of the 13 development fixture cases (largest deviation from the old C++
#: matrix, 4.4e-16).  The full sweep lives in `agent_scratch/`; this displacement
#: and the elements below are the part of it kept under version control.
WORST_DISPLACEMENT = [0.5292945859739524, 0.9176548842614971, -0.37938272178283206]

#: Column 0 of that case is the physically meaningful selection: translating a
#: unit charge writes the regular solid harmonics R_lm(-d) into every component,
#: so these 16 numbers exercise all four ranks at once.  Literals are the old
#: C++ reference values, which agree with the fixture to 4.4e-16.
WORST_CHARGE_COLUMN = [
    1.0,                                            # 00
    -0.37938272178283206, 0.5292945859739524, 0.9176548842614971,     # 1m
    -0.4171903730878099, -0.34780484459384997, -0.6030003383164476,
    -0.4866523476782863, 0.8412739855718974,                          # 2m
    0.5840345161670512, -0.1771404535123017, -0.30711404702306916,
    0.4128396231286102, -0.7136742210498999, -0.9398751435667386,
    -0.0011841194803787836,                                           # 3m
]

#: The single element where our translation and the old C++ one disagree most
#: over all 13 cases: the 2c <- 1s transport coefficient.
WORST_ELEMENT = ((7, 3), -1.5894248833546503)


def test_lw_translation_worst_fixture_case_selected_elements():
    """The retained elements of the fixture's worst case, not the full matrix."""
    matrix = core.isa_multipole_translation(3, WORST_DISPLACEMENT).to_array()
    assert matrix.shape == (16, 16)
    np.testing.assert_allclose(matrix[:, 0], WORST_CHARGE_COLUMN, atol=5e-15, rtol=0)
    (row, col), value = WORST_ELEMENT
    np.testing.assert_allclose(matrix[row, col], value, atol=5e-15, rtol=0)
    # Structure the full comparison also pinned: unit diagonal, no back-transport
    # of higher ranks into the charge.
    np.testing.assert_allclose(np.diag(matrix), np.ones(16), atol=0, rtol=0)
    np.testing.assert_array_equal(matrix[0, 1:], np.zeros(15))


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
