# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
# LW math tests ported from our LGPL atomic_polarizability_math tests.
# No runtime reference-tree reads; graph/translation guard tests remain separate.
import math
from pathlib import Path

import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]


def _residual_values(residuals):
    return tuple(getattr(residuals, name) for name in
                 ("off_site", "charge_sum", "reciprocity", "molecular_sum", "local_charge"))


def _transfer_values(transfers):
    return [(t.first, t.second, t.first_component, t.second_component, t.fixed_site, t.amount)
            for t in transfers]


def _matrix(values):
    matrix = psi4.core.Matrix(len(values), len(values[0]))
    for row, entries in enumerate(values):
        for column, value in enumerate(entries):
            matrix.set(row, column, value)
    return matrix


def _working_l3_matrix():
    return [[0.0] * 16 for _ in range(16)]


def _lw_localize(positions, values, bonds, tolerance=1.0e-9, frequency=0.0):
    # Legacy synthetic helper tolerances are deliberately retained, not API defaults.
    return psi4.core.isa_localize_lw(
        _matrix(positions), [_matrix(block) for block in values], frequency, bonds, tolerance
    )


def test_lw_localization_requires_explicit_finite_frequency_identity():
    with pytest.raises(Exception, match="frequency must be finite"):
        _lw_localize([[0.0, 0.0, 0.0]], [_working_l3_matrix()], [], frequency=float("nan"))


def test_lw_two_site_charge_flow_localizes_to_full_rank3_fixture():
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0], values[1][0][0] = -2.0, 2.0
    values[2][0][0], values[3][0][0] = 2.0, -2.0
    result = _lw_localize([[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]], values, [(0, 1)])
    local = result.local
    assert len(local) == 2
    # The core storage is rank-4 wide; a rank-3 declaration leaves every component
    # above rank 3 identically zero rather than absent, and the leading 15x15 is
    # bitwise what the rank-3-wide storage produced.
    assert local[0].shape == (24, 24)
    assert all(local[s].get(r, c) == 0.0 for s in (0, 1)
               for r in range(24) for c in range(24) if r >= 15 or c >= 15)
    assert local[0].get(0, 0) == pytest.approx(-0.16, abs=1.0e-11)
    assert local[0].get(0, 1) == pytest.approx(-0.08, abs=1.0e-11)
    assert local[0].get(0, 2) == pytest.approx(0.12, abs=1.0e-11)
    assert local[0].get(0, 3) == pytest.approx(-0.038, abs=1.0e-11)
    assert local[1].get(0, 3) == pytest.approx(0.038, abs=1.0e-11)
    assert local[0].get(14, 14) == pytest.approx(-0.000050625, abs=1.0e-12)
    assert local[1].get(14, 14) == pytest.approx(-0.000050625, abs=1.0e-12)
    assert max(_residual_values(result.residuals)) < 1.0e-10


def test_lw_three_site_preserves_sum_reciprocity_and_transfers_only_on_bonds():
    positions = [[0.0, 0.0, 0.0], [0.7, -0.2, 0.1], [1.1, 0.4, -0.3]]
    graph_operator = [[-1.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -1.0]]
    values = [_working_l3_matrix() for _ in range(9)]
    for a in range(3):
        for b in range(3):
            values[3 * a + b][0][0] = 1.7 * graph_operator[a][b]
    result = _lw_localize(positions, values, [(0, 1), (1, 2)], 2.0e-9)
    assert max(_residual_values(result.residuals)) < 2.0e-9
    assert all(
        (first, second) in {(0, 1), (1, 2)}
        for first, second, _mu, _nu, _site, _amount in _transfer_values(result.transfers)
    )
    for matrix in result.local:
        assert all(
            matrix.get(row, column) == pytest.approx(matrix.get(column, row), abs=2.0e-9)
            for row in range(15) for column in range(15)
        )


def test_lw_rejects_postcondition_residual():
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0], values[1][0][0] = -2.0 + 1.0e-5, 2.0
    values[2][0][0], values[3][0][0] = 2.0, -2.0
    with pytest.raises(RuntimeError, match=r"residual tolerance"):
        _lw_localize([[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]], values, [(0, 1)], 1.0e-9)


def _matrix_values(matrix):
    rows, columns = matrix.shape
    return [[matrix.get(row, column) for column in range(columns)] for row in range(rows)]


def _local_rank3_values(matrix):
    """Leading 15x15 of a 24-wide local block, after asserting the rest is zero.

    A rank-3 declaration has no rank-4 components; in the rank-4-wide storage they
    are identically zero rather than absent, and these rank-3 oracles compare
    against exactly what the 15-wide storage produced, bitwise.
    """
    values = _matrix_values(matrix)
    assert len(values) == len(values[0]) == 24
    assert all(values[row][column] == 0.0 for row in range(24) for column in range(24)
               if row >= 15 or column >= 15)
    return [row[:15] for row in values[:15]]


def _matmul(first, second):
    return [
        [sum(first[row][k] * second[k][column] for k in range(len(second)))
         for column in range(len(second[0]))]
        for row in range(len(first))
    ]


def _transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def _assert_matrix_close(actual, expected, tolerance):
    assert len(actual) == len(expected)
    assert all(actual[row] == pytest.approx(expected[row], abs=tolerance)
               for row in range(len(expected)))


def _regular_harmonics(displacement):
    x, y, z = displacement
    rho2 = x * x + y * y + z * z
    return [
        1.0, z, x, y,
        (3 * z * z - rho2) / 2, math.sqrt(3) * x * z, math.sqrt(3) * y * z,
        math.sqrt(3) * (x * x - y * y) / 2, math.sqrt(3) * x * y,
        (5 * z**3 - 3 * z * rho2) / 2,
        math.sqrt(3 / 8) * x * (5 * z * z - rho2),
        math.sqrt(3 / 8) * y * (5 * z * z - rho2),
        math.sqrt(15) * z * (x * x - y * y) / 2, math.sqrt(15) * x * y * z,
        math.sqrt(10) * x * (x * x - 3 * y * y) / 4,
        math.sqrt(10) * y * (3 * x * x - y * y) / 4,
    ]


def _translation_matrix(displacement):
    return _matrix_values(psi4.core.isa_multipole_translation(3, displacement))


def _common_origin_response(positions, blocks, origin):
    translations = [
        _translation_matrix([coordinate - shift for coordinate, shift in zip(position, origin)])
        for position in positions
    ]
    result = [[0.0] * 16 for _ in range(16)]
    count = len(positions)
    for a in range(count):
        for b in range(count):
            contribution = _matmul(_matmul(translations[a], blocks[a * count + b]),
                                   _transpose(translations[b]))
            for row in range(16):
                for column in range(16):
                    result[row][column] += contribution[row][column]
    return result


def _assert_refined_invariants(result, positions, original, tolerance):
    # The workspace is rank-4 wide (25) while these are rank-3 declarations, so
    # every component above rank 3 must be identically zero -- not small -- and the
    # leading 16 must be exactly what the rank-3-wide storage produced. The oracle
    # below is a rank-3 oracle and reads only that leading block.
    stored = [_matrix_values(block) for block in result.refined_pairs]
    assert all(len(block) == len(block[0]) == 25 for block in stored)
    assert all(block[row][column] == 0.0 for block in stored
               for row in range(25) for column in range(25) if row >= 16 or column >= 16)
    refined = [[row[:16] for row in block[:16]] for block in stored]
    count = len(positions)
    assert max(abs(refined[a * count + b][row][column])
               for a in range(count) for b in range(count) if a != b
               for row in range(16) for column in range(16)) <= tolerance
    assert max(abs(refined[a * count + b][row][column] -
                   refined[b * count + a][column][row])
               for a in range(count) for b in range(count)
               for row in range(16) for column in range(16)) <= tolerance
    assert max(abs(sum(refined[a * count + b][component][0] for b in range(count)))
               for a in range(count) for component in range(16)) <= tolerance
    assert max(abs(sum(refined[b * count + a][0][component] for b in range(count)))
               for a in range(count) for component in range(16)) <= tolerance
    for origin in ([0.3, -0.2, 0.5], [-0.4, 0.6, -0.1]):
        before = _common_origin_response(positions, original, origin)
        after = _common_origin_response(positions, refined, origin)
        assert max(abs(before[row][column] - after[row][column])
                   for row in range(16) for column in range(16)) <= tolerance


def test_lw_refined_workspace_matches_full_two_site_oracle_and_reversed_edge():
    positions = [[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]]
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0], values[1][0][0] = -2.0, 2.0
    values[2][0][0], values[3][0][0] = 2.0, -2.0
    tail = _regular_harmonics(positions[1])[1:]
    negative_tail = [(-1 if rank in (1, 3) else 1) * value
                     for rank, value in zip([1] * 3 + [2] * 5 + [3] * 7, tail)]
    expected = [
        [[-left * right for right in tail] for left in tail],
        [[-left * right for right in negative_tail] for left in negative_tail],
    ]
    for bonds in ([(0, 1)], [(1, 0)]):
        result = _lw_localize(positions, values, bonds)
        for site in range(2):
            _assert_matrix_close(_local_rank3_values(result.local[site]), expected[site], 2.0e-11)
        _assert_refined_invariants(result, positions, values, 2.0e-10)


def test_lw_co_axis_aligned_charge_flow_has_closed_local_cartesian_oracle():
    """A C--O diatomic on +z needs no local/molecular frame conversion.

    For a two-site tree, a scalar charge-flow response q localizes to
    -q/2 R_l(d) R_l'(d) at C. At O, translating along -d adds (-1)^(l+l').
    Along z only m=0 survives, making every rank-1-through-3 entry a closed monomial.
    """
    bond_length = 2.132  # approximately the experimental C--O distance, bohr
    charge_flow = 0.4
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]]
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0] = values[3][0][0] = -charge_flow
    values[1][0][0] = values[2][0][0] = charge_flow

    result = _lw_localize(positions, values, [(0, 1)])
    carbon_tail = np.zeros(15)
    oxygen_tail = np.zeros(15)
    for rank, component in ((1, 0), (2, 3), (3, 8)):
        carbon_tail[component] = bond_length**rank
        oxygen_tail[component] = (-bond_length)**rank
    expected = [
        -0.5 * charge_flow * np.outer(carbon_tail, carbon_tail),
        -0.5 * charge_flow * np.outer(oxygen_tail, oxygen_tail),
    ]

    for site in range(2):
        _assert_matrix_close(_local_rank3_values(result.local[site]), expected[site], 3.0e-11)
    assert result.local[0].get(0, 3) == pytest.approx(
        -0.5 * charge_flow * bond_length**3, abs=3.0e-11)
    assert result.local[1].get(0, 3) == pytest.approx(
        0.5 * charge_flow * bond_length**3, abs=3.0e-11)
    assert all(result.local[0].get(index, index) == pytest.approx(
        result.local[1].get(index, index), abs=3.0e-11)
               for index in (0, 3, 8))
    _assert_refined_invariants(result, positions, values, 3.0e-10)


def test_lw_noncharge_seed_has_independent_reciprocity_sum_and_origin_oracles():
    positions = [[0.1, -0.2, 0.3], [0.8, 0.1, -0.4]]
    values = [_working_l3_matrix() for _ in range(4)]
    values[1][1][2] = 0.35
    values[2][2][1] = 0.35
    result = _lw_localize(positions, values, [(1, 0)], 1.0e-8)
    _assert_refined_invariants(result, positions, values, 2.0e-9)


def test_lw_disconnected_components_accept_zero_and_reject_inconsistent_flow():
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    zero = [_working_l3_matrix() for _ in range(9)]
    result = _lw_localize(positions, zero, [(0, 1)])
    assert len(result.local) == 3
    assert sum(abs(value) < 1.0e-12 for value in
               psi4.core.isa_lw_graph_math(3, [(0, 1)])[2]) == 2

    component_positions = [[0.0, 0.0, 0.0], [0.4, 0.1, -0.2],
                           [2.0, -0.3, 0.2], [2.2, 0.5, 0.4]]
    component_values = [_working_l3_matrix() for _ in range(16)]
    for first, second, scale in ((0, 1, 1.2), (2, 3, 0.8)):
        component_values[4 * first + first][0][0] = -scale
        component_values[4 * first + second][0][0] = scale
        component_values[4 * second + first][0][0] = scale
        component_values[4 * second + second][0][0] = -scale
    component_result = _lw_localize(
        component_positions, component_values, [(0, 1), (2, 3)], 3.0e-7
    )
    _assert_refined_invariants(
        component_result, component_positions, component_values, 3.0e-7
    )

    inconsistent = [_working_l3_matrix() for _ in range(9)]
    inconsistent[2][1][1] = 1.0
    inconsistent[6][1][1] = 1.0
    with pytest.raises(RuntimeError, match=r"component.*zero sum|graph solve"):
        _lw_localize(positions, inconsistent, [(0, 1)])


def test_lw_historical_omission_threshold_boundaries_and_diagnostics():
    def localize_with_amplitude(amplitude):
        values = [_working_l3_matrix() for _ in range(4)]
        values[1][1][1] = amplitude
        values[2][1][1] = amplitude
        return _lw_localize([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], values, [(0, 1)], 1.0e-5)

    candidate_below = localize_with_amplitude(math.nextafter(1.0e-7, 0.0))
    candidate_equal = localize_with_amplitude(1.0e-7)
    transfer_equal = localize_with_amplitude(4.0e-7)
    transfer_above = localize_with_amplitude(4.1e-7)
    assert (1, 1) in map(tuple, candidate_below.omitted_component_pairs)
    assert (1, 1) not in map(tuple, candidate_equal.omitted_component_pairs)
    assert candidate_equal.omitted_transfer_count > candidate_below.omitted_transfer_count
    assert not any(transfer[2:4] == (1, 1) for transfer in _transfer_values(candidate_equal.transfers))
    assert not any(transfer[2:4] == (1, 1) for transfer in _transfer_values(transfer_equal.transfers))
    assert any(transfer[2:4] == (1, 1) for transfer in _transfer_values(transfer_above.transfers))
    assert transfer_above.refined_pairs[1].get(1, 1) == pytest.approx(0.0, abs=1.0e-12)


def test_lw_finite_inputs_that_overflow_derived_math_fail_closed():
    zero = [_working_l3_matrix() for _ in range(4)]
    # The shared translation kernel raises ValueError for nonfinite transforms;
    # native LW arithmetic raises RuntimeError. Both must fail closed.
    with pytest.raises((RuntimeError, ValueError), match=r"finite|overflow"):
        _lw_localize([[0.0, 0.0, 0.0], [1.0e308, 0.0, 0.0]], zero, [(0, 1)])

    large = [_working_l3_matrix() for _ in range(4)]
    large[1][0][0] = 1.0e308
    large[2][0][0] = 1.0e308
    large[0][0][0] = -1.0e308
    large[3][0][0] = -1.0e308
    with pytest.raises(RuntimeError, match=r"finite|overflow"):
        _lw_localize([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], large, [(0, 1)], 1.0e300)



def test_lw_supplemental_0375_block_with_local_rank2_seed():
    # Supplemental synthetic oracle (not present in the old math test file).
    # q=-1 along +z gives +1/2 R_l R_l' on site 0; a pre-localized
    # rank-2 seed contributes -1/8 to 10,20 and -1/4 to 20,20.
    # Thus the complete 10/20 block is [[1/2, 3/8], [3/8, 1/4]].
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0] = values[3][0][0] = 1.0
    values[1][0][0] = values[2][0][0] = -1.0
    values[0][1][4] = values[0][4][1] = -0.125
    values[0][4][4] = -0.25
    result = _lw_localize([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], values, [(0, 1)])
    local = result.local[0]
    assert [local.get(i, j) for i in (0, 3) for j in (0, 3)] == pytest.approx(
        [0.5, 0.375, 0.375, 0.25], abs=1.0e-12)
    _assert_refined_invariants(result, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], values, 1.0e-11)


@pytest.mark.parametrize("frequency", [0.0, 0.125, 8.5])
def test_lw_frequency_identity_and_owned_readonly_copies(frequency):
    positions = _matrix([[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]])
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0] = values[3][0][0] = -2.0
    values[1][0][0] = values[2][0][0] = 2.0
    blocks = [_matrix(block) for block in values]
    result = psi4.core.isa_localize_lw(positions, blocks, frequency, [(0, 1)])
    assert isinstance(result, psi4.core.IsaLocalizedResponse)
    assert isinstance(result.residuals, psi4.core.IsaLocalizationResiduals)
    assert isinstance(result.transfers[0], psi4.core.IsaBondTransfer)
    assert result.frequency == frequency
    original_local = result.local[0].get(0, 0)
    original_refined = result.refined_pairs[0].get(1, 1)
    blocks[0].set(1, 1, 999.0)
    positions.set(1, 0, 999.0)
    assert result.positions.get(1, 0) == 0.2
    for name, row, column, expected in (
        ("local", 0, 0, original_local), ("refined_pairs", 1, 1, original_refined)
    ):
        copies = getattr(result, name)
        copies[0].set(row, column, 777.0)
        copies.clear()
        assert getattr(result, name)[0].get(row, column) == expected
        assert getattr(result, name)[0] is not getattr(result, name)[0]
    result.positions.set(0, 0, 123.0)
    assert result.positions.get(0, 0) == 0.0
    transfers = result.transfers
    transfers.clear()
    assert result.transfers
    assert result.transfers[0] is not result.transfers[0]
    omissions = result.omitted_component_pairs
    expected_omissions = result.omitted_component_pairs
    omissions[0][0] = 99
    omissions.clear()
    assert result.omitted_component_pairs == expected_omissions
    assert result.residuals is not result.residuals
    with pytest.raises(AttributeError):
        result.residuals.off_site = 999.0
    with pytest.raises(AttributeError):
        result.transfers[0].amount = 999.0
    for name in ("frequency", "positions", "local", "refined_pairs", "transfers",
                 "residuals", "omitted_component_pairs", "omitted_transfer_count"):
        with pytest.raises(AttributeError):
            setattr(result, name, getattr(result, name))


def test_lw_frequency_is_required_and_default_is_postcondition_gate():
    positions = _matrix([[0.0, 0.0, 0.0]])
    block = _matrix(_working_l3_matrix())
    with pytest.raises(TypeError):
        psi4.core.isa_localize_lw(positions, [block], bonds=[])
    with pytest.raises(TypeError):
        psi4.core.isa_localize_lw(positions, [block], frequency=0.0)
    block.set(0, 0, 5.0e-7)
    result = psi4.core.isa_localize_lw(positions, [block], 0.0, [])
    assert result.residuals.local_charge == 5.0e-7
    with pytest.raises(RuntimeError, match="postcondition.*off-site=.*charge-sum=.*reciprocity=.*molecular-sum=.*local-charge="):
        psi4.core.isa_localize_lw(positions, [block], 0.0, [], 1.0e-9)


@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_lw_invalid_frequency_fails_closed(frequency):
    with pytest.raises(RuntimeError, match="frequency"):
        _lw_localize([[0.0, 0.0, 0.0]], [_working_l3_matrix()], [], frequency=frequency)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_lw_invalid_residual_tolerance(tolerance):
    with pytest.raises(RuntimeError, match="tolerance"):
        _lw_localize([[0.0, 0.0, 0.0]], [_working_l3_matrix()], [], tolerance)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_lw_nonfinite_positions_and_blocks(bad):
    with pytest.raises(RuntimeError, match="finite"):
        _lw_localize([[bad, 0.0, 0.0]], [_working_l3_matrix()], [])
    block = _working_l3_matrix()
    block[15][15] = bad
    with pytest.raises(RuntimeError, match="finite"):
        _lw_localize([[0.0, 0.0, 0.0]], [block], [])


@pytest.mark.parametrize("rows,cols", [(0, 3), (1, 2), (1, 4), (257, 3)])
def test_lw_position_dimensions_and_site_cap_precede_conversion(rows, cols):
    with pytest.raises(RuntimeError, match="N by 3|1..256|resource"):
        psi4.core.isa_localize_lw(psi4.core.Matrix(rows, cols), [], 0.0, [])


@pytest.mark.parametrize("blocks", [[], [None], ["not a Matrix"]])
def test_lw_block_collection_rejects_invalid_structure(blocks):
    with pytest.raises((RuntimeError, TypeError)):
        psi4.core.isa_localize_lw(_matrix([[0.0, 0.0, 0.0]]), blocks, 0.0, [])


@pytest.mark.parametrize("rows,cols", [(15, 15), (16, 15), (15, 16), (17, 17),
                                       (24, 24), (25, 24), (24, 25), (26, 26)])
def test_lw_block_dimensions(rows, cols):
    with pytest.raises(RuntimeError, match="16 by 16 or 25 by 25"):
        psi4.core.isa_localize_lw(_matrix([[0.0, 0.0, 0.0]]),
                                  [psi4.core.Matrix(rows, cols)], 0.0, [])


def test_lw_oversized_python_sequences_rejected_without_element_conversion():
    class NeverRead:
        def __len__(self):
            return 10**9

        def __getitem__(self, index):
            raise AssertionError("resource guard must precede element conversion")

    positions = _matrix([[0.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="capacity"):
        psi4.core.isa_localize_lw(positions, [], 0.0, NeverRead())
    with pytest.raises(RuntimeError, match="ordered site pair"):
        psi4.core.isa_localize_lw(positions, NeverRead(), 0.0, [])


def test_lw_rejects_symmetry_blocked_matrices():
    blocked = psi4.core.Matrix.from_array([np.zeros((8, 8)), np.zeros((8, 8))])
    with pytest.raises(RuntimeError, match="single-block"):
        psi4.core.isa_localize_lw(_matrix([[0.0, 0.0, 0.0]]), [blocked], 0.0, [])
    positions = psi4.core.Matrix.from_array([np.zeros((1, 3)), np.zeros((1, 3))])
    with pytest.raises(RuntimeError, match="single-block"):
        psi4.core.isa_localize_lw(positions, [], 0.0, [])


@pytest.mark.parametrize("bonds", [[(0, 0)], [(0, 3)], [(0, 1), (1, 0)], [(0, 1)] * 4])
def test_lw_invalid_graph_before_response_allocation(bonds):
    # These are localizer boundary checks, not duplicated graph algebra tests.
    with pytest.raises(RuntimeError, match="bond|duplicate|capacity"):
        _lw_localize([[0.0, 0.0, 0.0]] * 3, [_working_l3_matrix() for _ in range(9)], bonds)


def test_lw_input_reciprocity_rejected():
    values = [_working_l3_matrix() for _ in range(4)]
    values[1][1][2] = 1.0
    with pytest.raises(RuntimeError, match="input reciprocity"):
        _lw_localize([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], values, [(0, 1)])


def test_lw_reversed_edge_keeps_canonical_transfer_amounts():
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0] = values[3][0][0] = -2.0
    values[1][0][0] = values[2][0][0] = 2.0
    positions = [[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]]
    first = _lw_localize(positions, values, [(0, 1)])
    reversed_edge = _lw_localize(positions, values, [(1, 0)])
    a, b = _transfer_values(first.transfers), _transfer_values(reversed_edge.transfers)
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert left[:5] == right[:5]
        assert left[5] == pytest.approx(right[5], abs=1.0e-12)
    for left, right in zip(first.refined_pairs, reversed_edge.refined_pairs):
        _assert_matrix_close(_matrix_values(left), _matrix_values(right), 2.0e-11)


def test_lw_source_is_pod_only_and_resource_bounded():
    root = Path(__file__).resolve().parents[2]
    source = (root / "psi4/src/psi4/libisapol/lw_localization.cc").read_text()
    header = (root / "psi4/src/psi4/libisapol/lw_localization.h").read_text()
    for forbidden in ("Wavefunction", "scf::HF", "BasisSet", "DirectJK", "libscf_solver",
                      "partitioned_response", "IsaPartitionedResponse", "IsaResponseOptions",
                      "IsaGrid", "IsaFit", "IsaPfit", "SCF", "real_to_complex", "binomial",
                      "regular_harmonics", "complex_index"):
        assert forbidden not in source
    assert source.count("isa_multipole_translation(static_cast<int>(kIsaLwMaxRank), displacement)") == 1
    assert "translation_matrix(position)" in source  # molecular origin shifts use the same seam
    assert "kElementTransferThreshold = 1.0e-7" in source
    assert "largest_candidate < kElementTransferThreshold" in source
    assert "std::abs(amount) <= kElementTransferThreshold" in source
    assert "kIsaLwGraphMaxSites = 256" in header
    assert "kIsaLwMaxRank = 4" in header
    assert "kIsaLwMaxTransfers = 1000000" in header
    assert "kIsaLwMaxWorkspaceBytes = 768 * 1024 * 1024" in header
    assert "result.transfers.size() + pending.size() >= kIsaLwMaxTransfers" in source
    assert source.index("isa_lw_validate_workspace(count, graph.bonds.size());") < source.index(
        "IsaSitePairResponse truncated = response;") < source.index(
        "IsaSitePairResponse refined = truncated;")


def _random_reciprocal_blocks(count, seed, width=16):
    """Full reciprocal ordered-pair input: blocks[a*n+b][t][u] == blocks[b*n+a][u][t].

    `width` DECLARES the rank of the generated data, 16 for rank 3 and 25 for
    rank 4; it is not a storage choice. The rank-limit tests need input that is
    nonzero in EVERY component, including the ones a declared limit discards, or
    the truncation would be a no-op and the commutation claim would go untested.
    """
    generator = np.random.default_rng(seed)
    raw = generator.normal(size=(count, count, width, width))
    blocks = [[[0.0] * width for _ in range(width)] for _ in range(count * count)]
    for a in range(count):
        for b in range(count):
            for row in range(width):
                for column in range(width):
                    blocks[a * count + b][row][column] = 0.5 * (raw[a, b, row, column] +
                                                                raw[b, a, column, row])
    return blocks


def _limited_localize(positions, values, bonds, rank_limit, tolerance=1.0e-9):
    # The supplied blocks generally break the charge-flow sum rule, which is a
    # property of the caller's data and not of the limit under test, so it is
    # measured and reported (infinity) rather than gated. The algorithm-controlled
    # residuals stay on the same `tolerance` gate at every limit.
    return psi4.core.isa_localize_lw(
        _matrix(positions), [_matrix(block) for block in values], 0.0, bonds, tolerance,
        float("inf"), rank_limit,
    )


def test_lw_declared_rank_limit_defaults_to_three_and_leaves_the_call_unchanged():
    positions = [[0.0, 0.0, 0.0], [0.2, -0.3, 0.4], [-0.5, 0.1, 0.2]]
    values = _random_reciprocal_blocks(3, 20260910)
    bonds = [(0, 1), (1, 2)]
    unlimited = _limited_localize(positions, values, bonds, 3)
    for explicit in (3, None):
        result = (unlimited if explicit is None else
                  _limited_localize(positions, values, bonds, explicit))
        assert result.localization_rank_limit == 3
        # Nothing was discarded at the full working space, so this is exactly 0,
        # not merely small: it is a report on the declaration, not a residual.
        assert result.truncated_input_maxabs == 0.0
        for site in range(3):
            assert (np.asarray(result.local[site]) ==
                    np.asarray(unlimited.local[site])).all()


@pytest.mark.parametrize("rank_limit", [1, 2])
def test_lw_declared_rank_limit_equals_rank_three_restricted_bitwise(rank_limit):
    """A declared limit L gives the rank-3 result restricted to (L+1)^2, exactly.

    This is a theorem about the algorithm, not a numerical observation. The pair
    loop is ordered first_component <= second_component; a transfer for (t, u)
    writes only into slot u with target weight delta(target, t) + T(+-d)[target][t],
    and translation is rank-raising, so that weight vanishes for rank(target) <
    rank(t). A pair whose t lies above the limit therefore writes only above it,
    and t <= u puts u above it too, while the screening decisions for the pairs
    below read only components below. Hence `abs=0.0`, not a tolerance.

    The consequence is negative and is the point of the test: a declared
    localization rank limit cannot change any rank <= L number, so it cannot
    explain a disagreement in one.
    """
    positions = [[0.0, 0.0, 0.0], [0.2, -0.3, 0.4], [-0.5, 0.1, 0.2], [0.7, 0.4, -0.6]]
    values = _random_reciprocal_blocks(4, 4090 + rank_limit)
    bonds = [(0, 1), (1, 2), (2, 3)]
    full = _limited_localize(positions, values, bonds, 3)
    limited = _limited_localize(positions, values, bonds, rank_limit)
    assert limited.localization_rank_limit == rank_limit
    # local output has the 00 component removed, so rank <= L occupies [0, (L+1)^2 - 1).
    width = (rank_limit + 1) ** 2 - 1
    scale = max(abs(np.asarray(full.local[site])).max() for site in range(4))
    assert scale > 1.0e-3  # the comparison would be vacuous on a null result
    for site in range(4):
        inside_full = np.asarray(full.local[site])[:width, :width]
        inside_limited = np.asarray(limited.local[site])[:width, :width]
        assert (inside_limited == inside_full).all()
        outside = np.asarray(limited.local[site]).copy()
        outside[:width, :width] = 0.0
        # Above the declared limit the components are identically zero, by
        # declaration rather than by truncation of a computed value.
        assert (outside == 0.0).all()
    # Truncating the input to the declared space by hand changes nothing either.
    prepared = [[[value if row < (rank_limit + 1) ** 2 and column < (rank_limit + 1) ** 2 else 0.0
                  for column, value in enumerate(entries)] for row, entries in enumerate(block)]
                for block in values]
    pretruncated = _limited_localize(positions, prepared, bonds, rank_limit)
    for site in range(4):
        assert (np.asarray(pretruncated.local[site]) == np.asarray(limited.local[site])).all()
    assert pretruncated.truncated_input_maxabs == 0.0
    assert limited.truncated_input_maxabs == pytest.approx(
        max(abs(values[block][row][column])
            for block in range(16) for row in range(16) for column in range(16)
            if row >= (rank_limit + 1) ** 2 or column >= (rank_limit + 1) ** 2), abs=0.0)
    # No tolerance was relaxed to achieve any of this: the same gate holds.
    assert max(_residual_values(limited.residuals)[:1] +
               _residual_values(limited.residuals)[2:4]) <= 1.0e-9


def test_lw_declared_rank_limit_four_is_a_new_model_extending_rank_three_bitwise():
    """Limit 4 localizes the rank-4 rows; limit 3 on the SAME data is its restriction.

    The theorem of the preceding test at L' = 4: translation is rank-raising, so a
    component pair above the declared limit writes only above it. Localizing the
    full rank-4 input at limit 4 and then reading ranks 1..3 is therefore bitwise
    the same computation as truncating to rank 3 first and localizing at limit 3.

    That exact consistency is NOT an agreement claim between the two models. The
    limit-4 result carries a rank-4 local tensor the limit-3 model does not have at
    all, and the two may never be quoted as agreeing on anything but their shared
    leading components.
    """
    positions = [[0.0, 0.0, 0.0], [0.2, -0.3, 0.4], [-0.5, 0.1, 0.2], [0.7, 0.4, -0.6]]
    values = _random_reciprocal_blocks(4, 20260911, width=25)
    bonds = [(0, 1), (1, 2), (2, 3)]
    four = _limited_localize(positions, values, bonds, 4)
    three = _limited_localize(positions, values, bonds, 3)
    assert four.localization_rank_limit == 4
    assert three.localization_rank_limit == 3
    # Limit 4 is the full working space, so it discards nothing; limit 3 discards
    # the caller's real rank-4 data, which is what makes the comparison nonvacuous.
    assert four.truncated_input_maxabs == 0.0
    assert three.truncated_input_maxabs > 1.0e-3
    for site in range(4):
        wide, narrow = np.asarray(four.local[site]), np.asarray(three.local[site])
        assert wide.shape == narrow.shape == (24, 24)
        assert (wide[:15, :15] == narrow[:15, :15]).all()
        # Ranks 1..3 are shared; rank 4 exists only in the limit-4 model, where it
        # is a computed number rather than the limit-3 model's declared zero.
        assert (narrow[15:, :] == 0.0).all() and (narrow[:, 15:] == 0.0).all()
        assert abs(wide[15:, 15:]).max() > 1.0e-3
    # No tolerance was relaxed to reach rank 4: the same gate holds.
    assert max(_residual_values(four.residuals)[:1] +
               _residual_values(four.residuals)[2:4]) <= 1.0e-9


def test_lw_rank_four_cannot_be_declared_on_rank_three_blocks():
    """The supplied width declares the caller's rank; rank 4 is not inferred into.

    A rank-3 caller must not get a rank-4 localization by accident of the widened
    storage: the rank-4 components of 16-wide input do not exist, and zero-filling
    them would declare a model the caller never supplied.
    """
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    narrow = _random_reciprocal_blocks(2, 20260912)
    with pytest.raises(Exception, match="rank 4 localization requires 25 by 25"):
        _limited_localize(positions, narrow, [(0, 1)], 4)
    with pytest.raises(Exception, match="rank_limit must be 1, 2, 3 or 4"):
        _limited_localize(positions, _random_reciprocal_blocks(2, 20260913, width=25), [(0, 1)], 5)
    # Zero-extending by hand to the full width is the caller DECLARING rank 4, and
    # is accepted. Its ranks 1..3 reproduce the rank-3 localization bitwise, but its
    # rank-4 block is NOT zero: translation is rank-raising, so transferring a
    # rank <= 3 source to a displaced site generates genuine rank-4 components.
    # Declaring rank 4 on rank-3 data therefore still yields a different model, not
    # a padded copy of the rank-3 one -- which is why the declaration is required.
    widened = [[[block[row][column] if row < 16 and column < 16 else 0.0
                 for column in range(25)] for row in range(25)] for block in narrow]
    four = _limited_localize(positions, widened, [(0, 1)], 4)
    three = _limited_localize(positions, narrow, [(0, 1)], 3)
    assert four.localization_rank_limit == 4
    assert four.truncated_input_maxabs == 0.0
    for site in range(2):
        wide, narrow_local = np.asarray(four.local[site]), np.asarray(three.local[site])
        assert (wide[:15, :15] == narrow_local[:15, :15]).all()
        assert (narrow_local[15:, :] == 0.0).all()
        assert abs(wide[15:, 15:]).max() > 1.0


def test_lw_declared_rank_limit_reports_and_does_not_gate_discarded_input():
    """`truncated_input_maxabs` reports the caller's own declaration, not a defect.

    The seed is a single rank-1/rank-3 charge-flow element on a diatomic, which is
    the cleanest separator available: at rank 3 it localizes to a large result,
    and at declared limit 2 its pair has second_component outside the declared
    space, so the algorithm never sees it and the surviving space is exactly null.
    """
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    values = [_working_l3_matrix() for _ in range(4)]
    values[1][1][9] = values[2][9][1] = 5.0e3
    for block in (0, 3):
        values[block][1][9] = values[block][9][1] = -5.0e3
    limited = _limited_localize(positions, values, [(0, 1)], 2)
    assert limited.truncated_input_maxabs == pytest.approx(5.0e3, abs=0.0)
    # Discarding 5000 does not fail the call; it is the caller's declaration.
    assert max(abs(np.asarray(limited.local[site])).max() for site in range(2)) == 0.0
    assert not limited.transfers
    full = _limited_localize(positions, values, [(0, 1)], 3)
    assert full.truncated_input_maxabs == 0.0
    assert len(full.transfers) == 2
    # The same seed does act at rank 3: 5000 lands on site 0's own (rank1, rank3)
    # element, and the translation to site 1 raises it by an order of magnitude.
    assert abs(np.asarray(full.local[0])).max() == pytest.approx(5.0e3, abs=0.0)
    assert abs(np.asarray(full.local[1])).max() > 1.0e4
    # The rank <= 2 window of that rank-3 result is itself null, which is the
    # commutation theorem on a case where the two limits are visibly different.
    for site in range(2):
        assert (np.asarray(full.local[site])[:8, :8] == 0.0).all()


@pytest.mark.parametrize("rank_limit", [0, 4, -1, 16])
def test_lw_declared_rank_limit_out_of_range_fails_closed(rank_limit):
    with pytest.raises(Exception, match="rank_limit"):
        _limited_localize([[0.0, 0.0, 0.0]], [_working_l3_matrix()], [], rank_limit)
