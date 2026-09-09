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
    assert local[0].shape == (15, 15)
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
    refined = [_matrix_values(block) for block in result.refined_pairs]
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
            _assert_matrix_close(_matrix_values(result.local[site]), expected[site], 2.0e-11)
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
        _assert_matrix_close(_matrix_values(result.local[site]), expected[site], 3.0e-11)
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


@pytest.mark.parametrize("rows,cols", [(15, 15), (16, 15), (15, 16), (17, 17)])
def test_lw_block_dimensions(rows, cols):
    with pytest.raises(RuntimeError, match="16 by 16"):
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
    assert source.count("isa_multipole_translation(3, displacement)") == 1
    assert "translation_matrix(position)" in source  # molecular origin shifts use the same seam
    assert "kElementTransferThreshold = 1.0e-7" in source
    assert "largest_candidate < kElementTransferThreshold" in source
    assert "std::abs(amount) <= kElementTransferThreshold" in source
    assert "kIsaLwGraphMaxSites = 256" in header
    assert "kIsaLwMaxTransfers = 1000000" in header
    assert "kIsaLwMaxWorkspaceBytes = 768 * 1024 * 1024" in header
    assert "result.transfers.size() + pending.size() >= kIsaLwMaxTransfers" in source
    assert source.index("isa_lw_validate_workspace(count, graph.bonds.size());") < source.index(
        "IsaSitePairResponse refined = response;")
