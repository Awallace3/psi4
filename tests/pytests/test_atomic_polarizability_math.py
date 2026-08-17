import math
from pathlib import Path

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

_PUBLIC_ARRAYS = (
    "ATOMIC POLARIZABILITIES",
    "ATOMIC DYNAMIC POLARIZABILITIES",
    "ATOMIC POLARIZABILITY FREQUENCIES",
    "ATOMIC C6",
    "ATOMIC C8",
    "ATOMIC C10",
    "ATOMIC C12",
)

_REVIEWED_FREQUENCIES = (
    0.0,
    0.0066096015960872435,
    0.03617481199863096,
    0.09544736369034827,
    0.1976442118453127,
    0.3704172128053672,
    0.6749146404580301,
    1.264899172436498,
    2.619244684547324,
    6.910885950408292,
    37.82376235021415,
)

_REVIEWED_WEIGHTS = (
    0.0,
    0.01711141976082999,
    0.04296478632573829,
    0.07767872676394183,
    0.13105411733467387,
    0.22389687302154546,
    0.4079488542407823,
    0.8387305806393448,
    2.131641821338567,
    8.208052005690483,
    97.92092081488428,
)


def _matrix(values):
    matrix = psi4.core.Matrix(len(values), len(values[0]))
    for row, entries in enumerate(values):
        for column, value in enumerate(entries):
            matrix.set(row, column, value)
    return matrix


def _as_packed_rows(matrix):
    return [matrix.get(row, column) for row in range(3) for column in range(3)]


def _solve_restricted_response(h1, h2, omega, rhs):
    return psi4.core._atomic_polarizability_solve_restricted_response(
        _matrix(h1), _matrix(h2), omega, _matrix(rhs)
    )


def _assemble_restricted_hessian(gaps, coulomb, exchange_direct, exchange_transpose, alda,
                                  chf_exchange=0.25, alda_coefficient=0.75):
    return psi4.core._atomic_polarizability_assemble_restricted_hessian(
        gaps,
        _matrix(coulomb),
        _matrix(exchange_direct),
        _matrix(exchange_transpose),
        _matrix(alda),
        chf_exchange,
        alda_coefficient,
    )


def _response_matrix_values(matrix):
    rows, columns = matrix.shape
    return [[matrix.get(row, column) for column in range(columns)] for row in range(rows)]


def test_restricted_hessian_one_transition_matches_literal_native_formula():
    result = _assemble_restricted_hessian([2.0], [[3.0]], [[5.0]], [[7.0]], [[11.0]])

    # Independent rational arithmetic:
    # H1 = 2 + 4*3 - (1/4)*(5+7) + 4*(3/4)*11 = 44
    # H2 = 2 - (1/4)*5 + (1/4)*7 = 5/2
    assert _response_matrix_values(result["H1"]) == [[44.0]]
    assert _response_matrix_values(result["H2"]) == [[2.5]]


def test_restricted_hessian_two_transition_matches_elementwise_rational_oracle():
    result = _assemble_restricted_hessian(
        [2.0, 3.0],
        [[1.0, 2.0], [2.0, 4.0]],
        [[5.0, 6.0], [6.0, 7.0]],
        [[9.0, 11.0], [11.0, 14.0]],
        [[11.0, 12.0], [12.0, 13.0]],
    )

    # Literal values keep every primitive distinct and do not reuse production algebra.
    assert _response_matrix_values(result["H1"]) == [[35.5, 39.75], [39.75, 52.75]]
    assert _response_matrix_values(result["H2"]) == [[3.0, 1.25], [1.25, 4.75]]
    assert all(result[name].get(i, j) == result[name].get(j, i)
               for name in ("H1", "H2") for i in range(2) for j in range(2))


def test_restricted_hessian_oracle_detects_every_omitted_or_swapped_component():
    inputs = {
        "gaps": [2.0, 3.0],
        "coulomb": [[1.0, 2.0], [2.0, 4.0]],
        "exchange_direct": [[5.0, 6.0], [6.0, 7.0]],
        "exchange_transpose": [[9.0, 11.0], [11.0, 14.0]],
        "alda": [[11.0, 12.0], [12.0, 13.0]],
    }
    expected = (
        [[35.5, 39.75], [39.75, 52.75]],
        [[3.0, 1.25], [1.25, 4.75]],
    )
    zero = [[0.0, 0.0], [0.0, 0.0]]
    mutations = [
        {"gaps": [1.0, 1.0]},
        {"coulomb": zero},
        {"exchange_direct": zero},
        {"exchange_transpose": zero},
        {"alda": zero},
        {"coulomb": inputs["alda"], "alda": inputs["coulomb"]},
        {"exchange_direct": inputs["exchange_transpose"],
         "exchange_transpose": inputs["exchange_direct"]},
    ]
    for mutation in mutations:
        candidate = {**inputs, **mutation}
        result = _assemble_restricted_hessian(**candidate)
        actual = (_response_matrix_values(result["H1"]), _response_matrix_values(result["H2"]))
        assert actual != expected


@pytest.mark.parametrize(
    "gaps,coulomb,exchange_direct,exchange_transpose,alda,message",
    [
        ([], [[1.0]], [[1.0]], [[1.0]], [[1.0]], r"orbital gaps.*nonempty"),
        ([0.0], [[1.0]], [[1.0]], [[1.0]], [[1.0]], r"orbital gaps.*positive"),
        ([-1.0], [[1.0]], [[1.0]], [[1.0]], [[1.0]], r"orbital gaps.*positive"),
        ([math.nan], [[1.0]], [[1.0]], [[1.0]], [[1.0]], r"orbital gaps.*finite"),
        ([1.0, 2.0], [[1.0]], [[1.0, 0.0], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], r"Coulomb.*dimension"),
        ([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], [[1.0]],
         [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], r"K_direct.*dimension"),
        ([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0]], [[1.0]],
         [[1.0, 0.0], [0.0, 1.0]], r"K_transpose.*dimension"),
        ([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0]],
         r"full ALDA.*dimension"),
        ([1.0, 2.0], [[1.0, 0.1], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0]], r"Coulomb.*symmetric"),
        ([1.0], [[1.0]], [[math.inf]], [[1.0]], [[1.0]], r"K_direct.*finite"),
        ([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0e-4], [0.0, 1.0]],
         [[1.0, 0.0], [0.0, 1.0]], r"K_transpose.*symmetric"),
        ([1.0], [[1.0]], [[1.0]], [[1.0]], [[math.nan]], r"full ALDA.*finite"),
    ],
)
def test_restricted_hessian_rejects_invalid_primitives(
    gaps, coulomb, exchange_direct, exchange_transpose, alda, message
):
    with pytest.raises(RuntimeError, match=message):
        _assemble_restricted_hessian(gaps, coulomb, exchange_direct, exchange_transpose, alda)


def test_restricted_hessian_requires_exact_response_kernel_and_no_ground_functional():
    unit = [[1.0]]
    with pytest.raises(RuntimeError, match=r"CHF exchange.*exactly 0.25"):
        _assemble_restricted_hessian(
            [1.0], unit, unit, unit, unit, math.nextafter(0.25, math.inf), 0.75
        )
    with pytest.raises(RuntimeError, match=r"ALDA coefficient.*exactly 0.75"):
        _assemble_restricted_hessian(
            [1.0], unit, unit, unit, unit, 0.25, math.nextafter(0.75, 0.0)
        )
    with pytest.raises(TypeError):
        psi4.core._atomic_polarizability_assemble_restricted_hessian(
            [1.0], _matrix(unit), _matrix(unit), _matrix(unit), _matrix(unit), 0.25, 0.75,
            ground_functional="must not be accepted",
        )


def test_restricted_response_one_transition_matches_imaginary_frequency_algebra():
    delta = 2.5
    omega = 0.7
    source = 1.2
    result = _solve_restricted_response([[delta]], [[delta]], omega, [[source]])

    expected_p = delta * source / (delta * delta + omega * omega)
    expected_q = omega * source / (delta * delta + omega * omega)
    assert result["P"].get(0, 0) == pytest.approx(expected_p, abs=1.0e-14)
    assert result["Q"].get(0, 0) == pytest.approx(expected_q, abs=1.0e-14)
    assert 0.0 < result["reciprocal_condition"] <= 1.0


def test_restricted_response_static_reduces_exactly_to_h1_solve():
    result = _solve_restricted_response(
        [[4.0, 1.0], [1.0, 3.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        0.0,
        [[1.0], [2.0]],
    )
    _assert_matrix_close(
        _response_matrix_values(result["P"]), [[1.0 / 11.0], [7.0 / 11.0]], 1.0e-15
    )
    assert _response_matrix_values(result["Q"]) == [[0.0], [0.0]]


def test_restricted_response_two_transition_diagonal_matches_componentwise_algebra():
    deltas = [1.5, 4.0]
    omega = 0.8
    source = [[2.0], [-3.0]]
    result = _solve_restricted_response(
        [[deltas[0], 0.0], [0.0, deltas[1]]],
        [[deltas[0], 0.0], [0.0, deltas[1]]],
        omega,
        source,
    )
    expected_p = [[deltas[row] * source[row][0] / (deltas[row] ** 2 + omega**2)]
                  for row in range(2)]
    expected_q = [[omega * source[row][0] / (deltas[row] ** 2 + omega**2)]
                  for row in range(2)]
    _assert_matrix_close(_response_matrix_values(result["P"]), expected_p, 1.0e-13)
    _assert_matrix_close(_response_matrix_values(result["Q"]), expected_q, 1.0e-13)


def test_restricted_response_supports_multiple_rhs_without_cross_column_mixing():
    result = _solve_restricted_response(
        [[2.0, 0.0], [0.0, 3.0]],
        [[2.0, 0.0], [0.0, 3.0]],
        0.5,
        [[1.0, 4.0, -2.0], [3.0, -1.0, 5.0]],
    )
    expected_p = [
        [2.0 * value / (2.0**2 + 0.5**2) for value in [1.0, 4.0, -2.0]],
        [3.0 * value / (3.0**2 + 0.5**2) for value in [3.0, -1.0, 5.0]],
    ]
    _assert_matrix_close(_response_matrix_values(result["P"]), expected_p, 1.0e-13)
    assert result["P"].shape == (2, 3)
    assert result["Q"].shape == (2, 3)


def test_restricted_response_distinct_noncommuting_blocks_match_full_numpy_oracle():
    h1 = np.array([[3.0, 0.7], [0.7, 1.8]])
    h2 = np.array([[2.1, -0.4], [-0.4, 3.2]])
    omega = 0.65
    rhs = np.array([[1.0, -0.3, 2.4], [0.2, 1.7, -0.8]])
    assert not np.allclose(h1 @ h2, h2 @ h1)

    doubled = np.block([[h1, omega * np.eye(2)], [-omega * np.eye(2), h2]])
    doubled_rhs = np.vstack((rhs, np.zeros_like(rhs)))
    oracle = np.linalg.solve(doubled, doubled_rhs)
    result = _solve_restricted_response(h1.tolist(), h2.tolist(), omega, rhs.tolist())
    p = np.array(_response_matrix_values(result["P"]))
    q = np.array(_response_matrix_values(result["Q"]))

    assert p == pytest.approx(oracle[:2], abs=2.0e-14)
    assert q == pytest.approx(oracle[2:], abs=2.0e-14)
    assert np.max(np.abs(h1 @ p + omega * q - rhs)) < 2.0e-14
    assert np.max(np.abs(-omega * p + h2 @ q)) < 2.0e-14

    wrong_h2_oracle = np.linalg.solve(
        np.block([[h1, omega * np.eye(2)], [-omega * np.eye(2), h1]]), doubled_rhs
    )
    assert np.max(np.abs(oracle - wrong_h2_oracle)) > 1.0e-2


def test_restricted_response_amplitude_contraction_is_reciprocal():
    h1 = [[3.0, 0.4], [0.4, 2.0]]
    h2 = [[2.5, -0.2], [-0.2, 1.7]]
    sources = [[1.0, -0.3], [0.2, 0.8]]
    p = _response_matrix_values(_solve_restricted_response(h1, h2, 0.6, sources)["P"])
    first_second = sum(sources[row][0] * p[row][1] for row in range(2))
    second_first = sum(sources[row][1] * p[row][0] for row in range(2))
    assert first_second == pytest.approx(second_first, abs=2.0e-14)


@pytest.mark.parametrize(
    "h1,h2,omega,rhs,message",
    [
        ([[1.0, 0.0]], [[1.0]], 0.0, [[1.0]], r"H1.*square"),
        ([[1.0, 0.0], [0.0, 1.0]], [[1.0]], 0.0, [[1.0], [2.0]], r"same dimension"),
        ([[1.0, 1.0e-4], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], 0.0,
         [[1.0], [2.0]], r"H1.*symmetric"),
        ([[1.0]], [[math.nan]], 0.0, [[1.0]], r"H2.*finite"),
        ([[1.0]], [[1.0]], -math.ulp(1.0), [[1.0]], r"omega.*nonnegative"),
        ([[1.0]], [[1.0]], math.inf, [[1.0]], r"omega.*finite"),
        ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], 0.0,
         [[1.0]], r"RHS.*dimensions"),
        ([[1.0]], [[1.0]], 0.0, [[math.nan]], r"RHS.*finite"),
    ],
)
def test_restricted_response_rejects_invalid_inputs(h1, h2, omega, rhs, message):
    with pytest.raises(RuntimeError, match=message):
        _solve_restricted_response(h1, h2, omega, rhs)


@pytest.mark.parametrize("small", [0.0, 1.0e-20])
def test_restricted_response_rejects_singular_or_numerically_singular_systems(small):
    with pytest.raises(RuntimeError, match=r"singular|condition"):
        _solve_restricted_response(
            [[1.0, 0.0], [0.0, small]],
            [[1.0, 0.0], [0.0, 1.0]],
            0.0,
            [[1.0], [1.0]],
        )


def test_restricted_response_rejects_finite_frequency_singular_doubled_system():
    with pytest.raises(RuntimeError, match=r"singular|condition"):
        _solve_restricted_response([[1.0]], [[-1.0]], 1.0, [[1.0]])


def _validate_response_diagnostics(rcond=1.0, pivot=1.0, ferr=(0.0,), berr=(0.0,), residual=(0.0,)):
    return psi4.core._atomic_polarizability_validate_response_diagnostics(
        rcond, pivot, list(ferr), list(berr), list(residual)
    )


def test_restricted_response_diagnostic_budget_accepts_exact_boundaries():
    assert _validate_response_diagnostics(
        rcond=1.0e-12, pivot=1.0e-12, ferr=(1.0e-8,), berr=(1.0e-11,), residual=(1.0e-11,)
    ) is None


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"rcond": math.nextafter(1.0e-12, 0.0)}, r"reciprocal condition"),
        ({"pivot": math.nextafter(1.0e-12, 0.0)}, r"pivot growth"),
        ({"ferr": (math.nextafter(1.0e-8, math.inf),)}, r"forward error"),
        ({"berr": (math.nextafter(1.0e-11, math.inf),)}, r"backward error"),
        ({"residual": (math.nextafter(1.0e-11, math.inf),)}, r"residual"),
        ({"rcond": math.nan}, r"reciprocal condition"),
        ({"pivot": math.inf}, r"pivot growth"),
        ({"ferr": (math.nan,)}, r"forward error"),
        ({"berr": (math.inf,)}, r"backward error"),
        ({"residual": (math.nan,)}, r"residual"),
        ({"ferr": (-math.ulp(0.0),)}, r"forward error"),
        ({"berr": (-math.ulp(0.0),)}, r"backward error"),
        ({"residual": (-math.ulp(0.0),)}, r"residual"),
    ],
)
def test_restricted_response_diagnostic_budget_rejects_threshold_neighbors_and_nonfinite(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_response_diagnostics(**kwargs)


def test_casimir_frequency_grid_matches_reviewed_eleven_point_protocol_exactly():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    assert tuple(frequencies) == _REVIEWED_FREQUENCIES
    assert tuple(weights) == _REVIEWED_WEIGHTS


def test_casimir_frequency_grid_scales_frequencies_and_weights():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.25)
    assert tuple(frequencies) == tuple(value * 0.5 for value in _REVIEWED_FREQUENCIES)
    assert tuple(weights) == tuple(value * 0.5 for value in _REVIEWED_WEIGHTS)


@pytest.mark.parametrize("nonzero_count", [0, 9, 11])
def test_casimir_frequency_grid_requires_ten_nonzero_points(nonzero_count):
    with pytest.raises(RuntimeError, match=r"exactly ten nonzero"):
        psi4.core._atomic_polarizability_make_casimir_grid(nonzero_count, 0.5)


@pytest.mark.parametrize(
    "scale",
    [
        0.0,
        -0.5,
        float("inf"),
        float("nan"),
        float.fromhex("0x1.fffffffffffffp+1023"),
        float.fromhex("0x0.0000000000001p-1022"),
    ],
)
def test_casimir_frequency_grid_rejects_invalid_scale(scale):
    with pytest.raises(RuntimeError, match=r"finite and positive"):
        psi4.core._atomic_polarizability_make_casimir_grid(10, scale)


def test_casimir_frequency_grid_rejects_weight_overflow():
    weight_overflow_scale = float.fromhex("0x1.fffffffffffffp+1023") / 100.0
    assert math.isfinite(weight_overflow_scale * (_REVIEWED_FREQUENCIES[-1] / 0.5))
    with pytest.raises(RuntimeError, match=r"finite and positive at every grid point"):
        psi4.core._atomic_polarizability_make_casimir_grid(10, weight_overflow_scale)


def test_local_spherical_dipole_maps_10_11c_11s_to_z_x_y():
    spherical = [[0.0] * 15 for _ in range(15)]
    dipole_block = (
        (1.0, 2.0, 3.0),
        (2.0, 4.0, 5.0),
        (3.0, 5.0, 6.0),
    )
    for row in range(3):
        for column in range(3):
            spherical[row][column] = dipole_block[row][column]

    cartesian = psi4.core._atomic_polarizability_local_spherical_dipole_to_cartesian(_matrix(spherical))
    assert _as_packed_rows(cartesian) == pytest.approx(
        [4.0, 5.0, 2.0, 5.0, 6.0, 3.0, 2.0, 3.0, 1.0]
    )


def test_rotate_tensor_applies_global_r_local_rt_with_right_handed_frame():
    local = _matrix([[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 4.0]])
    rotation = _matrix([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    rotated = psi4.core._atomic_polarizability_rotate_tensor(local, rotation)
    assert _as_packed_rows(rotated) == pytest.approx(
        [3.0, -0.5, 0.0, -0.5, 2.0, 0.0, 0.0, 0.0, 4.0]
    )


@pytest.mark.parametrize(
    "rotation",
    [
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_rotate_tensor_rejects_invalid_frames(rotation):
    with pytest.raises(RuntimeError, match=r"orthonormal right-handed"):
        psi4.core._atomic_polarizability_rotate_tensor(
            _matrix([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            _matrix(rotation),
        )


def test_pack_symmetric_tensor_uses_xx_xy_xz_yy_yz_zz_order():
    tensor = _matrix([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert psi4.core._atomic_polarizability_pack_symmetric_tensor(tensor) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )


def test_tensor_algebra_rejects_asymmetry_and_nonfinite_values():
    asymmetric = _matrix([[1.0, 2.0, 0.0], [2.1, 3.0, 0.0], [0.0, 0.0, 4.0]])
    with pytest.raises(RuntimeError, match=r"finite symmetric"):
        psi4.core._atomic_polarizability_pack_symmetric_tensor(asymmetric)

    nonfinite = _matrix([[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(RuntimeError, match=r"finite symmetric"):
        psi4.core._atomic_polarizability_rotate_tensor(
            nonfinite, _matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )


def _working_l3_matrix():
    return [[0.0] * 16 for _ in range(16)]


def _lw_localize(positions, values, bonds, tolerance=1.0e-9, frequency=0.0):
    return psi4.core._atomic_polarizability_localize_lw(
        _matrix(positions), [_matrix(block) for block in values], frequency, bonds, tolerance
    )


def test_lw_localization_requires_explicit_finite_frequency_identity():
    with pytest.raises(Exception, match="frequency must be finite"):
        _lw_localize([[0.0, 0.0, 0.0]], [_working_l3_matrix()], [], frequency=float("nan"))


def test_lw_graph_operator_is_symmetric_with_one_null_mode_for_connected_chain():
    operator, pseudoinverse, eigenvalues = psi4.core._atomic_polarizability_lw_graph_math(
        3, [(0, 1), (1, 2)]
    )
    assert [operator.get(i, j) for i in range(3) for j in range(3)] == [
        -1.0, 1.0, 0.0, 1.0, -2.0, 1.0, 0.0, 1.0, -1.0
    ]
    assert all(operator.get(i, j) == operator.get(j, i) for i in range(3) for j in range(3))
    assert all(sum(operator.get(i, j) for j in range(3)) == 0.0 for i in range(3))
    assert sum(abs(value) < 1.0e-12 for value in eigenvalues) == 1
    assert [
        sum(operator.get(i, k) * pseudoinverse.get(k, j) for k in range(3))
        for i in range(3) for j in range(3)
    ] == pytest.approx([
        2 / 3, -1 / 3, -1 / 3, -1 / 3, 2 / 3, -1 / 3, -1 / 3, -1 / 3, 2 / 3
    ], abs=1.0e-12)


def test_lw_rank3_translation_matches_arbitrary_displacement_fixtures():
    displacement = [0.2, -0.3, 0.4]
    expected_scalar = [
        2.0, 0.8, 0.4, -0.6, 0.19, 0.277128129211, -0.415692193817,
        -0.0866025403784, -0.207846096908, -0.028, 0.124923976882,
        -0.187385965323, -0.0774596669241, -0.185903200618,
        -0.0727323861839, -0.0142302494708,
    ]
    assert psi4.core._atomic_polarizability_translate_l3(
        [2.0] + [0.0] * 15, displacement
    ) == pytest.approx(expected_scalar, abs=5.0e-12)
    expected_dense = [
        0.1, 0.24, 0.32, 0.37, 0.7295, 0.890984535672, 0.852420471066,
        1.10743901834, 0.872287187079, 1.88348457268, 2.29457302188,
        1.30348504869, 3.00906108622, 2.04427885923, 2.75223325142,
        1.30215605855,
    ]
    assert psi4.core._atomic_polarizability_translate_l3(
        [0.1 * value for value in range(1, 17)], displacement
    ) == pytest.approx(expected_dense, abs=5.0e-11)


def test_lw_two_site_charge_flow_localizes_to_full_rank3_fixture():
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0], values[1][0][0] = -2.0, 2.0
    values[2][0][0], values[3][0][0] = 2.0, -2.0
    result = _lw_localize([[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]], values, [(0, 1)])
    local = result["local"]
    assert len(local) == 2
    assert local[0].shape == (15, 15)
    assert local[0].get(0, 0) == pytest.approx(-0.16, abs=1.0e-11)
    assert local[0].get(0, 1) == pytest.approx(-0.08, abs=1.0e-11)
    assert local[0].get(0, 2) == pytest.approx(0.12, abs=1.0e-11)
    assert local[0].get(0, 3) == pytest.approx(-0.038, abs=1.0e-11)
    assert local[1].get(0, 3) == pytest.approx(0.038, abs=1.0e-11)
    assert local[0].get(14, 14) == pytest.approx(-0.000050625, abs=1.0e-12)
    assert local[1].get(14, 14) == pytest.approx(-0.000050625, abs=1.0e-12)
    assert max(result["residuals"]) < 1.0e-10


def test_lw_three_site_preserves_sum_reciprocity_and_transfers_only_on_bonds():
    positions = [[0.0, 0.0, 0.0], [0.7, -0.2, 0.1], [1.1, 0.4, -0.3]]
    graph_operator = [[-1.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -1.0]]
    values = [_working_l3_matrix() for _ in range(9)]
    for a in range(3):
        for b in range(3):
            values[3 * a + b][0][0] = 1.7 * graph_operator[a][b]
    result = _lw_localize(positions, values, [(0, 1), (1, 2)], 2.0e-9)
    assert max(result["residuals"]) < 2.0e-9
    assert all(
        (first, second) in {(0, 1), (1, 2)}
        for first, second, _mu, _nu, _site, _amount in result["transfers"]
    )
    for matrix in result["local"]:
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
    columns = []
    for source in range(16):
        unit = [0.0] * 16
        unit[source] = 1.0
        columns.append(psi4.core._atomic_polarizability_translate_l3(unit, displacement))
    return [list(row) for row in zip(*columns)]


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
    refined = [_matrix_values(block) for block in result["refined"]]
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
            _assert_matrix_close(_matrix_values(result["local"][site]), expected[site], 2.0e-11)
        _assert_refined_invariants(result, positions, values, 2.0e-10)


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
    assert len(result["local"]) == 3
    assert sum(abs(value) < 1.0e-12 for value in
               psi4.core._atomic_polarizability_lw_graph_math(3, [(0, 1)])[2]) == 2

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
    assert (1, 1) in map(tuple, candidate_below["omitted_component_pairs"])
    assert (1, 1) not in map(tuple, candidate_equal["omitted_component_pairs"])
    assert candidate_equal["omitted_transfer_count"] > candidate_below["omitted_transfer_count"]
    assert not any(transfer[2:4] == (1, 1) for transfer in candidate_equal["transfers"])
    assert not any(transfer[2:4] == (1, 1) for transfer in transfer_equal["transfers"])
    assert any(transfer[2:4] == (1, 1) for transfer in transfer_above["transfers"])
    assert transfer_above["refined"][1].get(1, 1) == pytest.approx(0.0, abs=1.0e-12)


def test_lw_finite_inputs_that_overflow_derived_math_fail_closed():
    zero = [_working_l3_matrix() for _ in range(4)]
    with pytest.raises(RuntimeError, match=r"finite|overflow"):
        _lw_localize([[0.0, 0.0, 0.0], [1.0e308, 0.0, 0.0]], zero, [(0, 1)])

    large = [_working_l3_matrix() for _ in range(4)]
    large[1][0][0] = 1.0e308
    large[2][0][0] = 1.0e308
    large[0][0][0] = -1.0e308
    large[3][0][0] = -1.0e308
    with pytest.raises(RuntimeError, match=r"finite|overflow"):
        _lw_localize([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], large, [(0, 1)], 1.0e300)


def test_lw_lapack_pseudoinverse_identities_on_realistic_chain():
    count = 24
    bonds = [(site, site + 1) for site in range(count - 1)]
    operator, pseudoinverse, eigenvalues = psi4.core._atomic_polarizability_lw_graph_math(count, bonds)
    b = _matrix_values(operator)
    inverse = _matrix_values(pseudoinverse)
    projector = _matmul(b, inverse)
    assert sum(abs(value) < 1.0e-10 for value in eigenvalues) == 1
    assert max(abs(inverse[i][j] - inverse[j][i]) for i in range(count) for j in range(count)) < 1.0e-11
    _assert_matrix_close(_matmul(_matmul(b, inverse), b), b, 2.0e-10)
    _assert_matrix_close(_matmul(_matmul(inverse, b), inverse), inverse, 2.0e-10)
    _assert_matrix_close(projector, _transpose(projector), 2.0e-10)
    _assert_matrix_close(_matmul(projector, projector), projector, 2.0e-10)


def test_atomic_polarizabilities_api_is_registered():
    assert "ATOMIC_POLARIZABILITIES" in psi4.core.OEProp.valid_methods
    assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_N_FREQUENCIES") == 10
    assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_FREQUENCY_SCALE") == pytest.approx(0.5)


def test_atomic_polarizabilities_fail_closed_without_response_data():
    molecule = psi4.geometry(
        """
        H 0.0 0.0 0.0
        H 0.0 0.0 0.7
        symmetry c1
        units angstrom
        """
    )
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk"})
    _, wfn = psi4.energy("scf", return_wfn=True)

    oeprop = psi4.core.OEProp(wfn)
    oeprop.add("MULTIPOLE(2)")
    oeprop.add("ATOMIC_POLARIZABILITIES")

    with pytest.raises(RuntimeError, match=r"AtomicPolarizabilityCalculator.*response data"):
        oeprop.compute()

    unpublished = ("DIPOLE", "QUADRUPOLE", *_PUBLIC_ARRAYS)
    assert all(not wfn.has_array_variable(name) for name in unpublished)
    assert all(not psi4.core.has_array_variable(name) for name in unpublished)


def test_atomic_polarizabilities_reject_incomplete_wavefunction_prerequisites():
    molecule = psi4.geometry(
        """
        He 0.0 0.0 0.0
        symmetry c1
        """
    )
    wfn = psi4.core.Wavefunction.build(molecule, "sto-3g")
    calculator = psi4.core.AtomicPolarizabilityCalculator(wfn)

    with pytest.raises(RuntimeError, match=r"unsupported wavefunction.*orbital response data"):
        calculator.compute()

    assert all(not wfn.has_array_variable(name) for name in _PUBLIC_ARRAYS)


def test_native_atomic_polarizability_source_guard():
    from test_native_atomic_polarizability_source_guard import source_violations

    repo_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "psi4/src/psi4/libmints/atomic_polarizability.cc").is_file()
    )
    native_sources = (
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.cc",
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.h",
        repo_root / "psi4/src/psi4/libmints/oeprop.cc",
        repo_root / "psi4/src/psi4/libmints/oeprop.h",
        repo_root / "psi4/src/export_oeprop.cc",
    )

    violations = []
    for source in native_sources:
        violations.extend(f"{source.name}: {item}" for item in source_violations(source.read_text()))

    cmake_text = (repo_root / "psi4/src/psi4/libmints/CMakeLists.txt").read_text()
    assert "atomic_polarizability.cc" in cmake_text
    assert violations == []

    canary = 'void launch() { std::system("camcasp"); }'
    assert source_violations(canary) == ["forbidden process API: std::system(", "forbidden external term: camcasp"]


# Ordered rank pairs permitted by the isotropic L3 recoupling table, with the
# published prefactor numerators binom(2*la + 2*lb, 2*la); every K is that
# numerator divided by 2*pi. (1,4)/(4,1) also satisfy n = 12 but rank 4 is
# absent from an L3 model, so they are deliberately not present.
_DISPERSION_RANK_PAIRS = (
    (6, 1, 1, 6.0),
    (8, 1, 2, 15.0),
    (8, 2, 1, 15.0),
    (10, 1, 3, 28.0),
    (10, 3, 1, 28.0),
    (10, 2, 2, 70.0),
    (12, 2, 3, 210.0),
    (12, 3, 2, 210.0),
)

_DISPERSION_SITES = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]


def _dispersion_l3(rank_values, coupling=0.0):
    """Build a 15 by 15 L3 tensor whose rank-l block has isotropic mean rank_values[l - 1]."""
    values = [[0.0] * 15 for _ in range(15)]
    for start, end, value in ((0, 3, rank_values[0]), (3, 8, rank_values[1]), (8, 15, rank_values[2])):
        for index in range(start, end):
            values[index][index] = value
    for first, second in ((0, 1), (3, 4), (8, 9), (0, 4), (2, 9), (5, 13)):
        values[first][second] = coupling
        values[second][first] = coupling
    return values


def _mapped_casimir_quadrature(count, scale):
    """Half-line Gauss-Legendre mapping of the specification, ascending, static point first."""
    nodes, weights = np.polynomial.legendre.leggauss(count)
    points = sorted(
        (scale * (1.0 - node) / (1.0 + node), weight * 2.0 * scale / (1.0 + node) ** 2)
        for node, weight in zip(nodes, weights)
    )
    return [0.0] + [point[0] for point in points], [0.0] + [point[1] for point in points]


def _compute_dispersion(models, grid_frequencies, grid_weights, sites=None, protocol=True):
    """models[frequency][site] holds 15 by 15 nested values in grid-frequency order."""
    tensors = [_matrix(tensor) for frequency in models for tensor in frequency]
    entry = (
        psi4.core._atomic_polarizability_compute_dispersion
        if protocol
        else psi4.core._atomic_polarizability_test_compute_dispersion
    )
    return entry(
        _matrix(sites if sites is not None else _DISPERSION_SITES[: len(models[0])]),
        list(grid_frequencies),
        tensors,
        list(grid_frequencies),
        list(grid_weights),
    )


def _uniform_dispersion_models(profiles, frequency_count, coupling=0.0):
    return [[_dispersion_l3(profile, coupling) for profile in profiles] for _ in range(frequency_count)]


def _rank_pair_contribution(result, order, first_rank, second_rank, first_site, second_site):
    terms = result["rank_pair_terms"]
    site_count = result["site_count"]
    index = next(
        position
        for position, term in enumerate(terms)
        if (term["coefficient_order"], term["first_rank"], term["second_rank"])
        == (order, first_rank, second_rank)
    )
    return result["rank_pair_contributions"][
        index * site_count * site_count + first_site * site_count + second_site
    ]


@pytest.mark.parametrize("order,first_rank,second_rank,numerator", _DISPERSION_RANK_PAIRS)
def test_dispersion_rank_prefactor_matches_published_binomial_table(order, first_rank, second_rank, numerator):
    prefactor = psi4.core._atomic_polarizability_dispersion_rank_prefactor(first_rank, second_rank)
    assert prefactor == pytest.approx(numerator / (2.0 * math.pi), rel=1.0e-15)
    assert order == 2 * (first_rank + second_rank + 1)
    assert prefactor == psi4.core._atomic_polarizability_dispersion_rank_prefactor(
        second_rank, first_rank
    )


def test_dispersion_rank_prefactor_rejects_ranks_outside_the_l3_model():
    for first_rank, second_rank in ((0, 1), (1, 0), (4, 1), (1, 4)):
        with pytest.raises(RuntimeError, match=r"rank"):
            psi4.core._atomic_polarizability_dispersion_rank_prefactor(first_rank, second_rank)


def test_dispersion_recoupling_table_contains_exactly_the_permitted_ordered_pairs():
    result = _compute_dispersion(
        _uniform_dispersion_models([[1.0, 2.0, 3.0]], 11, coupling=0.25),
        *psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5),
    )
    assert [
        (term["coefficient_order"], term["first_rank"], term["second_rank"])
        for term in result["rank_pair_terms"]
    ] == [(order, first_rank, second_rank) for order, first_rank, second_rank, _ in _DISPERSION_RANK_PAIRS]
    assert [term["prefactor"] for term in result["rank_pair_terms"]] == pytest.approx(
        [numerator / (2.0 * math.pi) for _, _, _, numerator in _DISPERSION_RANK_PAIRS], rel=1.0e-15
    )


def test_dispersion_isotropic_rank_extraction_uses_block_trace_over_two_l_plus_one():
    values = [[0.0] * 15 for _ in range(15)]
    for index in range(15):
        values[index][index] = float(index + 1)
    for first, second in ((0, 2), (3, 7), (8, 14), (1, 9)):
        values[first][second] = 100.0
        values[second][first] = 100.0

    extract = psi4.core._atomic_polarizability_dispersion_isotropic_rank
    assert extract(_matrix(values), 1) == pytest.approx((1.0 + 2.0 + 3.0) / 3.0, rel=1.0e-15)
    assert extract(_matrix(values), 2) == pytest.approx(sum(range(4, 9)) / 5.0, rel=1.0e-15)
    assert extract(_matrix(values), 3) == pytest.approx(sum(range(9, 16)) / 7.0, rel=1.0e-15)


@pytest.mark.parametrize("rank", [0, 4])
def test_dispersion_isotropic_rank_extraction_rejects_ranks_outside_the_l3_model(rank):
    with pytest.raises(RuntimeError, match=r"rank"):
        psi4.core._atomic_polarizability_dispersion_isotropic_rank(
            _matrix(_dispersion_l3([1.0, 1.0, 1.0])), rank
        )


def test_dispersion_isotropic_c6_matches_analytic_three_over_pi_integration():
    # Higher ranks are populated because the L3 model must be rank complete, but
    # C6 depends only on the rank-1 isotropic means.
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    dipoles = (1.5, 0.75)
    models = _uniform_dispersion_models(
        [[dipoles[0], 0.4, 0.2], [dipoles[1], 0.3, 0.1]], len(frequencies)
    )
    result = _compute_dispersion(models, frequencies, weights)

    weight_sum = sum(weights)
    assert result["quadrature_weight_sum"] == pytest.approx(weight_sum, rel=1.0e-15)
    for first in range(2):
        for second in range(2):
            assert result["c6"].get(first, second) == pytest.approx(
                (3.0 / math.pi) * dipoles[first] * dipoles[second] * weight_sum, rel=1.0e-13
            )


def test_dispersion_pair_matrices_are_exactly_symmetric():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    models = [
        [
            _dispersion_l3([2.3 / (1.0 + frequency), 0.9 / (1.0 + frequency), 0.4 / (1.0 + frequency)]),
            _dispersion_l3([0.6 / (1.0 + 3.0 * frequency), 0.2, 0.05]),
            _dispersion_l3([0.5, 0.11 / (1.0 + frequency ** 2), 0.03]),
        ]
        for frequency in frequencies
    ]
    result = _compute_dispersion(models, frequencies, weights)

    for name in ("c6", "c8", "c10", "c12"):
        matrix = result[name]
        assert matrix.shape == (3, 3)
        assert all(
            matrix.get(first, second) == matrix.get(second, first)
            for first in range(3) for second in range(3)
        )
        assert all(matrix.get(first, second) > 0.0 for first in range(3) for second in range(3))

    # Individual ordered rank-pair terms are not themselves symmetric.
    assert _rank_pair_contribution(result, 8, 1, 2, 0, 1) != pytest.approx(
        _rank_pair_contribution(result, 8, 1, 2, 1, 0), rel=1.0e-6
    )


@pytest.mark.parametrize("order,first_rank,second_rank,numerator", _DISPERSION_RANK_PAIRS)
def test_dispersion_each_permitted_rank_pair_contributes_its_closed_form_term(
    order, first_rank, second_rank, numerator
):
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    profiles = ([1.3, 0.7, 0.21], [0.45, 0.19, 0.06])
    models = [
        [_dispersion_l3([value / (1.0 + frequency) for value in profile]) for profile in profiles]
        for frequency in frequencies
    ]
    result = _compute_dispersion(models, frequencies, weights)

    expected = (numerator / (2.0 * math.pi)) * sum(
        weight * profiles[0][first_rank - 1] * profiles[1][second_rank - 1] / (1.0 + frequency) ** 2
        for frequency, weight in zip(frequencies, weights)
    )
    assert _rank_pair_contribution(result, order, first_rank, second_rank, 0, 1) == pytest.approx(
        expected, rel=1.0e-13
    )

    coefficient = {6: "c6", 8: "c8", 10: "c10", 12: "c12"}[order]
    assembled = sum(
        _rank_pair_contribution(result, term_order, term_first, term_second, 0, 1)
        for term_order, term_first, term_second, _ in _DISPERSION_RANK_PAIRS
        if term_order == order
    )
    assert result[coefficient].get(0, 1) == pytest.approx(assembled, rel=1.0e-14)


def test_dispersion_quadrature_converges_for_a_single_pole_model():
    pole = 0.5
    dipole = 1.7
    # (3/pi) * integral of (a/(1+(w/w0)^2))^2 dw = (3/4) * a^2 * w0.
    exact = 0.75 * dipole * dipole * pole

    reviewed_frequencies, reviewed_weights = _mapped_casimir_quadrature(10, 0.5)
    assert reviewed_frequencies == pytest.approx(_REVIEWED_FREQUENCIES, abs=1.0e-13)
    assert reviewed_weights == pytest.approx(_REVIEWED_WEIGHTS, rel=1.0e-13)

    errors = []
    for count in (4, 8, 16):
        frequencies, weights = _mapped_casimir_quadrature(count, pole)
        models = [
            [
                _dispersion_l3([
                    dipole / (1.0 + (frequency / pole) ** 2),
                    0.4 / (1.0 + (frequency / pole) ** 2),
                    0.1 / (1.0 + (frequency / pole) ** 2),
                ])
            ]
            for frequency in frequencies
        ]
        result = _compute_dispersion(models, frequencies, weights, protocol=False)
        errors.append(abs(result["c6"].get(0, 0) - exact) / exact)

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1.0e-9


@pytest.mark.parametrize("rank", [2, 3])
def test_dispersion_rejects_a_model_missing_a_higher_rank_block(rank):
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    profile = [1.0, 0.5, 0.25]
    profile[rank - 1] = 0.0
    models = _uniform_dispersion_models([[1.0, 0.5, 0.25], profile], len(frequencies))

    with pytest.raises(RuntimeError, match=rf"rank[ -]{rank}"):
        _compute_dispersion(models, frequencies, weights)


def test_dispersion_static_point_carries_no_quadrature_weight():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    assert frequencies[0] == 0.0
    assert weights[0] == 0.0

    models = _uniform_dispersion_models([[1.1, 0.6, 0.2], [0.7, 0.3, 0.1]], len(frequencies))
    baseline = _compute_dispersion(models, frequencies, weights)
    models[0] = [_dispersion_l3([9.5, 8.5, 7.5]), _dispersion_l3([6.5, 5.5, 4.5])]
    perturbed = _compute_dispersion(models, frequencies, weights)

    for name in ("c6", "c8", "c10", "c12"):
        assert all(
            baseline[name].get(first, second) == perturbed[name].get(first, second)
            for first in range(2) for second in range(2)
        )


def test_dispersion_rejects_a_static_point_that_carries_quadrature_weight():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    weights[0] = 1.0e-6
    models = _uniform_dispersion_models([[1.1, 0.6, 0.2]], len(frequencies))

    with pytest.raises(RuntimeError, match=r"static"):
        _compute_dispersion(models, frequencies, weights)


def test_dispersion_accepts_any_positive_protocol_grid_scale():
    for scale in (0.25, 0.5, 1.0):
        frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, scale)
        models = _uniform_dispersion_models([[1.0, 0.5, 0.25]], len(frequencies))
        result = _compute_dispersion(models, frequencies, weights)
        assert result["inferred_scale"] == pytest.approx(scale, rel=1.0e-12)
        assert result["protocol_grid_enforced"] is True
        assert result["c6"].get(0, 0) == pytest.approx(
            (3.0 / math.pi) * sum(weights), rel=1.0e-13
        )


def test_dispersion_rejects_frequency_grid_mismatch():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    models = _uniform_dispersion_models([[1.0, 0.5, 0.25]], len(frequencies))

    with pytest.raises(RuntimeError, match=r"eleven"):
        _compute_dispersion(models[:-1], frequencies[:-1], weights[:-1])

    shifted = list(frequencies)
    shifted[5] *= 1.000001
    with pytest.raises(RuntimeError, match=r"make_casimir_grid"):
        _compute_dispersion(models, shifted, weights)

    halved = [0.5 * weight for weight in weights]
    with pytest.raises(RuntimeError, match=r"make_casimir_grid"):
        _compute_dispersion(models, frequencies, halved)


def test_dispersion_rejects_a_model_count_that_disagrees_with_the_grid():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    models = _uniform_dispersion_models([[1.0, 0.5, 0.25]], len(frequencies))
    tensors = [_matrix(tensor) for frequency in models[:-1] for tensor in frequency]

    with pytest.raises(RuntimeError, match=r"one model per grid frequency"):
        psi4.core._atomic_polarizability_compute_dispersion(
            _matrix(_DISPERSION_SITES[:1]),
            list(frequencies)[:-1],
            tensors,
            list(frequencies),
            list(weights),
        )


def test_dispersion_plan_bounds_storage_before_allocation():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    models = _uniform_dispersion_models([[1.0, 0.5, 0.25], [0.7, 0.3, 0.1]], len(frequencies))
    result = _compute_dispersion(models, frequencies, weights)
    plan = result["plan"]

    assert result["frequency_count"] == 11
    assert result["weighted_frequency_count"] == 10
    assert plan["frequency_count"] == 11
    assert plan["site_count"] == 2
    assert plan["coefficient_count"] == 4
    assert plan["rank_pair_count"] == len(_DISPERSION_RANK_PAIRS)
    assert plan["isotropic_elements"] == 11 * 2 * 3
    assert plan["coefficient_elements"] == 4 * 2 * 2
    assert plan["contribution_elements"] == len(_DISPERSION_RANK_PAIRS) * 2 * 2
    assert plan["work_terms"] == len(_DISPERSION_RANK_PAIRS) * 2 * 2 * 11
    assert len(result["rank_pair_contributions"]) == plan["contribution_elements"]
    assert plan["estimated_bytes"] <= plan["reserved_memory_bytes"]
    assert plan["algorithm"]
    assert plan["memory_semantics"]

    with pytest.raises(RuntimeError, match=r"site envelope"):
        psi4.core._atomic_polarizability_plan_dispersion(11, 4096, 1 << 30)
