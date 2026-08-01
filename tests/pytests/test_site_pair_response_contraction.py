from pathlib import Path

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

_COMPONENT_ORDER = "00;10,11c,11s;20,21c,21s,22c,22s;30,31c,31s,32c,32s,33c,33s"


def _contract(projection, response_map, site_count=None):
    """Acquire the immutable production carrier from a static identity-RHS solve."""
    response_map = np.asarray(response_map, dtype=float)
    h1 = np.linalg.inv(response_map)
    h2 = np.eye(response_map.shape[0])
    return _solve_and_contract(projection, h1, h2, 0.0, site_count)


def _solve_and_contract(projection, h1, h2, omega, site_count=None):
    projection = np.asarray(projection, dtype=float)
    if site_count is None:
        site_count = projection.shape[0] // 16
    result = psi4.core._atomic_polarizability_test_solve_and_contract_site_pair_response(
        site_count,
        psi4.core.Matrix.from_array(projection),
        psi4.core.Matrix.from_array(np.asarray(h1, dtype=float)),
        psi4.core.Matrix.from_array(np.asarray(h2, dtype=float)),
        omega,
    )
    return np.asarray(result["values"]), result


def _validate_symmetry(response_map, q_map, forward_errors):
    return psi4.core._atomic_polarizability_test_validate_response_map_symmetry(
        psi4.core.Matrix.from_array(np.asarray(response_map, dtype=float)),
        psi4.core.Matrix.from_array(np.asarray(q_map, dtype=float)),
        np.atleast_1d(forward_errors).tolist(),
    )


def _translation_matrix(displacement):
    columns = []
    for component in range(16):
        unit = np.zeros(16)
        unit[component] = 1.0
        columns.append(
            psi4.core._atomic_polarizability_translate_l3(unit.tolist(), displacement)
        )
    return np.asarray(columns).T


def test_one_transition_literal_has_the_restricted_factor_exactly_once():
    projection = np.zeros((32, 1))
    projection[0, 0] = 2.0
    projection[16 + 3, 0] = -3.0

    actual, meta = _contract(projection, [[5.0]])

    expected = np.zeros((32, 32))
    expected[0, 0] = 80.0
    expected[0, 19] = -120.0
    expected[19, 0] = -120.0
    expected[19, 19] = 180.0
    np.testing.assert_array_equal(actual, expected)
    assert meta["restricted_factor"] == 4.0
    assert meta["component_order"] == _COMPONENT_ORDER
    assert meta["block_order"] == "row=(response_site,ISA_component); column=(source_site,ISA_component)"


def test_two_site_two_transition_matches_independent_numpy_oracle_and_reciprocity():
    projection = (np.arange(64, dtype=float).reshape(32, 2) - 17.0) / 11.0
    response_map = np.array([[1.25, -0.4], [-0.4, 0.75]])

    actual, meta = _contract(projection, response_map)
    expected = 4.0 * projection @ response_map @ projection.T

    assert actual == pytest.approx(expected, abs=2.0e-13)
    assert actual[:16, 16:] == pytest.approx(actual[16:, :16].T, abs=2.0e-13)
    assert meta["response_map_symmetry_residual"] == 0.0
    assert meta["reciprocity_enforced"] is True
    assert meta["site_count"] == 2
    assert meta["transition_count"] == 2


def test_rank_zero_charge_flow_sum_rule_covers_every_mixed_rank_column_and_row():
    rng = np.random.default_rng(731)
    projection = rng.normal(size=(48, 3))
    projection[0] = [1.0, -2.0, 0.4]
    projection[16] = [-0.25, 0.5, 1.1]
    projection[32] = -(projection[0] + projection[16])
    assert np.count_nonzero(projection[[*range(1, 16), *range(17, 32), *range(33, 48)]]) > 0
    response_map = np.array([[0.7, 0.2, -0.1], [0.2, 1.3, 0.35], [-0.1, 0.35, 0.9]])

    actual, _ = _contract(projection, response_map)
    monopoles = [0, 16, 32]

    assert actual[monopoles, :].sum(axis=0) == pytest.approx(np.zeros(48), abs=2.0e-14)
    assert actual[:, monopoles].sum(axis=1) == pytest.approx(np.zeros(48), abs=2.0e-14)


def test_common_origin_recovery_with_existing_l3_translation_matrices():
    rng = np.random.default_rng(814)
    sites = [[-0.4, 0.2, 0.1], [0.7, -0.3, 0.5]]
    projection = rng.normal(size=(32, 3))
    response_map = np.array([[1.2, -0.1, 0.3], [-0.1, 0.8, 0.25], [0.3, 0.25, 1.5]])

    atomic, _ = _contract(projection, response_map)
    translations = [_translation_matrix(site) for site in sites]
    common_projection = sum(
        translations[site] @ projection[16 * site:16 * (site + 1)]
        for site in range(2)
    )
    recovered = sum(
        translations[a] @ atomic[16 * a:16 * (a + 1), 16 * b:16 * (b + 1)] @ translations[b].T
        for a in range(2) for b in range(2)
    )

    assert recovered == pytest.approx(
        4.0 * common_projection @ response_map @ common_projection.T, abs=2.0e-10
    )


def test_site_permutation_covariance_and_component_order_untouched():
    rng = np.random.default_rng(912)
    projection = rng.normal(size=(48, 2))
    response_map = np.array([[0.9, -0.35], [-0.35, 1.1]])
    base, _ = _contract(projection, response_map)
    permutation = [2, 0, 1]
    permuted_projection = np.concatenate(
        [projection[16 * site:16 * (site + 1)] for site in permutation]
    )

    permuted, _ = _contract(permuted_projection, response_map)
    index = np.concatenate([np.arange(16 * site, 16 * (site + 1)) for site in permutation])

    assert permuted == pytest.approx(base[np.ix_(index, index)], abs=2.0e-13)


@pytest.mark.parametrize(
    "response_map",
    [
        pytest.param(np.linalg.inv([[2.0, 0.1], [0.1, 5.0]]), id="supplied-static-SPD-G-1"),
        pytest.param(np.linalg.inv([[4.25, -0.2], [-0.2, 7.5]]), id="supplied-static-SPD-G-2"),
    ],
)
def test_supplied_analytic_response_maps(response_map):
    projection = np.array([[1.0, -0.5], [0.25, 2.0]] + [[0.0, 0.0]] * 14)
    actual, _ = _contract(projection, response_map)
    assert actual == pytest.approx(4.0 * projection @ response_map @ projection.T, abs=2.0e-15)


def test_zero_projection_gives_zero_response_matrix_through_solved_carrier():
    actual, _ = _contract(np.zeros((16, 2)), np.linalg.inv([[2.0, 0.1], [0.1, 5.0]]))
    np.testing.assert_array_equal(actual, np.zeros((16, 16)))


@pytest.mark.parametrize(
    "projection,h1,h2,site_count,message",
    [
        (np.ones((16, 1)), [[1.0]], [[1.0]], 0, "site count"),
        (np.ones((16, 1)), [[1.0]], [[1.0]], 2, "dimensions"),
        (np.ones((16, 2)), [[1.0]], [[1.0]], 1, "dimensions"),
        (np.ones((16, 1)), [[1.0, 0.0]], [[1.0]], 1, "square"),
        (np.full((16, 1), np.nan), [[1.0]], [[1.0]], 1, "finite"),
        (np.ones((16, 1)), [[np.inf]], [[1.0]], 1, "finite"),
        (np.ones((16, 2)), [[1.0, 1.0e-4], [0.0, 1.0]], np.eye(2), 1, "symmetric"),
    ],
)
def test_malformed_nonfinite_and_asymmetric_inputs_fail(projection, h1, h2, site_count, message):
    with pytest.raises(RuntimeError, match=message):
        _solve_and_contract(projection, h1, h2, 0.0, site_count)


@pytest.mark.parametrize("omega", [0.0, 0.7])
def test_dense_solver_identity_rhs_response_map_passes_symmetry_gate(omega):
    h1 = np.array([[3.0, 0.4, 0.2], [0.4, 2.0, -0.3], [0.2, -0.3, 1.7]])
    h2 = np.array([[1.8, -0.2, 0.1], [-0.2, 2.6, 0.35], [0.1, 0.35, 2.2]])
    assert np.max(np.abs(h1 @ h2 - h2 @ h1)) > 0.1
    projection = np.arange(48, dtype=float).reshape(16, 3) / 17.0

    actual, meta = _solve_and_contract(projection, h1, h2, omega)
    response_map = np.asarray(meta["P"])
    q_map = np.asarray(meta["Q"])
    raw_asymmetry = np.max(np.abs(response_map - response_map.T))
    forward_errors = np.asarray(meta["forward_error"])
    column_scales = np.asarray(meta["solution_column_scales"])
    pair_bounds = [
        forward_errors[j] * column_scales[j]
        + forward_errors[i] * column_scales[i]
        + 64.0 * np.finfo(float).eps * max(1.0, column_scales[i], column_scales[j]) * response_map.shape[0]
        for i in range(response_map.shape[0]) for j in range(i + 1, response_map.shape[0])
    ]
    expected_bound = max(pair_bounds)
    assert raw_asymmetry <= expected_bound
    assert meta["response_map_forward_error_bound"] == max(forward_errors)
    assert meta["response_map_allowed_antisymmetry"] == pytest.approx(expected_bound, rel=2.0e-15)
    assert meta["response_map_symmetry_residual"] == pytest.approx(raw_asymmetry, abs=0.0)
    assert len(meta["forward_error"]) == response_map.shape[0]
    assert len(meta["backward_error"]) == response_map.shape[0]
    assert len(meta["scaled_residual"]) == response_map.shape[0]
    assert len(meta["solution_column_scales"]) == response_map.shape[0]
    assert meta["max_backward_error"] <= 1.0e-11
    assert meta["max_scaled_residual"] <= 1.0e-11
    assert meta["reciprocal_pivot_growth"] >= 1.0e-12
    assert np.all(np.isfinite(actual))


def test_nonzero_solver_symmetry_bound_uses_full_solution_when_q_dominates_p():
    h1 = np.array([[2.0, 0.1], [0.1, 1.7]])
    h2 = np.array([[1.0e-8, 2.0e-9], [2.0e-9, 2.0e-8]])
    projection = np.arange(32, dtype=float).reshape(16, 2) / 17.0

    actual, meta = _solve_and_contract(projection, h1, h2, 0.01)
    p_map = np.asarray(meta["P"])
    q_map = np.asarray(meta["Q"])
    column_scales = np.maximum(np.max(np.abs(p_map), axis=0), np.max(np.abs(q_map), axis=0))
    forward_errors = np.asarray(meta["forward_error"])
    full_scale = max(column_scales)
    p_only_scale = max(1.0, np.max(np.abs(p_map)))
    expected = (
        forward_errors[0] * column_scales[0]
        + forward_errors[1] * column_scales[1]
        + 64.0 * np.finfo(float).eps * max(1.0, *column_scales) * 2
    )

    assert np.max(np.abs(q_map)) > 1.0e4 * np.max(np.abs(p_map))
    assert meta["solution_column_scales"] == pytest.approx(column_scales, rel=2.0e-15)
    assert meta["response_map_solution_scale"] == pytest.approx(full_scale, rel=2.0e-15)
    assert meta["response_map_allowed_antisymmetry"] == pytest.approx(expected, rel=2.0e-15)
    assert meta["response_map_allowed_antisymmetry"] > (
        50.0 * 64.0 * np.finfo(float).eps * p_only_scale * 2
    )
    assert np.all(np.isfinite(actual))


def test_near_symmetric_roundoff_policy_uses_independently_symmetrized_numpy_oracle():
    projection = np.arange(32, dtype=float).reshape(16, 2) / 17.0
    forward_error_bound = 2.0e-11
    response_map = np.array([[1.0, 0.25 + 2.0e-11], [0.25, 0.8]])
    symmetric_response_map = (response_map + response_map.T) / 2.0

    validation = _validate_symmetry(
        response_map, np.zeros_like(response_map), [forward_error_bound] * 2
    )
    actual, meta = _contract(projection, symmetric_response_map)

    assert validation["response_map_forward_error_bound"] == forward_error_bound
    assert validation["response_map_symmetry_residual"] == pytest.approx(2.0e-11, rel=2.0e-6)
    assert actual == pytest.approx(
        4.0 * projection @ symmetric_response_map @ projection.T, abs=2.0e-13
    )
    assert meta["reciprocity_enforced"] is True


@pytest.mark.parametrize(
    "forward_error_bound,residual_factor,accepted",
    [
        pytest.param(0.0, 0.9, True, id="below-machine-roundoff-bound"),
        pytest.param(0.0, 1.1, False, id="above-machine-roundoff-bound"),
        pytest.param(1.0e-9, 0.99, True, id="below-forward-error-bound"),
        pytest.param(1.0e-9, 1.01, False, id="above-forward-error-bound"),
    ],
)
def test_symmetry_gate_derived_boundaries(forward_error_bound, residual_factor, accepted):
    scale = 3.0
    dimension = 2
    forward_errors = [0.5 * forward_error_bound, forward_error_bound]
    boundary = (
        forward_errors[0] * 2.0
        + forward_errors[1] * scale
        + 64.0 * np.finfo(float).eps * scale * dimension
    )
    response_map = np.array([[2.0, 0.25], [0.25 + residual_factor * boundary, scale]])
    q_map = np.zeros_like(response_map)

    if accepted:
        validation = _validate_symmetry(response_map, q_map, forward_errors)
        assert validation["response_map_allowed_antisymmetry"] == pytest.approx(boundary, rel=2.0e-15)
        assert validation["response_map_symmetry_residual"] == pytest.approx(
            residual_factor * boundary, rel=2.0e-3, abs=1.0e-17
        )
        assert validation["response_map_max_normalized_antisymmetry"] == pytest.approx(
            residual_factor, rel=2.0e-3, abs=1.0e-4
        )
    else:
        with pytest.raises(RuntimeError, match="symmetric"):
            _validate_symmetry(response_map, q_map, forward_errors)


@pytest.mark.parametrize("forward_error_bound", [-1.0, np.nan, np.inf, 1.0001e-8])
def test_response_map_forward_error_bound_is_fail_closed(forward_error_bound):
    with pytest.raises(RuntimeError, match="forward error"):
        _validate_symmetry([[1.0]], [[0.0]], forward_error_bound)


def test_symmetry_validator_rejects_mismatched_or_nonfinite_per_rhs_data():
    with pytest.raises(RuntimeError, match="cardinalit"):
        _validate_symmetry(np.eye(2), np.zeros((2, 2)), [0.0])
    with pytest.raises(RuntimeError, match="finite"):
        _validate_symmetry(np.eye(2), [[0.0, np.nan], [0.0, 0.0]], [0.0, 0.0])


def test_no_python_binding_can_fabricate_or_mutate_a_production_response_carrier():
    assert not hasattr(psi4.core, "DenseRestrictedResponse")
    assert not hasattr(psi4.core, "_atomic_polarizability_test_contract_site_pair_response")
    header = (
        Path(__file__).resolve().parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.h"
    ).read_text()
    assert "const Matrix& P()" not in header
    assert "const Matrix& Q()" not in header
    assert "SharedMatrix P_clone() const" in header
    assert "SharedMatrix Q_clone() const" in header
    implementation = (
        Path(__file__).resolve().parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    contraction_start = implementation.index(
        "SitePairResponseContraction contract_site_pair_response"
    )
    contraction = implementation[contraction_start:]
    assert contraction.index("plan_site_pair_response_contraction") < contraction.index(
        "response.P_clone()"
    )


def test_mutating_exported_response_matrices_cannot_change_later_contraction():
    projection = np.arange(32, dtype=float).reshape(16, 2) / 17.0
    h1 = np.array([[2.0, 0.1], [0.1, 1.7]])
    h2 = np.array([[1.4, -0.2], [-0.2, 2.1]])
    first_values, first = _solve_and_contract(projection, h1, h2, 0.3)

    np.asarray(first["P"])[:] = 1.0e6
    np.asarray(first["Q"])[:] = -1.0e6
    second_values, second = _solve_and_contract(projection, h1, h2, 0.3)

    np.testing.assert_array_equal(second_values, first_values)
    assert np.max(np.abs(np.asarray(second["P"]))) < 1.0
    assert np.max(np.abs(np.asarray(second["Q"]))) < 1.0


def test_plan_exact_incremental_allocation_arithmetic_and_success_boundary():
    estimate = psi4.core._atomic_polarizability_estimate_site_pair_response_contraction

    small = estimate(2, 3, 1 << 30)
    assert small["component_count"] == 32
    assert small["output_bytes"] == 32 * 32 * 8
    assert small["scratch_bytes"] == (32 * 3 + 2 * 3 * 3) * 8
    assert small["estimated_bytes"] == small["output_bytes"] + small["scratch_bytes"]
    assert small["work_terms"] == 32 * 3 * 3 + 32 * 32 * 3
    assert small["memory_semantics"] == (
        "INCREMENTAL_NUMERIC_PAYLOAD_CALLER_B_AND_DENSE_RESPONSE_EXCLUDED"
    )
    exact = estimate(2, 3, small["estimated_bytes"])
    assert exact["estimated_bytes"] == small["estimated_bytes"]
    with pytest.raises(RuntimeError, match="memory"):
        estimate(2, 3, small["estimated_bytes"] - 1)

    boundary = estimate(64, 512, (1 << 64) - 1)
    assert boundary["component_count"] == 1024
    assert boundary["output_bytes"] == 1024 * 1024 * 8
    assert boundary["scratch_bytes"] == (1024 * 512 + 2 * 512 * 512) * 8
    assert boundary["estimated_bytes"] == 16 * 1024 * 1024
    assert boundary["work_terms"] == 805306368


def test_result_nonfiniteness_and_resource_envelopes_fail():
    with pytest.raises(RuntimeError, match="nonfinite|overflow"):
        _contract(np.full((16, 1), 1.0e308), [[1.0]])

    estimate = psi4.core._atomic_polarizability_estimate_site_pair_response_contraction
    with pytest.raises(RuntimeError, match="overflow"):
        estimate((1 << 64) - 1, 2, (1 << 64) - 1)
    with pytest.raises(RuntimeError, match="site count"):
        estimate(65, 2, 1 << 30)
    with pytest.raises(RuntimeError, match="response envelope"):
        estimate(2, 513, 1 << 30)
    with pytest.raises(RuntimeError, match="memory"):
        estimate(4, 20, 1024)
