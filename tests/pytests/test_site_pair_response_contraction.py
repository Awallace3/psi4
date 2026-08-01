import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

_COMPONENT_ORDER = "00;10,11c,11s;20,21c,21s,22c,22s;30,31c,31s,32c,32s,33c,33s"
_SYMMETRY_ABSOLUTE_TOLERANCE = 1.0e-10
_SYMMETRY_RELATIVE_TOLERANCE = 1.0e-10


def _contract(projection, response_map, site_count=None):
    projection = np.asarray(projection, dtype=float)
    if site_count is None:
        site_count = projection.shape[0] // 16
    result = psi4.core._atomic_polarizability_test_contract_site_pair_response(
        site_count,
        psi4.core.Matrix.from_array(projection),
        psi4.core.Matrix.from_array(np.asarray(response_map, dtype=float)),
    )
    return np.asarray(result["values"]), result


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
        pytest.param(np.diag([1.0 / 2.0, 1.0 / 5.0]), id="supplied-static-analytic-G"),
        pytest.param(
            np.diag([2.0 / (4.0 + 0.75**2), 5.0 / (25.0 + 0.75**2)]),
            id="supplied-finite-frequency-analytic-G",
        ),
        pytest.param(np.zeros((2, 2)), id="supplied-zero-G"),
    ],
)
def test_supplied_analytic_response_maps(response_map):
    projection = np.array([[1.0, -0.5], [0.25, 2.0]] + [[0.0, 0.0]] * 14)
    actual, _ = _contract(projection, response_map)
    assert actual == pytest.approx(4.0 * projection @ response_map @ projection.T, abs=2.0e-15)


@pytest.mark.parametrize(
    "projection,response_map,site_count,message",
    [
        (np.ones((16, 1)), np.ones((1, 1)), 0, "site count"),
        (np.ones((16, 1)), np.ones((1, 1)), 2, "dimensions"),
        (np.ones((16, 2)), np.ones((1, 1)), 1, "dimensions"),
        (np.ones((16, 1)), np.ones((1, 2)), 1, "square"),
        (np.full((16, 1), np.nan), np.ones((1, 1)), 1, "finite"),
        (np.ones((16, 1)), np.full((1, 1), np.inf), 1, "finite"),
        (np.ones((16, 2)), np.array([[1.0, 1.0e-4], [0.0, 1.0]]), 1, "symmetric"),
    ],
)
def test_malformed_nonfinite_and_asymmetric_inputs_fail(projection, response_map, site_count, message):
    with pytest.raises(RuntimeError, match=message):
        _contract(projection, response_map, site_count)


@pytest.mark.parametrize("omega", [0.0, 0.7])
def test_dense_solver_identity_rhs_response_map_passes_symmetry_gate(omega):
    h1 = np.array([[3.0, 0.4, 0.2], [0.4, 2.0, -0.3], [0.2, -0.3, 1.7]])
    h2 = np.array([[1.8, -0.2, 0.1], [-0.2, 2.6, 0.35], [0.1, 0.35, 2.2]])
    assert np.max(np.abs(h1 @ h2 - h2 @ h1)) > 0.1
    identity = psi4.core.Matrix.from_array(np.eye(3))
    solved = psi4.core._atomic_polarizability_solve_restricted_response(
        psi4.core.Matrix.from_array(h1), psi4.core.Matrix.from_array(h2), omega, identity
    )
    response_map = np.asarray(solved["P"])
    raw_asymmetry = np.max(np.abs(response_map - response_map.T))
    projection = np.arange(48, dtype=float).reshape(16, 3) / 17.0

    actual, meta = _contract(projection, response_map)

    assert raw_asymmetry <= (
        _SYMMETRY_ABSOLUTE_TOLERANCE
        + _SYMMETRY_RELATIVE_TOLERANCE * np.max(np.abs(response_map))
    )
    assert meta["response_map_symmetry_residual"] == pytest.approx(raw_asymmetry, abs=0.0)
    assert np.all(np.isfinite(actual))


def test_near_symmetric_roundoff_policy_uses_independently_symmetrized_numpy_oracle():
    projection = np.arange(32, dtype=float).reshape(16, 2) / 17.0
    response_map = np.array([[1.0, 0.25 + 2.0e-11], [0.25, 0.8]])
    symmetric_response_map = (response_map + response_map.T) / 2.0

    actual, meta = _contract(projection, response_map)

    assert meta["response_map_symmetry_policy"] == "AVERAGE_WITHIN_DENSE_SOLVER_RESIDUAL_TOLERANCE"
    assert meta["response_map_symmetry_absolute_tolerance"] == _SYMMETRY_ABSOLUTE_TOLERANCE
    assert meta["response_map_symmetry_relative_tolerance"] == _SYMMETRY_RELATIVE_TOLERANCE
    assert meta["response_map_symmetry_residual"] == pytest.approx(2.0e-11, rel=2.0e-6)
    assert actual == pytest.approx(
        4.0 * projection @ symmetric_response_map @ projection.T, abs=2.0e-13
    )
    assert meta["reciprocity_enforced"] is True


@pytest.mark.parametrize(
    "scale,residual_factor,accepted",
    [
        pytest.param(0.0, 0.99, True, id="below-absolute-boundary"),
        pytest.param(0.0, 1.01, False, id="above-absolute-boundary"),
        pytest.param(1.0e6, 0.99, True, id="below-relative-boundary"),
        pytest.param(1.0e6, 1.01, False, id="above-relative-boundary"),
    ],
)
def test_symmetry_gate_absolute_and_relative_boundaries(scale, residual_factor, accepted):
    projection = np.ones((16, 2))
    boundary = _SYMMETRY_ABSOLUTE_TOLERANCE + _SYMMETRY_RELATIVE_TOLERANCE * scale
    response_map = np.array([[2.0, scale], [scale + residual_factor * boundary, 3.0]])

    if accepted:
        actual, meta = _contract(projection, response_map)
        assert np.all(np.isfinite(actual))
        assert meta["response_map_symmetry_residual"] == pytest.approx(
            residual_factor * boundary, rel=2.0e-6, abs=1.0e-16
        )
    else:
        with pytest.raises(RuntimeError, match="symmetric"):
            _contract(projection, response_map)


def test_plan_exact_incremental_allocation_arithmetic_and_success_boundary():
    estimate = psi4.core._atomic_polarizability_estimate_site_pair_response_contraction

    small = estimate(2, 3, 1 << 30)
    assert small["component_count"] == 32
    assert small["output_bytes"] == 32 * 32 * 8
    assert small["scratch_bytes"] == (32 * 3 + 3 * 3) * 8
    assert small["estimated_bytes"] == 1024 * 1024 + small["output_bytes"] + small["scratch_bytes"]
    assert small["work_terms"] == 32 * 3 * 3 + 32 * 32 * 3
    assert small["memory_semantics"] == "INCREMENTAL_INTERNAL_ALLOCATIONS_CALLER_B_AND_G_EXCLUDED"

    boundary = estimate(64, 512, (1 << 64) - 1)
    assert boundary["component_count"] == 1024
    assert boundary["output_bytes"] == 1024 * 1024 * 8
    assert boundary["scratch_bytes"] == (1024 * 512 + 512 * 512) * 8
    assert boundary["estimated_bytes"] == 15 * 1024 * 1024
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
