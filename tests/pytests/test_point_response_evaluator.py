from pathlib import Path

import numpy as np
import pytest

import psi4
from test_native_atomic_polarizability_source_guard import (
    _without_cpp_comments,
    source_violations,
)


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints, pytest.mark.scf]


@pytest.fixture(scope="module")
def h2o_point_response_case():
    psi4.core.be_quiet()
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "pk",
            "reference": "rhf",
            "dft_spherical_points": 50,
            "dft_radial_points": 12,
            "dft_density_tolerance": 1.0e-12,
            "dft_grac_shift": 0.0,
        }
    )
    neutral = psi4.geometry(
        """
        0 1
        O  0.000000  0.000000  0.000000
        H  0.000000  0.757160  0.586260
        H  0.000000 -0.757160  0.586260
        symmetry c1
        units angstrom
        """
    )
    _, precursor = psi4.energy("pbe0", molecule=neutral, return_wfn=True)
    cation = psi4.geometry(
        """
        1 2
        O  0.000000  0.000000  0.000000
        H  0.000000  0.757160  0.586260
        H  0.000000 -0.757160  0.586260
        symmetry c1
        units angstrom
        """
    )
    psi4.set_options({"reference": "uhf", "dft_grac_shift": 0.0})
    _, cation_wfn = psi4.energy("pbe0", molecule=cation, return_wfn=True)
    homo = np.max(np.asarray(precursor.epsilon_a_subset("SO", "OCC")))
    shift = cation_wfn.energy() - precursor.energy() + homo
    psi4.set_options({"reference": "rhf", "dft_grac_shift": shift})
    _, grac = psi4.energy("pbe0", molecule=neutral, return_wfn=True)
    psi4.set_options({"dft_grac_shift": 0.0})

    context = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation_wfn
    )
    c1 = psi4.core._atomic_polarizability_test_restricted_c1_primitives(context, {})
    alda = psi4.core._atomic_polarizability_test_restricted_alda_kernel(context, False)
    assembled = psi4.core._atomic_polarizability_assemble_restricted_hessian(
        c1["orbital_gaps"], c1["coulomb"], c1["exchange_direct"],
        c1["exchange_transpose"], alda["full_alda"], 0.25, 0.75
    )
    h1 = np.asarray(assembled["H1"])
    h2 = np.asarray(assembled["H2"])
    assert h1.shape[0] == 5 * 2  # occupied-major and genuinely multi-transition
    return context, grac, h1, h2


def _canonical(case, points, frequencies=(0.0,), **options):
    context, _, _, _ = case
    return psi4.core._atomic_polarizability_evaluate_point_response(
        context, 0.25, 0.75, points, list(frequencies), options
    )


def _raw(case, points, frequencies, h1, h2, permutation=None, **options):
    context, _, _, _ = case
    if permutation is None:
        permutation = []
    return psi4.core._test_only_raw_point_response(
        context, points, list(frequencies), psi4.core.Matrix.from_array(h1),
        psi4.core.Matrix.from_array(h2), permutation, options
    )


def _external_transition_potentials(wfn, points):
    coefficients = np.asarray(wfn.Ca())
    occupations = np.asarray(wfn.occupation_a()).ravel()
    occupied = np.flatnonzero(occupations == 1.0)
    virtual = np.flatnonzero(occupations == 0.0)
    expected = []
    for point in points:
        external = psi4.core.ExternalPotential()
        external.addCharge(-1.0, *point)
        ao_potential = np.asarray(external.computePotentialMatrix(wfn.basisset()))
        expected.append(
            (coefficients[:, occupied].T @ ao_potential @ coefficients[:, virtual]).ravel()
        )
    return np.asarray(expected)


def _numpy_response(h1, h2, omega, point_major_potentials):
    nov = h1.shape[0]
    rhs = point_major_potentials.T
    if omega == 0.0:
        amplitudes = np.linalg.solve(h1, rhs)
    else:
        doubled = np.block(
            [[h1, omega * np.eye(nov)], [-omega * np.eye(nov), h2]]
        )
        amplitudes = np.linalg.solve(
            doubled, np.vstack((rhs, np.zeros_like(rhs)))
        )[:nov]
    raw = 4.0 * point_major_potentials @ amplitudes
    return 0.5 * (raw + raw.T)


def test_canonical_one_two_point_static_dynamic_shapes_and_metadata(h2o_point_response_case):
    one = _canonical(h2o_point_response_case, [[0.3, 0.2, 2.5]], [0.0, 0.4])
    assert np.asarray(one["transition_potentials"]).shape == (1, 10)
    assert one["frequencies"] == [0.0, 0.4]
    assert len(one["responses"]) == len(one["diagnostics"]) == 2

    points = [[0.3, 0.2, 2.5], [-0.4, 0.1, -2.2]]
    result = _canonical(h2o_point_response_case, points, [0.0, 0.4])
    assert result["points"] == points
    assert result["frequencies"] == [0.0, 0.4]
    assert result["operator_provenance"] == "CANONICAL_C1_PLUS_FULL_ALDA"
    for matrix, diagnostic in zip(result["responses"], result["diagnostics"]):
        matrix = np.asarray(matrix)
        assert matrix.shape == (2, 2)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_array_equal(matrix, matrix.T)
        assert diagnostic["reciprocity_enforced"] is True
        assert diagnostic["max_scaled_residual"] <= 1.0e-11
        assert diagnostic["max_forward_error"] <= 1.0e-8
        assert diagnostic["max_solution_scale"] >= 0.0
        for transient_vector in (
            "forward_error", "backward_error", "scaled_residual",
            "solution_column_scales",
        ):
            assert transient_vector not in diagnostic


def test_multi_transition_potential_matches_independent_external_potential_layout(
    h2o_point_response_case,
):
    _, grac, _, _ = h2o_point_response_case
    points = [[0.31, -0.27, 2.13], [-0.45, 0.19, -2.44]]
    actual = _canonical(h2o_point_response_case, points, [0.0])
    expected = _external_transition_potentials(grac, points)
    assert expected.shape == (2, 10)
    assert np.asarray(actual["transition_potentials"]) == pytest.approx(
        expected, abs=3.0e-14
    )
    assert actual["potential_convention"] == (
        "native electronic AO multipole-potential sign; the sign cancels in the bilinear response"
    )
    assert actual["transition_order"] == "(i,a) occupied-major/virtual-minor"


@pytest.mark.parametrize("omega", [0.0, 0.35])
def test_canonical_response_matches_independent_numpy_static_and_doubled_solve(
    h2o_point_response_case, omega
):
    _, _, h1, h2 = h2o_point_response_case
    points = [[0.2, 0.4, 2.0], [-0.3, 0.1, -2.3]]
    actual = _canonical(h2o_point_response_case, points, [omega])
    potentials = np.asarray(actual["transition_potentials"])
    expected = _numpy_response(h1, h2, omega, potentials)
    assert np.asarray(actual["responses"][0]) == pytest.approx(expected, abs=4.0e-12)


def test_raw_operator_and_transition_permutation_are_test_only_and_do_not_mutate_production(
    h2o_point_response_case,
):
    _, _, h1, h2 = h2o_point_response_case
    points = [[0.2, 0.4, 2.0], [-0.3, 0.1, -2.3]]
    canonical_before = _canonical(h2o_point_response_case, points, [0.0])
    permutation = list(reversed(range(h1.shape[0])))
    permuted_h1 = h1[np.ix_(permutation, permutation)]
    permuted_h2 = h2[np.ix_(permutation, permutation)]
    raw = _raw(
        h2o_point_response_case, points, [0.0], permuted_h1, permuted_h2,
        permutation=permutation
    )
    canonical_after = _canonical(h2o_point_response_case, points, [0.0])
    assert raw["operator_provenance"] == "TEST_ONLY_UNPROVENANCED_RAW_H1_H2"
    assert np.asarray(raw["responses"][0]) == pytest.approx(
        np.asarray(canonical_before["responses"][0]), abs=4.0e-12
    )
    np.testing.assert_array_equal(
        np.asarray(canonical_after["responses"][0]),
        np.asarray(canonical_before["responses"][0]),
    )
    assert not hasattr(psi4.core, "_atomic_polarizability_test_evaluate_point_response")
    with pytest.raises(RuntimeError, match="ALDA coefficient"):
        psi4.core._atomic_polarizability_evaluate_point_response(
            h2o_point_response_case[0], 0.25, 0.0, points, [0.0], {}
        )


def test_far_field_transition_scaling_and_signed_potential(h2o_point_response_case):
    result = _canonical(
        h2o_point_response_case,
        [[0.0, 0.0, 30.0], [0.0, 0.0, 60.0], [0.0, 0.0, -30.0]], [0.0]
    )
    potential = np.asarray(result["transition_potentials"])
    assert np.linalg.norm(potential[0]) / np.linalg.norm(potential[1]) == pytest.approx(
        4.0, rel=1.5e-2
    )
    assert np.dot(potential[0], potential[2]) < 0.0
    response = np.asarray(result["responses"][0])
    assert response[0, 2] < 0.0 < response[0, 0]


def test_point_permutation_duplicate_and_site_separation_policies(h2o_point_response_case):
    points = [[0.2, 0.4, 2.0], [-0.3, 0.1, -2.3], [0.6, -0.2, 1.8]]
    permutation = [2, 0, 1]
    base = _canonical(h2o_point_response_case, points, [0.0])
    permuted = _canonical(
        h2o_point_response_case, [points[index] for index in permutation], [0.0]
    )
    expected = np.asarray(base["responses"][0])[np.ix_(permutation, permutation)]
    assert np.asarray(permuted["responses"][0]) == pytest.approx(expected, abs=4.0e-12)
    with pytest.raises(RuntimeError, match="distinct"):
        _canonical(h2o_point_response_case, [points[0], points[0]], [0.0])
    with pytest.raises(RuntimeError, match="minimum site distance"):
        _canonical(
            h2o_point_response_case, [[0.0, 0.0, 0.0]], [0.0],
            minimum_site_distance_bohr=0.2
        )
    assert np.all(np.isfinite(np.asarray(
        _canonical(h2o_point_response_case, [[0.0, 0.0, 0.0]], [0.0])["responses"][0]
    )))


@pytest.mark.parametrize(
    "points,frequencies,options,message",
    [
        ([], [0.0], {}, "at least one point"),
        ([[0.0, 0.0, np.nan]], [0.0], {}, "finite"),
        ([[0.0, 0.0, 2.0]], [], {}, "frequency"),
        ([[0.0, 0.0, 2.0]], [np.inf], {}, "finite"),
        ([[0.0, 0.0, 2.0]], [-0.1], {}, "nonnegative"),
        ([[0.0, 0.0, 2.0]], [0.0], {"minimum_site_distance_bohr": -1.0}, "nonnegative"),
    ],
)
def test_malformed_and_nonfinite_inputs_fail_closed(
    h2o_point_response_case, points, frequencies, options, message
):
    with pytest.raises(RuntimeError, match=message):
        _canonical(h2o_point_response_case, points, frequencies, **options)


@pytest.mark.parametrize(
    "h1,h2,message",
    [
        (np.eye(2), np.eye(2), "dimension"),
        (np.full((10, 10), np.inf), np.eye(10), "finite"),
        (np.eye(10) + np.diag([1.0e-4] * 9, 1), np.eye(10), "symmetric"),
    ],
)
def test_raw_unprovenanced_operator_validation(
    h2o_point_response_case, h1, h2, message
):
    with pytest.raises(RuntimeError, match=message):
        _raw(h2o_point_response_case, [[0.0, 0.0, 2.0]], [0.0], h1, h2)
    with pytest.raises(RuntimeError, match="permutation"):
        _raw(
            h2o_point_response_case, [[0.0, 0.0, 2.0]], [0.0],
            h2o_point_response_case[2], h2o_point_response_case[3],
            permutation=[0] * 10
        )


def test_aggregate_resource_gate_precedes_snapshot_and_canonical_construction():
    source = (
        Path(__file__).resolve().parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    start = source.index("PointResponseData evaluate_point_response(")
    end = source.index("Matrix lw_graph_operator", start)
    body = _without_cpp_comments(source[start:end])
    assert body.index("preflight_isapol_response_provider") < body.index(
        "plan_point_response_provider"
    ) < body.index("verify_basis_unchanged") < body.index(
        "construct_restricted_c1_primitives"
    ) < body.index("construct_restricted_alda_kernel")
    for required in (
        "construct_restricted_c1_primitives", "construct_restricted_alda_kernel",
        "assemble_restricted_singlet_hessian",
    ):
        assert body.count(required) == 1


def test_resource_plan_accounts_for_canonical_stages_and_python_output_clones(
    h2o_point_response_case,
):
    with pytest.raises(RuntimeError, match="500"):
        _canonical(
            h2o_point_response_case,
            [[0.0, 0.0, 3.0 + index] for index in range(501)], [0.0]
        )
    estimate = psi4.core._atomic_polarizability_estimate_point_response
    standalone = estimate(2, 7, 5, 2, 500, True, 1 << 30)
    assert standalone["output_clone_bytes"] == standalone["output_bytes"]
    assert standalone["retained_frequency_bytes"] == 2 * 8
    assert standalone["retained_points_bytes"] == 500 * 3 * 8
    assert standalone["native_diagnostics_bytes"] == (
        2 * standalone["native_diagnostic_record_bytes"]
    )
    with pytest.raises(RuntimeError, match="reserved memory"):
        estimate(2, 7, 5, 2, 500, True, standalone["estimated_bytes"] * 2 - 2)

    # With one point and one transition, the retained scalar diagnostics and
    # Python scalar-object budget dominate the numeric response payload.
    diagnostic_plan = estimate(64, 1, 1, 1, 1, True, 1 << 30)
    assert diagnostic_plan["max_frequency_count"] == 64
    assert diagnostic_plan["retained_frequency_bytes"] == 64 * 8
    assert diagnostic_plan["retained_points_bytes"] == 3 * 8
    assert diagnostic_plan["native_diagnostics_bytes"] == (
        64 * diagnostic_plan["native_diagnostic_record_bytes"]
    )
    assert diagnostic_plan["python_scalar_diagnostic_overhead_bytes"] == 64 * 512
    assert diagnostic_plan["python_metadata_overhead_bytes"] == 1024 + 64 * 32 + 128
    assert diagnostic_plan["python_export_overhead_bytes"] == (
        diagnostic_plan["python_scalar_diagnostic_overhead_bytes"]
        + diagnostic_plan["python_metadata_overhead_bytes"]
    )
    assert diagnostic_plan["retained_metadata_bytes"] == (
        diagnostic_plan["retained_frequency_bytes"]
        + diagnostic_plan["retained_points_bytes"]
        + diagnostic_plan["native_diagnostics_bytes"]
        + diagnostic_plan["container_overhead_bytes"]
    )
    assert diagnostic_plan["retained_metadata_bytes"] > 2 * diagnostic_plan["output_bytes"]
    estimate(64, 1, 1, 1, 1, True, diagnostic_plan["estimated_bytes"] * 2)
    with pytest.raises(RuntimeError, match="reserved memory"):
        estimate(64, 1, 1, 1, 1, True, diagnostic_plan["estimated_bytes"] * 2 - 2)
    with pytest.raises(RuntimeError, match="64"):
        estimate(65, 1, 1, 1, 1, True, 1 << 30)

    result = _canonical(h2o_point_response_case, [[0.0, 0.0, 2.0]], [0.0, 0.2])
    plan = result["plan"]
    assert plan["max_point_count"] == 500
    assert plan["c1_plan_estimated_bytes"] > 0
    assert plan["alda_plan_estimated_bytes"] > 0
    assert plan["retained_c1_bytes"] > 0
    assert plan["retained_alda_bytes"] > 0
    assert plan["hessian_bytes"] > 0
    assert plan["output_clone_bytes"] == plan["output_bytes"]
    assert plan["point_potential_stage_peak_bytes"] > plan["transition_potential_bytes"]
    assert plan["dense_solve_stage_peak_bytes"] > plan["dense_solve_peak_bytes"]
    assert plan["output_clone_stage_peak_bytes"] >= 2 * plan["output_bytes"]
    assert plan["estimated_bytes"] == max(
        plan["c1_stage_peak_bytes"], plan["alda_stage_peak_bytes"],
        plan["point_potential_stage_peak_bytes"], plan["dense_solve_stage_peak_bytes"],
        plan["output_clone_stage_peak_bytes"],
    )
    assert plan["memory_semantics"] == (
        "KNOWN_STORAGE_HARD_GATE_DIRECT_JK_WORKSPACE_ADVISORY_PYTHON_CLONES_SCALAR_DIAGNOSTICS_FREQ64"
    )


def test_source_guard_allows_only_native_order_zero_integrals_and_no_process_route():
    source = (
        Path(__file__).resolve().parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    start = source.index("PointResponseData::PointResponseData")
    end = source.index("Matrix lw_graph_operator", start)
    evaluator = source[start:end]
    active = _without_cpp_comments(evaluator)
    assert active.count("ao_multipole_potential(0,") == 1
    assert "ao_multipoles" not in active
    assert "compute_isa" not in active.lower()
    assert "generate" not in active.lower()
    assert source_violations(evaluator) == []
