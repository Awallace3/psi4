from pathlib import Path

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints, pytest.mark.scf]


@pytest.fixture(scope="module")
def h2_point_response_case():
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
        H 0 0 -0.70
        H 0 0  0.70
        symmetry c1
        units bohr
        """
    )
    _, precursor = psi4.energy("pbe0", molecule=neutral, return_wfn=True)

    cation = psi4.geometry(
        """
        1 2
        H 0 0 -0.70
        H 0 0  0.70
        symmetry c1
        units bohr
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
    primitives = psi4.core._atomic_polarizability_test_restricted_c1_primitives(
        context, {}
    )
    return context, grac, np.asarray(primitives["H1_zero_alda"]), np.asarray(
        primitives["H2_zero_alda"]
    )


def _evaluate(case, points, frequencies=(0.0,), **options):
    context, _, h1, h2 = case
    return psi4.core._atomic_polarizability_test_evaluate_point_response(
        context,
        points,
        list(frequencies),
        psi4.core.Matrix.from_array(h1),
        psi4.core.Matrix.from_array(h2),
        options,
    )


def test_one_and_two_point_static_dynamic_shapes_symmetry_and_finiteness(h2_point_response_case):
    one = _evaluate(h2_point_response_case, [[0.3, 0.2, 2.5]], [0.0, 0.4])
    assert np.asarray(one["transition_potentials"]).shape == (1, 1)
    assert len(one["responses"]) == 2
    assert [np.asarray(value).shape for value in one["responses"]] == [(1, 1), (1, 1)]

    points = [[0.3, 0.2, 2.5], [-0.4, 0.1, -2.2]]
    two = _evaluate(h2_point_response_case, points, [0.0, 0.4])
    assert two["points"] == points
    assert two["frequencies"] == [0.0, 0.4]
    assert len(two["diagnostics"]) == 2
    for matrix, diagnostic in zip(two["responses"], two["diagnostics"]):
        matrix = np.asarray(matrix)
        assert matrix.shape == (2, 2)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_array_equal(matrix, matrix.T)
        assert diagnostic["reciprocity_enforced"] is True
        assert diagnostic["max_scaled_residual"] <= 1.0e-11
        assert diagnostic["max_backward_error"] <= 1.0e-11


def test_transition_potential_matches_independent_mints_and_explicit_mo_layout(h2_point_response_case):
    _, grac, _, _ = h2_point_response_case
    points = [[0.31, -0.27, 2.13], [-0.45, 0.19, -2.44]]
    actual = _evaluate(h2_point_response_case, points, [0.0])

    coefficients = np.asarray(grac.Ca())
    occupations = np.asarray(grac.occupation_a()).ravel()
    occupied = np.flatnonzero(occupations == 1.0)
    virtual = np.flatnonzero(occupations == 0.0)
    mints = psi4.core.MintsHelper(grac.basisset())
    expected = []
    for point in points:
        ao_potential = np.asarray(mints.ao_multipole_potential(0, point)[0])
        expected.append((coefficients[:, occupied].T @ ao_potential @ coefficients[:, virtual]).ravel())

    assert np.asarray(actual["transition_potentials"]) == pytest.approx(
        np.asarray(expected), abs=2.0e-14
    )
    assert actual["potential_convention"] == (
        "native electronic AO multipole-potential sign; the sign cancels in the bilinear response"
    )
    assert actual["transition_order"] == "(i,a) occupied-major/virtual-minor"


@pytest.mark.parametrize("omega", [0.0, 0.35])
def test_point_response_matches_explicit_reviewed_solver_chain(h2_point_response_case, omega):
    _, _, h1, h2 = h2_point_response_case
    points = [[0.2, 0.4, 2.0], [-0.3, 0.1, -2.3]]
    actual = _evaluate(h2_point_response_case, points, [omega])
    potentials = np.asarray(actual["transition_potentials"])
    solved = psi4.core._atomic_polarizability_solve_restricted_response(
        psi4.core.Matrix.from_array(h1),
        psi4.core.Matrix.from_array(h2),
        omega,
        psi4.core.Matrix.from_array(potentials.T),
    )
    raw = 4.0 * potentials @ np.asarray(solved["P"])
    expected = 0.5 * (raw + raw.T)
    assert np.asarray(actual["responses"][0]) == pytest.approx(expected, abs=2.0e-13)


def test_point_permutation_covariance_and_duplicate_policy(h2_point_response_case):
    points = [[0.2, 0.4, 2.0], [-0.3, 0.1, -2.3], [0.6, -0.2, 1.8]]
    permutation = [2, 0, 1]
    base = _evaluate(h2_point_response_case, points, [0.0, 0.2])
    permuted = _evaluate(h2_point_response_case, [points[index] for index in permutation], [0.0, 0.2])
    for base_matrix, permuted_matrix in zip(base["responses"], permuted["responses"]):
        expected = np.asarray(base_matrix)[np.ix_(permutation, permutation)]
        assert np.asarray(permuted_matrix) == pytest.approx(expected, abs=2.0e-13)

    with pytest.raises(RuntimeError, match="distinct"):
        _evaluate(h2_point_response_case, [points[0], points[0]], [0.0])


def test_far_field_transition_scaling_and_signed_potential(h2_point_response_case):
    result = _evaluate(
        h2_point_response_case,
        [[0.0, 0.0, 30.0], [0.0, 0.0, 60.0], [0.0, 0.0, -30.0]],
        [0.0],
    )
    potential = np.asarray(result["transition_potentials"])[:, 0]
    assert potential[0] / potential[1] == pytest.approx(4.0, rel=8.0e-3)
    assert potential[0] == pytest.approx(-potential[2], rel=8.0e-3)
    response = np.asarray(result["responses"][0])
    assert response[0, 2] < 0.0 < response[0, 0]


def test_optional_minimum_site_separation_policy(h2_point_response_case):
    with pytest.raises(RuntimeError, match="minimum site distance"):
        _evaluate(
            h2_point_response_case,
            [[0.0, 0.0, 0.70]],
            [0.0],
            minimum_site_distance_bohr=0.2,
        )
    allowed = _evaluate(h2_point_response_case, [[0.0, 0.0, 0.70]], [0.0])
    assert np.all(np.isfinite(np.asarray(allowed["responses"][0])))


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
    h2_point_response_case, points, frequencies, options, message
):
    with pytest.raises(RuntimeError, match=message):
        _evaluate(h2_point_response_case, points, frequencies, **options)


def test_point_count_and_memory_resource_gates_are_up_front(h2_point_response_case):
    with pytest.raises(RuntimeError, match="500"):
        _evaluate(h2_point_response_case, [[0.0, 0.0, 3.0 + i] for i in range(501)], [0.0])

    estimate = psi4.core._atomic_polarizability_estimate_point_response
    plan = estimate(2, 7, 5, 2, 500, True, 1 << 30)
    assert plan["max_point_count"] == 500
    assert plan["ao_matrix_bytes"] == 7 * 7 * 8
    assert plan["transition_potential_bytes"] == 10 * 500 * 8
    assert plan["output_bytes"] == 2 * 500 * 500 * 8
    assert plan["dense_solve_peak_bytes"] > plan["transition_potential_bytes"]
    assert plan["estimated_bytes"] >= plan["output_bytes"]
    with pytest.raises(RuntimeError, match="reserved memory"):
        estimate(2, 7, 5, 2, 500, True, plan["estimated_bytes"] * 2 - 2)


@pytest.mark.parametrize(
    "h1,h2,message",
    [
        (np.eye(2), np.eye(2), "dimension"),
        ([[np.inf]], [[1.0]], "finite"),
        ([[1.0, 1.0e-4], [0.0, 1.0]], np.eye(2), "symmetric"),
    ],
)
def test_malformed_response_operator_is_rejected(h2_point_response_case, h1, h2, message):
    context, _, _, _ = h2_point_response_case
    with pytest.raises(RuntimeError, match=message):
        psi4.core._atomic_polarizability_test_evaluate_point_response(
            context,
            [[0.0, 0.0, 2.0]],
            [0.0],
            psi4.core.Matrix.from_array(np.asarray(h1, dtype=float)),
            psi4.core.Matrix.from_array(np.asarray(h2, dtype=float)),
            {},
        )


def test_source_uses_only_native_order_zero_point_potential_without_generator_conflation():
    source = (
        Path(__file__).resolve().parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    start = source.index("PointResponseData evaluate_point_response")
    end = source.index("Matrix lw_graph_operator", start)
    evaluator = source[start:end]
    assert evaluator.count("ao_multipole_potential(0,") == 1
    assert "ao_multipoles" not in evaluator
    assert "compute_isa" not in evaluator.lower()
    assert "camcasp" not in evaluator.lower()
    assert "generate" not in evaluator.lower()
