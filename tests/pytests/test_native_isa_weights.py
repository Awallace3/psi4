import math

import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


def _synthetic(sites, points, terms, **options):
    return psi4.core._atomic_polarizability_test_isa(
        sites,
        points,
        [1.0] * len(points),
        [1] * len(sites),
        terms,
        options,
    )


def _rows(result):
    nsite = result["site_count"]
    values = result["weights"]
    return [values[i : i + nsite] for i in range(0, len(values), nsite)]


def test_one_atom_is_exact_unity_and_conserves_population():
    points = [[-1.2, 0.1, 0.3], [0.0, 0.0, 0.0], [1.7, -0.2, 0.4]]
    result = _synthetic([[0.0, 0.0, 0.0]], points, [[0.0, 0.0, 0.0, 2.0, 0.7]])
    assert _rows(result) == [[1.0], [1.0], [1.0]]
    assert result["diagnostics"]["atomic_populations"] == pytest.approx(
        [result["diagnostics"]["electron_count"]]
    )
    assert result["diagnostics"]["iterations"] >= 1
    assert result["diagnostics"]["converged"] is True


def test_identical_two_center_density_has_inversion_symmetry_and_partition_unity():
    sites = [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]
    points = [[-1.4, 0.2, 0.0], [1.4, -0.2, 0.0], [0.0, 0.0, 0.0]]
    terms = [[-0.8, 0.0, 0.0, 1.0, 0.9], [0.8, 0.0, 0.0, 1.0, 0.9]]
    rows = _rows(_synthetic(sites, points, terms))
    assert rows[0][0] == pytest.approx(rows[1][1], abs=2.0e-9)
    assert rows[0][1] == pytest.approx(rows[1][0], abs=2.0e-9)
    assert rows[2] == pytest.approx([0.5, 0.5], abs=2.0e-9)
    for row in rows:
        assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in row)
        assert math.fsum(row) == pytest.approx(1.0, abs=1.0e-13)


def test_site_order_only_permutes_weight_columns():
    sites = [[-0.7, 0.0, 0.0], [0.9, 0.1, 0.0]]
    points = [[-1.0, 0.3, 0.0], [0.2, -0.1, 0.4], [1.4, 0.0, -0.2]]
    terms = [[-0.7, 0.0, 0.0, 1.4, 1.1], [0.9, 0.1, 0.0, 0.8, 0.6]]
    forward = _rows(_synthetic(sites, points, terms))
    reverse = _rows(_synthetic(list(reversed(sites)), points, terms))
    for first, second in zip(forward, reverse):
        assert first == pytest.approx(list(reversed(second)), abs=2.0e-9)


def test_density_scaling_preserves_probabilities_and_scales_populations():
    sites = [[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]
    points = [[-1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.0, 0.0]]
    terms = [[-0.6, 0.0, 0.0, 1.0, 0.8], [0.6, 0.0, 0.0, 0.7, 1.2]]
    base = _synthetic(sites, points, terms)
    scaled_terms = [term[:3] + [3.5 * term[3], term[4]] for term in terms]
    scaled = _synthetic(sites, points, scaled_terms)
    assert scaled["weights"] == pytest.approx(base["weights"], abs=2.0e-9)
    assert scaled["diagnostics"]["atomic_populations"] == pytest.approx(
        [3.5 * value for value in base["diagnostics"]["atomic_populations"]], rel=2.0e-8
    )


def test_log_pchip_is_shape_preserving_and_exponential_tail_is_continuous():
    nodes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    logs = [0.0, -0.2, -0.8, -1.4, -2.0, -3.0]
    query = [0.25, 0.75, 1.25, 1.5, 2.5, 6.0]
    result = psi4.core._atomic_polarizability_test_isa_profile(
        nodes, logs, query, 1.5, 0.75
    )
    for value, left, right in zip(result["log_values"][:3], logs, logs[1:]):
        assert min(left, right) <= value <= max(left, right)
    assert result["tail_alpha"] > 0.0
    assert result["join_log_left"] == pytest.approx(result["join_log_right"], abs=1.0e-13)
    assert result["log_values"][-1] < result["log_values"][-2]


def test_tail_fit_is_stable_under_radial_refinement():
    sites = [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]
    points = [[-1.4, 0.2, 0.0], [-0.7, 0.1, 0.0], [0.0, 0.2, 0.0],
              [0.7, -0.1, 0.0], [1.4, -0.2, 0.0]]
    terms = [[-0.8, 0.0, 0.0, 1.0, 0.9], [0.8, 0.0, 0.0, 0.7, 1.2]]
    options = {"angular_polar_points": 18, "angular_azimuthal_points": 24}
    grids = [
        _synthetic(sites, points, terms, radial_points=radial_points, **options)
        for radial_points in (120, 160)
    ]
    tail_alphas = [result["diagnostics"]["tail_alphas"] for result in grids]
    assert tail_alphas[1] == pytest.approx(tail_alphas[0], abs=2.0e-3)
    populations = [result["diagnostics"]["atomic_populations"] for result in grids]
    assert populations[1] == pytest.approx(populations[0], abs=1.0e-4)


def test_solver_fails_closed_for_nonconvergence_and_nonfinite_density():
    sites = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    points = [[0.0, 0.0, 0.0]]
    terms = [[-0.5, 0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 0.0, 1.0, 1.0]]
    with pytest.raises(RuntimeError, match=r"did not converge"):
        _synthetic(sites, points, terms, max_iterations=1, convergence=1.0e-16)
    bad = [[0.0, 0.0, 0.0, float("nan"), 1.0]]
    with pytest.raises(RuntimeError, match=r"finite|density"):
        _synthetic([[0.0, 0.0, 0.0]], points, bad)


@pytest.fixture(scope="module")
def frozen_h2o_context():
    psi4.core.be_quiet()
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "pk",
            "reference": "rhf",
            "dft_spherical_points": 50,
            "dft_radial_points": 12,
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
    homo = max(precursor.epsilon_a_subset("SO", "OCC").to_array().ravel())
    shift = cation_wfn.energy() - precursor.energy() + homo
    psi4.set_options({"reference": "rhf", "dft_grac_shift": shift})
    _, grac = psi4.energy("pbe0", molecule=neutral, return_wfn=True)
    psi4.set_options({"dft_grac_shift": 0.0})
    return psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation_wfn
    )


@pytest.mark.scf
def test_real_frozen_grac_h2o_conserves_population_and_refines(frozen_h2o_context):
    coarse = psi4.core._atomic_polarizability_compute_isa_weights(
        frozen_h2o_context,
        {"radial_points": 18, "angular_polar_points": 6, "angular_azimuthal_points": 8},
    )
    medium = psi4.core._atomic_polarizability_compute_isa_weights(
        frozen_h2o_context,
        {"radial_points": 24, "angular_polar_points": 8, "angular_azimuthal_points": 12},
    )
    fine = psi4.core._atomic_polarizability_compute_isa_weights(
        frozen_h2o_context,
        {"radial_points": 36, "angular_polar_points": 12, "angular_azimuthal_points": 16},
    )
    for result in (coarse, medium, fine):
        rows = _rows(result)
        assert all(math.fsum(row) == pytest.approx(1.0, abs=1.0e-13) for row in rows)
        diagnostics = result["diagnostics"]
        assert math.fsum(diagnostics["atomic_populations"]) == pytest.approx(
            diagnostics["electron_count"], abs=2.0e-10
        )
        assert diagnostics["converged"] is True
        assert diagnostics["max_overlap_residual"] <= 1.0e-9
    # These deliberately small, non-nested product grids qualify invariants and
    # fixed-point convergence, not monotonic quadrature convergence.
    profiles = [result["diagnostics"]["grid_profile"] for result in (coarse, medium, fine)]
    assert profiles[0]["shell_point_count"] < profiles[1]["shell_point_count"] < profiles[2]["shell_point_count"]


@pytest.mark.scf
def test_real_frozen_grac_h2o_is_deterministic(frozen_h2o_context):
    options = {"radial_points": 36, "angular_polar_points": 12, "angular_azimuthal_points": 16}
    first = psi4.core._atomic_polarizability_compute_isa_weights(frozen_h2o_context, options)
    second = psi4.core._atomic_polarizability_compute_isa_weights(frozen_h2o_context, options)
    assert second["weights"] == first["weights"]
    assert second["diagnostics"]["atomic_populations"] == first["diagnostics"]["atomic_populations"]
    assert second["diagnostics"]["log_profiles"] == first["diagnostics"]["log_profiles"]
