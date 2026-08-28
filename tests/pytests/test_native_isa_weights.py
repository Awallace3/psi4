import math
from pathlib import Path
import re

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


_BASIS_H2O_CONTEXT = None
_BASIS_H2O_AUXILIARY = None


def _synthetic(sites, points, terms, *, weights=None, atomic_numbers=None, **options):
    return psi4.core._atomic_polarizability_test_isa(
        sites,
        points,
        [1.0] * len(points) if weights is None else weights,
        [1] * len(sites) if atomic_numbers is None else atomic_numbers,
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


def test_exact_two_gaussian_fixed_point_matches_independent_probability_oracle():
    sites = [[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]
    points = [[-1.1, 0.2, 0.3], [0.0, 0.0, 0.0], [1.3, -0.1, 0.2]]
    terms = [[-0.75, 0.0, 0.0, 1.3, 0.8], [0.75, 0.0, 0.0, 0.7, 1.2]]
    result = psi4.core._atomic_polarizability_test_isa_gaussian_fixed_point(
        sites, points, terms, 48, 12, 16
    )
    expected = []
    for point in points:
        values = []
        for cx, cy, cz, coefficient, exponent in terms:
            radius2 = sum((value - center) ** 2 for value, center in zip(point, (cx, cy, cz)))
            values.append(coefficient * math.exp(-exponent * radius2))
        expected.extend(value / math.fsum(values) for value in values)
    assert result["weights"] == pytest.approx(expected, abs=2.0e-14)
    assert result["max_profile_relative_error"] < 2.0e-13


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


def test_log_pchip_is_shape_preserving_and_exponential_tail_has_independent_charge_oracle():
    nodes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    logs = [0.0, -0.2, -0.8, -1.4, -2.0, -3.0]
    query = [0.25, 0.75, 1.25, 1.5, 2.5, 6.0]
    known_alpha = 1.7
    join = 1.5
    value_at_join = math.exp(logs[3])
    tail_charge = 4.0 * math.pi * value_at_join * (
        join**2 / known_alpha + 2.0 * join / known_alpha**2 + 2.0 / known_alpha**3
    )
    result = psi4.core._atomic_polarizability_test_isa_profile(
        nodes, logs, query, join, tail_charge
    )
    for value, left, right in zip(result["log_values"][:3], logs, logs[1:]):
        assert min(left, right) <= value <= max(left, right)
    assert result["tail_alpha"] == pytest.approx(known_alpha, rel=2.0e-12)
    assert result["join_log_left"] == pytest.approx(result["join_log_right"], abs=1.0e-13)
    assert result["tail_charge"] == pytest.approx(tail_charge, rel=2.0e-12)
    for radius, actual in zip(query[4:], result["log_values"][4:]):
        expected = result["tail_log_amplitude"] - known_alpha * radius
        assert actual == pytest.approx(expected, abs=2.0e-12)


def test_overlap_uses_dedicated_inner_rule_plus_analytic_tail_and_refines():
    join = 1.5
    first_alpha, second_alpha = 0.7, 1.4
    nodes = [0.0, 0.3, 0.7, 1.1, join, 2.0]
    first_logs = [-first_alpha * radius for radius in nodes]
    second_logs = [-second_alpha * radius for radius in nodes]
    exact_overlap = (2.0 * math.sqrt(first_alpha * second_alpha) /
                     (first_alpha + second_alpha)) ** 3
    errors = []
    for integration_points in (4, 8, 16):
        result = psi4.core._atomic_polarizability_test_isa_overlap(
            nodes, first_logs, first_alpha, 0.0,
            nodes, second_logs, second_alpha, 0.0,
            join, integration_points,
        )
        errors.append(abs((1.0 - result["overlap_residual"]) - exact_overlap))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2.0e-13


def test_overlap_first_activation_handles_old_raw_and_new_fitted_tail():
    join = 1.5
    nodes = [0.0, 0.3, 0.7, 1.1, join, 2.0, 3.0]
    logs = [-radius for radius in nodes]
    result = psi4.core._atomic_polarizability_test_isa_overlap(
        nodes, logs, 0.0, 0.0,
        nodes, logs, 1.0, 0.0,
        join, 32,
    )
    assert result["overlap_residual"] < 2.0e-10


def test_far_field_distinct_analytic_tails_do_not_collapse_to_equal_floor():
    probabilities = psi4.core._atomic_polarizability_test_isa_tail_probabilities(
        [0.0, 0.0], [1.0, 1.002], [1000.0, 1000.0]
    )
    expected_first = 1.0 / (1.0 + math.exp(-2.0))
    assert probabilities == pytest.approx([expected_first, 1.0 - expected_first], abs=2.0e-14)
    assert probabilities != pytest.approx([0.5, 0.5], abs=1.0e-3)


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


def test_tail_fit_failure_reuses_complete_previous_profile_or_fails_closed():
    sites = [[-0.6, 0.0, 0.0], [0.7, 0.1, 0.0]]
    points = [[-0.9, 0.2, 0.0], [0.1, -0.3, 0.2], [1.1, 0.0, -0.1]]
    terms = [[-0.6, 0.0, 0.0, 1.4, 0.7], [0.7, 0.1, 0.0, 0.5, 1.6]]
    common = {
        "radial_points": 32,
        "angular_polar_points": 8,
        "angular_azimuthal_points": 12,
        "tail_activation_iteration": 1,
    }
    with pytest.raises(RuntimeError, match=r"no valid exponential tail"):
        _synthetic(sites, points, terms, inject_tail_fit_failure_iteration=1, **common)
    retained = _synthetic(
        sites, points, terms,
        inject_tail_fit_failure_iteration=2,
        **common,
    )
    assert retained["diagnostics"]["tail_fit_failures"] == len(sites)
    assert retained["diagnostics"]["tail_failure_reused_profiles"] == len(sites)
    assert retained["diagnostics"]["iterations"] > 2
    assert retained["diagnostics"]["converged"] is True
    assert all(math.isfinite(value) for profile in retained["diagnostics"]["log_profiles"] for value in profile)
    with pytest.raises(RuntimeError, match=r"did not converge"):
        _synthetic(
            sites, points, terms,
            inject_tail_fit_failure_iteration=2,
            max_iterations=2,
            **common,
        )


def test_solver_fails_closed_for_nonconvergence_nonfinite_density_and_negative_weights():
    sites = [[-0.6, 0.0, 0.0], [0.7, 0.1, 0.0]]
    points = [[-0.9, 0.2, 0.0], [0.1, -0.3, 0.2], [1.1, 0.0, -0.1]]
    terms = [[-0.6, 0.0, 0.0, 1.4, 0.7], [0.7, 0.1, 0.0, 0.5, 1.6]]
    with pytest.raises(RuntimeError, match=r"did not converge") as failure:
        _synthetic(sites, points, terms, max_iterations=1, convergence=1.0e-16)
    residual = re.search(r"max overlap residual = ([0-9.eE+-]+)", str(failure.value))
    assert residual is not None
    assert float(residual.group(1)) > 1.0e-16

    guard_points = [[0.0, 0.0, 0.0]]
    bad = [[0.0, 0.0, 0.0, float("nan"), 1.0]]
    with pytest.raises(RuntimeError, match=r"finite|density"):
        _synthetic([[0.0, 0.0, 0.0]], guard_points, bad)
    with pytest.raises(RuntimeError, match=r"integration weights.*nonnegative"):
        _synthetic(
            [[0.0, 0.0, 0.0]], guard_points,
            [[0.0, 0.0, 0.0, 1.0, 1.0]], weights=[-1.0]
        )


def test_basis_space_options_fail_closed_without_a_native_auxiliary_basis():
    real = _synthetic(
        [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0, 1.0]], basis_eigenvalue_cutoff=1.0
    )
    assert real["diagnostics"]["method"] == "real-space"
    with pytest.raises(RuntimeError, match=r"requires a sealed native auxiliary basis"):
        _synthetic(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0, 1.0]], method="basis-space-a"
        )
    with pytest.raises(RuntimeError, match=r"method must be"):
        _synthetic(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0, 1.0]], method="unknown"
        )
    with pytest.raises(RuntimeError, match=r"values are invalid"):
        _synthetic(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0, 1.0]], method="basis-space-a",
            basis_eigenvalue_cutoff=1.0
        )


@pytest.fixture(scope="module")
def frozen_h2o_context():
    global _BASIS_H2O_CONTEXT, _BASIS_H2O_AUXILIARY
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
    psi4.set_options({"reference": "uhf", "dft_density_tolerance": 1.0e-12,
                      "dft_grac_shift": 0.0})
    _, cation_wfn = psi4.energy("pbe0", molecule=cation, return_wfn=True)
    homo = max(precursor.epsilon_a_subset("SO", "OCC").to_array().ravel())
    shift = cation_wfn.energy() - precursor.energy() + homo
    psi4.set_options({"reference": "rhf", "dft_density_tolerance": 1.0e-12,
                      "dft_grac_shift": shift})
    _, grac = psi4.energy("pbe0", molecule=neutral, return_wfn=True)
    psi4.set_options({"dft_grac_shift": 0.0})
    context = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation_wfn
    )
    auxiliary_key = "DF_BASIS_ATOMIC_POLARIZABILITY"
    _BASIS_H2O_AUXILIARY = psi4.core.BasisSet.build(
        grac.molecule(), auxiliary_key, "cc-pvdz-ri", puream=1, quiet=True
    )
    grac.set_basisset(auxiliary_key, _BASIS_H2O_AUXILIARY)
    _BASIS_H2O_CONTEXT = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation_wfn, auxiliary_key
    )
    overlap = np.asarray(psi4.core.MintsHelper(grac.basisset()).ao_overlap())
    frozen_density = np.asarray(grac.Da()) + np.asarray(grac.Db())
    formal_from_density = float(np.einsum("ij,ji->", frozen_density, overlap))
    return context, formal_from_density, grac


@pytest.mark.scf
def test_real_frozen_grac_h2o_coarse_grid_invariants_and_formal_count(frozen_h2o_context):
    context, formal_from_density, _ = frozen_h2o_context
    options = {"radial_points": 36, "angular_polar_points": 12, "angular_azimuthal_points": 16}
    result = psi4.core._atomic_polarizability_compute_isa_weights(context, options)
    rows = _rows(result)
    assert all(math.fsum(row) == pytest.approx(1.0, abs=1.0e-13) for row in rows)
    diagnostics = result["diagnostics"]
    assert formal_from_density == pytest.approx(10.0, abs=2.0e-10)
    assert diagnostics["formal_electron_count"] == pytest.approx(formal_from_density, abs=2.0e-10)
    assert math.fsum(diagnostics["atomic_populations"]) == pytest.approx(
        diagnostics["electron_count"], abs=2.0e-10
    )
    # The deliberately coarse sealed response grid is used for invariants only;
    # its nonzero quadrature error is not an ISA shell-grid qualification.
    coarse_grid_error = abs(diagnostics["electron_count"] - formal_from_density)
    assert coarse_grid_error == pytest.approx(diagnostics["electron_count_absolute_error"], abs=2.0e-12)
    assert 1.0e-12 < coarse_grid_error < 0.1
    assert diagnostics["converged"] is True
    assert diagnostics["method"] == "real-space"
    assert diagnostics["density_source"] == "frozen AO density"
    assert diagnostics["auxiliary_function_count"] == 0
    assert diagnostics["max_overlap_residual"] <= 1.0e-9
    radial_nodes = diagnostics["radial_nodes"]
    assert len(radial_nodes) == 3
    assert all(len(nodes) == len(profile) for nodes, profile in zip(radial_nodes, diagnostics["log_profiles"]))
    assert radial_nodes[0] != radial_nodes[1]


@pytest.mark.scf
def test_basis_space_a_uses_a_sealed_spherical_auxiliary_basis(frozen_h2o_context):
    _, formal_from_density, _ = frozen_h2o_context
    assert _BASIS_H2O_AUXILIARY.has_puream() is True
    options = {
        "method": "basis-space-a",
        "radial_points": 36,
        "angular_polar_points": 12,
        "angular_azimuthal_points": 16,
        "convergence": 1.0e-8,
    }
    result = psi4.core._atomic_polarizability_compute_isa_weights(
        _BASIS_H2O_CONTEXT, options
    )
    rows = _rows(result)
    assert all(math.fsum(row) == pytest.approx(1.0, abs=1.0e-13) for row in rows)
    diagnostics = result["diagnostics"]
    assert diagnostics["method"] == "basis-space-a"
    assert diagnostics["density_source"] == "frozen AO density"
    assert diagnostics["auxiliary_function_count"] == _BASIS_H2O_AUXILIARY.nbf()
    expected_per_site = [
        sum(_BASIS_H2O_AUXILIARY.function_to_center(function) == site
            for function in range(_BASIS_H2O_AUXILIARY.nbf()))
        for site in range(3)
    ]
    assert diagnostics["auxiliary_functions_per_site"] == expected_per_site
    assert all(rank > 0 for rank in diagnostics["retained_basis_ranks"])
    assert math.isfinite(diagnostics["max_basis_condition_number"])
    assert diagnostics["nonpositive_shape_repairs"] == 0
    assert math.fsum(diagnostics["atomic_populations"]) == pytest.approx(
        diagnostics["electron_count"], abs=2.0e-10
    )
    assert diagnostics["formal_electron_count"] == pytest.approx(
        formal_from_density, abs=2.0e-10
    )
    assert diagnostics["atomic_populations"] == pytest.approx(
        [8.500677596728877, 0.7630705014779949, 0.7630705014779815], abs=5.0e-8
    )
    real = psi4.core._atomic_polarizability_compute_isa_weights(
        frozen_h2o_context[0], {key: value for key, value in options.items() if key != "method"}
    )
    assert max(abs(first - second) for first, second in zip(result["weights"], real["weights"])) > 0.05


@pytest.mark.scf
def test_driver_attaches_the_selected_spherical_isa_auxiliary_basis(frozen_h2o_context):
    from psi4.driver.procrouting.atomic_polarizability import (
        AUXILIARY_PARTITION_BASIS_KEY,
        _attach_partition_auxiliary_basis,
    )

    _, _, grac = frozen_h2o_context
    psi4.set_options({
        "atomic_polarizability_partition": "ISA",
        "atomic_polarizability_isa_method": "BASIS_SPACE_A",
        "atomic_polarizability_isa_aux_basis": "cc-pvdz-ri",
    })
    try:
        _attach_partition_auxiliary_basis(grac)
        attached = grac.get_basisset(AUXILIARY_PARTITION_BASIS_KEY)
        assert attached.name().upper() == "CC-PVDZ-RI"
        assert attached.has_puream() is True
    finally:
        psi4.set_options({"atomic_polarizability_isa_method": "REAL_SPACE"})


@pytest.mark.scf
def test_basis_space_a_requires_a_sealed_auxiliary_basis(frozen_h2o_context):
    context, _, _ = frozen_h2o_context
    with pytest.raises(RuntimeError, match=r"no sealed auxiliary basis"):
        psi4.core._atomic_polarizability_compute_isa_weights(
            context, {"method": "basis-space-a", "radial_points": 12,
                      "angular_polar_points": 6, "angular_azimuthal_points": 8}
        )


@pytest.mark.scf
def test_real_frozen_grac_h2o_digest_is_complete_deterministic_and_option_sensitive(frozen_h2o_context):
    context, _, _ = frozen_h2o_context
    options = {"radial_points": 30, "angular_polar_points": 10, "angular_azimuthal_points": 12}
    first = psi4.core._atomic_polarizability_compute_isa_weights(context, options)
    second = psi4.core._atomic_polarizability_compute_isa_weights(context, options)
    changed = psi4.core._atomic_polarizability_compute_isa_weights(
        context, dict(options, convergence=5.0e-9)
    )
    assert second["weights"] == first["weights"]
    assert second["diagnostics"]["context_digest"] == first["diagnostics"]["context_digest"]
    assert changed["diagnostics"]["context_digest"] != first["diagnostics"]["context_digest"]


@pytest.mark.scf
def test_native_provider_wires_reviewed_physical_chain_frequency_major(frozen_h2o_context):
    context, _, grac = frozen_h2o_context
    options = {"radial_points": 30, "angular_polar_points": 10, "angular_azimuthal_points": 12}
    isa = psi4.core._atomic_polarizability_compute_isa_weights(context, options)
    provider = psi4.core._atomic_polarizability_make_native_response_provider(context, options)
    frequencies = [0.0, 0.3]
    responses = provider.compute(frequencies, [0.0, 1.0])

    assert len(responses) == 2
    assert all(np.asarray(item["positions"]).shape == (3, 3) for item in responses)
    assert all(len(item["blocks"]) == 9 for item in responses)
    assert all(item["chf_exchange_coefficient"] == 0.25 for item in responses)
    assert all(item["alda_kernel_coefficient"] == 0.75 for item in responses)
    assert all(item["restricted_factor"] == 4.0 for item in responses)

    # Independent wiring oracle: invoke each already-reviewed underscored C1,
    # C2b, C3a, Hessian, dense-solve, and C3b seam explicitly.
    c1 = psi4.core._atomic_polarizability_test_restricted_c1_primitives(context, {})
    alda = psi4.core._atomic_polarizability_test_restricted_alda_kernel(context, False)
    assert c1["transitions"] == alda["transitions"]
    assembled = psi4.core._atomic_polarizability_assemble_restricted_hessian(
        c1["orbital_gaps"], c1["coulomb"], c1["exchange_direct"],
        c1["exchange_transpose"], alda["full_alda"], 0.25, 0.75
    )
    projection = psi4.core._atomic_polarizability_test_project_transition_multipoles_context(
        context, context, isa["weights"]
    )
    assert projection["transitions"] == c1["transitions"]

    explicit = []
    for omega in frequencies:
        solved = psi4.core._atomic_polarizability_test_solve_and_contract_site_pair_response(
            3, projection["values"], assembled["H1"], assembled["H2"], omega
        )
        assert solved["reciprocal_condition"] >= 1.0e-12
        assert solved["reciprocal_pivot_growth"] >= 1.0e-12
        assert solved["max_forward_error"] <= 1.0e-8
        assert solved["max_backward_error"] <= 1.0e-11
        assert solved["max_scaled_residual"] <= 1.0e-11
        explicit.append(np.asarray(solved["values"]))

    for response, expected in zip(responses, explicit):
        blocks = np.asarray([np.asarray(block) for block in response["blocks"]]).reshape(3, 3, 16, 16)
        actual = blocks.transpose(0, 2, 1, 3).reshape(48, 48)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)
        np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=1.0e-13)
        charge_rows = np.arange(0, 48, 16)
        noncharge = np.setdiff1d(np.arange(48), charge_rows)
        assert np.max(np.abs(actual[np.ix_(charge_rows, noncharge)])) > 1.0e-8

    def translation(position):
        columns = []
        for component in range(16):
            unit = [0.0] * 16
            unit[component] = 1.0
            columns.append(psi4.core._atomic_polarizability_translate_l3(unit, position))
        return np.asarray(columns).T

    positions = np.asarray(responses[0]["positions"])
    translations = [translation(position) for position in positions]
    molecular = []
    for response, expected in zip(responses, explicit):
        blocks = np.asarray([np.asarray(block) for block in response["blocks"]]).reshape(3, 3, 16, 16)
        provider_sum = sum(
            translations[a] @ blocks[a, b] @ translations[b].T
            for a in range(3) for b in range(3)
        )
        expected_blocks = expected.reshape(3, 16, 3, 16).transpose(0, 2, 1, 3)
        explicit_sum = sum(
            translations[a] @ expected_blocks[a, b] @ translations[b].T
            for a in range(3) for b in range(3)
        )
        np.testing.assert_allclose(provider_sum, explicit_sum, rtol=2.0e-12, atol=2.0e-12)
        molecular.append(provider_sum)
    static_dipole = molecular[0][1:4, 1:4]
    assert np.isfinite(static_dipole).all()
    assert np.linalg.eigvalsh(0.5 * (static_dipole + static_dipole.T)).min() > 0.0

    # Independent molecular dipole target from analytic AO dipole integrals,
    # transformed to the same ordered occupied-virtual basis and dense solve.
    ca = np.asarray(grac.Ca())
    ao_dipoles = [np.asarray(value) for value in psi4.core.MintsHelper(grac.basisset()).ao_dipole()]
    direct_projection = np.zeros((16, len(c1["transitions"])))
    for transition, (occupied, virtual) in enumerate(c1["transitions"]):
        cartesian = [ca[:, occupied] @ value @ ca[:, virtual] for value in ao_dipoles]
        direct_projection[1:4, transition] = [cartesian[2], cartesian[0], cartesian[1]]
    direct = psi4.core._atomic_polarizability_test_solve_and_contract_site_pair_response(
        1, psi4.core.Matrix.from_array(direct_projection), assembled["H1"], assembled["H2"], 0.0
    )
    direct_dipole = np.asarray(direct["values"])[1:4, 1:4]
    # The deliberately coarse 12x50 sealed DFT grid limits this quadrature-vs-
    # analytic-integral check; the smallest symmetry component differs by 2.1%.
    np.testing.assert_allclose(static_dipole, direct_dipole, rtol=2.5e-2, atol=2.0e-5)


def _restricted_c1_primitives(context, **test_overrides):
    return psi4.core._atomic_polarizability_test_restricted_c1_primitives(
        context, test_overrides
    )


def _independent_transition_eri_oracle(wfn, orbital_order=None):
    """Tiny-basis oracle independent of the native blocked-JK contractions."""
    coefficients = np.asarray(wfn.Ca())
    energies = np.asarray(wfn.epsilon_a()).ravel()
    occupations = np.asarray(wfn.occupation_a()).ravel()
    if orbital_order is not None:
        coefficients = coefficients[:, orbital_order]
        energies = energies[orbital_order]
        occupations = occupations[orbital_order]
    occupied = [index for index, value in enumerate(occupations) if value == 1.0]
    virtual = [index for index, value in enumerate(occupations) if value == 0.0]
    co = coefficients[:, occupied]
    cv = coefficients[:, virtual]

    # ao_eri is intentionally confined to this seven-function STO-3G oracle.
    # Derive each four-index quantity directly rather than sharing a native MO
    # transform or transition flattening with production.
    ao_eri = np.asarray(psi4.core.MintsHelper(wfn.basisset()).ao_eri())
    iajb = np.einsum("mi,na,lj,sb,mnls->iajb", co, cv, co, cv, ao_eri, optimize=True)
    ijab = np.einsum("mi,nj,la,sb,mnls->ijab", co, co, cv, cv, ao_eri, optimize=True)
    ajbi = np.einsum("ma,nj,lb,si,mnls->ajbi", cv, co, cv, co, ao_eri, optimize=True)
    transitions = [(i, a) for i in occupied for a in virtual]
    gaps = np.array([energies[a] - energies[i] for i, a in transitions])
    return (
        transitions,
        gaps,
        iajb.reshape(len(transitions), len(transitions)),
        ijab.transpose(0, 2, 1, 3).reshape(len(transitions), len(transitions)),
        ajbi.transpose(3, 0, 1, 2).reshape(len(transitions), len(transitions)),
    )


@pytest.mark.scf
def test_restricted_c1_primitives_match_independent_mints_oracle_and_zero_alda_hessian(
    frozen_h2o_context,
):
    context, _, grac = frozen_h2o_context
    result = _restricted_c1_primitives(context)
    transitions, gaps, coulomb, exchange_direct, exchange_transpose = (
        _independent_transition_eri_oracle(grac)
    )

    assert [tuple(pair) for pair in result["transitions"]] == transitions
    assert result["transition_order"] == "(i,a) occupied-major/virtual-minor"
    assert result["algorithm"] == "DIRECT_JK_CANONICAL_NONSYMMETRIC"
    assert result["batch_size"] == 1
    assert result["estimated_bytes"] > 3 * len(transitions) ** 2 * 8
    assert result["orbital_gaps"] == pytest.approx(gaps, abs=2.0e-13)
    for name, expected in (
        ("coulomb", coulomb),
        ("exchange_direct", exchange_direct),
        ("exchange_transpose", exchange_transpose),
    ):
        actual = np.asarray(result[name])
        assert actual.shape == (len(transitions), len(transitions))
        assert np.all(np.isfinite(actual))
        assert actual == pytest.approx(actual.T, abs=2.0e-12)
        assert actual == pytest.approx(expected, abs=2.0e-12)

    # C2a formula with ALDA deliberately zero: this oracle does not call the
    # native assembler and keeps all three ERI primitives visibly distinct.
    diagonal = np.diag(gaps)
    expected_h1 = diagonal + 4.0 * coulomb - 0.25 * (
        exchange_direct + exchange_transpose
    )
    expected_h2 = diagonal - 0.25 * exchange_direct + 0.25 * exchange_transpose
    assert np.asarray(result["H1_zero_alda"]) == pytest.approx(expected_h1, abs=3.0e-12)
    assert np.asarray(result["H2_zero_alda"]) == pytest.approx(expected_h2, abs=3.0e-12)
    assert "alda" not in result


@pytest.mark.scf
def test_restricted_c1_transition_order_tracks_an_orbital_permutation(frozen_h2o_context):
    context, _, grac = frozen_h2o_context
    # Move one virtual ahead of the occupied set and one occupied behind it,
    # while also swapping the two virtual sources. The output order must still
    # be occupied-major/virtual-minor in the permuted context view.
    permutation = [5, 1, 2, 3, 4, 0, 6]
    result = _restricted_c1_primitives(context, orbital_order=permutation)
    expected = _independent_transition_eri_oracle(grac, permutation)
    transitions, gaps, coulomb, exchange_direct, exchange_transpose = expected
    assert [tuple(pair) for pair in result["transitions"]] == transitions
    assert result["orbital_gaps"] == pytest.approx(gaps, abs=2.0e-13)
    assert np.asarray(result["coulomb"]) == pytest.approx(coulomb, abs=2.0e-12)
    assert np.asarray(result["exchange_direct"]) == pytest.approx(exchange_direct, abs=2.0e-12)
    assert np.asarray(result["exchange_transpose"]) == pytest.approx(
        exchange_transpose, abs=2.0e-12
    )


@pytest.mark.scf
@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"occupation_a": [0.5, 1, 1, 1, 1, 0, 0],
          "occupation_b": [0.5, 1, 1, 1, 1, 0, 0]}, r"integer occupations"),
        ({"occupation_a": [1, 1, 1, 1, 1, 0, 0],
          "occupation_b": [1, 1, 1, 1, 0, 0, 0]}, r"closed-shell.*occupations"),
        ({"epsilon_a": [-20, -1, -0.8, -0.6, -0.5, -21, 1],
          "epsilon_b": [-20, -1, -0.8, -0.6, -0.5, -21, 1]}, r"gaps.*positive"),
        ({"beta_orbital_delta": 1.0e-3}, r"Ca and Cb"),
        ({"beta_orbital_delta": float("nan")}, r"coefficients.*finite"),
        ({"epsilon_a": [float("nan"), -1, -0.8, -0.6, -0.5, 0.5, 1],
          "epsilon_b": [float("nan"), -1, -0.8, -0.6, -0.5, 0.5, 1]}, r"energies.*finite"),
        ({"epsilon_a": [-20, -1, -0.8, -0.6, -0.5, 0.5, 1],
          "epsilon_b": [-20.1, -1, -0.8, -0.6, -0.5, 0.5, 1]}, r"energies.*match"),
        ({"occupation_a": [float("nan"), 1, 1, 1, 1, 0, 0],
          "occupation_b": [float("nan"), 1, 1, 1, 1, 0, 0]}, r"integer occupations"),
        ({"occupation_a": [0, 0, 0, 0, 0, 0, 0],
          "occupation_b": [0, 0, 0, 0, 0, 0, 0]}, r"occupied and one virtual"),
        ({"occupation_a": [1, 1, 1, 1, 1, 1, 1],
          "occupation_b": [1, 1, 1, 1, 1, 1, 1]}, r"occupied and one virtual"),
    ],
)
def test_restricted_c1_primitives_fail_closed_for_unsupported_orbital_states(
    frozen_h2o_context, overrides, message
):
    context, _, _ = frozen_h2o_context
    with pytest.raises(RuntimeError, match=message):
        _restricted_c1_primitives(context, **overrides)


@pytest.mark.scf
def test_restricted_c1_override_seam_rejects_malformed_inputs(frozen_h2o_context):
    context, _, _ = frozen_h2o_context
    malformed = (
        ({"unknown": 1}, r"unknown test override"),
        ({"orbital_order": [0, 1]}, r"wrong dimension"),
        ({"orbital_order": [0, 1, 2, 3, 4, 5, 5]}, r"must be a permutation"),
        ({"epsilon_a": [0.0]}, r"wrong dimension"),
    )
    for overrides, message in malformed:
        with pytest.raises((RuntimeError, TypeError), match=message):
            _restricted_c1_primitives(context, **overrides)


@pytest.mark.scf
def test_restricted_c1_canonical_direct_jk_is_option_independent_and_nonmutating(
    frozen_h2o_context,
):
    context, _, grac = frozen_h2o_context
    _, _, coulomb, exchange_direct, exchange_transpose = (
        _independent_transition_eri_oracle(grac)
    )
    option_names = (
        "SCREENING",
        "INTS_TOLERANCE",
        "INCFOCK",
        "INCFOCK_FULL_FOCK_EVERY",
        "SCF_TYPE",
        "DF_INTS_NUM_THREADS",
    )
    before = {name: psi4.core.get_option("SCF", name) for name in option_names}
    old_threads = psi4.get_num_threads()
    psi4.set_options(
        {
            "screening": "density",
            "ints_tolerance": 1.0e-2,
            "incfock": True,
            "incfock_full_fock_every": 1,
            "scf_type": "df",
            "df_ints_num_threads": 3,
        }
    )
    psi4.set_num_threads(2)
    perturbed = {name: psi4.core.get_option("SCF", name) for name in option_names}
    try:
        result = _restricted_c1_primitives(context)
        after = {name: psi4.core.get_option("SCF", name) for name in option_names}
        assert after == perturbed
        assert psi4.get_num_threads() == 2
    finally:
        psi4.set_options({name.lower(): value for name, value in before.items()})
        psi4.set_num_threads(old_threads)

    assert result["algorithm"] == "DIRECT_JK_CANONICAL_NONSYMMETRIC"
    assert result["batch_size"] == 1
    assert result["jk_threads"] == 1
    assert result["integral_engine_thread_count"] == 1
    assert result["screening"] == "NONE"
    assert result["integral_cutoff"] == pytest.approx(1.0e-15)
    assert result["incfock"] is False
    assert np.asarray(result["coulomb"]) == pytest.approx(coulomb, abs=2.0e-12)
    assert np.asarray(result["exchange_direct"]) == pytest.approx(
        exchange_direct, abs=2.0e-12
    )
    assert np.asarray(result["exchange_transpose"]) == pytest.approx(
        exchange_transpose, abs=2.0e-12
    )


def test_direct_jk_standard_integral_backend_selector_preserves_default_behavior():
    selector = psi4.core._direct_jk_uses_brian_backend
    assert selector(False, False) is False
    assert selector(True, False) is True
    assert selector(False, True) is False
    assert selector(True, True) is False


def test_restricted_c1_production_source_forbids_in_core_eri_routes_and_enables_standard_backend():
    source = (
        Path(__file__).resolve().parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    constructor = source.split("construct_restricted_c1_primitives_impl", 1)[1].split(
        "RestrictedC1Primitives construct_restricted_c1_primitives(", 1
    )[0]
    assert "mo_eri" not in constructor
    assert "ao_eri" not in constructor
    assert "jk->set_standard_integral_backend_only(true);" in constructor


def test_restricted_c1_aug_cc_pvtz_estimator_reports_supported_water_envelope():
    molecule = psi4.geometry(
        """
        0 1
        O 0.0 0.0 0.0
        H 0.0 1.4 1.1
        H 0.0 -1.4 1.1
        symmetry c1
        units bohr
        """
    )
    basis = psi4.core.BasisSet.build(molecule, "ORBITAL", "aug-cc-pvtz")
    nbf = basis.nbf()
    nocc = 5
    nvir = nbf - nocc
    diagnostics = psi4.core._atomic_polarizability_estimate_restricted_c1_jk(
        nbf, nocc, nvir, psi4.get_memory()
    )
    assert diagnostics["algorithm"] == "DIRECT_JK_CANONICAL_NONSYMMETRIC"
    assert diagnostics["nbf"] == nbf
    assert diagnostics["nov"] == nocc * nvir
    assert diagnostics["batch_size"] == 1
    assert diagnostics["jk_threads"] == 1
    assert diagnostics["max_supported_nov"] == 512
    assert diagnostics["reserved_memory_bytes"] == psi4.get_memory() // 2
    assert diagnostics["retained_payload_bytes"] == 3 * (nocc * nvir) ** 2 * 8
    assert diagnostics["jk_ao_bytes"] == 3 * nbf**2 * 8
    assert diagnostics["direct_jk_scratch_bytes"] == 10 * nbf**2 * 8
    assert diagnostics["projection_bytes"] > 3 * (nocc * nvir) * 8
    assert diagnostics["estimated_bytes"] == sum(
        diagnostics[name]
        for name in (
            "retained_payload_bytes",
            "metadata_bytes",
            "coefficient_bytes",
            "matrix_overhead_bytes",
            "jk_coefficient_bytes",
            "jk_ao_bytes",
            "direct_jk_scratch_bytes",
            "integral_engine_allowance_bytes",
            "projection_bytes",
        )
    )
    # This is a component-accounting diagnostic, not an observed peak-memory test.
    # DirectJK exposes no supported peak estimator; only retained payload is a hard gate.
    assert diagnostics["memory_semantics"] == "RETAINED_PAYLOAD_HARD_GATE_WORKSPACE_ADVISORY"


def test_restricted_c1_estimator_one_entry_memory_boundary_and_envelope_fail_closed():
    estimator = psi4.core._atomic_polarizability_estimate_restricted_c1_jk
    baseline = estimator(7, 1, 2, 2**20)
    retained = baseline["retained_payload_bytes"]
    passing = estimator(7, 1, 2, 2 * retained)
    assert passing["reserved_memory_bytes"] == retained
    with pytest.raises(RuntimeError, match=r"retained.*reserved memory"):
        estimator(7, 1, 2, 2 * retained - 1)
    with pytest.raises(RuntimeError, match=r"supported transition envelope"):
        estimator(110, 5, 103, 2**40)
    with pytest.raises(RuntimeError, match=r"overflow|dimension exceeds"):
        estimator(2**63, 2**63, 2, 2**64 - 1)


def test_restricted_alda_one_point_permutation_cutoff_and_failure_oracles():
    contract = psi4.core._atomic_polarizability_test_contract_restricted_alda
    cutoff = 1.0e-4
    t = np.array([[2.0, -3.0]])
    actual = np.asarray(contract([0.25], psi4.core.Matrix.from_array(t), [0.7], [5.0], cutoff))
    assert actual == pytest.approx(1.25 * np.outer(t[0], t[0]), abs=1.0e-15)
    assert actual == pytest.approx(actual.T, abs=0.0)

    weights = np.array([0.2, 0.4, 0.1])
    values = np.array([[1.0, 2.0], [-0.5, 0.7], [3.0, -1.0]])
    rho = np.array([0.3, 0.8, 1.2])
    fxc = np.array([2.0, -0.4, 0.9])
    reference = np.asarray(contract(weights, psi4.core.Matrix.from_array(values), rho, fxc, cutoff))
    order = [2, 0, 1]
    permuted = np.asarray(contract(weights[order], psi4.core.Matrix.from_array(values[order]),
                                   rho[order], fxc[order], cutoff))
    assert permuted == pytest.approx(reference, abs=2.0e-15)

    below = np.nextafter(cutoff, -math.inf)
    above = np.nextafter(cutoff, math.inf)
    threshold_rho = [below, cutoff, above, -1.0e-16, cutoff * 2.0]
    threshold_weights = [1.0, 1.0, 1.0, 1.0, 0.0]
    threshold_t = psi4.core.Matrix.from_array(np.ones((5, 1)))
    threshold_fxc = [float("nan"), 5.0, 7.0, float("nan"), float("nan")]
    threshold = np.asarray(contract(threshold_weights, threshold_t, threshold_rho,
                                    threshold_fxc, cutoff))
    assert threshold == pytest.approx(np.array([[12.0]]), abs=0.0)

    with pytest.raises(RuntimeError, match=r"cutoff.*positive"):
        contract([1.0], psi4.core.Matrix.from_array(np.ones((1, 1))), [0.5], [1.0], 0.0)

    for w, v, d, f, message in (
        ([-1.0], [[1.0]], [0.5], [1.0], r"weights.*nonnegative"),
        ([1.0], [[1.0]], [float("nan")], [1.0], r"density.*finite"),
        ([1.0], [[1.0]], [0.5], [float("nan")], r"LibXC kernel.*finite"),
        ([1.0], [[float("inf")]], [0.5], [1.0], r"transition values.*finite"),
    ):
        with pytest.raises(RuntimeError, match=message):
            contract(w, psi4.core.Matrix.from_array(np.asarray(v)), d, f, cutoff)

    validate = psi4.core._atomic_polarizability_test_validate_restricted_alda_grid
    validate(3, 3, [0.1, 0.2, 0.3], [0, 2], [2, 1], [[2, 0], [1]])
    for arguments, message in (
        ((3, 3, [0.1, 0.2, 0.3], [0, 1], [2, 1], [[0], [1]]), r"offset"),
        ((3, 3, [0.1, 0.2, 0.3], [0], [3], [[0, 0]]), r"function map"),
        ((3, 3, [0.1, -0.2, 0.3], [0], [3], [[0]]), r"weights.*nonnegative"),
    ):
        with pytest.raises(RuntimeError, match=message):
            validate(*arguments)


def test_restricted_alda_plan_envelope_work_and_diagnostics_memory_gates():
    estimate = psi4.core._atomic_polarizability_estimate_restricted_alda
    block_points = [256] * 195 + [80]
    block_maps = [100] * len(block_points)
    plan_cutoff = 2.5e-9
    production = estimate(100, 5, 20, block_points, block_maps, 2**30, False, plan_cutoff)
    diagnostics = estimate(100, 5, 20, block_points, block_maps, 2**30, True, plan_cutoff)
    assert production["nov"] == 100
    assert production["point_count"] == 50000
    assert production["density_work_terms"] == 50000 * 100**2
    assert production["mo_transition_work_terms"] == 50000 * (100 * 100 + 100)
    assert production["ao_collocation_work_terms"] == 50000 * 100
    assert production["libxc_work_terms"] == 50000
    assert production["dgemm_work_terms"] == 50000 * 100**2
    assert production["work_terms"] == sum(
        production[name] for name in (
            "density_work_terms", "mo_transition_work_terms",
            "ao_collocation_work_terms", "libxc_work_terms", "dgemm_work_terms"))
    assert production["max_supported_nov"] == 512
    assert production["retained_payload_bytes"] == 100 * 100 * 8
    assert production["diagnostics_payload_bytes"] == 0
    assert diagnostics["diagnostics_payload_bytes"] == 50000 * (100 + 2) * 8
    assert diagnostics["estimated_bytes"] > production["estimated_bytes"]
    assert production["algorithm"] == "SEALED_BLOCK_DGEMM"
    assert production["memory_semantics"] == "CONSERVATIVE_SIMULTANEOUS_LIVE_RESERVATION"
    assert production["conservative_overhead_bytes"] >= 2**20
    assert production["point_scratch_bytes"] == sum(
        production[name] for name in (
            "block_coordinate_weight_bytes", "block_density_kernel_bytes",
            "functional_workspace_bytes"))
    assert production["block_mo_scratch_bytes"] > 0
    assert production["metadata_bytes"] > 0
    assert production["validation_scratch_bytes"] == 100 * np.dtype(np.int32).itemsize
    assert production["density_cutoff"] == pytest.approx(plan_cutoff)
    with pytest.raises(RuntimeError, match=r"supported transition envelope"):
        estimate(100, 5, 103, [100], [10], 2**40, False, plan_cutoff)
    with pytest.raises(RuntimeError, match=r"work bound"):
        estimate(100, 5, 100, [10**9], [10], 2**40, False, plan_cutoff)
    with pytest.raises(RuntimeError, match=r"reserved memory"):
        estimate(100, 5, 20, block_points, block_maps, 1000, False, plan_cutoff)
    with pytest.raises(RuntimeError, match=r"diagnostic.*reserved memory"):
        estimate(100, 5, 20, block_points, block_maps,
                 production["estimated_bytes"] * 2, True, plan_cutoff)
    with pytest.raises(RuntimeError, match=r"overflow|integer limits"):
        estimate(2**63, 2**63, 2, [10], [10], 2**64 - 1, False, plan_cutoff)
    with pytest.raises(RuntimeError, match=r"block metadata"):
        estimate(100, 5, 20, [10, 20], [10], 2**30, False, plan_cutoff)
    with pytest.raises(RuntimeError, match=r"tolerance.*positive"):
        estimate(100, 5, 20, [10], [10], 2**30, False, 0.0)

    validate_work = psi4.core._atomic_polarizability_test_validate_restricted_alda_work_bound
    assert validate_work(production["max_work_terms"]) == production["max_work_terms"]
    with pytest.raises(RuntimeError, match=r"work bound"):
        validate_work(production["max_work_terms"] + 1)

    # Dimension-derived work remains checked near the cap even though seven
    # terms/point cannot represent every integer work count.
    max_points = production["max_work_terms"] // 7
    int_max = 2**31 - 1
    boundary_blocks = [int_max] * (max_points // int_max)
    if max_points % int_max:
        boundary_blocks.append(max_points % int_max)
    boundary_maps = [1] * len(boundary_blocks)
    boundary = estimate(1, 1, 1, boundary_blocks, boundary_maps,
                        2**64 - 1, False, plan_cutoff)
    assert boundary["work_terms"] == 7 * max_points
    over_blocks = list(boundary_blocks)
    over_blocks[-1] += 1
    with pytest.raises(RuntimeError, match=r"work bound"):
        estimate(1, 1, 1, over_blocks, boundary_maps,
                 2**64 - 1, False, plan_cutoff)

    water = psi4.geometry("""
        0 1
        O 0.0 0.0 0.0
        H 0.0 1.4 1.1
        H 0.0 -1.4 1.1
        symmetry c1
        units bohr
    """)
    water_basis = psi4.core.BasisSet.build(water, "ORBITAL", "aug-cc-pvtz")
    canonical_points = [256] * 292 + [248]
    canonical_maps = [water_basis.nbf()] * len(canonical_points)
    canonical = estimate(water_basis.nbf(), 5, water_basis.nbf() - 5,
                         canonical_points, canonical_maps, 2**40, False, plan_cutoff)
    assert canonical["nov"] <= 512
    assert canonical["work_terms"] < canonical["max_work_terms"]


def _independent_vwn_potential(rho, density_cutoff):
    functional = psi4.core.SuperFunctional.blank()
    correlation = psi4.core.LibXCFunctional("XC_LDA_C_VWN", True)
    correlation.set_alpha(1.0)
    functional.add_c_functional(correlation)
    functional.set_density_tolerance(density_cutoff)
    functional.set_max_points(len(rho))
    functional.set_deriv(1)
    functional.allocate()
    values = functional.compute_functional(
        {"RHO_A": psi4.core.Vector.from_array(np.asarray(rho))}, -1, True)
    return np.asarray(values["V_RHO_A"])


def _finite_difference_vwn_fxc(rho, relative_step, density_cutoff):
    rho = np.asarray(rho)
    step = relative_step * rho
    return (_independent_vwn_potential(rho + step, density_cutoff)
            - _independent_vwn_potential(rho - step, density_cutoff)) / (2.0 * step)


def test_restricted_alda_components_match_analytic_x_and_refined_vwn_potential_difference():
    evaluate = psi4.core._atomic_polarizability_test_restricted_alda_fxc
    rho = np.array([0.02, 0.08, 0.3, 1.1, 3.0])
    cutoff = 1.0e-8
    full = evaluate(rho, True, cutoff)
    exchange = evaluate(rho, False, cutoff)
    diagnostics = full["diagnostics"]
    assert diagnostics["exchange_component"] == "XC_LDA_X"
    assert diagnostics["correlation_component"] == "XC_LDA_C_VWN"
    assert diagnostics["exchange_coefficient"] == 1.0
    assert diagnostics["correlation_coefficient"] == 1.0
    assert diagnostics["exchange_libxc_id"] == 1
    assert diagnostics["correlation_libxc_id"] == 7
    assert diagnostics["exchange_libxc_canonical_name"] == "lda_x"
    assert diagnostics["correlation_libxc_canonical_name"] == "lda_c_vwn"
    assert diagnostics["exchange_effective_parameters"] == {}
    assert diagnostics["correlation_effective_parameters"] == {}
    assert diagnostics["derivative_order"] == 2
    assert diagnostics["density_cutoff"] == pytest.approx(cutoff)
    assert "Da+Db" in diagnostics["restricted_normalization"]
    assert "4*b once" in diagnostics["restricted_normalization"]

    expected_x = -(1.0 / 3.0) * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (-2.0 / 3.0)
    assert exchange["fxc"] == pytest.approx(expected_x, rel=3.0e-13, abs=3.0e-13)
    actual_correlation = np.asarray(full["fxc"]) - np.asarray(exchange["fxc"])
    coarse = _finite_difference_vwn_fxc(rho, 2.0e-4, cutoff)
    fine = _finite_difference_vwn_fxc(rho, 1.0e-4, cutoff)
    assert np.max(np.abs(fine - actual_correlation)) < np.max(np.abs(coarse - actual_correlation))
    assert actual_correlation == pytest.approx(fine, rel=2.0e-7, abs=2.0e-9)
    assert exchange["diagnostics"]["correlation_component"] == ""

    below = np.nextafter(cutoff, -math.inf)
    above = np.nextafter(cutoff, math.inf)
    cutoff_values = evaluate([below, cutoff, above, -1.0e-16], True, cutoff)
    assert cutoff_values["fxc"][0] == 0.0
    equality_x = -(1.0 / 3.0) * (3.0 / np.pi) ** (1.0 / 3.0) * cutoff ** (-2.0 / 3.0)
    equality_vwn = _finite_difference_vwn_fxc(np.array([cutoff]), 1.0e-4, cutoff / 10.0)[0]
    assert cutoff_values["fxc"][1] != 0.0
    assert cutoff_values["fxc"][1] == pytest.approx(equality_x + equality_vwn,
                                                       rel=2.0e-7, abs=2.0e-9)
    assert math.isfinite(cutoff_values["fxc"][2])
    assert cutoff_values["fxc"][3] == 0.0
    with pytest.raises(RuntimeError, match=r"tolerance.*positive"):
        evaluate([cutoff], True, 0.0)


def _direct_sto3g_ao_values(basis, coordinates):
    """Independent contracted-Gaussian evaluator for the s/p-only STO-3G fixture."""
    coordinates = np.asarray(coordinates).reshape(-1, 3)
    phi = np.zeros((len(coordinates), basis.nbf()))
    for shell_index in range(basis.nshell()):
        shell = basis.shell(shell_index)
        angular_momentum = shell.am
        assert angular_momentum in (0, 1)
        assert shell.nfunction == (1 if angular_momentum == 0 else 3)
        center = np.array([shell.coord(axis) for axis in range(3)])
        displacement = coordinates - center
        radius_squared = np.einsum("pi,pi->p", displacement, displacement)
        radial = sum(shell.coef(primitive) * np.exp(-shell.exp(primitive) * radius_squared)
                     for primitive in range(shell.nprimitive))
        first = shell.function_index
        if angular_momentum == 0:
            phi[:, first] = radial
        else:
            # Psi4 is configured for Gaussian spherical ordering: p(z),p(x),p(y).
            # Cartesian CCA ordering remains p(x),p(y),p(z).
            components = displacement[:, [2, 0, 1]] if shell.is_pure() else displacement
            phi[:, first:first + 3] = components * radial[:, None]
    return phi


def _independent_real_grid_rho_and_transitions(context, grac):
    points, _, blocks = context.grid_snapshot()
    coordinates = np.asarray(points).reshape(-1, 3)
    phi = _direct_sto3g_ao_values(grac.basisset(), coordinates)
    # The native collocation seam is only a parity target; it is not used below.
    target = psi4.core._atomic_polarizability_test_restricted_alda_ao_collocation_target(context)
    native_phi = np.asarray(target["ao_values"]).reshape(phi.shape)
    selected_points = [0, len(phi) // 3, 2 * len(phi) // 3, len(phi) - 1]
    assert native_phi[selected_points] == pytest.approx(phi[selected_points], abs=3.0e-14)

    density = np.asarray(grac.Da()) + np.asarray(grac.Db())
    coefficients = np.asarray(grac.Ca())
    occupations = np.asarray(grac.occupation_a()).ravel()
    transitions = [(i, a) for i, oi in enumerate(occupations) if oi == 1.0
                   for a, oa in enumerate(occupations) if oa == 0.0]
    rho = np.zeros(len(phi))
    transition_values = np.zeros((len(phi), len(transitions)))
    for offset, count, function_map in blocks:
        p = slice(offset, offset + count)
        fmap = np.asarray(function_map, dtype=int)
        local_phi = phi[p][:, fmap]
        local_density = density[np.ix_(fmap, fmap)]
        rho[p] = np.einsum("pm,mn,pn->p", local_phi, local_density, local_phi,
                           optimize=True)
        orbitals = local_phi @ coefficients[fmap]
        transition_values[p] = np.column_stack(
            [orbitals[:, i] * orbitals[:, a] for i, a in transitions])
    return transitions, rho, transition_values


@pytest.mark.scf
def test_real_restricted_alda_streamed_kernel_independent_collocation_and_c2a_solver(
    frozen_h2o_context,
):
    context, _, grac = frozen_h2o_context
    before = context.state_checksum()
    result = psi4.core._atomic_polarizability_test_restricted_alda_kernel(context)
    assert context.state_checksum() == before
    assert result["densities"] == []
    assert result["fxc"] == []
    assert result["transition_values"] == []
    transitions, rho, transition_values = _independent_real_grid_rho_and_transitions(context, grac)
    assert [tuple(pair) for pair in result["transitions"]] == transitions
    weights = np.asarray(context.grid_snapshot()[1])
    cutoff = result["diagnostics"]["density_cutoff"]
    summary = context.summary()
    assert result["diagnostics"]["density_cutoff_source"] == "FROZEN_FUNCTIONAL_DENSITY_TOLERANCE"
    assert cutoff == summary["functional_density_tolerance"]
    assert cutoff == grac.V_potential().functional().density_tolerance()
    active = rho >= cutoff
    independent_fxc = np.zeros_like(rho)
    active_rho = rho[active]
    expected_x = -(1.0 / 3.0) * (3.0 / np.pi) ** (1.0 / 3.0) * active_rho ** (-2.0 / 3.0)
    independent_fxc[active] = expected_x + _finite_difference_vwn_fxc(
        active_rho, 1.0e-4, cutoff)
    expected = np.einsum("p,pi,p,pj->ij", weights, transition_values, independent_fxc,
                         transition_values, optimize=True)
    kernel = np.asarray(result["full_alda"])
    assert np.all(np.isfinite(kernel))
    assert kernel == pytest.approx(kernel.T, abs=2.0e-13)
    assert kernel == pytest.approx(expected, rel=2.0e-7, abs=2.0e-9)

    retained = psi4.core._atomic_polarizability_test_restricted_alda_kernel(context, True)
    assert len(retained["densities"]) == len(weights)
    assert len(retained["fxc"]) == len(weights)
    assert len(retained["transition_values"]) == len(weights) * len(transitions)
    selected = [0, len(rho) // 3, 2 * len(rho) // 3, len(rho) - 1]
    assert np.asarray(retained["densities"])[selected] == pytest.approx(rho[selected], abs=3.0e-13)
    retained_t = np.asarray(retained["transition_values"]).reshape(transition_values.shape)
    assert retained_t[selected] == pytest.approx(transition_values[selected], abs=3.0e-13)
    assert np.asarray(retained["full_alda"]) == pytest.approx(kernel, abs=2.0e-12)

    c1 = _restricted_c1_primitives(context)
    assembled = psi4.core._atomic_polarizability_assemble_restricted_hessian(
        c1["orbital_gaps"], c1["coulomb"], c1["exchange_direct"],
        c1["exchange_transpose"], result["full_alda"], 0.25, 0.75)
    expected_h1 = (np.diag(c1["orbital_gaps"]) + 4.0 * np.asarray(c1["coulomb"])
                   - 0.25 * (np.asarray(c1["exchange_direct"])
                             + np.asarray(c1["exchange_transpose"]))
                   + 4.0 * 0.75 * kernel)
    assert np.asarray(assembled["H1"]) == pytest.approx(expected_h1, abs=5.0e-12)
    rhs = psi4.core.Matrix.from_array(np.ones((len(transitions), 1)) * 1.0e-3)
    for omega in (0.0, 0.2):
        response = psi4.core._atomic_polarizability_solve_restricted_response(
            assembled["H1"], assembled["H2"], omega, rhs)
        assert np.all(np.isfinite(np.asarray(response["P"])))
        assert np.all(np.isfinite(np.asarray(response["Q"])))


def test_restricted_alda_source_guard_covers_component_factory_and_production():
    root = Path(__file__).resolve().parents[2]
    source = (root / "psi4/src/psi4/libmints/atomic_polarizability.cc").read_text()
    header = (root / "psi4/src/psi4/libmints/atomic_polarizability.h").read_text()
    region = source[source.index("build_restricted_alda_functional"):
                    source.index("std::shared_ptr<FrozenResponseContext> FrozenResponseContext::create")]
    assert "context->functional()" not in region
    assert "V_potential" not in region
    assert "PBE0" not in region.upper()
    assert "Matrix transition_values(npoints" not in region
    assert "C_DGEMM('T', 'N'" in region
    assert "bool retain_test_diagnostics = false" in header
    assert 'constexpr const char* kALDAX = "XC_LDA_X";' in source
    assert 'constexpr const char* kALDAC = "XC_LDA_C_VWN";' in source
    assert '"FROZEN_FUNCTIONAL_DENSITY_TOLERANCE"' in source
    production = source[source.index("RestrictedALDAPrimitive construct_restricted_alda_kernel"):
                        source.index("RestrictedALDACollocationTestResult")]
    assert production.index("preflight_restricted_alda_grid") < production.index("plan_restricted_alda")
    assert production.index("plan_restricted_alda") < production.index("validate_restricted_alda_duplicate_maps")
    assert "seen_generation" not in source
