import math
from pathlib import Path
import re

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


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
    context = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation_wfn
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
    assert diagnostics["max_overlap_residual"] <= 1.0e-9
    radial_nodes = diagnostics["radial_nodes"]
    assert len(radial_nodes) == 3
    assert all(len(nodes) == len(profile) for nodes, profile in zip(radial_nodes, diagnostics["log_profiles"]))
    assert radial_nodes[0] != radial_nodes[1]


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
    assert result["algorithm"] == "DIRECT_JK_NONSYMMETRIC"
    assert 1 <= result["batch_size"] <= len(transitions)
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
def test_restricted_c1_uses_exact_direct_jk_when_scf_type_is_df(frozen_h2o_context):
    context, _, _ = frozen_h2o_context
    psi4.set_options({"scf_type": "df"})
    try:
        result = _restricted_c1_primitives(context)
    finally:
        psi4.set_options({"scf_type": "pk"})
    assert result["algorithm"] == "DIRECT_JK_NONSYMMETRIC"


def test_restricted_c1_production_source_forbids_in_core_eri_routes():
    source = (
        Path(__file__).resolve().parents[2]
        / "psi4/src/psi4/libmints/atomic_polarizability.cc"
    ).read_text()
    constructor = source.split("construct_restricted_c1_primitives_impl", 1)[1].split(
        "RestrictedC1Primitives construct_restricted_c1_primitives(", 1
    )[0]
    assert "mo_eri" not in constructor
    assert "ao_eri" not in constructor


def test_restricted_c1_aug_cc_pvtz_scaling_estimator_avoids_ao_nbf4():
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
    assert diagnostics["algorithm"] == "DIRECT_JK_NONSYMMETRIC"
    assert diagnostics["nbf"] == nbf
    assert diagnostics["nov"] == nocc * nvir
    assert 1 <= diagnostics["batch_size"] <= 32
    assert diagnostics["estimated_bytes"] < 8 * nbf**4


def test_restricted_c1_estimator_fails_closed_for_memory_and_overflow():
    estimator = psi4.core._atomic_polarizability_estimate_restricted_c1_jk
    with pytest.raises(RuntimeError, match=r"exceeds configured memory"):
        estimator(100, 5, 95, 1024)
    with pytest.raises(RuntimeError, match=r"overflow|dimension exceeds"):
        estimator(2**63, 2**63, 2, 2**64 - 1)
