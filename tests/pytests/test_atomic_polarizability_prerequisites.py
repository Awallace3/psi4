import math

import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints, pytest.mark.scf]


@pytest.fixture(scope="module")
def grac_states():
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
        H 0.0 0.0 0.0
        H 0.0 0.0 0.7
        symmetry c1
        units angstrom
        """
    )
    _, precursor = psi4.energy("pbe0", molecule=neutral, return_wfn=True)

    cation = psi4.geometry(
        """
        1 2
        H 0.0 0.0 0.0
        H 0.0 0.0 0.7
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
    return grac, precursor, cation_wfn, shift


def _context(states):
    grac, precursor, cation, _ = states
    return psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, cation)


def test_response_kernel_is_exact_and_rejects_nextafter_neighbors():
    assert psi4.core._atomic_polarizability_validate_response_kernel(0.25, 0.75) == pytest.approx((0.25, 0.75))
    for chf, alda, message in (
        (math.nextafter(0.25, 0.0), 0.75, "CHF exchange coefficient.*0.25"),
        (math.nextafter(0.25, 1.0), 0.75, "CHF exchange coefficient.*0.25"),
        (0.25, math.nextafter(0.75, 0.0), "ALDA coefficient.*0.75"),
        (0.25, math.nextafter(0.75, 1.0), "ALDA coefficient.*0.75"),
    ):
        with pytest.raises(RuntimeError, match=message):
            psi4.core._atomic_polarizability_validate_response_kernel(chf, alda)


def test_actual_grac_context_is_verified_and_frozen(grac_states):
    grac, precursor, cation, shift = grac_states
    context = _context(grac_states)
    summary = context.summary()

    assert summary["reference"] == "RKS"
    assert summary["functional"] == "PBE0"
    assert summary["needs_grac"] is True
    assert summary["applied_shift"] == pytest.approx(shift, abs=1.0e-12)
    assert summary["derived_shift"] == pytest.approx(shift, abs=1.0e-12)
    assert summary["grac_x_functional"] == "XC_GGA_X_LB"
    assert summary["grac_c_functional"] == "XC_LDA_C_VWN"
    assert isinstance(summary["grac_x_parameters"], dict)
    assert isinstance(summary["grac_c_parameters"], dict)
    assert summary["neutral_precursor_energy"] == pytest.approx(precursor.energy())
    assert summary["cation_energy"] == pytest.approx(cation.energy())
    assert summary["site_count"] == 2
    assert summary["grid_point_count"] > 0
    assert summary["single_thread_immutable"] is True


def test_ordinary_pbe0_rejects_even_when_calculation_metadata_is_available(grac_states):
    _, precursor, cation, _ = grac_states
    with pytest.raises(RuntimeError, match=r"needs_grac|actual GRAC"):
        psi4.core._atomic_polarizability_make_frozen_response_context(precursor, precursor, cation)


def test_wrong_applied_shift_rejects_actual_grac(grac_states):
    grac, precursor, cation, _ = grac_states
    functional = grac.functional()
    old_shift = functional.grac_shift()
    functional.set_lock(False)
    functional.set_grac_shift(old_shift + 1.0e-4)
    functional.set_lock(True)
    try:
        with pytest.raises(RuntimeError, match=r"applied GRAC shift.*IP.*HOMO"):
            psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, cation)
    finally:
        functional.set_lock(False)
        functional.set_grac_shift(old_shift)
        functional.set_lock(True)


@pytest.mark.parametrize("mutation", ["x_identity", "c_identity"])
def test_wrong_grac_component_identity_rejects(grac_states, mutation):
    grac, precursor, cation, _ = grac_states
    token = psi4.core._atomic_polarizability_mutate_grac_component_for_test(grac, mutation)
    try:
        with pytest.raises(RuntimeError, match=r"GRAC.*(functional identity|parameter map)"):
            psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, cation)
    finally:
        psi4.core._atomic_polarizability_restore_grac_component_for_test(grac, mutation, token)


def test_wrong_cation_calculation_rejects(grac_states):
    grac, precursor, _, _ = grac_states
    with pytest.raises(RuntimeError, match=r"cation.*(charge|electron|identity)"):
        psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, precursor)


def test_factory_rejects_an_actual_scf_state_not_marked_converged(grac_states):
    grac, precursor, cation, _ = grac_states
    precursor._set_response_state_converged(False)
    try:
        with pytest.raises(RuntimeError, match=r"neutral precursor.*not converged"):
            psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, cation)
    finally:
        precursor._set_response_state_converged(True)


def test_frozen_context_is_unaffected_by_later_source_orbital_and_density_mutation(grac_states):
    grac, _, _, _ = grac_states
    context = _context(grac_states)
    before = context.state_checksum()
    ca = grac.Ca()
    da = grac.Da()
    old_ca = ca.get(0, 0)
    old_da = da.get(0, 0)
    ca.set(0, 0, old_ca + 0.125)
    da.set(0, 0, old_da + 0.25)
    try:
        assert context.state_checksum() == pytest.approx(before, rel=0.0, abs=0.0)
        provider = psi4.core._atomic_polarizability_make_test_response_provider(context, context)
        assert provider.expected_response_count([0.0, 0.5], [0.0, 1.0]) == 2
    finally:
        ca.set(0, 0, old_ca)
        da.set(0, 0, old_da)


def test_isa_weights_are_structurally_bound_to_one_context(grac_states):
    first = _context(grac_states)
    second = _context(grac_states)
    with pytest.raises(RuntimeError, match=r"ISA weights.*frozen response context"):
        psi4.core._atomic_polarizability_make_test_response_provider(first, second)


@pytest.mark.parametrize(
    "frequencies, weights, message",
    [
        ([], [], "at least one"),
        ([0.0], [], "dimensions"),
        ([math.nan], [0.0], "finite"),
        ([-math.ulp(1.0)], [0.0], "nonnegative"),
        ([math.ulp(1.0)], [0.0], "start.*zero"),
        ([0.0], [math.ulp(1.0)], "static.*weight.*zero"),
        ([0.0, 0.0], [0.0, 1.0], "strictly increasing"),
        ([0.0, -math.ulp(1.0)], [0.0, 1.0], "positive"),
        ([0.0, math.inf], [0.0, 1.0], "finite"),
        ([0.0, 0.5], [0.0, 0.0], "nonzero.*weight.*positive"),
        ([0.0, 0.5], [0.0, -math.ulp(1.0)], "nonzero.*weight.*positive"),
        ([0.0, 0.5], [0.0, math.inf], "finite"),
    ],
)
def test_frequency_grid_rejects_every_invalid_branch(grac_states, frequencies, weights, message):
    context = _context(grac_states)
    provider = psi4.core._atomic_polarizability_make_test_response_provider(context, context)
    with pytest.raises(RuntimeError, match=message):
        provider.expected_response_count(frequencies, weights)


def test_frequency_grid_accepts_exact_boundaries_and_never_fakes_response(grac_states):
    context = _context(grac_states)
    provider = psi4.core._atomic_polarizability_make_test_response_provider(context, context)
    smallest = math.nextafter(0.0, 1.0)
    assert provider.expected_response_count([0.0, smallest], [0.0, smallest]) == 2
    with pytest.raises(RuntimeError, match=r"not implemented.*no response"):
        provider.compute([0.0, smallest], [0.0, smallest])
