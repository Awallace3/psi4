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
    return grac, precursor, cation_wfn, shift


def _context(states):
    grac, precursor, cation, _ = states
    return psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, cation)


def test_hf_response_provenance_lifecycle_is_native_only():
    assert not hasattr(psi4.core, "_HFResponseProvenanceScope")
    for lifecycle_name in (
        "_response_provenance_scope",
        "_set_response_state_converged",
        "_seal_response_provenance",
        "_reset_response_provenance_tracking",
        "_mark_response_compute_failed",
        "_invalidate_response_provenance",
        "_record_response_iteration_state",
        "_capture_response_provenance_if_converged",
    ):
        assert not hasattr(psi4.core.HF, lifecycle_name)


def test_old_scope_success_exit_forge_sequence_is_impossible(grac_states):
    grac, _, _, _ = grac_states
    with pytest.raises(AttributeError):
        scope = grac._response_provenance_scope()
        scope.success()
        scope.__exit__(None, None, None)


def test_genuine_finalized_scfs_are_sealed_by_native_finalize(grac_states):
    grac, precursor, cation, _ = grac_states
    psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, cation)


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
    grac_molecule = grac.molecule()
    cation_molecule = cation.molecule()
    coordinate_pairs = [
        (getattr(grac_molecule, axis)(atom), getattr(cation_molecule, axis)(atom))
        for atom in range(grac_molecule.natom())
        for axis in "xyz"
    ]
    differences = [abs(first - second) for first, second in coordinate_pairs]
    molecular_scale_ulp = max(math.ulp(value) for pair in coordinate_pairs for value in pair)
    assert 0.0 < max(differences) <= molecular_scale_ulp

    # The neutral and cation were parsed independently; their sub-molecular-ULP
    # Bohr-coordinate roundoff must not invalidate a vertical-state context.
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
    assert summary["grac_alpha"] == pytest.approx(0.5)
    assert summary["grac_beta"] == pytest.approx(40.0)
    assert summary["cation_reference"] == "UKS"
    assert summary["basis_detached"] is False
    assert summary["site_count"] == 3
    assert summary["grid_point_count"] > 0
    assert summary["single_thread_no_basis_mutation_contract"] is True


def test_material_cation_geometry_change_rejects(grac_states):
    grac, precursor, _, _ = grac_states
    displaced_cation = psi4.geometry(
        """
        1 2
        O  0.000000  0.000000  0.000000
        H  0.000000  0.757170  0.586260
        H  0.000000 -0.757160  0.586260
        symmetry c1
        units angstrom
        """
    )
    psi4.set_options({"reference": "uhf", "basis": "sto-3g", "dft_grac_shift": 0.0})
    try:
        _, displaced_wfn = psi4.energy("pbe0", molecule=displaced_cation, return_wfn=True)
        with pytest.raises(RuntimeError, match=r"geometry/electron identity"):
            psi4.core._atomic_polarizability_make_frozen_response_context(
                grac, precursor, displaced_wfn
            )
    finally:
        psi4.set_options({"reference": "rhf"})


def test_seal_occurs_only_after_successful_finalize_energy(grac_states, monkeypatch):
    _, precursor, cation, shift = grac_states
    original_finalize = psi4.core.HF.finalize_energy
    observed_unsealed = []

    def observing_finalize(self):
        with pytest.raises(RuntimeError, match=r"provenance seal"):
            psi4.core._atomic_polarizability_make_frozen_response_context(self, precursor, cation)
        observed_unsealed.append(True)
        return original_finalize(self)

    monkeypatch.setattr(psi4.core.HF, "finalize_energy", observing_finalize)
    psi4.set_options({"reference": "rhf", "basis": "sto-3g", "dft_grac_shift": shift})
    try:
        _, finalized = psi4.energy("pbe0", molecule=precursor.molecule(), return_wfn=True)
    finally:
        monkeypatch.setattr(psi4.core.HF, "finalize_energy", original_finalize)
        psi4.set_options({"dft_grac_shift": 0.0})
    assert observed_unsealed
    psi4.core._atomic_polarizability_make_frozen_response_context(finalized, precursor, cation)


def test_unconverged_state_cannot_be_forged_after_failure(grac_states):
    _, precursor, cation, shift = grac_states
    old_e_convergence = psi4.core.get_option("SCF", "E_CONVERGENCE")
    old_d_convergence = psi4.core.get_option("SCF", "D_CONVERGENCE")
    psi4.set_options(
        {
            "reference": "rhf",
            "basis": "sto-3g",
            "dft_grac_shift": shift,
            "maxiter": 1,
            "fail_on_maxiter": False,
        }
    )
    replacement_jk = None
    try:
        _, unconverged = psi4.energy("pbe0", molecule=precursor.molecule(), return_wfn=True)

        # The old reset/fail/record/capture forge surface no longer exists.
        for lifecycle_name in (
            "_reset_response_provenance_tracking",
            "_mark_response_compute_failed",
            "_invalidate_response_provenance",
            "_record_response_iteration_state",
            "_capture_response_provenance_if_converged",
        ):
            assert not hasattr(unconverged, lifecycle_name)

        # Later option loosening, JK replacement, duplicate finalization, and a
        # direct public finalizer cannot manufacture distinct native iterations.
        psi4.set_options({"e_convergence": 1.0, "d_convergence": 1.0})
        replacement_jk = psi4.core.JK.build(unconverged.basisset())
        replacement_jk.initialize()
        unconverged.set_jk(replacement_jk)
        unconverged.finalize()
        with pytest.raises(RuntimeError, match=r"provenance seal"):
            psi4.core._atomic_polarizability_make_frozen_response_context(
                unconverged, precursor, cation
            )
    finally:
        if replacement_jk is not None:
            replacement_jk.finalize()
        psi4.set_options(
            {
                "dft_grac_shift": 0.0,
                "e_convergence": old_e_convergence,
                "d_convergence": old_d_convergence,
                "maxiter": 100,
                "fail_on_maxiter": True,
            }
        )


def test_ordinary_pbe0_rejects_even_when_calculation_metadata_is_available(grac_states):
    _, precursor, cation, _ = grac_states
    with pytest.raises(RuntimeError, match=r"needs_grac|actual GRAC"):
        psi4.core._atomic_polarizability_make_frozen_response_context(precursor, precursor, cation)


def test_post_seal_vbase_reinitialize_cannot_replace_frozen_grid(grac_states):
    grac, _, _, _ = grac_states
    before = _context(grac_states)
    before_summary = before.summary()
    before_grid = before.grid_snapshot()
    potential = grac.V_potential()
    try:
        potential.finalize()
        psi4.set_options({"dft_radial_points": 13})
        potential.initialize()
        after = _context(grac_states)
        assert after.summary() == before_summary
        assert after.grid_snapshot() == before_grid
    finally:
        potential.finalize()
        psi4.set_options({"dft_radial_points": 12})
        potential.initialize()


def test_factory_uses_sealed_shift_and_energies_not_later_mutable_values(grac_states):
    grac, precursor, cation, shift = grac_states
    functional = grac.functional()
    old_shift = functional.grac_shift()
    old_precursor_energy = precursor.energy()
    functional.set_lock(False)
    functional.set_grac_shift(old_shift + 1.0e-4)
    functional.set_lock(True)
    precursor.set_energy(old_precursor_energy + 0.25)
    try:
        summary = psi4.core._atomic_polarizability_make_frozen_response_context(
            grac, precursor, cation
        ).summary()
        assert summary["applied_shift"] == pytest.approx(shift)
        assert summary["neutral_precursor_energy"] == pytest.approx(old_precursor_energy)
    finally:
        functional.set_lock(False)
        functional.set_grac_shift(old_shift)
        functional.set_lock(True)
        precursor.set_energy(old_precursor_energy)


def test_no_production_grac_component_test_mutators_exist():
    assert not hasattr(psi4.core, "_atomic_polarizability_mutate_grac_component_for_test")
    assert not hasattr(psi4.core, "_atomic_polarizability_restore_grac_component_for_test")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dft_grac_alpha": 0.6}, r"GRAC alpha/beta"),
        ({"dft_grac_beta": 39.0}, r"GRAC alpha/beta"),
        ({"dft_grac_x_func": "XC_GGA_X_PBE"}, r"GRAC immutable LibXC identity"),
    ],
)
def test_actual_effective_grac_state_mismatches_reject(grac_states, overrides, message):
    _, precursor, cation, shift = grac_states
    options = {
        "reference": "rhf",
        "basis": "sto-3g",
        "dft_grac_shift": shift,
        "dft_grac_alpha": 0.5,
        "dft_grac_beta": 40.0,
        "dft_grac_x_func": "XC_GGA_X_LB",
    }
    options.update(overrides)
    psi4.set_options(options)
    try:
        _, mismatched = psi4.energy("pbe0", molecule=precursor.molecule(), return_wfn=True)
        with pytest.raises(RuntimeError, match=message):
            psi4.core._atomic_polarizability_make_frozen_response_context(mismatched, precursor, cation)
    finally:
        psi4.set_options(
            {
                "dft_grac_shift": 0.0,
                "dft_grac_alpha": 0.5,
                "dft_grac_beta": 40.0,
                "dft_grac_x_func": "XC_GGA_X_LB",
            }
        )


def test_post_finalize_mutation_cannot_rewrite_sealed_functional_provenance(grac_states, monkeypatch):
    grac, precursor, cation, _ = grac_states
    original_finalize = psi4.core.HF.finalize_energy

    def finalize_with_test_local_tweak(self):
        energy = original_finalize(self)
        # Native finalize has already captured the functional provenance. PBE0
        # stores its combined LibXC XC component in the correlation container.
        self.functional().c_functionals()[0].set_tweak({"_beta": 0.3})
        return energy

    monkeypatch.setattr(psi4.core.HF, "finalize_energy", finalize_with_test_local_tweak)
    psi4.set_options({"reference": "uhf", "basis": "sto-3g", "dft_grac_shift": 0.0})
    try:
        _, mutated_cation = psi4.energy("pbe0", molecule=cation.molecule(), return_wfn=True)
    finally:
        monkeypatch.setattr(psi4.core.HF, "finalize_energy", original_finalize)
        psi4.set_options({"reference": "rhf"})
    # The post-capture tweak cannot alter the sealed functional record, so the
    # factory reaches the fixed-shift check and rejects the recomputed energy.
    with pytest.raises(RuntimeError, match=r"actual applied GRAC shift must equal IP plus HOMO energy"):
        psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, mutated_cation)


def test_wrong_cation_calculation_rejects(grac_states):
    grac, precursor, _, _ = grac_states
    with pytest.raises(RuntimeError, match=r"cation.*(charge|doublet|UKS|identity)"):
        psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, precursor)


def test_wrong_cation_multiplicity_real_scf_fails_closed_before_protocol_validation(grac_states):
    grac, precursor, _, _ = grac_states
    quartet = psi4.geometry(
        """
        1 4
        O  0.000000  0.000000  0.000000
        H  0.000000  0.757160  0.586260
        H  0.000000 -0.757160  0.586260
        symmetry c1
        units angstrom
        """
    )
    psi4.set_options({"reference": "uhf", "basis": "sto-3g", "dft_grac_shift": 0.0})
    try:
        _, quartet_wfn = psi4.energy("pbe0", molecule=quartet, return_wfn=True)
        with pytest.raises(RuntimeError, match=r"no finalized provenance seal"):
            psi4.core._atomic_polarizability_make_frozen_response_context(grac, precursor, quartet_wfn)
    finally:
        psi4.set_options({"reference": "rhf"})


def test_complete_basis_mismatch_real_scf_fails_closed_before_protocol_validation(grac_states):
    grac, precursor, cation, _ = grac_states
    psi4.set_options({"reference": "uhf", "basis": "3-21g", "dft_grac_shift": 0.0})
    try:
        _, wrong_basis_cation = psi4.energy("pbe0", molecule=cation.molecule(), return_wfn=True)
        with pytest.raises(RuntimeError, match=r"no finalized provenance seal"):
            psi4.core._atomic_polarizability_make_frozen_response_context(
                grac, precursor, wrong_basis_cation
            )
    finally:
        psi4.set_options({"reference": "rhf", "basis": "sto-3g"})


def test_pure_vertical_protocol_validator_rejects_downstream_branches():
    validator = psi4.core._atomic_polarizability_validate_vertical_protocol
    assert validator(True, True) is None
    with pytest.raises(RuntimeError, match=r"charge \+1 doublet UKS"):
        validator(False, True)
    with pytest.raises(RuntimeError, match=r"complete basis structure"):
        validator(True, False)


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
