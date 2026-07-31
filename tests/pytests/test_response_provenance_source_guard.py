"""Dependency-free guards for the SCF response-provenance boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative):
    return (ROOT / relative).read_text()


def test_python_scf_callback_records_facts_and_captures_after_finalize():
    source = _text("psi4/driver/procrouting/scf_proc/scf_iterator.py")
    body = source[source.index("def scf_compute_energy"):source.index("def _build_jk")]
    assert "_response_provenance_scope" not in body
    assert "provenance.success" not in body
    assert body.index("_reset_response_provenance_tracking") < body.index("self.initialize()")
    assert body.index("self.finalize_energy()") < body.index("_capture_response_provenance_if_converged")
    assert "_invalidate_response_provenance" in body


def test_cpp_capture_has_no_caller_controlled_seal_capability():
    header = _text("psi4/src/psi4/libscf_solver/hf.h")
    implementation = _text("psi4/src/psi4/libscf_solver/hf.cc")
    exports = _text("psi4/src/export_wavefunction.cc")
    assert "HFResponseProvenanceScope" not in header
    assert "HFResponseProvenanceScope" not in implementation
    assert "_HFResponseProvenanceScope" not in exports
    assert "response_provenance_scope" not in exports
    assert "capture_response_provenance_if_converged()" in implementation
    assert 'def("_capture_response_provenance_if_converged"' in exports
    assert "set_response_state_converged" not in header
    assert "_set_response_state_converged" not in exports


def test_capture_checks_internal_metrics_thresholds_and_finalization():
    implementation = _text("psi4/src/psi4/libscf_solver/hf.cc")
    for fact in (
        "response_iteration_metrics_valid_",
        "response_finalize_completed_",
        'options_.get_double("E_CONVERGENCE")',
        'options_.get_double("D_CONVERGENCE")',
        "response_e_convergence_",
        "response_d_convergence_",
        "response_last_energy_change_",
        "response_last_density_norm_",
        "response_last_iteration_final_grid_",
    ):
        assert fact in implementation


def test_cosx_final_grid_is_observed_before_finite_break():
    driver = _text("psi4/driver/procrouting/scf_proc/scf_iterator.py")
    options = _text("psi4/src/read_options.cc")
    loop = driver[driver.index("# SCF iterations!"):driver.index("if self.iteration_ >= core.get_option('SCF', 'MAXITER')")]
    assert loop.index("self._record_response_iteration_state()") < loop.index("scf_iter_post_screening += 1")
    assert 'self.jk().set_COSX_grid("Final")' in loop
    assert "scf_maxiter_post_screening == 0" in loop
    assert "scf_maxiter_post_screening > 0" in loop
    assert "scf_maxiter_post_screening < -1" in driver
    assert 'options.add_int("COSX_MAXITER_FINAL", 1)' in options


def test_seal_records_complete_functional_basis_and_ordered_grid_state():
    hf = _text("psi4/src/psi4/libscf_solver/hf.h")
    basis = _text("psi4/src/psi4/libmints/basisset.h")
    factory = _text("psi4/src/psi4/libmints/atomic_polarizability.cc")
    for field in (
        "libxc_id", "libxc_canonical_name", "effective_parameters", "x_alpha", "x_beta",
        "c_ss_alpha", "c_os_alpha", "vv10_beta", "density_tolerance", "grac_shift",
        "grac_alpha", "grac_beta", "functional_workers", "sealed_functional", "occupied_homo",
        "grid_points", "grid_weights", "grid_blocks", "potential_grac_initialized",
    ):
        assert field in hf
    for field in (
        "shells", "ecp_shells", "puream", "unique_exponents", "unique_coefficients",
        "unique_original_coefficients", "unique_ecp_exponents", "unique_ecp_coefficients",
        "unique_ecp_radial_powers", "centers",
    ):
        assert field in basis
    assert "same_ground_state" in factory
    assert "GRAC master/worker effective state" in factory
    assert 'cation_seal.reference != "UKS"' in factory
    assert "structural_snapshot() != *grac_seal.basis" in factory
    assert "verify_basis_unchanged" in factory
    assert "V_potential()" not in factory[factory.index("FrozenResponseContext::create"):factory.index("ISAWeights::ISAWeights")]


def test_basis_alias_contract_defers_exclusive_ownership_to_production_compute():
    header = _text("psi4/src/psi4/libmints/atomic_polarizability.h")
    assert "Production compute must resolve exclusive ownership" in header
    assert "not a current response-success claim" in header


def test_production_test_mutator_exports_are_absent():
    exports = _text("psi4/src/export_oeprop.cc")
    assert "_atomic_polarizability_mutate_grac_component_for_test" not in exports
    assert "_atomic_polarizability_restore_grac_component_for_test" not in exports
