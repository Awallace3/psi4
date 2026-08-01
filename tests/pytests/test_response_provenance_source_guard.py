"""Dependency-free guards for the SCF response-provenance boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative):
    return (ROOT / relative).read_text()


def test_python_scf_callback_has_no_response_lifecycle_authority():
    source = _text("psi4/driver/procrouting/scf_proc/scf_iterator.py")
    body = source[source.index("def scf_compute_energy"):source.index("def _build_jk")]
    for lifecycle_name in (
        "_response_provenance_scope",
        "_reset_response_provenance_tracking",
        "_mark_response_compute_failed",
        "_invalidate_response_provenance",
        "_record_response_iteration_state",
        "_capture_response_provenance_if_converged",
    ):
        assert lifecycle_name not in source
    assert "self.initialize()" in body
    assert "self.iterations()" in body
    assert "self.finalize_energy()" in body


def test_cpp_lifecycle_is_native_only_and_capture_occurs_in_finalize():
    header = _text("psi4/src/psi4/libscf_solver/hf.h")
    implementation = _text("psi4/src/psi4/libscf_solver/hf.cc")
    exports = _text("psi4/src/export_wavefunction.cc")
    for binding in (
        "_reset_response_provenance_tracking",
        "_mark_response_compute_failed",
        "_invalidate_response_provenance",
        "_record_response_iteration_state",
        "_capture_response_provenance_if_converged",
    ):
        assert f'def("{binding}"' not in exports
    protected = header[header.index("class HF"):header.index("public:", header.index("class HF"))]
    assert "begin_response_iteration();" in protected
    assert "record_response_iteration_state();" in protected
    assert "capture_response_provenance_if_converged();" in protected
    finalizer = implementation[implementation.index("void HF::finalize()"):implementation.index("void HF::set_jk")]
    assert "capture_response_provenance_if_converged();" in finalizer


def test_capture_checks_native_iteration_metrics_thresholds_and_finalization():
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
        "response_native_iteration_id_",
        "response_last_observed_iteration_id_",
        "response_distinct_iterations_observed_ < 2",
    ):
        assert fact in implementation
    guess_start = implementation.index("void HF::guess()")
    assert guess_start < implementation.index("reset_response_provenance_tracking();", guess_start)


def test_cosx_final_grid_is_observed_by_native_density_lifecycle():
    driver = _text("psi4/driver/procrouting/scf_proc/scf_iterator.py")
    implementation = _text("psi4/src/psi4/libscf_solver/hf.cc")
    options = _text("psi4/src/read_options.cc")
    loop = driver[driver.index("# SCF iterations!"):driver.index("if self.iteration_ >= core.get_option('SCF', 'MAXITER')")]
    assert "_record_response_iteration_state" not in driver
    for reference in ("rhf.cc", "uhf.cc", "rohf.cc", "cuhf.cc"):
        native = _text(f"psi4/src/psi4/libscf_solver/{reference}")
        assert "begin_response_iteration();" in native
        assert "record_response_iteration_state();" in native
    assert 'composite_jk->get_COSX_grid() == "Final"' in implementation
    assert 'self.jk().set_COSX_grid("Final")' in loop
    assert "scf_maxiter_post_screening == 0" in loop
    assert "scf_maxiter_post_screening > 0" in loop
    assert "scf_maxiter_post_screening < -1" in driver
    assert 'options.add_int("COSX_MAXITER_FINAL", 1)' in options


def test_seal_records_complete_functional_basis_and_ordered_grid_state():
    hf = _text("psi4/src/psi4/libscf_solver/hf.h")
    hf_source = _text("psi4/src/psi4/libscf_solver/hf.cc")
    basis = _text("psi4/src/psi4/libmints/basisset.h")
    factory = _text("psi4/src/psi4/libmints/atomic_polarizability.cc")
    isa = _text("psi4/src/psi4/libmints/isa_weights.cc")
    superfunctional = _text("psi4/src/psi4/libfunctional/superfunctional.cc")
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
    assert 'cation_seal.reference == "UKS"' in factory
    assert "structural_snapshot() == *grac_seal.basis" in factory
    assert "verify_basis_unchanged" in factory
    assert "functional_density_tolerance" in factory
    assert "functional density tolerance must be finite and positive" in hf_source
    assert "!(functional_density_tolerance > 0.0)" in factory
    worker_region = superfunctional[superfunctional.index("SuperFunctional::build_worker"):
                                    superfunctional.index("SuperFunctional::print")]
    assert worker_region.count("set_density_tolerance(density_tolerance_)") == 2
    assert 'digest.string("native-real-space-isa-context-v3")' in isa
    assert "digest.scalar(context.functional_density_tolerance())" in isa
    assert "V_potential()" not in factory[factory.index("FrozenResponseContext::create"):factory.index("ISAWeights::ISAWeights")]


def test_basis_alias_contract_defers_exclusive_ownership_to_production_compute():
    header = _text("psi4/src/psi4/libmints/atomic_polarizability.h")
    assert "Production compute must resolve exclusive ownership" in header
    assert "not a current response-success claim" in header


def test_production_test_mutator_exports_are_absent():
    exports = _text("psi4/src/export_oeprop.cc")
    assert "_atomic_polarizability_mutate_grac_component_for_test" not in exports
    assert "_atomic_polarizability_restore_grac_component_for_test" not in exports


def test_vertical_protocol_test_seam_is_pure_and_read_only():
    exports = _text("psi4/src/export_oeprop.cc")
    factory = _text("psi4/src/psi4/libmints/atomic_polarizability.cc")
    start = exports.index('m.def("_atomic_polarizability_validate_vertical_protocol"')
    seam = exports[start:exports.index("m.def(", start + 6)]
    assert "[](bool cation_state_valid, bool complete_basis_valid)" in seam
    assert "detail::validate_vertical_protocol" in seam
    assert "Wavefunction" not in seam
    assert factory.count("detail::validate_vertical_protocol") == 3
