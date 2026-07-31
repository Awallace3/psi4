"""Dependency-free guards for the SCF response-provenance boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative):
    return (ROOT / relative).read_text()


def test_python_scf_callback_uses_one_scope_at_the_outermost_boundary():
    source = _text("psi4/driver/procrouting/scf_proc/scf_iterator.py")
    body = source[source.index("def scf_compute_energy"):source.index("def _build_jk")]
    assert "with self._response_provenance_scope() as provenance:" in body
    assert body.index("with self._response_provenance_scope()") < body.index("self.initialize()")
    assert body.index("self.finalize_energy()") < body.index("provenance.success()")
    assert "primary_converged and not self._response_stability_exhausted" in body
    assert "_set_response_state_converged" not in body


def test_cpp_scope_is_generation_bound_and_no_hf_setter_is_exported():
    header = _text("psi4/src/psi4/libscf_solver/hf.h")
    implementation = _text("psi4/src/psi4/libscf_solver/hf.cc")
    exports = _text("psi4/src/export_wavefunction.cc")
    assert "HFResponseProvenanceScope(const HFResponseProvenanceScope&) = delete" in header
    assert "generation != response_scope_generation_" in implementation
    assert "capture_response_provenance()" in implementation
    assert "set_response_state_converged" not in header
    assert "_set_response_state_converged" not in exports


def test_seal_records_complete_functional_and_basis_state():
    hf = _text("psi4/src/psi4/libscf_solver/hf.h")
    basis = _text("psi4/src/psi4/libmints/basisset.h")
    factory = _text("psi4/src/psi4/libmints/atomic_polarizability.cc")
    for field in (
        "libxc_id", "libxc_canonical_name", "effective_parameters", "x_alpha", "x_beta",
        "c_ss_alpha", "c_os_alpha", "vv10_beta", "density_tolerance", "grac_shift",
        "grac_alpha", "grac_beta", "functional_workers", "sealed_functional", "occupied_homo",
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


def test_production_test_mutator_exports_are_absent():
    exports = _text("psi4/src/export_oeprop.cc")
    assert "_atomic_polarizability_mutate_grac_component_for_test" not in exports
    assert "_atomic_polarizability_restore_grac_component_for_test" not in exports
