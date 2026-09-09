"""GRAC must promote LDA point workers before evaluating auxiliary GGA potentials."""

import numpy as np
import psi4
import pytest

pytestmark = [pytest.mark.psi, pytest.mark.api]


def test_lda_grac_cpu(tmp_path):
    """Regression for missing gradient buffers after HF installs GRAC post-initialize."""
    def run(shift):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.set_output_file(str(tmp_path / f"svwn-{shift}.out"), False)
        molecule = psi4.geometry("""
            0 1
            O 0.0 0.0 0.0
            H 0.0 0.0 0.96
            H 0.91 0.0 -0.24
            symmetry c1
        """)
        psi4.set_options({
            "basis": "cc-pvdz",
            "reference": "rhf",
            "scf_type": "df",
            "use_cuest": False,
            "dft_grac_shift": shift,
            "e_convergence": 9,
            "d_convergence": 8,
        })
        try:
            energy, wfn = psi4.energy("svwn", molecule=molecule, return_wfn=True)
            return energy, wfn.epsilon_a().np.copy()
        finally:
            psi4.core.close_outfile()
            psi4.core.clean()

    unshifted = run(0.0)
    shifted = run(0.136)
    assert np.isfinite(shifted[0])
    assert np.all(np.isfinite(shifted[1]))
    assert np.max(np.abs(shifted[1] - unshifted[1])) > 1.e-3
    assert "Asymptotic Correction" in (tmp_path / "svwn-0.136.out").read_text()


def test_point_worker_ansatz_rebuild_cache(tmp_path):
    """Changing the ansatz must release worker pointers into old cached maps."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.set_output_file(str(tmp_path / "cache-rebuild.out"), False)
    molecule = psi4.geometry("""
        0 1
        H 0 0 0
        H 0 0 0.74
        symmetry c1
    """)
    psi4.set_options({"basis": "sto-3g", "scf_type": "df", "use_cuest": False})
    try:
        _, wfn = psi4.energy("svwn", molecule=molecule, return_wfn=True)
        potential = wfn.V_potential()
        potential.build_collocation_cache(10_000_000)
        # Force every worker to reference a cached map, independent of scheduling.
        block = potential.get_block(0)
        for worker in potential.properties():
            worker.compute_points(block, False)
            worker.set_ansatz(1)
        potential.clear_collocation_cache()
        potential.build_collocation_cache(10_000_000)
        for worker in potential.properties():
            worker.compute_points(block, False)
            assert np.all(np.isfinite(worker.point_values()["GAMMA_AA"].np[:block.npoints()]))
    finally:
        psi4.core.close_outfile()
        psi4.core.clean()
