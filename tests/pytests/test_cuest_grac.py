"""Restricted GRAC with GPU XC: potentials must affect orbitals, not just energies."""

import numpy as np
import psi4
import pytest

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.cuest]


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.parametrize("functional", ["svwn", "pbe0", "tpss", "cam-b3lyp"])
def test_cuest_grac_scf(functional, tmp_path):
    """Cover LDA promoted to GGA by GRAC, hybrids, meta-GGA, and range separation.

    Both references use cuEST J/K to isolate the XC implementation. Compare
    densities and orbital energies too: GRAC modifies the potential, not the
    underlying XC energy expression. Also require a nonzero shift effect.
    """
    def run(gpu_xc, shift, threads=4):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.set_output_file(str(tmp_path / f"{functional}-{gpu_xc}-{shift}-{threads}.out"), False)
        psi4.set_num_threads(threads)
        molecule = psi4.geometry("""
            0 1
            O 0.0 0.0 0.0
            H 0.0 0.0 0.96
            H 0.91 0.0 -0.24
            symmetry c1
        """)
        psi4.set_options({
            "basis": "cc-pvdz",
            "scf_type": "df",
            "reference": "rhf",
            "use_cuest": True,
            "cuest_xc": gpu_xc,
            "cuest_mixed_precision": False,
            "dft_grac_shift": shift,
            "dft_grac_alpha": 0.6,
            "dft_grac_beta": 35.0,
            "e_convergence": 10,
            "d_convergence": 9,
            "maxiter": 150,
        })
        try:
            energy, wfn = psi4.energy(functional, molecule=molecule, return_wfn=True)
            if gpu_xc and shift:
                # Do not silently use the uncorrected analytic-gradient branch.
                with pytest.raises(RuntimeError, match="analytic gradients with GRAC"):
                    wfn.V_potential().compute_gradient()
            return energy, wfn.Da().np.copy(), wfn.epsilon_a().np.copy()
        finally:
            psi4.core.close_outfile()
            psi4.core.clean()

    ref = run(False, 0.136)
    gpu = run(True, 0.136)
    unshifted = run(True, 0.0)
    np.testing.assert_allclose(gpu[0], ref[0], atol=1.e-6, rtol=0)
    np.testing.assert_allclose(gpu[1], ref[1], atol=1.e-5, rtol=0)
    np.testing.assert_allclose(gpu[2], ref[2], atol=1.e-5, rtol=0)
    assert np.max(np.abs(gpu[2] - unshifted[2])) > 1.e-3
    if functional == "pbe0":
        # The cuEST GRAC host functional is block-parallel. Pin agreement with
        # its serial execution so worker ownership/order cannot change results.
        serial = run(True, 0.136, threads=1)
        np.testing.assert_allclose(gpu[0], serial[0], atol=1.e-10, rtol=0)
        np.testing.assert_allclose(gpu[1], serial[1], atol=1.e-10, rtol=0)
        np.testing.assert_allclose(gpu[2], serial[2], atol=1.e-10, rtol=0)
