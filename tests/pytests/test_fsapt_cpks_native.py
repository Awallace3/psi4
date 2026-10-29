"""Opt-in matching-build tests; never substitute a donor core for qualification.

Run only after building this candidate:
PSI4_RUN_FSAPT_CPKS_NATIVE=1 python -m unittest discover -s tests/pytests \
    -p 'test_fsapt_cpks_native.py' -v
"""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("PSI4_RUN_FSAPT_CPKS_NATIVE") == "1",
                     "requires an explicitly selected matching native build")
class NativeFindCPKS(unittest.TestCase):
    def run_reference(self, functional):
        import psi4
        from psi4.driver.procrouting.sapt.sapt_proc import sapt_dft

        self.assertTrue(hasattr(psi4.core.FISAPT, "set_cpks_response_callbacks"),
                        "Build this candidate; an older core cannot qualify the new path")
        psi4.set_num_threads(2)
        psi4.set_memory("256 MiB")
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.set_options({
            "basis": "aug-cc-pvdz", "scf_type": "df", "reference": "rhf" if functional == "HF" else "rks",
            "e_convergence": 1e-11, "d_convergence": 1e-10, "freeze_core": False,
            "dft_grac_shift": 0., "level_shift": 0.,
            "dft_radial_points": 75, "dft_spherical_points": 302,
            "sapt_dft_functional": functional, "sapt_dft_do_fsapt": "fisapt",
            "sapt_dft_induction_type": "cpks", "sapt_dft_do_ddft": False,
            "sapt_dft_do_dhf": False, "sapt_dft_grac_compute": "none",
            "sapt_dft_grac_shift_a": 0., "sapt_dft_grac_shift_b": 0.,
            "cphf_r_convergence": 1e-10, "maxiter": 128, "do_ind_exch_sinf": False,
            "fisapt_do_fsapt": True, "fisapt_do_fsapt_disp": False,
            "fisapt_fsapt_ind_response": True, "fisapt_fsapt_ind_xc_response": True,
            "fisapt_fsapt_ind_scale": False, "fisapt_link_assignment": "c",
            "ssapt0_scale": False, "fisapt_exch_parperp": False,
        })
        molecule = psi4.geometry("""
            0 1
            He 0 0 0
            --
            0 1
            Ne 0 0 4
            units angstrom
            symmetry c1
            no_reorient
            no_com
        """)
        with tempfile.TemporaryDirectory(prefix="fsapt-cpks-native-") as directory:
            psi4.set_options({"fisapt_fsapt_filepath": str(Path(directory) / "fsapt")})
            a = psi4.energy(functional, molecule=molecule.extract_subsets(1, 2), return_wfn=True)[1]
            b = psi4.energy(functional, molecule=molecule.extract_subsets(2, 1), return_wfn=True)[1]
            from psi4.driver.procrouting.sapt.sapt_fisapt_cpks import NativeMonomerCPKS
            self.assertFalse(a.has_external_cpscf_terms())
            a.set_external_cpscf_perturbation("admission-test", lambda density: density)
            try:
                with self.assertRaisesRegex(NotImplementedError, "Stored external"):
                    NativeMonomerCPKS(psi4.core, a, None, grac_shift=0., orbital_shift=0.,
                                      external_response=False)
                self.assertTrue(a.has_external_cpscf_terms())  # rejection does not clear caller state
            finally:
                a.clear_external_cpscf_perturbations()  # only this test's owned callback
            dimer = psi4.core.Wavefunction.build(molecule, a.basisset())
            data = sapt_dft(dimer, a, b, do_dft=functional != "HF", do_disp=False, delta_hf=False)
            for side, canonical in (("AB", "A<-B"), ("BA", "A->B")):
                induction = dimer.array_variable(f"FSAPT_CPKS_IND20R_{side}_RAW").np
                exchange = dimer.array_variable(f"FSAPT_CPKS_EXCHIND20R_{side}_RAW").np
                np.testing.assert_allclose(induction.sum(), data[f"Ind20,r ({canonical})"],
                                           rtol=1e-7, atol=2e-9)
                np.testing.assert_allclose(exchange.sum(), data[f"Exch-Ind20,r ({canonical})"],
                                           rtol=1e-7, atol=2e-9)
            for direction in ("A<-B", "B<-A"):
                self.assertEqual(dimer.scalar_variable(f"FSAPT CPKS RESPONSE SCALE ({direction})"), 1.)
                self.assertEqual(dimer.scalar_variable(f"FSAPT CPKS HF REFERENCE LIMIT ({direction})"),
                                 float(functional == "HF"))
            self.assertEqual(dimer.scalar_variable("FSAPT CPKS DELTAHF SCALE"), 1.)

    def tearDown(self):
        import psi4
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()

    def test_hf_reference_limit(self):
        self.run_reference("HF")

    def test_pbe_original_whole_monomer_xc(self):
        self.run_reference("PBE")

    def test_svwn_original_whole_monomer_xc(self):
        self.run_reference("SVWN")
