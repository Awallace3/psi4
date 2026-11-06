"""Opt-in matching-build tests; never substitute a donor core for qualification.

Run only after building this candidate:
PSI4_RUN_FSAPT_CPKS_NATIVE=1 python -m unittest discover -s tests/pytests \
    -p 'test_fsapt_cpks_native.py' -v
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(os.environ.get("PSI4_RUN_FSAPT_CPKS_NATIVE") == "1",
                     "requires an explicitly selected matching native build")
class NativeFindCPKS(unittest.TestCase):
    def check_source_cells(self, core, monomers, native, cache):
        """Independent physical RHSs + dense original-monomer Hessian oracle."""
        matrices = native.matrices()
        for side, other, wfn, order in (("A","B",monomers[0],"AB"), ("B","A",monomers[1],"BA")):
            co,cv=cache["Cocc_"+side].np,cache["Cvir_"+side].np
            u=matrices["Uocc0"+side].np;lo=matrices["Locc0"+other].np
            mol=wfn.molecule();jk=wfn.jk();columns=[]
            for atom,charge in enumerate(native.vectors()["Z"+other].np):
                potential=core.ExternalPotential()
                potential.addCharge(float(charge),mol.x(atom),mol.y(atom),mol.z(atom))
                columns.append(co.T@potential.computePotentialMatrix(wfn.basisset()).np@cv)
            for orbital in lo.T:
                jk.C_clear();c=core.Matrix.from_array(orbital[:,None])
                jk.C_left_add(c);jk.C_right_add(c);jk.compute()
                columns.append(2*co.T@jk.J()[0].np@cv)
            columns.append(np.zeros((co.shape[1],cv.shape[1])))
            sources=np.array(columns);shape=sources.shape[1:];n=np.prod(shape)
            h=np.column_stack([-wfn.cphf_Hx([core.Matrix.from_array(
                np.eye(n)[a].reshape(shape))])[0].np.ravel() for a in range(n)])
            kappa=-np.linalg.solve(h,sources.reshape(len(sources),-1).T).T.reshape(sources.shape)
            S=cache["S"].np;da,db=cache["D_"+side].np,cache["D_"+other].np
            wa=cache["V_"+side].np+2*cache["J_"+side].np
            wb=cache["V_"+other].np+2*cache["J_"+other].np
            ka,kb=cache["K_"+side].np,cache["K_"+other].np
            ko=cache["K_O"].np if side=="A" else cache["K_O"].np.T
            left=-(wa-ka)+S@da@wb+wa@db@S-ko.T
            right=-(wb-kb)+wb@da@S-ko
            ex=(-kb-2*cache["J_O"].np+ko+2*cache["J_P_"+other].np
                +S@db@left+right@db@S)
            local=np.einsum("il,siv->slv",u,kappa)
            for name,target in (("IND20R",u.T@co.T@wb@cv),("EXCHIND20R",u.T@co.T@ex@cv)):
                expected=2*np.einsum("slv,lv->ls",local,target)
                actual=native.matrices()[f"CPKS_{'Ind20r' if name=='IND20R' else 'ExchInd20r'}_{order}_raw"].np
                np.testing.assert_allclose(actual if side=="A" else actual.T,expected,atol=2e-9,rtol=1e-7)

    def run_reference(self, functional):
        import psi4
        from psi4.driver.procrouting.sapt import saptdft_fisapt
        from psi4.driver.procrouting.sapt.sapt_proc import sapt_dft

        self.assertTrue(hasattr(psi4.core.FISAPT, "set_cpks_monomers"),
                        "Build this candidate; an older core cannot qualify the new path")
        psi4.set_num_threads(2)
        psi4.set_memory("256 MiB")
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.set_options({
            "basis": "aug-cc-pvdz", "scf_type": "df", "reference": "rhf" if functional == "HF" else "rks",
            "e_convergence": 1e-11, "d_convergence": 1e-10, "freeze_core": False, "save_jk": True,
            "dft_grac_shift": 0., "level_shift": 0.,
            "dft_radial_points": 75, "dft_spherical_points": 302,
            "sapt_dft_functional": functional, "sapt_dft_do_fsapt": "fisapt",
            "sapt_dft_induction_type": "cpks", "sapt_dft_do_ddft": False,
            "sapt_dft_do_dhf": False, "sapt_dft_do_disp": False, "sapt_dft_grac_compute": "none",
            "sapt_dft_grac_shift_a": 0., "sapt_dft_grac_shift_b": 0.,
            "cphf_r_convergence": 1e-10, "maxiter": 128, "do_ind_exch_sinf": False,
            "fisapt_do_fsapt": True, "fisapt_do_fsapt_disp": False,
            "fisapt_fsapt_ind_response": True, "fisapt_fsapt_ind_xc_response": True,
            "fisapt_fsapt_ind_scale": False, "fisapt_link_assignment": "c",
            "ssapt0_scale": False, "fisapt_exch_parperp": False,
        })
        molecule = psi4.geometry("""
            0 1
            H -.37 0 0
            H .37 0 0
            --
            0 1
            O 0 0 3
            H .76 0 3.58
            H -.76 0 3.58
            units angstrom
            symmetry c1
            no_reorient
            no_com
        """)
        with tempfile.TemporaryDirectory(prefix="fsapt-cpks-native-") as directory:
            psi4.set_options({"fisapt_fsapt_filepath": str(Path(directory) / "fsapt")})
            psi4.set_options({"reference": "rhf"})
            dimer = psi4.energy("HF", molecule=molecule, return_wfn=True)[1]
            psi4.set_options({"reference": "rhf" if functional == "HF" else "rks"})
            a = psi4.energy(functional, molecule=molecule.extract_subsets(1, 2), return_wfn=True)[1]
            b = psi4.energy(functional, molecule=molecule.extract_subsets(2, 1), return_wfn=True)[1]
            self.assertFalse(a.has_external_cpscf_terms())
            a.set_external_cpscf_perturbation("admission-test", lambda density: density)
            try:
                psi4.core.prepare_options_for_module("FISAPT")
                with self.assertRaises(RuntimeError):
                    psi4.core.FISAPT(dimer).set_cpks_monomers(a, b)
                self.assertTrue(a.has_external_cpscf_terms())  # rejection does not clear caller state
            finally:
                a.clear_external_cpscf_perturbations()  # only this test's owned callback
            setup=saptdft_fisapt.setup_fisapt_object;captured=[]
            def capture(*args,**kwargs):
                result=setup(*args,**kwargs)
                captured.append((result,dict(args[3])))  # preserve canonical J_P convention before cache export
                return result
            with patch.object(saptdft_fisapt,"setup_fisapt_object",side_effect=capture):
                data = sapt_dft(dimer, a, b, do_dft=functional != "HF", do_disp=False,
                                delta_hf=False,cleanup_jk=False)
            native,cache=captured.pop()
            self.check_source_cells(psi4.core,(a,b),native,cache)
            del native,cache
            a.jk().finalize()
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
            raw = {key: dimer.array_variable(key).np.copy() for key in (
                "FSAPT_CPKS_INDAB_RAW", "FSAPT_CPKS_INDBA_RAW")}
            psi4.set_options({"fisapt_fsapt_ind_scale": True, "sapt_dft_do_dhf": True})
            data["Delta HF Correction"] = .25*(data["Ind20,r"]+data["Exch-Ind20,r"])
            sapt_dft(dimer, a, b, do_dft=functional != "HF", do_disp=False, delta_hf=True, data=data)
            self.assertAlmostEqual(dimer.scalar_variable("FSAPT CPKS DELTAHF SCALE"), 1.25, places=10)
            for key, value in raw.items():
                np.testing.assert_allclose(dimer.array_variable(key).np, value, atol=2e-9, rtol=1e-7)
            psi4.set_options({"sapt_dft_do_dhf": False, "fisapt_fsapt_ind_scale": False})
            _, production = psi4.energy("sapt(dft)", molecule=molecule, return_wfn=True)
            total = sum(production.array_variable(key).np.sum() for key in raw)
            np.testing.assert_allclose(total, production.scalar_variable("SAPT IND ENERGY"),
                                       atol=2e-9, rtol=1e-7)

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
