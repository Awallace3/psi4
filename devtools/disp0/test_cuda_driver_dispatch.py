"""CPU-only public SAPT driver dispatch contracts; no Psi4/CUDA execution."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock
import numpy as np


class ValidationError(ValueError):
    pass


class AutomaticCudaDriver(unittest.TestCase):
    def fixture(self):
        path=Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt/sapt_proc.py"
        names={"_validate_einsums_dispersion","_automatic_cuda_dispersion","sapt_dft","_run_sapt_dft"}
        tree=ast.Module(body=[n for n in ast.parse(path.read_text()).body
                            if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        options={
            ("SCF","REFERENCE"):"RHF",
            ("SAPT","SAPT_DFT_INDUCTION_TYPE"):"NONE",
            ("SAPT","SAPT_DFT_DO_FSAPT"):"NONE",
            ("SAPT","SAPT_DFT_MP2_DISP_ALG"):"SAPT",
            ("SAPT","SAPT_DFT_USE_EINSUMS"):True,
            ("SAPT","SAPT_DFT_FUNCTIONAL"):"HF",
            ("SAPT","SAPT_DFT_DO_DISP"):True,
            ("SAPT","SAPT_DFT_GRAC_COMPUTE"):"NONE",
            ("SAPT","SAPT_DFT_EINSUMS_DISP_BLOCK"):24,
            ("FISAPT","FISAPT_FSAPT_FILEPATH"):"NONE",
        }
        global_options={"USE_CUEST":True,"BASIS":"cc-pvdz","FREEZE_CORE":False}
        core=types.SimpleNamespace(Wavefunction=object,JK=object,
            get_option=lambda module,name:options.get((module,name),False),
            get_global_option=lambda name:global_options.get(name,False),
            prepare_options_for_module=Mock(),has_global_option_changed=lambda name:False,
            set_global_option=Mock(),
            get_active_molecule=lambda:object(),
            timer_on=Mock(),timer_off=Mock(),print_out=Mock(),set_variable=Mock(),
            has_option_changed=lambda *args:False,release_freed_memory=Mock(),
            BasisSet=types.SimpleNamespace(build=Mock(return_value=object())))
        terms=types.SimpleNamespace(
            build_sapt_jk_cache=Mock(return_value={}),
            electrostatics=Mock(return_value=({"Elst10,r":-1.},0.)),
            exchange=Mock(return_value={"Exch10":.1}),
            dispersion0_cuda=Mock(return_value={"Disp20,u":-.01,"Exch-Disp20,u":.001,
                                               "Disp0 execution":{"backend":"fixture-CUDA"}}))
        mp2=types.SimpleNamespace(df_mp2_sapt_dispersion=Mock(
            return_value={"Disp20,u":-.02,"Exch-Disp20,u":.002}),
            df_mp2_einsums_dispersion=Mock(side_effect=AssertionError("Unexpected E2 dispersion")),
            df_mp2_fisapt_dispersion=Mock(side_effect=AssertionError("Unexpected native FISAPT dispersion")))
        ns={"core":core,"np":np,"ValidationError":ValidationError,"einsums_available":True,
            "sapt_jk_terms_ein":terms,"sapt_mp2_terms_ein":mp2,
            "sapt_dft_header":Mock(),"print_sapt_dft_summary":Mock(return_value="")}
        ns["proc_util"]=types.SimpleNamespace(prepare_sapt_molecule=lambda mol,mode:(mol,mol,mol))
        ns["scf_helper"]=Mock(side_effect=AssertionError("SCF must not start before missing-factory rejection"))
        exec(compile(tree,str(path),"exec"),ns)
        functional=types.SimpleNamespace(is_x_lrc=lambda:False,is_x_hybrid=lambda:False)
        basis=types.SimpleNamespace(n_frozen_core=lambda *args:0)
        wave=types.SimpleNamespace(functional=lambda:functional,basisset=lambda:basis,
                                   molecule=lambda:object())
        return ns,core,options,global_options,terms,mp2,wave

    def test_use_cuest_hf_without_factory_fails_before_cache_setup(self):
        ns,_,_,_,terms,mp2,wave=self.fixture()
        with self.assertRaisesRegex(ValidationError,"audited.*factory"):
            ns["sapt_dft"](wave,wave,wave,do_dft=False,sapt_jk=Mock(),print_header=False)
        terms.build_sapt_jk_cache.assert_not_called()
        mp2.df_mp2_sapt_dispersion.assert_not_called()

    def test_use_cuest_hf_selects_cuda_even_with_default_sapt_algorithm(self):
        ns,_,_,_,terms,mp2,wave=self.fixture()
        factory=Mock()
        actual=ns["sapt_dft"](wave,wave,wave,do_dft=False,sapt_jk=Mock(),print_header=False,
                              cuda_provider_factory=factory,cuda_dispersion_block_size=5)
        self.assertEqual(actual["Disp20,u"],-.01)
        self.assertEqual(actual["Exch-Disp20,u"],.001)
        self.assertNotIn("Disp0 execution",actual)  # Scalar QCVariables cannot hold dictionaries.
        self.assertEqual(wave._sapt_cuda_dispersion_execution,{"backend":"fixture-CUDA"})
        self.assertIs(terms.dispersion0_cuda.call_args.kwargs["provider_factory"],factory)
        self.assertEqual(terms.dispersion0_cuda.call_args.kwargs["block_size"],5)
        mp2.df_mp2_sapt_dispersion.assert_not_called()

    def test_unsupported_cuda_requests_fail_before_cache_setup(self):
        cases=(
            ("fragment",{("SAPT","SAPT_DFT_DO_FSAPT"):"SAPTDFT"},{},{}),
            ("native-fragment",{("SAPT","SAPT_DFT_DO_FSAPT"):"FISAPT"},{},{}),
            ("no-e2",{("SAPT","SAPT_DFT_USE_EINSUMS"):False},{},{}),
            ("conflicting-gpu",{("SAPT","SAPT_DFT_EINSUMS_GPU"):True},{},{}),
            ("external",{},{"external_potentials":{"A":[[1.,0.,0.,0.]]}},{}),
            ("zero-block",{},{"cuda_dispersion_block_size":0},{}),
            ("boolean-block",{},{"cuda_dispersion_block_size":True},{}),
            ("fractional-block",{},{"cuda_dispersion_block_size":1.5},{}),
            ("missing-e2-package",{},{},{"einsums_available":False}),
        )
        for label,changes,arguments,globals_ in cases:
            with self.subTest(label=label):
                ns,_,options,_,terms,_,wave=self.fixture();options.update(changes);ns.update(globals_)
                with self.assertRaises(ValidationError):
                    ns["sapt_dft"](wave,wave,wave,do_dft=False,sapt_jk=Mock(),print_header=False,
                                   cuda_provider_factory=Mock(),**arguments)
                terms.build_sapt_jk_cache.assert_not_called()

    def test_energy_route_rejects_missing_factory_before_scf(self):
        ns,_,_,_,_,_,wave=self.fixture()
        with self.assertRaisesRegex(ValidationError,"audited.*factory"):
            ns["_run_sapt_dft"]("sapt(dft)",molecule=wave.molecule())
        ns["scf_helper"].assert_not_called()

    def test_native_failure_never_retries_a_cpu_dispersion_algorithm(self):
        ns,_,_,_,terms,mp2,wave=self.fixture()
        terms.dispersion0_cuda.side_effect=RuntimeError("native execution failed")
        with self.assertRaisesRegex(RuntimeError,"native execution failed"):
            ns["sapt_dft"](wave,wave,wave,do_dft=False,sapt_jk=Mock(),print_header=False,
                           cuda_provider_factory=Mock())
        self.assertEqual(terms.dispersion0_cuda.call_count,1)
        mp2.df_mp2_sapt_dispersion.assert_not_called()
        mp2.df_mp2_einsums_dispersion.assert_not_called()
        mp2.df_mp2_fisapt_dispersion.assert_not_called()

    def test_automatic_hf_requires_explicit_native_fp64_policy(self):
        ns,_,_,global_options,terms,_,wave=self.fixture()
        global_options["CUEST_MIXED_PRECISION"]=True
        with self.assertRaisesRegex(ValidationError,"FP64"):
            ns["sapt_dft"](wave,wave,wave,do_dft=False,sapt_jk=Mock(),print_header=False,
                           cuda_provider_factory=Mock())
        terms.build_sapt_jk_cache.assert_not_called()

    def test_normalized_absent_external_fields_still_select_cuda(self):
        for external in ({},{"A":None,"B":None,"C":None}):
            with self.subTest(external=external):
                ns,_,_,_,terms,mp2,wave=self.fixture()
                actual=ns["sapt_dft"](wave,wave,wave,do_dft=False,sapt_jk=Mock(),print_header=False,
                    cuda_provider_factory=Mock(),external_potentials=external)
                self.assertEqual(actual["Disp20,u"],-.01)
                terms.dispersion0_cuda.assert_called_once()
                mp2.df_mp2_sapt_dispersion.assert_not_called()

    def test_use_cuest_false_preserves_existing_cpu_and_e2_choices(self):
        for algorithm,method in (("SAPT","df_mp2_sapt_dispersion"),("EINSUMS","df_mp2_einsums_dispersion"),
                                 ("FISAPT","df_mp2_fisapt_dispersion")):
            with self.subTest(algorithm=algorithm):
                ns,_,options,global_options,terms,mp2,wave=self.fixture()
                options[("SAPT","SAPT_DFT_MP2_DISP_ALG")]=algorithm;global_options["USE_CUEST"]=False
                compute=getattr(mp2,method);compute.side_effect=None
                compute.return_value={"Disp20,u":-.02,"Exch-Disp20,u":.002}
                actual=ns["sapt_dft"](wave,wave,wave,do_dft=False,sapt_jk=Mock(),print_header=False)
                self.assertEqual(actual["Disp20,u"],-.02)
                compute.assert_called_once();terms.dispersion0_cuda.assert_not_called()
                self.assertFalse(hasattr(wave,"_sapt_cuda_dispersion_execution"))

    def test_non_hf_and_empirical_only_paths_do_not_require_a_factory(self):
        for do_dft,empirical in ((True,None),(False,"SAPT_DFT_D3_IE"),(False,"SAPT_DFT_D4_IE")):
            with self.subTest(do_dft=do_dft,empirical=empirical):
                ns,_,options,_,terms,_,wave=self.fixture()
                if empirical:options[("SAPT",empirical)]=True
                ns["sapt_dft"](wave,wave,wave,do_dft=do_dft,do_disp=False,
                               sapt_jk=Mock(),print_header=False)
                terms.dispersion0_cuda.assert_not_called()


if __name__=="__main__":unittest.main()
