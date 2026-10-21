"""Execute actual callback/setup functions with CPU fixtures, not Psi4/CUDA."""
import ast
import contextlib
from functools import reduce
from pathlib import Path
import time
import types
import unittest

import numpy as np


class Matrix:
    def __init__(self, name, rows, cols):
        self.np = np.zeros((rows, cols))

    @classmethod
    def from_array(cls, value):
        obj = cls.__new__(cls)
        obj.np = np.array(value, dtype=np.float64, copy=True)
        return obj

    @property
    def shape(self):
        return self.np.shape


class SetupBoundary(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(911)
        nao, qa, na, nb, nr, ns = 7, 9, 3, 4, 5, 4
        cache = {}
        for name, cols in (("Cocc_A",na),("Cocc_B",nb),("Cvir_A",nr),("Cvir_B",ns)):
            cache[name] = Matrix.from_array(rng.normal(size=(nao,cols)))
        for name in ("S","P_A","V_A","J_A","K_A","P_B","V_B","J_B","K_B","K_O"):
            cache[name] = Matrix.from_array(rng.normal(size=(nao,nao)))
        for side in ("A","B"):
            c = cache["Cocc_"+side].np
            cache["D_"+side] = Matrix.from_array(c@c.T)
        for name, value in (("eps_occ_A",-np.arange(na)-1.),
                            ("eps_occ_B",-np.arange(nb)-2.),
                            ("eps_vir_A",np.arange(nr)+.2),
                            ("eps_vir_B",np.arange(ns)+.3)):
            cache[name] = Matrix.from_array(value)
        options = {("SAPT","SAPT_DFT_FUNCTIONAL"):"HF",
                   ("SAPT","SAPT_DFT_DO_DISP"):True,
                   ("SAPT","SAPT_DFT_D3_IE"):False,
                   ("SAPT","SAPT_DFT_D4_IE"):False,
                   ("SAPT","SAPT_DFT_DO_FSAPT"):"NONE",
                   ("SAPT","DO_DISP_EXCH_SINF"):False,
                   ("SAPT","SAPT_DFT_EINSUMS_GPU"):False,
                   ("FISAPT","FISAPT_LINK_ASSIGNMENT"):"SAO0",
                   ("DFMP2","DF_BASIS_MP2"):"fixture-ri"}
        primary = types.SimpleNamespace(nbf=lambda:nao)
        auxiliary = types.SimpleNamespace(nbf=lambda:qa)
        wave = types.SimpleNamespace(nirrep=lambda:1, basisset=lambda:primary,
                                     molecule=lambda:types.SimpleNamespace(natom=lambda:2))
        timers = []

        def forbidden(*a,**kw):
            raise AssertionError("Callback must bypass DFHelper")

        core = types.SimpleNamespace(
            Matrix=Matrix, Vector=Matrix, Wavefunction=object, JK=object,
            DFHelper=forbidden, get_option=lambda m,k:options[(m,k)],
            get_global_option=lambda k:"fixture-orbital",
            BasisSet=types.SimpleNamespace(build=lambda *a:auxiliary),
            timer_on=lambda name:timers.append(("on",name)),
            timer_off=lambda name:timers.append(("off",name)),
            print_out=lambda message:None)

        def chain(factors, transposes):
            return reduce(np.matmul,[x.T if t=="T" else x for x,t in zip(factors,transposes)])

        def chain_into(out,factors,transposes,beta=0.,coef=1.,**kw):
            out[...] = beta*out+coef*chain(factors,transposes)

        def chain_sum(out,terms,**kw):
            out[...] = sum(coef*chain(factors,transposes) for coef,factors,transposes in terms)

        source = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt/sapt_jk_terms_ein.py"
        names = {"_store_dispersion0_components","dispersion0","dispersion0_cuda","fdisp0"}
        functions = [n for n in ast.parse(source.read_text()).body
                     if isinstance(n,ast.FunctionDef) and n.name in names]
        namespace = {"core":core,"np":np,"time":time,"FDISP_BLOCK":24,
                     "_ein":lambda x,*a,**k: x.np if hasattr(x,"np") else np.asarray(x),
                     "_mat":lambda x:Matrix.from_array(x.np if hasattr(x,"np") else x),
                     "_ein_zeros":lambda *shape,**kw:np.zeros(shape),
                     "graph_block":lambda *a:contextlib.nullcontext(),
                     "chain_into":chain_into,"chain_sum":chain_sum}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),"exec"),namespace)
        return namespace,cache,wave,primary,auxiliary,options,timers

    def test_callback_bypasses_dfhelper_and_preserves_setup_freezing_and_cache(self):
        ns,cache,wave,primary,auxiliary,_,timers=self.fixture()
        before={key:value.np.copy() for key,value in cache.items()}
        captured={}
        def callback(**kw):
            captured.update(kw)
            return np.full((2,2),-.25),np.full((2,2),.125),{"scope":"CPU FIXTURE ONLY"}
        result=ns["dispersion0"](cache,wave,1,2,do_print=False,_kernel_callback=callback)
        np.testing.assert_array_equal(captured["spaces"]["a"],before["Cocc_A"][:,1:])
        np.testing.assert_array_equal(captured["spaces"]["b"],before["Cocc_B"][:,2:])
        np.testing.assert_array_equal(captured["energies"]["a"],before["eps_occ_A"][1:])
        self.assertIs(captured["primary_basis"],primary)
        self.assertIs(captured["auxiliary_basis"],auxiliary)
        self.assertEqual(len(captured["spaces"]),12)
        self.assertEqual(len(captured["corrections"]),8)
        # Independent nonsymmetric K_O expression; freezing never changes DA/DB.
        bo=before["Cocc_B"][:,2:]; va=before["Cvir_A"]; S=before["S"]
        expected=(2*bo.T@before["J_A"]@va-bo.T@before["K_A"]@va
                  +bo.T@before["K_O"].T@va
                  -2*bo.T@S@before["D_A"]@before["J_B"]@va
                  -2*bo.T@before["J_A"]@before["D_B"]@S@va
                  -bo.T@S@before["D_A"]@before["V_B"]@va
                  +bo.T@before["V_A"]@before["P_B"]@S@va)
        np.testing.assert_allclose(captured["corrections"]["Qbr"],expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(result["Disp20,u"],-1.)
        self.assertEqual(result["Exch-Disp20,u"],.5)
        self.assertNotIn("Disp_AB",cache)
        for key,value in before.items():
            np.testing.assert_array_equal(cache[key].np,value)
        self.assertEqual(timers[-1],("off","F-SAPT Disp Compute"))

    def test_zero_active_space_does_not_call_factory_or_dfhelper(self):
        for empty in ("occupied","virtual"):
            ns,cache,wave,*_=self.fixture()
            cache["Disp0 execution"]={"backend":"STALE PREVIOUS GPU CALL"}
            nfa=3 if empty=="occupied" else 0
            if empty=="virtual":
                cache["Cvir_A"]=Matrix.from_array(np.empty((7,0)))
                cache["eps_vir_A"]=Matrix.from_array(np.empty(0))
            result=ns["dispersion0"](cache,wave,nfa,0,do_print=False,
                                     _kernel_callback=lambda **kw:self.fail("zero-space callback"))
            self.assertEqual(result["Disp20,u"],0.)
            self.assertEqual(result["Exch-Disp20,u"],0.)
            self.assertNotIn("Disp0 execution",result)
            self.assertEqual(cache["Disp0 execution"]["backend"],"STALE PREVIOUS GPU CALL")

    def test_default_path_still_reaches_original_dfhelper_without_cuda(self):
        ns,cache,wave,*_=self.fixture()
        with self.assertRaisesRegex(AssertionError,"bypass DFHelper"):
            ns["dispersion0"](cache,wave,do_print=False)

    def test_bad_callback_shape_and_conflicts_fail(self):
        ns,cache,wave,*_=self.fixture()
        with self.assertRaises(ValueError):
            ns["dispersion0"](cache,wave,do_print=False,
                              _kernel_callback=lambda **kw:(np.zeros((1,1)),np.zeros((1,1)),{}))
        with self.assertRaises(ValueError):
            ns["dispersion0"](cache,wave,do_print=False,_graph_device="cpu",
                              _kernel_callback=lambda **kw:None)
        with self.assertRaises(ValueError):
            ns["fdisp0"]({}, {}, None, None, None, None, False,
                         _canonical=False,_kernel_callback=lambda **kw:None)

    def test_public_cuda_guards_precede_wavefunction_or_gpu_access(self):
        for option,value in (("SAPT_DFT_FUNCTIONAL","PBE0"),("SAPT_DFT_DO_DISP",False),
                             ("SAPT_DFT_D3_IE",True),("SAPT_DFT_D4_IE",True),
                             ("SAPT_DFT_DO_FSAPT","FISAPT"),("DO_DISP_EXCH_SINF",True),
                             ("SAPT_DFT_EINSUMS_GPU",True)):
            ns,cache,_,_,_,options,_=self.fixture()
            options[("SAPT",option)]=value
            with self.subTest(option=option),self.assertRaises((ValueError,NotImplementedError)):
                ns["dispersion0_cuda"](cache,None,provider_factory=lambda *a:None)


if __name__=="__main__":
    unittest.main()
