"""CPU serial-shim exchange equations; NOT actual CUDA/native-factor evidence."""
import ctypes as ct
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

exchange=importlib.import_module(PACKAGE+".sapt_fisapt_exch_cuda")


def fixture(seed,K=7,Q=11,A=2,B=3,R=4,S=5):
    rng=np.random.default_rng(seed)
    # Deliberately nonsymmetric matrices/factors discriminate transpose errors.
    raw={n:rng.normal(size=(K,K))*.1 for n in ("S","V_A","J_A","V_B","J_B")}
    la,lb,ra,rb=(rng.normal(size=(K,n))*.2 for n in (A,B,R,S))
    ao=rng.normal(size=(Q,K,K))*.1
    sab=la.T@raw["S"]@lb;sba=lb.T@raw["S"]@la
    sas=la.T@raw["S"]@rb;sbr=lb.T@raw["S"]@ra
    wbab=(la.T@(raw["V_B"]+2*raw["J_B"])@ra)@sbr.T
    waba=(lb.T@(raw["V_A"]+2*raw["J_A"])@rb)@sas.T
    aar=np.einsum("Qmn,ma,nr->Qar",ao,la,ra)
    abs_=np.einsum("Qmn,mb,ns->Qbs",ao,lb,rb)
    bab=np.einsum("Qar,br->Qab",aar,sbr)
    bba=np.einsum("Qbs,as->Qba",abs_,sas)
    expected=-2*(sab*wbab+sba.T*waba.T+np.einsum("Qab,Qba->ab",bab,bba))
    return (ao,la,lb,ra,rb,raw),expected


def evaluate(session,data,natoms=4):
    ao,la,lb,ra,rb,raw=data
    K,A,B=la.shape[0],la.shape[1],lb.shape[1]
    provider=SimpleNamespace(session=session,nao=K,naux=ao.shape[0])
    # Explicit synthetic CPU test provider, NEVER production native admission.
    issued=[]
    def compute(pairs):
        issued.append(tuple(pairs))
        return {n:session.upload(np.einsum("Qmn,mi,nj->Qij",ao,host(l),host(r)))
                for n,(l,r) in pairs.items()}
    provider.compute_device=compute
    state=SimpleNamespace(provider=provider,require_provider=Mock(),
        matrix=session.empty((natoms+A+1,natoms+B+1),zero=True))
    device=[session.upload(v) for v in (la,lb,ra,rb)]
    fields={n:session.upload(v) for n,v in raw.items()}
    return exchange.fexch_device(state,*device,fields,natoms=natoms),issued,state


class Exchange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        specs={
            "matrix_window":[ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2,
            "linear_combination":[ct.c_longlong]*3+[ct.c_void_p]*3,
            "scatter_matrix":[ct.c_longlong]*6+[ct.c_void_p]*2,
            "reduce_values":[ct.c_longlong]+[ct.c_void_p]*2,
            "reduce_fragment_exchange":[ct.c_longlong]*3+[ct.c_void_p]*7}
        for name,args in specs.items():
            fn=getattr(cls.lib,name);fn.argtypes=args;fn.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_native_linearity_equations_transposes_unequal_changed_inputs_and_slots(self):
        for dims in ((7,11,2,3,4,5),(5,17,1,2,3,1),(9,7,4,3,2,5)):
            for seed in (63,64):
                data,expected=fixture(seed,*dims)
                session=Session(self.lib)
                session.download=lambda *_:self.fail("Exchange intermediate host staging")
                with session.scope():
                    value,issued,state=evaluate(session,data)
                    N=4
                    full=np.zeros_like(host(value.matrix))
                    full[N:-1,N:-1]=expected
                    np.testing.assert_allclose(host(value.matrix),full,rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(host(value.total),[full.sum()],rtol=1e-11,atol=1e-12)
                    self.assertEqual(issued,[("Bab","Bba")])
                    state.require_provider.assert_called_once_with(state.provider)
                self.assertEqual(session.live_bytes,0)

    def test_missing_upstream_matrices_and_layouts_fail_before_compute(self):
        session=Session(self.lib)
        with session.scope():
            provider=SimpleNamespace(session=session,nao=4)
            state=SimpleNamespace(provider=provider,require_provider=Mock(),
                matrix=session.empty((7,8)))
            orbitals=[session.empty((4,n)) for n in (2,3,1,2)]
            raw={n:session.empty((4,4)) for n in ("S","V_A","J_A","V_B","J_B")}
            with self.assertRaises(ValueError):
                exchange.fexch_device(state,*orbitals,dict(raw,extra=raw["S"]),natoms=4)
            with self.assertRaises(ValueError):
                exchange.fexch_device(state,*orbitals,raw,natoms=True)
            with self.assertRaises(ValueError):
                exchange.fexch_device(state,*orbitals,raw,natoms=5)


if __name__=="__main__":unittest.main()
