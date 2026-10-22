"""CPU serial-shim HF Hessian/PCG tests; NOT actual GPU or native DFMO proof."""
import ctypes as ct
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

response=importlib.import_module(PACKAGE+".sapt_fisapt_response_cuda")


def fixture(seed=121,O=2,V=3,Q=4):
    rng=np.random.default_rng(seed);K=O+V
    ao=rng.normal(size=(Q,K,K))*.04;ao=(ao+ao.transpose(0,2,1))/2
    C=np.linalg.qr(rng.normal(size=(K,K)))[0]
    co,cv=C[:,:O],C[:,O:]
    energies={"o":-rng.uniform(.5,1.3,O),"v":rng.uniform(.1,.9,V)}
    mo=np.einsum("Qmn,mp,nq->Qpq",ao,C,C)
    eri=np.einsum("Qpq,Qrs->pqrs",mo,mo)
    H=np.empty((O*V,O*V))
    for i in range(O):
        for a in range(V):
            for j in range(O):
                for b in range(V):
                    H[i*V+a,j*V+b]=(4*eri[i,O+a,j,O+b]-eri[i,j,O+a,O+b]-eri[i,O+b,j,O+a]
                        +((energies["v"][a]-energies["o"][i]) if i==j and a==b else 0.))
    np.testing.assert_allclose(H,H.T,rtol=1e-12,atol=1e-14)
    if np.linalg.eigvalsh(H).min()<=0:raise AssertionError("Unstable positive fixture")
    trials=[rng.normal(size=(O,V)) for _ in range(2)]
    return (ao,co,cv,energies),H,trials


def operator(session,data,**kwargs):
    ao,co,cv,energies=data
    provider=SimpleNamespace(session=session,nao=co.shape[0],naux=ao.shape[0],
        primary_basis=object(),auxiliary_basis=object(),closed=False,failed=False,
        validate_basis=Mock(),calls=0)
    def compute(pairs):
        provider.calls+=1
        if set(pairs)!={"ov","oo","vv"}:raise AssertionError("Missing RHF exchange factors")
        return {n:session.upload(np.einsum("Qmn,mi,nj->Qij",ao,host(l),host(r)))
                for n,(l,r) in pairs.items()}
    provider.compute_device=compute
    value=response.HFResponseOperator(provider,session.upload(co),session.upload(cv),
                                     {n:session.upload(v) for n,v in energies.items()},**kwargs)
    return value,provider


class Response(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        specs={
            "single_gap_error":[ct.c_longlong]*2+[ct.c_void_p]*3,
            "linear_combination":[ct.c_longlong]*3+[ct.c_void_p]*3,
            "rhf_response_product":[ct.c_longlong]*3+[ct.c_void_p]*8,
            "response_dot":[ct.c_longlong]+[ct.c_void_p]*3,
            "response_precondition":[ct.c_longlong]*2+[ct.c_void_p]*4,
            "response_step":[ct.c_longlong]+[ct.c_void_p]*6,
            "response_direction":[ct.c_longlong]+[ct.c_void_p]*4,
            "response_metrics":[ct.c_void_p]*4}
        for name,args in specs.items():
            fn=getattr(cls.lib,name);fn.argtypes=args;fn.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_full_four_index_hessian_and_physical_sign_solutions_changed_rhs(self):
        for dims in ((1,2,3),(2,3,4),(3,4,5)):
            for seed in (121,122):
                data,H,trials=fixture(seed,*dims)
                session=Session(self.lib)
                def scalar_only(array):
                    self.assertLessEqual(array.size,3,"Response staged an intermediate matrix")
                    return np.array(host(array))
                session.download=scalar_only
                with session.scope():
                    op,provider=operator(session,data)
                    for trial in trials:
                        dt=session.upload(trial)
                        actual=op.apply(dt)
                        np.testing.assert_allclose(host(actual).ravel(),H@trial.ravel(),rtol=1e-11,atol=1e-12)
                        solved=op.solve(dt,rtol=1e-12,atol=1e-14)
                        truth=-np.linalg.solve(H,trial.ravel()).reshape(trial.shape)
                        np.testing.assert_allclose(host(solved.amplitudes),truth,rtol=1e-10,atol=1e-12)
                        self.assertLessEqual(solved.relative_residual,1e-12)
                        self.assertGreater(solved.iterations,0)
                    zero=op.solve(session.upload(np.zeros(trials[0].shape)))
                    self.assertEqual(zero.iterations,0)
                    np.testing.assert_array_equal(host(zero.amplitudes),0.)
                    self.assertEqual(provider.calls,1)
                    # Dropping OO/VV (or treating the second exchange as the
                    # transpose of projected K) cannot reproduce this Hessian.
                    b=host(op.factors["ov"]).reshape(op.Q,op.P)
                    gaps=(data[3]["v"][None,:]-data[3]["o"][:,None]).ravel()
                    wrong=np.diag(gaps)+4*b.T@b
                    self.assertGreater(np.max(np.abs(wrong-H)),1e-7)
                    with self.assertRaises(ValueError):op.solve(dt,max_iterations=True)
                    provider.closed=True
                    with self.assertRaises(RuntimeError):op.apply(dt)
                self.assertEqual(session.live_bytes,0)

    def test_negative_curvature_nonconvergence_and_work_admission_fail_closed(self):
        session=Session(self.lib)
        with session.scope():
            data=(np.array([[[2.,0.],[0.,2.]]]),np.array([[1.],[0.]]),np.array([[0.],[1.]]),
                  {"o":np.array([-.5]),"v":np.array([.5])})
            op,_=operator(session,data)
            with self.assertRaisesRegex(RuntimeError,"curvature"):op.solve(session.upload(np.ones((1,1))))
        session=Session(self.lib)
        with session.scope():
            data,H,trials=fixture()
            with self.assertRaisesRegex(ValueError,"admission"):operator(session,data,max_exchange_products=1)
            op,_=operator(session,data)
            with self.assertRaisesRegex(RuntimeError,"iteration limit"):
                op.solve(session.upload(trials[0]),max_iterations=1,rtol=1e-14,atol=1e-16)
        self.assertEqual(session.live_bytes,0)


if __name__=="__main__":unittest.main()
