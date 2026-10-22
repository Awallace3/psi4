"""CPU shader/native-call doubles for spectral algebra; NOT cuSOLVER proof."""
import ctypes as ct
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

power=importlib.import_module(PACKAGE+".sapt_cuda_symmetric_power")


def reference(eigenvalues,vectors,condition):
    w=np.zeros_like(eigenvalues);threshold=condition*np.max(np.abs(eigenvalues))
    keep=(eigenvalues>0)&~(np.abs(eigenvalues)<threshold)
    w[keep]=1/np.sqrt(eigenvalues[keep])
    return (vectors*w)@vectors.T,int(keep.sum())


def mock_solver(session,*,info=0,workspace_limit=1024**2):
    """NumPy is an explicit test native-call double, NEVER runtime fallback."""
    owner=SimpleNamespace(session=session,assert_live=Mock(),_invalidate=Mock(),_raw=lambda v:v)
    def status(label,code):
        if code:raise RuntimeError("mock native status")
    owner.api=SimpleNamespace(status=status)
    def query(handle,jobz,uplo,N,A,lda,W,lwork):
        if (jobz,uplo,lda)!=(1,0,N):raise AssertionError("Wrong native layout/enum")
        ct.cast(lwork,ct.POINTER(ct.c_int))[0]=N*N+10
        return 0
    def compute(handle,jobz,uplo,N,A,lda,W,workspace,lwork,observed):
        matrix=np.ctypeslib.as_array(A,shape=(N*N,)).reshape(N,N)
        values,vectors=np.linalg.eigh(matrix.copy())
        matrix[...]=vectors.T  # Native column-major eigenvectors viewed as C.
        np.ctypeslib.as_array(W,shape=(N,))[...]=values
        ct.cast(observed,ct.POINTER(ct.c_int))[0]=info
        return 0
    session.stream=1
    session.runtime=SimpleNamespace(cudaMemsetAsync=lambda p,v,n,s:ct.memset(p,v,n))
    session.rt=lambda result:result
    solver=power.CuSolverSymmetricPower.__new__(power.CuSolverSymmetricPower)
    solver.owner,solver.session=owner,session
    solver.workspace_limit_bytes,solver.max_dimension=workspace_limit,512
    solver.native=SimpleNamespace(cusolverDnDsyevd_bufferSize=Mock(side_effect=query),
                                  cusolverDnDsyevd=Mock(side_effect=compute))
    solver.handle=ct.c_void_p(123);solver._bind=Mock()
    return solver


class SymmetricPower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        specs={
            "inverse_sqrt_weights":[ct.c_longlong]+[ct.c_void_p]*4,
            "scale_eigenvector_rows":[ct.c_longlong]+[ct.c_void_p]*3,
            "linear_combination":[ct.c_longlong]*3+[ct.c_void_p]*3,
            "matrix_window":[ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2,
            "matrix_error":[ct.c_longlong]+[ct.c_void_p]*2}
        for name,args in specs.items():
            fn=getattr(cls.lib,name);fn.argtypes=args;fn.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_spectral_weights_relative_strict_cut_negative_and_native_transpose(self):
        for seed in (141,142):
            rng=np.random.default_rng(seed);vectors=np.linalg.qr(rng.normal(size=(6,6)))[0]
            eigenvalues=np.array([-8.,0.,.5,np.nextafter(1.,0.),1.,8.])
            for condition in (0.,.125,1.):
                expected,rank=reference(eigenvalues,vectors,condition)
                session=Session(self.lib)
                with session.scope():
                    actual,observed=power.reconstruct_inverse_sqrt(
                        session,session.upload(vectors.T),session.upload(eigenvalues),condition=condition)
                    np.testing.assert_allclose(host(actual),expected,rtol=1e-11,atol=1e-12)
                    self.assertEqual(host(observed)[0],rank)
                self.assertEqual(session.live_bytes,0)

    def test_mock_native_info_layout_and_outputs_survive_eigen_scratch_scope(self):
        rng=np.random.default_rng(143)
        for eigenvalues in (np.array([.1,.3,1.,2.]),np.array([-2.,0.,.5,4.]),np.zeros(4)):
            vectors=np.linalg.qr(rng.normal(size=(4,4)))[0]
            matrix=(vectors*eigenvalues)@vectors.T
            expected,rank=reference(eigenvalues,vectors,1e-7)
            session=Session(self.lib)
            with session.scope():
                solver=mock_solver(session);value=session.upload(matrix)
                actual=solver.inverse_sqrt(value)
                np.testing.assert_allclose(host(actual.matrix),expected,rtol=1e-11,atol=1e-12)
                self.assertEqual(host(actual.rank)[0],rank)
                np.testing.assert_array_equal(host(value),matrix)
                self.assertFalse(actual.matrix.allocation.freed or actual.eigenvalues.allocation.freed)
                solver.owner._invalidate.assert_not_called()
            self.assertEqual(session.live_bytes,0)

    def test_invalid_input_native_info_and_workspace_fail_closed(self):
        with self.assertRaises(TypeError):power.CuSolverSymmetricPower(object())
        for value,info,limit,exception in (
                (np.array([[1.,.5],[0.,1.]]),0,1024**2,ValueError),
                (np.full((2,2),np.nan),0,1024**2,ValueError),
                (np.eye(2),3,1024**2,RuntimeError),(np.eye(2),0,8,MemoryError)):
            session=Session(self.lib)
            with session.scope():
                solver=mock_solver(session,info=info,workspace_limit=limit)
                with self.assertRaises(exception):solver.inverse_sqrt(session.upload(value))
                solver.owner._invalidate.assert_called_once()
            self.assertEqual(session.live_bytes,0)


if __name__=="__main__":unittest.main()
