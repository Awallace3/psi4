"""Independent CPU JK/PCG reference fixtures, not native CPU/GPU evidence."""
import importlib
from types import SimpleNamespace
import unittest
import numpy as np
from test_cuda_backend import PACKAGE
from test_cuda_fisapt_response import fixture

cpu=importlib.import_module(PACKAGE+".sapt_fisapt_cpu_response")


class Matrix:
    def __init__(self,array):self.np=np.array(array,copy=True)
    @staticmethod
    def from_array(array):return Matrix(array)


class CPUJK:
    def __init__(self,ao):
        self.eri=np.einsum("Qmn,Qls->mnls",ao,ao)
        self.left=[];self.right=[];self.initialized=self.finalized=False
    def set_memory(self,value):self.memory=value
    def initialize(self):self.initialized=True
    def finalize(self):self.finalized=True
    def C_clear(self):self.left.clear();self.right.clear()
    def C_left_add(self,value):self.left.append(value.np)
    def C_right_add(self,value):self.right.append(value.np)
    def compute(self):
        density=self.left[0]@self.right[0].T
        self.j=Matrix(np.einsum("mnls,ls->mn",self.eri,density))
        self.k=Matrix(np.einsum("mlns,ls->mn",self.eri,density))
    def J(self):return [self.j]
    def K(self):return [self.k]


def core_double(ao,*,gpu=False):
    created=[]
    def build(*args,**kwargs):
        if kwargs!={"jk_type":"DF"}:raise AssertionError("Explicit CPU DF JK required")
        value=CPUJK(ao);created.append(value);return value
    return SimpleNamespace(Matrix=Matrix,JK=SimpleNamespace(build=build),
                           get_global_option=lambda name:gpu),created


class CPUResponse(unittest.TestCase):
    def test_native_AO_JK_action_matches_independent_four_index_Hessian(self):
        for dims in ((1,2,3),(2,3,4),(3,4,5)):
            data,H,trials=fixture(221,*dims)
            ao,co,cv,e=data;core,created=core_double(ao)
            with cpu.NativeCPUResponse(core,object(),object(),co,cv,e["o"],e["v"],
                                       memory_doubles=1024**2) as operator:
                for x in trials:
                    actual=operator.apply(x)
                    np.testing.assert_allclose(actual.ravel(),H@x.ravel(),rtol=1e-12,atol=1e-13)
                    solved=cpu.solve_positive(operator.apply,operator.gap,x,rtol=1e-12,atol=1e-14)
                    np.testing.assert_allclose(solved.amplitudes.ravel(),-np.linalg.solve(H,x.ravel()),
                                               rtol=1e-10,atol=1e-12)
                    self.assertLessEqual(solved.relative_residual,1e-12)
            self.assertEqual(len(created),1);self.assertTrue(created[0].finalized)
            with self.assertRaises(RuntimeError):operator.apply(trials[0])

    def test_source_partition_sign_rotation_and_changed_sources(self):
        data,H,_=fixture(222,2,3,4);ao,co,cv,e=data;core,created=core_double(ao)
        rng=np.random.default_rng(912)
        rotation=np.linalg.qr(rng.normal(size=(2,2)))[0]
        target,target_x=rng.normal(size=(2,3)),rng.normal(size=(2,3))
        for scale in (1.,-.3):
            sources=scale*rng.normal(size=(4,2,3));sources[-1]=0
            ind,ex,diagnostics=cpu.partition_response(core,object(),object(),co,cv,e["o"],e["v"],
                sources,rotation,target,target_x,memory_doubles=1024**2)
            amplitudes=-np.linalg.solve(H,sources.reshape(4,-1).T).T.reshape(sources.shape)
            for actual,field in ((ind,target),(ex,target_x)):
                expected=2*np.einsum("il,cir,lr->lc",rotation,amplitudes,field)
                np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)
            self.assertEqual(diagnostics[-1]["iterations"],0)
            self.assertTrue(created[-1].finalized)

    def test_curvature_iterations_dtype_and_GPU_rejection_do_not_fallback(self):
        gap=np.ones((2,2));rhs=np.ones((2,2))
        with self.assertRaisesRegex(RuntimeError,"curvature"):
            cpu.solve_positive(lambda x:-x,gap,rhs)
        H=np.diag([1.,2.,3.,4.])
        with self.assertRaisesRegex(RuntimeError,"iteration limit"):
            cpu.solve_positive(lambda x:(H@x.ravel()).reshape(2,2),gap,rhs,
                               rtol=1e-14,atol=1e-16,max_iterations=1)
        for options in ({"rtol":float("nan")},{"max_iterations":True},{"atol":-1.}):
            with self.assertRaises(ValueError):
                cpu.solve_positive(lambda x:x,gap,rhs,**options)
        with self.assertRaises(ValueError):
            cpu.solve_positive(lambda x:x,gap,rhs.astype(np.float32))
        calls=[]
        zero=cpu.solve_positive(lambda x:calls.append(x),gap,np.zeros((2,2)))
        self.assertEqual(zero.iterations,0);self.assertEqual(calls,[])
        data,_,_=fixture();ao,co,cv,e=data;core,created=core_double(ao,gpu=True)
        with self.assertRaisesRegex(ValueError,"USE_CUEST"):
            cpu.NativeCPUResponse(core,object(),object(),co,cv,e["o"],e["v"],memory_doubles=1024**2)
        self.assertEqual(created,[])


if __name__=="__main__":unittest.main()
