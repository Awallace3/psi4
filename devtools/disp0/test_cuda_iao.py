"""CPU algebra/native-power double for IAO construction, NOT GPU IBO proof."""
import ctypes as ct
import importlib
from types import SimpleNamespace
import unittest
import numpy as np
from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

iao=importlib.import_module(PACKAGE+".sapt_fisapt_iao_cuda")
overlap=importlib.import_module(PACKAGE+".sapt_cuest_overlap_cuda")


def root(value,condition=1e-7):
    values,vectors=np.linalg.eigh(value);weights=np.zeros_like(values)
    keep=(values>0)&~(np.abs(values)<condition*np.max(np.abs(values)))
    weights[keep]=1/np.sqrt(values[keep])
    return (vectors*weights)@vectors.T


def fixture(seed,N=9,M=6,O=3):
    rng=np.random.default_rng(seed)
    x,y=rng.normal(size=(N+M,N)),rng.normal(size=(N+M,M))
    s11,s22=x.T@x+np.eye(N),y.T@y+np.eye(M)
    s12=x.T@y
    c=root(s11)@np.linalg.qr(rng.normal(size=(N,O)))[0]
    # Independent explicit reference expression with the same spectral policy.
    h1,h2=root(s11),root(s22)
    t1=h2@s12.T;t2=h1@(t1.T@t1@c)
    tilde=h1@t2@root(t2.T@t2)
    d,dt=c@c.T,tilde@tilde.T
    an=(h1@h1+2*d@s11@dt-d-dt)@s12
    a=an@root(an.T@s11@an)
    return (s11,s12,s22,c),a,c.T@s11@a


class PowerDouble:
    """Explicit CPU reference double, never imported by production modules."""
    def __init__(self,session):
        self.session,self.calls=session,0
        self.workspace_limit_bytes=0  # This CPU double has no native workspace.
    def inverse_sqrt(self,value,*,condition):
        self.calls+=1
        return SimpleNamespace(matrix=self.session.upload(root(host(value),condition)))


class IAO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        cls.lib.linear_combination.argtypes=[ct.c_longlong]*3+[ct.c_void_p]*3
        cls.lib.matrix_window.argtypes=[ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2
        cls.lib.linear_combination.restype=cls.lib.matrix_window.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_native_recipe_changed_inputs_metric_reuse_and_occupied_rotation_invariance(self):
        for seed in (161,162):
            data,expected_a,expected_b=fixture(seed)
            s11,s12,s22,c=data;session=Session(self.lib)
            session.download=lambda *_:self.fail("IAO staged an intermediate matrix")
            with session.scope():
                blocks=overlap.OverlapBlocksDevice(*(session.upload(x) for x in (s11,s12,s22)))
                power=PowerDouble(session)
                metrics=iao.prepare_iao_metrics(blocks,power)
                baseline=session.live_bytes
                actual=iao.build_iao_device(session.upload(c),metrics)
                self.assertLessEqual(session.peak_live_bytes-baseline,
                    actual.work_plan["peak_increment_bytes"]+c.nbytes)
                np.testing.assert_allclose(host(actual.a),expected_a,rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(host(actual.occupied_iao),expected_b,rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(host(actual.a).T@s11@host(actual.a),np.eye(s22.shape[0]),rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(np.sum(host(actual.occupied_iao)**2,axis=1),1.,rtol=1e-10,atol=1e-12)
                rotation=np.linalg.qr(np.random.default_rng(seed+1).normal(size=(3,3)))[0]
                rotated=iao.build_iao_device(session.upload(c@rotation),metrics)
                np.testing.assert_allclose(host(rotated.a),expected_a,rtol=1e-10,atol=1e-12)
                self.assertEqual(power.calls,6)  # Two shared roots, two per occupied space.
                self.assertIs(actual.metrics,metrics)
                self.assertFalse(actual.a.allocation.freed or actual.occupied_iao.allocation.freed)
            self.assertEqual(session.live_bytes,0)

    def test_invalid_overlap_and_occupied_shapes_fail_before_power_or_arithmetic(self):
        session=Session(self.lib)
        with session.scope():
            power=PowerDouble(session)
            blocks=overlap.OverlapBlocksDevice(session.empty((5,5)),session.empty((5,3)),session.empty((4,4)))
            with self.assertRaises(ValueError):iao.prepare_iao_metrics(blocks,power)
            self.assertEqual(power.calls,0)
            data,_,_=fixture(163)
            blocks=overlap.OverlapBlocksDevice(*(session.upload(x) for x in data[:3]))
            metrics=iao.prepare_iao_metrics(blocks,power)
            with self.assertRaises(ValueError):iao.build_iao_device(session.empty((9,7)),metrics)
            with self.assertRaises(ValueError):iao.prepare_iao_metrics(blocks,power,condition=np.nan)


if __name__=="__main__":unittest.main()
