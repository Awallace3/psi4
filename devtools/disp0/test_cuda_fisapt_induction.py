"""CPU serial-shim source denominators/localization; NOT CUDA/physical proof."""
import ctypes as ct
import importlib
import unittest
import numpy as np
from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

induction=importlib.import_module(PACKAGE+".sapt_fisapt_induction_cuda")


def fixture(seed=81,O=3,V=7,C=5):
    rng=np.random.default_rng(seed)
    sources=rng.normal(size=(C,O,V))
    rotation=np.linalg.qr(rng.normal(size=(O,O)))[0]
    energies={"o":-rng.uniform(.5,1.5,O),"v":rng.uniform(.1,.9,V)}
    targets={"ind":rng.normal(size=(O,V)),"exch":rng.normal(size=(O,V))}
    canonical=sources/(energies["o"][None,:,None]-energies["v"][None,None,:])
    localized=np.einsum("il,cir->clr",rotation,canonical)
    expected={name:2*np.einsum("clr,lr->lc",localized,value) for name,value in targets.items()}
    wrong=np.einsum("il,cir->clr",rotation,sources)/(energies["o"][None,:,None]-energies["v"][None,None,:])
    wrong=2*np.einsum("clr,lr->lc",wrong,targets["ind"])
    return (sources,rotation,energies,targets),expected,wrong


def evaluate(session,data,*,exchange=True):
    sources,rotation,energies,targets=data
    C,O,V=sources.shape
    U=session.upload(rotation)
    e={n:session.upload(v) for n,v in energies.items()}
    t={n:session.upload(v) for n,v in targets.items() if n=="ind" or exchange}
    out={n:session.empty((O,C+1),zero=True) for n in t}
    for column in range(C):
        with session.scope():
            induction.uncoupled_source(session,session.upload(sources[column]),e,U,t,out,column=column)
    return out


class Induction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        for name,args in {
            "make_single_amplitudes":[ct.c_longlong]*2+[ct.c_void_p]*4,
            "reduce_induction_source":[ct.c_longlong]*4+[ct.c_void_p]*5}.items():
            fn=getattr(cls.lib,name);fn.argtypes=args;fn.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_signed_columns_canonical_denominator_order_changed_inputs_and_reserved_slot(self):
        for dims in ((1,3,2),(3,7,5),(4,129,3)):
            for seed in (81,82):
                data,expected,wrong=fixture(seed,*dims)
                for exchange in (False,True):
                    session=Session(self.lib)
                    session.download=lambda *_:self.fail("Induction intermediate host staging")
                    with session.scope():
                        out=evaluate(session,data,exchange=exchange)
                        for name,value in out.items():
                            np.testing.assert_allclose(host(value)[:,:-1],expected[name],rtol=1e-11,atol=1e-12)
                            np.testing.assert_array_equal(host(value)[:,-1],0.)
                    self.assertEqual(session.live_bytes,0)
                if dims[0]>1:self.assertGreater(np.max(np.abs(expected["ind"]-wrong)),1e-3)

    def test_layout_column_and_alias_rejection(self):
        session=Session(self.lib)
        with session.scope():
            source=session.empty((2,3));U=session.empty((2,2))
            e={"o":session.empty((2,)),"v":session.empty((3,))}
            targets={"ind":session.empty((2,3))};outputs={"ind":session.empty((2,5))}
            for column in (-1,True,5):
                with self.assertRaises(ValueError):
                    induction.uncoupled_source(session,source,e,U,targets,outputs,column=column)
            with self.assertRaises(ValueError):
                induction.uncoupled_source(session,source,e,U,targets,dict(ind=source),column=0)
            with self.assertRaises(ValueError):
                induction.uncoupled_source(session,source,e,U,targets,dict(ind=outputs["ind"],exch=outputs["ind"]),column=0)


if __name__=="__main__":unittest.main()
