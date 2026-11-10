"""CPU serial-shim localized dispersion tests, NOT actual CUDA/FISAPT proof."""
import ctypes as ct
import importlib
import unittest
import numpy as np

from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

fisapt=importlib.import_module(PACKAGE+".sapt_fisapt_cuda")


def inputs(seed=17,A=2,B=3,R=5,S=4):
    rng=np.random.default_rng(seed)
    ua=np.linalg.qr(rng.normal(size=(A,A)))[0]
    ub=np.linalg.qr(rng.normal(size=(B,B)))[0]
    v,x=rng.normal(size=(R,A,S,B)),rng.normal(size=(R,A,S,B))
    w=rng.normal(size=(S,A,R,B))
    e={"a":-rng.uniform(.6,1.5,A),"b":-rng.uniform(.7,1.4,B),
       "r":rng.uniform(.1,.6,R+3),"s":rng.uniform(.2,.7,S+3)}
    return v,x,w,e,ua,ub


def reference(v,x,w,e,ua,ub,r0=1,s0=2):
    R,A,S,B=v.shape
    denominator=(e["a"][None,:,None,None]+e["b"][None,None,None,:]
                 -e["r"][r0:r0+R,None,None,None]-e["s"][None,None,s0:s0+S,None])
    t=v/denominator
    def rotate(value):
        return np.einsum("al,rasb,bk->rlsk",ua,value,ub)
    tl,vl,xl=map(rotate,(t,v,x))
    wl=np.einsum("al,sarb,bk->slrk",ua,w,ub)
    d=4*np.einsum("rasb,rasb->ab",tl,vl)
    ex=-2*(np.einsum("rasb,rasb->ab",tl,xl)+np.einsum("rasb,sarb->ab",tl,wl))
    wrong=4*np.einsum("rasb,rasb->ab",vl/denominator,vl)
    return d,ex,wrong,(t,vl)


def factor_reference(ao,spaces,c,e,ua,ub):
    """Independent test-only full Coulomb/rank-tail tensors, never GPU input."""
    def f(left,right):
        return np.einsum("Qmn,mi,nj->Qij",ao,spaces[left],spaces[right])
    ar,bs=f("r","a"),f("s","b")
    far,fbs=f("r3","a")+f("r","a4"),f("s3","b")+f("s","b4")
    v=np.einsum("Qra,Qsb->rasb",ar,bs)
    x=(np.einsum("Qra,Qsb->rasb",ar,fbs)+np.einsum("Qra,Qsb->rasb",far,bs)
       +np.einsum("ar,bs->rasb",c["Qar"],c["SAbs"])
       +np.einsum("ar,bs->rasb",c["SBar"],c["Qbs"]))
    w=(np.einsum("Qsa,Qrb->sarb",f("s1","a"),f("r1","b"))
       +np.einsum("Qsa,Qrb->sarb",f("s","a2"),f("r","b2"))
       +np.einsum("as,br->sarb",c["Sas"],c["Qbr"])
       +np.einsum("as,br->sarb",c["Qas"],c["Sbr"]))
    return reference(v,x,w,e,ua,ub,0,0)[:2]


class Localized(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        specs={
            "permute_occupied_a":[ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2,
            "make_amplitudes":[ct.c_longlong]*6+[ct.c_void_p]*6,
            "rotation_error":[ct.c_longlong]+[ct.c_void_p]*2,
            "reduce_localized_dispersion":[ct.c_longlong]*4+[ct.c_void_p]*6}
        for name,args in specs.items():
            fn=getattr(cls.lib,name);fn.argtypes=args;fn.restype=None

    @classmethod
    def tearDownClass(cls):
        Kernels.tearDownClass.__func__(cls)

    def test_signed_localized_pairs_unequal_edges_and_repeat_accumulation(self):
        for A,B,R,S in ((1,3,1,7),(2,3,5,4),(4,2,3,5)):
            for seed in (17,18):
                v,x,w,e,ua,ub=inputs(seed,A,B,R,S)
                d,ex,wrong,_=reference(v,x,w,e,ua,ub)
                session=Session(self.lib)
                with session.scope():
                    da,db=session.upload(ua),session.upload(ub)
                    fisapt.validate_rotations(session,da,db)
                    dv,dx,dw=(session.upload(value.reshape(rows,cols)) for value,rows,cols in
                              ((v,R*A,S*B),(x,R*A,S*B),(w,S*A,R*B)))
                    de={name:session.upload(value) for name,value in e.items()}
                    dd,ee=session.empty((A,B),zero=True),session.empty((A,B),zero=True)
                    for repeat in (1,2):
                        fisapt.localized_dispersion_block(session,dv,dx,dw,de,da,db,dd,ee,
                                                         R=R,S=S,r0=1,s0=2)
                        np.testing.assert_allclose(host(dd),repeat*d,rtol=1e-11,atol=1e-12)
                        np.testing.assert_allclose(host(ee),repeat*ex,rtol=1e-11,atol=1e-12)
                    # Canonical global conservation despite non-diagonal LO denominators.
                    den=(e["a"][None,:,None,None]+e["b"][None,None,None,:]
                        -e["r"][1:1+R,None,None,None]-e["s"][None,None,2:2+S,None])
                    np.testing.assert_allclose(d.sum(),4*np.sum(v*v/den),rtol=1e-11,atol=1e-12)
                self.assertEqual(session.live_bytes,0)
                if A>1 and B>1:
                    self.assertGreater(np.max(np.abs(d-wrong)),1e-3)

    def test_rotation_kernel_matches_independent_occupied_einsum(self):
        v,_,_,_,ua,ub=inputs()
        R,A,S,B=v.shape
        session=Session(self.lib)
        with session.scope():
            out=fisapt.occupied_transform(session,session.upload(v.reshape(R*A,S*B)),
                                          session.upload(ua),session.upload(ub),R,S)
            expected=np.einsum("al,rasb,bk->rlsk",ua,v,ub)
            np.testing.assert_allclose(host(out).reshape(R,A,S,B),expected,rtol=1e-12,atol=1e-13)

    def test_orthogonality_and_nonfinite_validation_fail_closed(self):
        for invalid in (np.array([[1.,.2],[0.,1.]]),np.full((2,2),np.nan),np.full((2,2),np.inf)):
            session=Session(self.lib)
            with session.scope():
                with np.errstate(invalid="ignore"),self.assertRaisesRegex(ValueError,"finite/orthogonal"):
                    fisapt.validate_rotations(session,session.upload(invalid),session.upload(np.eye(3)))

    def test_layout_and_shape_guards_precede_numerical_launch(self):
        session=Session(self.lib)
        with session.scope():
            ua,ub=session.upload(np.eye(2)),session.upload(np.eye(3))
            v=session.empty((10,12))
            with self.assertRaises(ValueError):fisapt.occupied_transform(session,v,ua,ub,True,4)
            with self.assertRaises(ValueError):fisapt.occupied_transform(session,v,ua,ub,4,5)
            with self.assertRaises(ValueError):
                fisapt.occupied_transform(session,v.view(v.shape,order="F"),ua,ub,5,4)


if __name__=="__main__":
    unittest.main()
