"""CPU serial-shim fragment-elst equations; NOT actual GPU/native evidence."""
import ctypes as ct
import importlib
import unittest
import numpy as np

from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

elst=importlib.import_module(PACKAGE+".sapt_fisapt_elst_cuda")


def fixture(seed,N=7,A=3,B=2,Q=9,K=11):
    rng=np.random.default_rng(seed)
    xyz=rng.normal(size=(N,3))
    za=np.zeros(N);zb=np.zeros(N)
    za[:N//2]=rng.uniform(.2,2.,N//2)
    zb[N//2:]=rng.uniform(.3,2.,N-N//2)
    la,lb=rng.normal(size=(K,A)),rng.normal(size=(K,B))
    faa,fbb=rng.normal(size=(Q,A,A)),rng.normal(size=(Q,B,B))
    va,vb=rng.normal(size=(N,K,K)),rng.normal(size=(N,K,K))
    va=(va+va.transpose(0,2,1))/2;vb=(vb+vb.transpose(0,2,1))/2
    # Separate test-only AO potential fixtures include their respective charges.
    va*=za[:,None,None];vb*=zb[:,None,None]
    expected=np.zeros((N+A+1,N+B+1))
    for a in range(N):
        for b in range(N):
            if a!=b:
                expected[a,b]=za[a]*zb[b]/np.linalg.norm(xyz[a]-xyz[b])
    qa=np.diagonal(faa,axis1=1,axis2=2)
    qb=np.diagonal(fbb,axis1=1,axis2=2)
    expected[N:N+A,N:N+B]=4*qa.T@qb
    expected[:N,N:N+B]=2*np.einsum("mb,imn,nb->ib",lb,va,lb)
    expected[N:N+A,:N]=2*np.einsum("ma,imn,na->ai",la,vb,la)
    return (xyz,za,zb,la,lb,faa,fbb,va,vb),expected


def evaluate(session,data):
    xyz,za,zb,la,lb,faa,fbb,va,vb=data
    N,A,B=len(za),la.shape[1],lb.shape[1]
    # All these uploads are synthetic test inputs, NOT native-factor coverage.
    dxyz,dza,dzb,dla,dlb,dfaa,dfbb=(session.upload(x) for x in data[:7])
    qa,qb=elst.diagonal_factors(session,dfaa),elst.diagonal_factors(session,dfbb)
    output=session.empty((N+A+1,N+B+1),zero=True)
    with session.scope():
        elst.scatter_matrix(session,elst.nuclear_matrix(session,dxyz,dza,dzb),output,0,0)
    elst.electronic_pairs(session,qa,qb,output,N)
    for atom in range(N):
        with session.scope():
            elst.nuclear_electronic(session,session.upload(va[atom]),dlb,output,
                                    atom,side="A",natoms=N)
            elst.nuclear_electronic(session,session.upload(vb[atom]),dla,output,
                                    atom,side="B",natoms=N)
    return output,elst.sum_values(session,output),qa,qb


class Electrostatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        specs={
            "diagonal_factors":[ct.c_longlong]*2+[ct.c_void_p]*2,
            "scatter_matrix":[ct.c_longlong]*6+[ct.c_void_p]*2,
            "fragment_nuclear":[ct.c_longlong]+[ct.c_void_p]*4,
            "reduce_values":[ct.c_longlong]+[ct.c_void_p]*2}
        for name,args in specs.items():
            fn=getattr(cls.lib,name);fn.argtypes=args;fn.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_matrix_all_slots_unequal_spaces_changed_inputs_and_factors_retained(self):
        for dims in ((3,1,2,4,5),(7,3,2,9,11),(5,2,4,8,9)):
            for seed in (41,42):
                data,expected=fixture(seed,*dims)
                session=Session(self.lib)
                session.download=lambda *_:self.fail("Intermediate host staging")
                with session.scope():
                    out,total,qa,qb=evaluate(session,data)
                    np.testing.assert_allclose(host(out),expected,rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(host(total),[expected.sum()],rtol=1e-11,atol=1e-12)
                    np.testing.assert_array_equal(host(qa),np.diagonal(data[5],axis1=1,axis2=2))
                    np.testing.assert_array_equal(host(qb),np.diagonal(data[6],axis1=1,axis2=2))
                    self.assertFalse(qa.allocation.freed or qb.allocation.freed)
                    np.testing.assert_array_equal(host(out)[-1,:],0.)
                    np.testing.assert_array_equal(host(out)[:,-1],0.)
                self.assertEqual(session.live_bytes,0)

    def test_placement_shapes_weights_and_cross_session_rejected(self):
        session=Session(self.lib);other=Session(self.lib)
        with session.scope(),other.scope():
            out=session.empty((5,6),zero=True);value=session.empty((2,3))
            for row,col,weight in ((-1,0,1),(4,0,1),(0,4,1),(True,0,1),(0,0,1.5),(0,0,5)):
                with self.assertRaises(ValueError):
                    elst.scatter_matrix(session,value,out,row,col,weight=weight)
            with self.assertRaises(ValueError):elst.scatter_matrix(session,out,out,0,0)
            with self.assertRaises(ValueError):elst.scatter_matrix(session,other.empty((2,3)),out,0,0)
            with self.assertRaises(ValueError):elst.diagonal_factors(session,value)
            with self.assertRaises(ValueError):
                elst.diagonal_factors(session,session.empty((2,3,3)).view((2,3,3),order="F"))
            with self.assertRaises(ValueError):
                elst.electronic_pairs(session,value,value,out,1)
            with self.assertRaises(ValueError):
                elst.nuclear_electronic(session,session.empty((4,4)),value,out,0,side="C",natoms=2)

    def test_coincident_charged_centers_are_not_silently_clamped(self):
        session=Session(self.lib)
        with session.scope():
            xyz=session.upload(np.zeros((2,3)))
            za=session.upload(np.array([1.,0.]));zb=session.upload(np.array([0.,1.]))
            out=elst.nuclear_matrix(session,xyz,za,zb)
            self.assertTrue(np.isinf(host(out)[0,1]))
            np.testing.assert_array_equal(host(out)[1,:],0.)


if __name__=="__main__":unittest.main()
