"""CPU serial-shim fragment-elst equations; NOT actual GPU/native evidence."""
import ctypes as ct
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np

from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_hf_setup import Session

elst=importlib.import_module(PACKAGE+".sapt_fisapt_elst_cuda")
point=importlib.import_module(PACKAGE+".sapt_cuest_potential_cuda")


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

    def test_native_orchestration_preserves_factors_and_original_provider_gauge(self):
        data,expected=fixture(41)
        xyz,za,zb,la,lb,faa,fbb,va,vb=data
        session=Session(self.lib)
        primary,auxiliary=object(),object()
        owner=SimpleNamespace(handle=7,plan=8,primary_snapshot=SimpleNamespace(ncenter=len(za)),
                              auxiliary_snapshot=SimpleNamespace(owner=auxiliary))
        provider=SimpleNamespace(session=session,handle=7,plan=8,keepalive=(owner,),
            nao=la.shape[0],naux=faa.shape[0],primary_basis=primary,auxiliary_basis=auxiliary,
            validate_basis=Mock())
        potentials=SimpleNamespace(session=session,owner=owner,num_charges=1,primary_basis=primary)
        issued=[]
        def factors(pairs):
            self.assertEqual(set(pairs),{"aa","bb"})
            issued.append("DFMO")
            return {"aa":session.upload(faa),"bb":session.upload(fbb)}
        provider.compute_device=factors
        def potential(x,q):
            index=len([i for i in issued if i=="potential"])
            atom=index//2
            # Sign and device subviews are checked separately from reference data.
            np.testing.assert_array_equal(host(x),xyz[atom:atom+1])
            np.testing.assert_array_equal(host(q),-(za if index%2==0 else zb)[atom:atom+1])
            issued.append("potential")
            return session.upload((va if index%2==0 else vb)[atom])
        potentials.compute_device=potential
        with session.scope():
            inputs=[session.upload(value) for value in (la,lb,xyz,za,zb)]
            state=elst.felst_device(provider,potentials,*inputs,jkfit_basis=auxiliary)
            self.assertIs(state.require_provider(provider),state)
            np.testing.assert_allclose(host(state.matrix),expected,rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(host(state.total),[expected.sum()],rtol=1e-11,atol=1e-12)
            self.assertFalse(state.qa.allocation.freed or state.qb.allocation.freed)
            with self.assertRaises(ValueError):state.require_provider(object())
            with self.assertRaises(ValueError):
                elst.felst_device(provider,potentials,*inputs,jkfit_basis=object())
        with self.assertRaises(RuntimeError):state.require_provider(provider)
        self.assertEqual(issued.count("DFMO"),1)
        self.assertEqual(issued.count("potential"),2*len(za))
        self.assertEqual(session.live_bytes,0)

    def test_point_potential_failure_retains_pending_inputs_and_invalidates_parent(self):
        for fail_sync in (False,True):
            session=Session(self.lib)
            owner=SimpleNamespace(_context=ct.c_void_p(1),_raw=lambda v:v,
                assert_live=Mock(),_invalidate=Mock())
            def status(label,code):
                if code:raise RuntimeError("native compute failed")
            native=SimpleNamespace(cuestPotentialCompute=Mock(return_value=0 if fail_sync else 7))
            owner.api=SimpleNamespace(native=native,status=status)
            potential=point.CuESTPointPotential.__new__(point.CuESTPointPotential)
            potential.owner,potential.session=owner,session
            potential.num_charges,potential.nao=1,3
            potential.failed,potential.pending=False,[]
            potential.plan,potential.parameters=ct.c_void_p(2),ct.c_void_p(3)
            potential.workspace=point.abi._Workspace();potential.calls=0
            potential._bind=Mock();potential._precision=Mock()
            session.synchronize_device=Mock(side_effect=RuntimeError("sync failed")) if fail_sync else Mock()
            with session.scope():
                xyz,q=session.upload(np.ones((1,3))),session.upload(np.ones(1))
                with self.assertRaises(RuntimeError):potential.compute_device(xyz,q)
                self.assertEqual(len(potential.pending),1)
                self.assertTrue(potential.failed);owner._invalidate.assert_called_once()
                with self.assertRaises(RuntimeError):potential.compute_device(xyz,q)
                self.assertEqual(native.cuestPotentialCompute.call_count,1)

    def test_point_potential_precision_drift_is_read_only_and_rejected(self):
        potential=point.CuESTPointPotential.__new__(point.CuESTPointPotential)
        for ffloat,jit in ((2,0),(0,0),(2,1)):
            def query(kind,parameters,attribute,value,size):
                ct.cast(value,ct.POINTER(ct.c_int))[0]=(ffloat,jit)[attribute]
                return 0
            native=SimpleNamespace(cuestParametersQuery=Mock(side_effect=query))
            owner=SimpleNamespace(_raw=lambda x:x,handle=1,
                api=SimpleNamespace(native=native,status=lambda *args:None))
            potential.owner,potential.parameters=owner,ct.c_void_p(2)
            with patch.object(point.identity,"native_math_mode",return_value=1):
                if (ffloat,jit)==(2,0):potential._precision()
                else:
                    with self.assertRaisesRegex(RuntimeError,"parameter drift"):potential._precision()
            self.assertFalse(hasattr(native,"cuestParametersConfigure"))


if __name__=="__main__":unittest.main()
