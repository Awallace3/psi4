"""CPU ordered-Jacobi equation/shader tests, NOT native/GPU localization proof."""
import ctypes as ct
import importlib
import math
from types import SimpleNamespace
import unittest
import numpy as np
from test_cuda_backend import PACKAGE,Kernels,host
from test_cuda_overlap import MetadataSession

localization=importlib.import_module(PACKAGE+".sapt_fisapt_localization_cuda")


def fixture(seed=171,O=4,M=9,K=3):
    rng=np.random.default_rng(seed)
    b=np.linalg.qr(rng.normal(size=(M,O)))[0].T
    energies=np.linspace(-1.4,-.3,O)
    atoms=np.arange(M,dtype=np.int64)%K
    return b,energies,atoms


def reference(b,energies,atoms,*,nfrozen=0,power=4,convergence=1e-12,max_iterations=50):
    B=b.copy();O,M=B.shape;K=int(atoms.max())+1;R=np.eye(O);history=[]
    ranges=((0,nfrozen),(nfrozen,O))
    pairs=[(i,j) for start,stop in ranges for i in range(start,stop) for j in range(start,i)]
    converged=False
    for iteration in range(1,max_iterations+1):
        metric=0.
        for i in range(O):
            for atom in range(K):
                q=0.
                for m in range(M):
                    if atoms[m]==atom:q+=float(B[i,m])*float(B[i,m])
                metric+=math.pow(q,power)
        metric=math.pow(metric,1./power);gradient=0.
        for i,j in pairs:
            a=bb=0.
            for atom in range(K):
                x=y=z=0.
                for m in range(M):
                    if atoms[m]!=atom:continue
                    xi,xj=float(B[i,m]),float(B[j,m])
                    x+=xi*xi;z+=xi*xj;y+=xj*xj
                if power==2:
                    a+=4*z*z-(x-y)*(x-y);bb+=4*z*(x-y)
                else:
                    a+=-x*x*x*x-y*y*y*y+6*(x*x+y*y)*z*z+x*x*x*y+x*y*y*y
                    bb+=4*z*(x*x*x-y*y*y)
            phi=.25*math.atan2(bb,-a);c,s=math.cos(phi),math.sin(phi)
            for value in (B,R):
                ri,rj=value[i].copy(),value[j].copy()
                value[i]=c*ri+s*rj;value[j]=c*rj-s*ri
            gradient+=bb*bb
        gradient=math.sqrt(gradient)
        history.append({"iteration":iteration,"objective":metric,"gradient":gradient})
        converged=gradient<convergence
        if converged:break
    diagonal=np.array([sum(float(energies[k])*float(R[i,k])*float(R[i,k]) for k in range(O)) for i in range(O)])
    P=np.zeros((O,O))
    for start,stop in ranges:
        order=sorted(range(start,stop),key=lambda i:(diagonal[i],i))
        for i,old in enumerate(order,start):P[i,old]=1.
    U=R.T@P;B=P.T@B
    Q=np.zeros((K,O))
    for i in range(O):
        for m in range(M):Q[atoms[m],i]+=B[i,m]*B[i,m]
    return SimpleNamespace(u=U,f=U.T@np.diag(energies)@U,b=B,q=Q,p=P,history=history,converged=converged)


class Localization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        specs={
            "set_identity":[ct.c_longlong]+[ct.c_void_p],
            "ibo_objective":[ct.c_longlong]*4+[ct.c_void_p]*3,
            "ibo_pair":[ct.c_longlong]*6+[ct.c_void_p]*4,
            "ibo_sweep_finish":[ct.c_void_p]*3,
            "ibo_fock_diagonal":[ct.c_longlong]+[ct.c_void_p]*3,
            "ibo_permutation":[ct.c_longlong]*2+[ct.c_void_p]*2,
            "ibo_charges":[ct.c_longlong]*3+[ct.c_void_p]*3,
            "ibo_population_error":[ct.c_longlong]*2+[ct.c_void_p]*2,
            "matrix_error":[ct.c_longlong]+[ct.c_void_p]*2,
            "rotation_error":[ct.c_longlong]+[ct.c_void_p]*2,
            "matrix_window":[ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2,
            "scale_eigenvector_rows":[ct.c_longlong]+[ct.c_void_p]*3,
            "linear_combination":[ct.c_longlong]*3+[ct.c_void_p]*3}
        for name,args in specs.items():
            fn=getattr(cls.lib,name);fn.argtypes=args;fn.restype=None

    @classmethod
    def tearDownClass(cls):Kernels.tearDownClass.__func__(cls)

    def test_ordered_pairs_both_powers_frozen_ranges_history_and_literal_reordering(self):
        for seed in (171,172):
            data=fixture(seed)
            for nfrozen in (0,1,2,4):
                for power in (2,4):
                    expected=reference(*data,nfrozen=nfrozen,power=power)
                    session=MetadataSession(self.lib)
                    with session.scope():
                        actual=localization.jacobi_localize_device(session,session.upload(data[0]),
                            session.upload(data[1]),data[2],nfrozen=nfrozen,power=power,require_converged=False)
                        for value,truth in ((actual.u,expected.u),(actual.f,expected.f),(actual.q,expected.q),
                                           (actual.occupied_iao,expected.b),(actual.permutation,expected.p)):
                            np.testing.assert_allclose(host(value),truth,rtol=1e-11,atol=1e-12)
                        np.testing.assert_allclose([[x["objective"],x["gradient"]] for x in actual.history],
                            [[x["objective"],x["gradient"]] for x in expected.history],rtol=1e-10,atol=1e-12)
                        self.assertEqual(actual.converged,expected.converged)
                        np.testing.assert_allclose(host(actual.u).T@host(actual.u),np.eye(4),rtol=1e-12,atol=1e-12)
                        np.testing.assert_array_equal(host(actual.u)[:nfrozen,nfrozen:],0.)
                    self.assertEqual(session.live_bytes,0)

    def test_permutation_three_cycle_and_equal_energy_tie_index(self):
        session=MetadataSession(self.lib)
        with session.scope():
            diagonal=session.upload(np.array([3.,1.,2.,8.,8.]))
            permutation=session.empty((5,5),zero=True)
            session.launch("ibo_permutation",1,256,[5,3,diagonal,permutation])
            expected=np.zeros((5,5));expected[0,1]=expected[1,2]=expected[2,0]=expected[3,3]=expected[4,4]=1.
            np.testing.assert_array_equal(host(permutation),expected)
            self.assertFalse(np.array_equal(expected,expected.T))

    def test_complete_ao_outputs_and_mandatory_convergence(self):
        b=np.zeros((2,4));theta=.27
        b[:,:2]=np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
        energies=np.array([-1.3,-.7]);atoms=np.arange(4,dtype=np.int64)
        expected=reference(b,energies,atoms)
        self.assertTrue(expected.converged)
        session=MetadataSession(self.lib)
        with session.scope():
            S=session.upload(np.eye(4));c=session.upload(b.T)
            iao=SimpleNamespace(a=session.upload(np.eye(4)),coefficients=c,occupied_iao=session.upload(b),
                                metrics=SimpleNamespace(session=session,overlaps=SimpleNamespace(s11=S)))
            actual=localization.localize_iao_device(iao,session.upload(energies),atoms)
            np.testing.assert_allclose(host(actual.l),b.T@expected.u,rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(host(actual.q),expected.q,rtol=1e-11,atol=1e-12)
            with self.assertRaises(ValueError):
                localization.localize_iao_device(iao,session.upload(energies),atoms,require_converged=False)

    def test_nonconvergence_and_bad_metadata_fail_closed(self):
        b,e,atoms=fixture()
        session=MetadataSession(self.lib)
        with session.scope():
            db,de=session.upload(b),session.upload(e)
            with self.assertRaisesRegex(RuntimeError,"sweep limit"):
                localization.jacobi_localize_device(session,db,de,atoms,max_iterations=1,convergence=0.)
            for bad in (np.full(9,3,dtype=np.int64),np.full(9,2**60,dtype=np.int64),np.arange(9,dtype=np.float64)):
                with self.assertRaises(ValueError):
                    localization.jacobi_localize_device(session,db,de,bad)
            with self.assertRaises(ValueError):
                localization.jacobi_localize_device(session,db,de,atoms,nfrozen=True)


if __name__=="__main__":unittest.main()
