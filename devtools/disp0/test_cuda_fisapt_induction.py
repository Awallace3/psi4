"""CPU serial-shim source denominators/localization; NOT CUDA/physical proof."""
import ctypes as ct
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
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


def target_fixture(seed,K=9,A=3,V=5,Q=17,O=4):
    rng=np.random.default_rng(seed)
    matrices={n:rng.normal(size=(K,K))*.1 for n in
              ("S","D_A","D_B","V_A","V_B","J_A","J_B","K_A","K_B","J_O","K_O")}
    localized,virtual=rng.normal(size=(K,A)),rng.normal(size=(K,V))
    density,factors=rng.normal(size=(Q,O,O)),rng.normal(size=(Q,A,V))
    jp=np.einsum("Qkk,Qar->ar",density,factors)
    expected={}
    for side,other in (("A","B"),("B","A")):
        # Expanded independent transcription of all fourteen AO product terms.
        S=matrices["S"];da,db=matrices["D_"+side],matrices["D_"+other]
        va,vb=matrices["V_"+side],matrices["V_"+other]
        ja,jb=matrices["J_"+side],matrices["J_"+other]
        ka,kb=matrices["K_"+side],matrices["K_"+other]
        ko=matrices["K_O"] if side=="A" else matrices["K_O"].T
        operator=(-kb-2*matrices["J_O"]+ko
            -S@db@va-2*S@db@ja+S@db@ka
            +S@db@S@da@vb+2*S@db@S@da@jb
            +S@db@va@db@S+2*S@db@ja@db@S-S@db@ko.T
            -vb@db@S-2*jb@db@S+kb@db@S
            +vb@da@S@db@S+2*jb@da@S@db@S-ko@db@S)
        expected[side]=localized.T@operator@virtual+2*jp
    return (matrices,localized,virtual,density,factors),jp,expected


class Induction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Kernels.setUpClass.__func__(cls)
        for name,args in {
            "make_single_amplitudes":[ct.c_longlong]*2+[ct.c_void_p]*4,
            "reduce_induction_source":[ct.c_longlong]*4+[ct.c_void_p]*5,
            "matrix_error":[ct.c_longlong]+[ct.c_void_p]*2,
            "single_gap_error":[ct.c_longlong]*2+[ct.c_void_p]*3,
            "linear_combination":[ct.c_longlong]*3+[ct.c_void_p]*3,
            "matrix_window":[ct.c_longlong]*4+[ct.c_int]+[ct.c_void_p]*2,
            "rotation_error":[ct.c_longlong]+[ct.c_void_p]*2,
            "scatter_matrix":[ct.c_longlong]*6+[ct.c_void_p]*2,
            "reduce_values":[ct.c_longlong]+[ct.c_void_p]*2,
            "trace_factors":[ct.c_longlong]*2+[ct.c_void_p]*2}.items():
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

    def test_full_occupied_rotation_span_and_gap_validation(self):
        rng=np.random.default_rng(9)
        canonical=np.linalg.qr(rng.normal(size=(7,7)))[0][:,:3]
        U=np.linalg.qr(rng.normal(size=(3,3)))[0]
        session=Session(self.lib)
        for gap_bad,span_bad in ((False,False),(True,False),(False,True)):
            with session.scope():
                localized=canonical@U
                if span_bad:localized=localized+0.1
                args=[session.upload(x) for x in (canonical,localized,np.eye(7),
                    np.array([-.8,-.6,-.3]),np.array([-.4,.2]) if gap_bad else np.array([.1,.2]))]
                if gap_bad or span_bad:
                    with self.assertRaises(ValueError):induction.occupied_rotation(session,*args)
                else:
                    value=induction.occupied_rotation(session,*args)
                    np.testing.assert_allclose(host(value),U,rtol=1e-12,atol=1e-12)

    def test_native_source_orchestration_signed_directional_slots_and_source_sum(self):
        # Synthetic CPU provider is a test double, NOT native execution proof.
        for seed in (101,102):
            rng=np.random.default_rng(seed);K,Q,N,A,B=7,5,4,2,3
            ca=np.linalg.qr(rng.normal(size=(K,K)))[0]
            cb=np.linalg.qr(rng.normal(size=(K,K)))[0]
            occ={"A":ca[:,:A],"B":cb[:,:B]};vir={"A":ca[:,A:],"B":cb[:,B:]}
            rotations={s:np.linalg.qr(rng.normal(size=(n,n)))[0] for s,n in (("A",A),("B",B))}
            local={s:occ[s]@rotations[s] for s in ("A","B")}
            ao=rng.normal(size=(Q,K,K))*.1;ao=(ao+ao.transpose(0,2,1))/2
            def factor(left,right):return np.einsum("Qmn,mi,nj->Qij",ao,left,right)
            q={s:np.diagonal(factor(local[s],local[s]),axis1=1,axis2=2) for s in ("A","B")}
            za=np.array([1.,2.,0.,0.]);zb=np.array([0.,0.,1.,2.])
            xyz=rng.normal(size=(N,3))
            nuclear=rng.normal(size=(N,K,K))*.1;nuclear=(nuclear+nuclear.transpose(0,2,1))/2
            fields={"S":np.eye(K),"V_A":-np.einsum("i,imn->mn",za,nuclear),
                "V_B":-np.einsum("i,imn->mn",zb,nuclear)}
            fields.update({f"J_{s}":np.einsum("Q,Qmn->mn",q[s].sum(axis=1),ao) for s in ("A","B")})
            energies={"a":np.array([-.9,-.6]),"b":np.array([-.8,-.5,-.3]),
                "r":np.linspace(.1,.7,K-A),"s":np.linspace(.2,.8,K-B)}
            expected={}
            for side,other,charge,o,v in (("A","B",zb,"a","r"),("B","A",za,"b","s")):
                nuc=-np.einsum("mi,Amn,nr,A->Air",occ[side],nuclear,vir[side],charge)
                electronic=2*np.einsum("Qk,Qir->kir",q[other],factor(occ[side],vir[side]))
                sources=np.concatenate((nuc,electronic),axis=0)
                target=local[side].T@(fields["V_"+other]+2*fields["J_"+other])@vir[side]
                amplitudes=sources/(energies[o][None,:,None]-energies[v][None,None,:])
                pair=2*np.einsum("il,cir,lr->lc",rotations[side],amplitudes,target)
                canonical_target=occ[side].T@(fields["V_"+other]+2*fields["J_"+other])@vir[side]
                np.testing.assert_allclose(sources.sum(axis=0),canonical_target,rtol=1e-12,atol=1e-12)
                expected[side]=np.zeros((N+A+1,N+B+1))
                if side=="A":expected[side][N:N+A,:-1]=pair
                else:expected[side][:-1,N:N+B]=pair.T
            session=Session(self.lib)
            with session.scope():
                primary=object()
                owner=SimpleNamespace(handle=7,plan=8,primary_snapshot=SimpleNamespace(ncenter=N))
                provider=SimpleNamespace(session=session,handle=7,plan=8,keepalive=(owner,),
                    primary_basis=primary,nao=K,naux=Q)
                calls=[]
                def compute(pairs):
                    calls.append("DFMO")
                    return {n:session.upload(factor(host(l),host(r))) for n,(l,r) in pairs.items()}
                provider.compute_device=compute
                potential=SimpleNamespace(session=session,owner=owner,num_charges=1,primary_basis=primary)
                def potentials(point,charge):
                    calls.append("potential")
                    index=np.argmin(np.sum((xyz-host(point)[0])**2,axis=1))
                    return session.upload(nuclear[index]*host(charge)[0])
                potential.compute_device=potentials
                state=SimpleNamespace(provider=provider,require_provider=Mock(),
                    locc_a=session.upload(local["A"]),locc_b=session.upload(local["B"]),
                    qa=session.upload(q["A"]),qb=session.upload(q["B"]),
                    matrix=session.empty((N+A+1,N+B+1),zero=True))
                upload=lambda fields:{n:session.upload(x) for n,x in fields.items()}
                result=induction.ind20u_device(state,potential,upload(occ),upload(vir),upload(energies),
                    upload(fields),session.upload(xyz),session.upload(za),session.upload(zb))
                for side,value,total in (("A",result.ab,result.total_ab),("B",result.ba,result.total_ba)):
                    np.testing.assert_allclose(host(value),expected[side],rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(host(total),[expected[side].sum()],rtol=1e-11,atol=1e-12)
                self.assertFalse(state.qa.allocation.freed or state.qb.allocation.freed)
                self.assertEqual(calls.count("DFMO"),1)
                self.assertEqual(calls.count("potential"),2*N)
            self.assertEqual(session.live_bytes,0)

    def test_projected_coulomb_and_expanded_exchange_targets_both_ko_orientations(self):
        for seed in (111,112):
            data,expected_jp,expected=target_fixture(seed)
            matrices,localized,virtual,density,factors=data
            session=Session(self.lib)
            session.download=lambda *_:self.fail("Projected JP/target intermediate host staging")
            with session.scope():
                jp=induction.projected_coulomb(session,session.upload(density),session.upload(factors))
                np.testing.assert_allclose(host(jp),expected_jp,rtol=1e-11,atol=1e-12)
                raw={n:session.upload(v) for n,v in matrices.items()}
                dl,dv=session.upload(localized),session.upload(virtual)
                for side in ("A","B"):
                    target=induction.exchange_induction_target(session,raw,dl,dv,jp,side=side)
                    np.testing.assert_allclose(host(target),expected[side],rtol=1e-11,atol=1e-12)
                wrong=dict(raw,K_O=session.upload(matrices["K_O"].T))
                bad=induction.exchange_induction_target(session,wrong,dl,dv,jp,side="B")
                self.assertGreater(np.max(np.abs(host(bad)-expected["B"])),1e-3)
                with self.assertRaises(ValueError):
                    induction.projected_coulomb(session,session.empty((4,2,3)),session.empty((4,2,3)))
            self.assertEqual(session.live_bytes,0)


if __name__=="__main__":unittest.main()
