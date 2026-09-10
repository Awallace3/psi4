# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Source-loaded orchestration: synthetic contracts plus real native He full chain.

He is an explicit compact Gaussian model, NOT the requested water endpoint and
not a physical basis-limit claim. No reference properties or skipped stages.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native_partition as p

PATH = Path(__file__).resolve().parents[2]/'psi4/driver/procrouting/isapol_native.py'
SPEC = importlib.util.spec_from_file_location('psi4.driver.procrouting._test_native_pipeline', PATH)
n = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = n; SPEC.loader.exec_module(n)


def recipe(wfn):
    m = wfn.molecule(); c = (m.x(0),m.y(0),m.z(0))
    def s(a): return p.ShellRecipe(0,0,(a,),((2*a/np.pi)**.75,))
    auxiliary = p.BasisRecipe('explicit compact s/p/d/f AUX', 'test authored normalized s and raw angular primitives',
        'Cartesian',(c,),tuple(s(a) for a in (.25,.5,1.,2.,4.,8.,16.,32.))+
        tuple(p.ShellRecipe(0,l,(a,),(1.,)) for l in (1,2,3) for a in (.4,1.2)))
    atomic = p.BasisRecipe('distinct s AtomAux','test authored primitive recipe','Spherical',(c,),
                          tuple(s(a) for a in (.3,.7,1.5,3.5,9.,24.)))
    site = p.SiteRecipe('He',c,atomic,replace(atomic,name='Shape'),tuple(range(6)),3,1.5,True)
    return p.PartitionRecipe('He native smoke','test numeric basis definitions only','explicit_cartesian_drho_c_isa_a',
        auxiliary,(site,),p.GridRecipe(100,110,3,1.,'native_tabulated_bragg_slater','all_sites_unscreened_full_molecular_grid'),
        p.ControllerRecipe(1e-9,120,.17,.001,.2,True,0.,True,1e-36,1e-5,1e-5,1e-5,0.,20,20,True),'Drho1e-2')


def run(wfn,r=None,**kw):
    options=dict(bonds=(),frames=None,caller_converged=True,kernel='no_local',exact_exchange=1.,local_scale=0.,response_grid=None)
    options.update(kw)
    return n.native_properties(wfn,recipe(wfn) if r is None else r,**options)


@pytest.fixture(scope='module')
def helium():
    core.be_quiet()
    psi4.basis_helper('assign pipeline_he\n[pipeline_he]\nspherical\n****\nHe 0\n'
        'S 1 1.0\n1.7 1.0\nP 1 1.0\n.7 1.0\nD 1 1.0\n.8 1.0\nF 1 1.0\n.9 1.0\n****\n',name='NATIVE_PIPELINE_HE')
    mol=psi4.geometry('0 1\nHe .17 -.23 .31\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n')
    psi4.set_options({'basis':'NATIVE_PIPELINE_HE','puream':True,'reference':'rhf','scf_type':'pk',
                      'e_convergence':1e-12,'d_convergence':1e-12})
    _,wfn=psi4.energy('hf',molecule=mol,return_wfn=True)
    return wfn


@pytest.fixture(scope='module')
def full(helium):
    quad=n.Quadrature.from_casimir(core.CasimirGrid(10,.5))
    return run(helium,frequencies=quad.frequencies,quadrature=quad,pair_self=True)


def test_actual_native_all_stages(helium,full):
    assert not full.failures, full.failures
    assert full.partition.converged and full.dispersion is not None
    assert full.ov_fit.charge_penalty == 1.
    assert len(full.coefficient_responses)==len(full.frequencies)==11
    assert full.pair_tensors.shape==(11,1,1,16,16)
    assert full.local.metadata.production_postcondition_passed
    assert full.local.metadata.residual_tolerance==1e-6
    assert full.atomic_scalars.shape==(11,1,3)
    assert np.all(full.atomic_scalars.array>0)
    np.testing.assert_allclose(full.full_adapted_orbitals.array,
        full.partition.main.transform @ np.asarray(helium.Ca()),rtol=0,atol=0)
    d=np.asarray(full.ov_fit.coefficients)
    provider=full.context.response.provider
    h1,h2=np.asarray(provider.h1()),np.asarray(provider.h2())
    for xi,r in zip(full.frequencies,full.coefficient_responses):
        control=d.T @ np.linalg.solve(h2@h1+xi*xi*np.eye(len(d)),-4*h2@d)
        np.testing.assert_allclose(r.raw_coupled,control,atol=2e-12,rtol=2e-12)
    assert full.comparison_status=='not_measured_no_reference'


def test_actual_independent_C6_and_rank_coverage(full):
    assert not full.failures,full.failures
    a=full.atomic_scalars.array[:,0,0]
    c=full.dispersion.pairs[0].coefficients
    assert c[0].value==pytest.approx(6*np.dot(full.quadrature.cp_weights,a*a),rel=2e-14)
    assert [v.unrestricted_complete for v in c]==[True,True,True,False]
    assert c[-1].missing_rank_pairs==((1,4),(4,1))


def test_static_no_scf_no_weights_and_ownership(helium,full,monkeypatch):
    def no_scf(*a,**k): raise AssertionError('implicit SCF forbidden')
    monkeypatch.setattr(psi4,'energy',no_scf)
    r=run(helium,response_context=full.context)
    assert not r.failures,r.failures
    assert r.quadrature is None and r.dispersion is None and r.frequencies==(0.,)
    data=r.atomic_scalars.array; data[:]=123
    assert not np.any(r.atomic_scalars.array==123)
    provider=r.context.response.provider
    c=provider.orbitals(); c.np[:]=0
    assert np.any(np.asarray(provider.orbitals())!=0)
    r.partition.q.values.np[:]=0
    assert np.any(r.pair_tensors.array!=0)


@pytest.mark.parametrize('nodes',[(0.,.2,.1),(0.,0.),(-1.,),(float('nan'),),(),(0j,)])
def test_bad_nodes_rejected_before_partition(nodes):
    with pytest.raises(ValueError): n._frequencies(nodes)


def test_quadrature_complete_and_prefactor_once():
    g=core.CasimirGrid(10,.5); q=n.Quadrature.from_casimir(g)
    assert q.cp_weights==tuple(g.cp_weight(i) for i in range(11))
    assert q.frequencies==tuple(g.omega(i) for i in range(11))
    with pytest.raises(ValueError): replace(q,cp_weights=(0.,)*11)
    with pytest.raises(ValueError): replace(q,cp_weights=(1.,)+q.cp_weights[1:])
    with pytest.raises(ValueError): replace(q,cp_weights=q.cp_weights[:-1]+(0.,))


def test_missing_nodes_and_partner_rejected(helium,full):
    q=full.quadrature
    with pytest.raises(ValueError,match='complete authoritative'):
        run(helium,quadrature=q,frequencies=q.frequencies[:-1],pair_self=True)
    with pytest.raises(ValueError,match='quadrature'):
        run(helium,pair_self=True)
    with pytest.raises(ValueError,match='partner OR'):
        run(helium,quadrature=q,frequencies=q.frequencies,pair_self=True,partner=full)


def test_actual_explicit_partner(helium,full):
    assert not full.failures,full.failures
    q=full.quadrature
    r=run(helium,quadrature=q,frequencies=q.frequencies,partner=full,response_context=full.context)
    assert not r.failures,r.failures
    assert r.dispersion.pairs==full.dispersion.pairs


def test_context_policy_and_actual_energy_mutation_rejected(helium,full):
    with pytest.raises(ValueError,match='context mismatch'):
        run(helium,response_context=full.context,exact_exchange=0.)
    eps=helium.epsilon_a().np; before=eps.copy()
    try:
        eps[-1]+=.01
        with pytest.raises(ValueError,match='context mismatch'):
            run(helium,response_context=full.context)
    finally: eps[:]=before


def test_mutation_during_partition_retains_failure(helium,monkeypatch):
    original=n.native_partition; eps=helium.epsilon_a().np; before=eps.copy()
    def changed(*a,**kw):
        r=original(*a,**kw); eps[-1]+=.01; return r
    monkeypatch.setattr(n,'native_partition',changed)
    try:
        r=run(helium)
        assert r.local is None and r.partition.converged
        assert r.failures[0].stage=='context'
        assert 'changed' in r.failures[0].message
    finally: eps[:]=before


def test_LW_failure_retains_every_node_no_fake_atoms(helium,full,monkeypatch):
    calls=[]
    def reject(**kw):
        calls.append(kw['frequencies'][0])
        assert kw['residual_policy']=='production'
        raise RuntimeError('inspectable charge_sum=0.01')
    monkeypatch.setattr(n.lw,'supplied_nonlocal_properties',reject)
    q=full.quadrature
    r=run(helium,frequencies=q.frequencies,quadrature=q,pair_self=True,response_context=full.context)
    assert tuple(calls)==q.frequencies
    assert len(r.failures)==len(r.coefficient_responses)==11
    assert r.local is None and r.dispersion is None and r.distributed is not None
    assert r.pair_tensors.shape[0]==11
    with pytest.raises(RuntimeError,match='charge_sum'): _=r.atomic_scalars


def test_actual_partition_nonconvergence_retained(helium):
    a=recipe(helium); a=replace(a,controller=replace(a.controller,max_iterations=1))
    r=run(helium,a)
    assert not r.partition.converged and r.partition.q is None
    assert r.context is None and r.local is None
    assert r.failures[0].stage=='partition'


def test_synthetic_LW_dispersion_no_anisotropic_conversion():
    q=n.Quadrature.from_casimir(core.CasimirGrid(10,.5))
    raw=np.zeros((11,1,1,16,16))
    alpha=2/(1+np.array(q.frequencies)**2)
    for k,a in enumerate(alpha): raw[k,0,0,1:,1:]=np.eye(15)*a
    prov=n.lw.Provenance('synthetic tensors','0'*64,'test','analytic Lorentz model, not native water')
    local=n.lw.supplied_nonlocal_properties(labels=('X',),origins=((0.,0.,0.),),bonds=(),frames=None,
        frequencies=q.frequencies,tensors=raw,input_rank=3,provenance=prov)
    disp=n.lw.isotropic_dispersion(local,local,cp_weights=q.cp_weights,quadrature_provenance=q.provenance)
    assert disp.pairs[0].coefficients[0].value==pytest.approx(6*np.dot(q.cp_weights,alpha**2),rel=2e-14)


@pytest.mark.parametrize('penalty',[1,0,0.,-0.,-1.,float('nan'),float('inf'),'1e4',None,(1.,),1+0j,
                                    np.float64(1e4)])
def test_bad_ov_charge_penalty_rejected_before_any_fit(helium,penalty):
    with pytest.raises(ValueError,match='ov_charge_penalty must be an explicit finite positive float'):
        run(helium,ov_charge_penalty=penalty)


def test_direct_ov_forms_no_transition_fit_so_declares_no_penalty(helium,full):
    with pytest.raises(ValueError,match='no charge penalty applies'):
        run(helium,response_basis='direct_ov',ov_charge_penalty=1.e4)
    r=run(helium,response_basis='direct_ov',response_context=full.context)
    assert r.ov_fit is None and 'direct_ov' in r.model and 'lambda' not in r.model


def test_actual_penalty_recorded_and_one_context_serves_every_lambda(helium,full):
    """H1/H2 come from the orbitals, so the penalty is deliberately outside the
    response policy hash: the same native context is reusable at every lambda."""
    r=run(helium,frequencies=full.frequencies,quadrature=full.quadrature,pair_self=True,
          response_context=full.context,ov_charge_penalty=1.e4)
    assert not r.failures,r.failures
    assert r.context is full.context
    assert full.ov_fit.charge_penalty==1. and 'fitted_auxiliary lambda=1.0;' in full.model
    assert r.ov_fit.charge_penalty==1.e4 and 'fitted_auxiliary lambda=10000.0;' in r.model
    assert 'lambda=10000.0' in r.ov_fit.provenance and 'lambda=1.0' in full.ov_fit.provenance


def test_actual_penalty_changes_the_declared_model_and_costs_conditioning(helium,full):
    """He already satisfies the constraint at lambda1, so raising it only costs.

    The fitted transition charge is exactly zero for an OV transition density by
    MO orthonormality, and on this one-site AUX it is already at machine level at
    the archived lambda1 -- so nothing here is repaired by a larger penalty,
    while the conditioning of ``A = J + lambda q q^T`` degrades linearly in
    lambda and moves the atomic tensors.  That movement is the reason a chain run
    at another lambda is a differently declared model and must never be compared
    against a recorded lambda1 number; it is measured, not assumed.
    """
    assert full.diagnostics['fitted_transition_charge_maxabs'] < 1e-13
    base=full.atomic_scalars.array
    moved=[]
    for lam in (1.e2,1.e6):
        r=run(helium,frequencies=full.frequencies,quadrature=full.quadrature,pair_self=True,
              response_context=full.context,ov_charge_penalty=lam)
        assert not r.failures,r.failures
        assert r.local.metadata.production_postcondition_passed
        moved.append(float(np.max(np.abs(r.atomic_scalars.array-base))))
    assert 0. < moved[0] < 1e-11 and moved[0]*100 < moved[1] < 1e-6
