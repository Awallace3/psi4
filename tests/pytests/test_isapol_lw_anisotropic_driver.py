# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Synthetic factory LW -> anisotropic chain, NOT molecular-frequency acceptance.

Explicit source loading against staged core; no installed-driver acceptance.
Analytic anchors are independent formulas, not adapter-generated expectations.
"""
from dataclasses import FrozenInstanceError
import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]
MODULE = Path(__file__).resolve().parents[2] / 'psi4/driver/procrouting/isapol_lw.py'
spec = importlib.util.spec_from_file_location('psi4.driver.procrouting._lw_anisotropic_test_source', MODULE)
lw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lw
spec.loader.exec_module(lw)
PROV = lw.Provenance('synthetic', '0'*64, 'analytic synthetic test', 'Not molecular/native acceptance')


def placement(t=(0.,0.,0.), r=None):
    return lw.Placement(np.eye(3) if r is None else r, t)


def factory(block=None, origins=((0.,0.,0.),), frequencies=(0.,1.), frames=None):
    n, nf = len(origins), len(frequencies)
    raw = np.zeros((nf,n,n,16,16))
    if block is None:
        block = np.diag(np.arange(1.,16.))
        block[0,3] = block[3,0] = .4
        block[3,8] = block[8,3] = -.2
        block[0,8] = block[8,0] = .1
    for f,xi in enumerate(frequencies):
        for s in range(n):
            raw[f,s,s,1:,1:] = block*(s+1)/(1+xi*xi)
    return lw.supplied_nonlocal_properties(labels=[f'S{s}' for s in range(n)], origins=origins,
        bonds=[], frequencies=list(frequencies), tensors=raw, input_rank=3, frames=frames, provenance=PROV)


def evaluate(a, b=None, **kw):
    args = dict(placement_a=placement(), placement_b=placement((0.,0.,2.)),
                cp_weights=[0.,.25], quadrature_provenance=PROV)
    args.update(kw)
    return lw.anisotropic_dispersion(a, a if b is None else b, **args)


def test_dipole_lorentz_normalization():
    grid = psi4.core.CasimirGrid(10,.5)
    freq = [grid.omega(i) for i in range(11)]
    weights = [grid.cp_weight(i) for i in range(11)]
    def lorentz(alpha):
        # factory denominator 1+xi^2; rescale nodes to make omega=.5.
        raw = np.zeros((11,1,1,16,16))
        for k,xi in enumerate(freq):
            raw[k,0,0,1:4,1:4] = np.eye(3)*alpha/(1+(xi/.5)**2)
        return lw.supplied_nonlocal_properties(labels=['D'], origins=[[0,0,0]], bonds=[],
            frequencies=freq, tensors=raw, input_rank=3, provenance=PROV)
    r = evaluate(lorentz(3), lorentz(5), cp_weights=weights)
    # Integral: C6 = 3/4 alphaA alphaB omega for equal Lorentz poles.
    assert r.pairs[0].coefficients[0].value == pytest.approx(.75*3*5*.5, rel=2e-7)
    assert all(c.value == 0 for c in r.pairs[0].coefficients[1:])
    assert r.truncated_energy == pytest.approx(-(.75*3*5*.5)/2**6, rel=2e-7)


@pytest.mark.parametrize('order', range(6,13))
def test_mixed_rank_odd_anchor_and_coverage(order):
    aa, bb = np.zeros((15,15)), np.zeros((15,15))
    aa[0,0], bb[0,0] = 2.,3.
    aa[0,3] = aa[3,0] = .4
    bb[0,3] = bb[3,0] = -.7
    aa[8,8], bb[8,8] = 1.2,.8  # nonzero rank3, no C7 effect
    r = evaluate(factory(aa), factory(bb), max_order=order)
    coeffs = r.pairs[0].coefficients
    assert [c.order for c in coeffs] == list(range(6,order+1))
    if order >= 7:
        # tau11=-2, tau12=+3, tau21=-3; two ordered mixed terms.
        # Each dynamic tensor is half its supplied static value, weight=.25.
        assert coeffs[1].value == pytest.approx(.25/4*12*(.4*3-2*(-.7)), abs=1e-14)
    for c in coeffs:
        expected = tuple(q for q in itertools.product(range(1,c.order-4), repeat=4) if sum(q)+2 == c.order)
        assert c.included_rank_quadruples == tuple(q for q in expected if max(q) <= 3)
        assert c.missing_rank_quadruples == tuple(q for q in expected if max(q) > 3)
        assert c.declared_model_complete
        assert c.unrestricted_complete == (c.order <= 8)
        assert c.energy == pytest.approx(-c.value/2**c.order, abs=1e-15)
    assert r.truncated_energy == pytest.approx(sum(c.energy for c in coeffs), abs=1e-14)


def test_all_pairs_placements_direct_core_and_global_not_roundtripped(monkeypatch):
    # Nontrivial source frames may produce roundoff asymmetry in raw_local.
    angle = .37
    f = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    a = factory(origins=((.2,-.1,.3),), frames=[f])
    b = factory(origins=((-1.,.2,0.),(.4,.6,-.2)), frames=[f,f])
    assert np.array_equal(a.raw_global.array, a.raw_global.array.transpose(0,1,3,2))
    pa = placement((.1,.2,-.5), f)
    pb = placement((.7,-.4,4.), [[1,0,0],[0,0,-1],[0,1,0]])
    # Do not depend on whether this BLAS introduces asymmetric rounding.
    # Observe access instead: conversion must never read either raw_local snapshot.
    original_array = lw.ArraySnapshot.array.fget
    def guarded_array(snapshot):
        if snapshot is a.raw_local or snapshot is b.raw_local:
            raise AssertionError('anisotropic conversion accessed raw_local')
        return original_array(snapshot)
    with monkeypatch.context() as patch:
        patch.setattr(lw.ArraySnapshot, 'array', property(guarded_array))
        r = evaluate(a,b,placement_a=pa,placement_b=pb)
    assert r.model_a is a and r.model_b is b
    assert r.placement_a is pa and r.placement_b is pb
    def direct(m,p,g):
        sites = []
        for j,label in enumerate(m.labels):
            s = psi4.core.IsaAnisotropicSite()
            s.label, s.ranks = label, [1,2,3]
            s.origin = (p.rotation.array@m.origins.array[j]+p.translation.array).tolist()
            s.frame = p.rotation.array.tolist()
            s.responses = [psi4.core.Matrix.from_array(t) for t in m.raw_global.array[:,j]]
            sites.append(s)
        np.testing.assert_allclose(g.origins.array, [s.origin for s in sites], atol=1e-15)
        np.testing.assert_array_equal(g.source_frames.array, p.rotation.array@m.frames.array)
        np.testing.assert_array_equal(g.component_frames.array, [p.rotation.array]*len(sites))
        assert m.provenance.source_sha256 in g.core_provenance
        return psi4.core.IsaAnisotropicModel(list(m.frequencies),sites,'supplied_local_response',g.core_provenance)
    d = psi4.core.isa_anisotropic_dispersion(direct(a,pa,r.placed_a),direct(b,pb,r.placed_b),[0,.25],12)
    assert len(r.pairs) == 2
    assert r.truncated_energy == d.truncated_energy
    for p,q in zip(r.pairs,d.pairs):
        assert p.label_a == a.labels[p.site_a] and p.label_b == b.labels[p.site_b]
        assert p.distance == q.distance and p.truncated_energy == q.truncated_energy
        assert p.displacement == tuple(q.displacement) and p.direction == tuple(q.direction)
        assert [c.value for c in p.coefficients] == [c.value for c in q.coefficients]
    assert 'not_computed' in a.metadata.anisotropic_status


@pytest.mark.parametrize('defect', [1e-8, np.spacing(1.)])
def test_strict_asymmetric_factory_result_rejected(defect):
    block = np.eye(15); block[0,1] = defect
    a = factory(block)
    before = a.raw_global.data
    with pytest.raises(ValueError, match='exact reciprocity'):
        evaluate(a)
    assert a.raw_global.data == before


def test_snapshot_placement_bound_before_copy(monkeypatch):
    oversized = lw.ArraySnapshot.of(np.zeros((4,3)))
    def forbidden(_):
        raise AssertionError('invalid snapshot must be rejected before copying')
    monkeypatch.setattr(lw.ArraySnapshot, 'array', property(forbidden))
    with pytest.raises(ValueError, match='bounded shape'):
        lw.Placement(oversized, [0,0,0])


def test_cartesian_anisotropic_dipole_anchor():
    aa, bb = np.zeros((15,15)), np.zeros((15,15))
    aa[:3,:3] = [[2,.7,-.4],[.7,-1,.2],[-.4,.2,3]]
    bb[:3,:3] = [[1,-.3,.8],[-.3,4,-.6],[.8,-.6,2]]
    xyz = np.array([.7,-.9,1.3])
    radius = np.linalg.norm(xyz)
    u = xyz[[2,0,1]]/radius  # Racah dipoles z,x,y
    tau = np.eye(3)-3*np.outer(u,u)
    r = evaluate(factory(aa),factory(bb),placement_b=placement(xyz))
    expected = .25/4*np.trace(aa[:3,:3]@tau@bb[:3,:3]@tau.T)
    assert r.pairs[0].coefficients[0].value == pytest.approx(expected, abs=1e-14)
    assert r.truncated_energy == pytest.approx(-expected/radius**6, abs=1e-14)


def test_immutable_ownership():
    rarray, tarray, weights = np.eye(3), np.array([0.,0.,2.]), [0,.25]
    p = lw.Placement(rarray,tarray)
    r = evaluate(factory(),placement_b=p,cp_weights=weights)
    rarray.fill(9); tarray.fill(9); weights[1] = 9
    assert r.cp_weights == (0.,.25)
    np.testing.assert_array_equal(p.rotation.array,np.eye(3))
    np.testing.assert_array_equal(p.translation.array,[0,0,2])
    for s in (p.rotation,p.translation,r.placed_b.origins,r.placed_b.source_frames,r.placed_b.component_frames):
        before = s.data; s.array.fill(99); assert s.data == before
    with pytest.raises(FrozenInstanceError): r.truncated_energy = 3
    with pytest.raises(FrozenInstanceError): r.pairs[0].coefficients[0].value = 3
    with pytest.raises(TypeError): r.placed_a.origins.data[0] = 3


@pytest.mark.parametrize('rotation', [True, np.ones((4,4)), np.eye(3)*2, np.diag([-1,1,1]),
    [[1,.1,0],[0,1,0],[0,0,1]], [[True,0,0],[0,1,0],[0,0,1]], np.full((3,3),np.nan),
    np.full((3,3),np.inf), np.eye(3,dtype=complex), np.full((3,3),1e308)])
def test_invalid_rotation(rotation):
    with pytest.raises(ValueError): lw.Placement(rotation,[0,0,0])


@pytest.mark.parametrize('translation', [[True,0,0], [0,0], [0,0,0,0], [0,0,np.inf],
    [0,0,np.nan], ['0',0,0], iter([0,0,0]), np.zeros((100000,3))])
def test_invalid_translation(translation):
    with pytest.raises(ValueError): lw.Placement(np.eye(3),translation)


@pytest.mark.parametrize('order', [True,False,6.,np.int64(6),5,13,None,'7'])
def test_invalid_orders(order):
    with pytest.raises(ValueError): evaluate(factory(),max_order=order)


@pytest.mark.parametrize('weights', [[0,0],[1,1],[0,-1],[0,np.nan],[0,np.inf],[0,True],
    [0,1j],[0,'1'],[1],[[0,1]],iter([0,1])])
def test_invalid_weights(weights):
    with pytest.raises(ValueError): evaluate(factory(),cp_weights=weights)


def test_grid_types_static_coincidence_and_nonfinite_geometry():
    a = factory()
    with pytest.raises(ValueError,match='grids'): evaluate(a,factory(frequencies=(0.,2.)))
    with pytest.raises(ValueError): evaluate(a,placement_a=None)
    with pytest.raises(ValueError): evaluate(a,placement_b=np.eye(3))
    with pytest.raises(ValueError): evaluate(a,quadrature_provenance={})
    with pytest.raises(ValueError): evaluate(object())
    with pytest.raises(TypeError): lw.anisotropic_dispersion(a,a,cp_weights=[0,1],quadrature_provenance=PROV)
    with pytest.raises(ValueError,match='Coincident'): evaluate(a,placement_b=placement())
    static = factory(frequencies=(0.,))
    for w in ([0],[1]):
        with pytest.raises(ValueError): evaluate(static,cp_weights=w)
    # Finite placements whose A/B displacement overflows must fail in core.
    with pytest.raises((FloatingPointError,ValueError,RuntimeError)):
        evaluate(a,placement_a=placement((1e308,0,0)),placement_b=placement((-1e308,0,0)))
