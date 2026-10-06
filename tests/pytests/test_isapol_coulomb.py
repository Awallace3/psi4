"""Analytic native Libint2 AUX checks, the analytic Drho-C fit, and exact-residual Drho-C refinement against rational, Hilbert and frozen-water references; not end-to-end acceptance."""
from fractions import Fraction
from math import erf, exp, pi, sqrt
import hashlib
import itertools
import json
import os
import platform
import struct
import subprocess
import sys
import numpy as np
import pytest
from psi4 import core
pytestmark=[pytest.mark.psi,pytest.mark.api,pytest.mark.quick]


def basis(shells, centres, role=None, representation=None):
    data=[]
    for site,l,exponents,coefficients in shells:
        s=core.IsaGaussianShell()
        s.centre,s.l,s.exponents,s.coefficients=site,l,exponents,coefficients
        data.append(s)
    return core.IsaExplicitBasis(role if role is not None else core.IsaBasisRole.MolecularAux,
        representation if representation is not None else core.IsaBasisRepresentation.Cartesian,centres,data)


def boys0(t):
    return 1. if t==0 else sqrt(pi)*erf(sqrt(t))/(2*sqrt(t))


@pytest.mark.parametrize("representation,width", [
    (core.IsaBasisRepresentation.Cartesian, 6),
    (core.IsaBasisRepresentation.Spherical, 5)])
def test_native_shell_layout_preserves_centres_and_representation(representation, width):
    auxiliary = basis([(1, 2, [.7], [.5]), (0, 0, [.8], [.6]),
                       (1, 1, [.9], [.4])], [[0., 0., 0.], [.2, -.3, .4]],
                      representation=representation)
    expected = [[0, width, 1, 2], [width, 1, 0, 0], [width+1, 3, 1, 1]]
    layout = auxiliary.shell_layout()
    assert layout == expected
    layout[0][2] = 0
    assert auxiliary.shell_layout() == expected


@pytest.mark.parametrize("representation", [
    core.IsaBasisRepresentation.Cartesian, core.IsaBasisRepresentation.Spherical])
def test_alda_shell_screen_uses_effective_coefficients_and_signed_sum(representation):
    aux = basis([(0, 4, [1.], [2.]), (1, 2, [1.], [-.5]),
                 (0, 1, [1., 1.], [1., -1.])],
                [[0., 0., 0.], [2., 0., 0.]], representation=representation)
    # Equal exponents: normalized-s overlap is exp(-R²/2). The screen
    # deliberately uses stored effective coefficients, not unit shell norms.
    expected = np.array([[4., -exp(-2.), 0.],
                         [-exp(-2.), .25, 0.], [0., 0., 0.]])
    screened = aux.screening_s_overlap(max_bytes=9*8)
    np.testing.assert_allclose(screened, expected, atol=1e-15, rtol=1e-15)
    np.asarray(screened)[:] = 0.
    np.testing.assert_allclose(aux.screening_s_overlap(), expected, atol=1e-15, rtol=1e-15)
    with pytest.raises(ValueError, match="byte resource"):
        aux.screening_s_overlap(max_bytes=9*8-1)


@pytest.mark.parametrize("representation", [
    core.IsaBasisRepresentation.Cartesian, core.IsaBasisRepresentation.Spherical])
def test_three_center_shell_blocks_reassemble_full_integrals(representation):
    centres = [[0., 0., 0.], [.2, -.3, .4]]
    auxiliary = basis([(0, 0, [.8], [.6]), (1, 2, [.7], [.5]),
                       (0, 1, [.9], [.4])], centres, representation=representation)
    main = basis([(0, 0, [1.1], [.7]), (1, 1, [.6], [.3])], centres,
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    provider = core.IsaAuxCoulomb(auxiliary)
    whole = np.asarray(provider.three_center(main))
    chunks = [np.asarray(provider.three_center_shell_block(main, i, 1)) for i in range(3)]
    np.testing.assert_array_equal(np.concatenate(chunks), whole)
    np.testing.assert_array_equal(provider.three_center_shell_block(main, 1, 2),
                                   np.concatenate(chunks[1:]))


@pytest.mark.parametrize("representation", [
    core.IsaBasisRepresentation.Cartesian, core.IsaBasisRepresentation.Spherical])
def test_mo_shell_blocks_match_full_native_integral_transformation(representation):
    from psi4.driver.procrouting.isapol_factorized_response import mo_three_center_shell
    centres = [[0., 0., 0.], [.2, -.3, .4]]
    auxiliary = basis([(0, 0, [.8], [.6]), (1, 4, [.7], [.5]),
                       (0, 1, [.9], [.4])], centres, representation=representation)
    main = basis([(0, 0, [1.1], [.7]), (1, 1, [.6], [.3])], centres,
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    provider = core.IsaAuxCoulomb(auxiliary)
    # Rectangular and noncontiguous; coefficients are explicitly in MAIN order.
    coefficients = np.array([[1., .2, -.1], [.3, .8, .2],
                             [0., -.4, .7], [.1, .2, .3]])[:, ::-1]
    ao = np.asarray(provider.three_center(main)).reshape(-1, 4, 4)
    expected = np.einsum("mp,Pmn,nq->Ppq", coefficients, ao, coefficients)
    blocks = [mo_three_center_shell(provider, main, coefficients, i) for i in range(3)]
    np.testing.assert_allclose(np.concatenate(blocks), expected, atol=2e-13, rtol=2e-13)
    coefficients[:] = 0.
    np.testing.assert_allclose(np.concatenate(blocks), expected, atol=2e-13, rtol=2e-13)


def test_mo_shell_transform_resource_and_input_contracts():
    from psi4.driver.procrouting.isapol_factorized_response import mo_three_center_shell
    aux = basis([(0, 0, [.8], [.6])], [[0., 0., 0.]])
    main = basis([(0, 0, [1.1], [.7])], [[.2, 0., 0.]],
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    provider = core.IsaAuxCoulomb(aux)
    coefficients = np.ones((1, 1))
    # The documented conservative ledger reserves 24 AO, 18 MO and
    # four coefficient/workspace doubles even for a one-function shell.
    admitted_bytes = 46*8
    result = mo_three_center_shell(provider, main, coefficients, 0,
                                   max_bytes=admitted_bytes)
    np.testing.assert_array_equal(result.ravel(), provider.three_center(main).np.ravel())
    with pytest.raises(ValueError, match="byte resource"):
        mo_three_center_shell(provider, main, coefficients, 0, max_bytes=admitted_bytes-1)
    for index in (-1, True):
        with pytest.raises(ValueError, match="shell_index"):
            mo_three_center_shell(provider, main, coefficients, index)
    with pytest.raises(ValueError, match="range"):
        mo_three_center_shell(provider, main, coefficients, 1)
    for bad in (np.ones((2, 1)), np.ones((1, 0)), np.ones(1),
                np.ones((1, 1), dtype=np.float32), np.full((1, 1), np.nan)):
        with pytest.raises(ValueError, match="coefficients"):
            mo_three_center_shell(provider, main, bad, 0)
    with pytest.raises(ValueError, match="Orbital"):
        mo_three_center_shell(provider, aux, coefficients, 0)
    for budget in (0, True):
        with pytest.raises(ValueError, match="max_bytes"):
            mo_three_center_shell(provider, main, coefficients, 0, max_bytes=budget)
    result[:] = 0.
    assert mo_three_center_shell(provider, main, coefficients, 0)[0, 0, 0] > 0.


def test_mo_shell_budget_includes_native_spherical_accumulation():
    from psi4.driver.procrouting.isapol_factorized_response import mo_three_center_shell
    aux = basis([(0, 4, [.8], [.6])], [[0., 0., 0.]],
                representation=core.IsaBasisRepresentation.Spherical)
    main = basis([(0, 4, [1.1], [.7])], [[.2, 0., 0.]],
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    # Old ledger admits 10800 bytes. Native AO output plus spherical
    # accumulation alone need 2*9*9*9*8 = 11664, before MO work.
    with pytest.raises(ValueError, match="byte resource"):
        mo_three_center_shell(core.IsaAuxCoulomb(aux), main, np.ones((9, 1)),
                              0, max_bytes=10800)


def _factors(ops):
    """Owned factor record: gaps, OO/OV and the gathered (naux, columns) dual OV/VV tiles."""
    return (ops._gaps, ops._oo, ops._ov, np.column_stack(ops._dual_ov), np.column_stack(ops._dual_vv))


def _factor_h1_h2(ops, exchange):
    """Dense H1/H2 contracted directly from the record factors; rows a*nocc+i."""
    gaps, oo, ov, dual_ov, dual_vv = _factors(ops)
    na, no, nv = ov.shape
    dual_ov, dual_vv = dual_ov.reshape(na, no, nv), dual_vv.reshape(na, nv, nv)
    v = np.einsum("pia,pjb->aibj", ov, dual_ov)
    x = np.einsum("pij,pab->aibj", oo, dual_vv)
    y = np.einsum("pib,pja->aibj", ov, dual_ov)
    shape = (no*nv, no*nv)
    h1 = np.diag(gaps)+(4*v-exchange*(x+y)).reshape(shape)
    return h1, np.diag(gaps)-exchange*(x-y).reshape(shape)


@pytest.mark.parametrize("representation", [
    core.IsaBasisRepresentation.Cartesian, core.IsaBasisRepresentation.Spherical])
@pytest.mark.parametrize("exchange", [0., .25, 1.])
def test_streamed_plain_df_operators_match_full_native_contractions(representation, exchange):
    from psi4.driver.procrouting.isapol_native_factors import native_plain_df_operators
    aux = basis([(0, 0, [.8], [.6]), (0, 2, [.7], [.5])], [[0., 0., 0.]],
                representation=representation)
    main = basis([(0, 0, [1.1], [.7]), (0, 2, [.6], [.3])], [[.2, 0., 0.]],
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    coefficients = np.random.default_rng(75).normal(scale=.1, size=(6, 5))[:, ::-1]
    energies = np.array([-1., -.8, .2, .6, 1.])
    provider = core.IsaAuxCoulomb(aux)
    na = aux.nfunction
    ao = np.asarray(provider.three_center(main)).reshape(na, 6, 6)
    mo = np.einsum("mp,Pmn,nq->Ppq", coefficients, ao, coefficients)
    dual = np.linalg.solve(provider.metric().np, mo.reshape(na, -1)).reshape(mo.shape)
    gaps = (energies[2:, None]-energies[None, :2]).ravel()
    h1, h2 = np.diag(gaps), np.diag(gaps)
    for a, i, b, j in itertools.product(range(3), range(2), range(3), range(2)):
        v = sum(mo[p, i, 2+a]*dual[p, j, 2+b] for p in range(na))
        x = sum(mo[p, i, j]*dual[p, 2+a, 2+b] for p in range(na))
        y = sum(mo[p, i, 2+b]*dual[p, j, 2+a] for p in range(na))
        h1[2*a+i, 2*b+j] += 4*v-exchange*(x+y)
        h2[2*a+i, 2*b+j] -= exchange*(x-y)
    # Neither six OV nor nine VV columns is divisible by four.
    streamed = native_plain_df_operators(
        aux, main, coefficients, energies, nocc=2, shell_count=2, tile_columns=4)
    expected_density = 2*sum(dual[:, i, i] for i in range(2))
    np.testing.assert_allclose(streamed.plain_density_coefficients, expected_density,
                               atol=2e-12, rtol=2e-12)
    with pytest.raises(ValueError):
        streamed.plain_density_coefficients.setflags(write=True)
    coefficients[:] = 0.
    energies[:] = 0.
    got1, got2 = _factor_h1_h2(streamed, exchange)
    np.testing.assert_allclose(got1, h1, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(got2, h2, atol=2e-12, rtol=2e-12)


def test_streamed_plain_df_admission_and_shell_coverage():
    from psi4.driver.procrouting.isapol_native_factors import native_plain_df_operators
    aux = basis([(0, 0, [.8], [.6]), (0, 1, [.7], [.5])], [[0., 0., 0.]])
    main = basis([(0, 0, [1.1], [.7]), (0, 1, [.6], [.3])], [[.2, 0., 0.]],
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    coeff = np.eye(4)
    energies = np.array([-1., -.8, .2, .6])
    kwargs = dict(nocc=2, shell_count=2, tile_columns=1)
    ops = native_plain_df_operators(aux, main, coeff, energies, **kwargs)
    budget = ops.construction_planned_bytes
    exact = native_plain_df_operators(aux, main, coeff, energies, **kwargs, max_bytes=budget)
    for got, want in zip(_factors(exact), _factors(ops)):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="byte resource"):
        native_plain_df_operators(aux, main, coeff, energies, **kwargs, max_bytes=budget-1)
    for change, message in [
        (dict(shell_count=1), "coverage"), (dict(shell_count=3), "range"),
        (dict(nocc=True), "nocc"), (dict(nocc=4), "nocc"),
        (dict(tile_columns=0), "tile_columns"), (dict(tile_columns=709), "tile_columns"),
        (dict(max_bytes=True), "max_bytes"),
    ]:
        with pytest.raises(ValueError, match=message):
            native_plain_df_operators(aux, main, coeff, energies, **(kwargs | change))
    for bad in (np.full(4, np.nan), np.zeros(4)):
        with pytest.raises(ValueError, match="finite|gaps"):
            native_plain_df_operators(aux, main, coeff, bad, **kwargs)
    with pytest.raises(ValueError, match="finite"):
        native_plain_df_operators(aux, main, coeff*np.nan, energies, **kwargs)


@pytest.mark.parametrize("representation", [
    core.IsaBasisRepresentation.Cartesian, core.IsaBasisRepresentation.Spherical])
@pytest.mark.parametrize("penalty,damping", [(0., 0.), (1000., 0.), (1000., .0005)])
def test_tiled_constrained_ov_matches_dense_fit(representation, penalty, damping):
    from psi4.driver.procrouting.isapol_native_factors import (
        native_plain_df_operators, native_constrained_ov)
    centres = [[0., 0., 0.], [.6, -.4, .3]]
    # Deliberately interleave owning centres; the d-shell width depends on
    # representation and must not come from a separate caller-provided map.
    aux = basis([(1, 2, [.7], [.5]), (0, 0, [.8], [.6]), (1, 0, [.9], [.4])],
                centres, representation=representation)
    main = basis([(0, 0, [1.1], [.7]), (1, 2, [.6], [.3])], centres,
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    coefficients = np.random.default_rng(38).normal(scale=.1, size=(6, 5))[:, ::-1]
    ops = native_plain_df_operators(
        aux, main, coefficients, np.array([-1., -.8, .2, .6, 1.]),
        nocc=2, shell_count=3, tile_columns=4)
    # Dense reference: A = (1-eta offsite) J + lambda q q^T, occupied-fast RHS.
    integrals = core.IsaAuxCoulomb(aux)
    q = np.asarray(integrals.charges())
    centre = np.concatenate([[c]*w for _, w, c, _ in aux.shell_layout()])
    metric = np.where(centre[:, None] == centre[None, :], 1., 1.-damping)*integrals.metric().np
    metric += penalty*np.outer(q, q)
    b = integrals.three_center(main).np.reshape(len(q), 6, 6)
    occ, vir = coefficients[:, :2], coefficients[:, 2:]
    rhs = np.array([(occ.T @ block @ vir).T.reshape(-1) for block in b]).T
    expected = np.linalg.solve(metric, rhs.T).T
    fitted = native_constrained_ov(ops, charge_penalty=penalty,
                                   offsite_metric_damping=damping, tile_columns=4)
    np.testing.assert_allclose(fitted.coefficients, expected,
                               atol=2e-11, rtol=2e-10)
    assert fitted.relative_backward_residual <= 1.e-10
    assert fitted.charge_penalty == penalty
    assert fitted.offsite_metric_damping == damping
    with pytest.raises(ValueError):
        fitted.coefficients.setflags(write=True)
    if penalty == 0.:
        exact = native_constrained_ov(
            ops, charge_penalty=penalty, offsite_metric_damping=damping,
            tile_columns=4, max_bytes=fitted.planned_bytes)
        np.testing.assert_array_equal(exact.coefficients, fitted.coefficients)
        with pytest.raises(ValueError, match="byte resource"):
            native_constrained_ov(
                ops, charge_penalty=penalty, offsite_metric_damping=damping,
                tile_columns=4, max_bytes=fitted.planned_bytes-1)
        defaults = dict(charge_penalty=penalty, offsite_metric_damping=damping)
        for change, message in [
            (dict(charge_penalty=-1.), "charge_penalty"),
            (dict(charge_penalty=np.nan), "charge_penalty"),
            (dict(offsite_metric_damping=-.1), "offsite_metric_damping"),
            (dict(offsite_metric_damping=1.), "offsite_metric_damping"),
            (dict(tile_columns=True), "tile_columns"),
            (dict(tile_columns=709), "tile_columns"),
            (dict(max_bytes=True), "max_bytes"),
        ]:
            with pytest.raises(ValueError, match=message):
                native_constrained_ov(ops, **(defaults | change))


def test_constrained_ov_rejects_arbitrary_non_native_factors():
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    from psi4.driver.procrouting.isapol_native_factors import native_constrained_ov
    ops = FactorizedDFOperators()
    with pytest.raises(ValueError, match="native plain-DF"):
        native_constrained_ov(ops, charge_penalty=1000., offsite_metric_damping=0.)


def test_three_center_shell_block_admission_and_owned_output():
    aux = basis([(0, 0, [.8], [.6]), (0, 2, [.7], [.5])], [[0., 0., 0.]])
    main = basis([(0, 0, [1.1], [.7])], [[.2, 0., 0.]],
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    provider = core.IsaAuxCoulomb(aux)
    # Six Cartesian d rows, one MAIN pair, eight bytes each.
    expected = np.asarray(provider.three_center_shell_block(main, 1, 1, 48)).copy()
    with pytest.raises(ValueError, match="byte resource"):
        provider.three_center_shell_block(main, 1, 1, 47)
    for start, count in [(0, 0), (2, 1), (1, 2)]:
        with pytest.raises(ValueError, match="range"):
            provider.three_center_shell_block(main, start, count)
    with pytest.raises((TypeError, OverflowError)):
        provider.three_center_shell_block(main, -1, 1)
    with pytest.raises(ValueError, match="Orbital"):
        provider.three_center_shell_block(aux, 0, 1)
    value = provider.three_center_shell_block(main, 1, 1)
    np.asarray(value)[:] = 0.
    np.testing.assert_array_equal(provider.three_center_shell_block(main, 1, 1), expected)


def test_three_center_shell_block_rejects_excessive_aux_rows():
    # Admission only: no expensive integrals are evaluated.
    aux = basis([(0, 0, [1.+i/1000], [1.]) for i in range(513)], [[0., 0., 0.]])
    main = basis([(0, 0, [1.], [1.])], [[.2, 0., 0.]],
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    with pytest.raises(ValueError, match="maximum 512 functions"):
        core.IsaAuxCoulomb(aux).three_center_shell_block(main, 0, 513)


def ss(a,b,A,B):
    t=a*b/(a+b)*sum((x-y)**2 for x,y in zip(A,B))
    return 2*pi**2.5/(a*b*sqrt(a+b))*boys0(t)


@pytest.mark.parametrize('displacement', [[0.,0.,0.],[.3,-.4,1.2]])
def test_raw_contracted_ss_metric_and_charges(displacement):
    A=[.2,.5,-.3]; B=(np.asarray(A)+displacement).tolist()
    shells=[(0,0,[.7,1.4],[2.,-.3]),(1,0,[.9],[.4])]
    provider=core.IsaAuxCoulomb(basis(shells,[A,B]))
    expected=np.zeros((2,2))
    for i,(ci,_,ai,di) in enumerate(shells):
        for j,(cj,_,aj,dj) in enumerate(shells):
            expected[i,j]=sum(x*y*ss(a,b,[A,B][ci],[A,B][cj]) for a,x in zip(ai,di) for b,y in zip(aj,dj))
    got=provider.metric().np
    np.testing.assert_allclose(got,expected,rtol=3e-14,atol=2e-13)
    np.testing.assert_array_equal(got,got.T)
    np.testing.assert_allclose(provider.charges(),[sum(c*(pi/a)**1.5 for a,c in zip(s[2],s[3])) for s in shells],rtol=2e-15)
    moved=core.IsaAuxCoulomb(basis(shells,(np.asarray([A,B])+[2.,-3.,4.]).tolist()))
    np.testing.assert_allclose(moved.metric().np,got,rtol=3e-14,atol=2e-13)
    got[:]=99.
    np.testing.assert_allclose(provider.metric().np,expected,rtol=3e-14,atol=2e-13)


def test_displaced_d_s_mapping_and_mixed_component_scaling():
    a,b=.7,1.1
    A=np.array([.3,.6,-.2]); B=np.array([-.2,.4,.7]); R=A-B
    rho=a*b/(a+b); t=rho*np.dot(R,R)
    x,w=np.polynomial.legendre.leggauss(80); u=(x+1)/2
    f=[np.dot(w/2,u**(2*n)*np.exp(-t*u*u)) for n in range(3)]
    k=2*pi**2.5/(a*b*sqrt(a+b))
    diagonal=[k*(2*a*f[0]-2*rho*f[1]+4*rho*rho*r*r*f[2])/(4*a*a) for r in R]
    mixed=[sqrt(3)*k*rho*rho*R[i]*R[j]*f[2]/(a*a) for i,j in [(0,1),(0,2),(1,2)]]
    p=core.IsaAuxCoulomb(basis([(0,2,[a],[1.]),(1,0,[b],[1.])],[A.tolist(),B.tolist()]))
    np.testing.assert_allclose(p.metric().np[:6,6],diagonal+mixed,rtol=5e-13,atol=2e-13)
    np.testing.assert_allclose(p.charges()[:6],[(pi/a)**1.5/(2*a)]*3+[0.]*3,rtol=2e-15)


@pytest.mark.parametrize('l',range(5))
def test_s_through_g_charge_by_independent_hermite_quadrature(l):
    a,c=.8,-.37
    b=basis([(0,l,[a],[c])],[[0.,0.,0.]])
    x,w=np.polynomial.hermite.hermgauss(6)
    triples=list(itertools.product(range(6),repeat=3))
    points=np.array([[x[i],x[j],x[k]] for i,j,k in triples])/sqrt(a)
    # Remove Gaussian weight already supplied by Hermite quadrature.
    values=b.evaluate(points.tolist()).np*np.exp(a*np.sum(points*points,axis=1))[:,None]
    weights=np.array([w[i]*w[j]*w[k] for i,j,k in triples])/a**1.5
    expected=weights@values
    np.testing.assert_allclose(core.IsaAuxCoulomb(b).charges(),expected,rtol=2e-14,atol=2e-14)


def test_native_three_center_contracted_s_and_closed_shell_trace():
    A=[0.,0.,0.]; B=[.2,.4,.7]; C=[-.1,.6,-.2]
    aux=basis([(0,0,[.7,1.4],[2.,-.3])],[A])
    main=basis([(0,0,[.9],[.4]),(1,0,[1.3],[-.7])],[B,C],
               role=core.IsaBasisRole.Orbital,representation=core.IsaBasisRepresentation.Spherical)
    p=core.IsaAuxCoulomb(aux)
    expected=np.zeros((2,2))
    for i,(b,db,X) in enumerate([(.9,.4,B),(1.3,-.7,C)]):
        for j,(c,dc,Y) in enumerate([(.9,.4,B),(1.3,-.7,C)]):
            centre=((b*np.array(X)+c*np.array(Y))/(b+c)).tolist()
            factor=db*dc*exp(-b*c/(b+c)*sum((x-y)**2 for x,y in zip(X,Y)))
            expected[i,j]=factor*sum(d*ss(a,b+c,A,centre) for a,d in [(.7,2.),(1.4,-.3)])
    actual=p.three_center(main).np.reshape(2,2)
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=2e-13)
    np.testing.assert_array_equal(actual,actual.T)
    occupied=np.array([[.8,.2],[-.3,.7]])
    rhs=p.closed_shell_rhs(main,core.Matrix.from_array(occupied))
    np.testing.assert_allclose(rhs,[2*np.einsum('mi,mn,ni',occupied,expected,occupied)],rtol=5e-14)
    with pytest.raises(ValueError,match='dimensions'):
        p.closed_shell_rhs(main,core.Matrix.from_array(np.ones((3,1))))
    occupied[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        p.closed_shell_rhs(main,core.Matrix.from_array(occupied))


@pytest.mark.parametrize('l',range(1,5))
def test_dalton_main_transform_by_coulomb_potential_quadrature(l):
    a,b=.7,1.1
    A=np.array([.2,-.3,.5]); centre=np.array([-.1,.4,-.2])
    main=basis([(0,l,[b],[1.]),(0,0,[b],[1.])],[centre.tolist()],
               role=core.IsaBasisRole.Orbital,representation=core.IsaBasisRepresentation.Spherical)
    p=core.IsaAuxCoulomb(basis([(0,0,[a],[1.])],[A.tolist()]))
    n=main.nfunction
    got=p.three_center(main).np.reshape(n,n)
    def quadrature(order):
        x,w=np.polynomial.hermite.hermgauss(order)
        triples=np.array(list(itertools.product(range(order),repeat=3)))
        local=x[triples]/sqrt(2*b)
        points=local+centre
        radius=np.linalg.norm(points-A,axis=1)
        potential=(pi/a)**1.5*np.array([erf(sqrt(a)*r)/r if r else 2*sqrt(a/pi) for r in radius])
        weights=np.prod(w[triples],axis=1)/(2*b)**1.5
        poly=main.evaluate(points.tolist()).np*np.exp(b*np.sum(local*local,axis=1))[:,None]
        return np.einsum('p,pi,pj->ij',weights*potential,poly,poly)
    lo,hi=quadrature(24),quadrature(28)
    np.testing.assert_allclose(lo,hi,rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(got,hi,rtol=2e-11,atol=2e-11)


def test_orbital_role_fails_closed_for_cartesian_and_atomic_metric():
    with pytest.raises(ValueError,match='DALTON spherical'):
        basis([(0,0,[1.],[1.])],[[0.,0.,0.]],role=core.IsaBasisRole.Orbital)
    main=basis([(0,0,[1.],[1.])],[[0.,0.,0.]],role=core.IsaBasisRole.Orbital,
               representation=core.IsaBasisRepresentation.Spherical)
    with pytest.raises(ValueError,match='Atomic overlap'): main.overlap()
    aux=basis([(0,0,[1.],[1.])],[[0.,0.,0.]])
    with pytest.raises(ValueError,match='Orbital'): core.IsaAuxCoulomb(aux).three_center(aux)


@pytest.mark.parametrize('penalty',[.25,1000.])
@pytest.mark.parametrize('aux_coefficient',[1.,-.7])
def test_native_drho_c_analytic_finite_penalty(penalty,aux_coefficient):
    a,b=.7,.9
    aux=basis([(0,0,[a],[aux_coefficient])],[[0.,0.,0.]])
    main=basis([(0,0,[b],[1.])],[[0.,0.,0.]],role=core.IsaBasisRole.Orbital,
               representation=core.IsaBasisRepresentation.Spherical)
    c=(2*b/pi)**.75
    p=core.IsaAuxCoulomb(aux)
    result=p.fit_drho_c(main,core.Matrix.from_array(np.array([[c]])),penalty)
    q=aux_coefficient*(pi/a)**1.5
    j=aux_coefficient**2*ss(a,a,[0.]*3,[0.]*3)
    raw=2*c*c*aux_coefficient*ss(a,2*b,[0.]*3,[0.]*3)
    metric=j+(penalty*q)*q
    rhs=raw+penalty*2*q
    expected=rhs/metric
    np.testing.assert_allclose(result.raw_rhs,[raw],rtol=5e-14)
    np.testing.assert_allclose(result.metric.np,[[metric]],rtol=5e-14)
    np.testing.assert_allclose(result.coefficients,[expected],rtol=5e-14)
    assert result.fitted_electrons==pytest.approx(q*expected,rel=5e-14)
    assert abs(result.fitted_electrons-2)>1e-8  # Not silently rescaled.
    assert result.relative_residual<1e-14
    density=core.IsaFixedDensity(aux,result.coefficients)
    np.testing.assert_allclose(density.evaluate([[0.,0.,0.],[1.,0.,0.]],[0]),
                               [expected*aux_coefficient,expected*aux_coefficient*exp(-a)],rtol=5e-14)


@pytest.mark.parametrize('penalty',[0.,float('nan'),float('inf')])
def test_native_drho_rejects_invalid_penalty(penalty):
    aux=basis([(0,0,[1.],[1.])],[[0.,0.,0.]])
    main=basis([(0,0,[1.],[1.])],[[0.,0.,0.]],role=core.IsaBasisRole.Orbital,
               representation=core.IsaBasisRepresentation.Spherical)
    with pytest.raises(ValueError,match='penalty'):
        core.IsaAuxCoulomb(aux).fit_drho_c(main,core.Matrix.from_array(np.ones((1,1))),penalty)


def test_native_aux_rejects_wrong_role():
    shells=[(0,0,[1.],[1.])]
    with pytest.raises(ValueError,match='molecular AUX'):
        core.IsaAuxCoulomb(basis(shells,[[0.,0.,0.]],role=core.IsaBasisRole.AtomAux))
    # A spherical molecular AUX is a supported DIFFERENT declared basis, not a
    # rejected one: it spans 2l+1 per shell where the Cartesian one spans
    # (l+1)(l+2)/2, so its fits may never be quoted against Cartesian ones.
    core.IsaAuxCoulomb(basis(shells,[[0.,0.,0.]],representation=core.IsaBasisRepresentation.Spherical))


@pytest.mark.parametrize('l',[0,1,2,3])
def test_spherical_aux_metric_and_charges_contract_the_cartesian_ones(l):
    """A spherical AUX shell is a fixed linear combination of its Cartesian one.

    The combination is read off the *samples* of the two declared bases, then
    required to carry the Coulomb metric, the analytic charges and the
    three-centre integrals -- so integrals and grid values of a spherical shell
    are shown to share one convention, and the GAMINT mixed-component factor is
    shown never to reach a transformed index. The two bases remain DIFFERENT
    declared bases: the spherical one spans 2l+1 of the (l+1)(l+2)/2 Cartesian
    functions, and their fits may never be quoted as agreeing.
    """
    A=[.2,.5,-.3]; B=[.1,-.4,.7]
    shells=[(0,l,[.8],[1.]),(1,l,[1.3],[1.])]
    centres=[A,B]
    cb=basis(shells,centres)
    sb=basis(shells,centres,representation=core.IsaBasisRepresentation.Spherical)
    rng=np.random.default_rng(20260912)
    points=(np.array(centres)[rng.integers(0,2,600)]+rng.normal(scale=.9,size=(600,3))).tolist()
    cart_values=cb.evaluate(points).np
    sph_values=sb.evaluate(points).np
    M,*_=np.linalg.lstsq(cart_values,sph_values,rcond=None)
    assert np.max(np.abs(cart_values@M-sph_values))<1e-12*max(1.,np.max(np.abs(sph_values)))
    cart=core.IsaAuxCoulomb(cb); sph=core.IsaAuxCoulomb(sb)
    jc=cart.metric().np; qc=np.asarray(cart.charges())
    np.testing.assert_allclose(sph.metric().np,M.T@jc@M,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(np.asarray(sph.charges()),M.T@qc,rtol=1e-11,atol=1e-12)
    main=basis([(0,0,[.9],[.4]),(1,0,[1.3],[-.7])],centres,
               role=core.IsaBasisRole.Orbital,representation=core.IsaBasisRepresentation.Spherical)
    bc=cart.three_center(main).np
    np.testing.assert_allclose(sph.three_center(main).np,M.T@bc,rtol=1e-11,atol=1e-12)
    if l:   # Only l=0 rows of a spherical AUX carry charge; l>0 cancels exactly.
        assert np.max(np.abs(np.asarray(sph.charges())))<1e-13


def _operands(aux, main):
    p=core.IsaAuxCoulomb(aux)
    return [p.metric().np, p.three_center(main).np,
            np.concatenate([p.three_center_shell_block(main,i,1).np for i in range(len(aux.shell_layout()))])]


# Centres 0 and 2 coincide; centre 3 sits inside Molecule.add_atom's 0.05 bohr guard.
IDENTITY_CENTRES=[[0.,0.,0.],[.31,-.42,.53],[0.,0.,0.],[.01,0.,0.]]
IDENTITY_AUX=[(2,2,[.73,1.91],[.37,-.12]),(0,1,[.43,2.3],[1.7,-.21]),(3,3,[.84],[.62]),
              (1,0,[.62,1.8],[-.8,.19]),(0,4,[.95],[.51]),(2,0,[1.1],[-.4])]
IDENTITY_MAIN=[(3,1,[.91,1.81],[.42,-.17]),(0,2,[.75],[1.3]),(2,0,[1.03,1.9],[.31,-.12]),(1,1,[.69],[.55])]


@pytest.mark.parametrize("representation", [
    core.IsaBasisRepresentation.Cartesian, core.IsaBasisRepresentation.Spherical])
def test_native_integrals_keep_explicit_shell_order_and_coincident_centres(representation):
    """Native shells are grouped by centre: explicit order must survive through a map.

    Interleaved shells are compared bitwise with the same shells declared in
    grouped order (rows permuted back), and coincident distinct centres with
    the merged single centre. Row order, not integral values, is under test;
    only the metric, whose mirrored shell pairs swap bra and ket, may round.
    """
    main=basis(IDENTITY_MAIN,IDENTITY_CENTRES,core.IsaBasisRole.Orbital,core.IsaBasisRepresentation.Spherical)
    aux=basis(IDENTITY_AUX,IDENTITY_CENTRES,representation=representation)
    got=_operands(aux,main)
    order=sorted(range(len(IDENTITY_AUX)),key=lambda s:IDENTITY_AUX[s][0])
    assert order!=list(range(len(IDENTITY_AUX)))
    grouped=basis([IDENTITY_AUX[s] for s in order],IDENTITY_CENTRES,representation=representation)
    layout=aux.shell_layout()
    rows=np.concatenate([np.arange(layout[s][0],layout[s][0]+layout[s][1]) for s in order])
    ref=_operands(grouped,main)
    np.testing.assert_allclose(got[0][np.ix_(rows,rows)],ref[0],rtol=1e-14,atol=1e-15)
    for g,r in zip(got[1:],ref[1:]):
        np.testing.assert_array_equal(g[rows],r)
    merged=[(0 if c==2 else c,l,a,d) for c,l,a,d in IDENTITY_AUX]
    for g,r in zip(got,_operands(basis(merged,IDENTITY_CENTRES,representation=representation),main)):
        np.testing.assert_array_equal(g,r)


def test_native_integrals_ignore_ambient_screening_options():
    """INTS_TOLERANCE/SCREENING only shape a native sieve these operands never consult."""
    import psi4
    main=basis(IDENTITY_MAIN,IDENTITY_CENTRES,core.IsaBasisRole.Orbital,core.IsaBasisRepresentation.Spherical)
    aux=basis(IDENTITY_AUX,IDENTITY_CENTRES)
    ref=_operands(aux,main)
    saved={k:(core.get_global_option(k),core.has_global_option_changed(k)) for k in ("INTS_TOLERANCE","SCREENING")}
    try:
        for tolerance,screening in [(1e-1,"SCHWARZ"),(1e-1,"CSAM"),(0.,"NONE")]:
            psi4.set_options({"ints_tolerance":tolerance,"screening":screening})
            for g,r in zip(_operands(aux,main),ref):
                np.testing.assert_array_equal(g,r)
    finally:
        for k,(value,changed) in saved.items():
            core.set_global_option(k,value)
            if not changed: core.revoke_global_option_changed(k)


# Drho-C exact residual and LU refinement. core._isa_exact_residual and
# core._isa_refined_lu_solve are test hooks into the production routines, not API.
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_isapol')
_U = 2.**-53
_DBL_MAX = sys.float_info.max


def _bits(values):
    return [struct.unpack('<Q', struct.pack('<d', float(v)))[0] for v in values]


def _oracle_residual(A, x, b):
    """Correctly rounded b - A x from exact rationals (float(Fraction) rounds half-even)."""
    return [float(Fraction(bi) - sum(Fraction(a)*Fraction(xj) for a, xj in zip(row, x)))
            for row, bi in zip(A, b)]


def _random_system(seed, n, spread):
    rng = np.random.default_rng(seed)
    def draw(*shape):
        return rng.choice([-1., 1.], shape)*np.ldexp(rng.uniform(1., 2., shape), rng.integers(-spread, spread+1, shape))
    return draw(n, n), draw(n), draw(n)


def _residual_cases():
    A, x, b = _random_system(20261006, 40, 300)
    tiny = 2.**-1074
    return {
        'random 40x40, exponents +-300': (A, x, b),
        'catastrophic cancellation': ([[1., 1.], [1., -1.]], [2.**60, 1+2.**-52], [2.**60, 2.**60]),
        'subnormal inputs and results': ([[tiny, 0.], [2.**-1060, 0.]], [3., 3.], [2.**-1073, 2.**-1058]),
        'products below 2^-1074': ([[tiny, 0.], [tiny, tiny]], [2.**-3, 1.5], [0., 0.]),
        'half-way subnormal ties': ([[tiny, 0.], [0., tiny]], [.5, 1.5], [0., 0.]),
        'signed zeros': ([[-0., 0.], [0., -0.]], [0., -0.], [-0., 0.]),
        'half-way normal ties': ([[1., 0.], [0., 1.]], [-1., -1.], [2.**53, 2.**53+2]),
        'products beyond binary64 range cancel': ([[_DBL_MAX, -_DBL_MAX], [_DBL_MAX, -_DBL_MAX]], [2., 2.], [1., -0.]),
    }


@pytest.mark.parametrize('case', list(_residual_cases()))
def test_exact_residual_matches_rational_oracle_bitwise(case):
    A, x, b = _residual_cases()[case]
    got = core._isa_exact_residual(A, x, b)
    assert _bits(got) == _bits(_oracle_residual(np.asarray(A).tolist(), list(x), list(b)))


def test_exact_residual_known_values_and_term_order():
    assert core._isa_exact_residual([[1., 1.], [1., -1.]], [2.**60, 1+2.**-52], [2.**60, 2.**60]) == \
        [-(1+2.**-52), 1+2.**-52]  # naive binary64 evaluation gives [0, 0]
    tiny = 2.**-1074
    assert _bits(core._isa_exact_residual([[tiny, 0.], [0., tiny]], [.5, 1.5], [0., 0.])) == _bits([-0., -2*tiny])
    assert _bits(core._isa_exact_residual([[-0., 0.], [0., -0.]], [0., -0.], [-0., 0.])) == _bits([0., 0.])
    A, x, b = _random_system(20261007, 40, 300)
    order = np.random.default_rng(1).permutation(40)
    assert _bits(core._isa_exact_residual(A, x, b)) == _bits(core._isa_exact_residual(A[:, order], x[order], b))


def test_exact_residual_refuses_overflow_and_invalid_input():
    with pytest.raises(ValueError, match='overflow'):
        core._isa_exact_residual([[_DBL_MAX, _DBL_MAX], [1., 0.]], [2., 2.], [0., 0.])
    good = ([[1., 2.], [3., 4.]], [1., 1.], [0., 0.])
    for position in range(3):
        for bad in (float('nan'), float('inf'), -float('inf')):
            args = [np.array(v, dtype=float) for v in good]
            args[position].flat[0] = bad
            with pytest.raises(ValueError, match='finite'):
                core._isa_exact_residual(*args)
    for args in (([[1., 2.]], [1., 1.], [0.]), ([[1.]], [1., 1.], [0.]), ([[1.]], [1.], [0., 0.]),
                 (np.zeros((0, 0)), [], []), ([1.], [1.], [0.])):
        with pytest.raises(ValueError):
            core._isa_exact_residual(*args)


def test_refined_solve_refuses_invalid_input():
    for iterations in (-1, 33):
        with pytest.raises(ValueError, match='iterations'):
            core._isa_refined_lu_solve([[2.]], [1.], iterations)
    with pytest.raises(TypeError):
        core._isa_refined_lu_solve([[2.]], [1.], 1.5)
    with pytest.raises(ValueError, match='constrained metric'):
        core._isa_refined_lu_solve([[float('nan')]], [1.], 10)
    with pytest.raises(ValueError, match='constrained RHS'):
        core._isa_refined_lu_solve([[2.]], [float('inf')], 0)
    for args in (([[1., 2.]], [1.]), ([[1.]], [1., 1.]), (np.zeros((0, 0)), [])):
        with pytest.raises(ValueError):
            core._isa_refined_lu_solve(*args, 10)
    with pytest.raises(ValueError, match='LU solve failed'):
        core._isa_refined_lu_solve([[0., 0.], [0., 0.]], [1., 1.], 10)


def test_refined_solve_zero_solution_converges_in_one_step():
    x, iterations, displacement = core._isa_refined_lu_solve([[2., 1.], [1., 3.]], [0., 0.], 10)
    assert x == [0., 0.] and iterations == 1 and displacement == 0.


def test_native_drho_c_refinement_on_the_analytic_fit():
    a, b, penalty = .7, .9, 1000.
    aux = basis([(0, 0, [a], [1.])], [[0., 0., 0.]])
    main = basis([(0, 0, [b], [1.])], [[0., 0., 0.]], role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    occupied = core.Matrix.from_array(np.array([[(2*b/pi)**.75]]))
    p = core.IsaAuxCoulomb(aux)
    default, plain, refined = (p.fit_drho_c(main, occupied, penalty), p.fit_drho_c(main, occupied, penalty, 0),
                               p.fit_drho_c(main, occupied, penalty, max_refinement_iterations=10))
    for name in ('coefficients', 'raw_rhs', 'rhs', 'charges'):
        assert _bits(getattr(plain, name)) == _bits(getattr(default, name))
    assert (plain.relative_residual, plain.fitted_electrons) == (default.relative_residual, default.fitted_electrons)
    assert default.refinement_iterations == 0 and default.refinement_displacement == 0.
    q = (pi/a)**1.5
    expected = (2*(2*b/pi)**1.5*ss(a, 2*b, [0.]*3, [0.]*3)+penalty*2*q)/(ss(a, a, [0.]*3, [0.]*3)+penalty*q*q)
    np.testing.assert_allclose(refined.coefficients, [expected], rtol=5e-14)
    assert 1 <= refined.refinement_iterations <= 3 and refined.refinement_displacement < 1e-14
    assert _bits(refined.metric.np.ravel()) == _bits(default.metric.np.ravel())
    for iterations in (-1, 33):
        with pytest.raises(ValueError, match='iterations'):
            p.fit_drho_c(main, occupied, penalty, max_refinement_iterations=iterations)


def _exact_solution(A, b):
    """Exact rational solution of the stored binary64 system (Gaussian elimination)."""
    n = len(b)
    rows = [[Fraction(v) for v in row] + [Fraction(bi)] for row, bi in zip(A, b)]
    for k in range(n):
        pivot = next(i for i in range(k, n) if rows[i][k] != 0)
        rows[k], rows[pivot] = rows[pivot], rows[k]
        for i in range(k+1, n):
            factor = rows[i][k]/rows[k][k]
            rows[i] = [v-factor*w for v, w in zip(rows[i], rows[k])]
    x = [Fraction(0)]*n
    for i in reversed(range(n)):
        x[i] = (rows[i][n]-sum(rows[i][j]*x[j] for j in range(i+1, n)))/rows[i][i]
    return np.array([float(v) for v in x])


def _hilbert(n):
    return [[1./(i+j+1) for j in range(n)] for i in range(n)]


def test_refined_solve_hilbert10_reaches_the_exact_solution():
    # Regression target (kappa ~ 1.6e13), not a forward-error guarantee.
    A, b = _hilbert(10), [1.]*10
    reference = _exact_solution(A, b)
    x, iterations, _ = core._isa_refined_lu_solve(A, b, 10)
    assert 1 <= iterations <= 10
    assert np.max(np.abs(np.array(x)-reference)) <= 8*_U*np.max(np.abs(reference))


def test_refined_solve_hilbert13_converges_accurately_or_refuses():
    # kappa ~ 1e18 > 1/u: either documented refusal is acceptable; a converged
    # result outside the bound is not.
    A, b = _hilbert(13), [1.]*13
    try:
        x, _, _ = core._isa_refined_lu_solve(A, b, 32)
    except ValueError as error:
        assert 'stagnated' in str(error) or 'iteration cap' in str(error)
        return
    reference = _exact_solution(A, b)
    assert np.max(np.abs(np.array(x)-reference)) <= 8*_U*np.max(np.abs(reference))


def _water_operands():
    record = json.load(open(os.path.join(_DATA, 'drhoc_water_reference.json')))
    with np.load(os.path.join(_DATA, 'drhoc_water_operands.npz')) as data:
        A, b = data['A'], data['b']
    for name, array in (('A', A), ('b', b)):
        assert hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest() == \
            record['operand_sha256'][name]
    return A, b, np.array([float(v) for v in record['solution_100_digits']])


def test_refined_solve_frozen_water_reaches_the_100_digit_reference():
    # Regression target on the frozen water Drho-C operands (kappa ~ 6e15), not a guarantee.
    A, b, reference = _water_operands()
    x, iterations, _ = core._isa_refined_lu_solve(A, b, 10)
    assert 1 <= iterations <= 10
    assert np.max(np.abs(np.array(x)-reference)) <= 8*_U*np.max(np.abs(reference))
    rows = np.random.default_rng(246).choice(len(b), 16, replace=False)
    assert _bits(np.array(core._isa_exact_residual(A, reference, b))[rows]) == \
        _bits(_oracle_residual(A[rows].tolist(), reference.tolist(), b[rows].tolist()))


def test_refined_solve_same_process_threads_agree():
    hilbert = (np.array(_hilbert(10)), np.ones(10))
    water = _water_operands()[:2]
    saved = core.get_num_threads()
    try:
        for A, b in (hilbert, water):
            core.set_num_threads(1)
            x1 = np.array(core._isa_refined_lu_solve(A, b, 10)[0])
            core.set_num_threads(8)
            x8 = np.array(core._isa_refined_lu_solve(A, b, 10)[0])
            assert np.max(np.abs(x1-x8)) <= 8*_U*np.max(np.abs(x1))
    finally:
        core.set_num_threads(saved)


_GLIBC_FE_UPWARD = {'x86_64': 0x800, 'aarch64': 0x400000}


@pytest.mark.skipif(not sys.platform.startswith('linux') or platform.machine() not in _GLIBC_FE_UPWARD
                    or platform.libc_ver()[0] != 'glibc',
                    reason='the fesetround(FE_UPWARD) refusal is exercised only on x86-64/aarch64 glibc')
def test_refinement_refuses_directed_rounding_but_residual_is_unaffected(tmp_path):
    A, x, b = _random_system(20261008, 12, 40)
    expected = _bits(_oracle_residual(A.tolist(), x.tolist(), b.tolist()))
    script = tmp_path / 'upward.py'
    script.write_text(f'''
import ctypes, struct
from psi4 import core
libm = ctypes.CDLL('libm.so.6')
assert libm.fesetround({_GLIBC_FE_UPWARD[platform.machine()]}) == 0
r = core._isa_exact_residual({A.tolist()!r}, {x.tolist()!r}, {b.tolist()!r})
assert [struct.unpack('<Q', struct.pack('<d', v))[0] for v in r] == {expected!r}
core._isa_refined_lu_solve([[2., 1.], [1., 3.]], [1., 2.], 0)
try:
    core._isa_refined_lu_solve([[2., 1.], [1., 3.]], [1., 2.], 1)
except ValueError as error:
    assert 'round-to-nearest' in str(error), error
else:
    raise SystemExit('directed rounding was not refused')
assert libm.fesetround(0) == 0
core._isa_refined_lu_solve([[2., 1.], [1., 3.]], [1., 2.], 1)
print('UPWARD-OK')
''')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))] +
                                        [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p])
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0 and 'UPWARD-OK' in done.stdout, done.stdout + done.stderr
