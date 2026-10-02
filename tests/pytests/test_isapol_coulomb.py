"""Analytic native Libint2 AUX checks; not native Drho-C/end-to-end acceptance."""
from math import erf, exp, pi, sqrt
import itertools
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
        aux, main, coefficients, energies, nocc=2, shell_count=2,
        exact_exchange=exchange, tile_columns=4)
    expected_density = 2*sum(dual[:, i, i] for i in range(2))
    np.testing.assert_allclose(streamed.plain_density_coefficients, expected_density,
                               atol=2e-12, rtol=2e-12)
    with pytest.raises(ValueError):
        streamed.plain_density_coefficients.setflags(write=True)
    coefficients[:] = 0.
    energies[:] = 0.
    np.testing.assert_allclose(streamed.apply_h1(np.eye(6)), h1, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(streamed.apply_h2(np.eye(6)), h2, atol=2e-12, rtol=2e-12)


def test_streamed_plain_df_admission_and_shell_coverage():
    from psi4.driver.procrouting.isapol_native_factors import native_plain_df_operators
    aux = basis([(0, 0, [.8], [.6]), (0, 1, [.7], [.5])], [[0., 0., 0.]])
    main = basis([(0, 0, [1.1], [.7]), (0, 1, [.6], [.3])], [[.2, 0., 0.]],
                 role=core.IsaBasisRole.Orbital,
                 representation=core.IsaBasisRepresentation.Spherical)
    coeff = np.eye(4)
    energies = np.array([-1., -.8, .2, .6])
    kwargs = dict(nocc=2, shell_count=2, exact_exchange=.25, tile_columns=1)
    ops = native_plain_df_operators(aux, main, coeff, energies, **kwargs)
    budget = ops.construction_planned_bytes
    exact = native_plain_df_operators(aux, main, coeff, energies, **kwargs, max_bytes=budget)
    np.testing.assert_array_equal(exact.apply_h1(np.eye(4)), ops.apply_h1(np.eye(4)))
    with pytest.raises(ValueError, match="byte resource"):
        native_plain_df_operators(aux, main, coeff, energies, **kwargs, max_bytes=budget-1)
    for change, message in [
        (dict(shell_count=1), "coverage"), (dict(shell_count=3), "range"),
        (dict(nocc=True), "nocc"), (dict(nocc=4), "nocc"),
        (dict(tile_columns=0), "tile_columns"), (dict(tile_columns=709), "tile_columns"),
        (dict(exact_exchange=np.nan), "exchange"), (dict(max_bytes=True), "max_bytes"),
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
        nocc=2, shell_count=3, exact_exchange=.25, tile_columns=4)
    original_h1 = ops.apply_h1(np.eye(6))
    original_h2 = ops.apply_h2(np.eye(6))
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
    np.testing.assert_array_equal(ops.apply_h1(np.eye(6)), original_h1)
    np.testing.assert_array_equal(ops.apply_h2(np.eye(6)), original_h2)
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
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
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


def _operands(aux, main, points):
    p=core.IsaAuxCoulomb(aux)
    return [p.metric().np, p.point_potentials(core.Matrix.from_array(points)).np, p.three_center(main).np,
            np.concatenate([p.three_center_shell_block(main,i,1).np for i in range(len(aux.shell_layout()))])]


# Centres 0 and 2 coincide; centre 3 sits inside Molecule.add_atom's 0.05 bohr guard.
IDENTITY_CENTRES=[[0.,0.,0.],[.31,-.42,.53],[0.,0.,0.],[.01,0.,0.]]
IDENTITY_AUX=[(2,2,[.73,1.91],[.37,-.12]),(0,1,[.43,2.3],[1.7,-.21]),(3,3,[.84],[.62]),
              (1,0,[.62,1.8],[-.8,.19]),(0,4,[.95],[.51]),(2,0,[1.1],[-.4])]
IDENTITY_MAIN=[(3,1,[.91,1.81],[.42,-.17]),(0,2,[.75],[1.3]),(2,0,[1.03,1.9],[.31,-.12]),(1,1,[.69],[.55])]
IDENTITY_POINTS=np.array(IDENTITY_CENTRES+[[.7,-.9,1.2]])


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
    got=_operands(aux,main,IDENTITY_POINTS)
    order=sorted(range(len(IDENTITY_AUX)),key=lambda s:IDENTITY_AUX[s][0])
    assert order!=list(range(len(IDENTITY_AUX)))
    grouped=basis([IDENTITY_AUX[s] for s in order],IDENTITY_CENTRES,representation=representation)
    layout=aux.shell_layout()
    rows=np.concatenate([np.arange(layout[s][0],layout[s][0]+layout[s][1]) for s in order])
    ref=_operands(grouped,main,IDENTITY_POINTS)
    np.testing.assert_allclose(got[0][np.ix_(rows,rows)],ref[0],rtol=1e-14,atol=1e-15)
    for g,r in zip(got[1:],ref[1:]):
        np.testing.assert_array_equal(g[rows],r)
    merged=[(0 if c==2 else c,l,a,d) for c,l,a,d in IDENTITY_AUX]
    for g,r in zip(got,_operands(basis(merged,IDENTITY_CENTRES,representation=representation),main,IDENTITY_POINTS)):
        np.testing.assert_array_equal(g,r)


def test_native_integrals_ignore_ambient_screening_options():
    """INTS_TOLERANCE/SCREENING only shape a native sieve these operands never consult."""
    import psi4
    main=basis(IDENTITY_MAIN,IDENTITY_CENTRES,core.IsaBasisRole.Orbital,core.IsaBasisRepresentation.Spherical)
    aux=basis(IDENTITY_AUX,IDENTITY_CENTRES)
    ref=_operands(aux,main,IDENTITY_POINTS)
    saved={k:(core.get_global_option(k),core.has_global_option_changed(k)) for k in ("INTS_TOLERANCE","SCREENING")}
    try:
        for tolerance,screening in [(1e-1,"SCHWARZ"),(1e-1,"CSAM"),(0.,"NONE")]:
            psi4.set_options({"ints_tolerance":tolerance,"screening":screening})
            for g,r in zip(_operands(aux,main,IDENTITY_POINTS),ref):
                np.testing.assert_array_equal(g,r)
    finally:
        for k,(value,changed) in saved.items():
            core.set_global_option(k,value)
            if not changed: core.revoke_global_option_changed(k)
