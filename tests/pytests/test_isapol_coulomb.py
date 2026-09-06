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


@pytest.mark.parametrize('penalty',[0.,-1.,float('nan'),float('inf')])
def test_native_drho_rejects_invalid_penalty(penalty):
    aux=basis([(0,0,[1.],[1.])],[[0.,0.,0.]])
    main=basis([(0,0,[1.],[1.])],[[0.,0.,0.]],role=core.IsaBasisRole.Orbital,
               representation=core.IsaBasisRepresentation.Spherical)
    with pytest.raises(ValueError,match='penalty'):
        core.IsaAuxCoulomb(aux).fit_drho_c(main,core.Matrix.from_array(np.ones((1,1))),penalty)


def test_native_aux_rejects_wrong_role_and_representation():
    shells=[(0,0,[1.],[1.])]
    with pytest.raises(ValueError,match='molecular AUX'):
        core.IsaAuxCoulomb(basis(shells,[[0.,0.,0.]],role=core.IsaBasisRole.AtomAux))
    with pytest.raises(ValueError,match='Cartesian'):
        core.IsaAuxCoulomb(basis(shells,[[0.,0.,0.]],representation=core.IsaBasisRepresentation.Spherical))
