"""Polynomial translation/rotation identities, not ORIENT localization parity."""
import math
import numpy as np
import pytest
import psi4
from numpy.polynomial import Legendre

c=psi4.core


def harmonics(rank, point):
    x,y,z=point
    radius=np.linalg.norm(point)
    if radius==0:
        result=np.zeros((rank+1)**2); result[0]=1.
        return result
    cos=z/radius; phi=np.arctan2(y,x)
    out=[]
    for l in range(rank+1):
        p=Legendre.basis(l)
        out.append(radius**l*p(cos))
        for m in range(1,l+1):
            value=radius**l*np.sqrt(2*math.factorial(l-m)/math.factorial(l+m))
            value*=max(0.,1-cos*cos)**(m/2)*p.deriv(m)(cos)
            out.extend([value*np.cos(m*phi),value*np.sin(m*phi)])
    return np.array(out)


def rotation():
    axis=np.array([1.,2.,-3.]); axis/=np.linalg.norm(axis)
    skew=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    angle=.72
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


@pytest.mark.parametrize("rank",range(5))
def test_translation_polynomials_and_group(rank):
    d=np.array([.3,-.7,.2]); e=np.array([-.4,.2,.8]); x=np.array([.7,.3,-.6])
    t=c.isa_multipole_translation(rank,d).np
    u=c.isa_multipole_translation(rank,e).np
    np.testing.assert_allclose(t@harmonics(rank,x),harmonics(rank,x+d),rtol=1e-13,atol=2e-14)
    np.testing.assert_allclose(t[:,0],harmonics(rank,d),rtol=1e-13,atol=2e-14)
    np.testing.assert_allclose(t@u,c.isa_multipole_translation(rank,d+e).np,atol=2e-14)
    np.testing.assert_allclose(t@c.isa_multipole_translation(rank,-d).np,np.eye(len(t)),atol=2e-14)
    np.testing.assert_allclose(c.isa_multipole_translation(rank,[0.,0.,0.]).np,np.eye(len(t)),atol=2e-16)
    for l in range(rank+1):
        np.testing.assert_allclose(t[l*l:(l+1)**2,l*l:(l+1)**2],np.eye(2*l+1),atol=2e-16)
        np.testing.assert_array_equal(t[l*l:(l+1)**2,(l+1)**2:],0.)


def test_axial_translation_binomial_coefficients():
    d=.67
    t=c.isa_multipole_translation(4,[0.,0.,d]).np
    for l in range(5):
        for k in range(l+1):
            np.testing.assert_allclose(t[l*l,k*k],math.comb(l,k)*d**(l-k),rtol=2e-15,atol=2e-16)


@pytest.mark.parametrize("rank",range(5))
def test_rotation_polynomials_and_covariance(rank):
    f=rotation(); x=np.array([.7,.3,-.6]); d=np.array([.3,-.7,.2])
    r=c.isa_multipole_rotation(rank,f).np
    np.testing.assert_allclose(r@harmonics(rank,x),harmonics(rank,f@x),rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(r@r.T,np.eye(len(r)),atol=2e-15)
    np.testing.assert_allclose(r.T,c.isa_multipole_rotation(rank,f.T).np,atol=2e-15)
    np.testing.assert_allclose(r@r,c.isa_multipole_rotation(rank,f@f).np,atol=2e-15)
    np.testing.assert_allclose(r@c.isa_multipole_translation(rank,d).np@r.T,
                               c.isa_multipole_translation(rank,f@d).np,atol=2e-14)
    for l in range(rank+1):
        np.testing.assert_array_equal(r[l*l:(l+1)**2,:l*l],0.)
        np.testing.assert_array_equal(r[l*l:(l+1)**2,(l+1)**2:],0.)


def test_rank_one_axis_convention_and_owned_results():
    f=rotation()
    r=c.isa_multipole_rotation(1,f)
    expected=np.eye(4); expected[1:,1:]=f[np.ix_([2,0,1],[2,0,1])]
    np.testing.assert_allclose(r.np,expected,atol=1e-15)
    r.np[:]=99.
    np.testing.assert_allclose(c.isa_multipole_rotation(1,f).np,expected,atol=1e-15)


def test_noncommuting_rotation_composition():
    f=rotation(); angle=.43
    g=np.array([[np.cos(angle),-np.sin(angle),0.],
                [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    assert not np.allclose(f@g,g@f)
    df=c.isa_multipole_rotation(4,f).np
    dg=c.isa_multipole_rotation(4,g).np
    np.testing.assert_allclose(df@dg,c.isa_multipole_rotation(4,f@g).np,atol=2e-15)


def test_translation_ownership_large_representable_displacement():
    d=np.array([1.e40,-2.e40,3.e40])
    t=c.isa_multipole_translation(4,d)
    expected=harmonics(4,d)
    np.testing.assert_allclose(t.np[:,0],expected,rtol=2e-14,atol=1e-14)
    t.np[:]=0.
    np.testing.assert_allclose(c.isa_multipole_translation(4,d).np[:,0],expected,rtol=2e-14,atol=1e-14)


def test_transform_against_existing_gaussian_harmonics():
    shells=[]
    for rank in range(5):
        s=c.IsaGaussianShell()
        s.l,s.centre,s.exponents,s.coefficients=rank,0,[.5],[1.]
        shells.append(s)
    basis=c.IsaExplicitBasis(c.IsaBasisRole.MolecularAux,
        c.IsaBasisRepresentation.Spherical,[[0.,0.,0.]],shells)
    order=[0,3,1,2]
    for rank in range(2,5):
        order.append(rank*rank+rank)
        for m in range(1,rank+1): order.extend([rank*rank+rank+m,rank*rank+rank-m])
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,0.,1.],[.2,-.5,.3]])
    def evaluate(p):
        return basis.evaluate(p).np[:,order]*np.exp(.5*np.sum(p*p,axis=1))[:,None]
    r=evaluate(points); d=np.array([.3,-.2,.4]); f=rotation()
    np.testing.assert_allclose(r@c.isa_multipole_translation(4,d).np.T,evaluate(points+d),atol=3e-14)
    np.testing.assert_allclose(r@c.isa_multipole_rotation(4,f).np.T,evaluate(points@f.T),atol=3e-14)


def test_frame_tolerance_boundaries():
    # The frame check is necessary, not a promise to accept every approximately
    # orthogonal frame: polynomial reconstruction also has a 1e-12 residual gate.
    near=np.eye(3); near[0,0]+=1.e-14
    assert np.isfinite(c.isa_multipole_rotation(4,near).np).all()
    outside=np.eye(3); outside[0,0]+=1.e-12
    with pytest.raises(ValueError,match="orthogonal"):
        c.isa_multipole_rotation(4,outside)


def test_full_rank_tensor_translation_conservation_primitive():
    # Opposite balanced endpoint transfers lie in the nullspace of the
    # molecular translation operator through every rank, including (4,4).
    a=np.array([.1,.2,-.3]); b=np.array([.6,-.3,.7]); origin=np.array([.2,-.4,.5])
    ta=c.isa_multipole_translation(4,a-origin).np
    tb=c.isa_multipole_translation(4,b-origin).np
    ab=c.isa_multipole_translation(4,a-b).np
    ba=c.isa_multipole_translation(4,b-a).np
    h=np.vstack((-(np.eye(25)+ba),np.eye(25)+ab))
    u=np.hstack((ta,tb))
    np.testing.assert_allclose(u@h,0.,atol=2e-14)
    rng=np.random.default_rng(3101)
    v=rng.normal(size=(50,25))
    update=h@v.T+v@h.T
    np.testing.assert_allclose(u@update@u.T,0.,atol=5e-13)


@pytest.mark.parametrize("rank",[-1,5])
def test_invalid_rank(rank):
    with pytest.raises(ValueError): c.isa_multipole_translation(rank,[0.,0.,0.])
    with pytest.raises(ValueError): c.isa_multipole_rotation(rank,np.eye(3))


@pytest.mark.parametrize("value",[np.nan,np.inf,1.e100])
def test_invalid_or_overflow_translation(value):
    with pytest.raises(ValueError): c.isa_multipole_translation(4,[value,0.,0.])


@pytest.mark.parametrize("frame",[np.eye(3)*2,np.diag([-1.,1.,1.]),np.full((3,3),np.nan),
                                  [[1.,.1,0.],[0.,1.,0.],[0.,0.,1.]]])
def test_invalid_frame(frame):
    with pytest.raises(ValueError): c.isa_multipole_rotation(4,frame)
