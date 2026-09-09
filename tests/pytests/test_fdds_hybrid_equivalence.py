"""Characterize inherited hybrid equations; do not alter SAPT policy to match full OV."""
import numpy as np
from psi4.driver.procrouting.sapt.fdds_response import prepare_fdds_hybrid_transform,solve_fdds_response


def supplied_hybrid(gaps,B,S,E,omega,a,W=None):
    """Literal C++ intermediate contractions, with identity auxiliary metric."""
    n=B.shape[1];J=np.eye(n)
    if W is None:W=np.zeros((n,n))
    delta=np.diag(gaps);L=np.diag(-4/(gaps*gaps+omega*omega))
    Q,R=np.linalg.qr(B,mode='reduced')
    h=dict(K1LD=B.T@L@delta@S@Q,K2LD=B.T@L@delta@E@Q,
           K2L=B.T@L@E@B,K21L=B.T@E.T@L@S@Q,
           Rtinv=prepare_fdds_hybrid_transform(R))
    result=solve_fdds_response(metric=J,metric_inv=J,W=W,uncoupled=B.T@L@delta@B,x_alpha=a,hybrid=h)
    h1=delta+4*B@W@B.T-a*S;h2=delta-a*E
    full=B.T@np.linalg.solve(h2@h1+omega*omega*np.eye(len(gaps)),-4*h2@B)
    return result.raw_coupled,full


def test_full_rank_commuting_hybrid_control():
    gaps=np.array([1.,2.]);S=np.array([[.2,.1],[.1,.3]]);E=np.diag([.13,.21])
    actual,full=supplied_hybrid(gaps,np.eye(2),S,E,.4,.25,np.full((2,2),.03))
    np.testing.assert_allclose(actual,full,rtol=1e-12,atol=1e-12)


def test_full_rank_noncommuting_ordering_difference():
    gaps=np.array([1.,2.]);delta=np.diag(gaps)
    S=np.array([[.2,.1],[.1,.3]]);E=np.array([[.13,.19],[.19,.21]])
    omega=.4;step=1e-5
    def difference(a):
        actual,full=supplied_hybrid(gaps,np.eye(2),S,E,omega,a)
        return actual-full
    derivative=(difference(step)-difference(-step))/(2*step)
    L=np.diag(-4/(gaps*gaps+omega*omega))
    predicted=-.25*L@(delta@E-E@delta)@L@delta
    np.testing.assert_allclose(derivative,predicted,rtol=1e-7,atol=1e-8)
    assert np.max(np.abs(difference(.25)))>1e-4


def test_rectangular_closed_and_leaking_exchange_actions():
    gaps=np.array([1.,1.5,2.]);B=np.eye(3)[:,:2];E=np.zeros((3,3))
    S=np.array([[.2,.1,0],[.1,.3,0],[0,0,.4]])
    actual,full=supplied_hybrid(gaps,B,S,E,.4,.25)
    np.testing.assert_allclose(actual,full,rtol=1e-12,atol=1e-12)
    S[0,2]=S[2,0]=.3
    actual,full=supplied_hybrid(gaps,B,S,E,.4,.25)
    assert np.max(np.abs(actual-full))>1e-4


def test_effective_baseline_coupling_preserves_exchange_physics():
    gaps=np.array([1.,1.5,2.]);delta=np.diag(gaps)
    D=np.array([[.2,.1],[.3,-.1],[-.1,.25]])
    S=np.array([[.2,.1,.2],[.1,.3,-.1],[.2,-.1,.4]])
    E=np.array([[.1,.02,.04],[.02,.2,.01],[.04,.01,.15]])
    K=np.array([[-.15,.02],[.02,-.12]]);a=.25;omega=.4
    # Actual Coulomb need not equal D D^T for these external fitting legs.
    V=np.diag([.2,.1,.15]);h10=delta+4*V-a*S;h2=delta-a*E
    U=D.T@np.linalg.solve(h2@h10+omega*omega*np.eye(3),-4*h2@D)
    result=solve_fdds_response(metric=np.eye(2),metric_inv=np.eye(2),W=(1-a)*K,uncoupled=U)
    h1=h10+4*(1-a)*D@K@D.T
    full=D.T@np.linalg.solve(h2@h1+omega*omega*np.eye(3),-4*h2@D)
    np.testing.assert_allclose(result.raw_coupled,full,rtol=1e-12,atol=1e-12)
    # Identity-metric diagnostic coordinates do not change the helper's fixed label.
    assert result.representation=='fdds_coulomb_auxiliary'
