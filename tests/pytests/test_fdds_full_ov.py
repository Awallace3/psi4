"""Explicit shared full-OV route, independent full coupled solves and ownership."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from psi4.driver.procrouting.sapt.fdds_response import FDDSFullOVResponse,solve_fdds_response


def inputs():
    return dict(h1_baseline=np.array([[2.,.1],[.1,3.]]),h2=np.array([[1.2,.07],[.07,1.7]]),
                transition_legs=np.array([[.3,.2],[-.1,.4]]),coupling=np.array([[-.2,.03],[.03,-.1]]),
                representation='fitted_density_coefficients')


@pytest.mark.parametrize('omega',[0.,.1,1.,10.])
def test_full_ov_matches_direct_coupled_solve(omega):
    p=inputs();out=FDDSFullOVResponse(**p).at_frequency(omega)
    d=p['transition_legs'];h1=p['h1_baseline']+4*d@p['coupling']@d.T;h2=p['h2']
    expected=d.T@np.linalg.solve(h2@h1+omega*omega*np.eye(2),-4*h2@d)
    baseline=d.T@np.linalg.solve(h2@p['h1_baseline']+omega*omega*np.eye(2),-4*h2@d)
    np.testing.assert_allclose(out.raw_coupled,expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(out.raw_baseline,baseline,atol=1e-12,rtol=1e-12)
    assert out.representation==p['representation'] and out.method=='full_ov_effective_baseline'
    assert not hasattr(out,'uncoupled')


def test_full_ov_owns_inputs_and_frequency_results():
    p=inputs();provider=FDDSFullOVResponse(**p);expected=provider.at_frequency(.4)
    for value in p.values():
        if isinstance(value,np.ndarray):value[:]=np.nan
    actual=provider.at_frequency(.4)
    np.testing.assert_array_equal(actual.raw_coupled,expected.raw_coupled)
    assert not np.shares_memory(actual.raw_baseline,actual.raw_coupled)
    actual.raw_baseline[:]=123;actual.raw_coupled[:]=456
    np.testing.assert_array_equal(provider.at_frequency(.4).raw_coupled,expected.raw_coupled)
    with pytest.raises(FrozenInstanceError):expected.representation='changed'


def test_full_ov_does_not_symmetrize_supplied_operators():
    p=inputs();p['h1_baseline'][0,1]=.4;p['coupling'][:]=0
    out=FDDSFullOVResponse(**p).at_frequency(.3)
    assert np.max(np.abs(out.raw_coupled-out.raw_coupled.T))>1e-3
    np.testing.assert_array_equal(out.raw_coupled,out.raw_baseline)


def test_full_ov_supports_more_legs_than_transitions():
    p=inputs();p['transition_legs']=np.array([[1.,.2,.3],[.1,1.,.4]])
    p['coupling']=np.eye(3)*.02
    out=FDDSFullOVResponse(**p).at_frequency(.5)
    d=p['transition_legs'];h2=p['h2'];h1=p['h1_baseline']+4*d@p['coupling']@d.T
    np.testing.assert_allclose(out.raw_coupled,d.T@np.linalg.solve(h2@h1+.25*np.eye(2),-4*h2@d),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('omega',[-1.,np.nan,np.inf,1+0j,[],None,'0.2',True,1e308])
def test_full_ov_rejects_bad_frequencies(omega):
    with pytest.raises(ValueError):FDDSFullOVResponse(**inputs()).at_frequency(omega)


@pytest.mark.parametrize('key,value',[
    ('h1_baseline',[[1.,2.]]),('h2',np.eye(3)),('h2',np.eye(2,dtype=complex)),
    ('h1_baseline',np.full((2,2),np.nan)),('transition_legs',np.eye(3)),
    ('transition_legs',np.zeros((2,0))),('transition_legs',np.full((2,2),np.inf)),
    ('transition_legs',np.eye(2,dtype=complex)),('coupling',np.eye(3)),
    ('representation',None),('representation','implicit')])
def test_full_ov_rejects_malformed_inputs(key,value):
    p=inputs();p[key]=value
    with pytest.raises(ValueError):FDDSFullOVResponse(**p)


def test_full_ov_singular_baseline_has_no_fallback():
    p=inputs();p['h1_baseline'][:]=0
    provider=FDDSFullOVResponse(**p)
    with pytest.raises(np.linalg.LinAlgError):provider.at_frequency(0.)
    assert np.isfinite(provider.at_frequency(1.).raw_coupled).all()


def test_full_ov_preserves_inherited_coupling_policy():
    # Singular coupling denominator: characterize inherited pinv behavior, not
    # equality to a singular physical inverse or an automatically stabilized fit.
    p=dict(h1_baseline=np.eye(2),h2=np.eye(2),transition_legs=np.eye(2),
           coupling=np.diag([-.25,0.]),representation='supplied_transition_leg_coordinates')
    out=FDDSFullOVResponse(**p).at_frequency(0.)
    old=solve_fdds_response(metric=np.eye(2),metric_inv=np.eye(2),W=p['coupling'],uncoupled=-4*np.eye(2))
    np.testing.assert_array_equal(out.raw_coupled,old.raw_coupled)


def test_full_ov_rectangular_noncommuting_coupling():
    h10=np.array([[2.,.1,.2],[.1,3.,-.1],[.2,-.1,2.5]])
    h2=np.array([[1.,.03,.02],[.03,1.4,.01],[.02,.01,1.7]])
    d=np.array([[.3,.2],[-.1,.4],[.25,-.15]])
    w=np.array([[.01,.03],[-.02,.04]])
    provider=FDDSFullOVResponse(h1_baseline=h10,h2=h2,transition_legs=d,coupling=w,
                              representation='supplied_transition_leg_coordinates')
    out=provider.at_frequency(.4)
    expected=d.T@np.linalg.solve(h2@(h10+4*d@w@d.T)+.16*np.eye(3),-4*h2@d)
    np.testing.assert_allclose(out.raw_coupled,expected,rtol=1e-12,atol=1e-12)
    assert np.max(np.abs(out.raw_coupled-out.raw_coupled.T))>1e-7


@pytest.mark.parametrize('where',['baseline_product','rhs_scaling','frequency_addition','coupling'])
def test_full_ov_finite_input_overflow_fails_without_nonfinite_result(where):
    h1=h2=1.;w=0.;omega=0.
    if where=='baseline_product':h1=h2=1e308
    elif where=='rhs_scaling':h1,h2=1e-308,1e308
    elif where=='frequency_addition':h1,omega=1e308,1e154
    else:w=1e308
    # NumPy/BLAS failures may be ValueError, FloatingPointError or LinAlgError;
    # none is normalized into a fallback response. Suppress expected warnings only.
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises((ValueError,FloatingPointError,np.linalg.LinAlgError)):
            provider=FDDSFullOVResponse(h1_baseline=[[h1]],h2=[[h2]],transition_legs=[[1.]],
                coupling=[[w]],representation='supplied_transition_leg_coordinates')
            provider.at_frequency(omega)
