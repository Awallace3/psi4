"""Supplied-intermediate FDDS tests, not native CamCASP response parity."""
import numpy as np
import pytest

from psi4.driver.procrouting.sapt.fdds_response import (
    prepare_fdds_hybrid_transform, solve_fdds_response,
)


def inputs():
    j = np.array([[2., .3], [.3, 1.4]])
    return dict(metric=j, metric_inv=np.linalg.inv(j),
                W=np.array([[1.9, .1], [.1, 1.1]]),
                uncoupled=np.array([[-.5, -.12], [-.12, -.3]]))


@pytest.mark.parametrize('omega', [0., .4, 1.e6])
def test_fdds_scalar_sign_and_limits(omega):
    gap, b, j, w = 1.2, .7, 2., 2.1
    u = -4*b*b*gap/(gap*gap+omega*omega)
    r = solve_fdds_response(metric=[[j]], metric_inv=[[1/j]], W=[[w]], uncoupled=[[u]])
    np.testing.assert_allclose(r.coupled, [[u/(1-u*w/(j*j))]], rtol=2e-14)
    assert r.coupled[0, 0] < 0
    assert r.representation == 'fdds_coulomb_auxiliary'
    if omega > 1.e5:
        assert r.coupled[0, 0]*omega**2 == pytest.approx(-4*b*b*gap, rel=1e-10)


@pytest.mark.parametrize('omega', [0., .3, 1.e6])
def test_fdds_nonhybrid_direct_mo_oracle(omega):
    # Independent small MO-space solve, ordinary unregularized Coulomb fitting.
    rng = np.random.default_rng(823)
    b = rng.normal(size=(4, 3))*.2
    j = np.array([[2., .3, -.1], [.3, 1.4, .2], [-.1, .2, 1.1]])
    inv = np.linalg.inv(j)
    w = j + np.eye(3)*.03
    gaps = np.array([.6, 1., 1.4, 2.1])
    u = -4*b.T @ np.diag(gaps/(gaps*gaps+omega*omega)) @ b
    r = solve_fdds_response(metric=j, metric_inv=inv, W=w, uncoupled=u)
    d = b @ inv
    h2 = np.diag(gaps)
    h1 = h2 + 4*d @ w @ d.T
    x = np.linalg.solve(h2 @ h1 + omega*omega*np.eye(4), -4*h2 @ d)
    expected = d.T @ x
    np.testing.assert_allclose(inv @ r.raw_coupled @ inv, expected, rtol=2e-12, atol=1e-26)
    np.testing.assert_allclose(r.raw_coupled, r.raw_coupled.T, rtol=2e-12, atol=1e-26)


@pytest.mark.parametrize('alpha', [0., .25, 1.])
def test_fdds_hybrid_noncommuting_order_and_ownership(alpha):
    args = inputs()
    rng = np.random.default_rng(902)
    h = {k: rng.normal(size=(2, 2))*.03 for k in ['K1LD', 'K2LD', 'K2L', 'K21L']}
    h['Rtinv'] = prepare_fdds_hybrid_transform([[1., .2], [0., 1.7]])
    before = {k: v.copy() for k, v in {**args, **h}.items()}
    result = solve_fdds_response(**args, x_alpha=alpha, hybrid=h)
    x = args['uncoupled'] - alpha*h['K2L']
    k = -alpha*(h['K1LD']+h['K2LD']) + alpha**2*h['K21L']
    t = x @ args['metric_inv'] @ args['W'] + .25*k @ h['Rtinv'] @ args['metric']
    expected = args['metric'] @ np.linalg.solve(args['metric']-t, x)
    np.testing.assert_allclose(result.raw_coupled, expected, rtol=2e-14, atol=2e-16)
    np.testing.assert_allclose(result.coupled, .5*(expected+expected.T), rtol=2e-14, atol=2e-16)
    if alpha == 0:
        np.testing.assert_array_equal(result.coupled, solve_fdds_response(**args).coupled)
    else:
        assert np.max(np.abs(result.raw_coupled-result.raw_coupled.T)) > 1e-6
    for k, v in {**args, **h}.items():
        np.testing.assert_array_equal(v, before[k])
        for out in (result.raw_uncoupled, result.raw_coupled, result.uncoupled, result.coupled):
            assert not np.shares_memory(v, out)
    result.raw_uncoupled[:] = 0
    np.testing.assert_array_equal(result.uncoupled, args['uncoupled'])


def test_fdds_preserves_pseudoinverse_rank_policy():
    j = np.diag([1., 1.e-14])
    u = np.diag([-.2, -.1])
    r = solve_fdds_response(metric=j, metric_inv=np.diag([1., 0.]), W=np.zeros((2, 2)), uncoupled=u)
    np.testing.assert_array_equal(r.coupled, u)
    np.testing.assert_array_equal(prepare_fdds_hybrid_transform(j), np.diag([1., 0.]))
    # Coupling inverse itself drops the second near-null singular direction.
    r = solve_fdds_response(metric=j, metric_inv=np.eye(2), W=np.diag([0., 1.e-14]), uncoupled=u)
    np.testing.assert_array_equal(r.raw_coupled, u)


def test_fdds_R_sanitation_does_not_mutate():
    r = np.array([[1., np.nan], [0., 2.]])
    original = r.copy()
    np.testing.assert_array_equal(prepare_fdds_hybrid_transform(r), np.diag([1., .5]))
    np.testing.assert_array_equal(r, original)


@pytest.mark.parametrize('key', ['metric', 'metric_inv', 'W', 'uncoupled'])
@pytest.mark.parametrize('bad', [np.ones((2, 3)), np.eye(3), np.full((2, 2), np.nan), np.eye(2)*1j])
def test_fdds_rejects_bad_matrices(key, bad):
    args = inputs()
    args[key] = bad
    with pytest.raises(ValueError):
        solve_fdds_response(**args)


@pytest.mark.parametrize('alpha', [float('nan'), float('inf'), 1j, [0.]])
def test_fdds_rejects_invalid_exchange(alpha):
    with pytest.raises(ValueError, match='x_alpha'):
        solve_fdds_response(**inputs(), x_alpha=alpha)


def test_fdds_requires_complete_hybrid_inputs():
    with pytest.raises(ValueError, match='Missing hybrid'):
        solve_fdds_response(**inputs(), hybrid={'K2L': np.eye(2)})
