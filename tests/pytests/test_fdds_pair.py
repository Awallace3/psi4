"""Synthetic pair wiring tests; no claim of native monomer construction."""
import numpy as np
import pytest
from psi4 import core
from psi4.driver.procrouting.sapt import sapt_mp2_terms as terms


@pytest.mark.parametrize('hybrid', [False, True])
def test_pair_uses_shared_signed_response(monkeypatch, hybrid):
    j = np.array([[2., .3], [.3, 1.4]])
    inv = np.linalg.inv(j)
    rmat = np.array([[1., .2], [0., 1.7]])
    w = j + .02*np.eye(2)
    alpha = .25 if hybrid else 0.
    calls = []
    original = terms.solve_fdds_response

    def intermediates(label, omega):
        b = np.array([[.4, .8], [.7, .1]]) * (1. if label == 'A' else .7)
        gaps = np.array([.8, 1.3])
        u = -4*b.T @ np.diag(gaps/(gaps*gaps+omega*omega)) @ b
        base = np.array([[.01, .02], [-.01, .03]])/(1+omega*omega)
        return {'amp': u, 'K1LD': base, 'K2LD': base.T*.6,
                'K2L': base*.3, 'K21L': base.T*.2}

    class FakeFDDS:
        def __init__(self, *args):
            pass

        def metric(self):
            return core.Matrix.from_array(j)

        def metric_inv(self):
            return core.Matrix.from_array(inv)

        def aux_overlap(self):
            return core.Matrix.from_array(np.eye(2))

        def project_densities(self, densities):
            return densities

        def R_A(self):
            return core.Matrix.from_array(rmat)

        R_B = R_A

        def form_unc_amplitude(self, label, omega):
            return core.Matrix.from_array(-intermediates(label, omega)['amp'])

        def form_aux_matrices(self, label, omega):
            return {k: core.Matrix.from_array(v) for k, v in intermediates(label, omega).items()}

    def shared_spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(core, 'FDDS_Dispersion', FakeFDDS)
    monkeypatch.setattr(terms, 'solve_fdds_response', shared_spy)
    monkeypatch.setattr(terms, '_compute_fxc', lambda *args, **kwargs: core.Matrix.from_array(.02*np.eye(2)))
    cache = {key: core.Matrix.from_array(np.eye(2)) for key in
             ['Cocc_A', 'Cvir_A', 'Cocc_B', 'Cvir_B', 'D_A', 'D_B']}
    cache.update({key: core.Vector.from_array(np.ones(2)) for key in
                  ['eps_occ_A', 'eps_vir_A', 'eps_occ_B', 'eps_vir_B']})
    result = terms.df_fdds_dispersion(None, None, cache, hybrid, alpha, leg_points=3, do_print=False)
    assert len(calls) == 6
    total = np.zeros(2)
    for point, weight in zip(*np.polynomial.legendre.leggauss(3)):
        omega = .3*(1-point)/(1+point)
        responses = []
        for label in ['A', 'B']:
            h = intermediates(label, omega)
            u = h['amp']
            x = u - alpha*h['K2L']
            t = x @ inv @ w
            if hybrid:
                k = -alpha*(h['K1LD']+h['K2LD']) + alpha**2*h['K21L']
                t += .25*k @ np.linalg.inv(rmat).T @ j
            # Independent full-rank form, not the production pseudoinverse expression.
            c = j @ np.linalg.solve(j-t, x)
            responses.append((.5*(u+u.T), .5*(c+c.T)))
        for i in [0, 1]:
            total[i] += np.trace(inv @ responses[0][i] @ inv @ responses[1][i])*weight*.6/(1+point)**2
    expected = -total/(2*np.pi)
    np.testing.assert_allclose([result['Disp20,FDDS (unc)'], result['Disp20']], expected, rtol=3e-13)
    for i, kwargs in enumerate(calls):
        assert (kwargs['hybrid'] is not None) == hybrid
        assert np.all(np.diag(kwargs['uncoupled']) < 0)
        point = np.polynomial.legendre.leggauss(3)[0][i//2]
        omega = .3*(1-point)/(1+point)
        np.testing.assert_array_equal(kwargs['uncoupled'], intermediates('A' if i%2 == 0 else 'B', omega)['amp'])
