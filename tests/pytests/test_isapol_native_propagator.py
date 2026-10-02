# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The declared ALDA kernel smoothing (floor and cap) used by BOUNDED_DF."""
import numpy as np
import pytest

from psi4.driver.procrouting import isapol_native_propagator as pr

#: CamCASP's KERNEL-INTEGRAL-PARAMETERS, as declared by the BOUNDED_DF presets.
FD = pr.KernelSmoothing(1.e-8, 1000., .01, 1., 'FD')


def test_smoothing_requires_explicit_finite_positive_parameters():
    for bad in (dict(rho_epsilon=0.), dict(f_max=-1.), dict(fd_delta=float('inf')),
                dict(fd_alpha=1)):
        kw = dict(rho_epsilon=1.e-8, f_max=1000., fd_delta=.01, fd_alpha=1., method='FD')
        kw.update(bad)
        with pytest.raises(ValueError):
            pr.KernelSmoothing(**kw)
    with pytest.raises(ValueError):
        pr.KernelSmoothing(1.e-8, 1000., .01, 1., 'SMOOTH')


def test_fd_smoothing_records_the_reference_parameter_set():
    s = FD
    assert (s.rho_epsilon, s.f_max, s.fd_delta, s.fd_alpha, s.method) == (1.e-8, 1000., .01, 1., 'FD')
    assert s.method_code == 3
    assert 'F-MAX=1000.0' in s.declaration


def test_fd_limit_matches_the_declared_scalar_form_in_both_branches():
    s = FD
    q = np.array([0., 1., -1., 500., 999.5, 1000., 1000.5, 1010., -1010., 1e5, -1e5, 1e8])

    def scalar(x):
        z = (abs(x)/s.f_max - 1.)/s.fd_delta
        fd = (1./(1.+np.exp(z)))**s.fd_alpha if z <= 40. else np.exp(-s.fd_alpha*z)
        return x*fd if x <= s.f_max else s.f_max*fd

    np.testing.assert_allclose(s.limit(q), [scalar(x) for x in q], rtol=1.e-14, atol=0.)
    assert np.isfinite(s.limit(q)).all()
    # The cap is a cap: nothing survives far above F-MAX, and small values are untouched.
    assert abs(s.limit(np.array([1.]))[0] - 1.) < 1.e-12
    assert abs(s.limit(np.array([1.e8]))[0]) < 1.e-30


def test_zero_and_constant_methods_are_their_own_declared_forms():
    q = np.array([-2000., -10., 10., 2000.])
    np.testing.assert_array_equal(pr.KernelSmoothing(1.e-8, 1000., .01, 1., 'ZERO').limit(q),
                                  [0., -10., 10., 0.])
    np.testing.assert_array_equal(pr.KernelSmoothing(1.e-8, 1000., .01, 1., 'CONSTANT').limit(q),
                                  [-1000., -10., 10., 1000.])


def test_floor_raises_the_density_and_never_drops_a_row():
    s = FD
    d = np.array([-1.e-30, 0., 1.e-12, 1.e-3])
    out = s.floor(d)
    assert out.shape == d.shape and np.all(out >= s.rho_epsilon)
    assert out[-1] == d[-1]
