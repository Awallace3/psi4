"""Analytic/quadrature tail-kernel checks, not native ISA tail/controller parity."""
from math import exp, pi, erfc, sqrt, pow
import numpy as np
import pytest
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def shape(coefficient=1., contracted=False, centre=(0., 0., 0.)):
    s = core.IsaGaussianShell()
    s.exponents = [.4, 1.3] if contracted else [.7]
    s.coefficients = [2., -.3] if contracted else [1.]
    b = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical, [centre], [s])
    return core.IsaGaussianShape(b, [coefficient])


def tail(amplitude=2., exponent=2., cutoff=1.):
    t = core.IsaExponentialTail()
    t.defined, t.amplitude, t.exponent, t.cutoff = True, amplitude, exponent, cutoff
    return t


def quadrature(function, lower, upper):
    x, w = np.polynomial.legendre.leggauss(200)
    r = lower+(x+1)*(upper-lower)/2
    return np.dot(w, 4*pi*r*r*function(r))*(upper-lower)/2


@pytest.mark.parametrize('radius', [0., .5, 1.5, 4.])
def test_gaussian_shape_contracted_exterior_charge(radius):
    s = shape(2., contracted=True)
    want = lambda r: 4*np.exp(-.4*r*r)-.6*np.exp(-1.3*r*r)
    assert s.value(radius) == pytest.approx(want(radius), rel=2e-15)
    numerical = quadrature(want, radius, radius+20)
    assert s.exterior_charge(radius) == pytest.approx(numerical, rel=5e-13, abs=1e-13)
    moved = shape(2., contracted=True, centre=(2., -3., 5.))
    np.testing.assert_array_equal(moved.sample([[2+radius, -3., 5.]]), s.sample([[radius, 0, 0]]))


@pytest.mark.parametrize('radius', [0., .7, 1.7])
def test_exterior_charge_preserves_reference_primitive_order(radius):
    s = core.IsaGaussianShell()
    s.exponents, s.coefficients = [.4, 1.3, 2.1], [2., -.3, .017]
    b = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical,
                              [[0., 0., 0.]], [s])
    d = 3.7
    expected = 0.
    for a,c in zip(s.exponents, s.coefficients):
        integral = pow(pi/a, 1.5)*erfc(radius*sqrt(a))
        integral += (2*pi*radius/a)*exp((-a*radius)*radius)
        expected += (d*c)*integral
    assert core.IsaGaussianShape(b, [d]).exterior_charge(radius) == expected


def test_fit3_slope_charge_not_continuity():
    s = shape()
    result = s.fit_tail(1.5)
    t = result.tail
    assert t.defined and not result.used_previous_exponent and result.status == 'fitted'
    assert t.exponent == pytest.approx(2*.7*1.5, rel=3e-8)
    numerical_charge = quadrature(lambda r: t.amplitude*np.exp(-t.exponent*r), t.cutoff, t.cutoff+50)
    assert numerical_charge == pytest.approx(s.exterior_charge(t.cutoff), rel=5e-13)
    assert result.gaussian_tail_charge == s.exterior_charge(1.5)
    assert result.ionization_potential == pytest.approx(t.exponent*t.exponent/8)
    # Fit-Type 3 conserves exterior charge, not the value at the cutoff.
    assert abs(t.amplitude*exp(-t.exponent*t.cutoff)-s.value(t.cutoff)) > 1e-3
    t.amplitude = 99
    assert result.tail.amplitude != 99


@pytest.mark.parametrize('z', [0., -1.12168732, 100.])
def test_shape_and_fit3_use_translated_ordered_shell_evaluation(z):
    shells = [core.IsaGaussianShell(), core.IsaGaussianShell()]
    shells[0].exponents, shells[0].coefficients = [.4, 1.3], [2., -.3]
    shells[1].exponents, shells[1].coefficients = [.9], [.7]
    basis = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical,
                                  [[0., 0., z]], shells)
    coefficients = [3.7, -.2]
    radius, step = 1.5, 1e-8
    points = [[0., 0., z+r] for r in [radius, radius-step, radius+step]]
    values = basis.evaluate(points).np
    w = [sum(d*float(values[p,k]) for k,d in enumerate(coefficients)) for p in range(3)]
    expected = -((w[2]-w[1])/(2*step))/w[0]
    gaussian = core.IsaGaussianShape(basis, coefficients)
    np.testing.assert_array_equal(gaussian.sample(points), w)
    fitted = gaussian.fit_tail(radius)
    assert fitted.tail.defined
    assert fitted.tail.exponent == expected


def test_tail_fallback_and_deterministic_undefined():
    s = shape()
    undefined = s.fit_tail(.1)  # b=.14 outside strict range
    assert not undefined.tail.defined and undefined.status == 'undefined_slope'
    assert undefined.tail.amplitude == undefined.tail.exponent == undefined.ionization_potential == 0
    result = s.fit_tail(.1, tail(amplitude=99., exponent=2.5))
    assert result.tail.defined and result.used_previous_exponent
    assert result.tail.exponent == 2.5 and result.tail.amplitude != 99
    assert result.status == 'previous_exponent'
    zero = shape(0.).fit_tail(1.5)
    assert not zero.tail.defined
    far = s.fit_tail(1000., tail())
    assert not far.tail.defined and far.status == 'undefined_tail_integral'
    assert far.tail.amplitude == far.tail.exponent == far.ionization_potential == 0


def test_active_tail_keeps_negative_interior_and_strict_cutoff():
    s = shape(-1.)
    points = [[0, 0, 0], [1, 0, 0], [1.1, 0, 0]]
    t = tail(cutoff=1.)
    np.testing.assert_array_equal(s.sample(points), [0, 0, 0])
    np.testing.assert_array_equal(s.sample(points, t, False), [0, 0, 0])
    np.testing.assert_allclose(s.sample(points, t, True), [-1., -exp(-.7), 2*exp(-2*1.1)], atol=1e-15)
    np.testing.assert_array_equal(s.sample(points, core.IsaExponentialTail(), True), [0, 0, 0])
    # A charge-conserving fitted tail can itself be signed; no hidden positivity claim.
    fitted = s.fit_tail(1.5)
    assert fitted.tail.defined and fitted.tail.amplitude < 0
    assert s.sample([[2, 0, 0]], fitted.tail, True)[0] < 0


@pytest.mark.parametrize('radius', [-1., np.nan, np.inf, 1e308])
def test_shape_rejects_invalid_radius(radius):
    for method in (shape().value, shape().exterior_charge):
        with pytest.raises(ValueError, match='radius'):
            method(radius)


@pytest.mark.parametrize('cutoff', [0., 1e-10, -1., np.nan, np.inf])
def test_fit3_rejects_invalid_cutoff(cutoff):
    with pytest.raises(ValueError, match='cutoff'):
        shape().fit_tail(cutoff)


@pytest.mark.parametrize('field,value', [('amplitude', np.inf), ('exponent', 0.),
    ('exponent', np.nan), ('cutoff', -1.)])
def test_shape_rejects_invalid_active_tail(field, value):
    t = tail()
    setattr(t, field, value)
    with pytest.raises(ValueError, match='Tail'):
        shape().sample([[0, 0, 0]], t, True)


@pytest.mark.parametrize('exponent', [1., 4.])
def test_fit3_previous_exponent_strict_bounds(exponent):
    with pytest.raises(ValueError, match='1<b<4'):
        shape().fit_tail(.1, tail(exponent=exponent))


def test_shape_rejects_invalid_descriptors_and_points():
    s = core.IsaGaussianShell()
    s.exponents, s.coefficients = [.7], [1.]
    b = core.IsaExplicitBasis(core.IsaBasisRole.AtomAux, core.IsaBasisRepresentation.Spherical, [[0, 0, 0]], [s])
    with pytest.raises(ValueError, match='Shape basis role'):
        core.IsaGaussianShape(b, [1.])
    b = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical, [[0, 0, 0]], [s])
    for coefficients, match in [([], 'dimension'), ([np.nan], 'finite')]:
        with pytest.raises(ValueError, match=match):
            core.IsaGaussianShape(b, coefficients)
    s.centre = 1
    second = core.IsaGaussianShell()
    second.exponents, second.coefficients = [.7], [1.]
    b = core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical,
                              [[0, 0, 0], [1, 0, 0]], [second, s])
    with pytest.raises(ValueError, match='co-centred'):
        core.IsaGaussianShape(b, [1., 1.])
    with pytest.raises(ValueError, match='finite'):
        shape().sample([[np.nan, 0, 0]])
