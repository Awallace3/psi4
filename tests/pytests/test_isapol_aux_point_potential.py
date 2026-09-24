"""Exact AUX point potentials: a fitted-target prerequisite, not parity evidence."""
from math import erf, pi, sqrt

import numpy as np
import pytest
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def auxiliary(l=0, representation=core.IsaBasisRepresentation.Cartesian):
    shell = core.IsaGaussianShell()
    shell.centre, shell.l = 0, l
    shell.exponents, shell.coefficients = [.7, 1.4], [2., -.3]
    return core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux, representation,
                                 [[.2, -.4, .1]], [shell])


def test_contracted_s_potential_including_at_nucleus():
    points = np.array([[.2, -.4, .1], [1.2, -.1, -.7], [20., 10., -30.]])
    radii = np.linalg.norm(points - [.2, -.4, .1], axis=1)
    expected = [sum(c * (2*pi/a if r == 0 else
                         (pi/a)**1.5 * erf(sqrt(a)*r)/r)
                    for a, c in zip([.7, 1.4], [2., -.3])) for r in radii]
    provider = core.IsaAuxCoulomb(auxiliary())
    actual = provider.point_potentials(core.Matrix.from_array(points)).np
    assert actual.shape == (1, 3)
    np.testing.assert_allclose(actual[0], expected, rtol=3e-14, atol=2e-14)


@pytest.mark.parametrize('l', [1, 2])
def test_displaced_cartesian_p_and_d_against_boys_integrals(l):
    """Independent monomial integrals, including GAMINT mixed-d factors."""
    displacement = np.array([.6, -.9, .4])
    points = core.Matrix.from_array(np.array([[.2, -.4, .1]]) + displacement)
    x, w = np.polynomial.legendre.leggauss(80)
    u, weights = (x+1)/2, w/2
    expected = np.zeros(3 if l == 1 else 6)
    for a, c in zip([.7, 1.4], [2., -.3]):
        t = a * np.dot(displacement, displacement)
        f = [np.dot(weights, u**(2*k)*np.exp(-t*u*u)) for k in range(3)]
        if l == 1:
            expected += c * 2*pi/a * displacement * f[1]
        else:
            expected[:3] += c * (pi/a**2*(f[0]-f[1]) +
                                 2*pi/a*displacement**2*f[2])
            expected[3:] += c * sqrt(3)*2*pi/a*f[2] * np.array(
                [displacement[i]*displacement[j] for i, j in [(0, 1), (0, 2), (1, 2)]])
    actual = core.IsaAuxCoulomb(auxiliary(l)).point_potentials(points).np
    np.testing.assert_allclose(actual[:, 0], expected, rtol=6e-13, atol=3e-14)


@pytest.mark.parametrize('l', range(5))
def test_spherical_potentials_use_the_same_basis_as_grid_values(l):
    """The separately tested explicit-basis evaluator fixes the harmonic map."""
    cart = auxiliary(l)
    pure = auxiliary(l, core.IsaBasisRepresentation.Spherical)
    samples = np.random.default_rng(814).normal(size=(80, 3))
    transform, *_ = np.linalg.lstsq(cart.evaluate(samples.tolist()).np,
                                   pure.evaluate(samples.tolist()).np, rcond=None)
    points = core.Matrix.from_array(np.array([[.2, -.4, .1], [1.1, -.7, 2.],
                                             [-3., 4., -.1]]))
    cart_values = core.IsaAuxCoulomb(cart).point_potentials(points).np
    pure_values = core.IsaAuxCoulomb(pure).point_potentials(points).np
    np.testing.assert_allclose(pure_values, transform.T @ cart_values, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('points, message', [
    (np.zeros((0, 3)), 'nonempty'),
    (np.zeros((2, 2)), 'npoint x 3'),
    (np.zeros((513, 3)), 'maximum 512'),
    (np.array([[np.nan, 0., 0.]]), 'Nonfinite'),
    (np.array([[0., np.inf, 0.]]), 'Nonfinite'),
])
def test_invalid_points_refused(points, message):
    with pytest.raises(ValueError, match=message):
        core.IsaAuxCoulomb(auxiliary()).point_potentials(core.Matrix.from_array(points))


def test_byte_boundary_and_results_are_owned():
    provider = core.IsaAuxCoulomb(auxiliary())
    points = core.Matrix.from_array(np.array([[.2, -.4, .1]]))
    with pytest.raises(ValueError, match='byte resource'):
        provider.point_potentials(points, max_bytes=7)
    with pytest.raises(ValueError, match='byte resource'):
        provider.point_potentials(points, max_bytes=0)
    first = provider.point_potentials(points, max_bytes=8)
    expected = first.np.copy()
    first.np[:] = -99.
    np.testing.assert_array_equal(provider.point_potentials(points, max_bytes=8).np, expected)
    np.testing.assert_array_equal(points.np, [[.2, -.4, .1]])


@pytest.mark.parametrize('l, indices', [(3, [0, 9]), (4, [0, 9])])
def test_f_g_potentials_by_independent_gaussian_moment_quadrature(l, indices):
    """Laplace transform of 1/r reduces the integral to Gaussian moments."""
    r = np.array([.6, -.9, .4])
    x, w = np.polynomial.legendre.leggauss(80)
    u, weights = (x+1)/2, w/2
    expected = np.zeros(2)
    for a, c in zip([.7, 1.4], [2., -.3]):
        mean = r[:, None]*u**2
        variance = (1-u*u)/(2*a)
        if l == 3:  # x^3 and sqrt(15)*xyz in GAMINT order
            moments = [mean[0]**3 + 3*mean[0]*variance,
                       sqrt(15)*mean[0]*mean[1]*mean[2]]
        else:  # x^4 and sqrt(35/3)*x^2 y^2
            moments = [mean[0]**4 + 6*mean[0]**2*variance + 3*variance**2,
                       sqrt(35/3)*(mean[0]**2+variance)*(mean[1]**2+variance)]
        kernel = weights*np.exp(-a*np.dot(r, r)*u*u)
        expected += c*2*pi/a*np.array([np.dot(kernel, m) for m in moments])
    points = core.Matrix.from_array(np.array([[.2, -.4, .1]]) + r)
    actual = core.IsaAuxCoulomb(auxiliary(l)).point_potentials(points).np
    np.testing.assert_allclose(actual[indices, 0], expected, rtol=8e-13, atol=5e-14)


def test_multiple_shells_centres_and_translation():
    centres = np.array([[.2, -.4, .1], [-.8, .3, 1.2]])
    points = np.array([[.2, -.4, .1], [.7, -.9, 2.]])
    shells, separate = [], []
    for site, l in [(0, 2), (1, 0), (1, 3)]:
        shell = core.IsaGaussianShell()
        shell.centre, shell.l = site, l
        shell.exponents, shell.coefficients = [.7], [.4]
        shells.append(shell)
        # Keep the declared centres but a single shell, testing row offsets.
        basis = core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux,
                                      core.IsaBasisRepresentation.Cartesian,
                                      centres.tolist(), [shell])
        separate.append(core.IsaAuxCoulomb(basis).point_potentials(
            core.Matrix.from_array(points)).np.copy())
    expected = np.concatenate(separate)
    for shift in [np.zeros(3), np.array([2., -1., .5])]:
        combined = core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux,
                                         core.IsaBasisRepresentation.Cartesian,
                                         (centres+shift).tolist(), shells)
        actual = core.IsaAuxCoulomb(combined).point_potentials(
            core.Matrix.from_array(points+shift)).np
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-14)


def test_symmetry_blocked_coordinates_refused():
    rows = core.Dimension.from_list([1, 1])
    cols = core.Dimension.from_list([3, 3])
    with pytest.raises(ValueError, match='npoint x 3'):
        core.IsaAuxCoulomb(auxiliary()).point_potentials(core.Matrix('blocked', rows, cols))
