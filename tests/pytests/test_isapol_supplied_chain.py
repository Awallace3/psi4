"""Composed synthetic stages, explicitly NOT native wavefunction-to-property parity."""
from itertools import product
import numpy as np
import psi4

c = psi4.core


def monosite_response(exponent, strength, gap, omega, label):
    """Gaussian p AUX, unity partition, supplied isotropic Lorentz coefficient operator."""
    shell = c.IsaGaussianShell()
    shell.centre, shell.l = 0, 1
    shell.exponents, shell.coefficients = [exponent], [1.]
    basis = c.IsaExplicitBasis(c.IsaBasisRole.MolecularAux,
        c.IsaBasisRepresentation.Cartesian, [[0., 0., 0.]], [shell])
    nodes, weights = np.polynomial.hermite.hermgauss(4)
    indices = np.array(list(product(range(4), repeat=3)))
    points = nodes[indices]/np.sqrt(exponent)
    quadrature = np.prod(weights[indices], axis=1)/exponent**1.5
    quadrature *= np.exp(exponent*np.sum(points**2, axis=1))
    samples = c.IsaMultipoleSamples()
    samples.points, samples.weights = points.tolist(), quadrature.tolist()
    samples.shape = samples.shape_sum = [1.]*len(points)
    samples.auxiliary_sites = [0]
    site = c.IsaMultipoleSite()
    site.label, site.rank, site.samples = label, 1, samples
    partition = c.IsaPartitionedMultipoles(basis, [site], "synthetic unity partition")
    # Integral x*x*exp(-a*r^2) = (pi/a)^(3/2)/(2a).
    dipole_leg = (np.pi/exponent)**1.5/(2*exponent)
    scalar = strength*gap**2/(gap**2+omega**2)
    operators = [c.Matrix.from_array(-np.eye(3)*x/dipole_leg**2) for x in scalar]
    response = c.IsaDistributedResponse(partition, omega.tolist(), operators,
        "fitted_density_coefficients", "supplied Lorentz operators; not a native fit/kernel")
    extracted = []
    for f, expected in enumerate(scalar):
        alpha = response.at_index(f).np
        np.testing.assert_allclose(alpha[1:, 1:], np.eye(3)*expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(alpha[0], 0., atol=2e-14)
        extracted.append(np.trace(alpha[1:, 1:])/3.)
    # This one-site object is already local. This extraction is NOT a general
    # localization prescription for multisite distributed responses.
    local = c.IsaIsotropicSite()
    local.label, local.ranks = label, [1]
    local.polarizabilities = c.Matrix.from_array(np.array(extracted)[:, None])
    return c.IsaIsotropicModel(omega.tolist(), [local], "synthetic single-site scalar response")


def test_supplied_partition_response_dispersion_composition():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    omega = (1+nodes)/(1-nodes)
    cp = weights/(np.pi*(1-nodes)**2)
    a = monosite_response(.7, 2.3, .8, omega, "A")
    b = monosite_response(1.1, 4.1, 1.7, omega, "B")
    result = c.isa_isotropic_dispersion(a, b, cp.tolist())
    c6, c8, c10, c12 = result.pairs[0].coefficients
    exact_c6 = 1.5*2.3*4.1*.8*1.7/(.8+1.7)
    np.testing.assert_allclose(c6.value, exact_c6, rtol=2e-14)
    assert c6.complete and c6.included_rank_pairs == [[1, 1]]
    for truncated in [c8, c10, c12]:
        assert not truncated.complete and truncated.value == 0.
        assert truncated.included_rank_pairs == [] and truncated.missing_rank_pairs
