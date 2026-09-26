"""Supplied isotropic models composed into dispersion; not native wavefunction parity."""
import numpy as np
import psi4

c = psi4.core


def monosite_response(strength, gap, omega, label):
    """One site carrying a supplied isotropic Lorentz dipole polarizability."""
    local = c.IsaIsotropicSite()
    local.label, local.ranks = label, [1]
    scalar = strength*gap**2/(gap**2+omega**2)
    local.polarizabilities = c.Matrix.from_array(scalar[:, None])
    return c.IsaIsotropicModel(omega.tolist(), [local], "synthetic single-site scalar response")


def test_supplied_isotropic_dispersion_composition():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    omega = (1+nodes)/(1-nodes)
    cp = weights/(np.pi*(1-nodes)**2)
    a = monosite_response(2.3, .8, omega, "A")
    b = monosite_response(4.1, 1.7, omega, "B")
    result = c.isa_isotropic_dispersion(a, b, cp.tolist())
    c6, c8, c10, c12 = result.pairs[0].coefficients
    exact_c6 = 1.5*2.3*4.1*.8*1.7/(.8+1.7)
    np.testing.assert_allclose(c6.value, exact_c6, rtol=2e-14)
    assert c6.complete and c6.included_rank_pairs == [[1, 1]]
    for truncated in [c8, c10, c12]:
        assert not truncated.complete and truncated.value == 0.
        assert truncated.included_rank_pairs == [] and truncated.missing_rank_pairs
