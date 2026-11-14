"""GRAC asymptotic-correction options from Cencek & Szalewicz, J. Chem. Phys. 139, 024104 (2013).

DFT_GRAC_DENSITY_HESSIAN adds their Eq. 17 switching-gradient term; DFT_GRAC_SPLICE STRETCH is
their Sec. IV splice for range-separated hybrids.  Both are checked against a numpy rebuild of the
potential matrix on Psi4's own grid.
"""
import numpy as np
import pytest

import psi4
from psi4.driver.procrouting.dft import build_superfunctional

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

SHIFT = 0.12
ALPHA, BETA = 0.5, 40.0


@pytest.fixture
def water():
    psi4.core.clean_options()
    mol = psi4.geometry("""
    0 1
    O  -1.551007  -0.114520   0.000000
    H  -1.934259   0.762503   0.000000
    H  -0.599677   0.040712   0.000000
    symmetry c1
    no_reorient
    no_com
    """)
    psi4.set_options({"basis": "aug-cc-pvdz", "scf_type": "df", "dft_radial_points": 50,
                      "dft_spherical_points": 194, "d_convergence": 1e-8})
    yield mol
    psi4.core.clean_options()
    psi4.core.clean()


def _grac_V(mol, func, Da, shift, **options):
    """Build the RKS potential for a fixed density with the given GRAC options."""
    psi4.set_options({"dft_grac_shift": shift, **options})
    wfn = psi4.driver.scf_wavefunction_factory(func, psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS")), "RHF")
    V = wfn.V_potential()
    M = psi4.core.Matrix(Da.rows(), Da.cols())
    V.set_D([Da])
    V.compute_V([M])
    return V, M.np.copy()


def _blocks(V, Da):
    """Yield (w, phi, fmap, point values, npoints, numpy values, xyz) for every grid block with functions."""
    pw = V.properties()[0]
    pw.set_pointers(Da)
    for blk in V.grid().blocks():
        n = blk.npoints()
        fmap = np.array(blk.functions_local_to_global())
        if len(fmap) == 0:
            continue
        pw.compute_points(blk, True)
        pv = pw.point_values()
        xyz = np.stack([np.array(c.np[:n]) for c in (blk.x(), blk.y(), blk.z())], 1)
        yield (np.array(blk.w().np[:n]), np.array(pw.basis_values()["PHI"].np[:n, :len(fmap)]), fmap, pv, n,
               {k: np.array(pv[k].np[:n]) for k in pv}, xyz)


def _switch(rho, sigma):
    ok = (rho >= 1e-16) & (sigma > 0)
    g = np.where(ok, np.sqrt(np.where(ok, sigma, 1.0)) / np.where(ok, rho, 1.0)**(4 / 3), 1e2)
    return ok, 1.0 / (1.0 + np.exp(-ALPHA * (g - BETA)))


def test_grac_density_hessian_is_eq17(water):
    """V(Hessian) - V(no Hessian) = -int 2 v_gamma (grad rho . grad f) phi_mu phi_nu."""
    _, wfn = psi4.energy("pbe0", return_wfn=True, molecule=water)
    Da = wfn.Da()
    _, V_off = _grac_V(water, "pbe0", Da, SHIFT, dft_grac_density_hessian=False)
    V_on_obj, V_on = _grac_V(water, "pbe0", Da, SHIFT, dft_grac_density_hessian=True)
    assert V_on_obj.functional().grac_density_hessian()

    bulk = build_superfunctional("pbe0", True)[0]
    bulk.set_max_points(psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS"))
    bulk.set_deriv(1)
    bulk.allocate()
    ref = np.zeros_like(V_on)
    samples = []
    for w, phi, fmap, pv, n, g, xyz in _blocks(V_on_obj, Da):
        vg = np.array(bulk.compute_functional(pv, n)["V_GAMMA_AA"].np[:n])
        rho, sig = g["RHO_A"], g["GAMMA_AA"]
        ok, f = _switch(rho, sig)
        grad = np.stack([g["RHO_AX"], g["RHO_AY"], g["RHO_AZ"]], 1)
        H = np.empty((n, 3, 3))
        for i, a in enumerate("XYZ"):
            for j, b in enumerate("XYZ"):
                H[:, i, j] = g["RHO_A" + min(a, b) + max(a, b)]
        s = np.sqrt(np.where(ok, sig, 1.0))
        r = np.where(ok, rho, 1.0)
        dgr = np.einsum("pi,pij,pj->p", grad, H, grad) / (s * r**(4 / 3)) - 4 / 3 * s * sig / r**(7 / 3)
        gdf = np.where(ok, ALPHA * f * (1 - f) * dgr, 0.0)
        ref[np.ix_(fmap, fmap)] -= np.einsum("p,pm,pn->mn", w * 2 * vg * gdf, phi, phi)
        samples += [(xyz[p], H[p]) for p in np.where(ok & (f > 0.05) & (f < 0.95))[0][:1]]

    assert np.linalg.norm(ref) > 1e-4, "the switching region must be sampled"
    np.testing.assert_allclose(V_on - V_off, ref, atol=1e-12, rtol=0)

    # The density Hessian itself against finite differences of the density.
    basis, D = wfn.basisset(), Da.np

    def rho_at(p):
        ph = np.array(basis.compute_phi(*p))
        return 2 * ph @ D @ ph

    def grad_at(p, h=1e-4):
        return np.array([(rho_at(p + h * e) - rho_at(p - h * e)) / (2 * h) for e in np.eye(3)])

    assert len(samples) >= 5
    for p, H in samples[:5]:
        Hfd = np.array([(grad_at(p + 2e-3 * e) - grad_at(p - 2e-3 * e)) / 4e-3 for e in np.eye(3)])
        np.testing.assert_allclose(H, Hfd, atol=1e-5 * np.abs(H).max(), rtol=0)


def test_grac_stretch_is_bulk_minus_shift(water):
    """STRETCH: V = V_bulk - int (1 - f) shift phi_mu phi_nu, with no LB94 splice."""
    psi4.set_options({"dft_grac_shift": 0.0})
    _, wfn = psi4.energy("lc-wpbe", return_wfn=True, molecule=water)
    Da = wfn.Da()
    _, V_bulk = _grac_V(water, "lc-wpbe", Da, 0.0)
    V_obj, V_st = _grac_V(water, "lc-wpbe", Da, SHIFT, dft_grac_splice="STRETCH")
    assert V_obj.functional().grac_stretch()

    ref = np.zeros_like(V_st)
    for w, phi, fmap, pv, n, g, _ in _blocks(V_obj, Da):
        _, f = _switch(g["RHO_A"], g["GAMMA_AA"])
        ref[np.ix_(fmap, fmap)] -= np.einsum("p,pm,pn->mn", w * (1 - f) * SHIFT, phi, phi)
    np.testing.assert_allclose(V_st - V_bulk, ref, atol=1e-12, rtol=0)


def test_grac_stretch_needs_full_long_range_exchange(water):
    psi4.set_options({"dft_grac_shift": SHIFT, "dft_grac_splice": "STRETCH"})
    with pytest.raises(RuntimeError, match="100% long-range exact"):
        psi4.energy("pbe0", molecule=water)
