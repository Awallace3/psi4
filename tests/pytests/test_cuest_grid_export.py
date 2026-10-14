import numpy as np
import psi4
import pytest

from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api]


@uusing("cuest")
@uusing("cuda_cc8")
def test_live_cuest_grid_export():
    """Check count, units/layout, total weights, ownership and plan lifetime."""
    psi4.set_options({"basis": "cc-pvdz", "use_cuest": True, "cuest_xc": True,
                      "dft_radial_points": 75, "dft_spherical_points": 302,
                      "dft_pruning_scheme": "none"})
    mol = psi4.geometry("He 1 2 3\nunits bohr\nsymmetry c1\nno_com\nno_reorient")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    functional, _ = psi4.driver.dft.build_superfunctional("pbe", True)
    potential = psi4.core.VBase.build(basis, functional, "RV")
    with pytest.raises(RuntimeError, match="initialized"):
        potential.cuest_grid_points()
    potential.initialize()
    try:
        npoints = potential.cuest_grid_points()
        coords_matrix, weights_matrix = potential.export_cuest_grid()
        coords = np.asarray(coords_matrix)
        weights = np.asarray(weights_matrix).reshape(-1)
        assert npoints > 0
        assert coords.shape == (npoints, 3)
        assert weights.shape == (npoints,)
        assert np.isfinite(coords).all() and np.isfinite(weights).all()
        radius2 = ((coords - [1, 2, 3]) ** 2).sum(axis=1)
        assert np.dot(weights, np.exp(-radius2)) == pytest.approx(np.pi ** 1.5, abs=1e-6)
        # Export is a read-only diagnostic, not a replacement for the live mesh.
        coords_copy = coords.copy()
        coords[0, 0] += 100
        again, again_weights = potential.export_cuest_grid()
        np.testing.assert_array_equal(np.asarray(again), coords_copy)
        np.testing.assert_array_equal(np.asarray(again_weights).reshape(-1), weights)
    finally:
        potential.finalize()
    # Owned matrices remain usable after the live GPU plan/grid have been freed.
    assert np.isfinite(np.asarray(coords_matrix)).all()
    with pytest.raises(RuntimeError, match="initialized"):
        potential.export_cuest_grid()


@uusing("cuest")
@uusing("cuda_cc8")
def test_grid_export_preserves_xc_integration(monkeypatch):
    """Export from a live SCF plan and integrate XC again without rebuilding it."""
    psi4.set_options({"basis": "cc-pvdz", "use_cuest": True, "cuest_xc": True,
                      "scf_type": "df", "df_basis_scf": "def2-universal-jkfit"})
    mol = psi4.geometry("He 0 0 0\nunits bohr\nsymmetry c1\nno_com\nno_reorient")
    original = psi4.core.HF.iterations
    checked = []

    def audited_iterations(wfn, *args, **kwargs):
        answer = original(wfn, *args, **kwargs)
        potential = wfn.V_potential()
        baseline = psi4.core.Matrix("V before export", wfn.basisset().nbf(), wfn.basisset().nbf())
        after = baseline.clone()
        potential.compute_V([baseline])
        potential.export_cuest_grid()
        potential.compute_V([after])
        np.testing.assert_allclose(np.asarray(after), np.asarray(baseline), atol=1e-12, rtol=0)
        checked.append(True)
        return answer

    monkeypatch.setattr(psi4.core.HF, "iterations", audited_iterations)
    energy = psi4.energy("pbe", molecule=mol)
    assert np.isfinite(energy)
    assert checked
