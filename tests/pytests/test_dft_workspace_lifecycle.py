import numpy as np
import psi4
import pytest

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def make_potential(method, unrestricted=False):
    psi4.set_num_threads(2)
    psi4.set_options({"dft_radial_points": 30, "dft_spherical_points": 110})
    molecule = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 103.1
        symmetry c1
    """)
    basis = psi4.core.BasisSet.build(molecule, "ORBITAL", "sto-3g")
    functional = psi4.driver.dft.build_superfunctional(method, not unrestricted)[0]
    potential = psi4.core.VBase.build(basis, functional, "UV" if unrestricted else "RV")
    potential.initialize()
    densities = [psi4.core.Matrix.from_array(0.5 * np.eye(basis.nbf()))]
    if unrestricted:
        densities.append(psi4.core.Matrix.from_array(0.4 * np.eye(basis.nbf())))
    potential.set_D(densities)
    return potential, densities


def potential_matrices(potential, densities, response=False):
    result = [density.clone() for density in densities]
    if response:
        potential.compute_Vx(densities, result)
    else:
        potential.compute_V(result)
    return [np.asarray(matrix).copy() for matrix in result]


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method", ["SVWN", "PBE", "TPSS"])
def test_dft_workspace_reuse(method, unrestricted):
    potential, densities = make_potential(method, unrestricted)
    try:
        grid = potential.grid()
        expected = potential_matrices(potential, densities)
        gradient = np.asarray(potential.compute_gradient()).copy()
        quadrature = dict(potential.quadrature_values())
        potential.release_workspaces()
        potential.release_workspaces()
        assert potential.grid() is grid
        assert dict(potential.quadrature_values()) == quadrature
        # Gradient is the first use after release, not compute_V.
        np.testing.assert_allclose(potential.compute_gradient(), gradient, rtol=1e-10, atol=1e-10)
        potential.release_workspaces()
        for actual, wanted in zip(potential_matrices(potential, densities), expected):
            np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-10)
        if method != "TPSS":
            response = potential_matrices(potential, densities, response=True)
            potential.release_workspaces()
            for actual, wanted in zip(potential_matrices(potential, densities, response=True), response):
                np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-10)
        if method == "SVWN":
            hessian = np.asarray(potential.compute_hessain()).copy()
            potential.release_workspaces()
            np.testing.assert_allclose(potential.compute_hessain(), hessian, rtol=1e-10, atol=1e-10)
    finally:
        potential.finalize()


def test_dft_exported_workers_and_cache():
    potential, densities = make_potential("PBE")
    grid = potential.grid()
    block = grid.blocks()[0]
    potential.build_collocation_cache(2_000_000)
    potential_matrices(potential, densities)
    worker = potential.properties()[0]
    worker.compute_points(block, False)
    potential.clear_collocation_cache()
    assert np.asarray(worker.basis_values()["PHI"]).shape[0] == grid.max_points()
    potential.build_collocation_cache(2_000_000)
    worker.compute_points(block, False)
    potential.release_workspaces()
    # Previously exported workers must not point into the erased cache.
    worker.compute_points(block, False)
    assert np.asarray(worker.basis_values()["PHI"]).shape[0] == grid.max_points()
    fresh = potential.properties()
    assert len(fresh) == 2
    fresh[0].compute_points(block)  # Density pointers are restored without set_pointers().
    np.testing.assert_allclose(
        np.asarray(fresh[0].point_values()["RHO_A"])[:block.npoints()],
        np.asarray(worker.point_values()["RHO_A"])[:block.npoints()],
    )
    potential.initialize()
    assert len(potential.properties()) == 2
    potential.finalize()
    worker.compute_points(block, False)
    potential.initialize()
    assert len(potential.properties()) == 2
    potential.build_collocation_cache(2_000_000)
    exported = potential.properties()[0]
    new_block = potential.grid().blocks()[0]
    exported.compute_points(new_block, False)
    del potential
    exported.compute_points(new_block, False)


def test_dft_vv10_workspace_reuse():
    potential, densities = make_potential("B97M-V")
    try:
        expected = potential_matrices(potential, densities)
        vv10 = potential.quadrature_values()["VV10"]
        potential.release_workspaces()
        for actual, wanted in zip(potential_matrices(potential, densities), expected):
            np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-10)
        assert potential.quadrature_values()["VV10"] == pytest.approx(vv10, abs=1e-12)
    finally:
        potential.finalize()


@pytest.mark.parametrize("grac_shift", [0.0, 0.1])
def test_returned_scf_workspace_reuse(grac_shift):
    psi4.set_num_threads(2)
    psi4.set_options({
        "basis": "sto-3g", "scf_type": "df", "scf_subtype": "auto",
        "reference": "rhf", "dft_grac_shift": grac_shift,
        "dft_radial_points": 30, "dft_spherical_points": 110,
        "e_convergence": 1e-10, "d_convergence": 1e-10,
    })
    molecule = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 103.1
        symmetry c1
    """)
    _, wfn = psi4.energy("pbe0", molecule=molecule, return_wfn=True)
    potential = wfn.V_potential()
    densities = [wfn.Da()]
    # Terminal SCF cleanup has already released the workers.
    expected = potential_matrices(potential, densities)
    assert potential.quadrature_values()["FUNCTIONAL"] == pytest.approx(
        wfn.variable("DFT XC ENERGY"), abs=1e-8)
    potential.release_workspaces()
    for actual, wanted in zip(potential_matrices(potential, densities), expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-10)
    if grac_shift == 0.0:
        response = potential_matrices(potential, densities, response=True)
        potential.release_workspaces()
        for actual, wanted in zip(potential_matrices(potential, densities, response=True), response):
            np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-10)
        gradient = np.asarray(psi4.gradient("pbe0", ref_wfn=wfn)).copy()
        potential.release_workspaces()
        np.testing.assert_allclose(psi4.gradient("pbe0", ref_wfn=wfn), gradient, rtol=1e-9, atol=1e-9)
