import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

SHELL_ARRAYS = ("MBIS SHELL COUNTS", "MBIS SHELL POPULATIONS", "MBIS SHELL WIDTHS")
GEOMETRIES = {
    "water": "0 1\nO\nH 1 1.0\nH 1 1.0 2 104.5",
    "oxygen": "0 3\nO 0 0 0",
    "ghost": "0 1\nHe 0 0 0\n@He 0 0 3",
}
MBIS_OPTIONS = {
    "mbis_radial_points": 99,
    "mbis_spherical_points": 350,
    "mbis_pruning_scheme": "robust",
    "mbis_d_convergence": 1e-8,
}


@pytest.fixture(scope="module")
def scf_cache():
    return {}


def _scf(species):
    mol = psi4.geometry(GEOMETRIES[species] + "\nsymmetry c1\nno_reorient\nno_com")
    psi4.set_options({
        "basis": "cc-pvdz",
        "reference": "uhf" if species == "oxygen" else "rhf",
        "scf_type": "df",
        "e_convergence": 1e-10,
        "d_convergence": 1e-8,
    })
    wfn = psi4.energy("hf", molecule=mol, return_wfn=True)[1]
    return wfn, wfn.Da().clone(), wfn.Db().clone(), mol.molecular_charge()


def _restore(wfn, Da, Db, charge):
    # Tests mutate the shared wavefunction; put back everything MBIS reads or writes.
    wfn.Da().copy(Da)
    wfn.Db().copy(Db)
    wfn.molecule().set_molecular_charge(charge)
    for name in list(wfn.scalar_variables()):
        if name.startswith("MBIS"):
            wfn.del_scalar_variable(name)
    for name in list(wfn.array_variables()):
        if name.startswith("MBIS"):
            wfn.del_array_variable(name)


@pytest.fixture
def mbis_wfn(request, scf_cache):
    """One SCF per species per module; each test gets restored density, charge and no MBIS state."""
    species = getattr(request, "param", "water")
    if species not in scf_cache:
        scf_cache[species] = _scf(species)
    wfn, *saved = scf_cache[species]
    _restore(wfn, *saved)
    psi4.set_options(MBIS_OPTIONS)
    yield wfn
    _restore(wfn, *saved)


def assert_invalid_snapshot(wfn):
    assert wfn.scalar_variable("MBIS CONVERGED") == 0
    for name in SHELL_ARRAYS:
        assert not wfn.has_array_variable(name)


@pytest.mark.parametrize("mbis_wfn", ["water", "oxygen"], indirect=True)
def test_mbis_shell_snapshot(mbis_wfn):
    wfn = mbis_wfn
    psi4.oeprop(wfn, "MBIS_CHARGES")
    counts, populations, widths = [wfn.array_variable(name).np for name in SHELL_ARRAYS]
    mol = wfn.molecule()
    nat = mol.natom()
    assert counts.shape == (nat, 1)
    assert populations.shape == widths.shape == (nat, 7)
    np.testing.assert_array_equal(counts[:, 0], [2, 1, 1] if nat == 3 else [2])
    for a, count in enumerate(counts[:, 0].astype(int)):
        for array in (populations, widths):
            assert np.all(np.isfinite(array[a, :count]))
            assert np.all(array[a, :count] > 0)
            np.testing.assert_array_equal(array[a, count:], 0)
        assert widths[a, count - 1] == wfn.array_variable("MBIS VALENCE WIDTHS").np[a, 0]
        assert -populations[a, count - 1] == wfn.array_variable("MBIS VALENCE CHARGES").np[a, 0]
    assert wfn.scalar_variable("MBIS CONVERGED") == 1
    assert 0 < wfn.scalar_variable("MBIS ITERATIONS") < 500
    assert 0 <= wfn.scalar_variable("MBIS DENSITY RESIDUAL") < 1e-8
    np.testing.assert_allclose(populations.sum(), wfn.scalar_variable("MBIS GRID ELECTRONS"), atol=1e-10)
    charges = wfn.array_variable("MBIS CHARGES").np
    np.testing.assert_allclose(
        populations.sum(axis=1), [mol.Z(a) - charges[a, 0] for a in range(nat)], atol=1e-6, rtol=0
    )

    # Independently sample the exported model, not a second implementation of its iteration.
    grid = psi4.core.DFTGrid.build(
        mol, wfn.basisset(),
        {"DFT_RADIAL_POINTS": 99, "DFT_SPHERICAL_POINTS": 350},
        {"DFT_PRUNING_SCHEME": "ROBUST"},
    )
    points = psi4.core.UKSFunctions(wfn.basisset(), grid.max_points(), grid.max_functions())
    points.set_pointers(wfn.Da_subset("AO"), wfn.Db_subset("AO"))
    reconstructed_pop = np.zeros(nat)
    reconstructed_dipoles = np.zeros((nat, 3))
    electrons = 0.0
    for block in grid.blocks():
        points.compute_points(block)
        n = block.npoints()
        rho = points.point_values()["RHO_A"].np[:n] + points.point_values()["RHO_B"].np[:n]
        xyz = np.column_stack((block.x().np, block.y().np, block.z().np))
        displacement = xyz[None, :, :] - mol.geometry().np[:, None, :]
        distance = np.linalg.norm(displacement, axis=2)
        proatoms = np.zeros((nat, n))
        for a, count in enumerate(counts[:, 0].astype(int)):
            sigma = widths[a, :count, None]
            proatoms[a] = np.sum(
                populations[a, :count, None] * np.exp(-distance[a] / sigma) / (8 * np.pi * sigma**3), axis=0
            )
        weights = proatoms / proatoms.sum(axis=0)
        np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-14, rtol=0)
        atomic_electrons = weights * (block.w().np * rho)
        electrons += np.dot(block.w().np, rho)
        reconstructed_pop += atomic_electrons.sum(axis=1)
        reconstructed_dipoles -= np.einsum("ap,apx->ax", atomic_electrons, displacement)
    np.testing.assert_allclose(electrons, wfn.scalar_variable("MBIS GRID ELECTRONS"), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        np.array([mol.Z(a) for a in range(nat)]) - reconstructed_pop, charges[:, 0], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        reconstructed_dipoles, wfn.array_variable("MBIS DIPOLES").np, atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("maxiter", [0, 1, 2])
def test_mbis_failed_retry(mbis_wfn, maxiter):
    wfn = mbis_wfn
    psi4.oeprop(wfn, "MBIS_CHARGES")
    saved = wfn.array_variable("MBIS SHELL POPULATIONS").clone()
    psi4.set_options({"mbis_maxiter": maxiter})
    with pytest.raises(RuntimeError, match="MBIS"):
        psi4.oeprop(wfn, "MBIS_CHARGES")
    assert_invalid_snapshot(wfn)
    assert wfn.scalar_variable("MBIS ITERATIONS") == max(0, maxiter - 1)
    residual = wfn.scalar_variable("MBIS DENSITY RESIDUAL")
    if maxiter <= 1:
        assert np.isposinf(residual)
    else:
        assert np.isfinite(residual) and residual > 1e-8
    assert wfn.has_scalar_variable("MBIS GRID ELECTRONS")
    # A caller-owned copy survives invalidation; successful retry publishes a fresh model.
    psi4.set_options({"mbis_maxiter": 500})
    psi4.oeprop(wfn, "MBIS_CHARGES")
    np.testing.assert_array_equal(saved.np, wfn.array_variable("MBIS SHELL POPULATIONS").np)
    assert wfn.scalar_variable("MBIS CONVERGED") == 1


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_mbis_invalid_threshold(mbis_wfn, threshold):
    psi4.oeprop(mbis_wfn, "MBIS_CHARGES")
    psi4.set_options({"mbis_d_convergence": threshold})
    with pytest.raises(RuntimeError, match="MBIS_D_CONVERGENCE must be finite and positive"):
        psi4.oeprop(mbis_wfn, "MBIS_CHARGES")
    assert_invalid_snapshot(mbis_wfn)
    assert mbis_wfn.scalar_variable("MBIS ITERATIONS") == 0
    assert np.isposinf(mbis_wfn.scalar_variable("MBIS DENSITY RESIDUAL"))
    assert not mbis_wfn.has_scalar_variable("MBIS GRID ELECTRONS")


@pytest.mark.parametrize("bad_density", ["nonfinite", "zero"])
def test_mbis_invalid_density(mbis_wfn, bad_density):
    wfn = mbis_wfn
    psi4.oeprop(wfn, "MBIS_CHARGES")
    if bad_density == "nonfinite":
        wfn.Da().np[:] = np.nan
        message = "MBIS grid electron count is nonfinite"
    else:
        # Zero-electron density passes the grid count check but gives 0/0 shell widths.
        wfn.Da().zero()
        wfn.Db().zero()
        wfn.molecule().set_molecular_charge(10)
        message = "MBIS invalid shell update"
    with pytest.raises(RuntimeError, match=message):
        psi4.oeprop(wfn, "MBIS_CHARGES")
    assert_invalid_snapshot(wfn)
    assert np.isposinf(wfn.scalar_variable("MBIS DENSITY RESIDUAL"))
    assert wfn.scalar_variable("MBIS ITERATIONS") == (1 if bad_density == "zero" else 0)


@pytest.mark.parametrize("free_volume", [None, float("inf"), 0.0, -1.0, float("nan")])
def test_mbis_postprocessing_failure(mbis_wfn, free_volume):
    wfn = mbis_wfn
    psi4.oeprop(wfn, "MBIS_CHARGES")
    volumes = wfn.array_variable("MBIS RADIAL MOMENTS <R^3>").np[:, 0]
    mol = wfn.molecule()
    # Bypass the driver's free-atom preparation so native postprocessing must fail.
    if free_volume is None:
        message = "MBIS FREE ATOM"
    else:
        wfn.set_variable("MBIS FREE ATOM H VOLUME", volumes[1])
        wfn.set_variable("MBIS FREE ATOM O VOLUME", free_volume)
        message = "MBIS FREE ATOM O VOLUME must be finite and positive"
    prop = psi4.core.OEProp(wfn)
    prop.add("MBIS_VOLUME_RATIOS")
    with pytest.raises(RuntimeError, match=message):
        prop.compute()
    assert_invalid_snapshot(wfn)
    assert wfn.scalar_variable("MBIS DENSITY RESIDUAL") < 1e-8

    # The molecule's own <r^3> as free volumes makes every ratio exactly one.
    assert mol.label(1) == mol.label(2) == "H"
    wfn.set_variable("MBIS FREE ATOM O VOLUME", volumes[0])
    wfn.set_variable("MBIS FREE ATOM H VOLUME", volumes[1])
    prop = psi4.core.OEProp(wfn)
    prop.add("MBIS_VOLUME_RATIOS")
    prop.compute()
    assert wfn.scalar_variable("MBIS CONVERGED") == 1
    assert all(wfn.has_array_variable(name) for name in SHELL_ARRAYS)
    np.testing.assert_allclose(
        wfn.array_variable("MBIS VOLUME RATIOS").np[:, 0], volumes / volumes[[0, 1, 1]], atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("mbis_wfn", ["ghost"], indirect=True)
def test_mbis_ghost_rejected(mbis_wfn):
    with pytest.raises(RuntimeError, match="MBIS does not support ghost"):
        psi4.oeprop(mbis_wfn, "MBIS_CHARGES")
    assert_invalid_snapshot(mbis_wfn)
