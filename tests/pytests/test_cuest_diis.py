"""Device DIIS against independent host histories and complete SCFs."""
import numpy as np
import pytest
import psi4

from addons import using
from psi4.driver.procrouting.diis import DIIS, StoragePolicy
from psi4.driver.procrouting.cuest_diis import CuestDIIS

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cuest,
              *using("cuest"), *using("cuda_cc8")]


@pytest.fixture(autouse=True)
def clean_diis_options():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_num_threads(2)
    yield
    psi4.core.clean()
    psi4.core.clean_options()


def matrices(arrays):
    return [psi4.core.Matrix.from_array(a) for a in arrays]


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("engines", [{"diis"}, {"adiis"}, {"ediis"},
                                    {"diis", "adiis"}, {"diis", "ediis"}])
def test_cuest_diis_history(nspin, engines):
    host = DIIS(3, "host parity", storage_policy=StoragePolicy.InCore,
                closed_shell=nspin == 1, engines=engines)
    gpu = CuestDIIS(3, "device parity", closed_shell=nspin == 1, engines=engines)
    rng = np.random.default_rng(50023)
    for step in range(7):  # Full, replacement, and oldest-added wraparound.
        entry = {"target": matrices([rng.normal(size=(8, 8)) for _ in range(nspin)])}
        if "diis" in engines:
            entry["error"] = matrices([rng.normal(size=(5, 5)) * (step+1) for _ in range(nspin)])
        if engines & {"adiis", "ediis"}:
            entry["densities"] = matrices([rng.normal(size=(8, 8)) for _ in range(nspin)])
        if "ediis" in engines:
            entry["energy"] = [-1.0 - step * 0.01]
        host.add_entry(entry)
        gpu.add_entry(entry)
        if step == 0:
            allocation_calls = gpu.backend.stats()["allocation_calls"]
        assert gpu.backend.stats()["allocation_calls"] == allocation_calls
        # Original host objects may be reused immediately; GPU history is owned.
        for key in ("target", "error", "densities"):
            for value in entry.get(key, []):
                value.np[:] = 123.
        n = len(host.stored_vectors)
        assert gpu.backend.size() == n
        if "diis" in engines:
            expected = [[host.get_dot_product(i, j) for j in range(n)] for i in range(n)]
            np.testing.assert_allclose(np.asarray(gpu.backend.error_gram()).reshape(n, n),
                                       expected, atol=1.e-10, rtol=1.e-12)
            np.testing.assert_allclose(gpu.diis_coefficients(), host.diis_coefficients(),
                                       atol=1.e-10, rtol=1.e-10)
        if "adiis" in engines:
            host.adiis_populate()
            gpu.adiis_populate()
            np.testing.assert_allclose(gpu.adiis_linear, host.adiis_linear, atol=1.e-11)
            np.testing.assert_allclose(gpu.adiis_quadratic, host.adiis_quadratic, atol=1.e-11)
        if "ediis" in engines:
            host.ediis_populate()
            gpu.ediis_populate()
            np.testing.assert_allclose(gpu.ediis_quadratic, host.ediis_quadratic, atol=1.e-11)
        weights = np.arange(1., n+1)
        weights /= weights.sum()
        output = matrices([np.zeros((8, 8)) for _ in range(nspin)])
        gpu.backend.extrapolate(list(weights), output)
        for spin, result in enumerate(output):
            expected = sum(weights[i] * host.load_quantity("target", i, spin, False).np for i in range(n))
            np.testing.assert_allclose(result.np, expected, atol=1.e-12)
    gpu.reset_subspace()
    assert gpu.backend.size() == 0 and not gpu.stored_vectors
    gpu.delete_diis_file()
    gpu.delete_diis_file()  # Idempotent and never opens/deletes a PSIO file.
    assert gpu.stats_record["allocated_device_bytes"] == 0


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("reduced", [False, True])
def test_cuest_diis_projected_error(nspin, reduced):
    n, r = 9, (6 if reduced else 9)
    rng = np.random.default_rng(505)
    x = rng.normal(size=(n, r)) / n
    s = rng.normal(size=(n, n)); s = s @ s.T
    gpu = psi4.core.CuestDIIS(4, 32, True, False, False)
    errors = []
    smat, xmat = matrices([s, x])
    for _ in range(3):
        f = [rng.normal(size=(n, n)) for _ in range(nspin)]
        d = [rng.normal(size=(n, n)) for _ in range(nspin)]
        f = [(a+a.T)/2 for a in f]; d = [(a+a.T)/2 for a in d]
        current = [x.T @ (a @ b @ s - (a @ b @ s).T) @ x for a, b in zip(f, d)]
        error = np.concatenate([a.ravel() for a in current])
        expected_rms = np.sqrt(np.dot(error, error) / error.size)
        expected_max = np.max(np.abs(error))
        fm, dm = matrices(f), matrices(d)
        assert gpu.scf_update(fm, dm, smat, xmat, True, True) == pytest.approx(expected_rms, abs=1.e-10)
        assert gpu.scf_update(fm, dm, smat, xmat, False, False) == pytest.approx(expected_max, abs=1.e-10)
        errors.append(error)
        expected = np.asarray(errors) @ np.asarray(errors).T
        np.testing.assert_allclose(np.asarray(gpu.error_gram()).reshape(len(errors), -1),
                                   expected, atol=1.e-9, rtol=1.e-12)
        for ref, actual in zip(current, gpu._error_vectors()):
            np.testing.assert_allclose(actual.np, ref, atol=1.e-10, rtol=1.e-12)
    stats = gpu.stats()
    assert stats["fixed_input_uploads"] == 1
    # Only the explicitly requested diagnostics downloaded full errors.
    assert stats["matrix_device_to_host_bytes"] == stats["error_matrix_device_to_host_bytes"] == 3*nspin*r*r*8
    gpu.reset_subspace()
    assert gpu.size() == 0 and gpu.stats()["allocated_device_bytes"] == 0


@pytest.mark.parametrize("zero", [False, True])
def test_cuest_diis_singular_errors(zero):
    host = DIIS(4, "singular host", storage_policy=StoragePolicy.InCore)
    gpu = CuestDIIS(4, "singular device")
    for i in range(4):
        error = np.zeros((4, 4)) if zero else np.ones((4, 4))
        entry = {"target": matrices([np.eye(4) * (i+1)]), "error": matrices([error])}
        host.add_entry(entry); gpu.add_entry(entry)
    coefficients = gpu.diis_coefficients()
    assert np.all(np.isfinite(coefficients))
    np.testing.assert_allclose(coefficients, host.diis_coefficients(), atol=1.e-12)
    output = matrices([np.zeros((4, 4))])
    gpu.extrapolate(*output, Dnorm=0.)
    assert np.all(np.isfinite(output[0].np))


def test_cuest_diis_memory_and_cleanup():
    psi4.set_options({"cuest_diis_max_memory_mb": 1})
    gpu = CuestDIIS(10, "bounded history", engines={"diis", "adiis"}, closed_shell=False)
    entry = {key: matrices([np.eye(64), np.eye(64)]) for key in ("error", "target", "densities")}
    gpu.add_entry(entry)
    assert 2 <= gpu.backend.capacity() < 10
    assert gpu.backend.stats()["allocated_device_bytes"] <= 1000000
    gpu.delete_diis_file()
    assert gpu.backend.stats()["allocated_device_bytes"] == 0
    with pytest.raises(psi4.ValidationError, match="history has been deleted"):
        gpu.add_entry(entry)
    too_large = psi4.core.CuestDIIS(10, 1, True, True, False)
    large = matrices([np.eye(256), np.eye(256)])
    with pytest.raises(RuntimeError, match="cannot hold its fixed workspace"):
        too_large.add_entry(large, large, large)
    too_large.clear()


def test_cuest_diis_shape_and_nonfinite_guards():
    gpu = psi4.core.CuestDIIS(3, 32, True, False, False)
    target = matrices([np.eye(4)])
    with pytest.raises(RuntimeError, match="compatible dense C1"):
        gpu.add_entry(matrices([np.ones((3, 2))]), target, [])
    assert gpu.size() == 0
    gpu.add_entry(matrices([np.eye(3)]), target, [])
    with pytest.raises(RuntimeError, match="shape changed"):
        gpu.add_entry(matrices([np.eye(4)]), target, [])
    with pytest.raises(RuntimeError, match="nonfinite"):
        gpu.add_entry(matrices([np.full((3, 3), np.nan)]), target, [])
    assert gpu.size() == 1
    with pytest.raises(RuntimeError, match="invalid extrapolation"):
        gpu.extrapolate([np.nan], target)
    gpu.reset_subspace()
    gpu.add_entry(matrices([np.eye(4)]), target, [])
    assert gpu.size() == 1
    missing_energy = CuestDIIS(3, "energy guard", engines={"ediis"})
    with pytest.raises(psi4.ValidationError, match="finite energy"):
        missing_energy.add_entry({"target": target, "densities": target})
    assert missing_energy.backend.size() == 0


@pytest.mark.parametrize("filled", [1, 2])
@pytest.mark.parametrize("poison", ["target", "densities"])
def test_cuest_diis_history_gram_rejection_recovery(filled, poison):
    gpu = CuestDIIS(2, "reject bad Gram", engines={"adiis"})
    for _ in range(filled):
        gpu.add_entry({key: matrices([np.eye(8)]) for key in ("target", "densities")})
    before = gpu.backend.stats()["allocated_device_bytes"]
    bad = {key: matrices([np.eye(8)]) for key in ("target", "densities")}
    bad[poison][0].np[:] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite Gram"):
        gpu.add_entry(bad)
    # Explicit invalidation is intentional: neither layer may retain a
    # partially overwritten or poisoned subspace after rejecting an entry.
    assert gpu.backend.size() == 0 and not gpu.stored_vectors
    assert gpu.backend.stats()["allocated_device_bytes"] == 0
    gpu.add_entry({key: matrices([np.eye(4)]) for key in ("target", "densities")})
    assert gpu.backend.size() == len(gpu.stored_vectors) == 1
    assert gpu.backend.stats()["allocated_device_bytes"] < before
    np.testing.assert_allclose(gpu.backend.density_fock_gram(), [4.], atol=1.e-12)


def test_cuest_diis_partial_allocation_failure_recovery():
    gpu = psi4.core.CuestDIIS(4, 32, True, True, False)
    gpu._test_fail_allocation_after(3)
    large = matrices([np.eye(64), np.eye(64)])
    with pytest.raises(RuntimeError, match="injected allocation failure"):
        gpu.add_entry(large, large, large)
    stats = gpu.stats()
    assert stats["allocation_calls"] == 3 and stats["peak_allocated_device_bytes"] > 0
    assert stats["allocated_device_bytes"] == 0 and gpu.size() == 0
    gpu._test_fail_allocation_after(-1)
    small = matrices([np.eye(4), np.eye(4)])
    gpu.add_entry(small, small, small)
    assert gpu.size() == 1
    assert gpu.stats()["allocated_device_bytes"] < stats["peak_allocated_device_bytes"]
    gpu.clear()
    assert gpu.stats()["allocated_device_bytes"] == 0


def scf_case(method, unrestricted, gridk, gpu_diis, initial="NONE", maxiter=100, extra_options=None):
    psi4.core.clean()
    psi4.core.clean_options()
    mol = psi4.geometry(f"""
        {1 if unrestricted else 0} {2 if unrestricted else 1}
        O
        H 1 0.96
        H 1 0.96 2 104.5
        symmetry c1
    """)
    psi4.set_options({
        "basis": "cc-pvdz", "df_basis_scf": "cc-pvdz-jkfit",
        "reference": ("uhf" if unrestricted else "rhf") if method == "hf" else
                     ("uks" if unrestricted else "rks"),
        "scf_type": "df", "guess": "sad", "use_cuest": True, "cuest_xc": True,
        "cuest_directj": gridk, "cuest_gridk": gridk, "cuest_gridk_workspace_mb": 64,
        "cuest_diis": gpu_diis, "scf_initial_accelerator": initial,
        "diis_max_vecs": 4, "diis_start": 3,
        "cuest_mixed_precision": False, "ints_tolerance": 0.0,
        "dft_radial_points": 35, "dft_spherical_points": 110, "dft_pruning_scheme": "none",
        "e_convergence": 1.e-10, "d_convergence": 1.e-9, "maxiter": maxiter,
    })
    if extra_options:
        psi4.set_options(extra_options)
    return psi4.energy(method, molecule=mol, return_wfn=True)


def test_cuest_diis_actual_linear_dependence():
    options = {"s_orthogonalization": "canonical", "s_tolerance": 0.1}
    expected_energy, expected = scf_case("hf", False, False, False, extra_options=options)
    energy, wfn = scf_case("hf", False, False, True, extra_options=options)
    assert wfn.nmo() < wfn.basisset().nbf()
    assert energy == pytest.approx(expected_energy, abs=1.e-8)
    np.testing.assert_allclose(wfn.Da().np, expected.Da().np, atol=3.e-7, rtol=0)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("gridk", [False, True])
@pytest.mark.parametrize("initial", ["NONE", "ADIIS", "EDIIS"])
def test_cuest_diis_scf(unrestricted, gridk, initial):
    expected_energy, expected = scf_case("pbe0", unrestricted, gridk, False, initial)
    energy, wfn = scf_case("pbe0", unrestricted, gridk, True, initial)
    assert energy == pytest.approx(expected_energy, abs=1.e-8)
    np.testing.assert_allclose(wfn.Da().np, expected.Da().np, atol=3.e-7, rtol=0)
    np.testing.assert_allclose(wfn.Db().np, expected.Db().np, atol=3.e-7, rtol=0)
    stats = wfn.cuest_diis_stats()
    assert stats["entries_added"] > 0 and stats["extrapolations"] > 0
    assert stats["device_bytes_retained_after_cleanup"] == stats["allocated_device_bytes"] == 0
    assert stats["peak_allocated_device_bytes"] <= stats["memory_limit_bytes"]
    n = wfn.basisset().nbf()
    assert stats["matrix_device_to_host_bytes"] == stats["extrapolations"] * n*n*(2 if unrestricted else 1)*8
    assert stats["matrix_host_to_device_bytes"] == (
        stats["fixed_input_host_to_device_bytes"] + stats["gradient_updates"] * n*n*(2 if unrestricted else 1)*16)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cuest_diis_hf_and_failure_cleanup(unrestricted):
    energy, _ = scf_case("hf", unrestricted, False, False)
    actual, wfn = scf_case("hf", unrestricted, False, True)
    assert actual == pytest.approx(energy, abs=1.e-8)
    assert wfn.cuest_diis_stats()["allocated_device_bytes"] == 0
    with pytest.raises(psi4.SCFConvergenceError) as caught:
        scf_case("hf", unrestricted, True, True, maxiter=4)
    failed = caught.value.wfn
    assert failed.cuest_diis_stats()["device_bytes_retained_after_cleanup"] == 0
