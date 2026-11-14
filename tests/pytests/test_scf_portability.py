"""SCF memory, LAPACK, and threading regressions shared by the SCF branches."""
from types import SimpleNamespace
import os

import numpy as np
import pytest

import psi4
from psi4.driver.procrouting.scf_proc import scf_iterator
from psi4.driver.p4util.exceptions import ValidationError

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("remaining", [-1, 0])
def test_diagonalization_memory_exhausted(monkeypatch, remaining):
    """Reject impossible budgets before constructing a JK or a grid cache."""
    monkeypatch.setattr(psi4.core, "get_memory", lambda: 8000)
    get_global_option = psi4.core.get_global_option
    monkeypatch.setattr(
        psi4.core, "get_global_option",
        lambda key: 1.0 if key == "SCF_MEM_SAFETY_FACTOR" else get_global_option(key),
    )
    wfn = SimpleNamespace(diag_scratch_size=lambda: 1000 - remaining)
    with pytest.raises(ValidationError, match="diagonalization workspace"):
        scf_iterator.scf_initialize(wfn)


@pytest.mark.parametrize("degenerate", [False, True])
def test_dsyevd_small_degenerate_and_triangle(degenerate):
    """Do not compare individual eigenvectors inside a degenerate eigenspace."""
    dimensions = psi4.core.Dimension([0, 1, 2, 7, 33])
    a = psi4.core.Matrix("A", dimensions, dimensions)
    rng = np.random.default_rng(894)
    expected = []
    for h, n in enumerate(dimensions.to_tuple()):
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        values = np.ones(n) if degenerate else np.arange(n, dtype=float)
        symmetric = (q * values) @ q.T
        # diagonalize() reads the row-major upper triangle, not the lower.
        a.nph[h][:] = np.triu(symmetric) + np.tril(np.full((n, n), 12345.0), -1)
        expected.append(symmetric)
    original = a.clone()
    c = psi4.core.Matrix("C", dimensions, dimensions)
    e = psi4.core.Vector("e", dimensions)
    a.diagonalize_dsyevd(c, e)
    for h, n in enumerate(dimensions.to_tuple()):
        if not n:
            continue
        assert np.array_equal(a.nph[h], original.nph[h])
        np.testing.assert_allclose(c.nph[h].T @ c.nph[h], np.eye(n), atol=2e-13)
        np.testing.assert_allclose(expected[h] @ c.nph[h], c.nph[h] * e.nph[h], atol=2e-12)


def _bases():
    mol = psi4.geometry("O 0 0 0\nH 0 0 1\nH 1 0 0\nsymmetry c1\n")
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pvdz")
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")
    return primary, aux


@pytest.mark.parametrize("wk", [False, True])
@pytest.mark.parametrize("threads", [1, 2])
def test_memdf_scratch_lifecycle(wk, threads):
    """Charge retained scratch for every build, including separate wK."""
    psi4.set_options({"scf_type": "mem_df", "scf_subtype": "incore"})
    primary, aux = _bases()
    jk = psi4.core.JK.build_JK(primary, aux)
    jk.set_omp_nthread(threads)
    jk.set_do_wK(wk)
    jk.set_omega(0.3)
    jk.set_wcombine(False)
    minimum = jk.memory_estimate()
    jk.set_memory(minimum)
    jk.initialize()
    dfh = jk.dfh()
    nbf = primary.nbf()
    rng = np.random.default_rng(895)
    left, right = rng.normal(size=(2, nbf, nbf)) / nbf
    for nocc, symmetric in [(1, True), (nbf, True), (nbf, False), (1, True)]:
        jk.C_clear()
        jk.C_left_add(psi4.core.Matrix.from_array(left[:, :nocc].copy()))
        if not symmetric:
            jk.C_right_add(psi4.core.Matrix.from_array(right[:, :nocc].copy()))
        jk.compute()
        resident = dfh.get_AO_size() * (3 if wk else 1)
        assert resident + dfh.JK_scratch_memory() <= dfh.get_memory()
    jk.release_scratch()
    assert dfh.JK_scratch_memory() == 0
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(left[:, :1].copy()))
    jk.compute()
    assert dfh.JK_scratch_memory() > 0
    jk.finalize()
    assert dfh.JK_scratch_memory() == 0


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_memdf_minimum_memory_boundary(delta):
    psi4.set_options({"scf_type": "mem_df", "scf_subtype": "incore"})
    mol = psi4.geometry("He 0 0 0\nsymmetry c1\n")
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    jk = psi4.core.JK.build_JK(basis, basis)
    minimum = jk.memory_estimate()
    jk.set_memory(minimum + delta)
    if delta < 0:
        with pytest.raises(RuntimeError, match="not enough memory"):
            jk.initialize()
        return
    jk.initialize()
    jk.C_left_add(psi4.core.Matrix.from_array(np.ones((1, 1))))
    jk.compute()
    assert jk.K()[0].np[0, 0] > 0
    assert jk.dfh().get_AO_size() + jk.dfh().JK_scratch_memory() <= jk.dfh().get_memory()
    jk.finalize()


@pytest.mark.parametrize("method", ["svwn", "pbe", "tpss", "wb97m-v"])
def test_xc_threaded_matrices_and_gradients(method):
    """Row-tile locks preserve the potential and its gradient, not just energy."""
    old_threads = psi4.core.get_num_threads()
    psi4.geometry("O 0 0 0\nH 0 0 1\nH 1 0 0\nsymmetry c1\n")
    psi4.set_options({
        "basis": "def2-svp", "scf_type": "mem_df",
        "e_convergence": 1e-10, "d_convergence": 1e-10,
        "dft_radial_points": 50, "dft_spherical_points": 110,
        "dft_vv10_radial_points": 20, "dft_vv10_spherical_points": 50,
    })
    results = []
    try:
        for threads in (1, 2, 4):
            psi4.set_num_threads(threads)
            energy, wfn = psi4.energy(method, return_wfn=True)
            density, fock = wfn.Da().np.copy(), wfn.Fa().np.copy()
            # VV10 may use finite-difference gradients on this branch. That
            # driver's returned shell wavefunction has no SCF density/Fock.
            gradient = psi4.gradient(method)
            results.append((energy, density, fock, gradient.np.copy()))
            psi4.core.clean()
    finally:
        psi4.set_num_threads(old_threads)
    for actual in results[1:]:
        for item, (ref, value) in enumerate(zip(results[0], actual)):
            # VV10 uses 0.005-Bohr finite differences here; SCF energy noise is
            # divided by that displacement. Keep matrix/energy checks tighter.
            tolerance = 5e-7 if method == "wb97m-v" and item == 3 else 2e-8
            np.testing.assert_allclose(value, ref, atol=tolerance, rtol=0)


@pytest.mark.parametrize("exponent", ["1.0", "1.0000001"])
def test_cholesky_singular_metric_fallback(tmp_path, exponent):
    """Duplicate/near-duplicate auxiliary shells exercise the actual fallback."""
    mol = psi4.geometry("He 0 0 0\nsymmetry c1\n")
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g")
    psi4.basis_helper(
        f"assign scf_fallback\n[scf_fallback]\n"
        f"spherical\n****\nHe 0\nS 1 1.0\n1.0 1.0\nS 1 1.0\n{exponent} 1.0\n****",
        name="SCF_FALLBACK", key="DF_BASIS_SCF",
    )
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "SCF_FALLBACK", "JKFIT", "sto-3g")
    results = []
    out = tmp_path / "fallback.out"
    psi4.set_output_file(str(out), False)
    try:
        for metric in ("eigen", "cholesky"):
            psi4.set_options({"scf_type": "mem_df", "scf_df_metric": metric})
            jk = psi4.core.JK.build_JK(primary, aux)
            jk.set_memory(1_000_000)
            jk.initialize()
            jk.C_left_add(psi4.core.Matrix.from_array(np.ones((1, 1))))
            jk.compute()
            results.append((jk.J()[0].np.copy(), jk.K()[0].np.copy()))
            jk.finalize()
    finally:
        psi4.set_output_file("pytest_output.dat", True)
    assert "metric Cholesky not used" in out.read_text()
    for expected, actual in zip(*results):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("wk", [False, True])
def test_cholesky_matrix_equivalence(tmp_path, wk):
    """Also run under the LAPACK interposer to force each fallback separately."""
    primary, aux = _bases()
    coefficients = np.random.default_rng(897).normal(size=(primary.nbf(), 5)) / primary.nbf()
    results = []
    out = tmp_path / "cholesky-matrices.out"
    psi4.set_output_file(str(out), False)
    try:
        for metric in ("eigen", "cholesky"):
            psi4.set_options({"scf_type": "mem_df", "scf_df_metric": metric})
            jk = psi4.core.JK.build_JK(primary, aux)
            jk.set_do_wK(wk)
            jk.set_wcombine(False)
            jk.set_omega(0.3)
            jk.set_memory(2_000_000)
            jk.initialize()
            jk.C_left_add(psi4.core.Matrix.from_array(coefficients))
            jk.compute()
            results.append([jk.J()[0].np.copy(), jk.K()[0].np.copy()] +
                           ([jk.wK()[0].np.copy()] if wk else []))
            jk.finalize()
    finally:
        psi4.set_output_file("pytest_output.dat", True)
    if not wk and (os.environ.get("PSI_TEST_DPOTRF_FAIL") or os.environ.get("PSI_TEST_DPOCON_SMALL_RCOND")):
        assert "metric Cholesky not used" in out.read_text()
    for expected, actual in zip(*results):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("reference", ["rhf", "uhf"])
def test_cholesky_diffuse_density_and_gradient(reference):
    """Opt-in Cholesky preserves diffuse-basis SCF observables in both spins."""
    psi4.geometry(
        ("0 1\n" if reference == "rhf" else "1 2\n") +
        "O 0 0 0\nH 0 0 1\nH 1 0 0\nsymmetry c1\n"
    )
    psi4.set_options({
        "basis": "aug-cc-pvdz", "reference": reference, "scf_type": "mem_df",
        "e_convergence": 1e-10, "d_convergence": 1e-10,
    })
    results = []
    for metric in ("eigen", "cholesky"):
        psi4.set_options({"scf_df_metric": metric})
        g, wfn = psi4.gradient("hf", return_wfn=True)
        results.append((wfn.energy(), wfn.Da().np.copy(), wfn.Db().np.copy(),
                        wfn.Fa().np.copy(), wfn.Fb().np.copy(), g.np.copy()))
        psi4.core.clean()
    for expected, actual in zip(*results):
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)


def test_cholesky_gradient_finite_difference():
    mol = psi4.geometry("O 0 0 0\nH 0 0 1\nH 1 0 0\nsymmetry c1\nno_reorient\nno_com\n")
    psi4.set_options({
        "basis": "cc-pvdz", "scf_type": "mem_df", "scf_df_metric": "cholesky",
        "e_convergence": 1e-11, "d_convergence": 1e-11,
    })
    analytic = psi4.gradient("hf").np[1, 2]
    geometry = mol.geometry().np.copy()  # Bohr
    energies = []
    step = 1e-3
    for sign in (-1, 1):
        displaced = geometry.copy()
        displaced[1, 2] += sign * step
        mol.set_geometry(psi4.core.Matrix.from_array(displaced))
        mol.update_geometry()
        psi4.core.clean()
        energies.append(psi4.energy("hf", molecule=mol))
    mol.set_geometry(psi4.core.Matrix.from_array(geometry))
    mol.update_geometry()
    np.testing.assert_allclose((energies[1] - energies[0]) / (2 * step), analytic, atol=2e-6, rtol=0)


def test_rohf_dsyevd_level_shift():
    """Both ROHF diagonalization call sites reuse correctly sized workspaces."""
    psi4.geometry("1 2\nF 0 0 0\nH 0 0 0.92\n")
    psi4.set_options({
        "reference": "rohf", "basis": "cc-pvdz", "scf_type": "pk",
        "e_convergence": 1e-10, "d_convergence": 1e-10,
    })
    exact, ref = psi4.energy("hf", return_wfn=True)
    densities = [ref.Da().to_array(), ref.Db().to_array()]
    psi4.core.clean()
    psi4.set_options({"level_shift": 0.5, "level_shift_cutoff": 1e-2})
    energy, shifted = psi4.energy("hf", return_wfn=True)
    np.testing.assert_allclose(energy, exact, atol=1e-9, rtol=0)
    for expected, matrix in zip(densities, [shifted.Da(), shifted.Db()]):
        for a, b in zip(expected, matrix.to_array()):
            np.testing.assert_allclose(a, b, atol=2e-8, rtol=0)
