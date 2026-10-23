"""New resident solver against the independent existing host Hessian/CG."""
import numpy as np
import pytest
import psi4
from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cuest,
              *using("cuest"), *using("cuda_cc8")]


def _wavefunction(method="pbe0", unrestricted=False, directj=True, gridk=True,
                  gpu_xc=True, symmetry="c1", grac=False, atom=False, neutral_uhf=False, guess=None,
                  mixed_precision=False, ints_tolerance=0.0):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_num_threads(2)
    if atom:
        mol = psi4.geometry("0 2\nH 0 0 0\nsymmetry c1")
    else:
        mol = psi4.geometry(f"""
            {1 if unrestricted and not neutral_uhf else 0} {2 if unrestricted and not neutral_uhf else 1}
            O
            H 1 0.96
            H 1 0.96 2 104.5
            symmetry {symmetry}
        """)
    psi4.set_options({
        "basis": "cc-pvdz", "df_basis_scf": "cc-pvdz-jkfit",
        "reference": ("uhf" if unrestricted else "rhf") if method == "hf" else
                     ("uks" if unrestricted else "rks"),
        # Core converges the water cation to an unstable excited determinant.
        # The independent host -H spectrum is positive with SAD preparation.
        "scf_type": "df", "use_cuest": True, "save_jk": True,
        "guess": guess or ("sad" if unrestricted and not atom and not neutral_uhf else "core"),
        "cuest_directj": directj, "cuest_gridk": gridk, "cuest_xc": gpu_xc,
        "cuest_device_cg": False, "cuest_mixed_precision": mixed_precision, "cuest_gemm_min_dim": 0,
        "dft_radial_points": 35, "dft_spherical_points": 110,
        # Test the linear solver independently of density-dependent quartet
        # truncation. Benchmarks may separately use converged finite screening.
        "dft_pruning_scheme": "none", "ints_tolerance": ints_tolerance,
        "e_convergence": 1.e-11, "d_convergence": 1.e-10,
    })
    if grac:
        psi4.set_options({"dft_grac_shift": 0.1})
    try:
        return psi4.energy(method, molecule=mol, return_wfn=True)[1]
    except RuntimeError as exc:
        if "requires cuEST" in str(exc):
            pytest.skip("Requested cuEST backend unavailable in the linked SDK")
        raise


def _rhs(wfn, unrestricted, count=3, mixed=True):
    rng = np.random.default_rng(250501)
    rhs = []
    for r in range(count):
        for spin in (("a", "b") if unrestricted else ("a",)):
            sub = getattr(wfn, f"C{spin}_subset")
            shape = (sum(sub("SO", "OCC").coldim()), sum(sub("SO", "VIR").coldim()))
            values = rng.normal(size=shape) * 0.01
            if mixed and r == 1:
                values *= 0
            if mixed and r == count - 1 and count > 3:
                values *= 1.e-18
            rhs.append(psi4.core.Matrix.from_array(values))
    return rhs


def _device_solve(wfn, rhs, tol=1.e-9, maxiter=100):
    psi4.set_options({"cuest_device_cg": True})
    try:
        return wfn.cphf_solve(rhs, tol, maxiter, 0)
    except RuntimeError as exc:
        if "requires a Psi4 build with cuEST, CUDA NVRTC" in str(exc):
            pytest.skip("Device CG kernels unavailable in this build")
        raise


def _check(wfn, rhs, result, atol=1.e-9, relative_tol=1.e-9):
    assert wfn.cphf_converged()
    for b, ax in zip(rhs, wfn.cphf_Hx(result)):
        np.testing.assert_allclose(ax.np, b.np, atol=atol, rtol=0)
    stats = wfn.cphf_device_stats()
    assert stats["iteration_bulk_host_to_device_bytes"] == 0
    assert stats["iteration_bulk_device_to_host_bytes"] == 0
    assert stats["operator_calls"] > 0
    assert wfn.cphf_nfock_builds() == stats["operator_calls"] * stats["rhs_count"]
    assert stats["iteration_scalar_device_to_host_bytes"] <= (
        stats["operator_calls"] * (stats["rhs_count"] * 8 + 4) + 4
    )
    assert stats["final_max_relative_residual"] <= relative_tol


def test_cuest_device_cg_initial_status():
    wfn = _wavefunction("hf")
    assert wfn.cphf_nfock_builds() == 0
    assert not wfn.cphf_converged()
    assert wfn.cphf_device_stats() == {}


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cuest_response_profile_preserves_solution(unrestricted):
    wfn = _wavefunction("pbe0", unrestricted)
    rhs = _rhs(wfn, unrestricted, count=1, mixed=False)
    psi4.set_options({"cuest_response_profile": True})
    expected = [m.np.copy() for m in wfn.cphf_solve(rhs, 1.e-10, 150, 0)]
    assert wfn.cphf_converged()
    result = _device_solve(wfn, rhs)
    _check(wfn, rhs, result)
    for ref, out in zip(expected, result):
        np.testing.assert_allclose(out.np, ref, atol=1.e-8, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("directj,gridk", [(False, False), (True, True)])
def test_cuest_response_density_batch_size(batch_size, unrestricted, directj, gridk):
    """Chunking and partial batches change dispatch, never the response operator."""
    wfn = _wavefunction("pbe0", unrestricted, directj, gridk)
    rhs = _rhs(wfn, unrestricted, count=3, mixed=False)
    expected = [m.np.copy() for m in wfn.cphf_solve(rhs, 1.e-10, 150, 0)]
    assert wfn.cphf_converged()
    psi4.set_options({"cuest_response_batch_size": batch_size})
    result = _device_solve(wfn, rhs)
    _check(wfn, rhs, result)
    assert wfn.cphf_device_stats()["jk_density_batch_size"] == batch_size
    for ref, out in zip(expected, result):
        np.testing.assert_allclose(out.np, ref, atol=1.e-8, rtol=0)


def test_cuest_response_negative_batch_size():
    wfn = _wavefunction("hf")
    psi4.set_options({"cuest_response_batch_size": -1})
    with pytest.raises(RuntimeError, match="CUEST_RESPONSE_BATCH_SIZE must be nonnegative"):
        _device_solve(wfn, _rhs(wfn, False))
    assert not wfn.cphf_converged()


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method", ["hf", "pbe0", "cam-b3lyp", "lrc-wpbe"])
def test_cuest_response_symmetric_gridk_sum(method, unrestricted):
    """Signed symmetric density may replace general D only for K(D)+K(D)^T."""
    wfn = _wavefunction(method, unrestricted)
    rhs = _rhs(wfn, unrestricted, count=3, mixed=False)
    expected = [m.np.copy() for m in wfn.cphf_solve(rhs, 1.e-10, 150, 0)]
    assert wfn.cphf_converged()
    psi4.set_options({"cuest_response_symmetric_k": True})
    result = _device_solve(wfn, rhs)
    _check(wfn, rhs, result)
    assert wfn.cphf_device_stats()["symmetric_gridk"] == 1
    for ref, out in zip(expected, result):
        np.testing.assert_allclose(out.np, ref, atol=1.e-8, rtol=0)


def test_cuest_response_symmetric_gridk_preserves_general_jk():
    """A response-only shortcut cannot change a borrowed builder's general K."""
    wfn = _wavefunction("hf")
    rhs = _rhs(wfn, False, count=1, mixed=False)
    rng = np.random.default_rng(12)
    nbf = wfn.basisset().nbf()
    left = psi4.core.Matrix.from_array(rng.normal(size=(nbf, 3)) * .1)
    right = psi4.core.Matrix.from_array(-rng.normal(size=(nbf, 3)) * .1)
    jk = wfn.jk()
    def general_exchange():
        jk.C_clear()
        jk.C_left_add(left)
        jk.C_right_add(right)
        jk.compute()
        return jk.K()[0].np.copy()
    expected = general_exchange()
    assert np.max(np.abs(expected-expected.T)) > 1.e-4
    psi4.set_options({"cuest_response_symmetric_k": True})
    result = _device_solve(wfn, rhs)
    np.testing.assert_allclose(general_exchange(), expected, atol=1.e-11, rtol=0)
    _check(wfn, rhs, result)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_cuest_response_symmetric_gridk_chunked_edges(unrestricted, batch_size):
    """Exercise signed symmetric K with tiled, partial, cross-spin and inactive slots."""
    wfn = _wavefunction("pbe0", unrestricted)
    rhs = _rhs(wfn, unrestricted, count=11)
    psi4.set_options({"cuest_response_symmetric_k": True, "cuest_response_batch_size": batch_size})
    result = _device_solve(wfn, rhs)
    _check(wfn, rhs, result)
    stats = wfn.cphf_device_stats()
    assert stats["symmetric_gridk"] == 1 and stats["ao_rhs_tile"] < 11
    assert stats["jk_density_batch_size"] == batch_size
    start = 2 if unrestricted else 1
    for out in result[start:2*start]:
        np.testing.assert_array_equal(out.np, 0)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cuest_response_symmetric_gridk_default_math(unrestricted):
    """Validate the default SDK math mode with finite production screening."""
    wfn = _wavefunction("pbe0", unrestricted, mixed_precision=True, ints_tolerance=1.e-14)
    rhs = _rhs(wfn, unrestricted, count=3, mixed=False)
    expected = [m.np.copy() for m in wfn.cphf_solve(rhs, 1.e-8, 150, 0)]
    assert wfn.cphf_converged()
    psi4.set_options({"cuest_response_symmetric_k": True})
    result = _device_solve(wfn, rhs, tol=1.e-8)
    _check(wfn, rhs, result, relative_tol=1.e-8)
    for ref, out in zip(expected, result):
        np.testing.assert_allclose(out.np, ref, atol=1.e-8, rtol=0)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method", ["hf", "svwn", "pbe0", "cam-b3lyp"])
@pytest.mark.parametrize("directj,gridk", [(False, False), (True, False), (False, True), (True, True)])
def test_cuest_device_cg_matches_host(method, unrestricted, directj, gridk):
    wfn = _wavefunction(method, unrestricted, directj, gridk)
    rhs = _rhs(wfn, unrestricted, count=4)
    expected = [m.np.copy() for m in wfn.cphf_solve(rhs, 1.e-10, 150, 0)]
    assert wfn.cphf_converged()
    assert wfn.cphf_device_stats() == {}
    assert wfn.cphf_nfock_builds() > 0
    got = _device_solve(wfn, rhs)
    _check(wfn, rhs, got)
    for ref, out in zip(expected, got):
        np.testing.assert_allclose(out.np, ref, atol=1.e-8, rtol=0)
    # Initially inactive/non-prefix systems remain exact zeros.
    for out in got[(2 if unrestricted else 1):(4 if unrestricted else 2)]:
        np.testing.assert_array_equal(out.np, 0)
    assert wfn.cphf_device_stats()["result_device_to_host_bytes"] == sum(m.np.nbytes for m in rhs)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cuest_device_cg_zero_empty_and_limit(unrestricted):
    wfn = _wavefunction("hf", unrestricted)
    rhs = _rhs(wfn, unrestricted, mixed=False)
    zero = [m.clone() for m in rhs]
    for m in zero:
        m.zero()
    for batch in ([], zero):
        out = _device_solve(wfn, batch)
        assert wfn.cphf_converged()
        assert wfn.cphf_device_stats()["operator_calls"] == 0
        for m in out:
            np.testing.assert_array_equal(m.np, 0)
    _device_solve(wfn, rhs, maxiter=0)
    assert not wfn.cphf_converged()
    with pytest.raises(RuntimeError, match="positive finite tolerance"):
        _device_solve(wfn, rhs, tol=0)
    assert not wfn.cphf_converged()


def test_cuest_device_cg_noncanonical_panels():
    wfn = _wavefunction("hf")
    nocc = sum(wfn.Ca_subset("SO", "OCC").coldim())
    nmo = wfn.Ca().np.shape[1]
    rng = np.random.default_rng(50)
    for start, stop in ((0, nocc), (nocc, nmo)):
        q, _ = np.linalg.qr(rng.normal(size=(stop-start, stop-start)))
        wfn.Ca().np[:, start:stop] = wfn.Ca().np[:, start:stop] @ q
    rhs = _rhs(wfn, False, mixed=False)
    _check(wfn, rhs, _device_solve(wfn, rhs))


@pytest.mark.parametrize("symmetric_k", [False, True])
def test_cuest_device_cg_tiled_rhs(symmetric_k):
    wfn = _wavefunction("pbe0")
    rhs = _rhs(wfn, False, count=11)
    psi4.set_options({"cuest_response_symmetric_k": symmetric_k})
    _check(wfn, rhs, _device_solve(wfn, rhs))
    assert wfn.cphf_device_stats()["ao_rhs_tile"] < 11


@pytest.mark.parametrize("symmetric_k", [False, True])
def test_cuest_device_cg_empty_beta(symmetric_k):
    wfn = _wavefunction("hf", unrestricted=True, atom=True)
    rhs = _rhs(wfn, True, mixed=False)
    psi4.set_options({"cuest_response_symmetric_k": symmetric_k})
    result = _device_solve(wfn, rhs)
    _check(wfn, rhs, result)
    for b in result[1::2]:
        assert b.np.shape[0] == 0


@pytest.mark.parametrize("symmetric_k", [False, True])
def test_cuest_device_cg_borrowed_bare_k(symmetric_k):
    wfn = _wavefunction()
    psi4.set_options({"cuest_response_symmetric_k": symmetric_k})
    wfn.jk().set_omega_alpha(1.)
    rhs = _rhs(wfn, False, mixed=False)
    _check(wfn, rhs, _device_solve(wfn, rhs))
    assert wfn.jk().get_omega_alpha() == 1.
    assert wfn.jk().get_omega_beta() == 0.


def test_cuest_device_cg_grac():
    wfn = _wavefunction(grac=True)
    rhs = _rhs(wfn, False, mixed=False)
    _check(wfn, rhs, _device_solve(wfn, rhs))


def test_cuest_device_cg_cpu_xc_rejected():
    wfn = _wavefunction(gpu_xc=False)
    with pytest.raises(RuntimeError, match="CUEST_XC true before SCF"):
        _device_solve(wfn, _rhs(wfn, False))
    assert not wfn.cphf_converged()


def test_cuest_device_cg_external_rejected():
    wfn = _wavefunction("hf")
    psi4.set_options({"pe": True})
    try:
        with pytest.raises(RuntimeError, match="external/PCM/DDX/PE"):
            _device_solve(wfn, _rhs(wfn, False))
    finally:
        psi4.set_options({"pe": False})


def test_cuest_device_cg_bad_rhs_and_preconditioner():
    wfn = _wavefunction("hf")
    with pytest.raises(RuntimeError, match="occupied x virtual"):
        _device_solve(wfn, [psi4.core.Matrix.from_array(np.ones((1, 1)))])
    rhs = _rhs(wfn, False, mixed=False)
    rhs[0].np[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        _device_solve(wfn, rhs)
    rhs[0].np[0, 0] = 1.
    wfn.Fa().zero()
    with pytest.raises(RuntimeError, match="positive orbital-gap"):
        _device_solve(wfn, rhs)


def test_cuest_device_cg_finalized_jk_rejected():
    wfn = _wavefunction("hf")
    wfn.jk().finalize()
    with pytest.raises(RuntimeError, match="initialized JK plan"):
        _device_solve(wfn, _rhs(wfn, False))


def test_cuest_device_cg_mismatched_jk_rejected():
    """The resident adapter must reject an otherwise valid JK with the wrong AO layout."""
    wfn = _wavefunction("hf")
    wrong_basis = psi4.core.BasisSet.build(wfn.molecule(), "ORBITAL", "sto-3g")
    wrong_jk = psi4.core.JK.build_JK(wrong_basis, wfn.get_basisset("DF_BASIS_SCF"))
    wrong_jk.initialize()
    # HF owns this invariant and rejects the builder before attachment. The
    # resident adapter must keep using the original valid plan after rejection.
    with pytest.raises(RuntimeError, match="basis functions does not match"):
        wfn.set_jk(wrong_jk)
    rhs = _rhs(wfn, False, count=1, mixed=False)
    _check(wfn, rhs, _device_solve(wfn, rhs))


def test_cuest_device_cg_symmetry_rejected():
    wfn = _wavefunction("hf", symmetry="c2v")
    with pytest.raises(RuntimeError, match="C1 molecular symmetry"):
        _device_solve(wfn, _rhs(wfn, False))


def test_cuest_device_cg_negative_curvature():
    """Positive preconditioner does not turn an unstable Hessian into convergence."""
    wfn = _wavefunction("hf", unrestricted=True, neutral_uhf=True)
    wfn.Cb().np[:] = wfn.Ca().np
    nocc = sum(wfn.Ca_subset("SO", "OCC").coldim())
    inv = np.linalg.inv(wfn.Ca().np)
    eps = np.zeros(inv.shape[0])
    eps[nocc:] = 1.e-6
    f = inv.T @ np.diag(eps) @ inv
    wfn.Fa().np[:] = f
    wfn.Fb().np[:] = f
    rhs = _rhs(wfn, True, count=1, mixed=False)
    for m in rhs:
        m.zero()
    rhs[0].np[0, 0] = 1.
    rhs[1].np[0, 0] = -1.
    with pytest.raises(RuntimeError, match="nonpositive curvature"):
        _device_solve(wfn, rhs)
    assert not wfn.cphf_converged()


def test_cuest_device_cg_unstable_scf_rejected():
    """A converged SCF flag and positive gaps do not establish Hessian stability."""
    wfn = _wavefunction("hf", unrestricted=True, directj=False, gridk=False, guess="core")
    with pytest.raises(RuntimeError, match="nonpositive curvature"):
        _device_solve(wfn, _rhs(wfn, True, count=1, mixed=False))
    assert not wfn.cphf_converged()
    assert wfn.cphf_device_stats()["operator_calls"] > 0


@pytest.mark.parametrize("route", ["native", "einsums"])
def test_sapt_device_bridge(route, monkeypatch):
    """Neither SAPT route may enter its Python host CG when device mode is requested."""
    import ast
    import importlib
    from pathlib import Path
    native = importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms")
    util = importlib.import_module("psi4.driver.procrouting.sapt.sapt_util")
    a = _wavefunction("hf")
    b = _wavefunction("svwn")
    ra, rb = _rhs(a, False, count=1, mixed=False)[0], _rhs(b, False, count=1, mixed=False)[0]
    expected = []
    for wfn, r in ((a, ra), (b, rb)):
        solution = wfn.cphf_solve([r], 1.e-10, 150, 0)[0]
        assert wfn.cphf_converged()
        expected.append(2. * solution.vector_dot(r))
    psi4.set_options({"cuest_device_cg": True})
    def forbidden(*args, **kwargs):
        raise AssertionError("SAPT fell back to Python host CG")
    monkeypatch.setattr(native.solvers, "cg_solver", forbidden)
    monkeypatch.setattr(native.solvers, "cg_solver_ein", forbidden)
    if route == "native":
        solve = native._sapt_cpscf_solve
        rhs = [ra, rb]
    else:
        # The resident branch has no einsums dependency. Exercise the exact
        # installed function even on a build without the optional einsums package.
        path = Path(psi4.__file__).parent / "driver/procrouting/sapt/sapt_jk_terms_ein.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sapt_cpscf_solve")
        namespace = {"core": psi4.core, "np": np, "solve_device_sapt_response": util.solve_device_sapt_response}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        solve = namespace["_sapt_cpscf_solve"]
        rhs = [ra.np.copy(), rb.np.copy()]
    result = solve({"wfn_A": a, "wfn_B": b}, a.jk(), *rhs, 100, 1.e-9, sapt_jk_B=b.jk())
    matrices = result if route == "native" else [psi4.core.Matrix.from_array(m) for m in result]
    for wfn, r, solution in zip((a, b), (ra, rb), matrices):
        _check(wfn, [r], [solution])
    # The same 2*x.w contraction used for SAPT's relaxed induction terms.
    np.testing.assert_allclose([2. * x.vector_dot(r) for x, r in zip(matrices, (ra, rb))],
                               expected, atol=1.e-10, rtol=0)
