"""Fixed-orbital CPSCF comparisons: separate JK, GEMM, and XC effects."""

import numpy as np
import pytest
import psi4

from addons import using, uusing

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def _wavefunction(method="hf", unrestricted=False, symmetry="c1"):
    psi4.set_num_threads(2)
    mol = psi4.geometry(f"""
        {1 if unrestricted else 0} {2 if unrestricted else 1}
        O
        H 1 0.96
        H 1 0.96 2 104.5
        symmetry {symmetry}
    """)
    psi4.set_options({
        "basis": "cc-pvdz", "df_basis_scf": "cc-pvdz-jkfit",
        "scf_type": "df", "reference": "uhf" if unrestricted else "rhf",
        "save_jk": True, "e_convergence": 1.e-11, "d_convergence": 1.e-10,
        "use_cuest": False, "cuest_xc": False, "cuest_mixed_precision": False,
        "dft_radial_points": 75, "dft_spherical_points": 302,
    })
    return psi4.energy(method, molecule=mol, return_wfn=True)[1]


def _rotations(wfn, unrestricted, count=2, blocked=False):
    rng = np.random.default_rng(451)
    ret = []
    for _ in range(count):
        for spin in (("a", "b") if unrestricted else ("a",)):
            subset = getattr(wfn, f"C{spin}_subset")
            occ, vir = subset("SO", "OCC"), subset("SO", "VIR")
            if blocked:
                arrays = [rng.normal(size=(o, v)) * 0.01
                          for o, v in zip(occ.coldim(), vir.coldim())]
            else:
                arrays = rng.normal(size=(sum(occ.coldim()), sum(vir.coldim()))) * 0.01
            ret.append(psi4.core.Matrix.from_array(arrays))
    return ret


def _assert_matrices(ref, got, atol=1.e-8):
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert list(a.rowdim()) == list(b.rowdim())
        assert list(a.coldim()) == list(b.coldim())
        assert a.symmetry() == b.symmetry()
        for aa, bb in zip(a.to_array(), b.to_array()) if a.nirrep() > 1 else [(a.np, b.np)]:
            np.testing.assert_allclose(bb, aa, rtol=0, atol=atol)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cpscf_separate_operator_projection(unrestricted):
    """Every J/K component, not just the first spin pair, must be MO-projected."""
    wfn = _wavefunction(unrestricted=unrestricted)
    rotations = _rotations(wfn, unrestricted)
    operators = [m.clone() for m in wfn.twoel_Hx(rotations, False, "SO")]
    mo = wfn.twoel_Hx(rotations, False, "MO")
    ref = []
    for i, op in enumerate(operators):
        spin = ("a", "b")[i % 2] if unrestricted else "a"
        subset = getattr(wfn, f"C{spin}_subset")
        ref.append(psi4.core.triplet(subset("SO", "OCC"), op, subset("SO", "VIR"),
                                     True, False, False))
    _assert_matrices(ref, mo, atol=1.e-12)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cpscf_mixed_symmetry_batch(unrestricted):
    """Flattened alpha/beta input indices must survive a mixed AO/SO batch."""
    wfn = _wavefunction(unrestricted=unrestricted, symmetry="c2v")
    dense = _rotations(wfn, unrestricted, count=1)
    blocked = _rotations(wfn, unrestricted, count=1, blocked=True)
    for apply in (wfn.onel_Hx, wfn.cphf_Hx,
                  lambda x: wfn.twoel_Hx(x, False, "MO")):
        ref = [m.clone() for x in (dense, blocked) for m in apply(x)]
        _assert_matrices(ref, apply(dense + blocked), atol=1.e-11)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method,message", [
    ("m06-l", "MGGA"), ("wb97m-v", "VV10")])
def test_cpscf_unsupported_native_kernels(method, message, unrestricted):
    wfn = _wavefunction(method, unrestricted)
    with pytest.raises(RuntimeError, match=message):
        wfn.cphf_Hx(_rotations(wfn, unrestricted, count=1))


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cpscf_mixed_symmetry_solver(unrestricted):
    wfn = _wavefunction(unrestricted=unrestricted, symmetry="c2v")
    rhs = (_rotations(wfn, unrestricted, count=1) +
           _rotations(wfn, unrestricted, count=1, blocked=True))
    solved = wfn.cphf_solve(rhs, 1.e-10, 100, 0)
    assert wfn.cphf_converged()
    _assert_matrices(rhs, wfn.cphf_Hx(solved), atol=1.e-10)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("unrestricted", [False, True])
def test_cpscf_gpu_xc_guard(unrestricted):
    """Toggling CUEST_XC after a CPU grid was built must not reach either kernel."""
    wfn = _wavefunction("blyp", unrestricted)
    psi4.set_options({"use_cuest": True, "cuest_xc": True})
    matrices = [wfn.Da().clone()]
    if unrestricted:
        matrices.append(wfn.Db().clone())
    try:
        with pytest.raises(RuntimeError, match="not initialized on the cuEST grid"):
            wfn.V_potential().compute_Vx(matrices, [m.clone() for m in matrices])
    finally:
        psi4.set_options({"use_cuest": False, "cuest_xc": False})


def _response_jk(wfn, use_cuest):
    psi4.set_options({"use_cuest": use_cuest})
    jk = psi4.core.JK.build_JK(wfn.basisset(), wfn.get_basisset("DF_BASIS_SCF"))
    functional = wfn.functional()
    jk.set_do_J(True)
    jk.set_do_K(functional.is_x_hybrid())
    jk.set_do_wK(functional.is_x_lrc())
    jk.set_wcombine(False)
    jk.set_omega(functional.x_omega())
    jk.set_omega_alpha(functional.x_alpha())
    jk.set_omega_beta(functional.x_beta())
    jk.initialize()
    return jk


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("unrestricted", [False, True])
def test_cuest_cpscf_symmetry_fallback(unrestricted):
    """Blocked GEMMs stay native while cuEST handles mixed AO/SO response JK."""
    wfn = _wavefunction("pbe0", unrestricted, symmetry="c2v")
    dense = _rotations(wfn, unrestricted, count=1)
    blocked = _rotations(wfn, unrestricted, count=1, blocked=True)
    original_jk = wfn.jk()
    expected = [m.clone() for m in wfn.cphf_Hx(dense + blocked)]
    gpu_jk = _response_jk(wfn, True)
    try:
        wfn.set_jk(gpu_jk)
        psi4.set_options({"cuest_gemm_min_dim": 0})
        psi4.core.clean_timers()
        wfn.onel_Hx(blocked)
        assert not any("cuEST CPSCF: cuBLAS" in k for k in psi4.core.get_timer_dict())
        _assert_matrices(expected, wfn.cphf_Hx(dense + blocked))
        assert any("cuEST CPSCF: cuBLAS" in k for k in psi4.core.get_timer_dict())
    finally:
        wfn.set_jk(original_jk)
        gpu_jk.finalize()
        psi4.set_options({"use_cuest": False})


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method", ["hf", "svwn", "blyp", "pbe0", "cam-b3lyp", "lrc-wpbe"])
def test_cuest_cpscf_fixed_orbitals(method, unrestricted):
    """Native XC + nonsymmetric cuEST JK; force or bypass GPU transforms."""
    wfn = _wavefunction(method, unrestricted)
    rotations = _rotations(wfn, unrestricted)
    original_jk = wfn.jk()
    cpu_jk = _response_jk(wfn, False)
    gpu_jk = None
    try:
        wfn.set_jk(cpu_jk)
        one = [m.clone() for m in wfn.onel_Hx(rotations)]
        ref = {}
        for combine in (True, False):
            for basis in ("SO", "MO"):
                ref[combine, basis] = [
                    m.clone() for m in wfn.twoel_Hx(rotations, combine, basis)]
        href = [m.clone() for m in wfn.cphf_Hx(rotations)]
        triplet = None if unrestricted else [
            m.clone() for m in wfn.twoel_Hx_full(rotations, True, "MO", False)]
        gpu_jk = _response_jk(wfn, True)
        assert gpu_jk.name() == "cuESTJK"
        wfn.set_jk(gpu_jk)
        for threshold in (10**9, 0):
            psi4.set_options({"cuest_gemm_min_dim": threshold})
            psi4.core.clean_timers()
            _assert_matrices(one, wfn.onel_Hx(rotations), atol=1.e-11)
            for (combine, basis), expected in ref.items():
                _assert_matrices(expected, wfn.twoel_Hx(rotations, combine, basis))
            _assert_matrices(href, wfn.cphf_Hx(rotations))
            if triplet is not None:
                _assert_matrices(triplet, wfn.twoel_Hx_full(rotations, True, "MO", False))
            calls = sum(v["n_calls"] for k, v in psi4.core.get_timer_dict().items()
                        if "cuEST CPSCF: cuBLAS" in k)
            assert (calls > 0) == (threshold == 0)
        # K's preweighting belongs to the attached builder, not USE_CUEST.
        psi4.set_options({"use_cuest": False})
        _assert_matrices(href, wfn.cphf_Hx(rotations))
        if method == "pbe0":
            # SAPT borrows this builder for bare exchange before CPKS.
            # Response must rescale K, not permanently change that shared plan.
            gpu_jk.set_omega_alpha(1.0)
            gpu_jk.set_omega_beta(0.0)
            psi4.set_options({"use_cuest": True, "cuest_gemm_min_dim": 0})
            _assert_matrices(href, wfn.cphf_Hx(rotations))
            _assert_matrices(ref[False, "MO"], wfn.twoel_Hx(rotations, False, "MO"))
            assert gpu_jk.get_omega_alpha() == 1.0
            assert gpu_jk.get_omega_beta() == 0.0
        if method == "cam-b3lyp":
            gpu_jk.set_omega(0.1)
            with pytest.raises(RuntimeError, match="combined-exchange JK plan is incompatible"):
                wfn.cphf_Hx(rotations)
    finally:
        wfn.set_jk(original_jk)
        cpu_jk.finalize()
        if gpu_jk is not None:
            gpu_jk.finalize()
        psi4.set_options({"use_cuest": False})


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method", ["hf", "pbe0"])
def test_cuest_cpscf_solver_residual(method, unrestricted):
    wfn = _wavefunction(method, unrestricted)
    rhs = _rotations(wfn, unrestricted, count=3)
    # Initially converged equations and shrinking active RHS batches.
    for m in rhs[-(2 if unrestricted else 1):]:
        m.zero()
    original_jk = wfn.jk()
    builders = []
    solutions = []
    try:
        for use_cuest in (False, True):
            jk = _response_jk(wfn, use_cuest)
            builders.append(jk)
            wfn.set_jk(jk)
            psi4.set_options({"cuest_gemm_min_dim": 0})
            solved = wfn.cphf_solve(rhs, 1.e-10, 100, 0)
            assert wfn.cphf_converged()
            for b, ax in zip(rhs, wfn.cphf_Hx(solved)):
                np.testing.assert_allclose(ax.np, b.np, atol=1.e-10, rtol=0)
            solutions.append([m.clone() for m in solved])
        _assert_matrices(*solutions, atol=1.e-8)
    finally:
        wfn.set_jk(original_jk)
        for jk in builders:
            jk.finalize()
        psi4.set_options({"use_cuest": False})


# => XC response kernel on the cuEST grid <= #
#
# cuEST exposes no grid coordinates, so the host and cuEST grids are different
# quadratures. Each grid is therefore checked against itself first: central finite
# differences of its own compute_V, the kernel's algebraic identities and the
# dense/factored/occupied/LibXC-route variants. Host-vs-cuEST comparisons are
# quadrature-level only.

_GRIDS = ["host", pytest.param("cuest", marks=[*using("cuest"), *using("cuda_cc8"), pytest.mark.cuest])]
_KINDS = ["singlet", "triplet", "uks"]


@pytest.fixture(autouse=True)
def _reset_cuest_options():
    yield
    psi4.set_options({"use_cuest": False, "cuest_xc": False, "cuest_xc_response_host_libxc": False,
                      "print": 1})


def _M(array):
    return psi4.core.Matrix.from_array(np.ascontiguousarray(array))


def _potential(wfn, grid, polarized=None, name=None):
    """A freshly initialized V on the requested grid for wfn's basis and functional.

    name builds that functional instead; with polarized=True, its polarized form for an
    RKS reference (the triplet oracle).
    """
    gpu = grid == "cuest"
    psi4.set_options({"use_cuest": gpu, "cuest_xc": gpu, "cuest_mixed_precision": False})
    functional = wfn.functional()
    restricted = wfn.same_a_b_orbs() and not polarized
    if name is not None:
        functional = psi4.driver.dft.build_superfunctional(name, restricted)[0]
    V = psi4.core.VBase.build(wfn.basisset(), functional, "RV" if restricted else "UV")
    V.initialize()
    assert V.uses_cuest_xc() == gpu
    return V


def _set_reference(V, Cs):
    """Reference density D = C C^T per spin, with matching occupied factors."""
    V.set_D([_M(c @ c.T) for c in Cs])
    V.set_Cocc([_M(c) for c in Cs])


def _potential_at(V, Cs):
    _set_reference(V, Cs)
    out = [psi4.core.Matrix(c.shape[0], c.shape[0]) for c in Cs]
    V.compute_V(out)
    return [o.np.copy() for o in out]


def _occupied(wfn, unrestricted):
    Cs = [wfn.Ca_subset("AO", "OCC").np.copy()]
    if unrestricted:
        Cs.append(wfn.Cb_subset("AO", "OCC").np.copy())
    return Cs


def _virtual(wfn, unrestricted):
    Cs = [wfn.Ca_subset("AO", "VIR").np.copy()]
    if unrestricted:
        Cs.append(wfn.Cb_subset("AO", "VIR").np.copy())
    return Cs


def _apply(V, kind, Dx):
    """Vx for a list of AO numpy densities (UKS: [a0, b0, a1, b1, ...])."""
    nbf = Dx[0].shape[0]
    ret = [psi4.core.Matrix(nbf, nbf) for _ in Dx]
    inputs = [_M(d) for d in Dx]
    if kind == "triplet":
        V.compute_Vx_full(inputs, ret, False)
    else:
        V.compute_Vx(inputs, ret)
    return [r.np.copy() for r in ret]


def _apply_factored(V, kind, left, right):
    nbf = left[0].shape[0]
    ret = [psi4.core.Matrix(nbf, nbf) for _ in left]
    L, R = [_M(x) for x in left], [_M(x) for x in right]
    if kind == "triplet":
        V.compute_Vx_full_factored(L, R, ret, False)
    else:
        V.compute_Vx_factored(L, R, ret)
    return [r.np.copy() for r in ret]


def _assert_close(ref, got, atol, rtol=0.0):
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b, a, atol=atol + rtol * np.abs(a).max(), rtol=0)


def _kind_setup(method, kind, grid):
    """(wfn, V for Vx, occupied factors, virtual factors)."""
    unrestricted = kind == "uks"
    wfn = _wavefunction(method, unrestricted)
    V = _potential(wfn, grid)
    Cs = _occupied(wfn, unrestricted)
    _set_reference(V, Cs)
    return wfn, V, Cs, _virtual(wfn, unrestricted)


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("method", ["svwn", "blyp", "pbe0", "cam-b3lyp", "lrc-wpbe"])
def test_vx_finite_difference(method, kind, grid):
    """Vx is the orbital derivative of this grid's own compute_V.

    RKS singlet: Vx[dDa] = dVa/2 (compute_Vx_full's half-alpha normalization).
    RKS triplet: the same with Ca = C + eU, Cb = C - eU on the polarized functional.
    UKS: Vx[dDa, dDb] = (dVa, dVb).
    Exercised through both the dense and the factored entry points; dD = U C^T + C U^T
    has the factors [U | C] and [C | U].
    """
    wfn, V, Cs, _ = _kind_setup(method, kind, grid)
    rng = np.random.default_rng(7)
    Us = [rng.normal(size=c.shape) * 0.1 for c in Cs]
    oracle = _potential(wfn, grid, polarized=True, name=method) if kind == "triplet" else V
    eps = 1.e-4
    if kind == "triplet":
        plus = _potential_at(oracle, [Cs[0] + eps * Us[0], Cs[0] - eps * Us[0]])[0]
        minus = _potential_at(oracle, [Cs[0] - eps * Us[0], Cs[0] + eps * Us[0]])[0]
        expected = [0.25 * (plus - minus) / eps]
    else:
        plus = _potential_at(oracle, [c + eps * u for c, u in zip(Cs, Us)])
        minus = _potential_at(oracle, [c - eps * u for c, u in zip(Cs, Us)])
        scale = 0.5 if kind == "singlet" else 1.0
        expected = [scale * (p - m) / (2 * eps) for p, m in zip(plus, minus)]
    _set_reference(V, Cs)
    Dx = [u @ c.T + c @ u.T for c, u in zip(Cs, Us)]
    dense = _apply(V, kind, Dx)
    _assert_close(expected, dense, atol=2.e-8)
    factored = _apply_factored(V, kind, [np.hstack([u, c]) for c, u in zip(Cs, Us)],
                               [np.hstack([c, u]) for c, u in zip(Cs, Us)])
    _assert_close(dense, factored, atol=1.e-11)


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("method", ["svwn", "pbe0"])
def test_vx_algebra(method, kind, grid):
    """Signed, nonsymmetric, transposed, zero, batched, cross-spin and factored inputs."""
    wfn, V, Cs, Cv = _kind_setup(method, kind, grid)
    rng = np.random.default_rng(11)
    nbf = Cs[0].shape[0]
    nspin = len(Cs)

    def response():
        # CPHF-shaped, nonsymmetric: Cocc X Cvir^T per spin.
        xs = [rng.normal(size=(o.shape[1], v.shape[1])) * 0.1 for o, v in zip(Cs, Cv)]
        return xs, [o @ x @ v.T for o, x, v in zip(Cs, xs, Cv)]

    x1, D1 = response()
    x2, D2 = response()
    V1, V2 = _apply(V, kind, D1), _apply(V, kind, D2)
    scale = max(np.abs(v).max() for v in V1)
    atol = 1.e-11 * max(1.0, scale)
    assert scale > 1.e-3
    for v in V1:
        np.testing.assert_allclose(v, v.T, atol=atol, rtol=0)
    _assert_close(V1, _apply(V, kind, [d.T for d in D1]), atol)
    _assert_close([-v for v in V1], _apply(V, kind, [-d for d in D1]), atol)
    _assert_close([np.zeros((nbf, nbf))] * nspin, _apply(V, kind, [d - d.T for d in D1]), atol)
    _assert_close([np.zeros((nbf, nbf))] * nspin, _apply(V, kind, [np.zeros((nbf, nbf))] * nspin), 0.0)
    _assert_close([2 * a - 3 * b for a, b in zip(V1, V2)],
                  _apply(V, kind, [2 * a - 3 * b for a, b in zip(D1, D2)]), 10 * atol)
    # Self-adjoint on symmetric densities (summed over spins).
    S1 = [d + d.T for d in D1]
    S2 = [d + d.T for d in D2]
    lhs = sum(np.vdot(s, v) for s, v in zip(S2, _apply(V, kind, S1)))
    rhs = sum(np.vdot(s, v) for s, v in zip(S1, _apply(V, kind, S2)))
    assert abs(lhs - rhs) <= 1.e-10 * max(1.0, abs(lhs))
    # A batch, in either order, is the individual builds; UKS keeps [a_i, b_i] pairs.
    _assert_close(V1 + V2, _apply(V, kind, D1 + D2), atol)
    _assert_close(V2 + V1, _apply(V, kind, D2 + D1), atol)
    if kind == "uks":
        alpha_only = _apply(V, kind, [D1[0], np.zeros((nbf, nbf))])
        beta_only = _apply(V, kind, [np.zeros((nbf, nbf)), D1[1]])
        _assert_close(V1, [a + b for a, b in zip(alpha_only, beta_only)], atol)
        # Cross-spin coupling must be present: an alpha response moves V_beta.
        assert np.abs(alpha_only[1]).max() > 1.e-4 * scale
    # Factored inputs (Cocc, Cvir X^T), including a zero-rank factor.
    factored = _apply_factored(V, kind, list(Cs), [v @ x.T for v, x in zip(Cv, x1)])
    _assert_close(V1, factored, atol)
    empty = [np.zeros((nbf, 0))] * nspin
    _assert_close([np.zeros((nbf, nbf))] * nspin, _apply_factored(V, kind, empty, empty), 0.0)


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("method", ["svwn", "pbe"])
def test_vx_empty_beta_spin(method, grid):
    """UKS with no beta electrons: zero-rank occupied factors and responses.

    V_beta is not finite-difference checkable at rho_beta = 0 (zeta = 1). For a GGA the
    error grows as the step shrinks. For LDA (VWN) LibXC's own vrho_b is ill-conditioned
    there: its finite difference along rho_a misses its analytic v2rho2_ab by 1% (LibXC
    7.0.0, rho_a = 0.1) to 3% (7.1.2, rho_a = 1e-6), while v2rho2_ab itself agrees between
    the versions to 3e-5. The beta block is therefore checked for finiteness and, on the
    cuEST grid, against the native host-grid V_beta at quadrature level.
    """
    psi4.set_num_threads(2)
    mol = psi4.geometry("""
        0 2
        H
        symmetry c1
    """)
    psi4.set_options({"basis": "cc-pvdz", "scf_type": "df", "df_basis_scf": "cc-pvdz-jkfit",
                      "reference": "uhf", "use_cuest": False, "cuest_xc": False,
                      "e_convergence": 1.e-10, "d_convergence": 1.e-9})
    wfn = psi4.energy(method, molecule=mol, return_wfn=True)[1]
    V = _potential(wfn, grid)
    Cs = _occupied(wfn, True)
    assert Cs[1].shape[1] == 0
    rng = np.random.default_rng(3)
    Us = [rng.normal(size=c.shape) * 0.1 for c in Cs]
    eps = 1.e-3
    plus = _potential_at(V, [c + eps * u for c, u in zip(Cs, Us)])
    minus = _potential_at(V, [c - eps * u for c, u in zip(Cs, Us)])
    expected = [(p - m) / (2 * eps) for p, m in zip(plus, minus)]
    _set_reference(V, Cs)
    Dx = [u @ c.T + c @ u.T for c, u in zip(Cs, Us)]
    native_beta = None
    if grid == "cuest":
        host = _potential(wfn, "host")
        _set_reference(host, Cs)
        native_beta = _apply(host, "uks", Dx)[1]
    for got in (_apply(V, "uks", Dx),
                _apply_factored(V, "uks", [np.hstack([u, c]) for c, u in zip(Cs, Us)],
                                [np.hstack([c, u]) for c, u in zip(Cs, Us)])):
        _assert_close(expected[:1], got[:1], atol=1.e-7)
        assert np.all(np.isfinite(got[1]))
        if native_beta is not None:
            # H200, CUDA LibXC 7.1.2: svwn 1.7e-7 (scale 7e-3), pbe 1.7e-6 (scale 4.4).
            scale = np.abs(native_beta).max()
            print(f"\n{method}: beta block, cuEST grid vs host grid: "
                  f"{np.abs(native_beta - got[1]).max():.3e} (scale {scale:.3e})")
            _assert_close([native_beta], [got[1]], atol=2.e-6 * max(1.0, scale))
    # A beta-only response density on an alpha-only reference.
    beta = np.outer(wfn.Ca_subset("AO", "OCC").np[:, 0], wfn.Ca_subset("AO", "VIR").np[:, 0])
    out = _apply(V, "uks", [np.zeros_like(beta), beta])
    assert np.all(np.isfinite(out[0])) and np.all(np.isfinite(out[1]))


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("unrestricted", [False, True])
def test_vx_symmetry_irreps(unrestricted, grid):
    """Symmetry-blocked inputs/outputs of every irrep equal the AO build, transformed."""
    wfn = _wavefunction("blyp", unrestricted, symmetry="c2v")
    V = _potential(wfn, grid)
    V.set_D([wfn.Da()] + ([wfn.Db()] if unrestricted else []))
    V.set_Cocc([wfn.Ca_subset("AO", "OCC")] + ([wfn.Cb_subset("AO", "OCC")] if unrestricted else []))
    U = wfn.aotoso()
    nsopi = wfn.nsopi()
    nbf = wfn.basisset().nbf()
    rng = np.random.default_rng(5)
    kind = "uks" if unrestricted else "singlet"
    for h in range(wfn.nirrep()):
        so = []
        for _ in range(2 if unrestricted else 1):
            m = psi4.core.Matrix("Dx", nsopi, nsopi, h)
            for k in range(m.nirrep()):
                block = m.nph[k]
                if block.size:
                    block[...] = rng.normal(size=block.shape) * 0.1
            so.append(m)
        ao = []
        for m in so:
            d = np.zeros((nbf, nbf))
            for k in range(m.nirrep()):
                if m.nph[k].size:
                    d += U.nph[k] @ m.nph[k] @ U.nph[k ^ h].T
            ao.append(d)
        expected = _apply(V, kind, ao)
        ret = [psi4.core.Matrix("Vx", nsopi, nsopi, h) for _ in so]
        V.compute_Vx(so, ret)
        for e, r in zip(expected, ret):
            for k in range(r.nirrep()):
                if r.nph[k].size:
                    np.testing.assert_allclose(r.nph[k], U.nph[k].T @ e @ U.nph[k ^ h], atol=1.e-11, rtol=0)
        # Symmetry-blocked factors take the same route.
        ident = psi4.core.Matrix("I", nsopi, nsopi)
        ident.identity()
        fret = [psi4.core.Matrix("Vx", nsopi, nsopi, h) for _ in so]
        V.compute_Vx_factored(so, [ident] * len(so), fret)
        _assert_matrices(ret, fret, atol=1.e-11)


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("kind", _KINDS)
def test_vx_reference_changes(kind, grid):
    """No numerical state survives a change of reference or grid."""
    wfn, V, Cs, Cv = _kind_setup("blyp", kind, grid)
    rng = np.random.default_rng(13)
    Dx = [o @ rng.normal(size=(o.shape[1], v.shape[1])) @ v.T * 0.1 for o, v in zip(Cs, Cv)]
    first = _apply(V, kind, Dx)
    _assert_close(first, _apply(V, kind, Dx), 1.e-13)
    # A different reference changes the kernel; returning restores it.
    other = [c @ np.linalg.qr(rng.normal(size=(c.shape[1], c.shape[1])))[0] * 1.05 for c in Cs]
    _set_reference(V, other)
    moved = _apply(V, kind, Dx)
    assert max(np.abs(a - b).max() for a, b in zip(first, moved)) > 1.e-6
    _set_reference(V, Cs)
    _assert_close(first, _apply(V, kind, Dx), 1.e-13)
    # The reference is D; occupied factors are an optimization used only when C C^T = D.
    V.set_Cocc([_M(c @ np.linalg.qr(rng.normal(size=(c.shape[1], c.shape[1])))[0]) for c in Cs])
    _assert_close(first, _apply(V, kind, Dx), 1.e-11)
    V.set_Cocc([_M(c * 1.01) for c in Cs])
    _assert_close(first, _apply(V, kind, Dx), 1.e-11)
    V.set_D([_M(c @ c.T) for c in Cs])  # Cocc still stale
    _assert_close(first, _apply(V, kind, Dx), 1.e-11)
    # An independent V object (another functional) interleaved with this one shares nothing.
    # For a triplet V, the other object is a polarized UV with an [a, b] = [Dx, Dx] pair.
    other_kind = "uks" if kind == "triplet" else kind
    other_Cs, other_Dx = (Cs + Cs, Dx + Dx) if kind == "triplet" else (Cs, Dx)
    other_V = _potential(wfn, grid, polarized=kind == "triplet", name="svwn")
    _set_reference(other_V, other_Cs)
    other_first = _apply(other_V, other_kind, other_Dx)
    _set_reference(V, Cs)
    _assert_close(first, _apply(V, kind, Dx), 1.e-11)
    _assert_close(other_first, _apply(other_V, other_kind, other_Dx), 1.e-13)
    other_V.finalize()
    # Rebuild the grid: through the initial grid (rejected) and back.
    psi4.set_options({"dft_radial_points_initial": 50, "dft_spherical_points_initial": 194})
    try:
        V.set_grid("Initial")
        assert V.initial_grid()
        _set_reference(V, Cs)
        if grid == "cuest":
            with pytest.raises(RuntimeError, match="final DFT grid"):
                _apply(V, kind, Dx)
        V.set_grid("Final")
    finally:
        psi4.core.revoke_global_option_changed("DFT_RADIAL_POINTS_INITIAL")
        psi4.core.revoke_global_option_changed("DFT_SPHERICAL_POINTS_INITIAL")
    _set_reference(V, Cs)
    _assert_close(first, _apply(V, kind, Dx), 1.e-12)
    # A failed call leaves the object usable, and finalize/initialize is clean.
    nbf = Dx[0].shape[0]
    with pytest.raises(RuntimeError, match="input and output sizes"):
        V.compute_Vx([_M(d) for d in Dx], [psi4.core.Matrix(nbf, nbf) for _ in Dx] * 2)
    if grid == "cuest":
        # (The native kernel does not check output dimensions.)
        with pytest.raises(RuntimeError, match="nbf x nbf"):
            V.compute_Vx([_M(d) for d in Dx], [psi4.core.Matrix(2, 2) for _ in Dx])
    _assert_close(first, _apply(V, kind, Dx), 1.e-12)
    V.finalize()
    V.initialize()
    _set_reference(V, Cs)
    _assert_close(first, _apply(V, kind, Dx), 1.e-12)
    V.finalize()


@pytest.mark.parametrize("grid", _GRIDS)
def test_vx_grac_conventions(grid):
    """RKS singlet response uses the uncorrected kernel; triplet with GRAC is rejected."""
    psi4.set_options({"dft_grac_shift": 0.1})
    try:
        wfn = _wavefunction("pbe0", False)
        gpu = grid == "cuest"
        psi4.set_options({"use_cuest": gpu, "cuest_xc": gpu, "cuest_mixed_precision": False})
        grac = psi4.core.VBase.build(wfn.basisset(), wfn.functional(), "RV")
        grac.initialize()
    finally:
        psi4.core.revoke_global_option_changed("DFT_GRAC_SHIFT")
        psi4.set_options({"dft_grac_shift": 0.0})
    assert wfn.functional().needs_grac()
    plain = _potential(wfn, grid, name="pbe0")
    assert not plain.functional().needs_grac()
    Cs, Cv = _occupied(wfn, False), _virtual(wfn, False)
    rng = np.random.default_rng(17)
    Dx = [Cs[0] @ rng.normal(size=(Cs[0].shape[1], Cv[0].shape[1])) @ Cv[0].T * 0.1]
    for V in (grac, plain):
        _set_reference(V, Cs)
    # The GRAC potential differs at deriv = 1 ...
    assert np.abs(_potential_at(grac, Cs)[0] - _potential_at(plain, Cs)[0]).max() > 1.e-6
    # ... but not in the response kernel.
    _assert_close(_apply(plain, "singlet", Dx), _apply(grac, "singlet", Dx), 1.e-12)
    with pytest.raises(RuntimeError, match="GRAC"):
        _apply(grac, "triplet", Dx)


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method,message", [("m06-l", "MGGA"), ("wb97m-v", "VV10")])
def test_vx_unsupported_on_grid(method, message, unrestricted, grid):
    wfn = _wavefunction("pbe", unrestricted)
    V = _potential(wfn, grid, name=method)
    Cs = _occupied(wfn, unrestricted)
    _set_reference(V, Cs)
    with pytest.raises(RuntimeError, match=message):
        _apply(V, "uks" if unrestricted else "singlet", [c @ c.T for c in Cs])


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("method", ["svwn", "blyp", "pbe0", "lrc-wpbe"])
def test_vx_cuest_routes_and_host_grid(method, kind, tmp_path):
    """Reference paths and LibXC routes agree on the cuEST grid; the host grid agrees
    at quadrature level (the two grids are different, see above)."""
    wfn, host, Cs, Cv = _kind_setup(method, kind, "host")
    rng = np.random.default_rng(19)
    Dx = [o @ rng.normal(size=(o.shape[1], v.shape[1])) @ v.T * 0.1 for o, v in zip(Cs, Cv)]
    expected_host = _apply(host, kind, Dx)
    out = tmp_path / "routes.out"
    psi4.core.set_output_file(str(out), False)
    try:
        psi4.set_options({"print": 3})
        gpu = _potential(wfn, "cuest")
        results = {}
        for host_libxc in (False, True):
            psi4.set_options({"cuest_xc_response_host_libxc": host_libxc})
            _set_reference(gpu, Cs)
            results["occupied", host_libxc] = _apply(gpu, kind, Dx)
            gpu.set_Cocc([_M(c * 1.01) for c in Cs])
            results["dense", host_libxc] = _apply(gpu, kind, Dx)
    finally:
        psi4.core.close_outfile()
    text = out.read_text()
    assert "reference from occupied factors" in text
    assert "reference from dense D" in text
    assert "host LibXC" in text
    cuda_libxc = "CUDA LibXC" in text
    print(f"\n{method} {kind}: CUDA LibXC route present: {cuda_libxc}")
    base = results["occupied", False]
    scale = max(np.abs(v).max() for v in base)
    for key, value in results.items():
        err = max(np.abs(a - b).max() for a, b in zip(base, value))
        print(f"  {key}: max |dVx| vs occupied/default = {err:.3e} (scale {scale:.3e})")
        assert err <= 1.e-11 * max(1.0, scale)
    err = max(np.abs(a - b).max() for a, b in zip(expected_host, base))
    print(f"  host grid vs cuEST grid: max |dVx| = {err:.3e}")
    assert err <= 1.e-4 * max(1.0, scale)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("kind", _KINDS)
def test_vx_cuest_response_budget(kind):
    """CUEST_XC_RESPONSE_BUDGET_MB changes only cuEST's workspace, never the response."""
    wfn, V, Cs, Cv = _kind_setup("pbe0", kind, "cuest")
    rng = np.random.default_rng(23)
    Dx = [o @ rng.normal(size=(o.shape[1], v.shape[1])) @ v.T * 0.1 for o, v in zip(Cs, Cv)]
    reference = _apply(V, kind, Dx)
    scale = max(np.abs(v).max() for v in reference)
    for budget in (256, 16):
        psi4.set_options({"cuest_xc_response_budget_mb": budget})
        small = _potential(wfn, "cuest")  # fresh workspace, sized under the new budget
        _set_reference(small, Cs)
        err = max(np.abs(a - b).max() for a, b in zip(reference, _apply(small, kind, Dx)))
        print(f"\n{kind} budget {budget} MB: max |dVx| vs 2000 MB = {err:.3e}")
        assert err <= 1.e-11 * max(1.0, scale)
    psi4.set_options({"cuest_xc_response_budget_mb": 0})
    bad = _potential(wfn, "cuest")
    _set_reference(bad, Cs)
    with pytest.raises(Exception, match="CUEST_XC_RESPONSE_BUDGET_MB must be positive"):
        _apply(bad, kind, Dx)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("kind", _KINDS)
def test_vx_cuest_grid_survives_option_changes(kind):
    """A cuEST-grid V keeps its kernel after another V turns CUEST_XC off.

    Building a host-grid V sets CUEST_XC false globally. The cuEST-grid V must keep
    using its own grid (Phoenix 14085249 segfaulted in the native kernel instead).
    """
    wfn, _, Cs, Cv = _kind_setup("blyp", kind, "host")
    rng = np.random.default_rng(31)
    Dx = [o @ rng.normal(size=(o.shape[1], v.shape[1])) @ v.T * 0.1 for o, v in zip(Cs, Cv)]
    gpu = _potential(wfn, "cuest")
    _set_reference(gpu, Cs)
    first = _apply(gpu, kind, Dx)
    host = _potential(wfn, "host")
    assert not psi4.core.get_option("SCF", "CUEST_XC")
    assert gpu.uses_cuest_xc() and not host.uses_cuest_xc()
    _set_reference(gpu, Cs)
    _assert_close(first, _apply(gpu, kind, Dx), 1.e-12)
    _set_reference(host, Cs)
    assert max(np.abs(a - b).max() for a, b in zip(first, _apply(host, kind, Dx))) < 1.e-4
    gpu.finalize()


def _gpu_wavefunction(method, unrestricted, symmetry="c1"):
    psi4.set_num_threads(2)
    mol = psi4.geometry(f"""
        {1 if unrestricted else 0} {2 if unrestricted else 1}
        O
        H 1 0.96
        H 1 0.96 2 104.5
        symmetry {symmetry}
    """)
    psi4.set_options({
        "basis": "cc-pvdz", "df_basis_scf": "cc-pvdz-jkfit",
        "scf_type": "df", "reference": "uhf" if unrestricted else "rhf",
        "save_jk": True, "e_convergence": 1.e-11, "d_convergence": 1.e-10,
        "use_cuest": True, "cuest_xc": True, "cuest_mixed_precision": False,
        "dft_radial_points": 75, "dft_spherical_points": 302,
    })
    wfn = psi4.energy(method, molecule=mol, return_wfn=True)[1]
    assert wfn.V_potential().uses_cuest_xc()
    return wfn


def _polarizability(wfn, unrestricted):
    """Static dipole polarizability from the wavefunction's own CPHF solver (phase-invariant)."""
    mints = psi4.core.MintsHelper(wfn.basisset())
    dipoles = mints.so_dipole()
    rhs = []
    for mu in dipoles:
        for spin in (("a", "b") if unrestricted else ("a",)):
            subset = getattr(wfn, f"C{spin}_subset")
            rhs.append(psi4.core.triplet(subset("SO", "OCC"), mu, subset("SO", "VIR"), True, False, False))
    solved = wfn.cphf_solve(rhs, 1.e-10, 100, 0)
    assert wfn.cphf_converged()
    for b, ax in zip(rhs, wfn.cphf_Hx(solved)):
        np.testing.assert_allclose(ax.np, b.np, atol=1.e-9, rtol=0)
    nspin = 2 if unrestricted else 1
    factor = 2.0 if unrestricted else 4.0
    alpha = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            alpha[i, j] = factor * sum(rhs[nspin * i + s].vector_dot(solved[nspin * j + s]) for s in range(nspin))
    return alpha


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("method", ["svwn", "blyp", "pbe0"])
def test_vx_cuest_cpks_solve(method, unrestricted):
    """CPKS converges with the cuEST-grid kernel, and the polarizability matches the host grid."""
    cpu = _polarizability(_wavefunction(method, unrestricted), unrestricted)
    psi4.core.clean_timers()
    wfn = _gpu_wavefunction(method, unrestricted)
    psi4.core.clean_timers()
    gpu = _polarizability(wfn, unrestricted)
    timers = psi4.core.get_timer_dict()
    label = "UV: Form Vx (cuEST)" if unrestricted else "RV: Form Vx (cuEST)"
    assert any(label in k for k in timers)
    err = np.abs(cpu - gpu).max()
    print(f"\n{method} {'UKS' if unrestricted else 'RKS'} alpha host {np.diag(cpu)} cuEST {np.diag(gpu)} "
          f"max diff {err:.3e}")
    np.testing.assert_allclose(gpu, gpu.T, atol=1.e-7)
    np.testing.assert_allclose(gpu, cpu, atol=1.e-3, rtol=1.e-4)


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("unrestricted", [False, True])
def test_vx_cuest_symmetry_cpks(unrestricted):
    """cphf_Hx on a c2v cuEST SCF: mixed AO/SO batches reach the cuEST kernel."""
    wfn = _gpu_wavefunction("pbe0", unrestricted, symmetry="c2v")
    dense = _rotations(wfn, unrestricted, count=1)
    blocked = _rotations(wfn, unrestricted, count=1, blocked=True)
    ref = [m.clone() for x in (dense, blocked) for m in wfn.cphf_Hx(x)]
    _assert_matrices(ref, wfn.cphf_Hx(dense + blocked), atol=1.e-11)
    # A zero (initially converged) RHS shrinks the active batch from the first iteration.
    zero = _rotations(wfn, unrestricted, count=1)
    for m in zero:
        m.zero()
    rhs = dense + blocked + zero
    solved = wfn.cphf_solve(rhs, 1.e-10, 100, 0)
    assert wfn.cphf_converged()
    _assert_matrices(rhs, wfn.cphf_Hx(solved), atol=1.e-10)


_SAPT_DIMER = """
0 1
O   -0.702196054   -0.056060256   0.009942262
H   -1.022193224    0.846775782   -0.011488714
H    0.257521062    0.042121496    0.005218999
--
0 1
O    2.268880784    0.026340101    0.000508029
H    2.645502399   -0.412039965    0.766632411
H    2.641145101   -0.449872874   -0.744894473
units angstrom
"""


@pytest.mark.saptdft
@pytest.mark.dftd4
@uusing("cuest")
@uusing("cuda_cc8")
@uusing("dftd4")
@pytest.mark.cuest
def test_vx_cuest_saptdft_cpks_induction(tmp_path):
    """SAPT(DFT) CPKS induction with the cuEST-grid kernel vs the host-grid kernel."""
    values = {}
    for cuest_xc in (False, True):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.set_output_file(str(tmp_path / f"sapt-{cuest_xc}.out"), False)
        psi4.geometry(_SAPT_DIMER)
        psi4.set_options({
            "basis": "cc-pvdz", "scf_type": "df", "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_GRAC_SHIFT_A": 0.136, "SAPT_DFT_GRAC_SHIFT_B": 0.136,
            "SAPT_DFT_INDUCTION_TYPE": "CPKS", "SAPT_DFT_DO_DHF": True,
            "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL", "USE_CUEST": True, "CUEST_XC": cuest_xc,
            "CUEST_MIXED_PRECISION": False,
        })
        psi4.core.clean_timers()
        psi4.energy("sapt(dft)-d4(i)")
        timers = psi4.core.get_timer_dict()
        assert any("RV: Form Vx (cuEST)" in k for k in timers) == cuest_xc
        values[cuest_xc] = {k: psi4.variable(k) for k in (
            "SAPT IND ENERGY", "SAPT ELST ENERGY", "SAPT EXCH ENERGY", "SAPT TOTAL ENERGY")}
        psi4.core.close_outfile()
    for key in values[False]:
        print(f"\n{key}: host grid {values[False][key]:.10f} cuEST grid {values[True][key]:.10f}")
        assert psi4.compare_values(values[False][key], values[True][key], 6, f"cuEST XC CPKS: {key}")
