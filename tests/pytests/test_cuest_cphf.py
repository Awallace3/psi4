"""Fixed-orbital CPSCF comparisons: separate JK, GEMM, and XC effects."""

import numpy as np
import pytest
import psi4

from addons import uusing

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


@uusing("cuest")
@uusing("cuda_cc8")
@pytest.mark.cuest
@pytest.mark.parametrize("unrestricted", [False, True])
def test_cpscf_gpu_xc_guard(unrestricted):
    wfn = _wavefunction("blyp", unrestricted)
    psi4.set_options({"use_cuest": True, "cuest_xc": True})
    matrices = [wfn.Da().clone()]
    if unrestricted:
        matrices.append(wfn.Db().clone())
    try:
        with pytest.raises(RuntimeError, match="Set CUEST_XC false"):
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
