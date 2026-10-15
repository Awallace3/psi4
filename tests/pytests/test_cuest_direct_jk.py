"""Conventional four-center cuEST JK, exercised through ordinary energy calls.

These are not DF-J tests. CPU SCF_TYPE=DIRECT is the numerical reference.
Disable GPU XC for the numerical comparisons to isolate the new JK route;
a separate smoke test enables the existing GPU XC route.
"""
import numpy as np
import pytest
import psi4

from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.cuest,
              *using("cuest"), *using("cuda_cc8")]

WATER = """
0 1
O 0 0 0
H 0 1.43233673 1.10715266
H 0 -1.43233673 1.10715266
units bohr
symmetry c1
no_reorient
no_com
"""


def _options(gpu, **extra):
    psi4.core.clean()
    psi4.core.clean_options()
    options = dict(
        basis="cc-pvdz", scf_type="direct", use_cuest=gpu,
        cuest_xc=False, cuest_mixed_precision=False, cuest_direct_jit="off",
        df_scf_guess=False, guess="core", reference="rhf",
        save_jk=True,
        e_convergence=1e-10, d_convergence=1e-9,
        ints_tolerance=1e-14, screening="schwarz",
        dft_radial_points=50, dft_spherical_points=194)
    options.update(extra)
    psi4.set_options(options)


def _energy(functional, gpu, geometry=WATER, **options):
    _options(gpu, **options)
    molecule = psi4.geometry(geometry)
    try:
        energy, wfn = psi4.energy(functional, molecule=molecule, return_wfn=True)
    except RuntimeError as exc:
        if gpu and "cuEST direct JK is unavailable" in str(exc):
            pytest.skip("This cuEST build predates the direct-JK interface")
        raise
    expected = "cuESTDirectJK" if gpu else "DirectJK"
    assert wfn.jk().name() == expected, "DIRECT must not silently select CPU or DF JK"
    return energy, wfn


@pytest.mark.parametrize("functional", ["svwn", "pbe", "m06-l", "b3lyp", "wb97x", "lc-wpbe"])
def test_cuest_direct_dft_energy(functional):
    cpu, _ = _energy(functional, False)
    gpu, _ = _energy(functional, True)
    assert abs(gpu - cpu) < 5e-9, f"{functional}: direct GPU minus CPU = {gpu-cpu:.12g} Eh"


@pytest.mark.parametrize("puream", [True, False])
def test_cuest_direct_cartesian_and_symmetry(puream):
    # Let Psi4 retain molecular symmetry, exercising JK's USO-to-AO adapter.
    geometry = WATER.replace("symmetry c1\n", "")
    cpu, _ = _energy("pbe0", False, geometry, puream=puream)
    gpu, _ = _energy("pbe0", True, geometry, puream=puream)
    assert abs(gpu - cpu) < 5e-9


def test_cuest_direct_uks_energy():
    geometry = "0 2\nO 0 0 0\nH 0 0 1.8\nunits bohr\nsymmetry c1\n"
    cpu, _ = _energy("b3lyp", False, geometry, reference="uhf")
    gpu, _ = _energy("b3lyp", True, geometry, reference="uhf")
    assert abs(gpu - cpu) < 5e-9


def test_cuest_direct_with_gpu_xc():
    # Native host/device quadratures are not asserted identical here.
    psi4.core.clean_timers()
    energy, wfn = _energy("pbe", True, cuest_xc=True)
    assert np.isfinite(energy)
    paths = ["/".join(t["timer_path"]) for t in psi4.core.get_timer_records().values()]
    # cuEST quadrature also supports host-only LibXC builds.
    assert any("cuEST XC: CUDA LibXC Functional" in p or "cuEST XC: Host Functional" in p for p in paths)
    assert wfn.jk().name() == "cuESTDirectJK"


def _matrix_jk(gpu, primary, pairs, alpha=1., beta=0., omega=0., do_k=True, do_j=True):
    _options(gpu)
    try:
        jk = psi4.core.JK.build_JK(primary, None)
    except RuntimeError as exc:
        if gpu and "cuEST direct JK is unavailable" in str(exc):
            pytest.skip("This cuEST build predates the direct-JK interface")
        raise
    assert jk.name() == ("cuESTDirectJK" if gpu else "DirectJK")
    jk.set_do_J(do_j)
    jk.set_do_K(do_k)
    jk.set_do_wK(beta != 0.)
    jk.set_omega_alpha(alpha)
    jk.set_omega_beta(beta)
    jk.set_omega(omega)
    jk.initialize()
    for left, right in pairs:
        jk.C_left_add(psi4.core.Matrix.from_array(left))
        if right is not None:
            jk.C_right_add(psi4.core.Matrix.from_array(right))
    try:
        jk.compute()
        js = [m.np.copy() for m in jk.J()] if do_j else []
        ks = [m.np.copy() for m in jk.K()] if do_k else []
        if not gpu and beta:
            ks = [alpha * k + beta * w.np for k, w in zip(ks, jk.wK())]
        elif not gpu:
            ks = [alpha * k for k in ks]
        return js, ks
    finally:
        jk.finalize()


@pytest.mark.parametrize("alpha,beta,omega", [(1., 0., 0.), (.25, 0., 0.),
                                             (.2, .8, .3), (.25, -.25, .3)])
def test_cuest_direct_density_batch_and_weighted_exchange(alpha, beta, omega):
    _options(False)
    primary = psi4.core.BasisSet.build(psi4.geometry(WATER), "ORBITAL", "cc-pvdz")
    rng = np.random.default_rng(41)
    pairs = [(rng.normal(size=(primary.nbf(), n)) / 5., None) for n in (3, 2)]
    cj, ck = _matrix_jk(False, primary, pairs, alpha, beta, omega)
    gj, gk = _matrix_jk(True, primary, pairs, alpha, beta, omega)
    for cpu, gpu in zip(cj + ck, gj + gk):
        np.testing.assert_allclose(gpu, cpu, atol=1e-9, rtol=0)


def test_cuest_direct_rejects_nonsymmetric_exchange():
    _options(False)
    primary = psi4.core.BasisSet.build(psi4.geometry(WATER), "ORBITAL", "cc-pvdz")
    rng = np.random.default_rng(42)
    pairs = [(rng.normal(size=(primary.nbf(), 3)), rng.normal(size=(primary.nbf(), 3)))]
    with pytest.raises(RuntimeError, match="nonsymmetric exchange"):
        _matrix_jk(True, primary, pairs)


def test_cuest_direct_nonsymmetric_coulomb_uses_symmetric_density_part():
    _options(False)
    primary = psi4.core.BasisSet.build(psi4.geometry(WATER), "ORBITAL", "cc-pvdz")
    rng = np.random.default_rng(43)
    pairs = [(rng.normal(size=(primary.nbf(), 3)), rng.normal(size=(primary.nbf(), 3)))]
    cpu, _ = _matrix_jk(False, primary, pairs, do_k=False)
    gpu, _ = _matrix_jk(True, primary, pairs, do_k=False)
    np.testing.assert_allclose(gpu, cpu, atol=1e-9, rtol=0)


def test_cuest_direct_k_only_and_zero_density():
    _options(False)
    primary = psi4.core.BasisSet.build(psi4.geometry(WATER), "ORBITAL", "cc-pvdz")
    rng = np.random.default_rng(44)
    pairs = [(rng.normal(size=(primary.nbf(), 2)) / 5., None),
             (np.zeros((primary.nbf(), 2)), None)]
    _, cpu = _matrix_jk(False, primary, pairs, do_j=False)
    _, gpu = _matrix_jk(True, primary, pairs, do_j=False)
    for c, g in zip(cpu, gpu):
        np.testing.assert_allclose(g, c, atol=1e-9, rtol=0)
    np.testing.assert_allclose(gpu[1], 0., atol=1e-14, rtol=0)


def test_cuest_direct_rebuilds_changed_operator():
    _options(False)
    primary = psi4.core.BasisSet.build(psi4.geometry(WATER), "ORBITAL", "cc-pvdz")
    coeff = np.random.default_rng(45).normal(size=(primary.nbf(), 2)) / 5.
    _, expected = _matrix_jk(False, primary, [(coeff, None)], .2, .8, .4)
    _options(True)
    try:
        jk = psi4.core.JK.build_JK(primary, None)
    except RuntimeError as exc:
        if "cuEST direct JK is unavailable" in str(exc):
            pytest.skip("This cuEST build predates the direct-JK interface")
        raise
    jk.set_do_wK(True)
    jk.set_omega_alpha(.25)
    jk.set_omega_beta(.75)
    jk.set_omega(.3)
    jk.initialize()
    try:
        jk.C_left_add(psi4.core.Matrix.from_array(coeff))
        jk.compute()
        jk.set_omega_alpha(.2)
        jk.set_omega_beta(.8)
        jk.set_omega(.4)
        jk.compute()
        np.testing.assert_allclose(jk.K()[0].np, expected[0], atol=1e-9, rtol=0)
    finally:
        jk.finalize()
    # Finalize/reinitialize must discard rather than reuse the destroyed plan.
    jk.initialize()
    try:
        jk.compute()
        np.testing.assert_allclose(jk.K()[0].np, expected[0], atol=1e-9, rtol=0)
    finally:
        jk.finalize()
