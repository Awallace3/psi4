# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Production native provider tests; require the parent's rebuilt Psi4.

Independent dense AO/MO contractions, not imported reference tolerances or a
comparison harness. No old checkout, reference files, or CamCASP parity claim.
"""
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.p4util import OptionsState
from psi4.driver.procrouting.isapol_native_response import native_response_from_wavefunction


@contextmanager
def scf_settings(basis, puream=True):
    saved = OptionsState(["BASIS"], ["PUREAM"], ["SCF_TYPE"], ["SCF", "REFERENCE"],
                         ["SCF", "E_CONVERGENCE"], ["SCF", "D_CONVERGENCE"])
    try:
        psi4.set_options({"basis": basis, "puream": puream, "scf_type": "pk",
                          "reference": "rhf", "e_convergence": 1.e-11,
                          "d_convergence": 1.e-11})
        yield
    finally:
        saved.restore()


@pytest.fixture(scope="module")
def water():
    mol = psi4.geometry("""
    0 1
    O 0 0 0
    H 0.757 0 0.586
    H -0.757 0 0.586
    symmetry c1
    no_reorient
    no_com
    """)
    with scf_settings("sto-3g"):
        _, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
    return wfn


def make(wfn, **kwargs):
    policy = dict(caller_converged=True, kernel="no_local", exact_exchange=0.0, local_scale=0.0)
    policy.update(kwargs)
    return native_response_from_wavefunction(wfn, **policy)


def dense_primitives(wfn):
    """Independent four-index AO->MO transformation, then explicit permutations."""
    c = wfn.Ca().np.copy()
    nbf, nmo = c.shape
    ao = core.MintsHelper(wfn.basisset()).ao_eri().np.reshape((nbf,)*4)
    eri = np.einsum("mnrs,mi,nj,rk,sl->ijkl", ao, c, c, c, c, optimize=True)
    no = wfn.nalpha()
    transitions = [(i, a) for a in range(no, nmo) for i in range(no)]
    v = np.array([[eri[i, a, j, b] for j, b in transitions] for i, a in transitions])
    x = np.array([[eri[i, j, a, b] for j, b in transitions] for i, a in transitions])
    y = np.array([[eri[i, b, a, j] for j, b in transitions] for i, a in transitions])
    delta = np.diag([wfn.epsilon_a().np[a]-wfn.epsilon_a().np[i] for i, a in transitions])
    return v, x, y, delta


@pytest.mark.parametrize("a", [0.0, 0.25, 1.0])
def test_native_eri_permutations_and_shared_frequency_solve(water, a):
    model = make(water, exact_exchange=a)
    p = model.provider
    v, x, y, delta = dense_primitives(water)
    assert np.max(np.abs(x-y)) > 1.e-3  # catches accidentally using one exchange twice
    for name, expected in [("coulomb", v), ("exchange_direct", x), ("exchange_transpose", y)]:
        np.testing.assert_allclose(getattr(p, name)().np, expected, rtol=2.e-10, atol=2.e-11)
    h1, h2 = delta+4*v-a*(x+y), delta-a*(x-y)
    np.testing.assert_allclose(p.h1().np, h1, rtol=2.e-10, atol=2.e-11)
    np.testing.assert_allclose(p.h2().np, h2, rtol=2.e-10, atol=2.e-11)
    assert model.representation == "supplied_transition_leg_coordinates"
    assert "not fitted AUX" in model.coordinate_declaration
    for omega in [0., .4, 50.]:
        actual = model.at_frequency(omega)
        expected = np.linalg.solve(h2@h1 + omega**2*np.eye(len(h1)), -4*h2)
        np.testing.assert_allclose(actual.raw_coupled, expected, rtol=2.e-9, atol=2.e-10)
        np.testing.assert_allclose(actual.raw_baseline, expected, rtol=2.e-9, atol=2.e-10)
        residual = (h2@h1+omega**2*np.eye(len(h1)))@actual.raw_coupled+4*h2
        assert np.linalg.norm(residual) < 1.e-8
    np.testing.assert_array_equal(model.at_frequency(.4).raw_coupled, model.at_frequency(.4).raw_coupled)


def p_polynomial_axes(pure):
    """Declared AO order, not coefficients obtained from the collocator.

    Real regular l=1 harmonics are (m=0,+1,-1) = (z,x,y), with
    unit polynomial coefficients under GaussianShell.coef normalization.
    libmints/solidharmonics.cc ipure_gaussian/ipure_standard declare the
    m sequences; libfock/points.cc selects the matching Gau2Grid order.
    Cartesian CCA order is x,y,z regardless of the spherical convention.
    """
    if not pure:
        return [0, 1, 2]
    order = core.libint2_solid_harmonics_ordering().lower()
    assert order in ("gaussian", "standard"), order
    return {"gaussian": [2, 0, 1], "standard": [1, 2, 0]}[order]


def sto_ao_values(wfn, xyz):
    """Independent contracted S/P polynomials in the declared Psi4 AO order."""
    basis = wfn.basisset()
    phi = np.zeros((len(xyz), basis.nbf()))
    geometry = wfn.molecule().geometry().np
    for s in range(basis.nshell()):
        sh = basis.shell(s)
        dr = xyz-geometry[sh.ncenter]
        r2 = np.sum(dr*dr, axis=1)
        radial = sum(sh.coef(p)*np.exp(-sh.exp(p)*r2) for p in range(sh.nprimitive))
        off = sh.function_index
        assert sh.am in (0, 1)
        if sh.am == 0:
            phi[:, off] = radial
        else:
            phi[:, off:off+3] = radial[:, None]*dr[:, p_polynomial_axes(sh.is_pure())]
    return phi


def sp_overlap(basis):
    """Analytic Gaussian-product moments, independent of integral/collocation code."""
    functions = []
    for s in range(basis.nshell()):
        sh = basis.shell(s)
        assert sh.am in (0, 1)
        axes = [None] if sh.am == 0 else p_polynomial_axes(sh.is_pure())
        for axis in axes:
            functions.append((sh, axis, basis.molecule().geometry().np[sh.ncenter]))
    overlap = np.zeros((len(functions), len(functions)))
    for i, (sa, ax, A) in enumerate(functions):
        for j, (sb, bx, B) in enumerate(functions):
            for a in range(sa.nprimitive):
                for b in range(sb.nprimitive):
                    aa, bb = sa.exp(a), sb.exp(b)
                    p = aa+bb
                    P = (aa*A+bb*B)/p
                    moment = 1.
                    if ax is not None:
                        moment *= P[ax]-A[ax]
                    if bx is not None:
                        moment *= P[bx]-B[bx]
                    if ax is not None and ax == bx:
                        moment += 1/(2*p)
                    overlap[i, j] += (sa.coef(a)*sb.coef(b)*(np.pi/p)**1.5
                                      * np.exp(-aa*bb/p*np.sum((A-B)**2))*moment)
    return overlap


@pytest.mark.parametrize("puream", [True, False], ids=["pure-P", "Cartesian-P"])
def test_independent_p_order_collocation_and_overlap(puream):
    # Not the water/STO fixture: two displaced, non-axis-aligned P centers.
    # Cross-center S/P and P/P overlaps detect permutations that same-center
    # P/P identity overlaps cannot. No native provider supplies the oracle.
    mol = psi4.geometry("0 1\nH .2 -.3 .4\nH 1.1 .8 -.5\nunits bohr\n"
                         "symmetry c1\nno_reorient\nno_com")
    with scf_settings("6-31g**", puream):
        basis = core.BasisSet.build(mol, "BASIS", "6-31g**")
    wfn = core.Wavefunction.build(mol, basis)
    ps = [basis.shell(s) for s in range(basis.nshell()) if basis.shell(s).am == 1]
    assert len(ps) == 2 and all(sh.is_pure() == puream for sh in ps)
    xyz = np.array([[.31, -.23, .47], [.7, .2, -.4], [-.5, .8, .1]])
    block = core.BlockOPoints(*[core.Vector.from_array(v) for v in (*xyz.T, np.ones(len(xyz)))],
                             core.BasisExtents(basis, 1.e-12))
    assert block.shells_local_to_global() == list(range(basis.nshell()))
    assert block.functions_local_to_global() == list(range(basis.nbf()))
    collocation = core.BasisFunctions(basis, 128, basis.nbf())
    collocation.compute_functions(block)
    actual = collocation.basis_values()["PHI"].np
    assert actual.shape == (128, basis.nbf())  # only first npoints rows are valid
    np.testing.assert_allclose(actual[:len(xyz)], sto_ao_values(wfn, xyz), rtol=2.e-10, atol=2.e-11)
    np.testing.assert_allclose(core.MintsHelper(basis).ao_overlap().np, sp_overlap(basis),
                               rtol=2.e-10, atol=2.e-11)


def grid_and_local(wfn, kernel, cutoff=1.e-10):
    rng = np.random.default_rng(28)
    xyz = rng.normal(size=(137, 3))*1.5  # crosses the C++ 128-point block boundary
    xyz[-1] = [100., 100., 100.]  # low-density cutoff
    weights = rng.uniform(.001, .08, size=len(xyz))
    weights[3] = 0.
    grid = np.column_stack((xyz, weights))
    mo = sto_ao_values(wfn, xyz)@wfn.Ca().np
    no = wfn.nalpha()
    rho = 2*np.sum(mo[:, :no]**2, axis=1)
    active = rho >= cutoff
    fxc = np.zeros(len(rho))
    # Analytic Slater derivative: E_x=-3/4*(3/pi)^(1/3) rho^(4/3).
    fxc[active] = -(3/np.pi)**(1/3)/3*rho[active]**(-2/3)
    if kernel != "alda_slater":
        name = {"alda_slater_pw92": "XC_LDA_C_PW", "alda_slater_vwn": "XC_LDA_C_VWN"}[kernel]
        libxc = core.LibXCFunctional(name, True)
        inp = {"RHO_A": core.Vector.from_array(np.maximum(rho, cutoff))}
        out = {k: core.Vector(len(rho)) for k in ("V", "V_RHO_A", "V_RHO_A_RHO_A")}
        libxc.compute_functional(inp, out, len(rho), 2)
        fxc[active] += out["V_RHO_A_RHO_A"].np[active]
    t = np.array([mo[:, i]*mo[:, a] for a in range(no, mo.shape[1]) for i in range(no)]).T
    return grid, t.T@((weights*fxc)[:, None]*t)


@pytest.mark.parametrize("kernel", ["alda_slater", "alda_slater_pw92", "alda_slater_vwn"])
@pytest.mark.parametrize("a,b", [(0., 1.), (.25, .75), (1., 0.)])
def test_explicit_numerical_alda_normalization(water, kernel, a, b):
    grid, local = grid_and_local(water, kernel)
    model = make(water, kernel=kernel, exact_exchange=a, local_scale=b, grid=grid)
    p = model.provider
    v, x, y, delta = dense_primitives(water)
    np.testing.assert_allclose(p.local_primitive().np, local, rtol=2.e-9, atol=2.e-11)
    np.testing.assert_allclose(p.h1().np, delta+4*v-a*(x+y)+4*b*local, rtol=2.e-9, atol=2.e-10)
    np.testing.assert_allclose(p.h2().np, delta-a*(x-y), rtol=2.e-10, atol=2.e-11)
    before = p.h1().np.copy()
    grid[:] = np.nan
    np.testing.assert_array_equal(p.h1().np, before)


def test_native_alda_last_block_and_complete_mapping(water):
    grid, expected = grid_and_local(water, "alda_slater")
    def local(g):
        return make(water, kernel="alda_slater", local_scale=1., grid=g).provider.local_primitive().np.copy()
    full = local(grid)
    np.testing.assert_allclose(full, expected, rtol=2.e-9, atol=2.e-11)
    for split in (1, 127, 128, 129):
        np.testing.assert_allclose(full, local(grid[:split])+local(grid[split:]), rtol=2.e-9, atol=2.e-11)
    # A remote point screened by ordinary BlockOPoints must still be a safe,
    # complete native block, with zero density and zero local contribution.
    np.testing.assert_array_equal(local(grid[-1:]), np.zeros_like(expected))


def test_slater_single_point_cutoff_boundary(water):
    # Independent density/Slater derivative; active rho is strictly between
    # cutoff and 2*cutoff, where an overwritten LibXC threshold loses exchange.
    xyz = np.array([[.31, -.23, .47]])
    mo = (sto_ao_values(water, xyz)@water.Ca().np)[0]
    no = water.nalpha()
    rho = 2*np.sum(mo[:no]**2)
    t = np.array([mo[i]*mo[a] for a in range(no, len(mo)) for i in range(no)])
    weight = .07
    expected = -weight*(3/np.pi)**(1/3)/3*rho**(-2/3)*np.outer(t, t)
    assert np.max(np.abs(expected)) > 1.e-8
    grid = np.column_stack((xyz, [weight]))
    active = make(water, kernel="alda_slater", local_scale=1., grid=grid, density_cutoff=rho/1.5)
    np.testing.assert_allclose(active.provider.local_primitive().np, expected, rtol=2.e-9, atol=2.e-11)
    inactive = make(water, kernel="alda_slater", local_scale=1., grid=grid, density_cutoff=rho*1.5)
    np.testing.assert_array_equal(inactive.provider.local_primitive().np, np.zeros_like(expected))


def test_frozen_wavefunction_ignores_ambient_integral_settings(water):
    saved = OptionsState(["INTS_TOLERANCE"], ["SCREENING"])
    threads = core.get_num_threads()
    c, eps, da = water.Ca().np.copy(), water.epsilon_a().np.copy(), water.Da().np.copy()
    names = ("h1", "h2", "coulomb", "exchange_direct", "exchange_transpose", "local_primitive")
    providers = []
    grid, _ = grid_and_local(water, "alda_slater")
    try:
        # Reuse exactly one frozen SCF state, never reconverge under the options.
        for tolerance, screening, nthread in [(1.e-15, "SCHWARZ", 1), (1.e-2, "NONE", 2)]:
            core.set_global_option("INTS_TOLERANCE", tolerance)
            core.set_global_option("SCREENING", screening)
            core.set_num_threads(nthread)
            providers.append(make(water, exact_exchange=.25, local_scale=.75,
                                  kernel="alda_slater", grid=grid).provider)
            assert core.get_global_option("INTS_TOLERANCE") == tolerance
            assert core.get_global_option("SCREENING") == screening
            assert core.get_num_threads() == nthread
        for name in names:
            np.testing.assert_allclose(getattr(providers[0], name)().np, getattr(providers[1], name)().np,
                                       rtol=2.e-12, atol=2.e-13)
        np.testing.assert_array_equal(water.Ca().np, c)
        np.testing.assert_array_equal(water.epsilon_a().np, eps)
        np.testing.assert_array_equal(water.Da().np, da)
    finally:
        saved.restore()
        core.set_num_threads(threads)


@pytest.mark.parametrize("puream", [True, False], ids=["pure-D", "Cartesian-D"])
def test_uniform_high_am_integral_controls(puream):
    # He/cc-pVTZ includes D shells but keeps the ordered-quartet test small.
    mol = psi4.geometry("0 1\nHe 0 0 0\nsymmetry c1\nno_reorient\nno_com")
    with scf_settings("cc-pvtz", puream):
        _, wfn = psi4.energy("hf", molecule=mol, return_wfn=True)
        basis = wfn.basisset()
        ds = [basis.shell(s) for s in range(basis.nshell()) if basis.shell(s).am == 2]
        assert ds and basis.max_am() == 2
        assert all(sh.is_pure() == puream and sh.nfunction == (5 if puream else 6) for sh in ds)
        p = make(wfn, exact_exchange=.25).provider
        v, x, y, delta = dense_primitives(wfn)
        for name, expected in [("coulomb", v), ("exchange_direct", x), ("exchange_transpose", y),
                               ("h1", delta+4*v-.25*(x+y)), ("h2", delta-.25*(x-y))]:
            np.testing.assert_allclose(getattr(p, name)().np, expected, rtol=2.e-10, atol=2.e-11)


# No malformed/mixed-Libint mutation test: BasisSet.shell() returns a copy,
# Libint shells are not exposed, and constructing a mixed basis then building a
# Wavefunction could reach unsafe integral code before this provider's guard.
# Uniform D controls above exercise the supported representations numerically.


def test_fitted_legs_are_explicit_owned_and_not_refitted(water):
    direct = make(water, exact_exchange=.25)
    nov = direct.provider.nocc*direct.provider.nvir
    d = np.random.default_rng(5).normal(size=(nov, 3))
    expected_d = d.copy()
    fitted = make(water, exact_exchange=.25, transition_legs=d,
                  representation="fitted_density_coefficients")
    d[:] = np.nan
    expected = expected_d.T@direct.at_frequency(.3).raw_coupled@expected_d
    assert fitted.representation == "fitted_density_coefficients"
    np.testing.assert_allclose(fitted.at_frequency(.3).raw_coupled, expected, rtol=2.e-10, atol=2.e-10)
    with pytest.raises(FrozenInstanceError):
        fitted.provider = None
    with pytest.raises(ValueError):
        make(water, representation="fitted_density_coefficients")
    with pytest.raises(ValueError):
        make(water, transition_legs=np.eye(nov))
    with pytest.raises(ValueError):
        make(water, transition_legs=np.ones((1, 2)), representation="fitted_density_coefficients")


def test_snapshot_and_no_input_mutation(water):
    c, eps, da = water.Ca().np.copy(), water.epsilon_a().np.copy(), water.Da().np.copy()
    xyz = water.molecule().geometry().np.copy()
    model = make(water)
    before = model.at_frequency(.2).raw_coupled.copy()
    np.testing.assert_array_equal(water.Ca().np, c)
    np.testing.assert_array_equal(water.epsilon_a().np, eps)
    np.testing.assert_array_equal(water.Da().np, da)
    np.testing.assert_array_equal(water.molecule().geometry().np, xyz)
    try:
        water.Ca().np[:] = np.nan
        water.epsilon_a().np[:] = np.nan
        water.Da().np[:] = np.nan
        displaced = xyz.copy(); displaced[0, 0] += .1
        water.molecule().set_geometry(core.Matrix.from_array(displaced))
        for name in ("h1", "h2", "coulomb", "exchange_direct", "exchange_transpose", "local_primitive", "orbitals", "density_alpha"):
            snapshot = getattr(model.provider, name)().np.copy()
            getattr(model.provider, name)().np[:] = np.nan
            np.testing.assert_array_equal(getattr(model.provider, name)().np, snapshot)
        model.provider.energies().np[:] = np.nan
        np.testing.assert_array_equal(model.provider.energies().np, eps)
        np.testing.assert_array_equal(model.at_frequency(.2).raw_coupled, before)
        result = model.at_frequency(.2); result.raw_coupled[:] = np.nan
        np.testing.assert_array_equal(model.at_frequency(.2).raw_coupled, before)
    finally:
        water.Ca().np[:] = c; water.epsilon_a().np[:] = eps; water.Da().np[:] = da
        water.molecule().set_geometry(core.Matrix.from_array(xyz))


@pytest.mark.parametrize("kwargs", [
    {"caller_converged": False}, {"caller_converged": 1}, {"kernel": "camcasp"},
    {"exact_exchange": np.nan}, {"exact_exchange": 1j}, {"exact_exchange": -1.},
    {"local_scale": 1.}, {"density_cutoff": 0.}, {"density_cutoff": np.nextafter(0., 1.)},
    {"density_cutoff": 2*np.nextafter(0., 1.)}, {"max_bytes": 1}, {"max_nov": 1},
    {"max_nov": True}, {"kernel": "alda_slater_pw92"},
    {"kernel": "alda_slater", "grid": np.zeros((3, 3))},
    {"kernel": "alda_slater", "grid": np.array([[0., 0., 0., -1.]])},
    {"kernel": "alda_slater", "grid": np.array([[np.nan, 0., 0., 1.]])},
    {"kernel": "alda_slater", "grid": np.ones((2, 4), dtype=complex)},
    {"transition_legs": np.ones((10, 1), dtype=complex), "representation": "fitted_density_coefficients"},
])
def test_invalid_policy_resources_and_arrays(water, kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        make(water, **kwargs)


@pytest.mark.parametrize("target,value", [("Ca", np.nan), ("Da", .01), ("epsilon_a", np.inf),
                                          ("epsilon_a", -100.)])
def test_native_state_validation(water, target, value):
    obj = getattr(water, target)()
    saved = obj.np.copy()
    try:
        if target == "epsilon_a":
            obj.np[-1] = value
        else:
            obj.np[0, 0] = value
        with pytest.raises((ValueError, RuntimeError)):
            make(water)
    finally:
        obj.np[:] = saved


def test_native_binding_cannot_bypass_grid_or_resource_checks(water):
    args = [water, True, "no_local", .0, .0, None, 1.e-10, 512*1024**2, 512]
    for index, value in [(1, False), (2, "undefined"), (3, float("nan")), (6, np.nextafter(0., 1.)),
                         (6, 2*np.nextafter(0., 1.)), (7, 1), (8, 1)]:
        bad = args.copy(); bad[index] = value
        with pytest.raises((ValueError, RuntimeError)):
            core.NativeResponseProvider(*bad)
    args[2] = "alda_slater"
    args[5] = core.Matrix.from_array(np.ones((2, 3)))
    with pytest.raises((ValueError, RuntimeError)):
        core.NativeResponseProvider(*args)


def test_non_c1_and_unrestricted_rejected():
    # No SCF here: these rejection checks precede orbital/integral allocation.
    for geom in ["0 1\nH 0 0 0\nH 0 0 1", "0 2\nH 0 0 0\nsymmetry c1"]:
        mol = psi4.geometry(geom)
        basis = core.BasisSet.build(mol, "BASIS", "sto-3g")
        wfn = core.Wavefunction.build(mol, basis)
        with pytest.raises((ValueError, RuntimeError)):
            make(wfn)


@pytest.mark.parametrize("omega", [-1., np.nan, np.inf, 1j, 1.e300])
def test_invalid_frequency(water, omega):
    with pytest.raises(ValueError):
        make(water).at_frequency(omega)


def test_preflight_real_dimensions_before_snapshot_or_provider(water, monkeypatch):
    from psi4.driver.procrouting import isapol_native_response as api
    calls = []
    original = api.estimate_response_work
    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('native snapshot/provider entered after resource rejection')
    monkeypatch.setattr(api, 'estimate_response_work', observe)
    monkeypatch.setattr(core.Matrix, 'from_array', forbidden)
    monkeypatch.setattr(core, 'NativeResponseProvider', forbidden)
    nov = water.nalpha() * (water.nmo() - water.nalpha())
    with pytest.raises(ValueError, match='NativeResponseProvider: dense OV resource limit'):
        make(water, kernel='alda_slater', local_scale=1., grid=np.ones((7, 4)), max_nov=nov-1)
    assert calls == [((water.basisset().nbf(), water.nmo(), water.nalpha(), 7),
                      {'max_nov': nov-1, 'algorithm': 'ordered_pairwise'})]


def test_small_native_default_preflight_is_observational(water, monkeypatch):
    from psi4.driver.procrouting import isapol_native_response as api
    from types import SimpleNamespace
    original = api.estimate_response_work
    # Bypass only the new Python gate to model the previous small default path;
    # both runs still use the real provider and every authoritative C++ guard.
    monkeypatch.setattr(api, 'estimate_response_work',
                        lambda *a, **k: SimpleNamespace(require_pass=lambda: None))
    baseline = make(water).at_frequency(.4).raw_coupled.copy()
    calls = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(api, 'estimate_response_work', observe)
    model = make(water)
    np.testing.assert_array_equal(model.at_frequency(.4).raw_coupled, baseline)
    assert len(calls) == 1 and calls[0].passes and calls[0].grid_rows == 0
    assert model.coordinate_declaration.startswith('identity direct OV')


def test_shared_sweep_reproduces_ordered_pairwise(water):
    # The same ordered quartet sweep and the same ordered quadrature, arranged
    # to visit the sweep once. V/X/Y must be bitwise identical because every
    # addend arrives in the same order as the same left-associated product;
    # the blocked BLAS3 local primitive only reorders summation inside a
    # 128-row block, so it is checked against the independent oracle and
    # against the accumulator at rounding level, never claimed bitwise.
    grid, local = grid_and_local(water, "alda_slater_pw92")
    policy = dict(kernel="alda_slater_pw92", exact_exchange=.25, local_scale=.75, grid=grid)
    pairwise = make(water, **policy).provider
    sweep = make(water, algorithm="shared_sweep", **policy).provider
    assert pairwise.algorithm == "ordered_pairwise" and sweep.algorithm == "shared_sweep"
    for name in ("coulomb", "exchange_direct", "exchange_transpose"):
        np.testing.assert_array_equal(getattr(sweep, name)().np, getattr(pairwise, name)().np)
    np.testing.assert_allclose(sweep.local_primitive().np, local, rtol=2.e-9, atol=2.e-11)
    for name in ("local_primitive", "h1", "h2"):
        reference = getattr(pairwise, name)().np
        np.testing.assert_allclose(getattr(sweep, name)().np, reference,
                                   rtol=1.e-13, atol=1.e-13*max(1., np.abs(reference).max()))
    # Holding one J and one K AO operator per transition is what buys the
    # single sweep, and it is declared in the same gated envelope.
    nov = pairwise.nocc*pairwise.nvir
    assert (sweep.planned_bytes - pairwise.planned_bytes
            == 2*nov*water.basisset().nbf()**2*np.dtype(float).itemsize)


def test_shared_sweep_only_moves_its_own_alda_gate(water):
    from psi4.driver.procrouting.isapol_response_preflight import (
        ALDA_WORK_LIMITS, estimate_response_work)
    assert ALDA_WORK_LIMITS == {"ordered_pairwise": 2_000_000_000,
                                "shared_sweep": 64_000_000_000}
    # aug-cc-pVTZ water dimensions of the reference protocol on the full
    # unpruned IsaGrid(99,590): rejected by the accumulator's calibrated
    # limit, admitted by the blocked primitive's own calibrated limit.
    dimensions = dict(nbf=92, nmo=92, nocc=5, grid_rows=173460)
    pairwise = estimate_response_work(**dimensions)
    sweep = estimate_response_work(**dimensions, algorithm="shared_sweep")
    assert pairwise.alda_work == sweep.alda_work == 32_822_968_500
    assert pairwise.ao_work == sweep.ao_work == 31_163_093_760
    assert pairwise.failures == ("ALDA work resource limit",) and not pairwise.passes
    assert sweep.failures == () and sweep.passes
    assert (pairwise.max_grid_rows, sweep.max_grid_rows) == (10569, 338221)
    # Every other limit is identical; no caller argument raises either limit.
    for name in ("nbf_limit", "native_nov_limit", "ao_work_limit", "grid_rows_limit"):
        assert getattr(pairwise, name) == getattr(sweep, name)
    with pytest.raises(TypeError):
        estimate_response_work(**dimensions, alda_work_limit=10**12)


@pytest.mark.parametrize("name", ["", "ordered", "shared", "blas3", None, 3, True])
def test_unknown_response_algorithm_is_never_inferred(water, name):
    from psi4.driver.procrouting.isapol_response_preflight import estimate_response_work
    with pytest.raises(ValueError, match="algorithm"):
        make(water, algorithm=name)
    with pytest.raises(ValueError, match="algorithm"):
        estimate_response_work(2, 2, 1, 0, algorithm=name)
    args = [water, True, "no_local", .0, .0, None, 1.e-10, 512*1024**2, 512, name]
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.NativeResponseProvider(*args)
