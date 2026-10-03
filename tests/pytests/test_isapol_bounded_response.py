# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded response runner: reference H2H1 primitives and the explicit native FDDS branch.

Small independent identities, not large reference fixtures. No PFIT, LW,
dispersion or ISA code is used; toy admissions are not molecular acceptance.
"""
from fractions import Fraction
import hashlib
import itertools
import math
import weakref
import numpy as np
import pytest
from psi4 import core
from psi4.driver.procrouting import isapol_bounded_response as backend


def ledger():
    return backend._Ledger(backend.BoundedResources(512*1024**2, 6_000_000_000_000, 64*1024**3))


@pytest.mark.parametrize('field,value', [('max_bytes', True), ('max_work', 6_000_000_000_001),
                                      ('max_io_bytes', 0)])
def test_resource_declarations_fail_closed(field, value):
    args = dict(max_bytes=512*1024**2, max_work=6_000_000_000_000, max_io_bytes=64*1024**3)
    args[field] = value
    with pytest.raises(ValueError):
        backend.BoundedResources(**args)


def test_byte_budget_has_no_fixed_ceiling():
    assert backend.BoundedResources(8*1024**3, 1, 1).max_bytes == 8*1024**3


def test_ledger_never_resets_between_stages():
    budget = backend._Ledger(backend.BoundedResources(100, 100, 100), reserved=20)
    budget.admit('one', 80, 60)
    with pytest.raises(ValueError, match='cumulative work'):
        budget.admit('two', 80, 41)
    budget.admit('three', 80, 40)
    with pytest.raises(ValueError, match='shared numeric'):
        budget.admit('four', 81)
    budget.charge_io(60)
    with pytest.raises(ValueError, match='I/O'):
        budget.charge_io(41)
    assert budget.work == 100 and budget.peak == 100 and budget.io == 60


def test_checkpoint_identity_and_duplicate_refusals(tmp_path):
    store = backend._Store(tmp_path, ledger())
    store.save('a', np.eye(3))
    np.testing.assert_array_equal(store.load('a'), np.eye(3))
    with pytest.raises(ValueError, match='duplicate'):
        store.save('a', np.eye(3))
    np.save(tmp_path/'a.npy', np.ones((3, 3)))
    with pytest.raises(ValueError, match='identity'):
        store.load('a')


def _factor_action(gaps, oo, ov, dual_ov, dual_vv, exchange, rhs, first):
    """Matrix-free H1 (first) or H2 action, one RHS column and AUX row at a time; rows a*nocc+i."""
    nocc, nvir = ov.shape[1:]
    result = gaps[:, None]*rhs
    for col in range(rhs.shape[1]):
        z = rhs[:, col].reshape(nvir, nocc).T
        accum = np.zeros((nocc, nvir))
        for p in range(len(ov)):
            if first:
                accum += 4*ov[p]*np.sum(dual_ov[p]*z)
            x = (oo[p] @ z) @ dual_vv[p].T
            y = (ov[p] @ z.T) @ dual_ov[p]
            accum -= exchange*(x+y if first else x-y)
        result[:, col] += accum.T.reshape(-1)
    return result


@pytest.mark.parametrize('exchange', [0., .25, 1.])
@pytest.mark.parametrize('independent', [False, True])
def test_factor_action_oracle_matches_explicit_four_index_contractions(independent, exchange):
    rng = np.random.default_rng(17)
    nocc, nvir, naux = 2, 3, 4
    raw = rng.normal(size=(naux, nocc+nvir, nocc+nvir))
    mo = (raw+raw.transpose(0, 2, 1))/2
    dual = np.linalg.solve(np.diag([1., 2., 3., 4.]), mo.reshape(naux, -1)).reshape(mo.shape)
    if independent:
        mo, dual = raw[:, :, ::-1], rng.normal(size=mo.shape)
    gaps = np.arange(1., 7.)
    h1, h2 = np.diag(gaps), np.diag(gaps)
    for a, i, b, j in itertools.product(range(nvir), range(nocc), range(nvir), range(nocc)):
        v = sum(mo[p, i, nocc+a]*dual[p, j, nocc+b] for p in range(naux))
        x = sum(mo[p, i, j]*dual[p, nocc+a, nocc+b] for p in range(naux))
        y = sum(mo[p, i, nocc+b]*dual[p, j, nocc+a] for p in range(naux))
        h1[a*nocc+i, b*nocc+j] += 4*v-exchange*(x+y)
        h2[a*nocc+i, b*nocc+j] -= exchange*(x-y)
    factors = (gaps, mo[:, :nocc, :nocc], mo[:, :nocc, nocc:], dual[:, :nocc, nocc:], dual[:, nocc:, nocc:])
    rhs = rng.normal(size=(6, 3))
    np.testing.assert_allclose(_factor_action(*factors, exchange, rhs, True), h1 @ rhs, atol=1e-12)
    np.testing.assert_allclose(_factor_action(*factors, exchange, rhs, False), h2 @ rhs, atol=1e-12)


@pytest.mark.parametrize('p,no,nv', [(3, 1, 4), (3, 4, 1), (2, 1, 1), (3, 2, 3)])
def test_gram_exchange_term_is_one_owned_copy(p, no, nv):
    """y is the same values as the transposed v, in its own buffer even when the reshape could be a view."""
    rng = np.random.default_rng(no*10+nv)
    ov, dual = rng.standard_normal((p, no, nv)), rng.standard_normal((p, no, nv))
    v, y = backend._gram_terms(ov, dual)
    n = no*nv
    expected = v.reshape(nv, no, nv, no).transpose(2, 1, 0, 3).reshape(n, n).copy()
    np.testing.assert_array_equal(y, expected)
    assert y.shape == (n, n) and y.flags.c_contiguous and not np.shares_memory(y, v)
    v += 1.
    np.testing.assert_array_equal(y, expected)


@pytest.mark.parametrize('array', [
    np.arange(12.).reshape(3, 4), np.asfortranarray(np.arange(12.).reshape(3, 4)),
    np.arange(40.).reshape(5, 8)[::2, 1::3], np.arange(7.), np.arange(30.)[::4],
    np.arange(60.).reshape(3, 4, 5).transpose(2, 0, 1), np.arange(12, dtype=np.int64).reshape(4, 3).T],
    ids=['c', 'fortran', 'strided', 'vector', 'strided-vector', 'transposed-3d', 'int'])
def test_streamed_digest_is_the_tobytes_digest(array):
    assert backend._streamed_sha256(array).hexdigest() == hashlib.sha256(array.tobytes()).hexdigest()


@pytest.mark.parametrize('exchange', [0., .25, 1.])
def test_exchange_indexing_matches_independent_factor_actions(exchange):
    rng = np.random.default_rng(818)
    oo, ov, dual, vv = [rng.normal(size=shape) for shape in
                        [(5, 2, 2), (5, 2, 3), (5, 2, 3), (5, 3, 3)]]
    gaps = np.arange(1., 7.)
    h1 = _factor_action(gaps, oo, ov, dual, vv, exchange, np.eye(6), True)
    h2 = _factor_action(gaps, oo, ov, dual, vv, exchange, np.eye(6), False)
    v, y = backend._gram_terms(ov, dual)
    x = backend._exchange_x(oo, [vv.reshape(5, 9)[:, :4], vv.reshape(5, 9)[:, 4:]], 3)
    np.testing.assert_allclose(np.diag(gaps)+4*v-exchange*(x+y), h1, atol=1e-13)
    np.testing.assert_allclose(np.diag(gaps)-exchange*(x-y), h2, atol=1e-13)
    with pytest.raises(ValueError, match='coverage'):
        backend._exchange_x(oo, [vv.reshape(5, 9)[:, :4]], 3)


@pytest.mark.parametrize('omega', [0., .7])
def test_response_uses_distinct_target_and_anchor_legs(tmp_path, omega):
    store = backend._Store(tmp_path, ledger())
    rng = np.random.default_rng(231)
    a, b = rng.normal(size=(6, 6)), rng.normal(size=(6, 6))
    h1, h2 = a.T@a+np.eye(6), b.T@b+np.eye(6)
    d, anchor = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))
    for name, value in [('h1', h1), ('h2', h2), ('target', d), ('anchor', anchor)]:
        store.save(name, value)
    response = backend._H2H1Response(store, (4, 2, 3), 3)
    try:
        coefficient, raw, residual = response.solve(omega)
    finally:
        response.close()
    effective = h1+omega**2*np.linalg.inv(h2)
    np.testing.assert_allclose(coefficient, -4*d.T@np.linalg.solve(effective, d), atol=1e-12)
    np.testing.assert_allclose(raw, 4*anchor.T@np.linalg.solve(effective, anchor), atol=1e-12)
    assert residual < 1e-12
    assert backend._stability(store, 6)['h1']['minimum_symmetric_eigenvalue'] > 0


def test_minimum_eigenvalue_finds_modes_orthogonal_to_symmetric_starts():
    n = 120
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(n, n)))
    q[:, 0] -= q[:, 0].mean()  # lowest mode orthogonal to the all-ones vector
    q, _ = np.linalg.qr(q)
    values = np.linspace(.2, 30., n)
    a = (q*values)@q.T
    a = (a+a.T)*.5
    assert abs(backend._minimum_eigenvalue(a)-values[0]) < 1e-12
    assert backend._minimum_eigenvalue(a-.3*np.eye(n)) <= 0


def test_instability_is_not_repaired(tmp_path):
    store = backend._Store(tmp_path, ledger())
    store.save('h1', np.diag([-1., 1.]))
    with pytest.raises(ValueError, match='stability'):
        backend._stability(store, 2)






def _aux_basis(centres, shells, representation=core.IsaBasisRepresentation.Cartesian):
    """Unit-exponent, unit-coefficient AUX shells given as (centre, l)."""
    data = []
    for centre, l in shells:
        shell = core.IsaGaussianShell()
        shell.centre, shell.l, shell.exponents, shell.coefficients = centre, l, [1.], [1.]
        data.append(shell)
    return core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux, representation, centres, data)


def _slater_pw92_fxc(rho):
    """Independent unpolarized Slater + PW92 kernel, analytic in rs.

    f_c = rs/(9*rho) * (rs*e_c''(rs) - 2*e_c'(rs)), original PW92 table-I A
    (also CamCASP dft_Sx_PW92c.F90:147); .0310907 belongs to the modified
    parameterization, not XC_LDA_C_PW.
    """
    rs = (3/(4*np.pi*rho))**(1/3)
    A, alpha = .031091, .21370
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, .49294
    q = 2*A*(b1*np.sqrt(rs)+b2*rs+b3*rs**1.5+b4*rs**2)
    dq = 2*A*(b1/(2*np.sqrt(rs))+b2+1.5*b3*np.sqrt(rs)+2*b4*rs)
    ddq = 2*A*(-b1/(4*rs**1.5)+.75*b3/np.sqrt(rs)+2*b4)
    dl = -dq/(q*(q+1))
    ddl = -ddq/(q*(q+1))+dq**2*(2*q+1)/(q*q*(q+1)**2)
    de = -2*A*(alpha*np.log1p(1/q)+(1+alpha*rs)*dl)
    dde = -2*A*(2*alpha*dl+(1+alpha*rs)*ddl)
    return -(3/np.pi)**(1/3)/3*rho**(-2/3) + rs/(9*rho)*(rs*dde-2*de)


@pytest.mark.parametrize('cutoff', [0., np.exp(-2.), np.nextafter(np.exp(-2.), np.inf)])
def test_kernel_floors_caps_and_screens_the_full_grid(cutoff):
    """Every grid row enters; the mask keeps shell pairs at exactly the cutoff."""
    from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
    # Equal unit exponents: the normalized-s screen of the pair is exp(-R**2/2).
    aux = _aux_basis([[0., 0., 0.], [2., 0., 0.]], [(0, 0), (1, 0)])
    grid = np.array([[0., 0., 0., .2], [.5, .1, 0., .3], [1., 0., .2, .4],
                     [2., 0., 0., .1], [3.5, 0., 0., .25]])
    density = np.array([.3, -.05])
    smoothing = KernelSmoothing(1e-3, 2., .1, .1, 'CONSTANT')
    chi = np.asarray(aux.evaluate(grid[:, :3].tolist()))
    rho = chi @ density
    assert (rho < smoothing.rho_epsilon).any()  # negative rows are floored, not dropped
    fxc = _slater_pw92_fxc(np.maximum(rho, smoothing.rho_epsilon))
    assert (np.abs(fxc) > smoothing.f_max).any()  # and the cap is active
    expected = chi.T @ ((grid[:, 3]*np.clip(fxc, -smoothing.f_max, smoothing.f_max))[:, None]*chi)
    if cutoff > np.exp(-2.):
        expected[0, 1] = expected[1, 0] = 0.
    actual = backend._kernel(aux, density, grid, smoothing, cutoff, ledger())
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize('representation,widths', [
    (core.IsaBasisRepresentation.Cartesian, [1, 15, 6, 3]),
    (core.IsaBasisRepresentation.Spherical, [1, 9, 5, 3])])
def test_kernel_screens_whole_mixed_shell_blocks(representation, widths):
    from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
    centres = np.array([[0., 0., 0.], [6., 0., 0.], [1., 0., 0.]])
    sites = [0, 1, 0, 2]
    aux = _aux_basis(centres.tolist(), list(zip(sites, [0, 4, 2, 1])), representation)
    grid = np.array([[0., .2, .3, .1], [.5, .1, 0., .3], [1., 0., .2, .4], [2., .1, 0., .1],
                     [6., .3, .2, .2], [5., .1, -.1, .3], [1., -.2, .4, .5]])
    density = np.linspace(.1, 2., sum(widths))
    smoothing = KernelSmoothing(1e-3, 400., .1, .1, 'CONSTANT')
    chi = np.asarray(aux.evaluate(grid[:, :3].tolist()))
    fxc = _slater_pw92_fxc(np.maximum(chi @ density, smoothing.rho_epsilon))
    expected = chi.T @ ((grid[:, 3]*np.clip(fxc, -400., 400.))[:, None]*chi)
    offsets = np.cumsum([0]+widths)
    for i, j in itertools.product(range(4), repeat=2):
        distance = centres[sites[i]]-centres[sites[j]]
        if np.exp(-distance @ distance/2) < np.exp(-2.):
            expected[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = 0.
    actual = backend._kernel(aux, density, grid, smoothing, np.exp(-2.), ledger())
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)



GEOMETRIES = {'water': 'O 0 0 0\nH .757 0 .586\nH -.757 0 .586',
              'ammonia': 'N 0 0 .116\nH 0 .939 -.271\nH .813197854 -.4695 -.271\nH -.813197854 -.4695 -.271'}


def _recipe(molecule, name='aug-cc-pVTZ-RI', representation='Cartesian'):
    """Declared AUX recipe from a Psi4 shipped basis (Psi4 primitive data)."""
    from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe
    basis = core.BasisSet.build(molecule, 'DF_BASIS_SCF', name, puream=0)
    shells = []
    for i in range(basis.nshell()):
        s = basis.shell(i)
        shells.append(ShellRecipe(int(basis.shell_to_center(i)), int(s.am),
            tuple(s.exp(k) for k in range(s.nprimitive)), tuple(s.coef(k) for k in range(s.nprimitive))))
    return BasisRecipe(name, 'Psi4 shipped basis', representation, tuple(map(tuple, molecule.geometry().np)),
                       tuple(shells))


def _sites(molecule):
    sites = []
    for label, origin in zip(('A', 'H1', 'H2', 'H3'), molecule.geometry().np):
        site = core.IsaMultipoleSite()
        site.label, site.origin, site.rank = label, list(origin), 4
        sites.append(site)
    return sites


def _resources(max_bytes=512*1024**2, max_work=6_000_000_000_000, max_io_bytes=64*1024**3):
    return backend.BoundedResources(max_bytes, max_work, max_io_bytes)


def _response_args(molecule, radial=20, spherical=50):
    from psi4.driver.procrouting.isapol_native import Quadrature
    from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = radial, spherical
    grid = core.IsaGrid(molecule.clone(), options)
    return dict(caller_converged=True, quadrature=Quadrature.from_casimir(core.CasimirGrid(2, .5)),
                response_grid=np.column_stack((grid.x(), grid.y(), grid.z(), grid.w())),
                smoothing=KernelSmoothing(1e-8, 1000., .01, 1., 'FD'), shell_cutoff=1e-7,
                charge_penalty=1000., anchor_metric_damping=.0005, resources=_resources(), scf_correction='NONE')


def _pbe0(geometry, options):
    import psi4
    core.be_quiet()
    core.clean_options()
    molecule = psi4.geometry(geometry)
    psi4.set_options(options)
    _, wfn = psi4.energy('pbe0', molecule=molecule, return_wfn=True)
    return wfn


TOY = {'basis': 'sto-3g', 'reference': 'rks', 'scf_type': 'pk', 'e_convergence': 1e-10, 'd_convergence': 1e-10,
       'dft_radial_points': 50, 'dft_spherical_points': 110}


@pytest.fixture(scope='module', params=('water', 'ammonia'))
def small_water(request):
    """Response-only toy: PBE0/sto-3g, declared Cartesian aug-cc-pVTZ-RI AUX, rank-4 DF-centre sites."""
    wfn = _pbe0('0 1\n'+GEOMETRIES[request.param]+'\nsymmetry c1\nno_com\nno_reorient', TOY)
    molecule = wfn.molecule()
    return wfn, _recipe(molecule), dict(_response_args(molecule), sites=_sites(molecule))


def _runner(wfn, recipe, args, tmp_path, **extra):
    keys = ('caller_converged', 'quadrature', 'response_grid', 'smoothing', 'shell_cutoff',
            'charge_penalty', 'anchor_metric_damping', 'resources', 'scf_correction')
    keywords = {k: args[k] for k in keys}
    keywords.update(extra)
    return backend.BoundedResponse(wfn, recipe, args['sites'], scratch_directory=tmp_path, **keywords)


def _fdds(**extra):
    return dict(response='native_fdds', fdds=backend.NativeFDDSOptions(1, 1 << 30), **extra)


def _nothing_left(path):
    return not list(path.iterdir())


# ---------------------------------------------------------------- reference H2H1 runner ----

@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_response_runner_lifecycle_and_owned_results(small_water, tmp_path):
    from psi4.driver.procrouting.isapol_distribution import analytic_df_moments
    wfn, recipe, args = small_water
    frequencies = args['quadrature'].frequencies
    runner = _runner(wfn, recipe, args, tmp_path)
    assert runner.response == 'reference_h2h1' and runner.lw_residual_policy == 'production'
    with runner as response:
        assert len(list(tmp_path.iterdir())) == 1
        with pytest.raises(RuntimeError, match='prepared'):
            response.solve(frequencies[0])
        produced = []
        def provider(ledger):
            q = analytic_df_moments(recipe, response.sites, 4)
            produced.append(weakref.ref(q))
            return q
        response.prepare(provider)
        assert produced[0]() is None  # the runner held the only Q reference
        with pytest.raises(RuntimeError, match='prepare once'):
            response.prepare()
        with pytest.raises(ValueError, match='declared quadrature'):
            response.solve(frequencies[-1]+1.)
        solved = [response.solve(omega) for omega in frequencies]
        response.release()
        with pytest.raises(RuntimeError, match='unreleased'):
            response.solve(frequencies[0])
    assert _nothing_left(tmp_path)
    with pytest.raises(RuntimeError, match='single-use'):
        runner.__enter__()
    p, q = response.dimensions[0], 25*len(args['sites'])
    assert response.partition['model'] == 'df_centre_analytic'
    for omega, result in zip(frequencies, solved):
        assert result.frequency == omega and result.model == response.model and result.diagnostics == ()
        assert result.target_response.shape == (p, p) and result.nonlocal_response.shape == (q, q)
        assert np.isfinite(result.nonlocal_response).all() and result.residual < 1e-10


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_response_runner_failed_solve_cleans_scratch(small_water, tmp_path, monkeypatch):
    wfn, recipe, args = small_water
    def fail(self, omega):
        raise RuntimeError('injected solve failure')
    monkeypatch.setattr(backend._H2H1Response, 'solve', fail)
    with pytest.raises(RuntimeError, match='injected solve'):
        with _runner(wfn, recipe, args, tmp_path) as response:
            response.prepare()
            response.solve(args['quadrature'].frequencies[0])
    assert response._operator is None and _nothing_left(tmp_path)


@pytest.mark.parametrize('failure', ['provider', 'late'])
@pytest.mark.parametrize('response', [{}, _fdds()], ids=['reference', 'fdds'])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_failed_prepare_is_final(small_water, tmp_path, monkeypatch, response, failure):
    """A prepare that raised is never retried and never reaches solve: late failures leave H1/H2 checkpoints
    (reference stability gate) or a refused native instance (FDDS metric guard) behind."""
    wfn, recipe, args = small_water
    omega = args['quadrature'].frequencies[0]
    def provider(ledger):
        raise RuntimeError('injected provider failure')
    if failure == 'late':
        provider = None
        if response:
            monkeypatch.setattr(backend, 'FDDS_METRIC_RCOND_CUTOFF', 1.)
        else:
            def unstable(store, n):
                assert {'h1', 'h2'} <= set(store.records)
                raise ValueError('h1 reciprocity/stability gate')
            monkeypatch.setattr(backend, '_stability', unstable)
    with _runner(wfn, recipe, args, tmp_path, **response) as runner:
        with pytest.raises((RuntimeError, ValueError), match='injected provider|stability gate|metric resolution'):
            runner.prepare(provider)
        with pytest.raises(RuntimeError, match='solve requires a prepared'):
            runner.solve(omega)
        with pytest.raises(RuntimeError, match='prepare once'):
            runner.prepare()
        assert runner._operator is None and runner._fdds is None
    assert _nothing_left(tmp_path)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_reference_preflight_mirrors_the_first_solve(small_water, tmp_path, monkeypatch):
    """The entry preflight is the larger of the operator-formation and node admissions of the first solve,
    so one byte less is refused at entry, before any factor or scratch. The toy's state snapshot outweighs
    its operator, so both plans are inflated by the same constant to make the sweep bind the budget."""
    wfn, recipe, args = small_water
    frequencies = args['quadrature'].frequencies
    plan_bytes, extra = backend._H2H1Response.plan_bytes, 64*1024**2
    def inflated(n, p, q):
        retained, construction, node = plan_bytes(n, p, q)
        return retained, construction+extra, node+extra
    monkeypatch.setattr(backend._H2H1Response, 'plan_bytes', staticmethod(inflated))
    with _runner(wfn, recipe, args, tmp_path) as runner:
        runner.prepare()
        for omega in frequencies:
            runner.solve(omega)
    stages = {}
    for stage in runner.ledger.stages:
        stages[stage['stage']] = max(stages.get(stage['stage'], 0), stage['numeric_bytes'])
    p, no, nv = runner.dimensions
    retained, construction, node = inflated(no*nv, p, 25*len(args['sites']))
    assert stages['H2H1 response operator'] == retained+construction+runner._retained
    assert stages['frequency response'] == retained+node+runner._retained
    plan = max(stages['H2H1 response operator'], stages['frequency response'])
    n, q = no*nv, 25*len(args['sites'])
    # The former preflight did not mirror these admissions (and was 8 (1536 n - n^2) bytes low for
    # n < 1536): plan - 1 passed it.
    assert n < 1536 and plan-1 >= 8*(3*n*n+4*n*(p+q)+p*p+q*q+16*n)+8*1024**2+runner._retained
    def forbidden(*a, **k):
        pytest.fail('factor construction happened before the preflight refusal')
    monkeypatch.setattr(backend, 'native_plain_df_operators', forbidden)
    with pytest.raises(ValueError, match='complete frequency plan'):
        with _runner(wfn, recipe, dict(args, resources=_resources(plan-1)), tmp_path):
            pass
    with _runner(wfn, recipe, dict(args, resources=_resources(plan)), tmp_path):
        pass
    assert _nothing_left(tmp_path)


@pytest.mark.parametrize('supplied', [False, True], ids=['analytic', 'supplied'])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_charges_both_live_q_copies(small_water, tmp_path, supplied):
    """moments.values and the owned FDDS copy are live together; the analytic producer is admitted for its
    own value and copy before it runs."""
    from psi4.driver.procrouting.isapol_distribution import analytic_df_moments
    wfn, recipe, args = small_water
    provider = (lambda ledger: analytic_df_moments(recipe, args['sites'], 4)) if supplied else None
    with _runner(wfn, recipe, args, tmp_path, **_fdds()) as runner:
        runner.prepare(provider)
    stages = {s['stage']: s['numeric_bytes'] for s in runner.ledger.stages}
    qp = 8*25*len(args['sites'])*runner.dimensions[0]
    assert stages['retained distributed moments'] == stages['native FDDS inputs']+2*qp
    assert ('distributed moments' in stages) is not supplied
    if not supplied:
        assert stages['distributed moments'] == stages['native FDDS inputs']+2*qp
    assert _nothing_left(tmp_path)


@pytest.mark.parametrize('response', [{}, _fdds()], ids=['reference', 'fdds'])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_missing_scf_seal_is_not_manufactured(small_water, tmp_path, monkeypatch, response):
    wfn, recipe, args = small_water
    monkeypatch.delattr(wfn, '_scf_convergence_evidence')
    with pytest.raises(ValueError, match='convergence evidence'):
        with _runner(wfn, recipe, args, tmp_path, **response):
            pass
    assert _nothing_left(tmp_path)


@pytest.mark.parametrize('response', [{}, _fdds()], ids=['reference', 'fdds'])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_work_preflight_rejects_before_factor_construction(small_water, tmp_path, monkeypatch, response):
    wfn, recipe, args = small_water
    def forbidden(*args, **kwargs):
        pytest.fail('factor construction happened before work refusal')
    monkeypatch.setattr(backend, 'native_plain_df_operators', forbidden)
    monkeypatch.setattr(backend, 'native_plain_density', forbidden)
    args = dict(args, resources=_resources(max_work=1))
    with pytest.raises(ValueError, match='work resource limit'):
        with _runner(wfn, recipe, args, tmp_path, **response) as runner:
            runner.prepare()
    assert _nothing_left(tmp_path)


@pytest.mark.parametrize('response', [{}, _fdds()], ids=['reference', 'fdds'])
def test_late_failure_cleans_owned_checkpoints(small_water, tmp_path, monkeypatch, response):
    from psi4.driver.procrouting.isapol_logging import StageLog
    messages = []
    wfn, recipe, args = small_water
    def fail(*args, **kwargs):
        raise RuntimeError('injected kernel failure')
    monkeypatch.setattr(backend, '_kernel', fail)
    with pytest.raises(RuntimeError, match='injected kernel'):
        with _runner(wfn, recipe, args, tmp_path, log=StageLog(3, writer=messages.append), **response) as runner:
            runner.prepare()
    assert 'Stage: Full-grid ALDA kernel' in ''.join(messages)
    assert 'Stage complete: Full-grid ALDA kernel' not in ''.join(messages)
    assert _nothing_left(tmp_path)


@pytest.mark.parametrize('key,value,error,match', [
    ('caller_converged', False, ValueError, 'caller_converged'),
    ('scf_correction', 'DECLARED_MULTPOLE_AC', ValueError, 'NONE or FIXED_GRAC'),
    ('anchor_metric_damping', True, ValueError, 'anchor_metric'), ('shell_cutoff', -1., ValueError, 'shell_cutoff'),
    ('response', 'fdds', ValueError, 'response must be'), ('response', 'native_fdds', TypeError, 'NativeFDDSOptions'),
    ('fdds', backend.NativeFDDSOptions(1, 1), ValueError, 'only to response=native_fdds'),
])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_public_declaration_refusals(small_water, tmp_path, key, value, error, match):
    wfn, recipe, args = small_water
    keywords = {} if key in ('response', 'fdds') else None
    with pytest.raises(error, match=match):
        if keywords is None:
            runner = _runner(wfn, recipe, dict(args, **{key: value}), tmp_path)
        else:
            runner = _runner(wfn, recipe, args, tmp_path, **{key: value})
        with runner:
            pass
    assert _nothing_left(tmp_path)


@pytest.mark.parametrize('field,value', [('nthread', 0), ('nthread', True), ('disk_bytes', 0),
                                         ('disk_bytes', 1.), ('subalgo', 'DISK')])
def test_native_fdds_options_fail_closed(field, value):
    args = dict(nthread=1, disk_bytes=1, subalgo='OUT_OF_CORE')
    args[field] = value
    with pytest.raises(ValueError):
        backend.NativeFDDSOptions(**args)


# ---------------------------------------------------------------- native FDDS bridge and algebra ----

def _raw_metric(primary, raw):
    """Independent Libint J_raw over the bridge basis (MintsHelper of another basis; see native_auxiliary)."""
    zero = core.BasisSet.zero_ao_basis_set()
    return np.asarray(core.MintsHelper(primary).ao_eri(raw, zero, raw, zero)).reshape(raw.nbf(), raw.nbf())


@pytest.mark.parametrize('representation,shape', [(core.IsaBasisRepresentation.Cartesian, (35, 35)),
                                                  (core.IsaBasisRepresentation.Spherical, (25, 35))])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_native_auxiliary_bridge_reproduces_the_declared_metric(small_water, representation, shape):
    wfn = small_water[0]
    centres = np.asarray(wfn.molecule().geometry()).tolist()
    aux = _aux_basis(centres, [(0, 0), (0, 1), (1, 2), (2, 3), (0, 4)], representation)
    provider = core.IsaAuxCoulomb(aux)
    raw, transform = provider.native_auxiliary()
    T = transform.to_array()
    assert T.shape == shape and not raw.has_puream() and raw.molecule().natom() == 3
    declared = np.asarray(provider.metric())
    np.testing.assert_allclose(T @ _raw_metric(wfn.basisset(), raw) @ T.T, declared,
                               rtol=0, atol=2e-14*np.abs(declared).max())
    # Fresh caller-owned copies: changing one returned map changes neither metric() nor a later map.
    transform.np[:] = 0.
    again = provider.native_auxiliary()[1].to_array()
    np.testing.assert_array_equal(again, T)
    np.testing.assert_array_equal(np.asarray(provider.metric()), declared)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_native_plain_density_is_the_factor_density(small_water):
    from psi4.driver.procrouting.isapol_basis import adapt_main
    from psi4.driver.procrouting.isapol_native_factors import native_plain_df_operators, native_plain_density
    wfn, recipe = small_water[:2]
    auxiliary, main = recipe.build('MolecularAux'), adapt_main(wfn, caller_converged=True)
    coefficients = main.transform @ np.asarray(wfn.Ca())
    no = wfn.nalpha()
    ops = native_plain_df_operators(auxiliary, main.basis, coefficients, np.asarray(wfn.epsilon_a()).copy(),
                                    nocc=no, shell_count=len(recipe.shells), tile_columns=512)
    density, planned = native_plain_density(auxiliary, main.basis, coefficients, nocc=no,
                                            shell_count=len(recipe.shells))
    np.testing.assert_array_equal(density, ops.plain_density_coefficients)
    assert not density.flags.writeable and planned < ops.construction_planned_bytes
    with pytest.raises(ValueError, match='byte resource'):
        native_plain_density(auxiliary, main.basis, coefficients, nocc=no, shell_count=len(recipe.shells),
                             max_bytes=planned-1)


def _monomer(wfn, raw, transform, tmp_path, nthread=1):
    no = wfn.nalpha()
    C, eps = np.asarray(wfn.Ca()), np.asarray(wfn.epsilon_a())
    req = core.FDDS_Monomer.requirement(wfn.basisset(), raw, no, C.shape[1]-no, transform.rows(), True,
                                        'OUT_OF_CORE', nthread)
    return core.FDDS_Monomer(wfn.basisset(), raw, core.Matrix.from_array(C[:, :no]), core.Matrix.from_array(C[:, no:]),
                             core.Vector.from_array(eps[:no]), core.Vector.from_array(eps[no:]), True,
                             memory_bytes=req['memory_bytes'], disk_bytes=req['disk_bytes'],
                             scratch_dir=str(tmp_path), nthread=nthread, aux_transform=transform)


def _plain_operands(wfn, recipe):
    """Independent declared (P|ia) (runner MAIN-order factors), gaps and J from IsaAuxCoulomb."""
    from psi4.driver.procrouting.isapol_basis import adapt_main
    from psi4.driver.procrouting.isapol_native_factors import native_plain_df_operators, native_plain_density
    auxiliary, main = recipe.build('MolecularAux'), adapt_main(wfn, caller_converged=True)
    coefficients = main.transform @ np.asarray(wfn.Ca())
    no = wfn.nalpha()
    ops = native_plain_df_operators(auxiliary, main.basis, coefficients, np.asarray(wfn.epsilon_a()).copy(),
                                    nocc=no, shell_count=len(recipe.shells), tile_columns=512)
    B = np.asarray(ops._ov).transpose(0, 2, 1).reshape(auxiliary.nfunction, -1)  # columns a*nocc+i
    return auxiliary, B, np.asarray(ops._gaps), np.asarray(core.IsaAuxCoulomb(auxiliary).metric()), \
        np.asarray(ops.plain_density_coefficients)


def _dense_zero_exchange(B, gaps, J, W, omega):
    b = np.linalg.solve(J, B)
    chi0 = (b*(-4*gaps/(gaps**2+omega**2))) @ b.T
    chi = np.linalg.solve(np.eye(len(J))-chi0 @ W, chi0)
    return .5*(chi+chi.T)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_zero_exchange_kernel_factor_is_discriminated(small_water, tmp_path):
    """Well-conditioned spherical (rectangular T) declared AUX, x_alpha=0: native
    chi = sym((I - chi0 W)^-1 chi0) for W = 0 (uncoupled, tests the declared B) and W = J + cK."""
    wfn, _, args = small_water
    recipe = _recipe(wfn.molecule(), 'cc-pVDZ-RI', 'Spherical')
    auxiliary, B, gaps, J, density = _plain_operands(wfn, recipe)
    K = backend._kernel(auxiliary, density, args['response_grid'], args['smoothing'], args['shell_cutoff'],
                        backend._Ledger(_resources()))
    raw, transform = core.IsaAuxCoulomb(auxiliary).native_auxiliary()
    monomer = _monomer(wfn, raw, transform, tmp_path)
    assert monomer.model()['metric_lu_rcond'] > 1e-8 and monomer.model()['metric_dropped'] == 0
    for omega in args['quadrature'].frequencies:
        results = {}
        for c in (None, 0., .75, 1.):
            W = np.zeros_like(J) if c is None else J+c*K
            native = monomer.form_coefficient_response(omega, 0., core.Matrix.from_array(W))['response'].to_array()
            expected = _dense_zero_exchange(B, gaps, J, W, omega)
            np.testing.assert_allclose(native, expected, rtol=0, atol=1e-10*np.abs(expected).max())
            results[c] = native
        scale = np.abs(results[.75]).max()
        for c in (None, 0., 1.):
            assert np.abs(results[c]-results[.75]).max() > 1e-4*scale


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_zero_exchange_matches_reference_h2h1_with_plain_legs(small_water, tmp_path):
    """Same well-conditioned limit through the reference code: H1/H2 from _assemble at cx = 0
    with plain legs D = (J^-1 B)^T, solved by _H2H1Response, equals native chi for W = J + cK."""
    from psi4.driver.procrouting.isapol_basis import adapt_main
    from psi4.driver.procrouting.isapol_native_factors import native_plain_df_operators
    wfn, _, args = small_water
    recipe = _recipe(wfn.molecule(), 'cc-pVDZ-RI', 'Spherical')
    auxiliary, main = recipe.build('MolecularAux'), adapt_main(wfn, caller_converged=True)
    no = wfn.nalpha()
    ops = native_plain_df_operators(auxiliary, main.basis, main.transform @ np.asarray(wfn.Ca()),
                                    np.asarray(wfn.epsilon_a()).copy(), nocc=no, shell_count=len(recipe.shells),
                                    tile_columns=512)
    p, nv = auxiliary.nfunction, wfn.nmo()-no
    J = np.asarray(core.IsaAuxCoulomb(auxiliary).metric())
    K = backend._kernel(auxiliary, np.asarray(ops.plain_density_coefficients), args['response_grid'],
                        args['smoothing'], args['shell_cutoff'], backend._Ledger(_resources()))
    # Plain legs (no charge constraint), rows a*nocc+i like the reference OV fits.
    plain = np.column_stack(ops._dual_ov).reshape(p, no, nv).transpose(0, 2, 1).reshape(p, -1).T
    raw, transform = core.IsaAuxCoulomb(auxiliary).native_auxiliary()
    monomer = _monomer(wfn, raw, transform, tmp_path)
    for c in (0., .75):
        (tmp_path/f'reference{c}').mkdir()
        store = backend._Store(tmp_path/f'reference{c}', ledger())
        for name, value in (('gaps', ops._gaps), ('oo', ops._oo), ('ov', ops._ov), ('target', plain),
                            ('anchor', plain), ('kernel', K)):
            store.save(name, np.asarray(value))
        for kind in ('ov', 'vv'):
            for i, tile in enumerate(getattr(ops, '_dual_'+kind)):
                store.save(f'dual{kind}{i}', tile)
        backend._assemble(store, (p, no, nv), (len(ops._dual_ov), len(ops._dual_vv)), 0., c)
        reference = backend._H2H1Response(store, (p, no, nv), p)
        try:
            for omega in args['quadrature'].frequencies:
                expected, _, residual = reference.solve(omega)
                native = monomer.form_coefficient_response(omega, 0., core.Matrix.from_array(J+c*K))
                assert residual < 1e-12
                np.testing.assert_allclose(native['response'].to_array(), expected, rtol=0,
                                           atol=1e-10*np.abs(expected).max())
        finally:
            reference.close()


@pytest.mark.parametrize('wrong', ['row_permuted', 'main_order'])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_wrong_orbital_rows_are_refused_before_native_inputs(small_water, tmp_path, monkeypatch, wrong):
    """A state whose AO-order orbitals are row-permuted, or in the DALTON MAIN order, is refused
    by the bitwise Ca/epsilon_a equality before any native input, scratch or integral."""
    from psi4.driver.procrouting.isapol_basis import adapt_main
    wfn, recipe, args = small_water
    C = np.asarray(wfn.Ca())
    bad = np.roll(C, 1, axis=0) if wrong == 'row_permuted' else adapt_main(wfn, caller_converged=True).transform @ C
    assert bad.shape == C.shape and not np.array_equal(bad, C)
    original = backend.native_restricted_state_from_wavefunction

    class WrongRows:
        def __init__(self, state):
            self._state = state
        def __getattr__(self, name):
            return getattr(self._state, name)
        def orbitals(self):
            return core.Matrix.from_array(bad)

    monkeypatch.setattr(backend, 'native_restricted_state_from_wavefunction',
                        lambda *a, **k: WrongRows(original(*a, **k)))
    def forbidden(*a, **k):
        pytest.fail('native requirement evaluated for a wrong-orbital state')
    monkeypatch.setattr(core.FDDS_Monomer, 'requirement', forbidden)
    runner = _runner(wfn, recipe, args, tmp_path, **_fdds())
    with pytest.raises(ValueError, match='native FDDS state must equal the sealed wavefunction Ca/epsilon_a'):
        runner.__enter__()
    assert runner._fdds is None and runner._temporary is None and _nothing_left(tmp_path)


def test_fdds_one_transition_analytic_limit(tmp_path):
    """H2 (one occupied, one virtual), x_alpha=0: chi = L b b^T / (1 - L b^T W b), b = J^-1 B."""
    wfn = _pbe0('0 1\nH 0 0 0\nH 0 0 .74\nsymmetry c1\nno_com\nno_reorient', TOY)
    molecule = wfn.molecule()
    recipe = _recipe(molecule, 'cc-pVDZ-RI', 'Spherical')
    auxiliary, B, gaps, J, density = _plain_operands(wfn, recipe)
    assert B.shape[1] == 1
    raw, transform = core.IsaAuxCoulomb(auxiliary).native_auxiliary()
    monomer = _monomer(wfn, raw, transform, tmp_path)
    rng = np.random.default_rng(5)
    noise = rng.normal(size=J.shape)*1e-3
    W = J+.5*(noise+noise.T)
    b = np.linalg.solve(J, B[:, 0])
    for omega in (0., .7, 2.):
        L = -4*gaps[0]/(gaps[0]**2+omega**2)
        expected = L*np.outer(b, b)/(1-L*b @ W @ b)
        native = monomer.form_coefficient_response(omega, 0., core.Matrix.from_array(W))['response'].to_array()
        np.testing.assert_allclose(native, expected, rtol=0, atol=1e-10*np.abs(expected).max())


# ---------------------------------------------------------------- explicit native FDDS runner ----

def _exact_contraction(Q, chi):
    """-Q chi Q^T of the given float64 operands in exact integer arithmetic, rounded once."""
    def exact_integers(a):
        nonzero = a[a != 0]
        shift = max(0, 53-min(math.frexp(x)[1] for x in nonzero.tolist())) if nonzero.size else 0
        return np.array([int(Fraction(x)*(1 << shift)) for x in a.ravel().tolist()], dtype=object
                        ).reshape(a.shape), shift
    (q, sq), (c, sc) = exact_integers(np.asarray(Q, float)), exact_integers(np.asarray(chi, float))
    denominator = 1 << (2*sq+sc)
    return np.array([-n/denominator for n in q.dot(c).dot(q.T).ravel().tolist()]).reshape(len(Q), len(Q))


def _contraction_bound(Q, chi):
    """Componentwise a priori float64 bound gamma_(2p+1) |Q||chi||Q|^T for fl(fl(Q chi) Q^T), symmetrized."""
    n, u = 2*Q.shape[1]+1, 2.**-53
    return n*u/(1-n*u)*(np.abs(Q) @ np.abs(chi) @ np.abs(Q).T)


def test_exact_contraction_oracle_is_exact():
    rng = np.random.default_rng(17)
    A = rng.integers(-9, 9, (4, 5)).astype(float)
    C = rng.integers(-9, 9, (5, 5)).astype(float)
    C = C+C.T
    np.testing.assert_array_equal(_exact_contraction(A, C), -(A @ C @ A.T))
    np.testing.assert_array_equal(_exact_contraction(A*2.**-60, C*2.**40), -(A @ C @ A.T)*2.**-80)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_contraction_meets_forward_target_when_well_conditioned(small_water, tmp_path):
    """The runner's own contraction, fed entrywise-positive Q and chi (no cancellation, kappa = 1),
    meets the planned forward target |nonlocal - exact| <= 1e-13 max|exact|."""
    wfn, recipe, args = small_water
    omega = args['quadrature'].frequencies[0]
    with _runner(wfn, recipe, args, tmp_path, **_fdds()) as runner:
        runner.prepare()
        p, q = runner._moments.shape[1], runner._moments.shape[0]
        rng = np.random.default_rng(2026)
        Q = rng.uniform(.5, 1.5, (q, p))
        c = rng.uniform(.5, 1.5, (p, p))
        chi = .5*(c+c.T)

        class Positive:
            def form_coefficient_response(self, *a):
                return dict(response=core.Matrix.from_array(chi), native_dyson_ratio=1., s2_dyson_ratio=1.,
                            solve_residual=0., masked_transitions=0)

        native, moments = runner._fdds, runner._moments
        runner._fdds, runner._moments = Positive(), Q
        try:
            result = runner.solve(omega)
        finally:
            runner._fdds, runner._moments = native, moments
    exact = _exact_contraction(Q, chi)
    np.testing.assert_allclose(np.abs(Q) @ np.abs(chi) @ np.abs(Q).T, -exact, rtol=1e-14, atol=0)  # kappa = 1
    assert np.abs(result.nonlocal_response-exact).max() <= 1e-13*np.abs(exact).max()


def test_fdds_toy_nodes_are_admitted_with_owned_outputs(small_water, tmp_path, monkeypatch):
    """Toy declared recipes (not molecular acceptance): every registered node is admitted."""
    from psi4.driver.procrouting.isapol_distribution import analytic_df_moments
    wfn, recipe, args = small_water
    kernels = []
    original = backend._kernel
    def capture(*a, **k):
        kernels.append(original(*a, **k).copy())
        return kernels[-1].copy()
    monkeypatch.setattr(backend, '_kernel', capture)
    frequencies = args['quadrature'].frequencies
    with _runner(wfn, recipe, args, tmp_path, **_fdds()) as runner:
        runner.prepare()
        p = runner.dimensions[0]
        q = 25*len(args['sites'])
        native_files = list((next(tmp_path.iterdir())/'fdds').iterdir())
        # The sweep reservation is exactly caller allowances + Q + native memory + W.
        memory = runner.provenance['plan']['requirement']['memory_bytes']
        assert runner._held == dict(retained=runner._retained, moments=8*q*p, native=memory, W=8*p*p)
        assert runner.ledger.reserved == runner._retained+8*q*p+memory+8*p*p
        J = runner._fdds.metric().to_array()
        W = runner._fdds_kernel.to_array()
        solved = [runner.solve(omega) for omega in frequencies]
    assert native_files and _nothing_left(tmp_path)
    np.testing.assert_array_equal(W, .75*kernels[0]+J)
    assert np.abs(W-J-kernels[0]).max() > 1e-3*np.abs(W).max()
    guard = runner.provenance['metric_guard']
    assert guard['cutoff'] == backend.FDDS_METRIC_RCOND_CUTOFF == 1e-14
    assert guard['metric_lu_rcond'] == runner.provenance['native_model']['metric_lu_rcond'] > 5e-14
    assert runner.provenance['metric_minus_isa_aux_coulomb_max_abs'] < 1e-13
    assert runner.lw_residual_policy == 'reported_input_sum_rule' and 'rank-0' in runner.lw_disclosure
    Q = analytic_df_moments(recipe, runner.sites, 4).values
    arrays = []
    for omega, result in zip(frequencies, solved):
        diagnostics = dict(result.diagnostics)
        assert result.frequency == omega and result.model == runner.model
        assert diagnostics['native_dyson_ratio'] > 2e-13 and diagnostics['s2_dyson_ratio'] > 2e-13
        assert result.residual == diagnostics['solve_residual'] and np.isfinite(result.residual)
        assert diagnostics['metric_lu_rcond'] == guard['metric_lu_rcond']
        chi, nonlocal_response = result.target_response, result.nonlocal_response
        assert chi.shape == (p, p) and nonlocal_response.shape == (q, q)
        np.testing.assert_array_equal(chi, chi.T)
        np.testing.assert_array_equal(nonlocal_response, nonlocal_response.T)
        # Not the planned 1e-13*max|output| forward target: this contraction cancels (kappa ~ 3e4-5e4),
        # so it is checked against the exact value at the rigorous componentwise float64 bound.
        exact = _exact_contraction(Q, chi)
        np.testing.assert_array_less(np.abs(nonlocal_response-exact), _contraction_bound(Q, chi))
        arrays += [chi, nonlocal_response]
    assert all(a.base is None or a.base.flags.owndata for a in arrays)
    assert not any(np.shares_memory(a, b) for a, b in itertools.combinations(arrays, 2))


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_metric_guard_refuses_before_any_response(small_water, tmp_path, monkeypatch):
    from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe
    wfn, recipe, args = small_water
    # An epsilon-level declared metric (native rcond ~1e-17): a relative 1e-5 near-duplicate O s shell.
    DELTA = 1e-5
    s = next(shell for shell in recipe.shells if shell.centre == 0 and shell.l == 0)
    twin = ShellRecipe(0, 0, tuple(e*(1+DELTA) for e in s.exponents), s.coefficients)
    epsilon = BasisRecipe('near-duplicate', 'test', 'Cartesian', recipe.centres, recipe.shells+(twin,))
    def forbidden(*a, **k):
        pytest.fail('a response was formed after the metric guard')
    monkeypatch.setattr(backend.BoundedResponse, '_solve_fdds', forbidden)
    for case, cutoff in ((epsilon, backend.FDDS_METRIC_RCOND_CUTOFF), (recipe, 1.)):
        monkeypatch.setattr(backend, 'FDDS_METRIC_RCOND_CUTOFF', cutoff)
        with pytest.raises(ValueError, match='metric resolution guard') as refusal:
            with _runner(wfn, case, args, tmp_path, **_fdds()) as runner:
                runner.prepare()
        guard = runner.provenance['metric_guard']
        assert guard['metric_lu_rcond'] < guard['cutoff'] == cutoff
        assert f"{guard['metric_lu_rcond']:.6e}" in str(refusal.value) and f'{cutoff:.0e}' in str(refusal.value)
        assert runner._fdds is None and _nothing_left(tmp_path)
    assert guard['metric_lu_rcond'] > 5e-14  # the toy recipe itself passes the real cutoff


def test_fdds_molecular_water_is_refused_by_native_dyson_admission(tmp_path):
    """Honest stage04 molecular outcome: the declared molecular water recipe passes the metric
    guard but the native dual Dyson admission refuses its first registered node (omega = 0)."""
    import psi4
    saved = core.get_memory()
    psi4.set_memory('4 GiB')
    try:
        wfn = _pbe0('0 1\nO 0 0 0\nH -1.45365196 0 -1.12168732\nH 1.45365196 0 -1.12168732\n'
                    'units bohr\nsymmetry c1\nno_com\nno_reorient',
                    {'basis': 'aug-cc-pvtz', 'puream': True, 'reference': 'rks', 'scf_type': 'pk',
                     'e_convergence': 1e-12, 'd_convergence': 1e-12})
        molecule = wfn.molecule()
        args = dict(_response_args(molecule, 100, 200), sites=_sites(molecule), shell_cutoff=1e-8)
        with pytest.raises(RuntimeError, match=r'Dyson admission refused .* at omega = 0;'):
            with _runner(wfn, _recipe(molecule), args, tmp_path, **_fdds()) as runner:
                runner.prepare()
                runner.solve(args['quadrature'].frequencies[0])
    finally:
        psi4.set_memory(saved, quiet=True)
    assert runner.provenance['metric_guard']['metric_lu_rcond'] > backend.FDDS_METRIC_RCOND_CUTOFF
    assert runner.provenance['native_model']['metric_dropped'] == 1 and _nothing_left(tmp_path)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_lifecycle(small_water, tmp_path):
    from psi4.driver.procrouting.isapol_distribution import analytic_df_moments
    wfn, recipe, args = small_water
    frequencies = args['quadrature'].frequencies
    runner = _runner(wfn, recipe, args, tmp_path, **_fdds())
    with runner as response:
        with pytest.raises(RuntimeError, match='prepared'):
            response.solve(frequencies[0])
        produced = []
        def provider(ledger):
            q = analytic_df_moments(recipe, response.sites, 4)
            produced.append(weakref.ref(q))
            return q
        response.prepare(provider)
        assert produced[0]() is None and response._moments.flags.owndata
        with pytest.raises(RuntimeError, match='prepare once'):
            response.prepare()
        with pytest.raises(ValueError, match='declared quadrature'):
            response.solve(frequencies[-1]+1.)
        response.solve(frequencies[1])
        response.release()
        assert response._fdds is None and response.ledger.reserved == response._retained
        with pytest.raises(RuntimeError, match='unreleased'):
            response.solve(frequencies[0])
    assert _nothing_left(tmp_path)
    with pytest.raises(RuntimeError, match='single-use'):
        runner.__enter__()


class _Forbidden:
    """Stands in for FDDS_Monomer: the static preflight passes through, construction fails the test."""
    requirement = staticmethod(core.FDDS_Monomer.requirement)

    def __init__(self, *args, **kwargs):
        pytest.fail('native FDDS was constructed after a resource refusal')


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_one_byte_short_preflight_refuses_before_native_construction(small_water, tmp_path, monkeypatch):
    wfn, recipe, args = small_water
    frequencies = args['quadrature'].frequencies
    def sweep(resources, fdds=backend.NativeFDDSOptions(1, 1 << 30)):
        with _runner(wfn, recipe, dict(args, resources=resources), tmp_path,
                     response='native_fdds', fdds=fdds) as runner:
            runner.prepare()
            for omega in frequencies:
                runner.solve(omega)
        return runner
    full = sweep(_resources())
    peak, work, io = full.ledger.peak, full.ledger.work, full.ledger.io
    disk = full.provenance['plan']['requirement']['disk_bytes']
    assert work == full.provenance['plan']['planned_work'] and io == full.provenance['plan']['planned_io_bytes']
    exact = sweep(_resources(peak, work, io), backend.NativeFDDSOptions(1, disk))
    assert exact.ledger.peak == peak
    monkeypatch.setattr(core, 'FDDS_Monomer', _Forbidden)
    for resources, fdds in ((_resources(peak-1, work, io), backend.NativeFDDSOptions(1, disk)),
                            (_resources(peak, work-1, io), backend.NativeFDDSOptions(1, disk)),
                            (_resources(peak, work, io-1), backend.NativeFDDSOptions(1, disk)),
                            (_resources(peak, work, io), backend.NativeFDDSOptions(1, disk-1))):
        with pytest.raises(ValueError, match='resource limit|disk_bytes'):
            sweep(resources, fdds)
        assert _nothing_left(tmp_path)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_native_and_kernel_admissions_charge_the_caller_grid(small_water, tmp_path, monkeypatch):
    """The caller's response_grid stays live through native construction and W: both admissions and
    their preflight mirrors charge it, so one byte short refuses before scratch or native storage.
    The native requirement is inflated so the FDDS plan, not the state snapshot, binds the budget."""
    wfn, recipe, args = small_water
    frequencies = args['quadrature'].frequencies
    held, runners = [], []
    native = core.FDDS_Monomer
    def requirement(*a):
        result = dict(native.requirement(*a))
        result['memory_bytes'] += 64*1024**2
        return result
    class Recording(native):
        def __init__(self, *a, **k):
            held.append(dict(runners[-1]._held))
            super().__init__(*a, **k)
    class Forbidden(_Forbidden):
        pass
    Recording.requirement = Forbidden.requirement = staticmethod(requirement)
    monkeypatch.setattr(core, 'FDDS_Monomer', Recording)
    def sweep(max_bytes):
        runner = _runner(wfn, recipe, dict(args, resources=_resources(max_bytes)), tmp_path, **_fdds())
        runners.append(runner)
        with runner:
            runner.prepare()
            for omega in frequencies:
                runner.solve(omega)
        return runner
    full = sweep(512*1024**2)
    stages = {s['stage']: s['numeric_bytes'] for s in full.ledger.stages}
    memory = full.provenance['plan']['requirement']['memory_bytes']
    assert held[0]['grid'] == args['response_grid'].nbytes
    assert stages['native FDDS instance'] == sum(held[0].values())+memory
    assert stages['FDDS kernel W'] == stages['native FDDS instance']-held[0]['native_inputs']+32*full.dimensions[0]**2
    assert sweep(full.ledger.peak).ledger.peak == full.ledger.peak
    monkeypatch.setattr(core, 'FDDS_Monomer', Forbidden)
    for stage in ('native FDDS instance', 'FDDS kernel W'):
        with pytest.raises(ValueError, match='complete native FDDS plan exceeds shared numeric'):
            sweep(stages[stage]-1)
        assert _nothing_left(tmp_path)


@pytest.mark.parametrize('stage', ['construction', 'response'])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_native_failures_clean_private_scratch(small_water, tmp_path, monkeypatch, stage):
    wfn, recipe, args = small_water
    class Failing(core.FDDS_Monomer):
        def __init__(self, *a, **k):
            if stage == 'construction':
                raise RuntimeError('injected native failure')
            super().__init__(*a, **k)

        def form_coefficient_response(self, *a, **k):
            raise RuntimeError('injected native failure')
    monkeypatch.setattr(core, 'FDDS_Monomer', Failing)
    with pytest.raises(RuntimeError, match='injected native'):
        with _runner(wfn, recipe, args, tmp_path, **_fdds()) as runner:
            runner.prepare()
            runner.solve(args['quadrature'].frequencies[0])
    assert runner._fdds is None and _nothing_left(tmp_path)


def test_response_modules_import_without_later_stages():
    import subprocess
    import sys
    code = ('import sys, psi4\n'
            'from psi4.driver.procrouting import isapol_bounded_response\n'
            'print(sorted(m for m in sys.modules if "isapol" in m))\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    loaded = {m.rsplit('.', 1)[-1] for m in eval(out.strip().splitlines()[-1])}
    later = {'isapol_bounded', 'isapol_refine', 'isapol_pfit', 'isapol_native_partition', 'isapol_bounded_oeprop',
             'isapol_dispersion', 'isapol_isa'}
    assert 'isapol_bounded_response' in loaded and not loaded & later
