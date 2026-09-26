# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Small independent bounded-backend identities, not large reference fixtures."""
import itertools
import numpy as np
import pytest
from psi4 import core
from psi4.driver.procrouting import isapol_bounded_response as backend
from psi4.driver.procrouting import isapol_bounded as driver
from psi4.driver.procrouting import isapol_refine as refine
from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators


def ledger():
    return backend._Ledger(backend.BoundedResources(512*1024**2, 6_000_000_000_000, 64*1024**3))


@pytest.mark.parametrize('field,value', [('max_bytes', True), ('max_work', 6_000_000_000_001),
                                      ('max_io_bytes', 0), ('max_bytes', 0)])
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


@pytest.mark.parametrize('exchange', [0., .25, 1.])
def test_exchange_indexing_matches_independent_factor_actions(exchange):
    rng = np.random.default_rng(818)
    oo, ov, dual, vv = [rng.normal(size=shape) for shape in
                        [(5, 2, 2), (5, 2, 3), (5, 2, 3), (5, 3, 3)]]
    gaps = np.arange(1., 7.)
    ops = FactorizedDFOperators(gaps, oo, ov, dual, vv, exact_exchange=exchange)
    v, y = backend._gram_terms(ov, dual)
    x = backend._exchange_x(oo, [vv.reshape(5, 9)[:, :4], vv.reshape(5, 9)[:, 4:]], 3)
    np.testing.assert_allclose(np.diag(gaps)+4*v-exchange*(x+y), ops.apply_h1(np.eye(6)), atol=1e-13)
    np.testing.assert_allclose(np.diag(gaps)-exchange*(x-y), ops.apply_h2(np.eye(6)), atol=1e-13)
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
    coefficient, raw, residual = backend._frequency(store, (4, 2, 3), 3, omega)
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


def test_streamed_fit_equals_dense_fit_with_same_native_targets():
    """One shared-design multi-frequency fit reproduces each node's dense fit."""
    site = refine.RefinementSite('A', 'A', (0., 0., 0.), np.eye(3), 1)
    rng = np.random.default_rng(811)
    points = rng.normal(size=(19, 3))+[4., 4., 4.]
    potentials = rng.normal(size=(7, len(points)))
    source = dict(state_sha256='synthetic-state', auxiliary_sha256='synthetic-aux', model='synthetic-equation')
    models, coefficients = [], [-np.eye(7), -np.diag(rng.random(7)+.5)]
    for omega, scale in [(0., 1.), (.6, .7)]:
        anchor = scale*np.diag([0., 2., 3., 4.])
        models.append(refine.refinement_model([site], [anchor], frequency_au=omega, cutoff=1e-4,
            weight_type=4, weight_coefficient=1e-5,
            declared_variables=models[0].parameter_labels if models else None, provenance='synthetic identity'))
    row_count = len(points)*(len(points)+1)//2
    packed = np.empty((row_count, 2))
    for t, coefficient in enumerate(coefficients):
        driver._pack_targets(packed, t, coefficient, potentials, ledger())
    streamed = driver._stream_fit(models, points, packed, ledger(), source)
    assert len(streamed) == 2
    for model, coefficient, fit in zip(models, coefficients, streamed):
        targets = -(potentials.T @ coefficient) @ potentials
        dense = refine.refine(model, points, targets[np.tril_indices(len(points))],
            target_origin=core.IsaPfitTargetOrigin.NativeFittedPointResponse,
            source_id=source['state_sha256'], auxiliary_basis_id=source['auxiliary_sha256'],
            response_representation='fitted_density_coefficients', generation_record=source['model'])
        np.testing.assert_allclose(fit.parameters, dense.parameters, rtol=1e-11, atol=1e-11)
        assert fit.status == core.IsaPfitStatus.Solved
        assert fit.result.frequency_au == model.frequency_au
    assert not np.allclose(streamed[0].parameters, streamed[1].parameters)


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


@pytest.fixture(scope='module', params=('water', 'ammonia'))
def small_water(request):
    import psi4
    from psi4.driver.procrouting.isapol_native_partition import BasisRecipe, ShellRecipe
    from psi4.driver.procrouting.isapol_native import Quadrature
    from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
    core.be_quiet()
    geometry = ('O 0 0 0\nH .757 0 .586\nH -.757 0 .586' if request.param == 'water' else
                'N 0 0 .116\nH 0 .939 -.271\nH .813197854 -.4695 -.271\nH -.813197854 -.4695 -.271')
    molecule = psi4.geometry('0 1\n'+geometry+'\nsymmetry c1\nno_com\nno_reorient')
    psi4.set_options({'basis': 'sto-3g', 'reference': 'rks', 'scf_type': 'pk',
                     'e_convergence': 1e-10, 'd_convergence': 1e-10,
                     'dft_radial_points': 50, 'dft_spherical_points': 110})
    _, wfn = psi4.energy('pbe0', molecule=molecule, return_wfn=True)
    origins = molecule.geometry().np
    basis = core.BasisSet.build(molecule, 'DF_BASIS_SCF', 'aug-cc-pVTZ-RI', puream=0)
    shells = []
    for i in range(basis.nshell()):
        s = basis.shell(i)
        shells.append(ShellRecipe(int(basis.shell_to_center(i)), int(s.am),
            tuple(s.exp(k) for k in range(s.nprimitive)), tuple(s.coef(k) for k in range(s.nprimitive))))
    recipe = BasisRecipe('aug-cc-pVTZ-RI', 'Psi4 shipped basis', 'Cartesian', tuple(map(tuple, origins)), tuple(shells))
    sites = [refine.RefinementSite(label, label, origin, np.eye(3), 1)
             for label, origin in zip(('A', 'H1', 'H2', 'H3'), origins)]
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 20, 50
    grid = core.IsaGrid(molecule.clone(), options)
    lattice = core.FitPointsOptions()
    lattice.npoints, lattice.seed, lattice.lolim, lattice.hilim = 32, 1, 2., 4.
    args = dict(caller_converged=True, distribution='df_centre_analytic', sites=sites,
        bonds=[(0, i) for i in range(1, len(sites))], quadrature=Quadrature.from_casimir(core.CasimirGrid(2, .5)),
        response_grid=np.column_stack((grid.x(), grid.y(), grid.z(), grid.w())),
        smoothing=KernelSmoothing(1e-8, 1000., .01, 1., 'FD'), shell_cutoff=1e-7,
        charge_penalty=1000., anchor_metric_damping=.0005, lattice_options=lattice,
        localization_rank_limit=2, weight_type=4, weight_coefficient=1e-5, cutoff=1e-4,
        resources=ledger().resources, scf_correction='NONE', max_order=6)
    return wfn, recipe, args


def test_live_small_molecule_driver_completes_and_cleans_checkpoints(small_water, tmp_path):
    from psi4.driver.procrouting.isapol_logging import StageLog
    messages = []
    log = StageLog(3, writer=messages.append)
    wfn, recipe, args = small_water
    result = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=log)
    text = ''.join(messages)
    for name in ('Native plain-DF factors', 'anchor constrained OV fit', 'target constrained OV fit',
                 'Full-grid ALDA kernel', 'H1/H2 assembly', 'Response stability checks',
                 'Native point cloud and Coulomb potentials', 'Complete-cloud PFIT',
                 'Casimir-Polder dispersion'):
        assert f'Stage complete: {name}' in text
    for node in range(1, 4):
        for name in ('Original H2H1 response', 'Production LW localization'):
            assert f'Stage complete: {name}: node {node}/3' in text
    assert 'relative response residual' in text
    assert 'maximum localization residual' in text
    assert 'solver status' in text
    assert len(result.refinements) == len(args['quadrature'].frequencies)
    assert all(r.status == core.IsaPfitStatus.Solved for r in result.refinements)
    assert result.resources['fit_rows'] == 3*32*33
    assert max(d['response_residual'] for d in result.diagnostics) < 1e-10
    assert not list(tmp_path.iterdir())
    assert result.provenance['distribution'] == 'df_centre_analytic'
    assert all(np.isfinite(c.value) for pair in result.dispersion.pairs for c in pair.coefficients)
    silent_messages = []
    silent_result = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path,
                                              log=StageLog(0, writer=silent_messages.append))
    assert not silent_messages
    for reported, silent in zip(result.refined_tensors, silent_result.refined_tensors):
        for a, b in zip(reported, silent):
            np.testing.assert_array_equal(a, b)


def test_missing_scf_seal_is_not_manufactured(small_water, tmp_path, monkeypatch):
    wfn, recipe, args = small_water
    monkeypatch.delattr(wfn, '_scf_convergence_evidence')
    with pytest.raises(ValueError, match='convergence evidence'):
        driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_work_preflight_rejects_before_factor_construction(small_water, tmp_path, monkeypatch):
    wfn, recipe, args = small_water
    def forbidden(*args, **kwargs):
        pytest.fail('factor construction happened before work refusal')
    monkeypatch.setattr(driver, 'native_plain_df_operators', forbidden)
    args = dict(args, resources=backend.BoundedResources(512*1024**2, 1, 64*1024**3))
    with pytest.raises(ValueError, match='work resource limit'):
        driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_late_failure_cleans_owned_checkpoints(small_water, tmp_path, monkeypatch):
    from psi4.driver.procrouting.isapol_logging import StageLog
    messages = []
    wfn, recipe, args = small_water
    def fail(*args, **kwargs):
        raise RuntimeError('injected kernel failure')
    monkeypatch.setattr(driver, '_kernel', fail)
    with pytest.raises(RuntimeError, match='injected kernel'):
        driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path,
                                  log=StageLog(3, writer=messages.append))
    assert 'Stage: Full-grid ALDA kernel' in ''.join(messages)
    assert 'Stage complete: Full-grid ALDA kernel' not in ''.join(messages)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('key,value,match', [
    ('distribution', 'ISA_A', 'df_centre'), ('caller_converged', False, 'caller_converged'),
    ('scf_correction', 'DECLARED_MULTPOLE_AC', 'supports NONE'),
    ('anchor_metric_damping', True, 'anchor_metric'), ('shell_cutoff', -1., 'shell_cutoff'),
    ('localization_rank_limit', 0, 'localization_rank'),
])
def test_public_declaration_refusals(small_water, tmp_path, key, value, match):
    wfn, recipe, args = small_water
    with pytest.raises(ValueError, match=match):
        driver.bounded_properties(wfn, recipe, **dict(args, **{key:value}), scratch_directory=tmp_path)
    assert not list(tmp_path.iterdir())
