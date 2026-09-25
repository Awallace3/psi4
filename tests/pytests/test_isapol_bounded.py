# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Small independent bounded-backend identities, not large reference fixtures."""
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


def test_instability_is_not_repaired(tmp_path):
    store = backend._Store(tmp_path, ledger())
    store.save('h1', np.diag([-1., 1.]))
    with pytest.raises(ValueError, match='stability'):
        backend._stability(store, 2)


def test_streamed_fit_equals_dense_fit_with_same_native_targets():
    site = refine.RefinementSite('A', 'A', (0., 0., 0.), np.eye(3), 1)
    anchor = np.diag([0., 2., 3., 4.])
    model = refine.refinement_model([site], [anchor], frequency_au=0., cutoff=1e-4,
                                    weight_type=4, weight_coefficient=1e-5, provenance='synthetic identity')
    rng = np.random.default_rng(811)
    points = rng.normal(size=(19, 3))+[4., 4., 4.]
    potentials = rng.normal(size=(7, len(points)))
    coefficient = -np.eye(7)
    source = dict(state_sha256='synthetic-state', auxiliary_sha256='synthetic-aux', model='synthetic-equation')
    streamed = driver._stream_fit(model, points, coefficient, potentials, ledger(), source)
    targets = -(potentials.T @ coefficient) @ potentials
    dense = refine.refine(model, points, targets[np.tril_indices(len(points))],
        target_origin=core.IsaPfitTargetOrigin.NativeFittedPointResponse,
        source_id=source['state_sha256'], auxiliary_basis_id=source['auxiliary_sha256'],
        response_representation='fitted_density_coefficients', generation_record=source['model'])
    np.testing.assert_allclose(streamed.parameters, dense.parameters, rtol=1e-11, atol=1e-11)
    assert streamed.status == core.IsaPfitStatus.Solved


def test_dense_kernel_is_the_same_screened_integral():
    from psi4.driver.procrouting.isapol_native_partition import BasisRecipe, ShellRecipe
    from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
    from psi4.driver.procrouting.isapol_auxiliary_kernel import screened_auxiliary_kernel
    recipe = BasisRecipe('two-centre test', 'synthetic', 'Cartesian',
        ((0., 0., 0.), (2., 0., 0.)),
        (ShellRecipe(0, 0, (1.,), (1.,)), ShellRecipe(1, 1, (.7,), (.8,))))
    auxiliary = recipe.build('MolecularAux')
    rng = np.random.default_rng(601)
    grid = np.column_stack((rng.normal(size=(129, 3)), rng.random(129)))
    density = np.array([1., .1, -.1, .2])
    smoothing = KernelSmoothing(1e-8, 400., .1, .1, 'FD')
    for cutoff in (0., .8):
        actual = backend._kernel(auxiliary, density, grid, smoothing, cutoff, ledger())
        expected = screened_auxiliary_kernel(auxiliary, density, grid,
            smoothing=smoothing, cutoff=cutoff, block_rows=64).matrix
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-13)


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
                 'Native point cloud and Coulomb potentials', 'Casimir-Polder dispersion'):
        assert f'Stage complete: {name}' in text
    for node in range(1, 4):
        for name in ('Original H2H1 response', 'Production LW localization', 'Complete-cloud PFIT'):
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
