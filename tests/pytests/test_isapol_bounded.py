# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Bounded property chain: response -> LW -> PFIT -> isotropic C_n on live toy wavefunctions.

The stages themselves are tested with their own modules; these tests check the
orchestration: stage order and resources, ownership, cleanup, the streamed
refinement against an independent dense fit, distribution and response
selection, and refusals. Toy admissions are not molecular acceptance.
"""
import math

import numpy as np
import pytest
from psi4 import core
from psi4.driver.procrouting import isapol_bounded as driver
from psi4.driver.procrouting import isapol_bounded_response as backend
from psi4.driver.procrouting import isapol_refine as refine
from psi4.driver.procrouting.isapol_logging import StageLog

GEOMETRIES = {'water': 'O 0 0 0\nH .757 0 .586\nH -.757 0 .586',
              'ammonia': 'N 0 0 .116\nH 0 .939 -.271\nH .813197854 -.4695 -.271\nH -.813197854 -.4695 -.271'}
NAMESPACE = ('ATOMIC REFINED DISPERSION', 'REFINED DISPERSION COEFFICIENT')


def _resources(max_bytes=512*1024**2, max_work=6_000_000_000_000, max_io_bytes=64*1024**3):
    return backend.BoundedResources(max_bytes, max_work, max_io_bytes)


@pytest.fixture(scope='module', params=('water', 'ammonia'))
def small_water(request):
    """PBE0/sto-3g, Cartesian aug-cc-pVTZ-RI AUX, rank-1 refinement on 32 lattice points, 3 nodes."""
    import psi4
    from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe
    from psi4.driver.procrouting.isapol_native import Quadrature
    from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing
    core.be_quiet()
    core.clean_options()
    molecule = psi4.geometry('0 1\n'+GEOMETRIES[request.param]+'\nsymmetry c1\nno_com\nno_reorient')
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
    recipe = BasisRecipe('aug-cc-pVTZ-RI', 'Psi4 shipped basis', 'Cartesian', tuple(map(tuple, origins)),
                         tuple(shells))
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
        resources=_resources(), scf_correction='NONE', max_order=6)
    return wfn, recipe, args


def _owned_names(wfn):
    from psi4.driver.procrouting.isapol_logging import _report_owned
    return {k for k in list(wfn.scalar_variables())+list(wfn.array_variables()) if _report_owned(k, *NAMESPACE)}


def _c6(result):
    return [c.value for pair in result.dispersion.pairs for c in pair.coefficients]


def _q(recipe, args):
    from psi4.driver.procrouting.isapol_distribution import analytic_df_moments
    sites = []
    for s in args['sites']:
        site = core.IsaMultipoleSite()
        site.label, site.origin, site.rank = s.label, list(s.origin_bohr), 4
        sites.append(site)
    return analytic_df_moments(recipe, sites, 4)


def _stage_names(result):
    return [s['stage'] for s in result.resources['stages']]


def test_live_chain_completes_in_order_and_cleans_checkpoints(small_water, tmp_path):
    messages = []
    wfn, recipe, args = small_water
    result = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path,
                                       log=StageLog(3, writer=messages.append))
    text = ''.join(messages)
    for name in ('Native plain-DF factors', 'anchor constrained OV fit', 'target constrained OV fit',
                 'Full-grid ALDA kernel', 'H1/H2 assembly', 'Response stability checks',
                 'Native point cloud', 'Complete-cloud PFIT',
                 'isotropic dispersion from refined tensors (Casimir-Polder)', 'Bounded resource totals'):
        assert f'Stage complete: {name}' in text
    for node in range(1, 4):
        for name in ('Original H2H1 response', 'Production LW localization', 'Point-response targets'):
            assert f'Stage complete: {name}: node {node}/3' in text
    # The fixed variable set is admitted once, between node 1's LW and its targets.
    assert text.count('Stage complete: Refinement model admission') == 1
    assert (text.index('Stage complete: Production LW localization: node 1/3')
            < text.index('Stage complete: Refinement model admission')
            < text.index('Stage: Point-response targets: node 1/3'))
    assert 'Stage FAILED' not in text and 'Pairwise isotropic dispersion coefficients' in text
    # Admission order: response, cloud, then per node solve -> LW -> targets, then PFIT.
    names = _stage_names(result)
    sweep = names[names.index('retained point-response targets')+1:names.index('complete-cloud refinement')]
    assert names.index('operator stability') < names.index('native fit cloud') < names.index(
        'retained point-response targets')
    assert sweep == ['H2H1 response operator']+['frequency response', 'production localization',
                                                'point-response targets']*3
    assert names[-1] == 'complete-cloud refinement'
    assert result.resources['maximum_numeric_plan'] == max(s['numeric_bytes'] for s in result.resources['stages'])
    assert result.resources['charged_work'] == sum(s['work'] for s in result.resources['stages'])
    assert len(result.refinements) == len(args['quadrature'].frequencies)
    assert all(r.status == core.IsaPfitStatus.Solved for r in result.refinements)
    assert result.resources['fit_rows'] == 3*32*33
    assert max(d['response_residual'] for d in result.diagnostics) < 1e-10
    assert all(d['response_diagnostics'] == () for d in result.diagnostics)
    assert result.provenance['distribution'] == 'df_centre_analytic'
    assert result.provenance['response'] == 'reference_h2h1'
    assert result.provenance['lw_residual_policy'] == 'production'
    assert result.provenance['partition']['model'] == 'df_centre_analytic'
    assert all(np.isfinite(c.value) for pair in result.dispersion.pairs for c in pair.coefficients)
    assert not list(tmp_path.iterdir())
    assert not _owned_names(wfn)  # bounded_properties alone publishes nothing
    silent_messages = []
    silent = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path,
                                       log=StageLog(0, writer=silent_messages.append))
    assert not silent_messages
    for reported, quiet in zip(result.refined_tensors, silent.refined_tensors):
        for a, b in zip(reported, quiet):
            np.testing.assert_array_equal(a, b)
    assert _c6(result) == _c6(silent) and silent.resources == result.resources


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_streamed_refinement_and_c6_match_independent_dense_oracles(small_water, tmp_path, monkeypatch):
    """Every node's streamed fit equals a dense QR fit of independently formed targets."""
    wfn, recipe, args = small_water
    captured = []
    original = driver.fitted_point_targets

    def capture(auxiliary, points, responses, **kwargs):
        captured.append((auxiliary, np.array(points), responses[0].copy()))
        return original(auxiliary, points, responses, **kwargs)

    monkeypatch.setattr(driver, 'fitted_point_targets', capture)
    result = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=StageLog(0))
    options = core.IsaPfitOptions()
    options.solver = core.IsaPfitSolver.StreamingQR
    source = result.provenance
    for (auxiliary, points, coefficient), fit in zip(captured, result.refinements):
        coulomb = core.IsaAuxCoulomb(auxiliary)
        potentials = np.column_stack([np.asarray(coulomb.point_potentials(core.Matrix.from_array(points[i:i+1])))
                                      for i in range(len(points))])
        targets = -np.einsum('ki,kl,lj->ij', potentials, coefficient, potentials)
        dense = refine.refine(fit.model, points, targets[np.tril_indices(len(points))], options=options,
            target_origin=core.IsaPfitTargetOrigin.NativeFittedPointResponse,
            source_id=source['state_sha256'], auxiliary_basis_id=source['auxiliary_sha256'],
            response_representation='fitted_density_coefficients', generation_record=source['model'])
        np.testing.assert_allclose(fit.parameters, dense.parameters, rtol=1e-9,
                                   atol=1e-10*np.abs(dense.parameters).max())
        assert fit.result.frequency_au == fit.model.frequency_au
    assert len(captured) == 3
    assert not np.allclose(result.refinements[0].parameters, result.refinements[2].parameters)
    # Rank-1 C6 = binomial(4, 2) sum_f w_f alpha_a(f) alpha_b(f), alpha = trace(alpha_11)/3.
    weights = result.dispersion.cp_weights
    for pair in result.dispersion.pairs:
        alpha = [[np.trace(np.asarray(r.refined_tensors[s])[1:4, 1:4])/3 for r in result.refinements]
                 for s in (pair.site_a, pair.site_b)]
        value = 0.
        for w, a, b in zip(weights, *alpha):
            if w != 0.:
                value += w*a*b
        assert pair.coefficients[0].value == pytest.approx(math.comb(4, 2)*value, rel=1e-13, abs=0.)
        assert pair.coefficients[0].unrestricted_complete


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_supplied_q_leaves_response_targets_independent(small_water, tmp_path, monkeypatch):
    from dataclasses import replace
    wfn, recipe, args = small_water
    q = _q(recipe, args)
    coefficients = []
    original = driver.fitted_point_targets

    def capture(auxiliary, points, responses, **kwargs):
        coefficients.append(responses[0].copy())
        return original(auxiliary, points, responses, **kwargs)

    monkeypatch.setattr(driver, 'fitted_point_targets', capture)
    default = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path)
    supplied_args = dict(args, distribution='supplied', distributed_moments=q)
    supplied = driver.bounded_properties(wfn, recipe, **supplied_args, scratch_directory=tmp_path)
    scaled = replace(q, values=1.01*q.values, model='scaled-Q test', provenance='explicit synthetic substitution')
    changed = driver.bounded_properties(wfn, recipe, **dict(supplied_args, distributed_moments=scaled),
                                        scratch_directory=tmp_path)
    nf = len(default.frequencies)
    for i in range(nf):
        np.testing.assert_array_equal(coefficients[i], coefficients[nf+i])
        np.testing.assert_array_equal(coefficients[i], coefficients[2*nf+i])
        np.testing.assert_array_equal(default.local_tensors[i].array, supplied.local_tensors[i].array)
        np.testing.assert_array_equal(default.refinements[i].parameters, supplied.refinements[i].parameters)
        np.testing.assert_allclose(changed.local_tensors[i].array,
                                   1.01**2*default.local_tensors[i].array, rtol=1e-11, atol=1e-11)
    assert not np.allclose(default.refinements[0].parameters, changed.refinements[0].parameters,
                           rtol=1e-10, atol=1e-10)
    assert _c6(default) == _c6(supplied)
    assert not np.allclose(_c6(default), _c6(changed), rtol=1e-10, atol=1e-10)
    assert supplied.provenance['distribution'] == 'supplied'
    assert changed.provenance['partition']['model'] == 'scaled-Q test'
    # The supplied record stays caller-owned and unchanged; the result does not alias it.
    np.testing.assert_array_equal(q.values, _q(recipe, args).values)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_results_are_owned_and_consumed_inputs_are_not_reread(small_water, tmp_path, monkeypatch):
    """Caller inputs changed mid-chain do not reach the result, and result arrays are owned."""
    from dataclasses import replace
    wfn, recipe, args = small_water
    first = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=StageLog(0))
    grid, lattice, q = args['response_grid'].copy(), core.FitPointsOptions(), _q(recipe, args)
    lattice.npoints, lattice.seed, lattice.lolim, lattice.hilim = 32, 1, 2., 4.
    sites = list(args['sites'])
    original = driver.fitted_point_targets

    def alter(*a, **k):
        grid[:] = 0.
        lattice.npoints, lattice.seed = 7, 99
        sites[0] = replace(sites[0], rank_limit=0)
        return original(*a, **k)

    monkeypatch.setattr(driver, 'fitted_point_targets', alter)
    altered = driver.bounded_properties(wfn, recipe, **dict(args, response_grid=grid, lattice_options=lattice,
                                        sites=sites, distribution='supplied', distributed_moments=q),
                                        scratch_directory=tmp_path, log=StageLog(0))
    assert _c6(altered) == _c6(first)
    assert altered.provenance['lattice_sha256'] == first.provenance['lattice_sha256']
    assert altered.resources['fit_rows'] == first.resources['fit_rows']
    with pytest.raises(ValueError):
        altered.refined_tensors[0][0][1, 1] = 1.
    local = altered.local_tensors[0].array
    altered.local_tensors[0].array[:] = 0.
    np.testing.assert_array_equal(altered.local_tensors[0].array, local)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_stale_scf_seal_is_refused_before_any_scratch(small_water, tmp_path):
    wfn, recipe, args = small_water
    ca = wfn.Ca().np
    saved = ca.copy()
    ca[0, 0] += 1e-9  # orbitals changed after the sealed SCF
    try:
        with pytest.raises(ValueError, match='stale'):
            driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=StageLog(0))
    finally:
        ca[:] = saved
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('target,stage', [
    ('backend._kernel', 'Full-grid ALDA kernel'),
    ('driver.fitted_point_targets', 'Point-response targets: node 1/3'),
    ('driver.refine_streamed', 'Complete-cloud PFIT'),
])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_failures_close_their_stage_publish_nothing_and_clean_scratch(small_water, tmp_path, monkeypatch,
                                                                      target, stage):
    wfn, recipe, args = small_water
    module, name = target.split('.')

    def fail(*a, **k):
        raise RuntimeError('injected failure')

    monkeypatch.setattr({'backend': backend, 'driver': driver}[module], name, fail)
    messages = []
    with pytest.raises(RuntimeError, match='injected failure'):
        driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, publish_qcvariables=True,
                                  log=StageLog(3, writer=messages.append))
    text = ''.join(messages)
    assert f'Stage FAILED: {stage}' in text and f'Stage complete: {stage}' not in text
    assert not _owned_names(wfn) and not list(tmp_path.iterdir())


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_parameter_cap_is_admitted_at_node_one_before_any_targets(small_water, tmp_path, monkeypatch):
    """The variable set is fixed at node 1, so a model over the cap stops before targets or later solves."""
    wfn, recipe, args = small_water
    count = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path,
                                      log=StageLog(0)).refinements[0].model.parameter_count
    solves, targets = [], []
    solve, fitted = backend.BoundedResponse.solve, driver.fitted_point_targets
    monkeypatch.setattr(backend.BoundedResponse, 'solve',
                        lambda self, omega: solves.append(omega) or solve(self, omega))
    monkeypatch.setattr(driver, 'fitted_point_targets', lambda *a, **k: targets.append(1) or fitted(*a, **k))
    monkeypatch.setattr(driver, 'MAX_PARAMETERS', count)  # at the cap: admitted, full sweep
    driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=StageLog(0))
    assert (len(solves), len(targets)) == (3, 3)
    solves.clear(), targets.clear()
    monkeypatch.setattr(driver, 'MAX_PARAMETERS', count-1)
    messages = []
    with pytest.raises(ValueError, match=f'at most {count-1} parameters'):
        driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, publish_qcvariables=True,
                                  log=StageLog(1, writer=messages.append))
    text = ''.join(messages)
    assert (len(solves), len(targets)) == (1, 0)
    assert 'Stage FAILED: Refinement model admission' in text
    assert 'Stage complete: Production LW localization: node 1/3' in text
    assert 'Stage FAILED: Production LW localization' not in text and 'Point-response targets' not in text
    assert not _owned_names(wfn) and not list(tmp_path.iterdir())


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_exact_resource_boundaries(small_water, tmp_path, monkeypatch):
    """The whole chain admits at its exact planned peak and work, and refuses one unit short."""
    wfn, recipe, args = small_water
    first = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=StageLog(0))
    peak, work = first.resources['maximum_numeric_plan'], first.resources['charged_work']
    exact = driver.bounded_properties(wfn, recipe, **dict(args, resources=_resources(peak, work)),
                                      scratch_directory=tmp_path, log=StageLog(0))
    assert exact.resources == first.resources and _c6(exact) == _c6(first)
    for resources, match in ((_resources(peak-1, work), 'numeric byte'), (_resources(peak, work-1), 'work')):
        with pytest.raises(ValueError, match=match):
            driver.bounded_properties(wfn, recipe, **dict(args, resources=resources),
                                      scratch_directory=tmp_path, log=StageLog(0))
    assert not list(tmp_path.iterdir())
    # The orchestrator's own plans cover the planned buffers of the stage-05 calls they admit.
    targets, streamed = driver.fitted_point_targets, driver.refine_streamed
    p = recipe.build('MolecularAux').nfunction
    npoint, nf = args['lattice_options'].npoints, len(args['quadrature'].frequencies)
    monkeypatch.setattr(driver, 'fitted_point_targets', lambda aux, points, responses, max_bytes: targets(
        aux, points, responses, max_bytes=driver._target_bytes(p, npoint)-8*p*p))
    def planned_fit(models, points, packed, **kwargs):
        plan = driver._pfit_plan(npoint, models[0].channel_count, models[0].parameter_count, nf)[0]
        return streamed(models, points, packed, **dict(kwargs, max_bytes=plan))
    monkeypatch.setattr(driver, 'refine_streamed', planned_fit)
    planned = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=StageLog(0))
    assert _c6(planned) == _c6(first)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_explicit_native_fdds_toy_chain(small_water, tmp_path):
    """The explicit FDDS choice reaches LW (reported-input policy), PFIT and C_n on the admitted toy."""
    wfn, recipe, args = small_water
    fdds = backend.NativeFDDSOptions(1, 1 << 30)
    reference = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, log=StageLog(0))
    messages = []
    result = driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, response='native_fdds',
                                       fdds=fdds, log=StageLog(1, writer=messages.append))
    text = ''.join(messages)
    assert 'Stage complete: Reported-input LW localization: node 3/3' in text
    assert result.provenance['response'] == 'native_fdds'
    assert result.provenance['lw_residual_policy'] == 'reported_input_sum_rule'
    assert 'rank-0 remainder' in result.provenance['lw_disclosure']
    guard = result.provenance['response_provenance']['metric_guard']
    assert guard['metric_lu_rcond'] >= backend.FDDS_METRIC_RCOND_CUTOFF
    for d in result.diagnostics:
        ratios = dict(d['response_diagnostics'])
        assert ratios['native_dyson_ratio'] > 2e-13 and ratios['s2_dyson_ratio'] > 2e-13
    assert all(r.status == core.IsaPfitStatus.Solved for r in result.refinements)
    assert result.resources['frequency_rhs'] is None
    assert 'complete-cloud refinement' in _stage_names(result)
    assert 'reported-input localization' in _stage_names(result)
    assert all(np.isfinite(_c6(result))) and not np.allclose(_c6(result), _c6(reference))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_fdds_metric_refusal_propagates_unchanged(small_water, tmp_path, monkeypatch):
    wfn, recipe, args = small_water
    monkeypatch.setattr(backend, 'FDDS_METRIC_RCOND_CUTOFF', 1.)
    messages = []
    with pytest.raises(ValueError, match='metric resolution guard'):
        driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, response='native_fdds',
                                  fdds=backend.NativeFDDSOptions(1, 1 << 30), publish_qcvariables=True,
                                  log=StageLog(1, writer=messages.append))
    assert 'Stage FAILED: Native FDDS instance' in ''.join(messages)
    assert not _owned_names(wfn) and not list(tmp_path.iterdir())


@pytest.mark.parametrize('key,value,error,match', [
    ('distribution', 'isa', TypeError, 'explicit PartitionRecipe'),
    ('distribution', 'mbis', ValueError, 'MBIS requires an explicit .* integration grid'),
    ('partition_recipe', 'recipe', ValueError, 'partition_recipe requires distribution=isa'),
    ('partition_grid', 'grid', ValueError, 'partition_grid requires distribution=isa or mbis'),
    ('distribution', 'supplied', TypeError, 'DistributedMoments'),
    ('distributed_moments', 'q', ValueError, 'requires distribution=supplied'),
    ('caller_converged', False, ValueError, 'caller_converged'),
    ('scf_correction', 'DECLARED_MULTPOLE_AC', ValueError, 'supports NONE'),
    ('anchor_metric_damping', True, ValueError, 'anchor_metric'),
    ('shell_cutoff', -1., ValueError, 'shell_cutoff'),
    ('localization_rank_limit', 4, ValueError, 'localization_rank'),
    ('max_order', 7, ValueError, 'max_order'),
    ('response', 'native_fdds', TypeError, 'NativeFDDSOptions'),
    ('sites', 'labels', ValueError, 'collide'),
])
@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_public_declaration_refusals(small_water, tmp_path, key, value, error, match):
    wfn, recipe, args = small_water
    if value == 'q':
        value = _q(recipe, args)
    if key == 'sites':
        from dataclasses import replace
        value = [replace(s, label=l) for s, l in zip(args['sites'], ('O', 'h', 'H'))]
    with pytest.raises(error, match=match):
        driver.bounded_properties(wfn, recipe, **dict(args, **{key: value}), scratch_directory=tmp_path,
                                  publish_qcvariables=True, log=StageLog(0))
    assert not list(tmp_path.iterdir()) and not _owned_names(wfn)


@pytest.mark.parametrize('small_water', ['water'], indirect=True)
def test_reserved_publication_labels_refuse_before_any_report_but_the_model_keeps_them(small_water, tmp_path):
    """TOTAL/INCOMPLETE labels refuse publication up front; unpublished runs keep them, look-alikes publish."""
    from dataclasses import replace
    wfn, recipe, args = small_water

    def labelled(*labels):
        return [replace(s, label=l) for s, l in zip(args['sites'], labels)]

    driver.bounded_properties(wfn, recipe, **args, scratch_directory=tmp_path, publish_qcvariables=True,
                              log=StageLog(0))
    published = {key: wfn.variable(key) for key in _owned_names(wfn) if wfn.has_scalar_variable(key)}
    for labels in (('O', 'Total', 'H2'), ('INCOMPLETE', 'H1', 'H2')):
        messages = []
        with pytest.raises(ValueError, match='reserved'):
            driver.bounded_properties(wfn, recipe, **dict(args, sites=labelled(*labels)),
                                      scratch_directory=tmp_path, publish_qcvariables=True,
                                      log=StageLog(1, writer=messages.append))
        text = ''.join(messages)
        assert 'Stage FAILED: Bounded input validation' in text and text.count('Stage complete') == 0
        assert {key: wfn.variable(key) for key in published} == published and not list(tmp_path.iterdir())
    kept = driver.bounded_properties(wfn, recipe, **dict(args, sites=labelled('O', 'Total', 'incomplete')),
                                     scratch_directory=tmp_path, log=StageLog(0))
    assert kept.dispersion.labels == ('O', 'Total', 'incomplete')
    assert {key: wfn.variable(key) for key in published} == published
    similar = driver.bounded_properties(wfn, recipe, **dict(args, sites=labelled('O', 'Totals', 'H_TOTAL')),
                                        scratch_directory=tmp_path, publish_qcvariables=True, log=StageLog(0))
    assert similar.dispersion.labels == ('O', 'Totals', 'H_TOTAL')
    assert wfn.has_variable('ATOMIC REFINED DISPERSION C6 O TOTALS')
    assert wfn.has_variable('ATOM H_TOTAL C6 REFINED DISPERSION COEFFICIENT')
    assert np.array_equal(_c6(kept), _c6(similar))
