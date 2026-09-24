# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Native fitted targets on a small water state, not CamCASP parity evidence."""
from dataclasses import replace
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native as native
from psi4.driver.procrouting.isapol_oeprop import generated_recipe

POINTS = np.array([[3., .2, -.1], [-.3, 4., .2], [.1, -.2, -3.]])


@pytest.fixture(scope='module')
def fitted_water():
    core.be_quiet()
    water = psi4.geometry('0 1\nO 0 0 0\nH .757 0 .586\nH -.757 0 .586\n'
                          'symmetry c1\nno_com\nno_reorient')
    psi4.set_options({'basis': 'sto-3g', 'reference': 'rhf', 'scf_type': 'pk',
                      'e_convergence': 1e-11, 'd_convergence': 1e-11})
    _, wfn = psi4.energy('hf', molecule=water, return_wfn=True)
    properties = native.native_properties(
        wfn, generated_recipe(wfn), bonds=[(0, 1), (0, 2)], frames=None,
        caller_converged=True, kernel='no_local', exact_exchange=1., local_scale=0.,
        frequencies=(0., .4), response_grid=None, response_basis='fitted_auxiliary',
        ov_charge_penalty=1000., ov_metric_damping=0.)
    assert len(properties.coefficient_responses) == 2, properties.failures
    return wfn, properties


def test_fitted_targets_equal_independent_ov_solve(fitted_water):
    from psi4.driver.procrouting.isapol_native_fitted_point_response import native_fitted_point_response
    wfn, properties = fitted_water
    targets = native_fitted_point_response(
        properties, wfn, POINTS, charge_penalty=1000., metric_damping=0.)
    # Solve directly in OV space with fitted point legs, independently of the
    # producer's retained AUX-response contraction.
    provider = properties.context.response.provider
    h1, h2 = np.asarray(provider.h1()), np.asarray(provider.h2())
    potentials = np.asarray(properties.partition.coulomb.point_potentials(
        core.Matrix.from_array(POINTS)))
    legs = np.asarray(properties.ov_fit.coefficients) @ potentials
    for k, omega in enumerate(properties.frequencies):
        expected = -legs.T @ np.linalg.solve(
            h2 @ h1 + omega**2*np.eye(h1.shape[0]), -4*h2 @ legs)
        np.testing.assert_allclose(targets.responses[k], expected, rtol=2e-11, atol=1e-13)
        np.testing.assert_array_equal(targets.packed_targets[k],
                                      targets.responses[k][np.tril_indices(len(POINTS))])
    provenance = targets.target_provenance('small native water fitted-target integration')
    assert provenance.origin == core.IsaPfitTargetOrigin.NativeFittedPointResponse
    assert provenance.auxiliary_basis_id
    assert targets.representation == 'fitted_density_coefficients'


def test_budget_includes_auxiliary_validation_and_fit_copy(fitted_water):
    from psi4.driver.procrouting.isapol_native_fitted_point_response import native_fitted_point_response
    wfn, properties = fitted_water
    naux = properties.partition.auxiliary.nfunction
    # One point makes the omitted AUX^2 finite mask larger than the point
    # buffers. A budget covering just those buffers must still be refused.
    point_buffers_only = 8*(3*naux + 2*2 + 3 + 3)
    with pytest.raises(ValueError, match='byte resource'):
        native_fitted_point_response(properties, wfn, POINTS[:1], charge_penalty=1000.,
                                     metric_damping=0., max_bytes=point_buffers_only)


def produce(fitted_water, **kwargs):
    from psi4.driver.procrouting.isapol_native_fitted_point_response import native_fitted_point_response
    wfn, properties = fitted_water
    declarations = dict(charge_penalty=1000., metric_damping=0.)
    declarations.update(kwargs)
    return native_fitted_point_response(properties, wfn, POINTS, **declarations)


@pytest.mark.parametrize('declaration', [
    dict(charge_penalty=999.), dict(metric_damping=.0005),
])
def test_target_fit_is_not_inferred_from_an_anchor(fitted_water, declaration):
    with pytest.raises(ValueError, match='fit declaration mismatch'):
        produce(fitted_water, **declaration)


@pytest.mark.parametrize('change', ['incomplete', 'reordered', 'representation', 'nonfinite'])
def test_coefficient_grid_contract(fitted_water, change):
    wfn, properties = fitted_water
    responses = list(properties.coefficient_responses)
    if change == 'incomplete':
        responses.pop()
    elif change == 'reordered':
        responses.reverse()
    elif change == 'representation':
        responses[0] = replace(responses[0], representation='supplied_transition_leg_coordinates')
    else:
        raw = responses[0].raw_coupled.copy()
        raw[0, 0] = np.nan
        responses[0] = replace(responses[0], raw_coupled=raw)
    invalid = replace(properties, coefficient_responses=tuple(responses))
    with pytest.raises(ValueError, match='coefficient-response'):
        produce((wfn, invalid))


def test_changed_wavefunction_refused(fitted_water):
    wfn, _ = fitted_water
    energies = wfn.epsilon_a().np
    saved = energies.copy()
    try:
        energies[0] += .001
        with pytest.raises(ValueError, match='wavefunction context'):
            produce(fitted_water)
    finally:
        energies[:] = saved


def test_owned_targets_and_context_digest(fitted_water):
    a, b = produce(fitted_water), produce(fitted_water)
    assert a.context_sha256 == b.context_sha256
    with pytest.raises(ValueError):
        a.responses[0][0, 0] = 9.
    with pytest.raises(ValueError):
        a.packed_targets[0][0] = 9.
    wfn, properties = fitted_water
    raw = properties.coefficient_responses[0].raw_coupled.copy()
    # Same-centre s potentials can be proportional outside their densities,
    # hiding AUX asymmetry from point space. Select linearly independent legs.
    potentials = np.asarray(a.operators)
    wedge = np.outer(potentials[:, 0], potentials[:, 1]) - np.outer(
        potentials[:, 1], potentials[:, 0])
    i, j = np.unravel_index(np.argmax(np.abs(wedge)), wedge.shape)
    assert abs(wedge[i, j]) > 1e-6
    raw[i, j] += .001  # Deliberate algebraic input, not a physical native model.
    changed = replace(properties.coefficient_responses[0], raw_coupled=raw)
    synthetic = replace(properties, coefficient_responses=(changed, properties.coefficient_responses[1]))
    c = produce((wfn, synthetic))
    assert c.context_sha256 != a.context_sha256
    assert c.reciprocity_defects[0] > a.reciprocity_defects[0] + 1e-12
    np.testing.assert_array_equal(c.packed_targets[0], c.responses[0][np.tril_indices(3)])
    np.testing.assert_array_equal(a.responses[0], b.responses[0])


def test_rebuilt_propagator_uses_retained_responses(fitted_water):
    from psi4.driver.procrouting.isapol_native_propagator import PropagatorDeclaration
    wfn, _ = fitted_water
    properties = native.native_properties(
        wfn, generated_recipe(wfn), bonds=[(0, 1), (0, 2)], frames=None,
        caller_converged=True, kernel='no_local', exact_exchange=1., local_scale=0.,
        frequencies=(0., .4), response_grid=None, response_basis='fitted_auxiliary',
        ov_charge_penalty=1000., ov_metric_damping=0.,
        propagator=PropagatorDeclaration('density_fitted', 'orbital_product', 'exact_orbital'))
    assert len(properties.coefficient_responses) == 2, properties.failures
    targets = produce((wfn, properties))
    b = np.asarray(targets.operators)
    for k, response in enumerate(properties.coefficient_responses):
        np.testing.assert_allclose(targets.responses[k], -b.T @ response.raw_coupled @ b,
                                   rtol=2e-14, atol=2e-14)
    # Replacing the rebuilt operators with the original provider must not give
    # this result. This assertion detects the prior operator-reuse hazard.
    provider = properties.context.response.provider
    h1, h2 = np.asarray(provider.h1()), np.asarray(provider.h2())
    legs = np.asarray(properties.ov_fit.coefficients) @ b
    original = -legs.T @ np.linalg.solve(h2 @ h1, -4*h2 @ legs)
    assert np.max(np.abs(original-targets.responses[0])) > 1e-12


def test_native_fitted_origin_reaches_pfit_and_refuses_false_labels(fitted_water):
    targets = produce(fitted_water)
    # One caller-declared constant channel: the exact least-squares solution
    # is the mean of the packed data. This is a handoff test, not an atom model.
    def matrix(value):
        out = core.IsaPfitMatrix()
        out.rows, out.cols, out.values = 1, 1, [value]
        return out
    model = core.IsaPfitModel()
    model.channel_labels, model.parameter_labels, model.parameter_units = ['q'], ['p'], ['au']
    model.parameter_tensors, model.fixed, model.fixed_values = [matrix(1.)], [False], [0.]
    model.provenance = 'explicit one-channel test model; no molecular inference'
    penalty = core.IsaPfitMatrixPenalty()
    penalty.matrix, penalty.anchor = matrix(0.), [0.]
    problem = core.IsaPfitProblem()
    problem.model, problem.penalty = model, penalty
    problem.batches = [targets.batch('native fitted data', 0, np.ones((3, 1)))]
    provenance = targets.target_provenance('water integration')
    problem.target_provenance = provenance
    result = core.isa_pfit_solve(problem, core.IsaPfitOptions())
    assert result.status == core.IsaPfitStatus.Solved
    assert result.parameters[0] == pytest.approx(np.mean(targets.packed_targets[0]), abs=1e-13)
    # The authoritative C++ boundary must refuse false representation claims.
    for representation, auxiliary in [('native_point_charge_ov_operators', targets.auxiliary_basis_id),
                                      ('fitted_density_coefficients', '')]:
        provenance.response_representation, provenance.auxiliary_basis_id = representation, auxiliary
        problem.target_provenance = provenance
        with pytest.raises(ValueError):
            core.isa_pfit_solve(problem, core.IsaPfitOptions())


@pytest.fixture(scope='module')
def separate_anchors(fitted_water):
    wfn, targets = fitted_water
    anchors = native.native_properties(
        wfn, targets.partition.recipe, bonds=[(0, 1), (0, 2)], frames=None,
        caller_converged=True, kernel='no_local', exact_exchange=1., local_scale=0.,
        frequencies=targets.frequencies, response_grid=None, response_basis='fitted_auxiliary',
        ov_charge_penalty=1000., ov_metric_damping=.0005,
        response_context=targets.context)
    assert anchors.local is not None, anchors.failures
    return anchors


def test_refinement_selects_separate_target_and_anchor_fits(fitted_water, separate_anchors):
    from psi4.driver.procrouting.isapol_native_refinement import native_refinement
    from psi4.driver.procrouting.isapol_native_fitted_point_response import native_fitted_point_response
    wfn, targets = fitted_water
    anchors = separate_anchors
    result = native_refinement(
        anchors, wfn, site_types=('O', 'H1', 'H2'), rank_limits={'O': 1, 'H1': 1, 'H2': 1},
        npoints=32, fitted_target_properties=targets,
        target_charge_penalty=1000., target_metric_damping=0.)
    assert result.solved
    expected = native_fitted_point_response(targets, wfn, result.lattice.points_bohr,
                                            charge_penalty=1000., metric_damping=0.)
    for actual, wanted in zip(result.targets.responses, expected.responses):
        np.testing.assert_array_equal(actual, wanted)
    assert result.targets.metric_damping == 0.
    assert anchors.ov_fit.offsite_metric_damping == .0005
    assert targets.ov_fit.offsite_metric_damping == 0.
    for node in result.refinements:
        provenance = node.result.target_provenance
        assert provenance.origin == core.IsaPfitTargetOrigin.NativeFittedPointResponse
        assert provenance.auxiliary_basis_id == result.targets.auxiliary_basis_id
        assert provenance.auxiliary_basis_id
        assert provenance.response_representation == 'fitted_density_coefficients'
        assert provenance.generation_record == result.targets.generation_record
        assert provenance.source_id == result.source_id


@pytest.mark.parametrize('mismatch, message', [
    ('missing', 'explicit target'), ('orphan', 'require fitted_target_properties'),
    ('state', 'wavefunction contexts'), ('policy', 'response policies'),
    ('grid', 'frequency grids'), ('target_type', 'native target calculation'),
    ('anchor_local', 'No accepted localized'),
])
def test_separate_fit_orchestration_refusals(fitted_water, separate_anchors, mismatch, message):
    from psi4.driver.procrouting.isapol_native_refinement import native_refinement
    wfn, target = fitted_water
    anchor = separate_anchors
    declaration = dict(fitted_target_properties=target, target_charge_penalty=1000.,
                       target_metric_damping=0.)
    if mismatch == 'missing':
        declaration.pop('target_metric_damping')
    elif mismatch == 'orphan':
        declaration['fitted_target_properties'] = None
    elif mismatch == 'state':
        declaration['fitted_target_properties'] = replace(
            target, context=replace(target.context, wavefunction_sha256='different state'))
    elif mismatch == 'policy':
        declaration['fitted_target_properties'] = replace(
            target, context=replace(target.context, policy_sha256='different response policy'))
    elif mismatch == 'grid':
        declaration['fitted_target_properties'] = replace(target, frequencies=(0., .5))
    elif mismatch == 'target_type':
        declaration['fitted_target_properties'] = object()
    else:
        anchor = replace(anchor, local=None)
    with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
        # Invalid point count would fail if execution reached cloud generation;
        # these declarations/anchor gates must take precedence.
        native_refinement(anchor, wfn, site_types=('O', 'H1', 'H2'),
                          rank_limits={'O': 1, 'H1': 1, 'H2': 1}, npoints=0, **declaration)


def test_target_localization_is_not_an_anchor_gate(fitted_water, separate_anchors):
    from psi4.driver.procrouting.isapol_native_refinement import native_refinement
    wfn, target = fitted_water
    unavailable_localization = replace(
        target, local=None, failures=(native.StageFailure('LW', 0., 'ValueError', 'test gate'),))
    result = native_refinement(
        separate_anchors, wfn, site_types=('O', 'H1', 'H2'),
        rank_limits={'O': 1, 'H1': 1, 'H2': 1}, npoints=32,
        fitted_target_properties=unavailable_localization,
        target_charge_penalty=1000., target_metric_damping=0.)
    assert result.solved
    assert result.targets.metric_damping == 0.


def test_default_refinement_still_uses_direct_targets(fitted_water, separate_anchors):
    from psi4.driver.procrouting.isapol_native_refinement import native_refinement
    from psi4.driver.procrouting.isapol_native_point_response import native_point_charge_response
    wfn, _ = fitted_water
    result = native_refinement(separate_anchors, wfn, site_types=('O', 'H1', 'H2'),
                               rank_limits={'O': 1, 'H1': 1, 'H2': 1}, npoints=32)
    expected = native_point_charge_response(
        separate_anchors.context.response, wfn, result.lattice.points_bohr,
        frequencies=separate_anchors.frequencies)
    assert result.targets.representation == 'native_point_charge_ov_operators'
    for actual, wanted in zip(result.targets.responses, expected.responses):
        np.testing.assert_array_equal(actual, wanted)
    for node in result.refinements:
        provenance = node.result.target_provenance
        assert provenance.origin == core.IsaPfitTargetOrigin.NativeDirectActualPointResponse
        assert provenance.auxiliary_basis_id == ''
        assert provenance.response_representation == expected.representation
        assert provenance.generation_record == expected.generation_record
