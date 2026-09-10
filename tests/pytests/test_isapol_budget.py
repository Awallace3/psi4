# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Property-anchored precision budget: rebuild fidelity and probe restrictions.

The budget answers "how well must intermediate X be known for property Y?" by
perturbing X and rebuilding downstream through the shipped objects.  Two things
must hold for an amplification to mean anything, and both are tested here:

* an unperturbed rebuild must reproduce the shipped property bit-exactly, so the
  rebuild path is the same computation and not a re-implementation of it, and
* a probe direction must preserve every structural invariant the unrelaxed
  production LW gate enforces, so the measured defect is physics and not the
  gate refusing malformed input.

Two probe geometries are covered.  The absolute (max-scaled) geometry is the
metric the shipped intermediate errors were recorded in; it is the right question
only for an intermediate whose elements share a scale.  The elementwise-relative
geometry is the right question for one spanning many decades, and the two are
never compared against each other -- that refusal is asserted here.

One intermediate is probed as raw parameters instead of as a sampled array --
the ISA-A exponential tail -- because that is the form its recorded error was
recorded in.  The identity that makes the comparison apples-to-apples is proved
here from the shipped evidence file itself, not asserted.

He is monatomic, so w_a/sum(w) == 1 identically: its ISA weights, and therefore
Q and every property, are exactly independent of the shape samples and of the
Drho-C coefficients.  That degeneracy is asserted rather than hidden, and the
non-degenerate partition measurement is a separate water test.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native_partition as p
from psi4.driver.procrouting import isapol_native as n
from psi4.driver.procrouting import isapol_budget as b
from psi4.driver.procrouting import isapol_oeprop as o

pytestmark = pytest.mark.quick

#: The provisionally recorded maximum joint-tail scaled error, from
#: ``PROVISIONAL_ACCEPTANCE.md`` TODO9 and the shipped evidence file.
RECORDED_RAW_TAIL = 2.3588804665973028e-8


def he_recipe(wfn):
    m = wfn.molecule(); c = (m.x(0), m.y(0), m.z(0))
    def s(a): return p.ShellRecipe(0, 0, (a,), ((2*a/np.pi)**.75,))
    auxiliary = p.BasisRecipe('explicit compact s/p/d/f AUX', 'test authored', 'Cartesian', (c,),
        tuple(s(a) for a in (.25, .5, 1., 2., 4., 8., 16., 32.)) +
        tuple(p.ShellRecipe(0, l, (a,), (1.,)) for l in (1, 2, 3) for a in (.4, 1.2)))
    atomic = p.BasisRecipe('distinct s AtomAux', 'test authored', 'Spherical', (c,),
                           tuple(s(a) for a in (.3, .7, 1.5, 3.5, 9., 24.)))
    site = p.SiteRecipe('He', c, atomic, replace(atomic, name='Shape'), tuple(range(6)), 3, 1.5, True)
    return p.PartitionRecipe('He native budget', 'test numeric basis definitions only',
        'explicit_cartesian_drho_c_isa_a', auxiliary, (site,),
        p.GridRecipe(100, 110, 3, 1., 'native_tabulated_bragg_slater',
                     'all_sites_unscreened_full_molecular_grid'),
        p.ControllerRecipe(1e-9, 120, .17, .001, .2, True, 0., True, 1e-36,
                           1e-5, 1e-5, 1e-5, 0., 20, 20, True), 'Drho1e-2')


@pytest.fixture(scope='module')
def helium():
    core.be_quiet()
    psi4.basis_helper('assign budget_he\n[budget_he]\nspherical\n****\nHe 0\n'
        'S 1 1.0\n1.7 1.0\nP 1 1.0\n.7 1.0\nD 1 1.0\n.8 1.0\nF 1 1.0\n.9 1.0\n****\n',
        name='NATIVE_BUDGET_HE')
    mol = psi4.geometry('0 1\nHe .17 -.23 .31\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n')
    psi4.set_options({'basis': 'NATIVE_BUDGET_HE', 'puream': True, 'reference': 'rhf',
                      'scf_type': 'pk', 'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('hf', molecule=mol, return_wfn=True)
    return wfn


def he_properties(wfn, **kw):
    quad = n.Quadrature.from_casimir(core.CasimirGrid(4, .5))
    options = dict(bonds=(), frames=None, caller_converged=True, kernel='no_local',
                   exact_exchange=1., local_scale=0., response_grid=None,
                   frequencies=quad.frequencies, quadrature=quad, pair_self=True)
    options.update(kw)
    return n.native_properties(wfn, he_recipe(wfn), **options)


@pytest.fixture(scope='module')
def fitted(helium):
    properties = he_properties(helium)
    assert not properties.failures, [f.message for f in properties.failures]
    return b.budget_chain(properties)


@pytest.fixture(scope='module')
def direct(helium):
    properties = he_properties(helium, response_basis='direct_ov')
    assert not properties.failures, [f.message for f in properties.failures]
    return b.budget_chain(properties)


def test_chain_records_the_shipped_declarations(fitted, direct):
    assert fitted.response_basis == 'fitted_auxiliary'
    assert direct.response_basis == 'direct_ov'
    for chain in (fitted, direct):
        assert chain.rank == 3 and chain.pair_self and chain.partner is None
        assert chain.max_order == 12
        assert chain.frequencies == chain.properties.frequencies
        np.testing.assert_allclose(chain.adapted, chain.properties.full_adapted_orbitals.array,
                                   rtol=0, atol=0)


def test_unperturbed_rebuild_is_bit_exact(fitted):
    """The rebuild path must BE the shipped computation, not resemble it."""
    baseline = b.property_groups(fitted.local, fitted.properties.dispersion)
    assert set(baseline) == {'alpha_iso_rank1', 'alpha_iso_rank2', 'alpha_iso_rank3',
                             'C6', 'C8', 'C10', 'C12'}
    for stage in b.STAGES:
        reference = b.reference_value(fitted, stage)
        local, dispersion = b.rebuild(fitted, stage, reference)
        defects = b._defects(baseline, local, dispersion)
        assert max(defects.values()) == 0., (stage, defects)


def test_direct_ov_legs_are_not_a_fitted_intermediate(direct):
    with pytest.raises(ValueError, match='coordinate declaration'):
        b.reference_value(direct, 'ov_transition_legs')
    assert b._charge_vector(direct) is None


def test_probe_defect_is_exactly_the_requested_epsilon(fitted):
    for stage in b.STAGES:
        reference = b.reference_value(fitted, stage)
        assert b.scaled_max(b.perturb(fitted, stage, reference, 0., 0), reference) == 0.
        for epsilon in (1.e-6, 5.e-7):
            value = b.perturb(fitted, stage, reference, epsilon, 1)
            assert b.scaled_max(value, reference) == pytest.approx(epsilon, rel=1e-14)
            assert value is not reference


def test_directions_are_deterministic_and_distinct(fitted):
    reference = b.reference_value(fitted, 'coefficient_responses')
    first = b.restricted_direction(fitted, 'coefficient_responses', reference, 1)
    again = b.restricted_direction(fitted, 'coefficient_responses', reference, 1)
    other = b.restricted_direction(fitted, 'coefficient_responses', reference, 2)
    np.testing.assert_allclose(first, again, rtol=0, atol=0)
    assert np.max(np.abs(first - other)) > 0.
    assert b.sign_direction('seed', 600).shape == (600,)
    assert set(np.unique(b.sign_direction('seed', 600))) == {-1., 1.}
    with pytest.raises(ValueError):
        b.sign_direction('seed', 0)


def test_leg_directions_stay_charge_neutral(fitted):
    charges = np.asarray(fitted.properties.ov_fit.charges, dtype=float)
    assert float(charges @ charges) > 0.
    reference = b.reference_value(fitted, 'ov_transition_legs')
    np.testing.assert_allclose(np.asarray(reference) @ charges, 0., atol=1e-8)
    for index in range(3):
        delta = b.restricted_direction(fitted, 'ov_transition_legs', reference, index)
        assert np.max(np.abs(delta @ charges)) < 1e-12 * max(1., float(np.max(np.abs(charges))))


def test_coupled_directions_stay_symmetric_and_charge_null(fitted):
    charges = np.asarray(fitted.properties.ov_fit.charges, dtype=float)
    reference = b.reference_value(fitted, 'coefficient_responses')
    for index in range(3):
        delta = b.restricted_direction(fitted, 'coefficient_responses', reference, index)
        for block in delta:
            np.testing.assert_allclose(block, block.T, rtol=0, atol=1e-15)
            assert np.max(np.abs(block @ charges)) < 1e-12


def test_pair_tensor_directions_keep_reciprocity_and_charge_sums(fitted):
    reference = b.reference_value(fitted, 'distributed_site_tensors')
    for index in range(3):
        delta = b.restricted_direction(fitted, 'distributed_site_tensors', reference, index)
        np.testing.assert_allclose(delta, delta.transpose(0, 2, 1, 4, 3), rtol=0, atol=1e-15)
        assert b._charge_sum_defect(delta) < 1e-12 * float(np.max(np.abs(delta)))


def test_pair_tensor_manifold_refuses_a_direction_it_cannot_project():
    class Fake:
        response_basis = 'direct_ov'
        provenance = 'fake'
    # Every component of a rank-0 pair tensor is a charge-charge element, so the
    # charge-sum-free subspace of a two-site rank-0 tensor kills the whole probe.
    with pytest.raises(RuntimeError, match='reciprocal charge-sum-free manifold'):
        b.restricted_direction(Fake(), 'distributed_site_tensors', np.zeros((1, 2, 2, 1, 1)), 0)


def test_budget_measures_the_last_stage_and_reports_a_requirement(fitted):
    budget = b.precision_budget(fitted, property_tolerances=1.e-6,
                                stages=('distributed_site_tensors',), epsilon=1.e-6, directions=1)
    assert budget.metric == b.METRIC == 'max(abs(actual-reference))/max(1,max(abs(reference)))'
    assert budget.status == b.STATUS and 'not_a_gate' in budget.status
    assert {probe.status for probe in budget.probes} == {'measured'}
    assert max(budget.self_consistency['distributed_site_tensors'].values()) == 0.
    quoted = {r.property_name: r for r in budget.requirements}
    assert 'alpha_iso_rank1' in quoted
    for requirement in budget.requirements:
        assert requirement.restriction == 'reciprocity_and_charge_sum_preserving'
        assert requirement.required_precision == pytest.approx(
            1.e-6 / requirement.amplification, rel=1e-12)
    # A rank-only distributed perturbation moves the static polarizability by
    # exactly its own size: the local model is linear in the supplied tensors.
    assert quoted['alpha_iso_rank1'].amplification == pytest.approx(1., rel=1e-6)


@pytest.mark.parametrize('stage', ['partition_shape_samples', 'raw_tail_parameters'])
def test_monatomic_partition_stages_are_structurally_insensitive(fitted, stage):
    """He has one site: Q cannot depend on the shapes, tails or Drho-C density."""
    budget = b.precision_budget(fitted, property_tolerances=1.e-6,
                                stages=(stage,), epsilon=1.e-6, directions=1,
                                recorded_errors={stage: RECORDED_RAW_TAIL})
    assert {probe.status for probe in budget.probes} == {'measured'}
    assert {a.amplification for a in budget.amplifications} == {0.}
    for requirement in budget.requirements:
        assert requirement.required_precision == float('inf')
        assert 'structurally insensitive' in requirement.note
        assert requirement.satisfied is True


def test_budget_rejects_inputs_it_cannot_defend(fitted):
    with pytest.raises(TypeError):
        b.precision_budget(fitted.properties, property_tolerances=1.e-6)
    with pytest.raises(TypeError):
        b.budget_chain(fitted)
    for bad in dict(epsilon=0.), dict(epsilon=1.), dict(epsilon=float('nan')):
        with pytest.raises(ValueError):
            b.precision_budget(fitted, property_tolerances=1.e-6, **bad)
    for bad in dict(directions=0), dict(directions=2.), dict(directions=33):
        with pytest.raises((ValueError, TypeError)):
            b.precision_budget(fitted, property_tolerances=1.e-6, **bad)
    with pytest.raises(ValueError):
        b.precision_budget(fitted, property_tolerances=1.e-6, stages=())
    with pytest.raises(ValueError):
        b.precision_budget(fitted, property_tolerances=1.e-6, stages=('nonsense',))
    with pytest.raises(ValueError):
        b.precision_budget(fitted, property_tolerances=1.e-6,
                           stages=('distributed_site_tensors', 'distributed_site_tensors'))
    with pytest.raises(ValueError):
        b.precision_budget(fitted, property_tolerances=-1.)
    with pytest.raises(ValueError):
        b.precision_budget(fitted, property_tolerances={'not_a_property': 1.e-6})
    with pytest.raises(ValueError):
        b.precision_budget(fitted, property_tolerances=1.e-6, recorded_errors={'nonsense': 1.})
    with pytest.raises(ValueError):
        b.reference_value(fitted, 'nonsense')
    with pytest.raises(ValueError):
        b.rebuild(fitted, 'nonsense', np.zeros(3))
    with pytest.raises(ValueError):
        b.scaled_max(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        b.scaled_max(np.array([np.nan]), np.zeros(1))
    with pytest.raises(ValueError):
        b.perturb(fitted, 'distributed_site_tensors',
                  b.reference_value(fitted, 'distributed_site_tensors'), -1.e-6, 0)
    with pytest.raises(ValueError):
        b.perturb(fitted, 'distributed_site_tensors',
                  b.reference_value(fitted, 'distributed_site_tensors'), 1.e-6, -1)


def test_recorded_joint_tail_error_is_this_modules_metric():
    """The recorded raw-tail number IS a ``scaled_max`` on the parameter array.

    The trajectory comparator scales each site's joint (amplitude, exponent)
    error by that site's own largest parameter; this module scales one array by
    its single largest.  On the shipped reference the site with the largest error
    also carries the largest parameter, so the two groupings coincide -- exactly,
    not approximately -- and the recorded number may be compared against a
    requirement derived in the absolute geometry.  Nothing else here may be.
    """
    evidence = json.loads((Path(__file__).parent
                           / 'data_isapol/psi4_provisional_acceptance_evidence.json').read_text())
    joint = [error for comparison in evidence['comparisons']
             for name, error in (comparison.get('errors') or {}).items() if name.endswith('_tail')]
    assert len(joint) == 3, 'three water sites carry a defined tail in the reference'
    # Recover each site's denominator from the pair the comparator reported.
    denominators = [e['max_absolute'] / e['max_scaled'] for e in joint]
    assert min(denominators) > 1., 'clamped denominators would make this identity vacuous'
    regrouped = max(e['max_absolute'] for e in joint) / max(1., max(denominators))
    assert regrouped == RECORDED_RAW_TAIL == max(e['max_scaled'] for e in joint)


def test_tail_parameters_are_probed_as_parameters_not_as_samples(fitted):
    """The probed intermediate is (amplitude, exponent) per applied tail."""
    state = fitted.properties.partition.trajectory.state
    sites = fitted.properties.partition.recipe.sites
    applied = b._applied_tails(state, sites)
    assert applied and all(state.tails[i].defined and sites[i].tail_allowed for i in applied)
    reference = b.reference_value(fitted, 'raw_tail_parameters')
    assert reference.shape == (len(applied), 2)
    np.testing.assert_allclose(reference, [[state.tails[i].amplitude, state.tails[i].exponent]
                                           for i in applied], rtol=0, atol=0)
    # The cutoff is supplied configuration, not a fitted intermediate: it is
    # outside the probed array and its invariance is part of the restriction.
    assert b.RESTRICTIONS['raw_tail_parameters'] == 'positive_exponent_supplied_cutoff_held_fixed'
    assert 'raw_tail_parameters' not in b.RELATIVE_ELIGIBLE
    with pytest.raises(ValueError, match='linear invariants'):
        b.perturb(fitted, 'raw_tail_parameters', reference, 1.e-6, 0, 'relative')
    with pytest.raises(ValueError, match='one \\(amplitude, exponent\\) row'):
        b.rebuild(fitted, 'raw_tail_parameters', reference[:, :1])
    negative = reference.copy()
    negative[0, 1] = -negative[0, 1]
    with pytest.raises(ValueError, match='positive exponent'):
        b.rebuild(fitted, 'raw_tail_parameters', negative)
    # A rebuild carries a surrogate state; the shipped tails stay untouched.
    b.rebuild(fitted, 'raw_tail_parameters', b.perturb(fitted, 'raw_tail_parameters',
                                                       reference, 1.e-3, 0))
    np.testing.assert_allclose(b.reference_value(fitted, 'raw_tail_parameters'), reference,
                               rtol=0, atol=0)
    assert [state.tails[i].cutoff for i in applied] == [sites[i].tail_cutoff for i in applied]


def test_budget_never_mutates_the_shipped_result(fitted):
    before = {stage: b.reference_value(fitted, stage).copy() for stage in b.STAGES}
    shipped = np.asarray(fitted.local.atomic_scalars.array).copy()
    b.precision_budget(fitted, property_tolerances=1.e-6,
                       stages=('distributed_site_tensors',), epsilon=1.e-6, directions=1)
    for stage, reference in before.items():
        np.testing.assert_allclose(b.reference_value(fitted, stage), reference, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(fitted.local.atomic_scalars.array), shipped,
                               rtol=0, atol=0)


def test_relative_defect_is_elementwise_and_keeps_zeros_zero():
    reference = np.array([[1.e3, 1.e-30], [0., -4.]])
    assert b.relative_max(reference, reference) == 0.
    assert b.scaled_max(reference, reference) == 0.
    # A max-scaled probe is invisible in the max-scaled metric where the value is
    # small and enormous in the relative one: that is the whole reason for the
    # second geometry, and it is asserted rather than described.
    absolute = reference + 1.e-6 * np.max(np.abs(reference)) * np.array([[1., 1.], [0., 1.]])
    assert b.scaled_max(absolute, reference) == pytest.approx(1.e-6, rel=1e-12)
    assert b.relative_max(absolute, reference) > 1.e20
    relative = reference * (1. + 1.e-6)
    assert b.relative_max(relative, reference) == pytest.approx(1.e-6, rel=1e-12)
    assert b.scaled_max(relative, reference) == pytest.approx(1.e-6, rel=1e-12)
    assert b.defect(relative, reference) == b.scaled_max(relative, reference)
    assert b.defect(relative, reference, 'relative') == b.relative_max(relative, reference)
    with pytest.raises(ValueError, match='probe geometry'):
        b.defect(relative, reference, 'nonsense')
    with pytest.raises(ValueError, match='exact zeros exactly zero'):
        b.relative_max(reference + np.array([[0., 0.], [1.e-40, 0.]]), reference)
    with pytest.raises(ValueError, match='nonzero reference'):
        b.relative_max(np.zeros(3), np.zeros(3))


def test_relative_probe_is_only_offered_where_it_preserves_the_invariants(fitted):
    assert set(b.RELATIVE_ELIGIBLE) == {'drho_c_coefficients', 'partition_shape_samples'}
    assert b.GEOMETRIES == ('absolute', 'relative')
    for stage in b.RELATIVE_ELIGIBLE:
        reference = b.reference_value(fitted, stage)
        value = b.perturb(fitted, stage, reference, 1.e-6, 1, 'relative')
        assert b.relative_max(value, reference) == pytest.approx(1.e-6, rel=1e-12)
        np.testing.assert_allclose(b.perturb(fitted, stage, reference, 0., 0, 'relative'),
                                   reference, rtol=0, atol=0)
    for stage in set(b.STAGES) - set(b.RELATIVE_ELIGIBLE):
        try:
            reference = b.reference_value(fitted, stage)
        except ValueError:
            continue
        with pytest.raises(ValueError, match='linear invariants'):
            b.perturb(fitted, stage, reference, 1.e-6, 1, 'relative')
    with pytest.raises(ValueError, match='probe geometry'):
        b.perturb(fitted, 'drho_c_coefficients',
                  b.reference_value(fitted, 'drho_c_coefficients'), 1.e-6, 1, 'nonsense')


def test_relative_budget_is_self_describing_and_refuses_a_mismatched_metric(fitted):
    budget = b.precision_budget(fitted, property_tolerances=1.e-6, epsilon=1.e-6, directions=1,
                                stages=('partition_shape_samples',), geometry='relative',
                                recorded_error_metric='relative')
    assert budget.geometry == 'relative'
    assert budget.metric == b.RELATIVE_METRIC != b.METRIC
    assert {probe.status for probe in budget.probes} == {'measured'}
    assert {probe.geometry for probe in budget.probes} == {'relative'}
    assert {probe.restriction for probe in budget.probes} == {'unperturbed', 'elementwise_relative'}
    assert {a.geometry for a in budget.amplifications} <= {'relative'}
    assert {r.geometry for r in budget.requirements} <= {'relative'}
    # An error recorded in one metric is never silently compared against a
    # requirement derived in the other.
    with pytest.raises(ValueError, match='cannot be compared'):
        b.precision_budget(fitted, property_tolerances=1.e-6, directions=1,
                           stages=('partition_shape_samples',), geometry='relative',
                           recorded_errors={'partition_shape_samples': RECORDED_RAW_TAIL})
    with pytest.raises(ValueError, match='cannot be compared'):
        b.precision_budget(fitted, property_tolerances=1.e-6, directions=1,
                           stages=('partition_shape_samples',), recorded_error_metric='relative',
                           recorded_errors={'partition_shape_samples': RECORDED_RAW_TAIL})
    for bad in dict(geometry='nonsense'), dict(recorded_error_metric='nonsense'):
        with pytest.raises(ValueError, match='geometry|metric'):
            b.precision_budget(fitted, property_tolerances=1.e-6, directions=1,
                               stages=('partition_shape_samples',), **bad)


@pytest.mark.long
def test_water_partition_stages_are_measurable():
    """The non-degenerate measurement: three sites, so Q depends on the shapes.

    Model declared here, not inferred: PBE0/cc-pVDZ, the shipped generated
    recipe, direct_ov, strict production LW.  Only the two partition stages are
    probed; this test asserts that they carry information at all, which is
    exactly what the monatomic fixture cannot show.
    """
    core.be_quiet()
    water = psi4.geometry('0 1\nO 0. 0. 0.\nH -1.45365196 0. -1.12168732\n'
                          'H 1.45365196 0. -1.12168732\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n')
    psi4.set_options({'basis': 'cc-pvdz', 'reference': 'rks', 'scf_type': 'pk',
                      'e_convergence': 1e-10, 'd_convergence': 1e-10,
                      'dft_radial_points': 99, 'dft_spherical_points': 590, 'dft_alpha': .25})
    _, wfn = psi4.energy('pbe0', molecule=water, return_wfn=True)
    recipe = o.generated_recipe(wfn)
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 99, 590
    grid = core.IsaGrid(wfn.molecule().clone(), options)
    response_grid = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    quadrature = n.Quadrature.from_casimir(core.CasimirGrid(4, .5))
    properties = n.native_properties(wfn, recipe, bonds=((1, 0), (2, 0)), frames=None,
        caller_converged=True, kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75,
        response_grid=response_grid, frequencies=quadrature.frequencies, quadrature=quadrature,
        pair_self=True, response_basis='direct_ov', response_algorithm='shared_sweep')
    assert not properties.failures, [f.message for f in properties.failures]
    chain = b.budget_chain(properties)
    budget = b.precision_budget(chain, property_tolerances=1.e-6, epsilon=1.e-6, directions=1,
                                stages=('partition_shape_samples', 'drho_c_coefficients'))
    assert {probe.status for probe in budget.probes} == {'measured'}
    for stage in ('partition_shape_samples', 'drho_c_coefficients'):
        assert max(budget.self_consistency[stage].values()) == 0.
        measured = [a.amplification for a in budget.amplifications if a.stage == stage]
        assert measured and max(measured) > 0., stage

    # The shape samples span many decades, so the max-scaled probe swamps the
    # tail and reports an enormous amplification; the elementwise-relative probe
    # is the property-relevant question and reports an attenuation instead.  The
    # two numbers are the same intermediate measured in different metrics.
    relative = b.precision_budget(chain, property_tolerances=1.e-6, epsilon=1.e-6, directions=1,
                                  stages=('partition_shape_samples',), geometry='relative')
    assert relative.geometry == 'relative' and relative.metric == b.RELATIVE_METRIC
    assert max(relative.self_consistency['partition_shape_samples'].values()) == 0.
    absolute_amplification = max(a.amplification for a in budget.amplifications
                                 if a.stage == 'partition_shape_samples')
    relative_amplification = max(a.amplification for a in relative.amplifications)
    assert absolute_amplification > 1.e4 > 1. > relative_amplification > 0.

    # The raw tail parameters, by contrast, are O(1) numbers sharing one scale,
    # so the absolute geometry IS their property-relevant error model: the
    # amplification is a converged derivative (it survives halving the probe at
    # two probe sizes four decades apart) and the recorded joint-tail error is
    # compared in exactly the metric it was recorded in.
    tails = [b.precision_budget(chain, property_tolerances=1.e-6, epsilon=eps, directions=1,
                                stages=('raw_tail_parameters',),
                                recorded_errors={'raw_tail_parameters': RECORDED_RAW_TAIL})
             for eps in (1.e-6, 1.e-10)]
    for measured in tails:
        assert {probe.status for probe in measured.probes} == {'measured'}
        assert max(measured.self_consistency['raw_tail_parameters'].values()) == 0.
        assert all(a.quoted for a in measured.amplifications)
        assert all(r.satisfied is True and r.recorded_error == RECORDED_RAW_TAIL
                   for r in measured.requirements)
    coarse, fine = ({a.property_name: a.amplification for a in measured.amplifications}
                    for measured in tails)
    assert set(coarse) == set(fine)
    for name, value in coarse.items():
        assert 1. < value < 1.e3, (name, value)
        assert fine[name] == pytest.approx(value, rel=1.e-3), name
