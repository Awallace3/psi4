# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The DF-centre distributed-multipole declaration: one rule, two forms, gated.

``ATOMIC_MULTIPOLE_DISTRIBUTION`` names which *model* distributes the response,
not how accurately one model is solved.  ``ISA_A`` charges each grid point to a
site by the stockholder ratio ``shape_a / sum_b shape_b``; ``DF_CENTRE_*`` is
CamCASP's ``DistPolAlgorithm = DF``, which charges every auxiliary function
*wholly* to the centre it sits on.  They answer different questions, so their
site multipoles, site polarizabilities and C_n are never quoted as agreeing --
the last test here measures how far apart they are on one wavefunction precisely
so that the declaration cannot be mistaken for a tolerance.

``DF_CENTRE_ANALYTIC`` and ``DF_CENTRE_GRID`` *are* the same rule.  Nothing is
approximated in the closed form and nothing is modelled differently in the grid
form; the only thing between them is the molecular quadrature, which is what
``test_the_two_forms_are_one_rule_...`` shows by refining the grid and watching
the gap fall.  Every number here is cheap and SCF-free except the two public
runs at the end.  The full aug-cc-pVTZ-RI comparison against the archived
reference protocol lives in the gitignored
``agent_scratch/pytests/test_isapol_df_centre_public_full.py``, which also
guards every literal recorded below.
"""
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_oeprop as api
from psi4.driver.procrouting import isapol_df_multipoles as dfm

#: The shipped Racah evaluator recovers exactly; these are the measured residuals
#: of the monomial least squares at ranks 0..3, seed 20260910.
HARMONIC_RESIDUALS = (6.661338147750939e-16, 1.3322676295501878e-15,
                      1.0658141036401503e-14, 2.3092638912203256e-14)
#: cc-pVDZ-JKFIT water AUX: 42 shells, 131 Cartesian functions, 3 sites.
NFUNCTION = 131
#: GAMINT convention check and the closed-form charge row, both on that AUX.
CONVENTION = (6.661338147750939e-16, 2.643465579626019)
CHARGE_ROW_ERROR = 7.105427357601002e-15
CHARGE_ROW_MAGNITUDE = 10.197510192046527
#: Grid-vs-closed-form maximum relative difference per rank, by (radial, spherical).
#: One rule, quadrature only: every entry falls when the grid is refined.
GRID_CONVERGENCE = {
    (50, 110): (5.263600e-05, 1.102143e-04, 1.018353e-04, 1.305225e-03),
    (75, 302): (2.27e-07, 3.64e-07, 3.00e-07, 1.37e-06),
    (99, 590): (9.95e-09, 4.62e-09, 9.00e-10, 9.24e-09),
}


@pytest.fixture(scope='module')
def declared():
    """An AUX recipe and its sites with no SCF at all: the rule needs neither."""
    core.be_quiet()
    wfn = core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'), 'cc-pvdz')
    recipe = api.generated_recipe(wfn)
    assert recipe.auxiliary.name == 'cc-pVDZ-JKFIT Cartesian molecular AUX'
    return wfn, recipe


@pytest.fixture(scope='module')
def analytic(declared):
    _, recipe = declared
    return dfm.analytic_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3)


def _grid(wfn, radial, spherical):
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = radial, spherical
    grid = core.IsaGrid(wfn.molecule().clone(), options)
    return np.column_stack((grid.x(), grid.y(), grid.z())), np.asarray(grid.w())


def test_harmonic_expansion_recovers_the_shipped_racah_evaluator():
    """The closed form's only non-elementary input, recovered and then gated.

    The Cartesian monomial coefficients of the Racah regular harmonics are not
    hard-coded: they are least-squares recovered from the shipped
    ``core.isa_regular_multipoles`` at a declared seed, so the closed form and
    the grid form are expanding the *same* harmonics by construction.  The
    residual is therefore a precondition, not a diagnostic.
    """
    expansion = dfm.harmonic_expansion(3)
    assert expansion.rank == 3 and expansion.seed == dfm.HARMONIC_SEED
    assert expansion.residuals == HARMONIC_RESIDUALS
    assert expansion.residual_max == max(HARMONIC_RESIDUALS) < dfm.HARMONIC_TOLERANCE
    assert [expansion.block(l).shape for l in range(4)] == [(1, 1), (3, 3), (6, 5), (10, 7)]
    assert [len(dfm.monomials(l)) for l in range(4)] == [1, 3, 6, 10]
    with pytest.raises(RuntimeError, match='monomial recovery failed'):
        dfm.harmonic_expansion(3, tolerance=1.e-18)


def test_the_cartesian_convention_is_verified_against_the_built_basis(declared):
    """That the recipe's shell list and the built basis agree function by function.

    The closed form walks ``CARTESIAN_POWERS`` in GAMINT order over the recipe's
    own shells; if the built basis ordered or normalized its columns differently
    the Q would be silently wrong, so the two are compared on random points.
    """
    _, recipe = declared
    built = recipe.auxiliary.build('MolecularAux')
    error, magnitude = dfm.verify_cartesian_convention(recipe.auxiliary, built)
    assert (error, magnitude) == CONVENTION
    assert error < dfm.CONVENTION_TOLERANCE*magnitude
    with pytest.raises(RuntimeError, match='GAMINT Cartesian convention check failed'):
        dfm.verify_cartesian_convention(recipe.auxiliary, built, tolerance=1.e-18)


def test_the_charge_row_is_checked_against_an_independent_gaussian_moment(declared, analytic):
    """``R_00 = 1``, so the rank-0 row is just ``int chi_k`` on each function's own centre.

    That derivation shares no code with the harmonic expansion, which is what
    makes it a real check on the closed form rather than a restatement of it.
    """
    _, recipe = declared
    rows = dfm.closed_form_charge_rows(recipe.auxiliary, 3)
    assert rows.shape == (3, NFUNCTION)
    assert analytic.values.shape == (3*16, NFUNCTION)
    np.testing.assert_allclose(analytic.values[[0, 16, 32]], rows, rtol=0, atol=1.e-13)
    assert analytic.diagnostics['charge_row_error'] == CHARGE_ROW_ERROR
    assert analytic.diagnostics['charge_row_magnitude'] == CHARGE_ROW_MAGNITUDE
    assert analytic.diagnostics['quadrature_defect'] == 'none; nothing is sampled'
    with pytest.raises(RuntimeError, match='charge row disagrees with the direct'):
        dfm.analytic_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3,
                                          charge_tolerance=1.e-18)


def test_the_two_forms_are_one_rule_with_only_quadrature_between_them(declared, analytic):
    """Refine the grid and the gap falls: there is no second model here.

    Each rank is followed separately because a single maximum would hide which
    moment carries the quadrature error -- it is spread over all four rows, and
    worst in relative terms on rank 3, not concentrated in the charge row.
    """
    wfn, recipe = declared
    m, previous = 16, None
    for (radial, spherical), recorded in GRID_CONVERGENCE.items():
        points, weights = _grid(wfn, radial, spherical)
        grid = dfm.grid_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3, points, weights)
        assert grid.form == 'grid' and grid.rank == 3
        assert grid.diagnostics['grid_points'] == points.shape[0]
        assert grid.diagnostics['negative_ratios'] == (0, 0, 0)
        assert not grid.diagnostics['excluded_denominators'][0]
        measured = []
        for l in range(4):
            rows = [i*m + l*l + t for i in range(3) for t in range(2*l+1)]
            difference = float(np.abs(grid.values[rows]-analytic.values[rows]).max())
            measured.append(difference/float(np.abs(analytic.values[rows]).max()))
        # The recorded table, to the digits it records, and strictly decreasing.
        np.testing.assert_allclose(measured, recorded, rtol=5.e-2, atol=0)
        if previous is not None:
            assert all(a < b for a, b in zip(measured, previous))
        previous = measured
    assert max(previous) < 1.e-8


def test_the_grid_form_may_declare_a_charge_row_tolerance_and_be_refused(declared):
    """The grid form carries quadrature error, so its tolerance is the caller's.

    It defaults to ``None`` -- reported, not enforced -- because the honest
    number depends on the grid the caller declared; asking for the closed form's
    accuracy from a 16k-point grid is refused rather than rounded away.
    """
    wfn, recipe = declared
    points, weights = _grid(wfn, 50, 110)
    with pytest.raises(RuntimeError, match='from the closed-form Gaussian moment'):
        dfm.grid_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3, points, weights,
                                      charge_tolerance=1.e-12)


def test_the_supplied_values_record_carries_the_declared_axes(declared, analytic):
    """``DFCentreMultipoles.partition()`` round-trips bitwise and declares its axes.

    Going through the shipped ``IsaPartitionedMultipoles`` is what puts the
    DF-centre Q on the same ``-Q C Q^T`` path as the stockholder Q, so there is
    no second contraction anywhere; that only works if the values survive the
    trip unchanged.
    """
    _, recipe = declared
    partition = analytic.partition()
    np.testing.assert_array_equal(np.asarray(partition.values), analytic.values)
    assert tuple(partition.labels) == ('O1', 'H2', 'H3') == analytic.labels
    assert tuple(partition.offsets) == (0, 16, 32, 48)
    assert tuple(partition.ranks) == (3, 3, 3)
    assert tuple(partition.components)[:5] == ('00', '10', '11c', '11s', '20')
    assert partition.representation == dfm.REPRESENTATION == 'fitted_density_coefficients'
    assert 'DistPolAlgorithm=DF' in partition.provenance
    np.testing.assert_array_equal(np.asarray(partition.origins),
                                  np.asarray(recipe.auxiliary.centres))
    # The sites it declares carry no samples: nothing was integrated.
    assert all(site.samples is None or not site.samples.points for site in analytic.sites())


def test_the_df_centre_q_reaches_the_shipped_minus_q_c_qt(analytic):
    """``alpha = -Q C Q^T`` through the shipped class, bitwise against numpy.

    ``-C(i xi)`` is positive semidefinite, which is why a negative site
    isotropic scalar is a broken model rather than an unconverged one; that is
    the premise ``site_isotropic_gate`` enforces, so it is checked here.
    """
    q = analytic.values
    rng = np.random.default_rng(3)
    legs = rng.normal(size=(q.shape[1], 4))
    coupled = legs @ legs.T
    response = core.IsaDistributedResponse(analytic.partition(), [0.],
                                           [core.Matrix.from_array(coupled)],
                                           dfm.REPRESENTATION, 'reduced in-repo probe')
    np.testing.assert_array_equal(np.asarray(response.at_index(0)), -(q @ coupled) @ q.T)
    assert tuple(response.frequencies) == (0.,)
    assert max(response.reciprocity_errors) < 1.e-14


def test_the_site_isotropic_gate_refuses_a_negative_static_site(analytic):
    """A negative site response must not reach a polarizability or a C_n.

    Measured on hydrogen at rank 3 with a MAIN-matched JKFIT auxiliary set, the
    DF rule can charge enough of a diffuse function's moment to a light centre
    to drive the site's own static response negative.  The refusal names the
    site, the rank, the value and the alternative declarations; it does not
    tighten anything, because there is nothing here to converge.
    """
    labels, m, rank = analytic.labels, 16, 3
    rng = np.random.default_rng(7)
    root = rng.normal(size=(3*m, 3*m))
    psd = (root @ root.T).reshape(3, m, 3, m).transpose(0, 2, 1, 3)
    scalars = dfm.site_isotropic_gate(psd, rank, labels=labels)
    assert set(scalars) == {(a, l) for a in labels for l in (1, 2, 3)}
    assert min(scalars.values()) > 0.
    broken = psd.copy()
    broken[1, 1, 1:4, 1:4] *= -1.
    with pytest.raises(RuntimeError) as excinfo:
        dfm.site_isotropic_gate(broken, rank, labels=labels)
    message = str(excinfo.value)
    assert 'site H2 rank 1' in message and 'aug-cc-pVTZ-RI' in message
    assert '1 of 9 site/rank scalars are negative' in message
    with pytest.raises(ValueError, match='site,site,component,component'):
        dfm.site_isotropic_gate(np.zeros((3, 3, 9, 9)), rank, labels=labels)


@pytest.mark.parametrize('kwargs,pattern', [
    (dict(rank=0), 'rank must be an integer 1 to 4, not 0'),
    (dict(rank=5), 'rank must be an integer 1 to 4, not 5'),
    (dict(rank=True), 'rank must be an integer 1 to 4, not True'),
    (dict(rank=3.), 'rank must be an integer 1 to 4, not 3.0'),
    (dict(rank=3, nsite=2), 'needs one site per auxiliary centre'),
    (dict(rank=3, displace=True), 'centre order must match site order exactly'),
])
def test_the_rule_refuses_anything_but_an_exact_site_to_centre_correspondence(
        declared, kwargs, pattern):
    """A displaced or miscounted site would make this a different, undeclared rule.

    "Every auxiliary function wholly on its own centre" has no meaning if the
    site is not *at* the centre, so the correspondence is exact rather than
    within a tolerance.
    """
    _, recipe = declared
    sites = list(recipe.sites)[:kwargs.pop('nsite', None) or len(recipe.sites)]
    if kwargs.pop('displace', False):
        moved = core.IsaMultipoleSite()
        moved.label, moved.origin, moved.rank = sites[0].label, [0., 0., 1.], 3
        sites = [moved] + sites[1:]
    with pytest.raises(ValueError, match=pattern):
        dfm.analytic_df_centre_multipoles(recipe.auxiliary, sites, **kwargs)


def test_the_dispatcher_refuses_a_form_handed_the_wrong_operands(declared):
    """The closed form samples nothing and the grid form integrates something."""
    wfn, recipe = declared
    points, weights = _grid(wfn, 50, 110)
    assert dfm.DISTRIBUTIONS == ('isa_a', 'df_centre_analytic', 'df_centre_grid')
    assert dfm.DF_CENTRE_DISTRIBUTIONS == ('df_centre_analytic', 'df_centre_grid')
    with pytest.raises(ValueError, match='Unknown DF-centre distribution'):
        dfm.df_centre_multipoles('isa_a', recipe.auxiliary, recipe.sites, 3)
    with pytest.raises(ValueError, match='samples nothing; do not hand it a grid'):
        dfm.df_centre_multipoles('df_centre_analytic', recipe.auxiliary, recipe.sites, 3,
                                 points=points, weights=weights)
    with pytest.raises(ValueError, match='needs the molecular quadrature'):
        dfm.df_centre_multipoles('df_centre_grid', recipe.auxiliary, recipe.sites, 3)
    with pytest.raises(ValueError, match=r'matching \(npoint,3\) points'):
        dfm.df_centre_multipoles('df_centre_grid', recipe.auxiliary, recipe.sites, 3,
                                 points=points, weights=weights[:-1])
    with pytest.raises(ValueError, match='Supplied harmonic expansion is rank 2'):
        dfm.analytic_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3,
                                          expansion=dfm.harmonic_expansion(2))


def test_the_public_options_are_declared_together_and_refused_together():
    """``lambda``/``eta`` belong to the fit; the DF rule needs the fit's columns.

    The DF-centre rule charges a *function* to a centre, so occupied-virtual
    product columns have no centre for it to sit on: ``DIRECT_OV`` is refused
    rather than silently promoted to the fitted route.  And ``DIRECT_OV`` forms
    no fit at all, so declaring a penalty or a damping under it is a
    contradiction, not an ignorable option.
    """
    base = {'ATOMIC_MULTIPOLE_DISTRIBUTION': 'ISA_A', 'ATOMIC_RESPONSE_BASIS': 'DIRECT_OV',
            'ATOMIC_OV_CHARGE_PENALTY': 1., 'ATOMIC_OV_METRIC_DAMPING': 0.}
    assert api._distribution_options(base) == dict(
        distribution='isa_a', response_basis='direct_ov',
        ov_charge_penalty=1., ov_metric_damping=0.)
    assert api._distribution_options(dict(
        base, ATOMIC_MULTIPOLE_DISTRIBUTION='DF_CENTRE_GRID',
        ATOMIC_RESPONSE_BASIS='FITTED_AUXILIARY', ATOMIC_OV_CHARGE_PENALTY=1000.,
        ATOMIC_OV_METRIC_DAMPING=5.e-4)) == dict(
            distribution='df_centre_grid', response_basis='fitted_auxiliary',
            ov_charge_penalty=1000., ov_metric_damping=5.e-4)
    with pytest.raises(ValueError, match='no centre for a function to sit on'):
        api._distribution_options(dict(base, ATOMIC_MULTIPOLE_DISTRIBUTION='DF_CENTRE_ANALYTIC'))
    for contradiction in ({'ATOMIC_OV_CHARGE_PENALTY': 1000.},
                          {'ATOMIC_OV_METRIC_DAMPING': 5.e-4}):
        with pytest.raises(ValueError, match='forms no transition-density fit'):
            api._distribution_options(dict(base, **contradiction))


def test_the_four_options_default_to_the_previously_hard_coded_model():
    """Adding the declaration must not move the public path's existing default.

    Before these options existed the public path was ``ISA_A`` with fit-free
    ``DIRECT_OV`` columns; those are exactly the defaults, so every existing
    result is the same model it was.
    """
    psi4.core.clean_options()
    assert psi4.core.get_global_option('ATOMIC_MULTIPOLE_DISTRIBUTION') == 'ISA_A'
    assert psi4.core.get_global_option('ATOMIC_RESPONSE_BASIS') == 'DIRECT_OV'
    assert psi4.core.get_global_option('ATOMIC_OV_CHARGE_PENALTY') == 1.
    assert psi4.core.get_global_option('ATOMIC_OV_METRIC_DAMPING') == 0.
    with pytest.raises(Exception, match='not a valid choice'):
        psi4.set_options({'atomic_multipole_distribution': 'stockholder'})
    psi4.core.clean_options()


@pytest.fixture(scope='module')
def public_models():
    """One PBE0/cc-pVDZ water endpoint run twice: the two declared distributions.

    Both are complete public ``oeprop`` runs -- options in, wavefunction
    variables out -- because the point of the declaration is that it reaches the
    public path, not that the module can be called directly.
    """
    out = {}
    for name, extra in (('isa_a', {}),
                        ('df_centre_analytic',
                         {'atomic_multipole_distribution': 'df_centre_analytic',
                          'atomic_response_basis': 'fitted_auxiliary',
                          'atomic_ov_charge_penalty': 1000.,
                          'atomic_ov_metric_damping': 5.e-4})):
        psi4.core.clean_options()
        psi4.core.be_quiet()
        mol = psi4.geometry('0 1\nO 0. 0. 0.\nH -1.45365196 0. -1.12168732\n'
                            'H 1.45365196 0. -1.12168732\nunits bohr\nsymmetry c1\n'
                            'no_com\nno_reorient\n')
        psi4.set_options(dict({'basis': 'cc-pvdz', 'reference': 'rks', 'scf_type': 'pk',
                               'e_convergence': 1e-10, 'd_convergence': 1e-10,
                               'atomic_property_print': 0}, **extra))
        _, wfn = psi4.energy('pbe0', molecule=mol, return_wfn=True)
        assert psi4.oeprop(wfn, 'ATOMIC_POLARIZABILITIES') is None
        out[name] = psi4.atomic_property_result(wfn)
    psi4.core.clean_options()
    return out


def test_the_public_path_runs_the_declared_df_centre_rule(public_models):
    """The declaration reaches the model string, the gates and the diagnostics.

    The ISA-A partition still runs and is still reported under a DF-centre
    declaration -- ``oeprop`` publishes a converged Drho-C partition either way
    -- but it is not this model's operand, and the diagnostics say so.
    """
    result = public_models['df_centre_analytic']
    properties = result.properties
    assert not properties.failures
    assert 'distribution=df_centre_analytic' in properties.model
    assert 'fitted_auxiliary lambda=1000.0; eta=0.0005' in properties.model
    diagnostics = properties.diagnostics
    assert diagnostics['distribution'] == 'df_centre_analytic'
    assert diagnostics['distribution_isa_a_converged'] is True
    assert result.partition.converged
    assert 'DistPolAlgorithm=DF' in diagnostics['distribution_provenance']
    assert diagnostics['distribution_harmonic_residuals'] == HARMONIC_RESIDUALS
    # The charge row is a pure Gaussian moment, so it does not depend on where
    # the molecule sits and is the same number the SCF-free fixture measured.
    assert diagnostics['distribution_charge_row_error'] == CHARGE_ROW_ERROR
    # The convention check evaluates the built basis on points drawn about the
    # molecule's own centres, so its residual moves with the geometry; only its
    # bound is a claim about the model.
    assert diagnostics['distribution_convention_error'] < dfm.CONVENTION_TOLERANCE
    # Every site/rank static isotropic scalar is published and every one passed.
    gate = {k: v for k, v in diagnostics.items() if 'site_isotropic' in k}
    assert len(gate) == 9 and min(gate.values()) > 0.
    assert set(gate) == {f'distribution_site_isotropic[xi=0.0][{a} rank{l}]'
                         for a in ('O1', 'H2', 'H3') for l in (1, 2, 3)}
    # The stockholder run declares no distribution diagnostics at all.
    plain = public_models['isa_a'].properties.diagnostics
    assert plain['distribution'] == 'isa_a'
    assert not [k for k in plain if k.startswith('distribution_')]
    assert 'distribution=isa_a; direct_ov' in public_models['isa_a'].properties.model


def test_the_two_distributions_are_different_models_not_two_tolerances(public_models):
    """How far apart the declarations are on one identical wavefunction.

    12% on oxygen and 22% on hydrogen: this is a model choice with visible
    consequences, which is why the two are published from separate declarations
    and never compared for agreement.
    """
    scalars = {}
    for name, result in public_models.items():
        labels = result.properties.require_local().labels
        assert tuple(labels) == ('O1', 'H2', 'H3')
        scalars[name] = np.array(result.atomic_scalars)[0, :, 0]
    relative = np.abs(scalars['df_centre_analytic']-scalars['isa_a'])/np.abs(scalars['isa_a'])
    assert relative[0] > .1 and relative[1] > .2 and relative[2] > .2
    # Both are sound models of their own: positive, and the two hydrogens agree.
    for values in scalars.values():
        assert (values > 0.).all()
        assert abs(values[1]-values[2]) < 1.e-8*values[1]
