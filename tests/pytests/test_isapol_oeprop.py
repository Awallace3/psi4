"""Native API validation and direct-OV independent sampled-polynomial checks."""
import numpy as np
import pytest
import psi4
from psi4.driver.procrouting import isapol_oeprop as api
from test_isapol_partitioned_response import basis, site, poly2


def test_direct_ov_polynomial_order_ownership_and_representation():
    b = basis(((0.,0.,0.), (.8,.1,-.3)), psi4.core.IsaBasisRole.Orbital)
    s = site(neighbours=(0,1))
    c = np.array([[.8,.2,.4],[-.1,.7,-.2]])
    out = psi4.core.IsaPartitionedMultipoles(b,[s],'synthetic direct OV',1e-36,
                                            psi4.core.Matrix.from_array(c),1)
    pts = np.array(s.samples.points)
    centers = np.array([[0.,0.,0.],[.8,.1,-.3]])
    mo = (1.2*np.exp(-.7*np.sum((pts[:,None,:]-centers)**2,axis=2))) @ c
    expected = poly2(pts) @ (mo[:,:1]*mo[:,1:])
    np.testing.assert_allclose(out.values.np, expected, atol=2e-15, rtol=2e-15)
    assert out.representation == 'direct_ov'
    val = out.values; val.np[:] = 0
    np.testing.assert_allclose(out.values.np, expected, atol=2e-15)
    with pytest.raises(ValueError, match='representation'):
        psi4.core.IsaDistributedResponse(out,[0.],[psi4.core.Matrix.from_array(-np.eye(2))],
                                         'fitted_density_coefficients','wrong coordinates')
    r = psi4.core.IsaDistributedResponse(out,[0.],[psi4.core.Matrix.from_array(-np.eye(2))],
                                         'direct_ov','direct response')
    np.testing.assert_allclose(r.at_index(0).np,expected@expected.T,atol=2e-15)


@pytest.mark.parametrize('nocc',[0,3,-1])
def test_direct_ov_bad_occupation(nocc):
    with pytest.raises(ValueError):
        psi4.core.IsaPartitionedMultipoles(basis(role=psi4.core.IsaBasisRole.Orbital),
            [site()], 'bad occupancy',1e-36,psi4.core.Matrix.from_array(np.ones((1,3))),nocc)


@pytest.mark.parametrize('tasks',[(),('NOT_A_TASK',),('ATOMIC_PARTITION','ATOMIC_PARTITION')])
def test_request_validation(tasks):
    with pytest.raises(ValueError, match='request'):
        api.validate_request(None,tasks)


@pytest.mark.parametrize('option,value,message',[
    ('PARTITION_SCHEME','MBIS','MBIS'),('ATOMIC_RESPONSE_LOCALIZATION','LS','LW')])
def test_unsupported_policy(option,value,message):
    old = psi4.core.get_global_option(option)
    try:
        psi4.set_options({option:value})
        with pytest.raises(ValueError,match=message):
            api.validate_request(None,('ATOMIC_PARTITION',))
    finally:
        psi4.core.set_global_option(option,old)


def test_missing_result_and_unconverged_wavefunction():
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    with pytest.raises(ValueError):
        psi4.atomic_property_result(w)
    with pytest.raises(ValueError):
        psi4.oeprop(w,'ATOMIC_PARTITION')


def test_generated_recipe_is_self_contained():
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    r = api.generated_recipe(w)
    assert 'NOT modern CamCASP' in r.origin
    assert r.auxiliary.representation == 'Cartesian'
    assert [len(s.shape.shells) for s in r.sites] == [17,11,11]
    assert min(s.exponents[0] for s in r.sites[0].shape.shells) == .1
    assert r.controller.convergence == 1e-9


def test_declared_molecular_aux_is_an_argument_never_inferred_from_main():
    """Naming the AUX names a partition: it also carries the Drho-C/ISA-A fit.

    So it is a declared argument with an unchanged default, not something read
    off BASIS or DF_BASIS_SCF.  Only the auxiliary moves; the site radial
    recipe, grid and controller are untouched.
    """
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    default, matched = api.generated_recipe(w), api.generated_recipe(w, aux_basis='aug-cc-pVTZ-JKFIT')
    assert default.auxiliary.name == 'cc-pVDZ-JKFIT Cartesian molecular AUX'
    assert matched.auxiliary.name == 'aug-cc-pVTZ-JKFIT Cartesian molecular AUX'
    assert 'aug-cc-pVTZ-JKFIT' in matched.origin and 'NOT modern CamCASP' in matched.origin
    assert len(matched.auxiliary.shells) > len(default.auxiliary.shells)
    assert default.name == matched.name and default.grid == matched.grid
    assert default.controller == matched.controller
    assert [[sh.exponents for sh in s.shape.shells] for s in default.sites] == \
           [[sh.exponents for sh in s.shape.shells] for s in matched.sites]


def test_undeclared_molecular_aux_is_rejected_before_any_compute():
    old = psi4.core.get_global_option('ATOMIC_PROPERTY_AUXILIARY_BASIS')
    assert old == 'cc-pVDZ-JKFIT'  # registered default, case preserved
    try:
        psi4.core.set_global_option('ATOMIC_PROPERTY_AUXILIARY_BASIS','   ')
        with pytest.raises(ValueError, match='AUXILIARY_BASIS'):
            api.validate_request(None,('ATOMIC_PARTITION',))
    finally:
        psi4.core.set_global_option('ATOMIC_PROPERTY_AUXILIARY_BASIS',old)


@pytest.mark.parametrize('tasks', [
    ('ATOMIC_POLARIZABILITY',),
    ('ATOMIC_PARTITION', 'ATOMIC_POLARIZABILITY'),
    ('DIPOLE', 'ATOMIC_POLARIZABILITY'),
    ('ATOMIC_PARTITION', 'ATOMIC_PARTITION'),
    ('ATOMIC_PARTITION', 42),
])
def test_public_rejection_invalidates_result_before_any_compute(tasks, monkeypatch):
    w = psi4.core.Wavefunction.build(psi4.geometry('H\nH 1 0.74\nsymmetry c1'), 'sto-3g')
    old = api.AtomicPropertyResult(('ATOMIC_PARTITION',), (), object(), None, 0.)
    w._native_atomic_property_result = old
    monkeypatch.setattr(psi4.core, 'OEProp', lambda *a: pytest.fail('ordinary compute entered'))
    monkeypatch.setattr(api, 'generated_recipe', lambda *a: pytest.fail('native compute entered'))
    # Psi4 ValidationError is not necessarily a ValueError.
    from psi4.driver.p4util.exceptions import ValidationError
    with pytest.raises((ValueError, ValidationError)):
        psi4.oeprop(w, *tasks)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(w)
    assert old.tasks == ('ATOMIC_PARTITION',) and old.partition is not None


@pytest.mark.parametrize('entry', ['public', 'direct'])
def test_policy_rejection_invalidates_previous_result(entry):
    from psi4.driver.p4util import OptionsStateCM
    w = psi4.core.Wavefunction.build(psi4.geometry('H\nH 1 0.74\nsymmetry c1'), 'sto-3g')
    old = api.AtomicPropertyResult(('ATOMIC_PARTITION',), (), object(), None, 0.)
    w._native_atomic_property_result = old
    with OptionsStateCM(['PARTITION_SCHEME']):
        psi4.core.set_global_option('PARTITION_SCHEME', 'MBIS')
        with pytest.raises(ValueError, match='MBIS'):
            if entry == 'public':
                psi4.oeprop(w, 'ATOMIC_PARTITION')
            else:
                api.run(w, ('ATOMIC_PARTITION',))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(w)
    assert old.tasks == ('ATOMIC_PARTITION',)


def test_ordinary_property_behavior_unchanged(monkeypatch):
    calls = []
    class Ordinary:
        def __init__(self, wfn):
            calls.append('init')
        def add(self, prop):
            calls.append(prop)
        def compute(self):
            calls.append('compute')
    monkeypatch.setattr(psi4.core, 'OEProp', Ordinary)
    class Wfn:
        pass
    w = Wfn()
    old = object()
    w._native_atomic_property_result = old
    assert psi4.oeprop(w, 'dipole') is None
    assert calls == ['init', 'DIPOLE', 'compute']
    assert w._native_atomic_property_result is old


def test_static_request_no_quadrature_or_pfit(monkeypatch):
    # Real request dispatch; forbid expensive prerequisites not needed for static.
    monkeypatch.setattr(api,'validate_request',lambda *a: 0.)
    monkeypatch.setattr(api.n.Quadrature,'from_casimir',lambda *a: pytest.fail('static quadrature'))
    class Stop(Exception): pass
    def capture(*a,**kw):
        assert kw['frequencies'] == (0.,) and kw['quadrature'] is None and not kw['pair_self']
        assert kw['response_basis'] == 'direct_ov'
        assert a[1].grid.radial_points == 160
        # Response has its own quadrature, rather than reusing the denser ISA grid.
        # Euler–MacLaurin n_r yields n_r-1 finite radial shells.
        assert kw['response_grid'].shape == (3 * (99 - 1) * 590, 4)
        raise Stop
    monkeypatch.setattr(api.n,'native_properties',capture)
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    with pytest.raises(Stop):
        api.run(w,('ATOMIC_POLARIZABILITIES',))


@pytest.mark.parametrize('option,expected',[(None,'ordered_pairwise'),
                                            ('ORDERED_PAIRWISE','ordered_pairwise'),
                                            ('SHARED_SWEEP','shared_sweep')])
def test_named_response_algorithm_reaches_preflight_and_factory(option,expected,monkeypatch):
    # The public option only names the arrangement; it is recorded in the
    # request options and forwarded to both gates, never inferred.
    monkeypatch.setattr(api,'validate_request',lambda *a: 0.)
    seen = []
    original = api.estimate_response_work
    def observe(*a,**kw):
        seen.append(kw['algorithm'])
        return original(*a,**kw)
    monkeypatch.setattr(api,'estimate_response_work',observe)
    class Stop(Exception): pass
    def capture(*a,**kw):
        assert kw['response_algorithm'] == expected
        raise Stop
    monkeypatch.setattr(api.n,'native_properties',capture)
    old = psi4.core.get_global_option('ATOMIC_RESPONSE_ALGORITHM')
    try:
        if option is not None:
            psi4.set_options({'ATOMIC_RESPONSE_ALGORITHM':option})
        w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
        with pytest.raises(Stop):
            api.run(w,('ATOMIC_POLARIZABILITIES',))
    finally:
        psi4.core.set_global_option('ATOMIC_RESPONSE_ALGORITHM',old)
    assert seen == [expected]
    with pytest.raises(RuntimeError,match='not a valid choice'):
        psi4.core.set_global_option('ATOMIC_RESPONSE_ALGORITHM','BLAS3')
    assert psi4.core.get_global_option('ATOMIC_RESPONSE_ALGORITHM') == old


def test_declared_tail_policy_is_a_model_parameter_never_a_tolerance():
    """``W-TAILS R1-Multiplier = 1.5`` is 1.5*R_Slater, not a flat 1.5 bohr.

    The two cutoffs are different declared models, so each gets its own recipe
    name and the demo default is left exactly where it was -- byte-identical
    origin, flat cutoff on every site -- rather than being silently retuned.
    """
    from psi4.driver.procrouting import isapol_native_partition as native
    w = psi4.core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'),'cc-pvdz')
    default, slater = api.generated_recipe(w), api.generated_recipe(w, tail_policy='bragg_slater_1.5')
    assert default.name == 'GENERATED_JKFIT_ISA_A'
    assert [s.tail_cutoff for s in default.sites] == [1.5, 1.5, 1.5]
    assert slater.name == 'GENERATED_JKFIT_BRAGG_SLATER_TAIL_ISA_A'
    assert slater.sites[0].tail_cutoff == native.bragg_slater_tail_cutoff(8, 1.5)
    assert [s.tail_cutoff for s in slater.sites[1:]] == [native.bragg_slater_tail_cutoff(1, 1.5)]*2
    assert slater.sites[0].tail_cutoff != slater.sites[1].tail_cutoff
    # Only the cutoff moves; the declaration is visible in the origin, and the
    # default recipe's provenance text is untouched.
    assert 'bragg_slater_1.5 W-TAILS cutoff' in slater.origin
    assert 'W-TAILS' not in default.origin
    assert default.auxiliary.shells == slater.auxiliary.shells
    assert default.auxiliary.centres == slater.auxiliary.centres and default.grid == slater.grid
    assert default.controller == slater.controller and default.track == slater.track
    # The three activation thresholds are declared model parameters carrying
    # CamCASP's declared module defaults (stockholder.F90): wEps_EpsNorm and
    # PositiveW_EpsNorm are 1e-5, TailFix_EpsNorm is 1e-6.  They are pinned here
    # because a run at another value is a different model, not a looser one.
    assert default.controller.w_eps_activation == 1e-5
    assert default.controller.positive_activation == 1e-5
    assert default.controller.tail_activation == 1e-6
    assert default.controller.tail_iteration_limit == 20
    assert [[sh.exponents for sh in s.shape.shells] for s in default.sites] == \
           [[sh.exponents for sh in s.shape.shells] for s in slater.sites]
    with pytest.raises(ValueError, match='Unknown declared tail policy'):
        api.generated_recipe(w, tail_policy='1.5')


def test_refinement_option_defaults_are_the_declared_reference_model():
    """The nine refinement options name one model, so their defaults are pinned.

    ``LOWER_LIMIT``/``UPPER_LIMIT`` are multiples of the van der Waals radius,
    not bohr, and the point count is the protocol's ``Random 500`` rather than
    the C++ ``FitPointsOptions`` production default of 2000, which the
    point-response cap refuses.
    """
    assert api._refinement_options() == {
        'npoints': 500, 'seed': 1, 'lower_limit': 2.0, 'upper_limit': 4.0,
        'weight_type': 4, 'weight_coefficient': 1.e-3, 'cutoff': 1.e-4,
        'rank_limit': 2, 'hydrogen_rank_limit': 1}
    assert api.REFINEMENT_TASKS < api.TASKS
    assert set(api.REFINEMENT_KEYS) == {'ATOMIC_REFINEMENT_' + n for n in
        ('POINTS','SEED','LOWER_LIMIT','UPPER_LIMIT','WEIGHT_TYPE','WEIGHT_COEFFICIENT',
         'CUTOFF','RANK_LIMIT','HYDROGEN_RANK_LIMIT')}


@pytest.mark.parametrize('option,value',[
    ('ATOMIC_REFINEMENT_POINTS',0),('ATOMIC_REFINEMENT_POINTS',513),
    ('ATOMIC_REFINEMENT_SEED',0),
    ('ATOMIC_REFINEMENT_WEIGHT_TYPE',7),('ATOMIC_REFINEMENT_WEIGHT_TYPE',-1),
    ('ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT',0.),
    ('ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT',-1.e-3),
    ('ATOMIC_REFINEMENT_CUTOFF',0.),
    ('ATOMIC_REFINEMENT_LOWER_LIMIT',0.),('ATOMIC_REFINEMENT_LOWER_LIMIT',4.),
    ('ATOMIC_REFINEMENT_UPPER_LIMIT',1.),
    ('ATOMIC_REFINEMENT_RANK_LIMIT',0),('ATOMIC_REFINEMENT_RANK_LIMIT',5),
    ('ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT',0),
])
def test_bad_refinement_option_is_refused_not_clamped(option,value):
    """A mis-declared model is rejected; nothing here is a tolerance to relax."""
    old = psi4.core.get_global_option(option)
    try:
        psi4.core.set_global_option(option,value)
        with pytest.raises(ValueError,match=option.split('ATOMIC_REFINEMENT_')[1].split('_')[0]):
            api._refinement_options()
        with pytest.raises(ValueError):
            api.validate_request(None,('ATOMIC_REFINED_DISPERSION',))
    finally:
        psi4.core.set_global_option(option,old)


def test_refinement_rank_limit_cannot_exceed_the_localization():
    """A rank never localized cannot be refined, whichever type asks for it."""
    for option in ('ATOMIC_REFINEMENT_RANK_LIMIT','ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT'):
        old = psi4.core.get_global_option(option)
        loc = psi4.core.get_global_option('ATOMIC_LOCALIZATION_RANK_LIMIT')
        try:
            psi4.core.set_global_option('ATOMIC_LOCALIZATION_RANK_LIMIT',1)
            psi4.core.set_global_option(option,2)
            with pytest.raises(ValueError,match='ATOMIC_LOCALIZATION_RANK_LIMIT'):
                api._refinement_options()
        finally:
            psi4.core.set_global_option(option,old)
            psi4.core.set_global_option('ATOMIC_LOCALIZATION_RANK_LIMIT',loc)


def test_refinement_options_are_read_only_for_a_refinement_request(monkeypatch):
    """An unrefined request neither validates nor records them: they did not apply."""
    class Sentinel(Exception): pass
    monkeypatch.setattr(api,'_refinement_options',lambda: (_ for _ in ()).throw(Sentinel()))
    for tasks in (('ATOMIC_PARTITION',),('ATOMIC_POLARIZABILITIES',),('ATOMIC_DISPERSION',)):
        # Reaches the wavefunction requirement, which is downstream of where the
        # refinement options would have been read.
        with pytest.raises(ValueError,match='restricted C1'):
            api.validate_request(None,tasks)
    for tasks in (('ATOMIC_REFINED_POLARIZABILITIES',),('ATOMIC_REFINED_DISPERSION',),
                  ('ATOMIC_DISPERSION','ATOMIC_REFINED_DISPERSION')):
        with pytest.raises(Sentinel):
            api.validate_request(None,tasks)
