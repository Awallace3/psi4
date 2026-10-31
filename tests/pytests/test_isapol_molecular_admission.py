"""Admission/planner regressions only; no molecular quantum calculation."""
import pytest


def test_small_cloud_large_fit_charges_cubic_solver_work_before_admission():
    from psi4.driver.procrouting import isapol_bounded as driver
    from psi4.driver.procrouting.isapol_bounded_response import BoundedResources, _Ledger
    # One point does not remove the per-node penalty eigendecomposition,
    # augmented-system rank diagnostics or DSYSV factorization.
    planned_bytes, planned_work = driver._pfit_plan(1, 153, 612, 11)
    factorization_floor = 840_476_736  # 11*(612**3)/3, DSYSV alone.
    assert planned_work > factorization_floor
    resources = BoundedResources(
        1024**3, factorization_floor, 1024**3, capacity='large')
    with pytest.raises(ValueError, match='work'):
        _Ledger(resources).admit('complete-cloud refinement', planned_bytes, planned_work)


def test_fit_solver_work_uses_the_shared_cumulative_budget():
    from psi4.driver.procrouting import isapol_bounded as driver
    from psi4.driver.procrouting.isapol_bounded_response import BoundedResources, _Ledger
    numeric, work = driver._pfit_plan(1, 153, 612, 11)
    exact = _Ledger(BoundedResources(numeric, work + 7, 1024**3, capacity='large'))
    exact.admit('earlier stage', 0, 7)
    exact.admit('complete-cloud refinement', numeric, work)
    assert exact.work == work + 7
    short = _Ledger(BoundedResources(numeric, work + 6, 1024**3, capacity='large'))
    short.admit('earlier stage', 0, 7)
    with pytest.raises(ValueError, match='cumulative work'):
        short.admit('complete-cloud refinement', numeric, work)
    assert short.work == 7


def test_five_independent_rank2_sites_need_explicit_replay_work():
    from psi4.driver.procrouting.isapol_pfit_stream import packed_design_plan
    dimensions = dict(npoint=2000, channel_count=45, parameter_count=180,
                      used_channels=40, block_rows=4096)
    with pytest.raises(ValueError, match='replay work'):
        packed_design_plan(**dimensions)
    plan = packed_design_plan(**dimensions, max_work=64_228_656_000)
    # Independently worked full triangular cloud:2,001,000rows, two passes.
    assert plan['row_count'] == 2_001_000
    assert plan['planned_work'] == 64_228_656_000
    assert plan['planned_bytes'] == 325_365_648
    assert plan['chunk'] == 32
    with pytest.raises(ValueError, match='replay work'):
        packed_design_plan(**dimensions, max_work=64_228_655_999)


def test_molecular_parameter_allowance_is_explicit_and_finite():
    from psi4.driver.procrouting.isapol_bounded import check_max_parameters
    with pytest.raises(ValueError, match='1..72'):
        check_max_parameters(612)
    assert check_max_parameters(612, capacity='molecular') == 612
    assert check_max_parameters(1024, capacity='molecular') == 1024
    with pytest.raises(ValueError, match='1..1024'):
        check_max_parameters(1025, capacity='molecular')
    with pytest.raises(ValueError, match='capacity'):
        check_max_parameters(64, capacity='unlimited')
    for value in (True, 612., '612'):
        with pytest.raises(ValueError):
            check_max_parameters(value, capacity='molecular')


def test_molecular_route_refuses_missing_large_resource_opt_in_before_state():
    from psi4.driver.procrouting.isapol_bounded import bounded_properties, BoundedResources
    kwargs = dict(caller_converged=True, sites=(), bonds=(), quadrature=None,
        response_grid=None, smoothing=None, shell_cutoff=1e-8,
        charge_penalty=1000., anchor_metric_damping=.0005, lattice_options=None,
        localization_rank_limit=2, weight_type=3, weight_coefficient=.001,
        cutoff=1e-4, resources=BoundedResources(1024**3, 6_000_000_000_000, 1024**3),
        scf_correction='NONE', max_parameters=612, refinement_capacity='molecular',
        max_replay_work=1_000_000_000_000)
    with pytest.raises(ValueError, match='large resource'):
        bounded_properties(None, None, **kwargs)


def test_retained_frequency_plan_scales_with_the_declared_parameter_limit():
    from psi4.driver.procrouting.isapol_bounded_response import retained_frequency_bytes
    assert retained_frequency_bytes(3, 11) == 2_101_792
    assert retained_frequency_bytes(17, 11, parameters=612) == 135_579_488
    for value in (True, 0, 4097, 612.):
        with pytest.raises(ValueError, match='parameters'):
            retained_frequency_bytes(17, 11, parameters=value)


def test_full17_atom_inventory_is_planned_without_allocating_design_buffers(monkeypatch):
    import numpy as np
    from psi4.driver.procrouting.isapol_pfit_stream import packed_design_plan
    def no_allocation(*args, **kwargs):
        raise AssertionError('dimension planner must not allocate an array')
    monkeypatch.setattr(np, 'empty', no_allocation)
    dimensions = dict(npoint=2000, channel_count=153, parameter_count=612,
                      used_channels=136, block_rows=4096, max_work=1_000_000_000_000)
    plan = packed_design_plan(**dimensions, max_bytes=4*1024**3)
    assert plan['planned_bytes'] == 2_044_348_752
    assert plan['planned_work'] == 718_632_624_000
    with pytest.raises(ValueError, match='byte resource'):
        packed_design_plan(**dimensions, max_bytes=2_044_348_751)


@pytest.mark.parametrize('budget', [True, 1., float('inf'), 0, 2_000_000_000_001])
def test_replay_budget_is_a_finite_explicit_integer(budget):
    from psi4.driver.procrouting.isapol_pfit_stream import packed_design_plan
    with pytest.raises(ValueError, match='max_work'):
        packed_design_plan(npoint=2, channel_count=4, parameter_count=1, used_channels=1,
                           max_work=budget)


def test_explicit_oeprop_declaration_admits_new_limits_without_a_molecular_preset():
    from psi4 import core
    from psi4.driver.procrouting import isapol_bounded_oeprop as api
    core.clean_options()
    declared_sites, declared_bonds = object(), object()
    values = api.settings('explicit', dict(
        sites=declared_sites, bonds=declared_bonds, refinement_capacity='molecular',
        max_refinement_parameters=612, max_replay_work=1_000_000_000_000))
    assert values['sites'] is declared_sites and values['bonds'] is declared_bonds
    assert values['max_refinement_parameters'] == 612
    assert values['refinement_capacity'] == 'molecular'
    assert values['max_replay_work'] == 1_000_000_000_000
    assert api.settings('water', {})['max_refinement_parameters'] == 64


def test_explicit_route_passes_inequivalent_sites_without_preset_topology(monkeypatch):
    import numpy as np
    import psi4
    from psi4 import core
    from psi4.driver.procrouting import isapol_bounded_oeprop as api
    from psi4.driver.procrouting.isapol_refine import RefinementSite
    from psi4.driver.procrouting.isapol_logging import StageLog
    core.clean_options()
    molecule = psi4.geometry(
        '0 1\nC 0 0 0\nH -1 0 0\nH 0 -1 0\nH 0 0 -1\nO 1 0 0\nH 2 0 0\n'
        'symmetry c1\nno_com\nno_reorient\nunits bohr')
    # Descriptor only: no SCF/ISA/response calculation is performed locally.
    wfn = core.Wavefunction.build(molecule, 'sto-3g')
    xyz = molecule.geometry().np
    labels = ('C1', 'H1', 'H2', 'H3', 'O1', 'H4')
    sites = tuple(RefinementSite(label, label, origin, np.eye(3), 2)
                  for label, origin in zip(labels, xyz))
    bonds = ((0, 1), (0, 2), (0, 3), (0, 4), (4, 5))
    captured = {}
    marker, auxiliary = object(), object()
    def backend(wavefunction, recipe, **kwargs):
        assert wavefunction is wfn and recipe is auxiliary
        captured.update(kwargs)
        return marker
    def no_preset(*args, **kwargs):
        raise AssertionError('generic route must not invoke water/benzene topology')
    monkeypatch.setattr(api, 'bounded_properties', backend)
    monkeypatch.setattr(api, 'preset_sites', no_preset)
    psi4.oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES',
        atomic_backend='BOUNDED_DF', preset='explicit', sites=sites, bonds=bonds,
        auxiliary_recipe=auxiliary, response_grid=np.zeros((1, 4)),
        lattice_options=core.FitPointsOptions(), quadrature=object(), scf_correction='NONE',
        resources=api.BoundedResources(1024**3, 1_000_000_000_000, 1024**3, capacity='large'),
        refinement_capacity='molecular', max_refinement_parameters=216,
        max_replay_work=1_000_000_000_000, log=StageLog(0))
    assert captured['sites'] == sites and captured['bonds'] == bonds
    assert len({s.site_type for s in captured['sites']}) == 6
    assert captured['max_parameters'] == 216 and captured['refinement_capacity'] == 'molecular'
    assert captured['max_replay_work'] == 1_000_000_000_000
    assert psi4.atomic_property_result(wfn) is marker


@pytest.mark.parametrize('overrides', [{}, {'sites': ()}, {'bonds': ()}])
def test_explicit_route_never_infers_missing_sites_or_bonds(overrides):
    from psi4.driver.procrouting import isapol_bounded_oeprop as api
    with pytest.raises(ValueError, match='caller-declared sites and bonds'):
        api.settings('explicit', overrides)


def test_campaign_chno_recipe_is_explicit_uniform_and_does_not_leak_puream():
    import psi4
    from psi4 import core
    from psi4.driver.procrouting.isapol_recipes import native_monomer_recipe
    core.clean_options()
    molecule = psi4.geometry(
        '0 1\nC 0 0 0\nN 2.7 0 0\nH -1 1 0\nH -1 -1 0\nH 0 0 -1\n'
        'H 3.5 1 0\nH 3.5 -1 0\nsymmetry c1\nno_com\nno_reorient\nunits bohr')
    core.set_global_option('PUREAM', True)
    try:
        recipe = native_monomer_recipe(molecule, profile='s66-chno-isa-a-v1')
        assert core.get_global_option('PUREAM') == 1
        assert core.has_global_option_changed('PUREAM')
        assert recipe.auxiliary.representation == 'Cartesian'
        assert [s.label for s in recipe.sites] == ['C1', 'N1', 'H1', 'H2', 'H3', 'H4', 'H5']
        assert recipe.sites[0].tail_cutoff == 1.5*.70/.529177249
        assert recipe.sites[1].tail_cutoff == 1.5*.65/.529177249
        assert recipe.sites[2].tail_cutoff == 1.5*.50/.529177249
        assert [s.exponents for s in recipe.sites[1].shape.shells] == [
            (256.,), (128.,), (64.,), (32.,), (16.,), (8.,),
            (4.,), (2.,), (1.,), (.5,), (.25,), (.125,)]
        assert recipe.atomic_initialization == 'zero_atomic_D0'
    finally:
        core.clean_options()
    with pytest.raises(ValueError, match='profile'):
        native_monomer_recipe(molecule, profile='inferred')
