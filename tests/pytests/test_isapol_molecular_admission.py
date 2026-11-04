"""Admission/planner regressions only; no molecular quantum calculation."""
import pytest


def test_independent_benzene_preflight_refuses_default_native_cap_before_allocating(monkeypatch):
    import numpy as np
    from psi4.driver.procrouting.isapol_bounded_response import BoundedResources
    from psi4.driver.procrouting.isapol_molecular import (
        MolecularDeclaration, PROFILE, plan_molecular_refinement)
    frame = tuple(map(tuple, np.eye(3)))
    declaration = MolecularDeclaration(PROFILE, (frame,)*12, tuple((i, i+1) for i in range(11)))
    resources = BoundedResources(64*1024**3, 100_000_000_000_000, 64*1024**3, capacity='large')
    def forbidden(*args, **kwargs):
        raise AssertionError('admission must refuse before allocating a design or scientific state')
    monkeypatch.setattr(np, 'empty', forbidden)
    monkeypatch.setattr(np, 'zeros', forbidden)
    # The legacy native base is688,747,264 bytes; the independent kernel cap
    # remains256MiB even when the outer Python budget has ample space.
    with pytest.raises(ValueError, match='native.*(budget|cap|workspace)'):
        plan_molecular_refinement(declaration, resources, max_replay_work=1_000_000_000_000)


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


@pytest.mark.parametrize('route', ['source', 'installed'])
def test_molecular_integration_grid_forwarding_is_dense_and_other_policies_unchanged(monkeypatch, route):
    """Actual route body with synthetic dependencies; no grid or quantum work."""
    import ast
    from pathlib import Path
    import sys
    from types import ModuleType, SimpleNamespace as NS
    import numpy as np
    if route == 'source':
        path = Path(__file__).resolve().parents[2]/'psi4/driver/procrouting/isapol_molecular.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'run_molecular_properties')
        namespace = dict(__name__='synthetic_molecular_route',
                         __package__='psi4.driver.procrouting')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        function = namespace['run_molecular_properties']
    else:
        from psi4.driver.procrouting.isapol_molecular import run_molecular_properties
        function = run_molecular_properties
        namespace = function.__globals__
    captured = {}
    marker, auxiliary, quadrature = object(), object(), object()
    class Molecule:
        def clone(self): return self
    molecule = Molecule()
    class Wavefunction:
        def molecule(self): return molecule
        def basisset(self):
            return NS(name=lambda: 'AUG-CC-PVTZ', has_puream=lambda: True)
    wfn = Wavefunction()
    class Grid:
        def __init__(self, mol, options):
            assert mol is molecule
            captured['integration_grid'] = (options.radial_points, options.spherical_points)
        def x(self): return np.array([0., 1.])
        def y(self): return np.array([2., 3.])
        def z(self): return np.array([4., 5.])
        def w(self): return np.array([.25, .75])
    def backend(state, *tasks, **kwargs):
        assert state is wfn
        captured.update(tasks=tasks, settings=kwargs)
    def casimir(*args):
        captured['casimir_arguments'] = args
        return 'SYNTHETIC_CASIMIR_INPUT'
    for name in ('psi4', 'psi4.driver', 'psi4.driver.p4util', 'psi4.driver.procrouting'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    util = ModuleType('psi4.driver.p4util.util')
    util.oeprop, util.atomic_property_result = backend, lambda state: marker
    api = ModuleType('psi4.driver.procrouting.isapol_bounded_oeprop')
    api.invalidate = lambda state: None
    monkeypatch.setitem(sys.modules, util.__name__, util)
    monkeypatch.setitem(sys.modules, api.__name__, api)
    replacements = dict(
        np=np,
        core=NS(Wavefunction=Wavefunction, get_memory=lambda: 40*1024**3,
                IsaGridOptions=NS, IsaGrid=Grid, FitPointsOptions=NS,
                CasimirGrid=casimir),
        plan_molecular_refinement=lambda *args, **kwargs: None,
        require_scf_seal=lambda state: None,
        validate_correction=lambda *args, **kwargs: None,
        independent_sites=lambda *args: ('SYNTHETIC_SITE',)*3,
        native_monomer_recipe=lambda *args, **kwargs: NS(auxiliary=auxiliary),
        Quadrature=NS(from_casimir=lambda value: quadrature),
        KernelSmoothing=lambda *args: ('SYNTHETIC_SMOOTHING', args),
        full_rank2_variables=lambda sites: ('SYNTHETIC_VARIABLE',))
    for name, value in replacements.items():
        monkeypatch.setitem(namespace, name, value)
    declaration = NS(profile='s66-chno-isa-a-v1', bonds=((0, 1), (0, 2)))
    resources = NS(max_bytes=32*1024**3)
    assert function(wfn, declaration, resources, max_replay_work=123,
                    scratch_directory='SYNTHETIC_SCRATCH',
                    pfit_maximum_work_bytes=456) is marker
    assert captured['integration_grid'] == (400, 2030)
    settings = captured['settings']
    assert settings['distribution'] == 'isa' and settings['auxiliary_recipe'] is auxiliary
    assert settings['partition_grid'].shape == (2, 4)
    assert settings['quadrature'] is quadrature
    assert captured['casimir_arguments'] == (10, .5)
    assert settings['radial_points'] == 100 and settings['spherical_points'] == 200
    assert settings['smoothing'] == ('SYNTHETIC_SMOOTHING', (1e-8, 600., .01, 1., 'FD'))
    lattice = settings['lattice_options']
    assert (lattice.npoints, lattice.seed, lattice.lolim, lattice.hilim) == (2000, 1, 2., 4.)
    assert settings['charge_penalty'] == 1000. and settings['anchor_metric_damping'] == .0005
    assert settings['localization_rank_limit'] == settings['rank_limit'] == settings['hydrogen_rank_limit'] == 2
    assert settings['weight_type'] == 3 and settings['weight_coefficient'] == .001
    assert settings['cutoff'] == 1e-4 and settings['shell_cutoff'] == 1e-8
    assert settings['max_refinement_parameters'] == 108 and settings['max_order'] == 10
    assert settings['refinement_capacity'] == 'molecular'
    assert settings['resources'] is resources and settings['max_replay_work'] == 123
    assert settings['pfit_maximum_work_bytes'] == 456
    assert settings['scf_correction'] == 'NONE' and settings['recoupled_dispersion'] is False
    assert captured['tasks'] == ('ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION')
