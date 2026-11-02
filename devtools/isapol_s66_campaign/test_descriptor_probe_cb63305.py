"""Descriptor-only extraction/forwarding with mocked state seams; no Psi4."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load('descriptor_probe_cb63305', 'descriptor_probe_cb63305.py')
fixtures = load('descriptor_extraction_fixtures', 'test_descriptor_probe.py')


def mock_boundaries(monkeypatch, *, producer='cb63305'):
    molecule, main, recipe, declaration = fixtures.fixtures()
    declaration.profile = 's66-chno-isa-a-v1'
    main.shell = lambda i: NS(am=1, nprimitive=1, exp=lambda j: 1., coef=lambda j: 1.)
    main.shell_to_center = lambda i: 0
    budget = NS(max_bytes=12*1024**3, max_work=6*10**12, max_io_bytes=16*1024**3, capacity='large')
    request = NS(declaration=declaration, resources=budget, max_replay_work=10**12,
                 pfit_maximum_work_bytes=256*1024**2, as_dict=lambda: {})
    calls = {}
    for name in ('psi4', 'psi4.driver', 'psi4.driver.procrouting'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    psi4 = sys.modules['psi4']
    psi4.__file__, psi4.__version__ = '/synthetic/psi4/__init__.py', 'synthetic-only'
    psi4.core = NS(__file__='/synthetic/psi4/core.so',
        Molecule=NS(from_schema=lambda _: molecule), BasisSet=NS(build=lambda *a, **k: main))
    psi4.set_memory = lambda value, **kwargs: calls.update(memory=value)
    psi4.set_options = lambda value: calls.update(options=value)
    definitions = {
        'isapol_schema': dict(admit_native_schema=lambda *a: (request, object()),
            native_molecule_schema=lambda value, req: value,
            validate_native_request=lambda *a, **k: (request, {'sha256': {'synthetic': 'not qualified'}}),
            check_native_geometry=lambda *a: 0.),
        'isapol_recipes': dict(native_monomer_recipe=lambda *a, **k: recipe),
        'isapol_distribution': dict(basis_identity=lambda value: 'f'*64),
        'isapol_native_partition': dict(DRHO_REFINEMENT_ITERATIONS=10 if producer == 'cb63305' else 32),
    }
    for name, attrs in definitions.items():
        module = ModuleType('psi4.driver.procrouting.'+name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    planner, descriptors = probe.helper('resource_plan_'+producer), probe.helper('descriptor_probe')
    original = planner.plan
    def plan(d, resources, **kwargs):
        calls.update(plan_kwargs=kwargs, resources=resources, dimensions=d)
        return original(d, resources, **kwargs)
    planner.plan = plan
    monkeypatch.setattr(probe, 'helper', lambda name: planner
                        if name == 'resource_plan_'+producer else descriptors)
    runtime = dict(molecule={}, specification=dict(
        driver='properties', model=dict(method='pbe0', basis='aug-cc-pvtz'),
        keywords=dict(reference='rhf', function_kwargs=dict(
            native_monomer={'schema': 'native-monomer-request/3'}, properties=['dipole']))))
    return runtime, calls, request


def test_new_closed_producer_requires_native32_and_reports_charged32(monkeypatch):
    runtime, calls, _ = mock_boundaries(monkeypatch, producer='ed561666')
    report = probe.scientific_probe(runtime, memory_bytes=20*1024**3, producer='ed561666')
    assert report['policy_id'] == 'ed561666-reference-h2h1-ledger-v3'
    assert report['plan']['dimensions']['drho_refinement_iterations'] == 32
    monkeypatch.setattr(sys.modules['psi4.driver.procrouting.isapol_native_partition'],
                        'DRHO_REFINEMENT_ITERATIONS', 10)
    with pytest.raises(ValueError, match='refinement policy changed'):
        probe.scientific_probe(runtime, memory_bytes=20*1024**3, producer='ed561666')


def test_unknown_producer_is_refused_before_importing_quantum_code():
    with pytest.raises(ValueError, match='closed installed descriptor'):
        probe.scientific_probe({}, memory_bytes=1, producer='../caller_module')


def test_descriptor_function_forwards_exact_request3_caps_without_inference(monkeypatch):
    runtime, calls, request = mock_boundaries(monkeypatch)
    report = probe.scientific_probe(runtime, memory_bytes=20*1024**3)
    assert calls['plan_kwargs'] == dict(max_replay_work=10**12, pfit_maximum_work_bytes=256*1024**2)
    assert calls['memory'] == 20*1024**3
    assert calls['dimensions'].main_nbf == 12  # toy shells, NOT formula-predicted92
    assert report['schema'] == 'native-campaign-descriptor-probe/2'
    assert report['pfit_maximum_work_bytes'] == request.pfit_maximum_work_bytes
    assert report['plan']['dimensions']['parameters'] == 108
    assert 'function_kwargs' not in calls['options']
    assert 'no SCF' in report['qualification']


def test_legacy_request_is_not_reinterpreted_as_expanded(monkeypatch):
    runtime, calls, _ = mock_boundaries(monkeypatch)
    runtime['specification']['keywords']['function_kwargs']['native_monomer']['schema'] = 'native-monomer-request/2'
    with pytest.raises(ValueError, match='explicit request3'):
        probe.scientific_probe(runtime, memory_bytes=20*1024**3)
    assert not calls


def test_extended_and_nondefault_caps_are_forwarded_and_reported_exactly(monkeypatch):
    runtime, calls, request = mock_boundaries(monkeypatch)
    request.resources = NS(max_bytes=32*1024**3, max_work=128*10**12,
                           max_io_bytes=16*1024**3, capacity='extended')
    request.pfit_maximum_work_bytes = 2*1024**3
    request.max_replay_work = 1100000000000
    report = probe.scientific_probe(runtime, memory_bytes=40*1024**3)
    assert calls['plan_kwargs'] == dict(max_replay_work=1100000000000,
                                       pfit_maximum_work_bytes=2*1024**3)
    assert calls['resources'] == report['resources'] == vars(request.resources)
    assert report['max_replay_work'] == request.max_replay_work
    assert report['pfit_maximum_work_bytes'] == request.pfit_maximum_work_bytes
    assert report['intended_memory_bytes'] == calls['memory'] == 40*1024**3


@pytest.mark.parametrize('memory', [None, True, 0, -1, 1])
def test_undeclared_or_insufficient_scientific_memory_is_refused(monkeypatch, memory):
    runtime, calls, _ = mock_boundaries(monkeypatch)
    with pytest.raises(ValueError, match='memory must cover'):
        probe.scientific_probe(runtime, memory_bytes=memory)
    assert not calls


def test_help_requires_no_quantum_or_manager_imports():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', probe.__file__, '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0 and '--monomer-id' in result.stdout


def test_child_constructs_no_state_grid_fit_points_or_integrals():
    tree = ast.parse(Path(probe.__file__).read_text())
    calls = [ast.unparse(n.func).split('.')[-1] for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = {'energy', 'properties', 'oeprop', 'Wavefunction', 'NativeRestrictedState',
                 'IsaGrid', 'FitPoints', 'native_partition', 'adapt_main', 'IntegralFactory',
                 'exec_module'}
    assert not forbidden.intersection(calls)
