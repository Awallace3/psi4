"""Finite manager binding/lifecycle with fake activation, no native execution."""
from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_execution_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding(name='s20'):
    cases, inputs, pool = load('test_grid400_bank_case').fixture()
    policy = cases.helper('grid400_bank_policy')
    profile = policy.execution_profile(name)
    row, report = pool['bank']['monomers'][0], pool['allocation']['reports'][0]
    numeric = dict(s20=12, s40=32, s60=48, s80=64, s120=96)[name]
    request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    request['resources']['max_bytes'] = numeric*1024**3
    row['intended_memory_bytes'] = profile['science_memory_GiB']*1024**3
    report['runtime_input_sha256'] = row['runtime_input_sha256'] = cases.helper(
        'prepare_descriptor_bank_cb63305').json_digest(row['runtime_input'])
    case = cases.prepare_case(inputs, pool, 'monomer-0001')
    execution = load('grid400_bank_execution')
    runtime = execution.helper('phoenix_worker_probe').runtime_input(case)
    runtime['specification']['protocols'] = execution.helper(
        'phoenix_pilot').execution_specification({})['protocols']
    kwargs = dict(runtime=runtime, compute_tag='ccasp-native-grid400-bank-monomer-0001-123',
        worker_shape=(profile['cores'], profile['science_memory_GiB']), record_id=9999)
    return execution, case, kwargs


@pytest.mark.parametrize('name', ['s20', 's40', 's60', 's80', 's120'])
def test_all_five_profiles_use_shared_single_claim_and_guarded_activation(monkeypatch, tmp_path, name):
    execution, case, kwargs = binding(name)
    assert execution.validate_binding(case, **kwargs) == tuple(range(5234314, 5234328))
    session = load('single_record_cb63305')
    effects = []

    class Client:
        pass

    @contextmanager
    def factory():
        effects.append('factory-enter')
        try:
            yield Client
        finally:
            effects.append('factory-exit')

    class Manager:
        def __init__(self, config):
            assert effects == ['factory-enter']
            effects.append('activation')
            self.manager_config = config
            self.name, self.client = 'SYNTHETIC_BANK_MANAGER', Client()
            self.dflow_kernel = None
            self.statistics = NS(dict=lambda: {})

        def start(self):
            assert effects[-1] == 'factory-exit'
            effects.append('start')
            self.target_task_id, self.delivery_outcome = 77, 'accepted'

    package = ModuleType('qcfractalcompute')
    package.__path__ = []
    compute = ModuleType('qcfractalcompute.compute_manager')
    compute.ComputeManager = Manager
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, compute.__name__, compute)
    original = session.helper
    monkeypatch.setattr(session, 'helper', lambda helper:
        NS(grid400_manager_client_factory=factory) if helper == 'dataset_fresh_cb63305' else original(helper))
    cores, memory = kwargs['worker_shape']
    config = NS(executors=dict(pilot=NS(type='local', max_workers=1,
        cores_per_worker=cores, memory_per_worker=float(memory), compute_tags=[kwargs['compute_tag']])))
    result = session.run_session(config, **kwargs, case=case, campaign_scope='eligible-bank',
        producer='grid400', payload=tmp_path, save=lambda *args: None)
    assert result['task_id'] == 77
    assert effects == ['factory-enter', 'activation', 'factory-exit', 'start']


@pytest.mark.parametrize('change', ['runtime', 'protocols', 'shape', 'tag', 'profile',
    'unbound', 'target', 'queue', 'hash', 'old_record', 'forensic_record', 'old_append'])
def test_unbound_or_changed_case_cannot_enter_manager(change):
    execution, case, kwargs = binding()
    if change == 'runtime':
        kwargs['runtime']['molecule']['geometry'][0] = 1.
    elif change == 'protocols':
        kwargs['runtime']['specification']['protocols']['error_correction']['default_policy'] = True
    elif change == 'shape':
        kwargs['worker_shape'] = (24, 120)
    elif change == 'tag':
        kwargs['compute_tag'] = 'ccasp-native-grid400-monomer-0008-123'
    elif change == 'profile':
        case['execution_profile'] = execution.helper('grid400_bank_policy').execution_profile('s60')
    elif change == 'unbound':
        case['dataset_mode'] = 'awaiting-audited-bootstrap'
    elif change == 'target':
        case['dataset_id'] = 999
    elif change == 'queue':
        case['queue_policy']['find_existing'] = True
    elif change == 'hash':
        case['runtime_input_sha256'] = 'f'*64
    elif change in ('old_record', 'forensic_record'):
        kwargs['record_id'] = 5234318 if change == 'old_record' else 5234327
    else:
        case.update(dataset_mode='append', dataset_id=356,
                    bootstrap_context_sha256='c'*64, checkpoint_sha256='d'*64)
    with pytest.raises(ValueError):
        execution.validate_binding(case, **kwargs)


def test_new_120_shape_is_not_silently_available_to_historical_sessions():
    session = load('single_record_cb63305')
    config = NS(executors=dict(pilot=NS(type='local', max_workers=1,
        cores_per_worker=24, memory_per_worker=120., compute_tags=['unit'])))
    with pytest.raises(ValueError):
        session.validate_session_config(config, 'unit', worker_shape=(24, 120.))
    session.validate_session_config(config, 'unit', worker_shape=(24, 120.),
                                    campaign_scope='eligible-bank')


@pytest.mark.parametrize('change', [None, 'missing_bootstrap', 'malformed_bootstrap',
                                  'missing_checkpoint', 'malformed_checkpoint'])
def test_append_binding_requires_both_exact_checkpoint_commitments(change):
    cases, inputs, pool = load('test_grid400_bank_case').fixture()
    case = cases.prepare_case(inputs, pool, 'monomer-0002')
    case.update(dataset_mode='append', dataset_id=999,
                bootstrap_context_sha256='c'*64, checkpoint_sha256='d'*64)
    execution = load('grid400_bank_execution')
    runtime = execution.helper('phoenix_worker_probe').runtime_input(case)
    runtime['specification']['protocols'] = execution.helper(
        'phoenix_pilot').execution_specification({})['protocols']
    if change:
        key = 'bootstrap_context_sha256' if 'bootstrap' in change else 'checkpoint_sha256'
        if change.startswith('missing'):
            case.pop(key)
        else:
            case[key] = 'not-a-hash'
    kwargs = dict(runtime=runtime, compute_tag='ccasp-native-grid400-bank-monomer-0002-123',
                  worker_shape=(8, 20), record_id=9999)
    if change:
        with pytest.raises(ValueError, match='audit-bound'):
            execution.validate_binding(case, **kwargs)
    else:
        # Binding only; the launch caller must verify actual context/proof bytes.
        assert execution.validate_binding(case, **kwargs) == tuple(range(5234314, 5234328))


@pytest.mark.parametrize('change', ['record', 'tag', 'runtime', 'unbound'])
def test_invalid_bank_binding_stops_before_manager_factory(monkeypatch, tmp_path, change):
    _, case, kwargs = binding()
    if change == 'record':
        kwargs['record_id'] = 5234327
    elif change == 'tag':
        kwargs['compute_tag'] = 'foreign'
    elif change == 'runtime':
        kwargs['runtime']['specification']['protocols'] = {}
    else:
        case['dataset_mode'] = 'awaiting-audited-bootstrap'
    session = load('single_record_cb63305')
    attempts = []
    @contextmanager
    def forbidden_factory():
        attempts.append('factory-entered')
        raise AssertionError('invalid binding constructed a manager')
        yield
    original = session.helper
    monkeypatch.setattr(session, 'helper', lambda name:
        NS(grid400_manager_client_factory=forbidden_factory)
        if name == 'dataset_fresh_cb63305' else original(name))
    with pytest.raises(ValueError):
        session.run_session(NS(), **kwargs, case=case, campaign_scope='eligible-bank',
            producer='grid400', payload=tmp_path, save=lambda *args: None)
    assert attempts == []
