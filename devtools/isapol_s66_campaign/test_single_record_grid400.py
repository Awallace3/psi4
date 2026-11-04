"""Synthetic single-manager guards/lifecycle; no server or quantum execution."""
from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load('_grid400_session_test', 'single_record_cb63305.py')
fixtures = load('_grid400_session_pilot_fixture', 'test_pilot_grid400.py')
TAG = 'ccasp-native-grid400-monomer-0008-123'


def setup(monkeypatch):
    inputs, _, _, _ = fixtures.fixture(monkeypatch)
    draft = fixtures.construct(inputs)
    runtime = fixtures.pilot.helper('phoenix_worker_probe').runtime_input(draft)
    runtime['specification']['protocols'] = session.helper('phoenix_pilot').execution_specification({})['protocols']
    original = session.helper
    monkeypatch.setattr(session, 'helper', lambda name:
                        fixtures.pilot if name == 'pilot_cb63305' else original(name))
    config = NS(executors=dict(pilot=NS(type='local', max_workers=1, cores_per_worker=16,
                                      memory_per_worker=60., compute_tags=[TAG])))
    return runtime, config


@pytest.mark.parametrize('record', list(range(5234314, 5234327)))
def test_all_old_records_are_refused_before_manager_construction(monkeypatch, tmp_path, record):
    runtime, config = setup(monkeypatch)
    with pytest.raises(ValueError, match='fresh expanded-pilot record'):
        session.run_session(config, record_id=record, compute_tag=TAG, runtime=runtime,
            payload=tmp_path, save=lambda *args: None, worker_shape=(16, 60.), producer='grid400')


@pytest.mark.parametrize('change', ['shape', 'tag', 'append', 'runtime', 'protocols', 'missing_protocols'])
def test_unbound_grid400_input_or_mode_cannot_construct_a_manager(monkeypatch, tmp_path, change):
    runtime, config = setup(monkeypatch)
    kwargs = dict(record_id=9999, compute_tag=TAG, runtime=runtime, payload=tmp_path,
        save=lambda *args: None, worker_shape=(16, 60.), producer='grid400')
    if change == 'shape':
        kwargs['worker_shape'] = (12, 40.)
    elif change == 'tag':
        kwargs['compute_tag'] = 'ccasp-native-ed561666-monomer-0008-123'
    elif change == 'append':
        kwargs['case'] = {'dataset_id': 355}
    elif change == 'runtime':
        runtime['specification']['keywords']['reference'] = 'uhf'
    elif change == 'protocols':
        runtime['specification']['protocols']['error_correction']['default_policy'] = True
    else:
        runtime['specification'].pop('protocols')
    with pytest.raises(ValueError):
        session.run_session(config, **kwargs)


def test_new_shape_requires_explicit_grid400_producer(monkeypatch, tmp_path):
    runtime, config = setup(monkeypatch)
    with pytest.raises(ValueError, match='explicit fresh producer'):
        session.run_session(config, record_id=9999, compute_tag=TAG, runtime=runtime,
            payload=tmp_path, save=lambda *args: None, worker_shape=(16, 60.))


@pytest.mark.parametrize('bad_client', [False, True])
def test_owned_manager_factory_is_active_before_constructor_and_restored_before_start(
        monkeypatch, tmp_path, bad_client):
    runtime, config = setup(monkeypatch)
    active = [False]
    effects = []

    class Client:
        pass

    @contextmanager
    def factory():
        active[0] = True
        effects.append('factory-enter')
        try:
            yield Client
        finally:
            active[0] = False
            effects.append('factory-exit')

    class Manager:
        def __init__(self, config):
            assert active[0]
            effects.append('synthetic-activation')
            self.manager_config = config
            self.name = 'SYNTHETIC_GRID400_MANAGER'
            self.client = NS() if bad_client else Client()
            self.dflow_kernel = None
            self.statistics = NS(dict=lambda: {})

        def start(self):
            assert not active[0]
            effects.append('start')
            self.target_task_id = 77
            self.delivery_outcome = 'accepted'

    package = ModuleType('qcfractalcompute')
    package.__path__ = []
    module = ModuleType('qcfractalcompute.compute_manager')
    module.ComputeManager = Manager
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = session.helper
    monkeypatch.setattr(session, 'helper', lambda name:
        NS(grid400_manager_client_factory=factory) if name == 'dataset_fresh_cb63305' else original(name))
    kwargs = dict(record_id=9999, compute_tag=TAG, runtime=copy.deepcopy(runtime), payload=tmp_path,
        save=lambda *args: None, worker_shape=(16, 60.), producer='grid400')
    if bad_client:
        with pytest.raises(ValueError, match='one-shot activation client'):
            session.run_session(config, **kwargs)
        assert effects == ['factory-enter', 'synthetic-activation', 'factory-exit']
    else:
        result = session.run_session(config, **kwargs)
        assert result['task_id'] == 77
        assert effects == ['factory-enter', 'synthetic-activation', 'factory-exit', 'start']
    assert not active[0]
