"""Mocked lifecycle only; real one-claim behavior has separate mixin tests."""
import importlib.util
from pathlib import Path
import signal
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location(
    'expanded_session_fixture', Path(__file__).with_name('single_record_cb63305.py'))
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)
TAG = 'ccasp-native-cb63305-monomer-0006-123'


def config():
    return NS(executors=dict(pilot=NS(type='local', max_workers=1, cores_per_worker=4,
                                     memory_per_worker=20., compute_tags=[TAG])))


def test_exact_single_worker_shape_and_fresh_record_required(tmp_path):
    session.validate_session_config(config(), TAG)
    for key, value in [('type', 'slurm'), ('max_workers', 2), ('cores_per_worker', 8),
                       ('memory_per_worker', 40.), ('compute_tags', ['*'])]:
        cfg = config()
        setattr(cfg.executors['pilot'], key, value)
        with pytest.raises(ValueError, match='exact single-worker'):
            session.validate_session_config(cfg, TAG)
    for record in (0, None, True, 5234314, 5234315):
        with pytest.raises(ValueError, match='fresh expanded-pilot'):
            session.run_session(config(), record_id=record, compute_tag=TAG,
                                runtime={}, payload=tmp_path, save=lambda *a: None)


@pytest.mark.parametrize('shape', [(8, 20.), (12, 40.), (24, 80.)])
def test_remaining_bank_requires_explicit_matching_worker_shape(shape):
    cfg = config()
    cfg.executors['pilot'].cores_per_worker, cfg.executors['pilot'].memory_per_worker = shape
    session.validate_session_config(cfg, TAG, worker_shape=shape)
    with pytest.raises(ValueError, match='exact single-worker'):
        session.validate_session_config(cfg, TAG)
    cfg.executors['pilot'].max_workers = 2
    with pytest.raises(ValueError, match='exact single-worker'):
        session.validate_session_config(cfg, TAG, worker_shape=shape)


@pytest.mark.parametrize('shape', [(True, 20.), (12, True), [12, 40.], (12, 80.),
                                   (24, 160.), (12, float('inf')), (12, float('nan'))])
def test_refuse_unbounded_or_implicit_worker_shapes(shape):
    with pytest.raises(ValueError, match='explicit finite'):
        session.validate_session_config(config(), TAG, worker_shape=shape)


@pytest.mark.parametrize('case', ['accepted', 'rejected', 'interrupted', 'callback_failure',
                                   'scientific_failure'])
def test_own_manager_cleanup_capture_and_signal_restoration(monkeypatch, tmp_path, case):
    effects, saved = [], {}
    stats = dict(total_successful_tasks=1, total_failed_tasks=0, total_rejected_tasks=0, active_tasks=0)
    runtime = dict(molecule=dict(symbols=['H', 'H'], geometry=[0., 0., 0., 0., 0., 2.],
        fix_com=True, fix_orientation=True, molecular_charge=0, molecular_multiplicity=1),
        specification=dict(driver='properties', model={'method': 'pbe0', 'basis': 'aug-cc-pvtz'},
                           keywords={}, protocols={'error_correction': {'default_policy': False}}))
    raw = dict(schema_name='qcschema_input', schema_version=1,
               molecule=runtime['molecule'], **runtime['specification'])
    task = NS(id=77, record_id=9999, compute_tag=TAG, function='qcengine.compute',
              required_programs=['psi4'], function_kwargs=dict(program='psi4', input_data=raw))
    class Manager:
        def __init__(self, config):
            self.manager_config = config
            self.name = 'SYNTHETIC_MANAGER_UNIT_ONLY'
            self.statistics = NS(active_tasks=0, active_cores=0, active_memory=0, total_cpu_hours=0,
                                 dict=lambda: stats)
            self.client = NS(deactivate=lambda **kw: effects.append('own_deactivate'))
            self.dflow_kernel = NS(cleanup=lambda: effects.append('own_cleanup'))
        def stop(self):
            effects.append('own_stop')
        def start(self):
            self.target_task_id = task.id
            self.preprocess_new_tasks([task])
            if case == 'callback_failure':
                raise RuntimeError('synthetic callback failure')
            result_data = {'extras': {'psi4:native_monomer': {'synthetic': 'not molecular evidence'}}}
            if case == 'scientific_failure':
                result_data = dict(success=False, extras=None,
                    error=dict(error_type='unknown_error', error_message='SYNTHETIC_ORIGINAL_SCIENTIFIC_ERROR'))
                stats.update(total_successful_tasks=0, total_failed_tasks=1)
            self.postprocess_results({task.id: NS(result_compressed=b'SYNTHETIC_UNIT_NOT_ZSTD',
                                                  result=result_data)})
            if case == 'scientific_failure':
                assert result_data['success'] is False and result_data['extras'] is None
                assert result_data['error']['error_message'] == 'SYNTHETIC_ORIGINAL_SCIENTIFIC_ERROR'
            if case == 'interrupted':
                signal.raise_signal(signal.SIGTERM)
            self.delivery_outcome = 'rejected' if case == 'rejected' else 'accepted'
    package = ModuleType('qcfractalcompute')
    package.__path__ = []
    module = ModuleType('qcfractalcompute.compute_manager')
    module.ComputeManager = Manager
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    kwargs = dict(record_id=9999, compute_tag=TAG, runtime=runtime, payload=tmp_path,
                  save=lambda name, data: saved.__setitem__(name, data))
    if case in ('accepted', 'scientific_failure'):
        result = session.run_session(config(), **kwargs)
        assert result['task_id'] == 77
        assert result['local_native'] == (None if case == 'scientific_failure'
                                         else {'synthetic': 'not molecular evidence'})
    else:
        with pytest.raises(RuntimeError):
            session.run_session(config(), **kwargs)
    assert effects.count('own_cleanup') == effects.count('own_deactivate') == 1
    assert saved['claim.json']['record_id'] == 9999
    assert saved['manager-delivery.json']['interrupted'] is (case == 'interrupted')
    assert saved['manager-delivery.json']['manager'] == 'SYNTHETIC_MANAGER_UNIT_ONLY'
    if case == 'scientific_failure':
        assert saved['manager-delivery.json']['outcome'] == 'accepted'
        assert saved['manager-delivery.json']['local_statistics']['total_failed_tasks'] == 1
        assert saved['manager-delivery.json']['local_statistics']['total_successful_tasks'] == 0
    assert {sig: signal.getsignal(sig) for sig in before} == before
    if case != 'callback_failure':
        assert (tmp_path/'local-result-77.zstd').read_bytes() == b'SYNTHETIC_UNIT_NOT_ZSTD'
