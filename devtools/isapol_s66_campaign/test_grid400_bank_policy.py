"""Finite resource policy tests, no calculations or descriptor admission."""
import copy
import importlib.util
from pathlib import Path

import pytest


def policy():
    path = Path(__file__).with_name('grid400_bank_policy.py')
    spec = importlib.util.spec_from_file_location('_grid400_bank_policy_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(numeric, science):
    return dict(intended_memory_bytes=science*1024**3, runtime_input=dict(
        specification=dict(keywords=dict(function_kwargs=dict(native_monomer=dict(
            schema='native-monomer-request/3', resources=dict(max_bytes=numeric*1024**3,
                max_work=128*10**12, max_io_bytes=16*1024**3, capacity='extended'),
            max_replay_work=10**12, pfit_maximum_work_bytes=2*1024**3))))))


@pytest.mark.parametrize('name,numeric,science,job,cores,duration', [
    ('s20', 12, 20, 32, 8, '02:00:00'), ('s40', 32, 40, 64, 12, '04:00:00'),
    ('s60', 48, 60, 96, 16, '04:00:00'), ('s80', 64, 80, 144, 24, '08:00:00'),
    ('s120', 96, 120, 186, 24, '08:00:00')])
def test_exact_finite_declared_shapes(name, numeric, science, job, cores, duration):
    p = policy()
    result = p.validate_execution_profile(row(numeric, science))
    assert result == p.execution_profile(name) == p.validate_profile(result)
    assert (result['science_memory_GiB'], result['job_memory_GiB'], result['cores'],
            result['time_limit']) == (science, job, cores, duration)
    assert result['qos'] == 'embers' and result['concurrency'] == 1
    assert result['scratch_bytes'] == 256*1024**3


@pytest.mark.parametrize('change', ['memory', 'numeric', 'work', 'large_work', 'io',
                                  'replay', 'pfit', 'schema', 'bool', 'extra'])
def test_no_automatic_promotion_or_cap_increase(change):
    r = row(48, 60)
    request = r['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    resources = request['resources']
    if change == 'memory':
        r['intended_memory_bytes'] = 80*1024**3
    elif change == 'numeric':
        resources['max_bytes'] = 49*1024**3
    elif change == 'work':
        resources['max_work'] += 1
    elif change == 'large_work':
        resources['capacity'] = 'large'
    elif change == 'io':
        resources['max_io_bytes'] = 65*1024**3
    elif change == 'replay':
        request['max_replay_work'] = 2*10**12+1
    elif change == 'pfit':
        request['pfit_maximum_work_bytes'] = resources['max_bytes']+1
    elif change == 'schema':
        request['schema'] = 'native-monomer-request/2'
    elif change == 'bool':
        resources['max_work'] = True
    else:
        resources['extra'] = 1
    with pytest.raises(ValueError):
        policy().validate_execution_profile(r)


@pytest.mark.parametrize('field,value', [('qos', 'inferno'), ('cores', 8),
    ('scratch_bytes', 512*1024**3), ('concurrency', 2), ('time_limit', '09:00:00')])
def test_profile_modification_is_not_admitted(field, value):
    p = policy()
    profile = copy.deepcopy(p.execution_profile('s60'))
    profile[field] = value
    with pytest.raises(ValueError):
        p.validate_profile(profile)


def test_historical_singleton_and_failed_forensic_record_are_protected():
    p = policy()
    assert p.PROTECTED_DATASETS == (353, 354, 355, 356)
    assert set(range(5234314, 5234328)) <= set(p.PROTECTED_RECORDS)
