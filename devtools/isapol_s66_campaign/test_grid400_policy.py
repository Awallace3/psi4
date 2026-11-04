"""Pure approved shape/declaration checks, never molecular admission."""
import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).with_name('grid400_policy.py')
spec = importlib.util.spec_from_file_location('_grid400_execution_test', path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
GIB = 1024**3


def row():
    return dict(monomer_id='monomer-0008', intended_memory_bytes=60*GIB,
        runtime_input=dict(specification=dict(keywords=dict(function_kwargs=dict(
            native_monomer=dict(resources=dict(max_bytes=48*GIB, max_work=75*10**12,
                max_io_bytes=16*GIB, capacity='large'), max_replay_work=10**12,
                pfit_maximum_work_bytes=2*GIB))))))


def test_approved_singleton_shape_is_fresh_finite_and_no_more_than_four_hours():
    profile = policy.validate_execution_profile(row())
    assert profile == policy.execution_profile() == policy.validate_profile(profile)
    assert profile['cores'] == 16 and profile['science_memory_GiB'] == 60
    assert profile['job_memory_GiB'] == 96 and profile['time_limit'] == '04:00:00'
    assert profile['science_memory_GiB']*100 <= profile['job_memory_GiB']*65
    assert profile['qos'] == 'embers' and profile['concurrency'] == 1
    assert policy.PROTECTED_DATASETS == (353, 354, 355)
    assert set(range(5234314, 5234327)) == set(policy.PROTECTED_RECORDS)
    profile['cores'] = 24
    assert policy.execution_profile()['cores'] == 16


@pytest.mark.parametrize('field,value', [
    ('cores', 24), ('science_memory_GiB', 80), ('job_memory_GiB', 144),
    ('time_limit', '08:00:00'), ('qos', 'inferno'), ('concurrency', 2),
    ('scratch_bytes', 512*GIB), ('cores', 16.)])
def test_shape_cannot_be_promoted_or_implicitly_reinterpreted(field, value):
    profile = policy.execution_profile()
    profile[field] = value
    with pytest.raises(ValueError, match='exact approved grid400'):
        policy.validate_profile(profile)


@pytest.mark.parametrize('field,value', [
    ('max_bytes', 32*GIB), ('max_bytes', 64*GIB), ('max_work', 100*10**12),
    ('max_io_bytes', 32*GIB), ('capacity', 'extended'), ('max_bytes', float(48*GIB))])
def test_declared_budget_is_exact_and_does_not_change_on_refusal(field, value):
    data = row()
    request = data['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    request['resources'][field] = value
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match='exact fresh grid400'):
        policy.validate_execution_profile(data)
    assert data == before


def test_nonselected_monomer_and_changed_frozen_work_limits_refuse():
    for field, value in (('max_replay_work', 2*10**12), ('pfit_maximum_work_bytes', GIB)):
        data = row()
        data['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer'][field] = value
        with pytest.raises(ValueError, match='exact fresh grid400'):
            policy.validate_execution_profile(data)
    data = row()
    data['monomer_id'] = 'monomer-0009'
    with pytest.raises(ValueError, match='exact fresh grid400'):
        policy.validate_execution_profile(data)
