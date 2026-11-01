"""Pilot orchestration guards; synthetic metadata only, no server or SCF."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('phoenix_pilot', Path(__file__).with_name('phoenix_pilot.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_standalone_config_has_one_tag_no_inherited_executors_or_idle_shutdown():
    server = dict(fractal_uri='http://128.61.254.96:7777', username='synthetic', password='not-a-secret')
    original = copy.deepcopy(server)
    config = pilot.pilot_config(server, '/owned/run', '/owned/scratch', 'owned-tag')
    assert server == original
    # compute0.63 validates an explicitly supplied None as a duration string.
    # Omit the field to use its real None default without calling that validator.
    assert 'max_idle_time' not in config
    assert config['environments'] == dict(use_manager_environment=True, conda=[], apptainer=[])
    assert set(config['executors']) == {'pilot'}
    executor = config['executors']['pilot']
    assert executor['compute_tags'] == ['owned-tag']
    assert executor['max_workers'] == 1 and executor['type'] == 'local'
    assert executor['cores_per_worker'] == 4 and executor['memory_per_worker'] == 20
    assert executor['environments'] == config['environments']
    assert not executor.get('worker_init')  # Local executor would ignore it.


def claim():
    runtime = dict(molecule=dict(symbols=['O', 'H', 'H'], geometry=[0.]*9,
                                 molecular_charge=0., molecular_multiplicity=1,
                                 fix_com=True, fix_orientation=True, fix_symmetry='c1'),
                   specification=dict(driver='properties', model=dict(method='pbe0', basis='aug-cc-pvtz'),
                                      keywords=dict(reference='rhf', function_kwargs=dict(
                                          native_monomer={'point_group': 'c1'})),
                                      protocols={'error_correction': {'default_policy': False}}))
    # QCArchive may reuse an existing molecule with different symmetry metadata.
    cached = copy.deepcopy(runtime['molecule'])
    cached['fix_symmetry'] = None
    data = dict(schema_name='qcschema_input', schema_version=1,
                molecule=cached, **copy.deepcopy(runtime['specification']))
    task = dict(record_id=42, compute_tag='owned-tag', function='qcengine.compute',
                required_programs=['psi4'], function_kwargs=dict(program='psi4', input_data=data))
    return task, runtime


def test_claimed_execution_is_exact_but_cached_symmetry_is_not_edited():
    task, runtime = claim()
    original = copy.deepcopy(task)
    pilot.validate_claim(task, runtime, record_id=42, compute_tag='owned-tag')
    assert task == original and task['function_kwargs']['input_data']['molecule']['fix_symmetry'] is None


@pytest.mark.parametrize('changed', ['record', 'tag', 'program', 'function', 'keywords',
                                    'geometry', 'order', 'psiapi', 'files', 'references', 'correction', 'schema'])
def test_foreign_execution_inputs_never_reach_application_submission(changed):
    task, runtime = claim()
    data = task['function_kwargs']['input_data']
    if changed == 'record':
        task['record_id'] = 43
    elif changed == 'tag':
        task['compute_tag'] = '*'
    elif changed == 'program':
        task['function_kwargs']['program'] = 'other'
    elif changed == 'function':
        task['function'] = 'qcengine.compute_procedure'
    elif changed == 'keywords':
        data['keywords']['function_kwargs']['native_monomer']['point_group'] = 'c2v'
    elif changed == 'geometry':
        data['molecule']['geometry'][0] = .01
    elif changed == 'order':
        data['molecule']['symbols'] = ['H', 'O', 'H']
    elif changed == 'psiapi':
        data['extras'] = {'psiapi': True}
    elif changed == 'files':
        data['extras'] = {'extra_infiles': {'orbitals': 'saved'}}
    elif changed == 'correction':
        data['protocols']['error_correction']['default_policy'] = True
    elif changed == 'schema':
        data['schema_version'] = 2
    else:
        data['extras'] = {'sapt_reference': -2.}
    with pytest.raises(ValueError):
        pilot.validate_claim(task, runtime, record_id=42, compute_tag='owned-tag')


def test_execution_specification_explicitly_disables_automatic_scientific_changes():
    original = dict(program='psi4', driver='properties', method='pbe0',
                    basis='aug-cc-pvtz', keywords={'maxiter': 100})
    result = pilot.execution_specification(original)
    assert 'protocols' not in original
    assert result['protocols'] == {'error_correction': {'default_policy': False}}
    assert result['keywords'] == original['keywords']
