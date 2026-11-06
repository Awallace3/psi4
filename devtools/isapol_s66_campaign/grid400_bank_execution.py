"""Bind one reconstructed eligible-bank case to one finite manager session.

The launch caller must first reproduce the case from closed measured inputs
and, for append mode, verify the ordinary-audit-created checkpoint. This pure
guard alone cannot create that authority or start a manager.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def helper(name):
    if name not in ('grid400_bank_policy', 'phoenix_worker_probe', 'phoenix_pilot',
                    'bank_wave_runtime'):
        raise ValueError('closed eligible bank execution helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_grid400_bank_execution_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def worker_shapes():
    policy = helper('grid400_bank_policy')
    return tuple((p['cores'], p['science_memory_GiB']) for p in (
        policy.execution_profile(name) for name in ('s20', 's40', 's60', 's80', 's120')))


def validate_binding(case, *, runtime, compute_tag, worker_shape, record_id,
                     campaign_scope='eligible-bank', wave_binding=None, allocation=None):
    policy = helper('grid400_bank_policy')
    if campaign_scope not in ('eligible-bank', 'eligible-wave'):
        raise ValueError('explicit finite bank execution scope required')
    wave = campaign_scope == 'eligible-wave'
    if not wave and (wave_binding is not None or allocation is not None):
        raise ValueError('wave authority cannot enter serial binding')
    if (type(case) is not dict or case.get('schema') != 'native-selected-molecular-pilot/1'
            or case.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
            or case.get('producer_profile') != 'grid400'
            or case.get('campaign_scope') != campaign_scope
            or case.get('campaign') != policy.CAMPAIGN
            or case.get('dataset_mode') not in ('fresh', 'append')
            or type(case.get('monomer_id')) is not str
            or re.fullmatch(r'monomer-(?:00[0-9]{2}|01[0-9]{2})', case['monomer_id']) is None
            or not 1 <= int(case['monomer_id'][-4:]) <= 131
            or type(case.get('eligibility_sha256')) is not str
            or re.fullmatch('[0-9a-f]{64}', case['eligibility_sha256']) is None
            or type(record_id) is not int or record_id <= 0 or record_id in policy.PROTECTED_RECORDS):
        raise ValueError('one reconstructed eligible grid400 bank session required')
    if wave:
        if wave_binding is None or allocation is None or case.get('dataset_mode') != 'append':
            raise ValueError('registered wave binding and actual science allocation required')
        adapter = helper('bank_wave_runtime')
        adapter.helper('bank_wave_campaign').revalidate_binding(wave_binding)
        slot = adapter.helper('bank_wave').validate_registered_slot(wave_binding['plan'],
            wave_binding['registration'], wave_binding['release_permit'], case, allocation=allocation)
        if record_id != slot['record_id'] or compute_tag != slot['compute_tag']:
            raise ValueError('actual wave session record/tag differs from registered slot')
    if re.fullmatch(re.escape('ccasp-native-grid400-bank-'+case['monomer_id']+'-')+
                    '[1-9][0-9]*', compute_tag) is None:
        raise ValueError('one dedicated eligible grid400 bank compute tag required')
    if case['dataset_mode'] == 'fresh':
        if any(k in case for k in ('dataset_id', 'bootstrap_context_sha256', 'checkpoint_sha256')):
            raise ValueError('fresh bank bootstrap cannot carry an existing dataset or checkpoint')
    elif (type(case.get('dataset_id')) is not int or case['dataset_id'] <= 0
            or case['dataset_id'] in policy.PROTECTED_DATASETS
            or any(type(case.get(key)) is not str or re.fullmatch('[0-9a-f]{64}', case[key]) is None
                   for key in ('bootstrap_context_sha256', 'checkpoint_sha256'))):
        raise ValueError('ordinary-audit-bound fresh bank append required')
    profile = policy.validate_profile(case.get('execution_profile'))
    expected = helper('phoenix_worker_probe').runtime_input(case)
    declared = policy.validate_execution_profile(dict(runtime_input=expected,
        intended_memory_bytes=case.get('intended_memory_bytes')))
    if (_json(declared) != _json(profile)
            or hashlib.sha256(_json(expected).encode()).hexdigest() != case.get('runtime_input_sha256')):
        raise ValueError('exact admitted bank runtime hash and declared shape required')
    expected['specification']['protocols'] = helper('phoenix_pilot').execution_specification({})['protocols']
    queue = case.get('queue_policy', {})
    if (_json(runtime) != _json(expected)
            or worker_shape != (profile['cores'], profile['science_memory_GiB'])
            or queue.get('cpus') != profile['cores']
            or queue.get('psi4_memory_GiB') != profile['science_memory_GiB']
            or queue.get('job_memory_GiB') != profile['job_memory_GiB']
            or queue.get('maximum_records') != 1 or queue.get('concurrency') != 1
            or queue.get('find_existing') is not False or queue.get('existing_ok') is not False):
        raise ValueError('exact admitted runtime/no-correction protocols and finite bank shape required')
    return policy.PROTECTED_RECORDS
