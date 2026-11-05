"""Verify complete measured grid400 descriptor pools without promoting refusals.

Authority is supplied by a closed campaign adapter, never by a launch argument.
This module verifies bindings/arithmetic, not molecular results or acceptance.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def helper(name):
    if name not in ('prepare_descriptor_bank_cb63305', 'descriptor_admission_cb63305',
                    'descriptor_profiles', 'resource_plan_grid400', 'phoenix_descriptor_bank_cb63305'):
        raise ValueError('closed full descriptor pool helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_descriptor_pool_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_pool(inputs, receipts, resources, bank_raw, allocation_raw, reports, *, authority):
    """Return all verified rows, including refused ones; no runtime side effects."""
    roles = {'bank_sha256', 'resources_sha256', 'allocation_sha256',
             'operational_commit', 'job_id', 'operational_sources_sha256'}
    if type(authority) is not dict or set(authority) != roles:
        raise ValueError('complete campaign-bound descriptor authority required')
    for key in ('bank_sha256', 'resources_sha256', 'allocation_sha256'):
        if (type(authority[key]) is not str or re.fullmatch('[0-9a-f]{64}', authority[key]) is None):
            raise ValueError('exact descriptor receipt hash required')
    if (re.fullmatch('[0-9a-f]{40}', authority['operational_commit']) is None
            or re.fullmatch('[1-9][0-9]*', authority['job_id']) is None):
        raise ValueError('exact measured operational commit/job required')
    for raw, key in ((bank_raw, 'bank_sha256'), (resources, 'resources_sha256'),
                     (allocation_raw, 'allocation_sha256')):
        if type(raw) is not bytes or digest(raw) != authority[key]:
            raise ValueError('descriptor pool byte identity differs: '+key)
    prepare = helper('prepare_descriptor_bank_cb63305')
    bank = prepare.prepare_grid400_full_bank(inputs, receipts, resources)
    if bank != json.loads(bank_raw):
        raise ValueError('full raw candidate bank does not reproduce')
    profile = helper('descriptor_profiles').profile('grid400', scope='full-bank')
    planner = helper('resource_plan_grid400')
    operations = authority['operational_sources_sha256']
    names = set(helper('phoenix_descriptor_bank_cb63305').OP_FILES+profile['parent_extra_sources'])
    if (type(operations) is not dict or set(operations) != names
            or any(type(sha) is not str or re.fullmatch('[0-9a-f]{64}', sha) is None
                   for sha in operations.values())):
        raise ValueError('complete measured parent/child operational source inventory required')
    allocation = json.loads(allocation_raw)
    if (allocation.get('schema') != 'native-campaign-descriptor-allocation/2'
            or allocation.get('campaign_scope') != 'full-bank'
            or allocation.get('runtime_unchanged') is not True
            or allocation.get('scientific_build_commit') != planner.SCIENTIFIC_COMMIT
            or allocation.get('operational_commit') != authority['operational_commit']
            or allocation.get('bank_sha256') != authority['bank_sha256']
            or allocation.get('policy_id') != planner.POLICY
            or allocation.get('source_policy_sha256') != planner.FILES
            or allocation.get('operational_sources_sha256') != operations
            or allocation.get('producer') != bank['producer']
            or allocation.get('allocation', {}).get('JobId') != authority['job_id']):
        raise ValueError('exact full-scope measured descriptor allocation required')
    admission = helper('descriptor_admission_cb63305')
    supervisor = helper('phoenix_descriptor_bank_cb63305')
    supervisor.admit_route(allocation['allocation'], scope='full-bank')
    memory = admission.admit_memory(bank, allocation['allocation'],
                                    nonnumeric_headroom_bytes=16*1024**3)
    if allocation.get('memory_contract') != memory:
        raise ValueError('actual descriptor allocation memory contract differs')
    expected_ids = {f'monomer-{i:04d}' for i in range(1, 132)}
    by_id = {row['monomer_id']: row for row in bank['monomers']}
    if len(bank['monomers']) != 131 or set(by_id) != expected_ids:
        raise ValueError('complete unique131 candidate runtime inventory required')
    measurements = allocation.get('reports')
    status = admission.verify_coverage(measurements, expected_ids)
    if allocation.get('status') != status:
        raise ValueError('global pool verdict must retain every refusal')
    if type(reports) is not dict or set(reports) != {
            mid+'-descriptors.json' for mid in expected_ids}:
        raise ValueError('complete exact131 measured child report bytes required')
    child_sources = {name: operations[name] for name in profile['child_sources']}
    identity = json.loads(receipts['imported_identity'])
    eligible, refused = [], []
    for measurement in measurements:
        mid = measurement['monomer_id']
        row, name = by_id[mid], mid+'-descriptors.json'
        raw = reports[name]
        runtime_sha = prepare.json_digest(row['runtime_input'])
        if (type(raw) is not bytes or row.get('runtime_input_sha256') != runtime_sha):
            raise ValueError('raw descriptor child/runtime bytes required')
        result = admission.verify_descriptor(json.loads(raw), row, identity,
            policy=planner.FILES, operational_sources_sha256=child_sources,
            runtime_sha256=runtime_sha, producer='grid400', scope='full-bank')
        expected = dict(result, descriptor_report=name, descriptor_report_sha256=digest(raw),
                        runtime_input_sha256=runtime_sha)
        if measurement != expected:
            raise ValueError('actual per-row descriptor admission differs: '+mid)
        (eligible if result['status'] == 'PASS_DESCRIPTOR_PLAN_ONLY' else refused).append(mid)
    return dict(status='VERIFIED_DESCRIPTOR_POOL_ELIGIBILITY_ONLY', bank=bank,
        allocation=allocation, eligible_ids=sorted(eligible), refused_ids=sorted(refused),
        global_pool_status=status, authority=dict(authority),
        qualification='Complete measured descriptor inventory and per-row eligibility only. '
                      'Refused rows remain refused; no SCF/model/whole-set acceptance.')
