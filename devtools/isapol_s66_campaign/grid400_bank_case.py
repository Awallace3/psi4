"""Construct raw cases only after complete measured-pool verification.

The caller must reconstruct the pool from its closed campaign authority at
every launch/audit. These pure dictionaries are not execution authorization.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

DATASET_NAME = 'S66x8-native-CCASP-grid400-1aaa9e0a30-eligible-request3-v1'
SPECIFICATION_NAME = 'pbe0-atz-native-grid400-request3'


def helper(name):
    if name not in ('pilot_cb63305', 'grid400_bank_policy', 'prepare_descriptor_bank_cb63305'):
        raise ValueError('closed eligible bank case helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_grid400_bank_case_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def prepare_case(inputs, pool, monomer_id):
    """Reproduce a case without records, old accepted seeds or scientific data."""
    if (pool.get('status') != 'VERIFIED_DESCRIPTOR_POOL_ELIGIBILITY_ONLY'
            or pool.get('eligible_ids') != sorted(set(pool.get('eligible_ids', [])))
            or not pool['eligible_ids'] or monomer_id not in pool['eligible_ids']
            or set(pool['eligible_ids']) & set(pool.get('refused_ids', []))
            or len(pool['eligible_ids'])+len(pool.get('refused_ids', [])) != 131):
        raise ValueError('one actually admitted member of the complete measured pool required')
    policy = helper('grid400_bank_policy')
    bank, allocation, authority = (pool[k] for k in ('bank', 'allocation', 'authority'))
    if (type(inputs) is not dict or set(inputs) != set(bank['input_sha256'])
            or any(type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != bank['input_sha256'][role]
                   for role, raw in inputs.items())):
        raise ValueError('exact measured-bank raw input bytes required, including reference metadata')
    row = next(r for r in bank['monomers'] if r['monomer_id'] == monomer_id)
    report = next(r for r in allocation['reports'] if r['monomer_id'] == monomer_id)
    if report['status'] != 'PASS_DESCRIPTOR_PLAN_ONLY':
        raise ValueError('refused descriptor row cannot become a dispatch case')
    execution = policy.validate_execution_profile(row)
    pins = dict(producer='grid400', monomer=monomer_id, dataset_name=DATASET_NAME,
        specification_name=SPECIFICATION_NAME, schema='native-selected-molecular-pilot/1',
        bank_sha256=authority['bank_sha256'], allocation_sha256=authority['allocation_sha256'],
        runtime_input_sha256=row['runtime_input_sha256'],
        selected_report_sha256=report['descriptor_report_sha256'],
        descriptor_ops=authority['operational_commit'])
    molecule = next(m for m in json.loads(inputs['inventory'])['monomers']
                    if m['id'] == monomer_id)
    result = helper('pilot_cb63305')._prepare_fresh_selected(
        inputs, bank, allocation, pins, execution, formula=molecule['formula'],
        qualification='Measured eligible-subset raw case only, not model or execution acceptance. '
            'References remain entry attributes; historical datasets353..356 and records are protected.')
    # Only the first eligible row can bootstrap a fresh dataset. All later
    # rows remain unbound until an ordinary-audit-created checkpoint authorizes
    # their single append; no forensic receipt or arbitrary dataset ID applies.
    if monomer_id != pool['eligible_ids'][0]:
        result['dataset_mode'] = 'awaiting-audited-bootstrap'
    result.update(campaign_scope='eligible-bank', campaign=policy.CAMPAIGN,
        eligibility_sha256=helper('prepare_descriptor_bank_cb63305').json_digest(dict(
            eligible_ids=pool['eligible_ids'], refused_ids=pool['refused_ids'],
            descriptor_authority=authority)),
        eligible_monomer_count=len(pool['eligible_ids']), refused_monomer_count=len(pool['refused_ids']))
    return result
