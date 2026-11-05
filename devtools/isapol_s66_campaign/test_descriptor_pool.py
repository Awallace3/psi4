"""Synthetic pool-binding transactions; actual shell arithmetic has its own tests."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def fixture(monkeypatch):
    path = Path(__file__).with_name('descriptor_pool.py')
    spec = importlib.util.spec_from_file_location('_descriptor_pool_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.helper
    prepare = NS(json_digest=lambda value: module.digest(encode(value)))
    rows = []
    for i in range(1, 132):
        runtime = dict(specification=dict(keywords=dict(function_kwargs=dict(
            native_monomer=dict(resources=dict(max_bytes=12*1024**3))))))
        rows.append(dict(monomer_id=f'monomer-{i:04d}', runtime_input=runtime,
            runtime_input_sha256=prepare.json_digest(runtime), intended_memory_bytes=20*1024**3))
    bank = dict(monomers=rows, producer={'synthetic': 'not acceptance'})
    prepare.prepare_grid400_full_bank = lambda *args: copy.deepcopy(bank)
    profile = original('descriptor_profiles').profile('grid400', scope='full-bank')
    supervisor = original('phoenix_descriptor_bank_cb63305')
    planner = original('resource_plan_grid400')
    operations = dict.fromkeys(supervisor.OP_FILES+profile['parent_extra_sources'], 'a'*64)
    admission = original('descriptor_admission_cb63305')
    # Binding/coverage are real. Only the separately tested measured-shell
    # arithmetic is replaced; these are not real descriptors or eligibility.
    calls = []
    module.descriptor_calls = calls
    def measured_stub(report, row, identity, **kw):
        calls.append(dict(monomer_id=row['monomer_id'], identity=identity, **copy.deepcopy(kw)))
        return dict(monomer_id=row['monomer_id'], status=report['status'])
    admission.verify_descriptor = measured_stub
    monkeypatch.setattr(module, 'helper', lambda name: {
        'prepare_descriptor_bank_cb63305': prepare,
        'descriptor_admission_cb63305': admission,
    }.get(name) or original(name))
    reports, measurements = {}, []
    for i, row in enumerate(rows):
        name = row['monomer_id']+'-descriptors.json'
        result = dict(monomer_id=row['monomer_id'], status=(
            'PASS_DESCRIPTOR_PLAN_ONLY' if i < 116 else 'REFUSED_DESCRIPTOR_PLAN'))
        reports[name] = encode(result)
        measurements.append(dict(result, descriptor_report=name,
            descriptor_report_sha256=module.digest(reports[name]),
            runtime_input_sha256=row['runtime_input_sha256']))
    route = dict(JobId='123', Account='gts-cs207-chemx', QOS='embers',
        Partition='cpu-small', NumNodes='1', NumCPUs='24', MinMemoryNode='180G', TimeLimit='00:45:00')
    allocation = dict(schema='native-campaign-descriptor-allocation/2', campaign_scope='full-bank',
        runtime_unchanged=True, scientific_build_commit=planner.SCIENTIFIC_COMMIT,
        operational_commit='b'*40, bank_sha256=module.digest(encode(bank)),
        policy_id=planner.POLICY, source_policy_sha256=planner.FILES,
        operational_sources_sha256=operations, producer=bank['producer'], allocation=route,
        memory_contract=admission.admit_memory(bank, route, nonnumeric_headroom_bytes=16*1024**3),
        reports=measurements, status='REFUSED_DESCRIPTOR_PLANS')
    authority = dict(bank_sha256=module.digest(encode(bank)),
        resources_sha256=module.digest(b'{}'), allocation_sha256=module.digest(encode(allocation)),
        operational_commit='b'*40, job_id='123', operational_sources_sha256=operations)
    receipts = {'imported_identity': b'{}'}
    return module, bank, allocation, reports, authority, receipts


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def verify(f):
    module, bank, allocation, reports, authority, receipts = f
    return module.verify_pool({}, receipts, b'{}', encode(bank), encode(allocation),
                              reports, authority=authority)


def test_complete_pool_preserves_all_refusals_without_claiming_full_admission(monkeypatch):
    f = fixture(monkeypatch)
    module, bank, _, _, authority, _ = f
    result = verify(f)
    assert len(result['eligible_ids']) == 116 and len(result['refused_ids']) == 15
    assert result['global_pool_status'] == 'REFUSED_DESCRIPTOR_PLANS'
    assert result['status'] == 'VERIFIED_DESCRIPTOR_POOL_ELIGIBILITY_ONLY'
    assert len(result['bank']['monomers']) == len(result['allocation']['reports']) == 131
    assert not set(result['eligible_ids']) & set(result['refused_ids'])
    profile = module.helper('descriptor_profiles').profile('grid400', scope='full-bank')
    sources = {name: authority['operational_sources_sha256'][name] for name in profile['child_sources']}
    assert len(module.descriptor_calls) == len({r['monomer_id'] for r in module.descriptor_calls}) == 131
    for call, row in zip(module.descriptor_calls, bank['monomers']):
        assert call == dict(monomer_id=row['monomer_id'], identity={},
            producer='grid400', scope='full-bank', runtime_sha256=row['runtime_input_sha256'],
            policy=module.helper('resource_plan_grid400').FILES, operational_sources_sha256=sources)


@pytest.mark.parametrize('change', ['missing_report', 'report_bytes', 'missing_row',
    'duplicate_row', 'global_pass', 'child_pass', 'runtime', 'scope', 'producer',
    'source', 'job', 'memory', 'wrong_bank_hash', 'wrong_allocation_hash', 'wrong_resources_hash'])
def test_incomplete_tampered_or_cross_scoped_pool_cannot_authorize_eligibility(monkeypatch, change):
    f = fixture(monkeypatch)
    module, bank, allocation, reports, authority, _ = f
    first = allocation['reports'][0]
    if change == 'missing_report':
        reports.pop(first['descriptor_report'])
    elif change == 'report_bytes':
        reports[first['descriptor_report']] += b' '
    elif change == 'missing_row':
        bank['monomers'].pop()
    elif change == 'duplicate_row':
        bank['monomers'][-1] = copy.deepcopy(bank['monomers'][0])
    elif change == 'global_pass':
        allocation['status'] = 'PASS_ALL_131_DESCRIPTOR_PLANS_ONLY'
    elif change == 'child_pass':
        allocation['reports'][-1]['status'] = 'PASS_DESCRIPTOR_PLAN_ONLY'
    elif change == 'runtime':
        bank['monomers'][0]['runtime_input']['changed'] = True
    elif change == 'scope':
        allocation['campaign_scope'] = 'default'
    elif change == 'producer':
        allocation['producer'] = {'foreign': True}
    elif change == 'source':
        allocation['operational_sources_sha256'] = dict(
            allocation['operational_sources_sha256'], extra='c'*64)
    elif change == 'job':
        allocation['allocation']['JobId'] = '124'
    elif change == 'memory':
        allocation['memory_contract']['maximum_intended_psi4_memory'] += 1
    # Recommit altered bytes to exercise semantic checks as well as hash checks.
    authority['bank_sha256'] = module.digest(encode(bank))
    allocation['bank_sha256'] = authority['bank_sha256']
    authority['allocation_sha256'] = module.digest(encode(allocation))
    if change.startswith('wrong_'):
        authority[{'wrong_bank_hash': 'bank_sha256',
                   'wrong_allocation_hash': 'allocation_sha256',
                   'wrong_resources_hash': 'resources_sha256'}[change]] = '0'*64
    with pytest.raises(ValueError):
        verify(f)
