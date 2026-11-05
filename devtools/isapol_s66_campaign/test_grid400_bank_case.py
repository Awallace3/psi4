"""Raw case construction, not installed science or runtime authorization."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def fixture():
    path = Path(__file__).with_name('grid400_bank_case.py')
    spec = importlib.util.spec_from_file_location('_grid400_bank_case_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare = module.helper('prepare_descriptor_bank_cb63305')
    entries, molecules, rows, reports = [], [], [], []
    for i in range(1, 132):
        mid = f'monomer-{i:04d}'
        molecule = dict(symbols=['O', 'H', 'H'], geometry=[0.]*9,
                        fix_com=True, fix_orientation=True, fix_symmetry='c1')
        request = dict(schema='native-monomer-request/3', point_group='c1',
            resources=dict(max_bytes=12*1024**3, max_work=6*10**12,
                           max_io_bytes=16*1024**3, capacity='large'),
            max_replay_work=10**12, pfit_maximum_work_bytes=2*1024**3)
        runtime = dict(molecule=copy.deepcopy(molecule), specification=dict(
            driver='properties', model=dict(method='pbe0', basis='aug-cc-pvtz'),
            keywords=dict(reference='rhf', function_kwargs=dict(native_monomer=request,
                                                               properties=['dipole']))))
        sha = prepare.json_digest(runtime)
        entry = dict(name='synthetic-'+mid, molecule=molecule, additional_keywords={},
                     attributes=dict(monomer_id=mid, reference_energy='metadata only'))
        entries.append(entry)
        molecules.append(dict(id=mid, formula='SYNTHETIC_ONLY'))
        rows.append(dict(monomer_id=mid, entry_name=entry['name'], runtime_input=runtime,
                         runtime_input_sha256=sha, intended_memory_bytes=20*1024**3))
        reports.append(dict(monomer_id=mid, runtime_input_sha256=sha, dimensions=dict(sites=3),
            descriptor_report_sha256='a'*64, status=(
                'PASS_DESCRIPTOR_PLAN_ONLY' if i <= 116 else 'REFUSED_DESCRIPTOR_PLAN')))
    inputs = dict(inventory=json.dumps(dict(monomers=molecules)).encode(),
                  entries=json.dumps(dict(entries=entries)).encode())
    pool = dict(status='VERIFIED_DESCRIPTOR_POOL_ELIGIBILITY_ONLY',
        eligible_ids=[r['monomer_id'] for r in rows[:116]],
        refused_ids=[r['monomer_id'] for r in rows[116:]],
        bank=dict(monomers=rows, producer={'synthetic': 'not acceptance'},
                  input_sha256={role: hashlib.sha256(raw).hexdigest() for role, raw in inputs.items()},
                  resource_declarations_sha256='b'*64),
        allocation=dict(reports=reports),
        authority=dict(bank_sha256='c'*64, allocation_sha256='d'*64, operational_commit='e'*40))
    return module, inputs, pool


def test_only_first_eligible_case_can_be_a_fresh_bootstrap():
    module, inputs, pool = fixture()
    first = module.prepare_case(inputs, pool, 'monomer-0001')
    later = module.prepare_case(inputs, pool, 'monomer-0002')
    assert first['dataset_mode'] == 'fresh'
    assert later['dataset_mode'] == 'awaiting-audited-bootstrap'
    for case in (first, later):
        assert 'dataset_id' not in case
        assert case['campaign_scope'] == 'eligible-bank'
        assert case['dataset_name'] == module.DATASET_NAME
        assert case['eligible_monomer_count'] == 116 and case['refused_monomer_count'] == 15
        assert case['execution_profile']['name'] == 'grid400-bank-s20'
        assert case['queue_policy']['maximum_records'] == case['queue_policy']['concurrency'] == 1
        assert case['entry']['attributes']['reference_energy'] == 'metadata only'
        assert 'reference_energy' not in json.dumps(
            module.helper('pilot_cb63305').helper('phoenix_worker_probe').runtime_input(case))
    assert first['eligibility_sha256'] == later['eligibility_sha256']


def test_refused_row_cannot_be_prepared_or_silently_promoted():
    module, inputs, pool = fixture()
    with pytest.raises(ValueError):
        module.prepare_case(inputs, pool, 'monomer-0131')
    pool['eligible_ids'].append(pool['refused_ids'].pop())
    pool['eligible_ids'].sort()
    with pytest.raises(ValueError, match='refused descriptor'):
        module.prepare_case(inputs, pool, 'monomer-0131')


def test_first_admitted_member_not_monomer0001_defines_fresh_bootstrap():
    module, inputs, pool = fixture()
    pool['refused_ids'].append(pool['eligible_ids'].pop(0))
    pool['refused_ids'].sort()
    pool['allocation']['reports'][0]['status'] = 'REFUSED_DESCRIPTOR_PLAN'
    assert module.prepare_case(inputs, pool, 'monomer-0002')['dataset_mode'] == 'fresh'
    assert module.prepare_case(inputs, pool, 'monomer-0003')['dataset_mode'] == 'awaiting-audited-bootstrap'
    with pytest.raises(ValueError):
        module.prepare_case(inputs, pool, 'monomer-0001')


@pytest.mark.parametrize('change', ['reference_attributes', 'formula'])
def test_metadata_changes_cannot_keep_original_input_provenance(change):
    module, inputs, pool = fixture()
    if change == 'reference_attributes':
        entries = json.loads(inputs['entries'])
        entries['entries'][0]['attributes']['reference_energy'] = 'changed'
        inputs['entries'] = json.dumps(entries).encode()
    else:
        inventory = json.loads(inputs['inventory'])
        inventory['monomers'][0]['formula'] = 'changed'
        inputs['inventory'] = json.dumps(inventory).encode()
    with pytest.raises(ValueError, match='raw input bytes'):
        module.prepare_case(inputs, pool, 'monomer-0001')


@pytest.mark.parametrize('change', ['geometry', 'runtime_hash', 'report_hash', 'point_group'])
def test_shared_raw_constructor_rejects_changed_inputs_or_runtime(change):
    module, inputs, pool = fixture()
    before = copy.deepcopy(pool)
    row, report = pool['bank']['monomers'][0], pool['allocation']['reports'][0]
    if change == 'geometry':
        entries = json.loads(inputs['entries'])
        entries['entries'][0]['molecule']['geometry'][0] = 1.
        inputs['entries'] = json.dumps(entries).encode()
    elif change == 'runtime_hash':
        row['runtime_input_sha256'] = 'f'*64
    elif change == 'report_hash':
        report['runtime_input_sha256'] = 'f'*64
    else:
        row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']['point_group'] = 'd2h'
    with pytest.raises(ValueError):
        module.prepare_case(inputs, pool, 'monomer-0001')
    assert before['eligible_ids'] == pool['eligible_ids']
