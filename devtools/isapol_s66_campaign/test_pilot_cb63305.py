"""Synthetic first expanded-pilot bookkeeping; never molecular acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('pilot_cb63305', Path(__file__).with_name('pilot_cb63305.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(monkeypatch, change=None):
    runtime = dict(molecule=dict(symbols=['C', 'N']+['H']*5, geometry=[0.]*21),
        specification=dict(driver='properties', model=dict(method='pbe0', basis='aug-cc-pvtz'),
            keywords=dict(reference='rhf', function_kwargs=dict(properties=['dipole'],
                native_monomer=dict(schema='native-monomer-request/3', pfit_maximum_work_bytes=256*1024**2,
                    max_replay_work=10**12, declaration={'synthetic': 'not acceptance'},
                    resources=dict(max_bytes=12*1024**3, max_work=12*10**12,
                                   max_io_bytes=16*1024**3, capacity='large'))))))
    row = dict(monomer_id=pilot.MONOMER, entry_name='raw-methylamine',
        intended_memory_bytes=20*1024**3, runtime_input=runtime,
        runtime_input_sha256=pilot.digest(encode(runtime)))
    rows = [copy.deepcopy(row) | dict(monomer_id=f'monomer-{i:04d}') for i in range(1, 132)]
    bank = dict(monomers=rows, producer=dict(scientific_commit='cb63305-synthetic-unit-only'),
                input_sha256={'synthetic': 'unit-only'}, resource_declarations_sha256='unit-only')
    reports = {f'monomer-{i:04d}-descriptors.json': encode({'synthetic': i}) for i in range(1, 132)}
    report_rows = [dict(monomer_id=f'monomer-{i:04d}', status='PASS_DESCRIPTOR_PLAN_ONLY',
        descriptor_report=name, descriptor_report_sha256=pilot.digest(raw))
        for i, (name, raw) in enumerate(reports.items(), 1)]
    allocation = dict(schema='native-campaign-descriptor-allocation/2',
        status='PASS_ALL_131_DESCRIPTOR_PLANS_ONLY', runtime_unchanged=True,
        scientific_build_commit=bank['producer']['scientific_commit'],
        operational_commit=pilot.DESCRIPTOR_OPS, allocation=dict(JobId='14970302'),
        reports=report_rows)
    inventory = dict(monomers=[dict(id=pilot.MONOMER, formula='CH5N', representative=dict(
        dimer_id='03_Water-MeNH2_0.90', role='B', R=0.9, fragment_indices=list(range(3, 10))))])
    entry = dict(name=row['entry_name'], molecule=copy.deepcopy(runtime['molecule']),
        attributes=dict(monomer_id=pilot.MONOMER,
                        sapt_reference_components_kcal_mol={'SAPT0/adz DISP': -1.}),
        additional_keywords={})
    if change == 'wrong_producer':
        allocation['scientific_build_commit'] = 'acf54'
    elif change == 'wrong_ops':
        allocation['operational_commit'] = 'a'*40
    elif change == 'partial':
        allocation['reports'].pop()
    elif change == 'refused':
        allocation['reports'][0]['status'] = 'REFUSED_DESCRIPTOR_PLAN'
    elif change == 'runtime_drift':
        allocation['runtime_unchanged'] = False
    elif change == 'changed_geometry':
        entry['molecule']['geometry'][0] = 1.
    elif change == 'crosswired_name':
        entry['name'] = 'foreign geometry'
    elif change == 'uncovered_memory':
        rows[5]['intended_memory_bytes'] = 40*1024**3
    elif change == 'changed_runtime':
        rows[5]['runtime_input_sha256'] = 'f'*64
    elif change == 'wrong_representative':
        inventory['monomers'][0]['representative']['R'] = 1.
    bank_raw = encode(bank)
    monkeypatch.setattr(pilot, 'BANK_SHA256', pilot.digest(bank_raw))
    allocation['bank_sha256'] = pilot.BANK_SHA256
    allocation_raw = encode(allocation)
    monkeypatch.setattr(pilot, 'DESCRIPTOR_ALLOCATION_SHA256', pilot.digest(allocation_raw))
    original = pilot.helper
    prepare = NS(prepare_bank=lambda *args: copy.deepcopy(bank),
                 json_digest=lambda value: pilot.digest(encode(value)))
    monkeypatch.setattr(pilot, 'helper', lambda name: prepare
        if name == 'prepare_descriptor_bank_cb63305' else original(name))
    if change == 'substituted_report':
        reports[next(iter(reports))] += b' '
    elif change == 'missing_report':
        reports.pop(next(iter(reports)))
    return dict(inputs=dict(inventory=encode(inventory), entries=encode({'entries': [entry]})),
                receipts={}, resources=b'synthetic', bank_raw=bank_raw,
                allocation_raw=allocation_raw, reports=reports)


def test_nitrogen_pilot_preserves_admitted_runtime_and_reference_attributes(monkeypatch):
    args = fixture(monkeypatch)
    result = pilot.prepare_pilot(**args)
    assert result['status'] == 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
    assert result['monomer_id'] == 'monomer-0006'
    assert result['queue_policy']['find_existing'] is False
    assert result['queue_policy']['existing_ok'] is False
    assert result['queue_policy']['maximum_records'] == result['queue_policy']['concurrency'] == 1
    assert result['dataset_name'] == pilot.DATASET_NAME
    assert 'sapt_reference' in json.dumps(result['entry']['attributes'])
    runtime = pilot.helper('phoenix_worker_probe').runtime_input(result)
    assert 'sapt_reference' not in json.dumps(runtime)
    assert pilot.digest(encode(runtime)) == result['runtime_input_sha256']
    assert 'dataset353 is never a mutation target' in result['qualification']


@pytest.mark.parametrize('change', ['wrong_producer', 'wrong_ops', 'partial', 'refused',
    'runtime_drift', 'changed_geometry', 'crosswired_name', 'uncovered_memory',
    'changed_runtime', 'wrong_representative', 'substituted_report', 'missing_report'])
def test_receipt_join_identity_and_complete_admission_are_not_optional(monkeypatch, change):
    with pytest.raises(ValueError):
        pilot.prepare_pilot(**fixture(monkeypatch, change))


def test_production_descriptor_and_source_pins_are_not_old_pilot_receipts():
    assert pilot.DESCRIPTOR_ALLOCATION_SHA256 == (
        '328191e2481d9e9e6b7c213fa0b98d94804443e136f2dd6761f33ba6bf841f02')
    assert pilot.DESCRIPTOR_OPS == 'cbed46c755f7c1b63a1df0d9ebf0eeac4088606e'
    assert pilot.SPECIFICATION_NAME == 'pbe0-atz-native-request3'
