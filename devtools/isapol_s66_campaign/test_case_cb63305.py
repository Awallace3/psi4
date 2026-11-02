"""Synthetic generic-case selection; no real descriptor or molecular acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('bank_case_fixture', Path(__file__).with_name('case_cb63305.py'))
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(monkeypatch):
    rows, entries, measured, reports = [], [], [], {}
    for i in range(1, 132):
        monomer = f'monomer-{i:04d}'
        ns, memory = {47: (12, 40), 66: (17, 80)}.get(i, (3, 20))
        molecule = dict(symbols=['C']+['H']*(ns-1), geometry=[float(j) for j in range(3*ns)])
        request = dict(schema='native-monomer-request/3', pfit_maximum_work_bytes=2*1024**3,
            max_replay_work=10**12, declaration=dict(frames=['SYNTHETIC_UNIT_ONLY']*ns),
            resources=dict(max_bytes={20: 12, 40: 32, 80: 64}[memory]*1024**3,
                max_work=128*10**12 if memory == 80 else 100*10**12, max_io_bytes=16*1024**3,
                capacity='extended' if memory == 80 else 'large'))
        runtime = dict(molecule=molecule, specification=dict(driver='properties',
            model=dict(method='pbe0', basis='aug-cc-pvtz'),
            keywords=dict(reference='rhf', function_kwargs=dict(native_monomer=request, properties=['dipole']))))
        row = dict(monomer_id=monomer, entry_name=monomer+'-synthetic-only',
            geometry_sha256=case.json_digest(molecule), runtime_input=runtime,
            runtime_input_sha256=case.json_digest(runtime), intended_memory_bytes=memory*1024**3)
        rows.append(row)
        entries.append(dict(name=row['entry_name'], molecule=copy.deepcopy(molecule),
            additional_keywords={}, attributes=dict(monomer_id=monomer,
                sapt_reference_components_kcal_mol={'SAPT0/adz DISP': -1.})))
        name = monomer+'-descriptors.json'
        reports[name] = encode({'synthetic_unit_only': monomer})
        measured.append(dict(monomer_id=monomer, descriptor_report=name,
            descriptor_report_sha256=case.digest(reports[name]), status='PASS_DESCRIPTOR_PLAN_ONLY',
            runtime_input_sha256=row['runtime_input_sha256'],
            dimensions=dict(sites=ns, parameters=36*ns, nodes=11)))
    bank = dict(monomers=rows, producer=dict(scientific_commit='CB_SYNTHETIC_UNIT_ONLY'),
                input_sha256={'synthetic': 'unit-only'}, resource_declarations_sha256='synthetic')
    bank_raw = encode(bank)
    monkeypatch.setattr(case, 'BANK_SHA256', case.digest(bank_raw))
    allocation = dict(schema='native-campaign-descriptor-allocation/2',
        status='PASS_ALL_131_DESCRIPTOR_PLANS_ONLY', runtime_unchanged=True,
        scientific_build_commit=bank['producer']['scientific_commit'],
        operational_commit='cbed46c755f7c1b63a1df0d9ebf0eeac4088606e',
        bank_sha256=case.BANK_SHA256, allocation=dict(JobId='14970302'), reports=measured)
    raw = encode(allocation)
    monkeypatch.setattr(case, 'DESCRIPTOR_SHA256', case.digest(raw))
    original = case.helper
    monkeypatch.setattr(case, 'helper', lambda name: NS(prepare_bank=lambda *args: copy.deepcopy(bank))
                        if name == 'prepare_descriptor_bank_cb63305' else original(name))
    return dict(inputs={'entries': encode({'entries': entries})}, receipts={}, resources=b'synthetic',
                bank_raw=bank_raw, allocation_raw=raw, reports=reports), bank


@pytest.mark.parametrize('monomer,profile,science,job,cores,parameters', [
    ('monomer-0001', 's20', 20, 32, 8, 108),
    ('monomer-0047', 's40', 40, 64, 12, 432),
    ('monomer-0066', 's80', 80, 144, 24, 612)])
def test_case_preserves_science_and_references_but_separates_allocation_headroom(
        monkeypatch, monomer, profile, science, job, cores, parameters):
    args, bank = fixture(monkeypatch)
    result = case.prepare_case(**args, monomer_id=monomer, profile=profile)
    original = next(row for row in bank['monomers'] if row['monomer_id'] == monomer)
    assert case.helper('phoenix_worker_probe').runtime_input(result) == original['runtime_input']
    assert result['runtime_input_sha256'] == original['runtime_input_sha256']
    assert result['intended_memory_bytes'] == science*1024**3
    assert result['execution_profile']['job_memory_GiB'] == job
    assert result['execution_profile']['cores'] == cores
    assert result['execution_profile']['qos'] == 'embers'
    assert result['descriptor_dimensions']['parameters'] == parameters
    assert result['dataset_id'] == 354 and not result['submission']['find_existing']
    assert 'sapt_reference' in json.dumps(result['entry']['attributes'])
    assert 'sapt_reference' not in json.dumps(original['runtime_input'])
    assert result['status'] == 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'


@pytest.mark.parametrize('monomer,profile', [
    ('monomer-0006', 's20'), ('monomer-9999', 's20'), ('monomer-0001', 's40'),
    ('monomer-0047', 's20'), ('monomer-0066', 's40'), ('monomer-0001', 'unbounded')])
def test_wrong_case_repeated_accepted_pilot_or_reinterpreted_memory_is_refused(monkeypatch, monomer, profile):
    args, _ = fixture(monkeypatch)
    with pytest.raises(ValueError):
        case.prepare_case(**args, monomer_id=monomer, profile=profile)


@pytest.mark.parametrize('change', ['bank_bytes', 'allocation_bytes', 'missing_child', 'changed_child'])
def test_actual_descriptor_bank_is_not_replaced_by_a_forecast_or_partial_receipt(monkeypatch, change):
    args, _ = fixture(monkeypatch)
    if change == 'bank_bytes':
        args['bank_raw'] += b' '
    elif change == 'allocation_bytes':
        args['allocation_raw'] += b' '
    elif change == 'missing_child':
        args['reports'].pop(next(iter(args['reports'])))
    else:
        args['reports'][next(iter(args['reports']))] += b' '
    with pytest.raises(ValueError):
        case.prepare_case(**args, monomer_id='monomer-0047', profile='s40')


def test_profile_choices_are_finite_embers_and_leave_at_least_35percent_allocation_headroom():
    for profile in case.PROFILES.values():
        assert profile['science_memory_GiB']*100 <= profile['job_memory_GiB']*65
        assert profile['cores'] in (8, 12, 24)
        assert profile['time_limit'] in ('02:00:00', '04:00:00', '08:00:00')
    with pytest.raises(ValueError, match='closed'):
        case.helper('arbitrary_loader')


@pytest.mark.parametrize('change', ['none', 'geometry', 'request', 'producer', 'profile', 'case_id'])
def test_stored_case_must_equal_fresh_closed_receipt_reconstruction(monkeypatch, tmp_path, change):
    args, _ = fixture(monkeypatch)
    prepared = case.prepare_case(**args, monomer_id='monomer-0047', profile='s40')
    expected = copy.deepcopy(prepared)
    calls = []
    def reconstruct(*paths, **selection):
        calls.append(selection)
        return copy.deepcopy(expected)
    monkeypatch.setattr(case, 'prepare_from_paths', reconstruct)
    if change == 'geometry':
        prepared['entry']['molecule']['geometry'][0] = .2
    elif change == 'request':
        prepared['entry']['additional_keywords']['function_kwargs']['native_monomer']['schema'] = 'request2'
    elif change == 'producer':
        prepared['producer']['scientific_commit'] = 'foreign'
    elif change == 'profile':
        prepared['execution_profile']['cores'] = 24
    elif change == 'case_id':
        prepared['case_id'] = 'foreign'
    path = tmp_path/'synthetic-case.json'
    path.write_bytes(encode(prepared))
    if change == 'none':
        assert case.verify_prepared_from_paths(path, None, None, None, None) == expected
    else:
        with pytest.raises(ValueError, match='pinned-input reconstruction'):
            case.verify_prepared_from_paths(path, None, None, None, None)
    assert calls == [dict(monomer_id='monomer-0047', profile='s40')]
