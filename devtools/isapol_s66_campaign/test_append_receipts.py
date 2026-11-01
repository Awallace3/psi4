"""Cross-wiring guard fixtures; neither molecular nor installed qualification."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('append_receipts', Path(__file__).with_name('append_receipts.py'))
receipts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipts)


def fixtures(monkeypatch):
    guard = receipts.helper('dataset_guard')
    draft = dict(monomer_id='monomer-0004', build_identity_sha256='build',
                 source_files_sha256={'installed_qualification': 'qualification'},
                 specification=dict(driver='properties', program='psi4', method='pbe0',
                                    basis='aug-cc-pvtz', keywords={}),
                 entry=dict(name='new-methanol', molecule={'symbols': ['C', 'O', 'H', 'H', 'H', 'H']},
                            additional_keywords={'function_kwargs': {'native_monomer': 'fixture-only'}},
                            attributes={'monomer_id': 'monomer-0004', 'geometry_sha256': 'new'}))
    runtime = receipts.helper('phoenix_worker_probe').runtime_input(draft)
    import hashlib, json
    runtime_sha = hashlib.sha256(json.dumps(runtime, allow_nan=False).encode()).hexdigest()
    water = {'implementation': {'sha256': {'core': 'core', 'isapol_python': 'python'}}}
    audit = dict(status='PASS_SINGLE_PILOT_EVIDENCE', record_id=5234314, runtime_unchanged=True,
                 scientific_build_commit=guard.SCIENTIFIC_COMMIT)
    descriptor = dict(status='PASS_DESCRIPTOR_PLAN_ONLY', coordinate_change_bohr=0,
                      runtime_input_sha256=runtime_sha, implementation=water['implementation'],
                      plan={'status': 'WITHIN_PREDICTED_LEDGER_LIMITS', 'refusals': []})
    allocation = dict(status='PASS_DESCRIPTOR_PLANS_ONLY', runtime_unchanged=True,
                      scientific_build_commit=guard.SCIENTIFIC_COMMIT,
                      operational_commit=receipts.DESCRIPTOR_COMMIT,
                      build_identity_sha256='build', qualification_sha256='qualification',
                      allocation={'JobId': '14953119'},
                      reports=[dict(monomer_id='monomer-0004',
                                    pilot_sha256=receipts.METHANOL_PILOT_SHA256,
                                    descriptor_report_sha256=receipts.METHANOL_DESCRIPTOR_SHA256,
                                    runtime_input_sha256=runtime_sha)])
    expected_spec = copy.deepcopy(draft['specification'])
    expected_spec['protocols'] = {'error_correction': {'default_policy': False}}
    snapshot = dict(
        metadata=dict(id=353, name=guard.DATASET_NAME, dataset_type='singlepoint', owner_user='awallace43',
                      provenance=dict(scientific_build_commit=guard.SCIENTIFIC_COMMIT,
                                      implementation_sha256=water['implementation']['sha256'])),
        specifications={guard.SPECIFICATION: {'specification': expected_spec}},
        entries={guard.WATER_ENTRY: {'attributes': {'monomer_id': 'monomer-0001', 'geometry_sha256': 'old'}}},
        associations={guard.WATER_ENTRY: {guard.SPECIFICATION: 5234314}},
        records={'5234314': dict(status='complete', tensor_sha256=guard.WATER_TENSOR,
                                native_envelope_sha256=guard.digest(water),
                                history=[{'manager_name': 'ccasp-native-s66-water-14948991-owned'}])})
    baseline = dict(schema='native-owned-append-baseline/1', next_monomer_id='monomer-0004',
                    previous_audit_sha256=receipts.WATER_AUDIT_SHA256,
                    previous_envelope_sha256=receipts.WATER_ENVELOPE_SHA256,
                    water_envelope_canonical_sha256=guard.digest(water), snapshot=snapshot)
    data = dict(pilot=(draft, receipts.METHANOL_PILOT_SHA256),
                previous_audit=(audit, receipts.WATER_AUDIT_SHA256),
                previous_envelope=(water, receipts.WATER_ENVELOPE_SHA256),
                descriptor_allocation=(allocation, receipts.DESCRIPTOR_ALLOCATION_SHA256),
                descriptor_report=(descriptor, receipts.METHANOL_DESCRIPTOR_SHA256),
                append_baseline=(baseline, 'baseline-hash'))

    def fake_load(path, expected=None):
        value, digest = data[str(path)]
        if expected is not None and digest != expected:
            raise ValueError('fixture byte pin mismatch')
        return copy.deepcopy(value), digest

    monkeypatch.setattr(receipts, 'load', fake_load)
    args = NS(pilot=Path('pilot'), **{name: Path(name) for name in receipts.PATH_ARGUMENTS})
    return args, data


def test_checkpoint_binds_pilot_baseline_prior_audit_and_actual_descriptor(monkeypatch):
    args, _ = fixtures(monkeypatch)
    checkpoint = receipts.verify(args)
    assert checkpoint['dataset_id'] == 353 and checkpoint['monomer_id'] == 'monomer-0004'
    assert checkpoint['descriptor_job'] == '14953119'
    assert checkpoint['baseline_sha256'] == 'baseline-hash'
    assert checkpoint['descriptor_operational_commit'] == receipts.DESCRIPTOR_COMMIT
    assert 'dataset_append.py' in checkpoint['source_files_sha256']


@pytest.mark.parametrize('change', [
    'descriptor-hash', 'runtime', 'pilot', 'build', 'qualification', 'source',
    'refused', 'water-audit', 'water-envelope', 'baseline-owner', 'baseline-records', 'next',
])
def test_cross_wired_or_unqualified_chain_stops_before_server_access(monkeypatch, change):
    args, data = fixtures(monkeypatch)
    if change == 'descriptor-hash':
        data['descriptor_allocation'][0]['reports'][0]['descriptor_report_sha256'] = 'wrong'
    elif change == 'runtime':
        data['descriptor_report'][0]['runtime_input_sha256'] = 'wrong'
    elif change == 'pilot':
        data['pilot'] = (data['pilot'][0], 'wrong')
    elif change in ('build', 'qualification'):
        key = 'build_identity_sha256' if change == 'build' else 'qualification_sha256'
        data['descriptor_allocation'][0][key] = 'wrong'
    elif change == 'source':
        data['descriptor_allocation'][0]['operational_commit'] = 'wrong'
    elif change == 'refused':
        data['descriptor_report'][0]['plan']['refusals'] = ['numeric_bytes']
    elif change == 'water-audit':
        data['previous_audit'][0]['status'] = 'NOT_QUALIFIED'
    elif change == 'water-envelope':
        data['previous_envelope'][0]['implementation']['sha256']['core'] = 'foreign'
    elif change == 'baseline-owner':
        data['append_baseline'][0]['snapshot']['metadata']['owner_user'] = 'other'
    elif change == 'baseline-records':
        data['append_baseline'][0]['snapshot']['records']['unexpected'] = {}
    else:
        data['pilot'][0]['monomer_id'] = 'monomer-0006'
    with pytest.raises(ValueError):
        receipts.verify(args)


def test_archived_water_mode_and_partial_checkpoint_are_distinct():
    assert receipts.verify(NS()) is None
    with pytest.raises(ValueError):
        receipts.verify(NS(append_baseline=Path('only-one')))


def test_receipt_bytes_are_actually_hashed(tmp_path):
    path = tmp_path/'receipt.json'
    path.write_text('{"value": 1}')
    _, sha = receipts.load(path)
    assert receipts.load(path, sha)[0] == {'value': 1}
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError):
        receipts.load(path, sha)
