"""Synthetic owned354 preservation guards; never real records or server calls."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'owned354_fixture', Path(__file__).with_name('owned_dataset_cb63305.py'))
owned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owned)


def fixture():
    molecule = dict(symbols=['N', 'H', 'H', 'C', 'H', 'H', 'H'],
        geometry=[v for i in range(7) for v in (float(i*3), 0., 0.)],
        molecular_charge=0, molecular_multiplicity=1, fix_com=True, fix_orientation=True)
    entry = dict(name=owned.FIRST_ENTRY, molecule=molecule,
        attributes=dict(monomer_id='monomer-0006', geometry_sha256='first-geometry-unit-only'),
        additional_keywords={'function_kwargs': {'native_monomer': {'schema': 'native-monomer-request/3'}}})
    accepted = dict(producer={'SYNTHETIC': 'not scientific evidence'}, entry=entry,
        manager='SYNTHETIC_FIRST_MANAGER', bank_sha256='synthetic-bank',
        accepted_sha256=dict(owned.ACCEPTED_SHA256))
    specification = dict(program='psi4', driver='properties', method='pbe0', basis='aug-cc-pvtz',
        keywords={}, protocols=dict(error_correction={'default_policy': False}))
    snapshot = dict(schema='native-owned-dataset-snapshot/1',
        metadata=dict(id=354, dataset_type='singlepoint', name=owned.DATASET_NAME,
            owner_user=owned.OWNER, provenance=dict(producer=accepted['producer'],
                                                   descriptor_bank_sha256=accepted['bank_sha256'])),
        specifications={owned.SPECIFICATION: {'specification': specification}},
        entries={owned.FIRST_ENTRY: copy.deepcopy(entry)},
        associations={owned.FIRST_ENTRY: {owned.SPECIFICATION: owned.FIRST_RECORD}},
        records={str(owned.FIRST_RECORD): dict(id=owned.FIRST_RECORD, status='complete',
            tensor_sha256=owned.FIRST_TENSOR, native_envelope_sha256=owned.FIRST_CANONICAL_ENVELOPE,
            history=[{'manager_name': accepted['manager']}])})
    return snapshot, accepted, specification


def new_case():
    return dict(schema='native-bank-molecular-case/1', status='PREPARED_NOT_REGISTERED_OR_SUBMITTED',
        monomer_id='monomer-0047', dataset_id=354, dataset_name=owned.DATASET_NAME,
        specification_name=owned.SPECIFICATION,
        producer={'scientific_commit': 'SYNTHETIC_UNIT_ONLY'},
        runtime_input_sha256='synthetic-runtime-only',
        execution_profile={'name': 's40', 'cores': 12, 'science_memory_GiB': 40},
        specification=dict(program='psi4', driver='properties', method='pbe0', basis='aug-cc-pvtz'),
        required_first_pilot_audit_sha256=owned.ACCEPTED_SHA256['audit'],
        submission=dict(find_existing=False, maximum_records=1, maximum_concurrency=1, qos='embers'),
        entry=dict(name='monomer-0047-unit-only', attributes=dict(
            monomer_id='monomer-0047', geometry_sha256='next-geometry-unit-only'),
            molecule=dict(symbols=['C'], geometry=[0., 0., 0.]),
            additional_keywords={'function_kwargs': {'native_monomer': {'schema': 'native-monomer-request/3'}}}))


def test_only_accepted_firstcase_can_seed_an_initial_owned354_checkpoint():
    snapshot, accepted, specification = fixture()
    checkpoint = owned.prepare_initial_checkpoint(snapshot, authenticated_owner=owned.OWNER,
                                                  specification=specification, accepted=accepted)
    assert checkpoint['dataset_id'] == 354 and checkpoint['status'] == 'READ_ONLY_INITIAL_BASELINE'
    assert checkpoint['snapshot_sha256'] == owned.base().digest(snapshot)
    assert 'not mutation authorization alone' in checkpoint['qualification']


@pytest.mark.parametrize('change', ['old353', 'foreign_owner', 'foreign_name', 'provenance',
    'old_spec', 'changed_envelope', 'changed_tensor', 'changed_history', 'waiting'])
def test_old_foreign_or_changed_firstcase_cannot_authorize_append(change):
    snapshot, accepted, specification = fixture()
    if change == 'old353':
        snapshot['metadata']['id'] = 353
    elif change == 'foreign_owner':
        snapshot['metadata']['owner_user'] = 'foreign'
    elif change == 'foreign_name':
        snapshot['metadata']['name'] = 'foreign'
    elif change == 'provenance':
        snapshot['metadata']['provenance'] = {}
    elif change == 'old_spec':
        snapshot['specifications'] = {'pbe0-atz-native-request2': {'specification': specification}}
    else:
        record = snapshot['records'][str(owned.FIRST_RECORD)]
        if change == 'changed_envelope':
            record['native_envelope_sha256'] = 'foreign'
        elif change == 'changed_tensor':
            record['tensor_sha256'] = 'foreign'
        elif change == 'changed_history':
            record['history'].append({'manager_name': 'foreign'})
        elif change == 'waiting':
            record['status'] = 'waiting'
    with pytest.raises(ValueError):
        owned.prepare_initial_checkpoint(snapshot, authenticated_owner=owned.OWNER,
                                          specification=specification, accepted=accepted)


def test_initial_baseline_does_not_bless_extra_completed_but_unaudited_records():
    snapshot, accepted, specification = fixture()
    snapshot['records']['9999'] = dict(status='complete')
    with pytest.raises(ValueError, match='exactly the audited first case'):
        owned.prepare_initial_checkpoint(snapshot, authenticated_owner=owned.OWNER,
                                          specification=specification, accepted=accepted)


def test_new_case_and_exact_request3_delta_preserve_the_existing_firstcase():
    before, _, _ = fixture()
    case = new_case()
    owned.validate_new_case(before, case, reproduced_case=copy.deepcopy(case))
    after = copy.deepcopy(before)
    name = case['entry']['name']
    after['entries'][name] = case['entry']
    after['associations'][name] = {owned.SPECIFICATION: None}
    owned.assert_append_delta(before, after, entry_name=name)
    after['records']['9999'] = dict(id=9999, status='waiting')
    after['associations'][name][owned.SPECIFICATION] = 9999
    owned.assert_append_delta(before, after, entry_name=name, record_id=9999)
    after['records'][str(owned.FIRST_RECORD)]['status'] = 'waiting'
    with pytest.raises(ValueError, match='preexisting record'):
        owned.assert_append_delta(before, after, entry_name=name, record_id=9999)


@pytest.mark.parametrize('change', ['old353', 'existing_monomer', 'existing_geometry',
                                   'old_request2_spec', 'reuse_records', 'missing_audit'])
def test_case_authorization_is_not_implicit(change):
    before, _, _ = fixture()
    case = new_case()
    if change == 'old353':
        case['dataset_id'] = 353
    elif change == 'existing_monomer':
        case['monomer_id'] = case['entry']['attributes']['monomer_id'] = 'monomer-0006'
    elif change == 'existing_geometry':
        case['entry']['attributes']['geometry_sha256'] = before['entries'][owned.FIRST_ENTRY]['attributes']['geometry_sha256']
    elif change == 'old_request2_spec':
        case['specification_name'] = 'pbe0-atz-native-request2'
    elif change == 'reuse_records':
        case['submission']['find_existing'] = True
    else:
        case['required_first_pilot_audit_sha256'] = 'foreign'
    with pytest.raises(ValueError):
        owned.validate_new_case(before, case, reproduced_case=copy.deepcopy(case))


@pytest.mark.parametrize('change', ['geometry', 'request', 'producer', 'runtime_hash', 'profile',
                                   'specification', 'missing_molecule'])
def test_case_labels_cannot_replace_full_pinned_input_reconstruction(change):
    before, _, _ = fixture()
    case = new_case()
    reproduced = copy.deepcopy(case)
    if change == 'geometry':
        case['entry']['molecule']['geometry'][0] = .25
    elif change == 'request':
        case['entry']['additional_keywords']['function_kwargs']['native_monomer']['schema'] = 'native-monomer-request/2'
    elif change == 'producer':
        case['producer']['scientific_commit'] = 'foreign'
    elif change == 'runtime_hash':
        case['runtime_input_sha256'] = 'foreign'
    elif change == 'profile':
        case['execution_profile']['cores'] = 24
    elif change == 'specification':
        case['specification']['basis'] = 'sto-3g'
    else:
        del case['entry']['molecule']
    with pytest.raises(ValueError, match='pinned-input reconstruction'):
        owned.validate_new_case(before, case, reproduced_case=reproduced)


def test_case_requires_explicit_reconstruction_binding():
    before, _, _ = fixture()
    with pytest.raises(TypeError, match='reproduced_case'):
        owned.validate_new_case(before, new_case())


def test_first_acceptance_does_not_accept_missing_or_changed_receipt_bytes():
    with pytest.raises(ValueError, match='roles'):
        owned.verify_first_acceptance({})
    with pytest.raises(ValueError, match='hash differs'):
        owned.verify_first_acceptance({name: b'SYNTHETIC_NOT_A_RECEIPT' for name in owned.ACCEPTED_SHA256})


def test_snapshot_uses_explicit354_and_bounded131_without_retargeting_legacy_globals(monkeypatch):
    seen = []
    core = owned.base()
    monkeypatch.setattr(core, 'snapshot_dataset', lambda client, **kwargs: seen.append(kwargs) or {'unit': True})
    assert owned.snapshot_dataset(object()) == {'unit': True}
    assert seen == [dict(dataset_id=354, max_entries=131)]
    assert core.DATASET_ID == 353 and core.SPECIFICATION == 'pbe0-atz-native-request2'
