"""Synthetic receipt commitments only, NOT molecular/numerical/SLURM evidence."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_advance_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


fixtures = load('test_owned_dataset_cb63305')
owned = fixtures.owned


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def initial(monkeypatch):
    snapshot, accepted, specification = fixtures.fixture_ed()
    accepted['producer'] = dict(scientific_commit='SYNTHETIC_SCIENCE_NOT_EVIDENCE',
                               implementation_sha256={'synthetic': 'unit-only'})
    snapshot['metadata']['provenance']['producer'] = copy.deepcopy(accepted['producer'])
    checkpoint = owned.prepare_initial_checkpoint(snapshot, authenticated_owner=owned.OWNER,
        specification=specification, accepted=accepted, producer='ed561666')
    raw = encode(checkpoint)
    # These explicit fake anchors never escape the test or confer qualification.
    monkeypatch.setattr(owned, 'SEED_CHECKPOINT_SHA256', sha(raw))
    monkeypatch.setattr(owned, 'SEED_SNAPSHOT_SHA256', owned.base().digest(snapshot))
    return raw, accepted


def delivered(previous_raw, accepted, *, monomer='monomer-0006', record_id=60001):
    previous = json.loads(previous_raw)
    before = copy.deepcopy(previous['snapshot'])
    case = fixtures.new_case()
    context = owned.context('ed561666')
    case.update(producer_profile='ed561666', dataset_mode='append', dataset_id=355,
        dataset_name=context['dataset_name'], monomer_id=monomer,
        producer=copy.deepcopy(accepted['producer']), runtime_input_sha256=sha(monomer.encode()),
        required_first_pilot_audit_sha256=context['accepted_sha256']['audit'])
    name = monomer+'-synthetic-unit-only'
    case['entry'].update(name=name)
    case['entry']['attributes'].update(monomer_id=monomer, geometry_sha256=sha(name.encode()))
    case['entry']['molecule'].update(molecular_charge=0, molecular_multiplicity=1,
        fix_com=True, fix_orientation=True, geometry=[float(record_id), 0., 0.])
    tensor = b'SYNTHETIC_UNIT_NOT_NUMERIC_NPZ_'+monomer.encode()
    envelope = dict(request=case['entry']['additional_keywords']['function_kwargs']['native_monomer'],
        implementation=dict(sha256=copy.deepcopy(accepted['producer']['implementation_sha256'])),
        artifacts={'tensors.npz': dict(sha256=sha(tensor), size=len(tensor),
            data=base64.b64encode(tensor).decode())})
    job, manager = str(record_id+100), 'SYNTHETIC_MANAGER_'+str(record_id)
    tag = 'ccasp-native-ed561666-'+monomer+'-'+job
    entry = copy.deepcopy(before)
    entry['entries'][name] = copy.deepcopy(case['entry'])
    entry['associations'][name] = {owned.SPECIFICATION: None}
    record = copy.deepcopy(entry)
    specification = copy.deepcopy(before['specifications'][owned.SPECIFICATION]['specification'])
    specification['keywords'].update(case['entry']['additional_keywords'])
    record['associations'][name][owned.SPECIFICATION] = record_id
    record['records'][str(record_id)] = dict(id=record_id, molecule_id=record_id+1, status='waiting',
        history=[], task=dict(compute_tag=tag), specification=specification)
    after = copy.deepcopy(record)
    after['records'][str(record_id)].update(status='complete', task=None,
        history=[dict(manager_name=manager)], tensor_sha256=sha(tensor),
        native_envelope_sha256=owned.base().digest(envelope),
        body_sha256=sha(b'SYNTHETIC_COMPLETED_RECORD_'+monomer.encode()))
    outcome = dict(status='DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION',
        producer_profile='ed561666', dataset_mode='append', dataset_id=355, record_id=record_id,
        monomer_id=monomer, allocation=dict(JobId=job), compute_tag=tag, manager=manager,
        checkpoint_sha256=sha(previous_raw), operational_commit='SYNTHETIC_OPS',
        scientific_build_commit=accepted['producer']['scientific_commit'],
        runtime_unchanged=True, runtime_input_sha256=case['runtime_input_sha256'],
        find_existing=False, maximum_records=1, pilot_sha256=sha(encode(case)),
        tensors_sha256=sha(tensor), tensor_bytes=len(tensor))
    outcome['dispatch_intent_sha256'] = sha(b'SYNTHETIC_INTENT_'+monomer.encode())
    audit = dict(status='PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE',
        producer_profile='ed561666', dataset_mode='append', dataset_id=355, record_id=record_id,
        monomer_id=monomer, case_job=job, case_operational_commit=outcome['operational_commit'],
        scientific_build_commit=accepted['producer']['scientific_commit'], runtime_unchanged=True,
        case_outcome_sha256=sha(encode(outcome)), pilot_sha256=sha(encode(case)),
        file_gate_sha256='b70b0c621d766a16a9dae94d97ce251e70b9f9d5aeeabf949b452f977e8f64ec',
        roles=dict(molecular_qcschema=1), traces={'unit-only': {'status': 'passed'}},
        numeric=dict(status='PASS_NUMERIC_TRANSPORT_ONLY', tensor_sha256=sha(tensor)),
        captured_preservation=dict(status='PASS_CAPTURED_OWNED_DATASET_PRESERVATION',
            checkpoint_sha256=sha(previous_raw), before_sha256=owned.base().digest(before),
            after_sha256=owned.base().digest(after), audited_tensor_sha256=sha(tensor),
            audited_native_envelope_sha256=owned.base().digest(envelope),
            dispatch_intent_sha256=outcome['dispatch_intent_sha256']))
    return dict(receipts={k: encode(v) for k, v in dict(
        audit=audit, outcome=outcome, case=case, envelope=envelope).items()},
        captured=dict(before=before, entry=entry, record=record, delivered=after), tensor_raw=tensor)


def advance(raw, accepted, evidence, proofs=None, expected=None):
    return owned.advance_checkpoint(raw, sha(raw), accepted=accepted, **evidence,
        expected_audit_sha256=expected or sha(evidence['receipts']['audit']), proof_blobs=proofs)


def proof_bytes(*evidence):
    return {sha(raw): raw for item in evidence for raw in item['receipts'].values()}


def test_two_successive_audited_advances_preserve_immutable_prefixes(monkeypatch):
    raw, accepted = initial(monkeypatch)
    first = delivered(raw, accepted)
    second_checkpoint = advance(raw, accepted, first)
    second_raw = encode(second_checkpoint)
    owned.validate_checkpoint_contents(second_checkpoint, accepted, proof_bytes(first))
    second = delivered(second_raw, accepted, monomer='monomer-0001', record_id=60002)
    third = advance(second_raw, accepted, second, proof_bytes(first))
    owned.validate_checkpoint_contents(third, accepted, proof_bytes(first, second))
    assert json.loads(raw)['generation'] == 1
    assert second_checkpoint['generation'] == 2 and third['generation'] == 3
    assert third['ledger'][:1] == second_checkpoint['ledger']
    assert third['audited_records'] == [5234318, 60001, 60002]
    assert third['previous_checkpoint_sha256'] == sha(second_raw)
    assert len(third['snapshot']['entries']) == 3


@pytest.mark.parametrize('change', ['missing_proof', 'changed_proof', 'extra_record',
    'old_record', 'ledger', 'generation', 'audited_records', 'seed_anchor', 'extra_proof'])
def test_offline_continuation_never_blesses_complete_but_unaudited_state(monkeypatch, change):
    raw, accepted = initial(monkeypatch)
    evidence = delivered(raw, accepted)
    checkpoint = advance(raw, accepted, evidence)
    proofs = proof_bytes(evidence)
    if change == 'missing_proof':
        proofs.pop(next(iter(proofs)))
    elif change == 'changed_proof':
        key = next(iter(proofs))
        proofs[key] += b' '
    elif change == 'extra_record':
        checkpoint['snapshot']['records']['60002'] = dict(status='complete')
    elif change == 'old_record':
        checkpoint['snapshot']['records']['5234318']['status'] = 'waiting'
    elif change == 'ledger':
        checkpoint['ledger'][0]['manager'] = 'foreign'
    elif change == 'generation':
        checkpoint['generation'] = 3
    elif change == 'audited_records':
        checkpoint['audited_records'].append(60002)
    elif change == 'seed_anchor':
        checkpoint['seed_snapshot_sha256'] = 'foreign'
    else:
        proofs[sha(b'extra unit-only bytes')] = b'extra unit-only bytes'
    checkpoint['snapshot_sha256'] = owned.base().digest(checkpoint['snapshot'])
    with pytest.raises(ValueError):
        owned.validate_checkpoint_contents(checkpoint, accepted, proofs)


@pytest.mark.parametrize('change', ['audit_status', 'numeric_status', 'failed_trace', 'two_calls',
    'old_dataset', 'wrong_runtime', 'wrong_parent', 'tensor', 'audit_hash', 'old_record',
    'waiting_record', 'two_histories', 'captured_before'])
def test_advancement_requires_new_successful_exact_evidence(monkeypatch, change):
    raw, accepted = initial(monkeypatch)
    evidence = delivered(raw, accepted)
    audit = json.loads(evidence['receipts']['audit'])
    outcome = json.loads(evidence['receipts']['outcome'])
    if change == 'audit_status':
        audit['status'] = 'DELIVERED_ONLY'
    elif change == 'numeric_status':
        audit['numeric']['status'] = 'failed'
    elif change == 'failed_trace':
        audit['traces']['unit-only']['status'] = 'failed'
    elif change == 'two_calls':
        audit['roles']['molecular_qcschema'] = 2
    elif change == 'old_dataset':
        audit['dataset_id'] = 354
    elif change == 'wrong_runtime':
        outcome['runtime_input_sha256'] = 'foreign'
    elif change == 'wrong_parent':
        outcome['checkpoint_sha256'] = 'f'*64
    elif change == 'tensor':
        evidence['tensor_raw'] += b'changed'
    elif change == 'old_record':
        evidence['captured']['delivered']['records']['5234318']['status'] = 'waiting'
    elif change == 'waiting_record':
        evidence['captured']['delivered']['records']['60001']['status'] = 'waiting'
    elif change == 'two_histories':
        evidence['captured']['delivered']['records']['60001']['history'].append({'manager_name': 'foreign'})
    elif change == 'captured_before':
        evidence['captured']['before']['metadata']['name'] = 'foreign'
    evidence['receipts']['outcome'] = encode(outcome)
    audit['case_outcome_sha256'] = sha(evidence['receipts']['outcome'])
    audit['captured_preservation']['after_sha256'] = owned.base().digest(evidence['captured']['delivered'])
    evidence['receipts']['audit'] = encode(audit)
    with pytest.raises(ValueError):
        advance(raw, accepted, evidence, expected='f'*64 if change == 'audit_hash' else None)


def test_content_addressed_proofs_are_required_and_revalidated_before_continuation(monkeypatch, tmp_path):
    raw, accepted = initial(monkeypatch)
    evidence = delivered(raw, accepted)
    checkpoint = advance(raw, accepted, evidence)
    encoded = encode(checkpoint)
    append = load('dataset_owned354_cb63305')
    original = append.helper
    monkeypatch.setattr(append, 'helper', lambda name: owned if name == 'owned_dataset_cb63305'
                        else original(name))
    root = tmp_path/'proofs'
    root.mkdir()
    for digest, blob in proof_bytes(evidence).items():
        (root/(digest+'.json')).write_bytes(blob)
    observed = append.read_checkpoint_proofs(encoded, sha(encoded), root)
    assert observed == proof_bytes(evidence)
    assert append.validate_checkpoint(encoded, sha(encoded), accepted, producer='ed561666',
                                      proof_blobs=observed) == checkpoint
    with pytest.raises(ValueError, match='proof'):
        append.validate_checkpoint(encoded, sha(encoded), accepted, producer='ed561666')
    with pytest.raises(ValueError, match='proof directory'):
        append.read_checkpoint_proofs(encoded, sha(encoded))
    member = root/(next(iter(observed))+'.json')
    member.write_bytes(member.read_bytes()+b' ')
    with pytest.raises(ValueError, match='bytes changed'):
        append.read_checkpoint_proofs(encoded, sha(encoded), root)


def test_proof_reader_rejects_aliases_and_wrong_checkpoint_before_file_reads(monkeypatch, tmp_path):
    raw, accepted = initial(monkeypatch)
    evidence = delivered(raw, accepted)
    checkpoint = advance(raw, accepted, evidence)
    encoded = encode(checkpoint)
    append = load('dataset_owned354_cb63305')
    original = append.helper
    monkeypatch.setattr(append, 'helper', lambda name: owned if name == 'owned_dataset_cb63305'
                        else original(name))
    with pytest.raises(ValueError, match='before proof-path reads'):
        append.read_checkpoint_proofs(encoded+b' ', sha(encoded), tmp_path/'not-opened')
    root = tmp_path/'proofs'
    root.mkdir()
    for digest, blob in proof_bytes(evidence).items():
        (root/(digest+'.json')).write_bytes(blob)
    member = root/(next(iter(proof_bytes(evidence)))+'.json')
    outside = tmp_path/'outside.json'
    outside.write_bytes(member.read_bytes())
    member.unlink()
    member.symlink_to(outside)
    with pytest.raises(ValueError, match='plain'):
        append.read_checkpoint_proofs(encoded, sha(encoded), root)


def legacy_receipts(evidence):
    result = copy.deepcopy(evidence)
    outcome = json.loads(result['receipts']['outcome'])
    outcome.pop('dispatch_intent_sha256')
    outcome['operational_commit'] = owned.LEGACY_INITIAL_OPS
    audit = json.loads(result['receipts']['audit'])
    audit['case_operational_commit'] = owned.LEGACY_INITIAL_OPS
    audit['captured_preservation'].pop('dispatch_intent_sha256')
    result['receipts']['outcome'] = encode(outcome)
    audit['case_outcome_sha256'] = sha(result['receipts']['outcome'])
    result['receipts']['audit'] = encode(audit)
    return result


def test_only_the_frozen_manual_initial_append_can_lack_a_durable_intent(monkeypatch):
    raw, accepted = initial(monkeypatch)
    first = legacy_receipts(delivered(raw, accepted))
    checkpoint = advance(raw, accepted, first)
    assert checkpoint['ledger'][0]['dispatch_intent_sha256'] is None
    second_raw = encode(checkpoint)
    second = legacy_receipts(delivered(second_raw, accepted, monomer='monomer-0001', record_id=60002))
    with pytest.raises(ValueError, match='dispatch intent'):
        advance(second_raw, accepted, second, proof_bytes(first))


def test_proof_store_is_immutable_and_does_not_overwrite_collisions(monkeypatch, tmp_path):
    raw, accepted = initial(monkeypatch)
    evidence = delivered(raw, accepted)
    append = load('dataset_owned354_cb63305')
    monkeypatch.setattr(append, 'PROOF_ROOT', tmp_path/'proofs')
    links = append.store_checkpoint_proofs(evidence['receipts'])
    assert links == {role: sha(blob) for role, blob in evidence['receipts'].items()}
    assert append.store_checkpoint_proofs(evidence['receipts']) == links
    path = append.PROOF_ROOT/(links['audit']+'.json')
    path.write_bytes(b'SYNTHETIC_COLLISION_NOT_VALID_RECEIPT')
    with pytest.raises(ValueError, match='never overwrite'):
        append.store_checkpoint_proofs(evidence['receipts'])
    assert path.read_bytes() == b'SYNTHETIC_COLLISION_NOT_VALID_RECEIPT'
