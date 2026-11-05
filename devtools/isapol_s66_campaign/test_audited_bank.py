"""Synthetic receipt reduction; no solver, QCA, scientific accuracy or attestation."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_audited_bank_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup():
    core = load('audited_bank')
    factory, inputs, pool = load('test_grid400_bank_case').fixture()
    pool['eligible_ids'] = pool['eligible_ids'][:3]
    pool['refused_ids'] = [f'monomer-{i:04d}' for i in range(4, 132)]
    pool['bank']['producer'] = dict(scientific_commit=core.SCIENCE,
        implementation_sha256=dict(core='a'*64, isapol_python='b'*64),
        qualification='SYNTHETIC_NOT_MEASURED')
    entries = json.loads(inputs['entries'])
    for index, entry in enumerate(entries['entries']):
        entry['attributes']['geometry_sha256'] = hashlib.sha256(str(index).encode()).hexdigest()
        pool['allocation']['reports'][index]['status'] = (
            'PASS_DESCRIPTOR_PLAN_ONLY' if index < 3 else 'REFUSED_DESCRIPTOR_PLAN')
    inputs['entries'] = json.dumps(entries).encode()
    pool['bank']['input_sha256']['entries'] = hashlib.sha256(inputs['entries']).hexdigest()
    cases = {mid: factory.prepare_case(inputs, pool, mid) for mid in pool['eligible_ids']}
    return core, pool, cases


def delivered(core, case, before=None):
    """Synthetic complete audit chain and before/entry/waiting/complete captures."""
    base = core.helper('dataset_guard')
    index = int(case['monomer_id'][-4:])
    identifier, job, manager = 10000+index, str(500+index), 'SYNTHETIC_MANAGER_'+str(index)
    name, specification = case['entry']['name'], case['specification_name']
    tensor = b'SYNTHETIC_TENSOR_TRANSPORT_ONLY'
    artifact = dict(size=len(tensor), sha256=core.digest(tensor),
                    data=base64.b64encode(tensor).decode())
    envelope = dict(schema='native-monomer-artifacts/2', artifacts={'tensors.npz': artifact},
        request=case['entry']['additional_keywords']['function_kwargs']['native_monomer'],
        implementation=dict(sha256=case['producer']['implementation_sha256']))
    stored = copy.deepcopy(case['entry'])
    stored['molecule'].update(molecular_charge=0, molecular_multiplicity=1)
    spec = core.normalize_specification(case['specification'])
    if before is None:
        before = dict(schema='native-owned-dataset-snapshot/1',
            metadata=dict(id=9999, dataset_type='singlepoint', name=case['dataset_name'],
                owner_user='awallace43', provenance=dict(producer=case['producer'],
                    descriptor_bank_sha256=case['descriptor_bank_sha256'])),
            specifications={specification: dict(name=specification, specification=spec)},
            entries={}, associations={}, records={})
    after_entry = copy.deepcopy(before)
    after_entry['entries'][name] = stored
    after_entry['associations'][name] = {specification: None}
    after_record = copy.deepcopy(after_entry)
    after_record['associations'][name][specification] = identifier
    effective = copy.deepcopy(spec)
    effective['keywords'].update(case['entry']['additional_keywords'])
    tag = 'ccasp-native-grid400-bank-'+case['monomer_id']+'-'+job
    after_record['records'][str(identifier)] = dict(id=identifier, molecule_id=9000+index,
        status='waiting', history=[], task=dict(compute_tag=tag), specification=effective,
        tensor_sha256=None, native_envelope_sha256=base.digest(None), body_sha256='0'*64)
    after = copy.deepcopy(after_record)
    after['records'][str(identifier)].update(status='complete', task=None,
        history=[dict(manager_name=manager)], tensor_sha256=artifact['sha256'],
        native_envelope_sha256=base.digest(envelope), body_sha256='1'*64)
    case_raw = core.encode(case)
    common = dict(producer_profile='grid400', campaign_scope='eligible-bank',
        dataset_mode=case['dataset_mode'], dataset_id=9999, record_id=identifier,
        monomer_id=case['monomer_id'], scientific_build_commit=core.SCIENCE, runtime_unchanged=True)
    outcome = dict(common, status='DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION',
        allocation=dict(JobId=job), operational_commit='c'*40, manager=manager,
        pilot_sha256=core.digest(case_raw), compute_tag=tag,
        runtime_input_sha256=case['runtime_input_sha256'], find_existing=False, maximum_records=1,
        tensor_bytes=len(tensor), tensors_sha256=artifact['sha256'])
    preservation = None
    if case['dataset_mode'] == 'append':
        outcome.update(checkpoint_sha256=case['checkpoint_sha256'], dispatch_intent_sha256='d'*64)
        preservation = dict(status='PASS_CAPTURED_OWNED_DATASET_PRESERVATION',
            checkpoint_sha256=case['checkpoint_sha256'], dispatch_intent_sha256='d'*64,
            before_sha256=base.digest(before), after_sha256=base.digest(after),
            audited_tensor_sha256=artifact['sha256'], audited_native_envelope_sha256=base.digest(envelope))
    outcome_raw = core.encode(outcome)
    audit = dict(common, status=('PASS_SELECTED_FRESH_SINGLETON_EVIDENCE'
        if case['dataset_mode'] == 'fresh' else 'PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE'),
        case_job=job, case_operational_commit='c'*40, case_outcome_sha256=core.digest(outcome_raw),
        pilot_sha256=core.digest(case_raw), file_gate_sha256=core.FILE_GATE_SHA256,
        roles=dict(molecular_qcschema=1), traces={'unit': dict(status='passed')},
        numeric=dict(status='PASS_NUMERIC_TRANSPORT_ONLY', tensor_sha256=artifact['sha256']),
        envelope_identity=dict(status='PASS_NATIVE_ENVELOPE_IDENTITY', policy='exact-json-diagnostic-number/1',
            local_json_sha256=core.digest(core.encode(envelope)),
            server_json_sha256=core.digest(core.encode(envelope)), epsilon_tolerance_used=False,
            diagnostic_json_number_equivalence=None),
        captured_preservation=preservation)
    receipts = dict(audit=core.encode(audit), outcome=outcome_raw, case=case_raw,
                    envelope=core.encode(envelope))
    captured = dict(before=before, entry=after_entry, record=after_record, delivered=after)
    return receipts, captured, tensor


def initialized():
    core, pool, cases = setup()
    receipts, captured, tensor = delivered(core, cases['monomer-0001'])
    context, checkpoint = core.initialize(pool, cases, receipts, core.digest(receipts['audit']),
                                          captured['delivered'], tensor)
    proofs = {core.digest(raw): raw for raw in receipts.values()}
    return core, pool, cases, context, checkpoint, proofs


def test_fresh_bootstrap_then_exact_successors_finish_only_eligible_members():
    core, pool, cases, context, checkpoint, proofs = initialized()
    context_sha = core.digest(context)
    for mid in ('monomer-0002', 'monomer-0003'):
        raw = core.encode(checkpoint)
        core.validate_checkpoint(raw, core.digest(raw), context, context_sha, proofs)
        case = core.bind_next_case(cases[mid], raw, core.digest(raw), context, context_sha, proofs)
        receipts, captured, tensor = delivered(core, case, checkpoint['snapshot'])
        checkpoint, proofs = core.advance(raw, core.digest(raw), context, context_sha, proofs,
            receipts=receipts, expected_audit_sha256=core.digest(receipts['audit']),
            captured=captured, tensor_raw=tensor)
    assert checkpoint['generation'] == len(pool['eligible_ids']) == 3
    assert checkpoint['audited_records'] == [10001, 10002, 10003]
    assert len(checkpoint['snapshot']['entries']) == 3
    with pytest.raises(ValueError, match='complete'):
        core.bind_next_case(cases['monomer-0003'], core.encode(checkpoint), core.digest(
            core.encode(checkpoint)), context, context_sha, proofs)


@pytest.mark.parametrize('change', ['audit_hash', 'forensic', 'numeric', 'trace', 'old_dataset',
                                   'old_record', 'tensor', 'extra_snapshot'])
def test_failed_forensic_or_unbound_first_receipts_cannot_bootstrap(change):
    core, pool, cases = setup()
    receipts, captured, tensor = delivered(core, cases['monomer-0001'])
    audit = json.loads(receipts['audit'])
    snapshot = captured['delivered']
    if change == 'forensic':
        audit['status'] = 'PASS_FORENSIC_COMPONENT_AUDIT_NOT_ACCEPTANCE'
    elif change == 'numeric':
        audit['numeric']['status'] = 'FAIL'
    elif change == 'trace':
        audit['traces']['unit']['status'] = 'failed'
    elif change == 'old_dataset':
        audit['dataset_id'] = 356
    elif change == 'old_record':
        audit['record_id'] = 5234327
    elif change == 'tensor':
        tensor += b'changed'
    elif change == 'extra_snapshot':
        snapshot['entries']['unaudited'] = {}
    receipts['audit'] = core.encode(audit)
    expected = 'f'*64 if change == 'audit_hash' else core.digest(receipts['audit'])
    with pytest.raises(ValueError):
        core.initialize(pool, cases, receipts, expected, snapshot, tensor)


@pytest.mark.parametrize('change', ['missing_proof', 'proof_bytes', 'old_snapshot',
                                   'extra_record', 'context', 'generation'])
def test_checkpoint_cannot_bless_changed_proofs_or_snapshot(change):
    core, _, _, context, checkpoint, proofs = initialized()
    context_sha = core.digest(context)
    if change == 'missing_proof':
        proofs.pop(next(iter(proofs)))
    elif change == 'proof_bytes':
        key = next(iter(proofs))
        proofs[key] += b' '
    elif change == 'old_snapshot':
        checkpoint['snapshot']['records']['10001']['history'] = []
    elif change == 'extra_record':
        checkpoint['snapshot']['records']['99999'] = {}
    elif change == 'context':
        context += b' '
    else:
        checkpoint['generation'] = 2
    raw = core.encode(checkpoint)
    with pytest.raises(ValueError):
        core.validate_checkpoint(raw, core.digest(raw), context, context_sha, proofs)


@pytest.mark.parametrize('change', ['preservation', 'intent', 'before', 'tensor', 'case_metadata'])
def test_bad_append_receipts_or_captures_cannot_advance(change):
    core, _, cases, context, checkpoint, proofs = initialized()
    raw, context_sha = core.encode(checkpoint), core.digest(context)
    case = core.bind_next_case(cases['monomer-0002'], raw, core.digest(raw), context, context_sha, proofs)
    receipts, captured, tensor = delivered(core, case, checkpoint['snapshot'])
    audit = json.loads(receipts['audit'])
    if change == 'preservation':
        audit['captured_preservation']['after_sha256'] = 'f'*64
    elif change == 'intent':
        audit['captured_preservation']['dispatch_intent_sha256'] = None
    elif change == 'before':
        captured['before']['metadata']['name'] = 'changed'
    elif change == 'tensor':
        tensor += b'changed'
    else:
        case['entry']['attributes']['reference_energy'] = 'changed'
        receipts['case'] = core.encode(case)
    receipts['audit'] = core.encode(audit)
    with pytest.raises(ValueError):
        core.advance(raw, core.digest(raw), context, context_sha, proofs, receipts=receipts,
            expected_audit_sha256=core.digest(receipts['audit']), captured=captured, tensor_raw=tensor)


def test_fixed_legitimate_audit_cannot_bless_a_changed_bootstrap_envelope():
    core, pool, cases = setup()
    receipts, captured, tensor = delivered(core, cases['monomer-0001'])
    audit_hash = core.digest(receipts['audit'])
    envelope = json.loads(receipts['envelope'])
    envelope['result'] = dict(changed_diagnostic=1.)
    receipts['envelope'] = core.encode(envelope)
    captured['delivered']['records']['10001']['native_envelope_sha256'] = (
        core.helper('dataset_guard').digest(envelope))
    assert core.digest(receipts['audit']) == audit_hash
    with pytest.raises(ValueError, match='artifact chain'):
        core.initialize(pool, cases, receipts, audit_hash, captured['delivered'], tensor)


@pytest.mark.parametrize('change', ['method', 'basis', 'driver', 'keywords', 'record_specification',
                                  'association', 'top_level', 'schema'])
def test_wrong_specification_or_snapshot_structure_cannot_be_a_bootstrap_anchor(change):
    core, pool, cases = setup()
    receipts, captured, tensor = delivered(core, cases['monomer-0001'])
    snapshot = captured['delivered']
    spec = snapshot['specifications'][cases['monomer-0001']['specification_name']]['specification']
    if change in ('method', 'basis', 'driver'):
        spec[change] = 'changed'
    elif change == 'keywords':
        spec['keywords']['reference'] = 'uhf'
    elif change == 'record_specification':
        snapshot['records']['10001']['specification']['method'] = 'changed'
    elif change == 'association':
        snapshot['associations']['unrelated'] = {}
    elif change == 'top_level':
        snapshot['extra'] = 'not a snapshot'
    else:
        snapshot['schema'] = 'foreign'
    with pytest.raises(ValueError):
        core.initialize(pool, cases, receipts, core.digest(receipts['audit']), snapshot, tensor)
