"""Closed owned354/355 identities, accepted state and audited append guards.

Never authorize dataset353 or reinterpret its historical append checkpoint.
Later owned355 checkpoints require actual recorded audit-proof bytes, not just
completed records. Ordinary-OS receipts and snapshots are not host attestation
or an atomic server lock.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import re

DATASET_ID = 354
DATASET_NAME = 'S66x8-native-CCASP-cb63305f41-request3-v1'
SPECIFICATION = 'pbe0-atz-native-request3'
OWNER = 'awallace43'
FIRST_ENTRY = 'monomer-0006-b1411a2d1677'
FIRST_RECORD = 5234316
FIRST_TENSOR = '1bdb8fcc0f40565a093b98240ccc722f4ffa373c38285868287edf9404e50e79'
FIRST_CANONICAL_ENVELOPE = '1186a34fe9d511821284b0ebf7792378e6a6c427d7a7a2b384cd0bcd51d4ebda'
ACCEPTED_SHA256 = {
    'audit': '1e4e4d638200dfecfec76fa11d7b480138386346d3cfac61af32d1f5eede033d',
    'outcome': '5cd351c46582a8221ee84fbe4e3303d8c2cacc1150d1e2bcc902d9917fa0eb45',
    'pilot': '229fba9a456eff50804761bbba9c0d74c14c66929e8ab7b56e923741eb0e133a',
    'envelope': '52c47332ece5abde60125cd43ad32b534052827ebba8f68501461c0bbf0675d1',
}
_BASE = None
SEED_CHECKPOINT_SHA256 = '5a033c70462ba409f4ad72b0a4cef9cf8b11494a40ba46041dded70c8846b294'
SEED_SNAPSHOT_SHA256 = '03eccb9f147d155f4f3a9d3ee7562ec63fa898ddef7cbc0595fb36530febe5e5'
PROOF_ROLES = frozenset(('audit', 'outcome', 'case', 'envelope'))
MAX_PROOF_BYTES = 512*1024**2
LEGACY_INITIAL_OPS = '7438d9b04ea87423080f5ce75e7a81862bccd8eb'


def context(producer='cb63305'):
    """Closed owned-dataset identities; never caller-supplied target IDs."""
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed owned-dataset producer required')
    if producer == 'cb63305':
        return dict(producer=producer, dataset_id=DATASET_ID, dataset_name=DATASET_NAME,
            first_entry=FIRST_ENTRY, first_record=FIRST_RECORD, first_monomer='monomer-0006',
            first_tensor=FIRST_TENSOR, first_canonical_envelope=FIRST_CANONICAL_ENVELOPE,
            accepted_sha256=dict(ACCEPTED_SHA256))
    return dict(producer=producer, dataset_id=355,
        dataset_name='S66x8-native-CCASP-ed561666fc-request3-v1',
        first_entry='monomer-0047-4d362ebb54d7', first_record=5234318, first_monomer='monomer-0047',
        first_tensor='c60c5f9c2635c87d65720e504e4a28e842e97380a70e3291aa2779e3f6661cb6',
        accepted_sha256=dict(
            audit='f07ec8ca5a4f2f5788655520e66a12436bca9cfa5782c7f5c9fbb7a48a526f53',
            outcome='727fc45c9f6ea57b959c745a85ff3dafdac7b3a6643635f712ce4e6bf814abca',
            pilot='547a7f0126d0f020e9b5c396698ca400d0c79bc9a2f04ac83d42b39930ccf1ef',
            envelope='35296450423f5b1fe5450c961623b34708d2bfc41ee1901960dff2a1729c2c37'))


def base():
    global _BASE
    if _BASE is None:
        path = Path(__file__).with_name('dataset_guard.py')
        spec = importlib.util.spec_from_file_location('_owned354_snapshot_core', path)
        _BASE = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec'), _BASE.__dict__)
    return _BASE


def verify_first_acceptance(blobs, *, producer='cb63305'):
    import hashlib
    selected = context(producer)
    if producer == 'ed561666':
        expected = selected['accepted_sha256']
        if set(blobs) != set(expected):
            raise ValueError('exact accepted first-case receipt roles required')
        for name, sha in expected.items():
            if type(blobs[name]) is not bytes or hashlib.sha256(blobs[name]).hexdigest() != sha:
                raise ValueError('accepted benzene receipt hash differs: '+name)
        audit, outcome, pilot, envelope = (json.loads(blobs[k]) for k in ('audit', 'outcome', 'pilot', 'envelope'))
        if (audit.get('status') != 'PASS_SELECTED_FRESH_SINGLETON_EVIDENCE'
                or audit.get('producer_profile') != producer or audit.get('dataset_mode') != 'fresh'
                or audit.get('dataset_id') != selected['dataset_id']
                or audit.get('record_id') != selected['first_record']
                or audit.get('runtime_unchanged') is not True
                or audit['case_outcome_sha256'] != expected['outcome']
                or audit['pilot_sha256'] != expected['pilot']
                or audit['scientific_build_commit'] != 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
                or audit['roles'].get('molecular_qcschema') != 1
                or not audit['traces'] or any(t['status'] != 'passed' for t in audit['traces'].values())
                or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
                or outcome.get('dataset_id') != selected['dataset_id']
                or outcome.get('record_id') != selected['first_record']
                or outcome.get('pilot_sha256') != expected['pilot']
                or pilot.get('monomer_id') != selected['first_monomer']
                or pilot['entry']['name'] != selected['first_entry']
                or envelope['artifacts']['tensors.npz']['sha256'] != selected['first_tensor']):
            raise ValueError('accepted cap32 first-case producer/artifact chain differs')
        return dict(producer=pilot['producer'], entry=pilot['entry'], manager=outcome['manager'],
            bank_sha256=pilot['descriptor_bank_sha256'], accepted_sha256=dict(expected),
            canonical_envelope_sha256=base().digest(envelope))
    if set(blobs) != set(ACCEPTED_SHA256):
        raise ValueError('exact accepted first-case receipt roles required')
    for name, expected in ACCEPTED_SHA256.items():
        if type(blobs[name]) is not bytes or hashlib.sha256(blobs[name]).hexdigest() != expected:
            raise ValueError('accepted methylamine receipt hash differs: '+name)
    audit, outcome, pilot, envelope = (json.loads(blobs[k]) for k in ('audit', 'outcome', 'pilot', 'envelope'))
    if (audit.get('status') != 'PASS_EXPANDED_SINGLE_PILOT_EVIDENCE'
            or audit.get('dataset_id') != DATASET_ID or audit.get('record_id') != FIRST_RECORD
            or audit.get('runtime_unchanged') is not True
            or audit['case_outcome_sha256'] != ACCEPTED_SHA256['outcome']
            or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
            or outcome.get('record_id') != FIRST_RECORD or outcome.get('dataset_id') != DATASET_ID
            or outcome.get('pilot_sha256') != ACCEPTED_SHA256['pilot']
            or pilot.get('monomer_id') != 'monomer-0006' or pilot['entry']['name'] != FIRST_ENTRY
            or base().digest(envelope) != FIRST_CANONICAL_ENVELOPE
            or envelope['artifacts']['tensors.npz']['sha256'] != FIRST_TENSOR):
        raise ValueError('accepted first-case producer/artifact chain differs')
    return dict(producer=pilot['producer'], entry=pilot['entry'], manager=outcome['manager'],
                bank_sha256=pilot['descriptor_bank_sha256'], accepted_sha256=dict(ACCEPTED_SHA256))


def snapshot_dataset(client, *, producer='cb63305'):
    return base().snapshot_dataset(client, dataset_id=context(producer)['dataset_id'], max_entries=131)


def validate_owned354(snapshot, *, authenticated_owner, specification, accepted):
    return validate_owned_dataset(snapshot, authenticated_owner=authenticated_owner,
                                  specification=specification, accepted=accepted)


def validate_owned_dataset(snapshot, *, authenticated_owner, specification, accepted,
                           producer='cb63305'):
    selected = context(producer)
    dataset_id, first_record = selected['dataset_id'], selected['first_record']
    first_entry, first_tensor = selected['first_entry'], selected['first_tensor']
    canonical = (selected['first_canonical_envelope'] if producer == 'cb63305'
                 else accepted['canonical_envelope_sha256'])
    metadata = snapshot['metadata']
    owner = metadata.get('owner_user', metadata.get('creator_user'))
    if (authenticated_owner != OWNER or owner != OWNER or metadata.get('id') != dataset_id
            or metadata.get('dataset_type') != 'singlepoint' or metadata.get('name') != selected['dataset_name']
            or metadata.get('provenance') != dict(producer=accepted['producer'],
                                                descriptor_bank_sha256=accepted['bank_sha256'])):
        raise ValueError('owned354 identity, owner or scientific provenance differs')
    if set(snapshot['specifications']) != {SPECIFICATION}:
        raise ValueError('one existing immutable request3 specification required')
    stored = snapshot['specifications'][SPECIFICATION]['specification']
    if (stored != base().plain(specification)
            or stored.get('protocols', {}).get('error_correction', {}).get('default_policy') is not False):
        raise ValueError('existing request3 specification/no-correction policy differs')
    if snapshot['associations'].get(first_entry, {}).get(SPECIFICATION) != first_record:
        raise ValueError('accepted first-case association changed')
    record = snapshot['records'].get(str(first_record), {})
    history = record.get('history', [])
    if (record.get('status') != 'complete' or len(history) != 1
            or history[0].get('manager_name') != accepted['manager']
            or record.get('tensor_sha256') != first_tensor
            or record.get('native_envelope_sha256') != canonical):
        raise ValueError('accepted first-case record/history/artifacts changed')
    base().validate_stored_entry(snapshot['entries'][first_entry], accepted['entry'])


def validate_new_case(before, case, *, reproduced_case, producer='cb63305'):
    """Require the full result of case.verify_prepared_from_paths, not case labels.

    The caller reconstructs from the closed bank/receipt bytes immediately before
    this check. This pure guard does not confer trust on arbitrary supplied dicts.
    """
    selected = context(producer)
    if base().digest(case) != base().digest(reproduced_case):
        raise ValueError('entire case must match pinned-input reconstruction')
    if (case.get('schema') != 'native-bank-molecular-case/1'
            or case.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
            or case.get('dataset_id') != selected['dataset_id'] or case.get('dataset_name') != selected['dataset_name']
            or case.get('specification_name') != SPECIFICATION
            or case.get('monomer_id') == selected['first_monomer']
            or case.get('required_first_pilot_audit_sha256') != selected['accepted_sha256']['audit']
            or case.get('submission') != dict(find_existing=False, maximum_records=1,
                                             maximum_concurrency=1, qos='embers')):
        raise ValueError('one admitted remaining-case checkpoint required')
    if producer == 'ed561666' and (
            case.get('producer_profile') != producer or case.get('dataset_mode') != 'append'):
        raise ValueError('explicit reconstructed cap32 append mode required')
    entry = case['entry']
    if entry['attributes']['monomer_id'] != case['monomer_id'] or entry['name'] in before['entries']:
        raise ValueError('new monomer/entry identity must be absent')
    for previous in before['entries'].values():
        attrs = previous['attributes']
        if (attrs.get('monomer_id') == case['monomer_id']
                or attrs.get('geometry_sha256') == entry['attributes']['geometry_sha256']):
            raise ValueError('monomer/geometry already present; no reuse or repair')
    if any(record['status'] != 'complete' for record in before['records'].values()):
        raise ValueError('all prior owned records must be complete and preserved')


def assert_append_delta(before, after, *, entry_name, record_id=None):
    base().assert_append_delta(before, after, entry_name=entry_name, record_id=record_id,
                              specification_name=SPECIFICATION)


def prepare_initial_checkpoint(snapshot, *, authenticated_owner, specification, accepted,
                               producer='cb63305'):
    """Only the verified singleton is an initial baseline; never bless extras."""
    selected = context(producer)
    validate_owned_dataset(snapshot, authenticated_owner=authenticated_owner,
                           specification=specification, accepted=accepted, producer=producer)
    if (set(snapshot['entries']) != {selected['first_entry']}
            or set(snapshot['records']) != {str(selected['first_record'])}):
        raise ValueError('initial owned354 baseline must contain exactly the audited first case')
    result = dict(schema='native-owned354-checkpoint/1', status='READ_ONLY_INITIAL_BASELINE',
        dataset_id=selected['dataset_id'], accepted_receipt_sha256=accepted['accepted_sha256'],
        snapshot=snapshot, snapshot_sha256=base().digest(snapshot),
        qualification='Read-only accepted-first-case baseline, not mutation authorization alone '
                      'or an atomic server lock. The live before-state must match exactly.')
    if producer == 'ed561666':
        result.update(schema='native-owned-dataset-checkpoint/2', producer_profile=producer,
                      generation=1, previous_checkpoint_sha256=None,
                      audited_records=[selected['first_record']])
    return result


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_string(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _prefix(snapshot, entries, records):
    if set(snapshot) != {'schema', 'metadata', 'specifications', 'entries', 'associations', 'records'}:
        raise ValueError('exact captured snapshot structure required')
    try:
        return dict(snapshot, entries={name: snapshot['entries'][name] for name in entries},
            associations={name: snapshot['associations'][name] for name in entries},
            records={str(identifier): snapshot['records'][str(identifier)] for identifier in records})
    except KeyError as error:
        raise ValueError('audited snapshot member is missing') from error


def _case_acceptance(receipts, accepted, before, after, previous_sha256):
    """Reduce previously audited bytes; this does not rerun a numerical audit.

    The expected audit hash comes from the recorded, approved audit execution.
    Ordinary-OS receipt trust is required: hashes are bindings, not attestation.
    """
    if (set(receipts) != PROOF_ROLES or any(type(raw) is not bytes for raw in receipts.values())
            or not _sha_string(previous_sha256)):
        raise ValueError('exact audited case receipt bytes and predecessor hash required')
    audit, outcome, case, envelope = (json.loads(receipts[k]) for k in ('audit', 'outcome', 'case', 'envelope'))
    selected = context('ed561666')
    identifier, name, monomer = audit['record_id'], case['entry']['name'], case['monomer_id']
    job = outcome['allocation']['JobId']
    tag = 'ccasp-native-ed561666-'+monomer+'-'+job
    preservation, numeric = audit['captured_preservation'], audit['numeric']
    artifact = envelope['artifacts']['tensors.npz']
    intent_sha = outcome.get('dispatch_intent_sha256')
    if intent_sha is None:
        # The already reviewed/manual first append predates durable intents.
        # It cannot authorize a later transition or silently relax new jobs.
        if (outcome.get('operational_commit') != LEGACY_INITIAL_OPS
                or base().digest(before) != SEED_SNAPSHOT_SHA256
                or preservation.get('dispatch_intent_sha256') is not None):
            raise ValueError('audited one-writer dispatch intent required after the legacy initial append')
    elif not _sha_string(intent_sha) or preservation.get('dispatch_intent_sha256') != intent_sha:
        raise ValueError('audited dispatch-intent commitment differs')
    links = {role: _sha(raw) for role, raw in receipts.items()}
    if (type(identifier) is not int or identifier <= 0 or identifier in range(5234314, 5234319)
            or type(monomer) is not str or re.fullmatch(r'monomer-[0-9]{4}', monomer) is None
            or not 1 <= int(monomer[-4:]) <= 131 or monomer == selected['first_monomer']
            or type(job) is not str or re.fullmatch(r'[1-9][0-9]*', job) is None
            or audit.get('status') != 'PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE'
            or any(item.get('producer_profile') != 'ed561666' or item.get('dataset_mode') != 'append'
                   for item in (audit, outcome, case))
            or any(item.get('dataset_id') != selected['dataset_id'] for item in (audit, outcome, case))
            or outcome.get('record_id') != identifier or audit.get('monomer_id') != monomer
            or outcome.get('monomer_id') != monomer or audit.get('case_job') != job
            or audit.get('case_outcome_sha256') != links['outcome']
            or audit.get('pilot_sha256') != links['case'] or outcome.get('pilot_sha256') != links['case']
            or audit.get('case_operational_commit') != outcome.get('operational_commit')
            or audit.get('scientific_build_commit') != accepted['producer']['scientific_commit']
            or outcome.get('scientific_build_commit') != accepted['producer']['scientific_commit']
            or case.get('producer') != accepted['producer']
            or case.get('dataset_name') != selected['dataset_name']
            or case.get('specification_name') != SPECIFICATION
            or case.get('required_first_pilot_audit_sha256') != selected['accepted_sha256']['audit']
            or audit.get('runtime_unchanged') is not True or outcome.get('runtime_unchanged') is not True
            or outcome.get('find_existing') is not False or outcome.get('maximum_records') != 1
            or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
            or outcome.get('runtime_input_sha256') != case.get('runtime_input_sha256')
            or outcome.get('compute_tag') != tag or outcome.get('checkpoint_sha256') != previous_sha256
            or preservation.get('status') != 'PASS_CAPTURED_OWNED_DATASET_PRESERVATION'
            or preservation.get('checkpoint_sha256') != previous_sha256
            or preservation.get('before_sha256') != base().digest(before)
            or preservation.get('after_sha256') != base().digest(after)
            or audit.get('file_gate_sha256') != 'b70b0c621d766a16a9dae94d97ce251e70b9f9d5aeeabf949b452f977e8f64ec'
            or audit.get('roles', {}).get('molecular_qcschema') != 1
            or not audit.get('traces') or any(t.get('status') != 'passed' for t in audit['traces'].values())
            or numeric.get('status') != 'PASS_NUMERIC_TRANSPORT_ONLY'
            or envelope.get('request') != case['entry']['additional_keywords']['function_kwargs']['native_monomer']
            or envelope.get('implementation', {}).get('sha256') != accepted['producer']['implementation_sha256']
            or type(artifact.get('size')) is not int or not 0 < artifact['size'] <= 16*1024**2
            or outcome.get('tensor_bytes') != artifact['size']
            or not _sha_string(artifact.get('sha256'))
            or outcome.get('tensors_sha256') != artifact['sha256']
            or numeric.get('tensor_sha256') != artifact['sha256']
            or preservation.get('audited_tensor_sha256') != artifact['sha256']
            or preservation.get('audited_native_envelope_sha256') != base().digest(envelope)):
        raise ValueError('independently audited cap32 case/provenance/artifact chain differs')
    validate_new_case(before, case, reproduced_case=case, producer='ed561666')
    assert_append_delta(before, after, entry_name=name, record_id=identifier)
    base().validate_stored_entry(after['entries'][name], case['entry'])
    record = after['records'][str(identifier)]
    if (record.get('status') != 'complete' or len(record.get('history', [])) != 1
            or record['history'][0].get('manager_name') != outcome.get('manager')
            or record.get('native_envelope_sha256') != base().digest(envelope)
            or record.get('tensor_sha256') != artifact['sha256']):
        raise ValueError('audited completed record/history/artifacts changed')
    return dict(monomer_id=monomer, entry_name=name, record_id=identifier,
        manager=outcome['manager'], compute_tag=tag, previous_checkpoint_sha256=previous_sha256,
        before_snapshot_sha256=base().digest(before), after_snapshot_sha256=base().digest(after),
        completed_record_sha256=base().digest(record), receipts_sha256=links,
        native_envelope_sha256=base().digest(envelope), tensor_sha256=artifact['sha256'],
        tensor_bytes=artifact['size'], dispatch_intent_sha256=intent_sha)


def checkpoint_proof_hashes(checkpoint):
    """Bounded content-addressed receipt inventory; no paths are accepted."""
    ledger = checkpoint.get('ledger', [])
    if type(ledger) is not list or len(ledger) > 130:
        raise ValueError('bounded audited append ledger required')
    result = set()
    for link in ledger:
        hashes = link.get('receipts_sha256', {})
        if set(hashes) != PROOF_ROLES or any(not _sha_string(sha) for sha in hashes.values()):
            raise ValueError('exact content-addressed audit receipt roles required')
        result.update(hashes.values())
    return result


def validate_checkpoint_contents(checkpoint, accepted, proof_blobs=None):
    """Validate all non-seed acceptance proofs against final-snapshot prefixes.

    Old tensor/trace files are not re-read. Their commitments are checked against
    the actual recorded audit receipts and current completed-record fingerprints.
    Historical raw parent hashes remain recorded audit commitments; only an
    advance with the actual parent bytes proves that immediate raw-byte link.
    """
    selected = context('ed561666')
    proofs = {} if proof_blobs is None else proof_blobs
    required = checkpoint_proof_hashes(checkpoint)
    if (type(proofs) is not dict or set(proofs) != required
            or any(type(raw) is not bytes or len(raw) > 32*1024**2 or _sha(raw) != sha
                   for sha, raw in proofs.items())
            or sum(len(raw) for raw in proofs.values()) > MAX_PROOF_BYTES):
        raise ValueError('complete exact bounded audit-proof bytes required')
    snapshot = checkpoint['snapshot']
    generation = checkpoint.get('generation')
    if (checkpoint.get('schema') != 'native-owned-dataset-checkpoint/2'
            or checkpoint.get('producer_profile') != 'ed561666'
            or checkpoint.get('dataset_id') != selected['dataset_id']
            or checkpoint.get('accepted_receipt_sha256') != selected['accepted_sha256']
            or accepted['accepted_sha256'] != selected['accepted_sha256']
            or type(generation) is not int or not 1 <= generation <= 131
            or checkpoint.get('snapshot_sha256') != base().digest(snapshot)):
        raise ValueError('owned355 audited checkpoint identity/bytes differ')
    entries, records = [selected['first_entry']], [selected['first_record']]
    prefix = _prefix(snapshot, entries, records)
    if base().digest(prefix) != SEED_SNAPSHOT_SHA256:
        raise ValueError('coded audited benzene snapshot anchor changed')
    ledger = checkpoint.get('ledger', [])
    if generation == 1:
        if (ledger or checkpoint.get('status') != 'READ_ONLY_INITIAL_BASELINE'
                or checkpoint.get('previous_checkpoint_sha256') is not None):
            raise ValueError('exact initial audited checkpoint required')
    else:
        if (checkpoint.get('status') != 'READ_ONLY_AUDITED_ADVANCE' or len(ledger) != generation-1
                or checkpoint.get('seed_checkpoint_sha256') != SEED_CHECKPOINT_SHA256
                or checkpoint.get('seed_snapshot_sha256') != SEED_SNAPSHOT_SHA256
                or ledger[0].get('previous_checkpoint_sha256') != SEED_CHECKPOINT_SHA256
                or checkpoint.get('previous_checkpoint_sha256') != ledger[-1].get('previous_checkpoint_sha256')):
            raise ValueError('monotonic audited checkpoint ledger required')
        for link in ledger:
            name, identifier = link['entry_name'], link['record_id']
            if name in entries or identifier in records:
                raise ValueError('one unique new audited entry/record per transition required')
            after = _prefix(snapshot, entries+[name], records+[identifier])
            receipts = {role: proofs[sha] for role, sha in link['receipts_sha256'].items()}
            observed = _case_acceptance(receipts, accepted, prefix, after,
                                        link['previous_checkpoint_sha256'])
            if base().digest(observed) != base().digest(link):
                raise ValueError('audited ledger transition changed')
            entries.append(name)
            records.append(identifier)
            prefix = after
    if (checkpoint.get('audited_records') != records or len(records) != generation
            or set(snapshot['entries']) != set(entries)
            or set(snapshot['associations']) != set(entries)
            or set(snapshot['records']) != {str(identifier) for identifier in records}):
        raise ValueError('checkpoint cannot bless unaudited or unassociated extras')
    return checkpoint


def advance_checkpoint(previous_raw, previous_sha256, *, accepted, receipts,
                       expected_audit_sha256, captured, tensor_raw, proof_blobs=None):
    """Pure one-step acceptance reducer; never write, repair or contact QCArchive."""
    import base64
    import copy
    if type(previous_raw) is not bytes or _sha(previous_raw) != previous_sha256:
        raise ValueError('exact immediate predecessor checkpoint bytes required')
    previous = json.loads(previous_raw)
    proofs = {} if proof_blobs is None else dict(proof_blobs)
    validate_checkpoint_contents(previous, accepted, proofs)
    if (previous['generation'] == 1 and previous_sha256 != SEED_CHECKPOINT_SHA256
            or previous['generation'] >= 131):
        raise ValueError('coded initial checkpoint or bounded remaining generation required')
    if (set(receipts) != PROOF_ROLES or not _sha_string(expected_audit_sha256)
            or _sha(receipts['audit']) != expected_audit_sha256):
        raise ValueError('independently recorded new audit receipt hash required')
    if set(captured) != {'before', 'entry', 'record', 'delivered'} or captured['before'] != previous['snapshot']:
        raise ValueError('captured before-state must equal the exact accepted predecessor')
    before, entry, record, after = (captured[k] for k in ('before', 'entry', 'record', 'delivered'))
    link = _case_acceptance(receipts, accepted, before, after, previous_sha256)
    name, identifier = link['entry_name'], link['record_id']
    case = json.loads(receipts['case'])
    for snapshot, record_id in ((entry, None), (record, identifier)):
        assert_append_delta(before, snapshot, entry_name=name, record_id=record_id)
        base().validate_stored_entry(snapshot['entries'][name], case['entry'])
    specification = before['specifications'][SPECIFICATION]['specification']
    base().validate_fresh_record(record['records'][str(identifier)], tag=link['compute_tag'],
                                specification=specification, entry=case['entry'])
    envelope = json.loads(receipts['envelope'])
    artifact = envelope['artifacts']['tensors.npz']
    if (type(tensor_raw) is not bytes or len(tensor_raw) != link['tensor_bytes']
            or _sha(tensor_raw) != link['tensor_sha256']
            or base64.b64decode(artifact['data'], validate=True) != tensor_raw):
        raise ValueError('actual new tensor transport bytes differ')
    for role, raw in receipts.items():
        sha = link['receipts_sha256'][role]
        if sha in proofs and proofs[sha] != raw:
            raise ValueError('content-addressed audit receipt collision')
        proofs[sha] = raw
    ledger = copy.deepcopy(previous.get('ledger', []))+[link]
    result = dict(schema='native-owned-dataset-checkpoint/2', status='READ_ONLY_AUDITED_ADVANCE',
        producer_profile='ed561666', dataset_id=355, generation=previous['generation']+1,
        previous_checkpoint_sha256=previous_sha256, seed_checkpoint_sha256=SEED_CHECKPOINT_SHA256,
        seed_snapshot_sha256=SEED_SNAPSHOT_SHA256, accepted_receipt_sha256=dict(accepted['accepted_sha256']),
        audited_records=list(previous['audited_records'])+[identifier],
        ledger=ledger, snapshot=copy.deepcopy(after), snapshot_sha256=base().digest(after),
        qualification='One audited append advanced under ordinary-OS recorded-receipt trust; '
            'hashes bind bytes, not execution attestation. No scientific accuracy/grid/solver claim, '
            'automatic recovery or server lock. All prior actual audit-proof bytes remain required.')
    validate_checkpoint_contents(result, accepted, proofs)
    return result
