"""Read-only owned354 identity, accepted checkpoint and append-delta guards.

Never authorize dataset353 or reinterpret its historical append checkpoint.
Snapshots detect changes; they do not create an atomic server lock.
"""
import importlib.util
import json
from pathlib import Path

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
