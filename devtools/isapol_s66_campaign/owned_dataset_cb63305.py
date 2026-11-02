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


def base():
    global _BASE
    if _BASE is None:
        path = Path(__file__).with_name('dataset_guard.py')
        spec = importlib.util.spec_from_file_location('_owned354_snapshot_core', path)
        _BASE = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec'), _BASE.__dict__)
    return _BASE


def verify_first_acceptance(blobs):
    import hashlib
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


def snapshot_dataset(client):
    return base().snapshot_dataset(client, dataset_id=DATASET_ID, max_entries=131)


def validate_owned354(snapshot, *, authenticated_owner, specification, accepted):
    metadata = snapshot['metadata']
    owner = metadata.get('owner_user', metadata.get('creator_user'))
    if (authenticated_owner != OWNER or owner != OWNER or metadata.get('id') != DATASET_ID
            or metadata.get('dataset_type') != 'singlepoint' or metadata.get('name') != DATASET_NAME
            or metadata.get('provenance') != dict(producer=accepted['producer'],
                                                descriptor_bank_sha256=accepted['bank_sha256'])):
        raise ValueError('owned354 identity, owner or scientific provenance differs')
    if set(snapshot['specifications']) != {SPECIFICATION}:
        raise ValueError('one existing immutable request3 specification required')
    stored = snapshot['specifications'][SPECIFICATION]['specification']
    if (stored != base().plain(specification)
            or stored.get('protocols', {}).get('error_correction', {}).get('default_policy') is not False):
        raise ValueError('existing request3 specification/no-correction policy differs')
    if snapshot['associations'].get(FIRST_ENTRY, {}).get(SPECIFICATION) != FIRST_RECORD:
        raise ValueError('accepted first-case association changed')
    record = snapshot['records'].get(str(FIRST_RECORD), {})
    history = record.get('history', [])
    if (record.get('status') != 'complete' or len(history) != 1
            or history[0].get('manager_name') != accepted['manager']
            or record.get('tensor_sha256') != FIRST_TENSOR
            or record.get('native_envelope_sha256') != FIRST_CANONICAL_ENVELOPE):
        raise ValueError('accepted methylamine record/history/artifacts changed')
    base().validate_stored_entry(snapshot['entries'][FIRST_ENTRY], accepted['entry'])


def validate_new_case(before, case, *, reproduced_case):
    """Require the full result of case.verify_prepared_from_paths, not case labels.

    The caller reconstructs from the closed bank/receipt bytes immediately before
    this check. This pure guard does not confer trust on arbitrary supplied dicts.
    """
    if base().digest(case) != base().digest(reproduced_case):
        raise ValueError('entire case must match pinned-input reconstruction')
    if (case.get('schema') != 'native-bank-molecular-case/1'
            or case.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
            or case.get('dataset_id') != DATASET_ID or case.get('dataset_name') != DATASET_NAME
            or case.get('specification_name') != SPECIFICATION
            or case.get('monomer_id') == 'monomer-0006'
            or case.get('required_first_pilot_audit_sha256') != ACCEPTED_SHA256['audit']
            or case.get('submission') != dict(find_existing=False, maximum_records=1,
                                             maximum_concurrency=1, qos='embers')):
        raise ValueError('one admitted remaining-case checkpoint required')
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


def prepare_initial_checkpoint(snapshot, *, authenticated_owner, specification, accepted):
    """Only the verified singleton is an initial baseline; never bless extras."""
    validate_owned354(snapshot, authenticated_owner=authenticated_owner,
                      specification=specification, accepted=accepted)
    if set(snapshot['entries']) != {FIRST_ENTRY} or set(snapshot['records']) != {str(FIRST_RECORD)}:
        raise ValueError('initial owned354 baseline must contain exactly the audited first case')
    return dict(schema='native-owned354-checkpoint/1', status='READ_ONLY_INITIAL_BASELINE',
        dataset_id=DATASET_ID, accepted_receipt_sha256=accepted['accepted_sha256'],
        snapshot=snapshot, snapshot_sha256=base().digest(snapshot),
        qualification='Read-only accepted-first-case baseline, not mutation authorization alone '
                      'or an atomic server lock. The live before-state must match exactly.')
