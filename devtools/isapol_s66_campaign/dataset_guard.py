"""Read-only snapshots and strict append deltas for the owned native dataset.

No mutation endpoint is available here. Snapshots detect concurrent changes;
they are not a server transaction lock or hostile-host security boundary.
"""
import hashlib
import json
import math

DATASET_ID = 353
DATASET_NAME = 'S66x8-native-CCASP-acf54c28e5-v1'
SPECIFICATION = 'pbe0-atz-native-request2'
SCIENTIFIC_COMMIT = 'acf54c28e50822b5b825f4d26e5d9f9a2835d773'
WATER_ENTRY = 'monomer-0001-0459be17cd38'
WATER_RECORD = 5234314
WATER_TENSOR = '6a15173b95eee548fed36d740a3705836d94f203e77708b0ab16996cb27a8cc2'


def plain(value):
    if hasattr(value, 'dict'):
        return plain(value.dict(by_alias=True))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, bytes):
        # RecordTask.function_kwargs_compressed is bytes in QCPortal0.63.
        # Snapshot its identity, not an unbounded decompression or binary dump.
        return dict(bytes_sha256=hashlib.sha256(value).hexdigest(), size_bytes=len(value))
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError('unsupported snapshot value: '+type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def snapshot_dataset(client, *, max_entries=256, dataset_id=DATASET_ID):
    """Use fresh objects/forced enumerations; never trust cached absences."""
    from typing import Any, Dict
    if type(dataset_id) is not int or dataset_id <= 0:
        raise ValueError('one explicit positive dataset identity required for read-only snapshot')
    raw = client.make_request('get', f'api/v1/datasets/{dataset_id}', Dict[str, Any])
    dataset = client.get_dataset_by_id(dataset_id)
    dataset.fetch_specifications(force_refetch=True)
    specifications = plain(dataset.specifications)
    dataset.fetch_entry_names()
    names = list(dataset.entry_names)
    if len(names) > max_entries or len(names) != len(set(names)):
        raise RuntimeError('bounded unique owned-entry inventory required')
    entries = {e.name: plain(e) for e in dataset.iterate_entries(force_refetch=True)}
    if set(entries) != set(names):
        raise RuntimeError('entry enumeration changed during snapshot')
    associations = {name: {spec: None for spec in specifications} for name in entries}
    records = {}
    include = ['compute_history', 'task', 'molecule']
    for entry, spec, record in dataset.iterate_records(force_refetch=True, include=include):
        if (entry not in associations or spec not in associations[entry]
                or associations[entry][spec] is not None):
            raise RuntimeError('unexpected/duplicate record association')
        associations[entry][spec] = record.id
        body = plain(record)
        envelope = record.extras.get('psi4:native_monomer')
        records[str(record.id)] = dict(
            id=record.id, molecule_id=record.molecule_id, status=record.status.value,
            body_sha256=digest(body), native_envelope_sha256=digest(envelope),
            tensor_sha256=(envelope or {}).get('artifacts', {}).get('tensors.npz', {}).get('sha256'),
            history=[plain(h) for h in record.compute_history],
            specification=plain(record.specification),
            task=body.get('task'))
    # An independent record query must agree with the refreshed association set.
    queried = {r.id for r in client.query_singlepoints(dataset_id=dataset_id)}
    if queried != {r['id'] for r in records.values()}:
        raise RuntimeError('record query and association inventory differ')
    return dict(schema='native-owned-dataset-snapshot/1', metadata=plain(raw),
                specifications=specifications, entries=entries,
                associations=associations, records=records)


def validate_owned(snapshot, *, authenticated_owner, implementation, specification,
                   water_envelope_sha256):
    metadata = snapshot['metadata']
    owner = metadata.get('owner_user', metadata.get('creator_user'))
    if (metadata.get('id') != DATASET_ID or metadata.get('dataset_type') != 'singlepoint'
            or metadata.get('name') != DATASET_NAME or not authenticated_owner
            or owner != authenticated_owner
            or metadata.get('provenance') != dict(
                scientific_build_commit=SCIENTIFIC_COMMIT, implementation_sha256=implementation)):
        raise ValueError('owned dataset identity/ownership/provenance differs')
    if set(snapshot['specifications']) != {SPECIFICATION}:
        raise ValueError('unexpected owned specification inventory')
    stored = snapshot['specifications'][SPECIFICATION]['specification']
    if stored != plain(specification):
        raise ValueError('existing immutable specification differs')
    if stored.get('protocols', {}).get('error_correction', {}).get('default_policy') is not False:
        raise ValueError('explicit no-correction policy required')
    if snapshot['associations'].get(WATER_ENTRY, {}).get(SPECIFICATION) != WATER_RECORD:
        raise ValueError('accepted water association differs')
    water = snapshot['records'].get(str(WATER_RECORD), {})
    history = water.get('history', [])
    if (water.get('status') != 'complete' or len(history) != 1
            or not history[0]['manager_name'].startswith('ccasp-native-s66-water-14948991-')
            or water.get('tensor_sha256') != WATER_TENSOR
            or water.get('native_envelope_sha256') != water_envelope_sha256):
        raise ValueError('accepted water result/history/artifact differs')


def validate_new_entry(before, draft):
    if draft['monomer_id'] != 'monomer-0004':
        raise ValueError('only the next methanol pilot is authorized by this checkpoint')
    entry = draft['entry']
    if entry['attributes']['monomer_id'] != draft['monomer_id']:
        raise ValueError('draft and entry monomer identities differ')
    if entry['name'] in before['entries']:
        raise ValueError('new entry must be absent, never reuse or repair it')
    identity = entry['attributes']['monomer_id']
    geometry = entry['attributes']['geometry_sha256']
    for previous in before['entries'].values():
        attributes = previous['attributes']
        if attributes.get('monomer_id') == identity or attributes.get('geometry_sha256') == geometry:
            raise ValueError('monomer/geometry already present')
    if any(r['status'] != 'complete' for r in before['records'].values()):
        raise ValueError('all previously owned records must be complete and preserved')


def assert_append_delta(before, after, *, entry_name, record_id=None, specification_name=SPECIFICATION):
    """Every old entry/specification/record/association is immutable."""
    if (before['metadata'] != after['metadata']
            or before['specifications'] != after['specifications']
            or set(after['entries']) != set(before['entries']) | {entry_name}
            or entry_name in before['entries']
            or set(after['associations']) != set(after['entries'])):
        raise ValueError('unexpected dataset metadata/specification/entry delta')
    for name, entry in before['entries'].items():
        if entry != after['entries'].get(name) or before['associations'][name] != after['associations'].get(name):
            raise ValueError('preexisting entry or record association changed')
    for identifier, record in before['records'].items():
        if after['records'].get(identifier) != record:
            raise ValueError('preexisting record changed or disappeared')
    expected = dict(before['records'])
    if record_id is not None:
        if str(record_id) in before['records'] or str(record_id) not in after['records']:
            raise ValueError('exactly one fresh record required')
        expected[str(record_id)] = after['records'][str(record_id)]
    if after['records'] != expected:
        raise ValueError('unexpected extra/deleted records')
    associations = {s: None for s in before['specifications']}
    if record_id is not None:
        if specification_name not in associations:
            raise ValueError('expected immutable specification is absent')
        associations[specification_name] = record_id
    if after['associations'][entry_name] != associations:
        raise ValueError('unexpected new record association')


def validate_stored_entry(stored, expected):
    if (stored['name'] != expected['name'] or stored['attributes'] != expected['attributes']
            or stored['additional_keywords'] != expected['additional_keywords']):
        raise ValueError('stored new entry metadata/runtime differs')
    molecule, raw = stored['molecule'], expected['molecule']
    geometry = molecule['geometry']
    flat = [v for row in geometry for v in row] if geometry and isinstance(geometry[0], list) else geometry
    if (molecule['symbols'] != raw['symbols'] or molecule['molecular_charge'] != 0
            or molecule['molecular_multiplicity'] != 1 or molecule['fix_com'] is not True
            or molecule['fix_orientation'] is not True or len(flat) != len(raw['geometry'])
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v-w) > 1e-10
                   for v, w in zip(flat, raw['geometry']))):
        raise ValueError('stored raw molecular coordinates/state differ')
    # Cached symmetry is never repaired: the bound native request owns private C1.


def validate_fresh_record(record, *, tag, specification, entry):
    effective = plain(specification)
    effective['keywords'] = dict(effective['keywords'], **entry['additional_keywords'])
    if (record['status'] != 'waiting' or record['history']
            or not record['task'] or record['task']['compute_tag'] != tag
            or record['specification'] != effective):
        raise ValueError('fresh record state/tag/effective specification differs')
