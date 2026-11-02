"""One next-case append to owned354, never dataset creation or automatic retry.

The first audited singleton is the only initial checkpoint admitted here.
Advancing beyond that append requires a separately audited checkpoint protocol.
No manager or scientific calculation is started by this helper.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

_MODULES = {}


def helper(name):
    if name not in ('owned_dataset_cb63305', 'dataset_fresh_cb63305',
                    'case_cb63305', 'phoenix_pilot'):
        raise ValueError('closed owned354 append helpers required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_owned354_append_'+name, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def selection(producer):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed owned append producer required')
    return {} if producer == 'cb63305' else dict(producer=producer)


def client_class(*, producer='cb63305'):
    return helper('dataset_fresh_cb63305').client_class(**selection(producer))


def transport_binding(*, producer='cb63305'):
    return helper('dataset_fresh_cb63305').transport_binding(**selection(producer))


def validate_checkpoint(raw, expected_sha256, accepted, *, producer='cb63305'):
    selected = selection(producer)
    if (type(raw) is not bytes
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError('exact read-only owned354 checkpoint bytes required')
    checkpoint = json.loads(raw)
    guard = helper('owned_dataset_cb63305')
    if selected:
        context = guard.context(producer)
        if (checkpoint.get('schema') != 'native-owned-dataset-checkpoint/2'
                or checkpoint.get('status') != 'READ_ONLY_INITIAL_BASELINE'
                or checkpoint.get('producer_profile') != producer
                or checkpoint.get('dataset_id') != context['dataset_id']
                or checkpoint.get('generation') != 1
                or checkpoint.get('previous_checkpoint_sha256') is not None
                or checkpoint.get('audited_records') != [context['first_record']]
                or checkpoint.get('accepted_receipt_sha256') != accepted['accepted_sha256']
                or checkpoint.get('snapshot_sha256') != guard.base().digest(checkpoint['snapshot'])
                or set(checkpoint['snapshot']['entries']) != {context['first_entry']}
                or set(checkpoint['snapshot']['records']) != {str(context['first_record'])}):
            raise ValueError('only the exact audited cap32 initial checkpoint is supported')
        return checkpoint
    if (checkpoint.get('schema') != 'native-owned354-checkpoint/1'
            or checkpoint.get('status') != 'READ_ONLY_INITIAL_BASELINE'
            or checkpoint.get('dataset_id') != 354
            or checkpoint.get('accepted_receipt_sha256') != accepted['accepted_sha256']
            or checkpoint.get('snapshot_sha256') != guard.base().digest(checkpoint['snapshot'])
            or set(checkpoint['snapshot']['entries']) != {guard.FIRST_ENTRY}
            or set(checkpoint['snapshot']['records']) != {str(guard.FIRST_RECORD)}):
        raise ValueError('only the exact accepted-first-case initial checkpoint is supported')
    return checkpoint


def append_once(client, *, case_path, inputs, build_identity, qualification, descriptors,
                first_receipt_blobs, checkpoint_raw, checkpoint_sha256,
                authenticated_owner, compute_tag, save, producer='cb63305'):
    """Reconstruct, preserve, insert one entry, then submit one fresh record.

    `save` must use exclusive creation. Lost replies, nonfresh counts, concurrent
    changes or any refusal stop without retry, repair, rollback or cancellation.
    The snapshot protocol detects races; it is not an atomic server lock.
    """
    selected = selection(producer)
    transport = helper('dataset_fresh_cb63305')
    transport.require_client(client, **selected)
    guard = helper('owned_dataset_cb63305')
    accepted = guard.verify_first_acceptance(first_receipt_blobs, **selected)
    checkpoint = validate_checkpoint(checkpoint_raw, checkpoint_sha256, accepted, **selected)
    dataset_id = guard.context(producer)['dataset_id'] if selected else 354
    case = helper('case_cb63305').verify_prepared_from_paths(
        case_path, inputs, build_identity, qualification, descriptors, **selected)
    # Re-read the submitted draft separately; require equality with reconstruction.
    prepared = json.loads(case_path.read_bytes())
    if re.fullmatch('ccasp-native-'+producer+'-'+re.escape(case['monomer_id'])+
                    (r'-[1-9][0-9]*' if selected else r'-[0-9]+'),
                    compute_tag) is None:
        raise ValueError('one exact next-case compute tag required')
    info = client.get_server_information()
    if info.get('name') != 'CDS QCFractal Server' or info.get('version') != '0.70':
        raise ValueError('approved owned deployment identity changed')
    from qcportal.singlepoint import QCSpecification, SinglepointDatasetNewEntry
    from qcportal.dataset_models import DatasetSubmitBody
    from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata
    entry = SinglepointDatasetNewEntry(**case['entry'])
    specification = QCSpecification(
        **helper('phoenix_pilot').execution_specification(case['specification']))
    snapshot = lambda: guard.snapshot_dataset(client, **selected)
    before = snapshot()
    validate_owned = guard.validate_owned_dataset if selected else guard.validate_owned354
    validate_owned(before, authenticated_owner=authenticated_owner,
                   specification=specification, accepted=accepted, **selected)
    if before != checkpoint['snapshot']:
        raise ValueError('live owned354 state differs from the sealed initial checkpoint')
    guard.validate_new_case(before, prepared, reproduced_case=case, **selected)
    managers = list(client.query_managers(status=['active'], limit=200))
    if len(managers) == 200 or any('*' in m.compute_tags or compute_tag in m.compute_tags for m in managers):
        raise ValueError('new singleton tag is not isolated from existing managers')
    save('owned-before-entry.json', before)
    save('entry-insert-attempt.json', dict(dataset_id=dataset_id, entry=entry.name,
        case_sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(),
        checkpoint_sha256=checkpoint_sha256, transport_retries=False))
    if snapshot() != before:
        raise ValueError('owned354 changed immediately before entry insertion')
    meta = client.make_request('post', f'api/v1/datasets/singlepoint/{dataset_id}/entries/bulkCreate',
        InsertMetadata, body=[entry], allow_retries=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('entry insertion was not exactly one fresh entry; do not retry')
    after_entry = snapshot()
    guard.assert_append_delta(before, after_entry, entry_name=entry.name)
    guard.base().validate_stored_entry(after_entry['entries'][entry.name], case['entry'])
    save('owned-after-entry.json', after_entry)
    save('record-submit-attempt.json', dict(dataset_id=dataset_id, entry=entry.name,
        specification=guard.SPECIFICATION, compute_tag=compute_tag,
        find_existing=False, transport_retries=False))
    if snapshot() != after_entry:
        raise ValueError('owned354 changed immediately before record submission')
    body = DatasetSubmitBody(entry_names=[entry.name], specification_names=[guard.SPECIFICATION],
                             compute_tag=compute_tag, find_existing=False)
    meta = client.make_request('post', f'api/v1/datasets/singlepoint/{dataset_id}/submit',
        InsertCountsMetadata, body=body, allow_retries=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('record insertion was not exactly one fresh record; do not retry')
    after_record = snapshot()
    record_id = after_record['associations'][entry.name][guard.SPECIFICATION]
    if (type(record_id) is not int or record_id <= 0
            or selected and record_id in (5234314, 5234315, 5234316, 5234317, 5234318)):
        raise RuntimeError('one exact fresh record identity required; do not retry')
    guard.assert_append_delta(before, after_record, entry_name=entry.name, record_id=record_id)
    guard.base().validate_stored_entry(after_record['entries'][entry.name], case['entry'])
    guard.base().validate_fresh_record(after_record['records'][str(record_id)],
        tag=compute_tag, specification=specification, entry=case['entry'])
    save('owned-after-record.json', after_record)
    return dict(dataset_id=dataset_id, record_id=record_id,
        molecule_id=after_record['records'][str(record_id)]['molecule_id'],
        n_inserted=meta.n_inserted, n_existing=meta.n_existing,
        checkpoint_sha256=checkpoint_sha256, compute_tag=compute_tag)
