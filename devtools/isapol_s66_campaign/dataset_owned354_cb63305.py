"""One guarded owned-dataset append, never dataset creation or automatic retry.

Historical354 retains its initial-only protocol. Closed355 continuation requires
all actual audit-proof bytes and an immutable one-writer intent per accepted
snapshot, so alternate JSON encodings cannot authorize competing writers.
No manager or scientific calculation is started by this helper.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

_MODULES = {}
INTENT_ROOT = Path('/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-ed561666fc-dispatch-intents')
PROOF_ROOT = Path('/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-ed561666fc-checkpoint-proofs')


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


def reserve_dispatch_intent(*, checkpoint_sha256, snapshot_sha256, case_sha256,
                            monomer_id, entry_name, compute_tag):
    """Atomic one-writer intent per accepted state, including JSON byte aliases."""
    job = os.environ.get('SLURM_JOB_ID')
    if (type(job) is not str or re.fullmatch(r'[1-9][0-9]*', job) is None
            or not compute_tag.endswith('-'+job)
            or any(type(sha) is not str or re.fullmatch(r'[0-9a-f]{64}', sha) is None
                   for sha in (checkpoint_sha256, snapshot_sha256, case_sha256))):
        raise ValueError('actual job/tag and exact checkpoint/case hashes required for dispatch intent')
    INTENT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if INTENT_ROOT.is_symlink():
        raise ValueError('plain private dispatch-intent directory required')
    target = INTENT_ROOT/snapshot_sha256
    target.mkdir(mode=0o700, exist_ok=False)
    body = dict(schema='native-owned-dataset-dispatch-intent/2', status='RESERVED_BEFORE_ENTRY',
        dataset_id=355, checkpoint_sha256=checkpoint_sha256, case_sha256=case_sha256,
        snapshot_sha256=snapshot_sha256,
        monomer_id=monomer_id, entry_name=entry_name, compute_tag=compute_tag, job_id=job,
        qualification='Ordinary-OS atomic one-writer intent, not server locking or execution attestation. '
                      'Never automatically clear an unresolved intent or repeat a mutation.')
    path = target/'intent.json'
    with path.open('x') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
    return dict(body, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_dispatch_intent(saved, *, checkpoint_sha256, snapshot_sha256, case_sha256,
                           monomer_id, entry_name, compute_tag):
    """Read the immutable original intent, not a caller-selected filesystem path."""
    path = INTENT_ROOT/snapshot_sha256/'intent.json'
    if path.is_symlink() or path.parent.is_symlink() or INTENT_ROOT.is_symlink():
        raise ValueError('plain immutable dispatch-intent files required')
    raw = path.read_bytes()
    body = json.loads(raw)
    expected = dict(checkpoint_sha256=checkpoint_sha256, case_sha256=case_sha256,
        snapshot_sha256=snapshot_sha256,
        monomer_id=monomer_id, entry_name=entry_name, compute_tag=compute_tag,
        job_id=compute_tag.rsplit('-', 1)[-1])
    if (body.get('schema') != 'native-owned-dataset-dispatch-intent/2'
            or body.get('status') != 'RESERVED_BEFORE_ENTRY' or body.get('dataset_id') != 355
            or any(body.get(k) != v for k, v in expected.items())
            or saved != dict(body, sha256=hashlib.sha256(raw).hexdigest())):
        raise ValueError('owned checkpoint/job dispatch intent changed')
    return saved['sha256']


def read_checkpoint_proofs(raw, expected_sha256, directory=None):
    """Read only exact content-addressed proof members of a byte-bound checkpoint."""
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('exact checkpoint bytes required before proof-path reads')
    guard = helper('owned_dataset_cb63305')
    required = guard.checkpoint_proof_hashes(json.loads(raw))
    if not required:
        return {}
    if directory is None:
        raise ValueError('advanced checkpoint requires its actual audit-proof directory')
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('content-addressed audit-proof directory required')
    paths = {sha: root/(sha+'.json') for sha in required}
    if (any(path.is_symlink() or path.resolve().parent != root
            or path.stat().st_size > 32*1024**2 for path in paths.values())
            or sum(path.stat().st_size for path in paths.values()) > guard.MAX_PROOF_BYTES):
        raise ValueError('bounded plain content-addressed audit-proof files required')
    result = {sha: path.read_bytes() for sha, path in paths.items()}
    if any(hashlib.sha256(blob).hexdigest() != sha for sha, blob in result.items()):
        raise ValueError('content-addressed audit-proof bytes changed')
    return result


def store_checkpoint_proofs(receipts):
    """Store actual audit receipts once; identical existing blobs may be reused."""
    guard = helper('owned_dataset_cb63305')
    if (set(receipts) != guard.PROOF_ROLES
            or any(type(raw) is not bytes or len(raw) > 32*1024**2 for raw in receipts.values())):
        raise ValueError('bounded actual audit receipt bytes required for proof storage')
    PROOF_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if PROOF_ROOT.is_symlink():
        raise ValueError('plain private proof store required')
    links = {}
    for role, raw in receipts.items():
        sha = hashlib.sha256(raw).hexdigest()
        path = PROOF_ROOT/(sha+'.json')
        if path.is_symlink():
            raise ValueError('content-addressed proof aliases are forbidden')
        try:
            with path.open('xb') as stream:
                stream.write(raw)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError('existing content-addressed proof differs; never overwrite')
        links[role] = sha
    return links


def validate_checkpoint(raw, expected_sha256, accepted, *, producer='cb63305', proof_blobs=None):
    selected = selection(producer)
    if (type(raw) is not bytes
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError('exact read-only owned354 checkpoint bytes required')
    checkpoint = json.loads(raw)
    guard = helper('owned_dataset_cb63305')
    if selected:
        if checkpoint.get('generation') == 1 and expected_sha256 != guard.SEED_CHECKPOINT_SHA256:
            raise ValueError('coded initial checkpoint byte identity required')
        return guard.validate_checkpoint_contents(checkpoint, accepted, proof_blobs)
    if proof_blobs:
        raise ValueError('historical owned354 does not admit checkpoint advancement proofs')
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
                authenticated_owner, compute_tag, save, producer='cb63305',
                checkpoint_proof_blobs=None):
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
    checkpoint = validate_checkpoint(checkpoint_raw, checkpoint_sha256, accepted, **selected,
                                     proof_blobs=checkpoint_proof_blobs)
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
    intent = None
    if selected:
        intent = reserve_dispatch_intent(checkpoint_sha256=checkpoint_sha256,
            snapshot_sha256=checkpoint['snapshot_sha256'],
            case_sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(),
            monomer_id=case['monomer_id'], entry_name=entry.name, compute_tag=compute_tag)
        save('dispatch-intent.json', intent)
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
    result = dict(dataset_id=dataset_id, record_id=record_id,
        molecule_id=after_record['records'][str(record_id)]['molecule_id'],
        n_inserted=meta.n_inserted, n_existing=meta.n_existing,
        checkpoint_sha256=checkpoint_sha256, compute_tag=compute_tag)
    if intent is not None:
        result['dispatch_intent_sha256'] = intent['sha256']
    return result
