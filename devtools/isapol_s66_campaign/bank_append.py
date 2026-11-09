"""One checkpoint-bound entry/record append, never recovery or job execution.

The closed launch adapter supplies its reviewed dataset adapter and a prepared
case reconstructed from the measured pool. Complete audit proofs bind the
new dataset; snapshots and a durable intent protect every prior member.
"""
import importlib.util
import json
from pathlib import Path

_MODULES = {}


def helper(name):
    if name not in ('audited_bank', 'bank_store', 'dataset_guard', 'phoenix_pilot',
                    'grid400_bank_policy','qomp_execution'):
        raise ValueError('closed audited-bank append helpers required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_bank_append_'+name, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def append_once(client, scope, case, *, dataset_adapter, prepared_case,
                checkpoint_raw, checkpoint_sha256, context_raw, context_sha256,
                proof_blobs, authenticated_owner, save,producer='grid400',
                qomp_binding=None,before_submit=None):
    """Exclusive save callback; an ambiguous write always stops without replay."""
    dataset_adapter._validate(client, scope, case)
    if (producer not in ('grid400','qomp') or getattr(scope,'producer','grid400')!=producer
            or producer=='qomp' and not callable(before_submit)
            or producer=='grid400' and (qomp_binding is not None or before_submit is not None)):
        raise ValueError('closed append producer/live release admission required')
    options=dict(producer=producer,qomp_binding=qomp_binding)
    core, base, store = (helper(name) for name in ('audited_bank', 'dataset_guard', 'bank_store'))
    checkpoint = core.validate_checkpoint(
        checkpoint_raw,checkpoint_sha256,context_raw,context_sha256,proof_blobs,**options)
    reproduced = core.bind_next_case(prepared_case, checkpoint_raw, checkpoint_sha256,
                                     context_raw,context_sha256,proof_blobs,**options)
    context = json.loads(context_raw)
    identifier, specification_name = context['dataset_id'], context['specification_name']
    if (scope.mode != 'append' or scope.dataset_id != identifier
            or core.encode(case) != core.encode(reproduced)
            or scope.case_sha256 != core.digest(core.encode(case))
            or authenticated_owner != 'awallace43'):
        raise ValueError('one entirely reconstructed ordinary-audited bank append required')
    info = client.get_server_information()
    if info.get('name') != 'CDS QCFractal Server' or info.get('version') != '0.70':
        raise ValueError('approved bank QCArchive deployment identity changed')
    from qcportal.singlepoint import QCSpecification, SinglepointDatasetNewEntry
    from qcportal.dataset_models import DatasetSubmitBody
    from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata
    entry = SinglepointDatasetNewEntry(**case['entry'])
    specification = QCSpecification(**helper('phoenix_pilot').execution_specification(case['specification']))
    snapshot = lambda: base.snapshot_dataset(client, dataset_id=identifier,
                                             max_entries=len(context['eligible_ids']))
    before = snapshot()
    if base.digest(before) != checkpoint['snapshot_sha256'] or any(
            r.get('status') != 'complete' for r in before['records'].values()):
        raise ValueError('live bank differs from the sealed ordinary-audited predecessor')
    managers = list(client.query_managers(status=['active'], limit=200))
    if len(managers) == 200 or any('*' in m.compute_tags or scope.compute_tag in m.compute_tags for m in managers):
        raise ValueError('new bank singleton tag is not isolated from active managers')
    if producer=='qomp':
        helper('qomp_execution').manager_isolation(managers,scope.compute_tag)
    save('owned-before-entry.json', before)
    intent = store.reserve_intent(dataset_id=identifier, checkpoint_sha256=checkpoint_sha256,
        snapshot_sha256=checkpoint['snapshot_sha256'], context_sha256=context_sha256,
        case_sha256=scope.case_sha256, eligibility_sha256=case['eligibility_sha256'],
        monomer_id=case['monomer_id'],compute_tag=scope.compute_tag,producer=producer)
    save('dispatch-intent.json', intent)
    save('entry-insert-attempt.json', dict(dataset_id=identifier, entry=entry.name,
        case_sha256=scope.case_sha256, checkpoint_sha256=checkpoint_sha256, transport_retries=False))
    if base.digest(snapshot()) != base.digest(before):
        raise ValueError('bank changed immediately before entry insertion')
    meta = client.make_request('post', f'api/v1/datasets/singlepoint/{identifier}/entries/bulkCreate',
        InsertMetadata, body=[entry], allow_retries=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('entry insertion was not exactly one fresh bank entry; no retry')
    after_entry = snapshot()
    base.assert_append_delta(before, after_entry, entry_name=entry.name,
                             specification_name=specification_name)
    base.validate_stored_entry(after_entry['entries'][entry.name], case['entry'])
    save('owned-after-entry.json', after_entry)
    save('record-submit-attempt.json', dict(dataset_id=identifier, entry=entry.name,
        specification=specification_name, compute_tag=scope.compute_tag,
        find_existing=False, transport_retries=False))
    if base.digest(snapshot()) != base.digest(after_entry):
        raise ValueError('bank changed immediately before record submission')
    body = DatasetSubmitBody(entry_names=[entry.name], specification_names=[specification_name],
                             compute_tag=scope.compute_tag, find_existing=False)
    if producer=='qomp':
        save('scratch-release-admission.json',before_submit())
    meta = client.make_request('post', f'api/v1/datasets/singlepoint/{identifier}/submit',
                               InsertCountsMetadata, body=body, allow_retries=False)
    if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
        raise RuntimeError('record insertion was not exactly one fresh bank record; no retry')
    after_record = snapshot()
    record_id = after_record['associations'][entry.name][specification_name]
    protected=helper('grid400_bank_policy').PROTECTED_RECORDS
    if producer=='qomp':
        protected=helper('qomp_execution').helper('qomp_campaign').PROTECTED_RECORDS+tuple(checkpoint['audited_records'])
    if (type(record_id) is not int or record_id <= 0 or record_id in protected):
        raise RuntimeError('one exact fresh bank record identity required; no retry')
    base.assert_append_delta(before, after_record, entry_name=entry.name, record_id=record_id,
                             specification_name=specification_name)
    if base.digest(after_record['entries'][entry.name]) != base.digest(after_entry['entries'][entry.name]):
        raise ValueError('new bank entry changed during record submission')
    base.validate_stored_entry(after_record['entries'][entry.name], case['entry'])
    base.validate_fresh_record(after_record['records'][str(record_id)],
        tag=scope.compute_tag, specification=specification, entry=case['entry'])
    save('owned-after-record.json', after_record)
    return dict(dataset_id=identifier, record_id=record_id,
        molecule_id=after_record['records'][str(record_id)]['molecule_id'],
        n_inserted=meta.n_inserted, n_existing=meta.n_existing,
        checkpoint_sha256=checkpoint_sha256, compute_tag=scope.compute_tag,
        dispatch_intent_sha256=intent['sha256'])
