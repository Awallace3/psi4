"""Captured/live bank preservation and ordinary-audit checkpoint publication.

Only explicit read-only clients contact QCArchive. Proofs/intents and output
metadata are ordinary-OS bindings, not server locks or host attestation.
This module never runs a solver or changes records/datasets/managers/jobs.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_MODULES = {}


def helper(name):
    if name not in ('audited_bank', 'dataset_guard', 'bank_store', 'bank_campaign',
                    'bank_dataset','bank_readonly','phoenix_pilot_cb63305','native_controller',
                    'qomp_expansion'):
        raise ValueError('closed read-only eligible-bank audit helpers required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_bank_audit_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def _raw(path, maximum=32*1024**2):
    return helper('bank_store')._read_regular(Path(path), maximum=maximum)


def _read(path):
    return json.loads(_raw(path))


def captured(data):
    return {role: _read(data/name) for role, name in dict(
        before='owned-before-entry.json', entry='owned-after-entry.json',
        record='owned-after-record.json', delivered='owned-after-delivery.json').items()}


def _verify_intent(data,outcome,draft,append_inputs,*,producer='grid400'):
    core, base = helper('audited_bank'), helper('dataset_guard')
    checkpoint = None if append_inputs is None else json.loads(append_inputs['checkpoint_raw'])
    fields = dict(dataset_id=outcome['dataset_id'] if append_inputs is not None else None,
        checkpoint_sha256=None if checkpoint is None else append_inputs['checkpoint_sha256'],
        snapshot_sha256=None if checkpoint is None else base.digest(checkpoint['snapshot']),
        context_sha256=None if checkpoint is None else append_inputs['context_sha256'],
        case_sha256=core.digest(core.encode(draft)), eligibility_sha256=draft['eligibility_sha256'],
        monomer_id=draft['monomer_id'], compute_tag=outcome['compute_tag'])
    sha = helper('bank_store').verify_intent(_read(data/'dispatch-intent.json'),
        **fields,**(dict(producer=producer) if producer!='grid400' else {}))
    if outcome.get('dispatch_intent_sha256') != sha:
        raise ValueError('bank delivered outcome and immutable dispatch intent differ')
    return sha


def validate_preservation(data,outcome,draft,append_inputs,specification,*,producer='grid400'):
    """Independently reduce all four append snapshots and the immutable intent."""
    core, base = helper('audited_bank'), helper('dataset_guard')
    options=(helper('qomp_expansion').reducer_options(append_inputs['qomp_binding'])
             if producer=='qomp' else {})
    if producer not in ('grid400','qomp'):
        raise ValueError('closed captured-preservation producer required')
    checkpoint = core.validate_checkpoint(append_inputs['checkpoint_raw'],
        append_inputs['checkpoint_sha256'], append_inputs['context_raw'],
        append_inputs['context_sha256'],append_inputs['proof_blobs'],**options)
    views = captured(data)
    before, entry, record, delivered = (views[k] for k in ('before', 'entry', 'record', 'delivered'))
    name, identifier, spec = draft['entry']['name'], outcome['record_id'], draft['specification_name']
    if (base.digest(before) != checkpoint['snapshot_sha256']
            or outcome.get('checkpoint_sha256') != append_inputs['checkpoint_sha256']):
        raise ValueError('captured bank append differs from the sealed audited predecessor')
    base.assert_append_delta(before, entry, entry_name=name, specification_name=spec)
    for snapshot in (record, delivered):
        base.assert_append_delta(before, snapshot, entry_name=name, record_id=identifier,
                                 specification_name=spec)
        if base.digest(snapshot['entries'][name]) != base.digest(entry['entries'][name]):
            raise ValueError('captured new bank entry changed after insertion')
    for snapshot in (entry, record, delivered):
        base.validate_stored_entry(snapshot['entries'][name], draft['entry'])
    base.validate_fresh_record(record['records'][str(identifier)], tag=outcome['compute_tag'],
                               specification=specification, entry=draft['entry'])
    finished = delivered['records'][str(identifier)]
    envelope = _read(data/'server-native-envelope.json')
    tensor = _raw(data/'server-tensors.npz', 16*1024**2)
    sha = hashlib.sha256(tensor).hexdigest()
    effective = core.normalize_specification(draft['specification'])
    effective['keywords'].update(draft['entry']['additional_keywords'])
    if (finished.get('status') != 'complete' or len(finished.get('history', [])) != 1
            or finished['history'][0].get('manager_name') != outcome['manager']
            or finished.get('specification') != effective
            or finished.get('native_envelope_sha256') != base.digest(envelope)
            or finished.get('tensor_sha256') != sha
            or envelope['artifacts']['tensors.npz']['sha256'] != sha
            or outcome.get('tensors_sha256') != sha or outcome.get('tensor_bytes') != len(tensor)):
        raise ValueError('captured completed bank record/history/envelope/tensors differ')
    intent = _verify_intent(data,outcome,draft,append_inputs,producer=producer)
    return dict(status='PASS_CAPTURED_OWNED_DATASET_PRESERVATION',
        checkpoint_sha256=append_inputs['checkpoint_sha256'],
        before_sha256=base.digest(before), after_sha256=base.digest(delivered),
        audited_native_envelope_sha256=base.digest(envelope), audited_tensor_sha256=sha,
        dispatch_intent_sha256=intent, preserved_records=len(before['records']),
        qualification='Captured snapshot reduction and original durable intent; not an atomic server lock.')


def validate_initial_capture(data, outcome, draft):
    """The ordinary audit still owns numeric/FD/witness checks, then bootstrap."""
    snapshot = _read(data/'bank-initial-snapshot.json')
    if (set(snapshot.get('entries', {})) != {draft['entry']['name']}
            or set(snapshot.get('associations', {})) != {draft['entry']['name']}
            or set(snapshot.get('records', {})) != {str(outcome['record_id'])}
            or snapshot.get('metadata', {}).get('id') != outcome['dataset_id']):
        raise ValueError('captured initial bank must contain only the fresh singleton')
    _verify_intent(data, outcome, draft, None)
    return snapshot


def approved_client(draft,outcome,append_inputs,*,producer='grid400'):
    import yaml
    server = yaml.safe_load(Path(
        '/storage/project/r-cs207-0/awallace43/qca/resources_losii_ac.yml').read_text())['server']
    helper('phoenix_pilot_cb63305').validate_server_config(server)
    adapter = helper('bank_dataset')
    if producer not in ('grid400','qomp') or producer=='qomp' and append_inputs is None:
        raise ValueError('QOMP audit requires its actual audited append binding')
    extra=(dict(producer='qomp',qomp_binding=append_inputs['qomp_binding'],
                context_raw=append_inputs['context_raw'],proof_blobs=append_inputs['proof_blobs'])
           if producer=='qomp' else {})
    scope = adapter.prepare_transaction(draft,draft,outcome['compute_tag'],
        checkpoint_raw=None if append_inputs is None else append_inputs['checkpoint_raw'],**extra)
    cls = helper('bank_readonly').readonly_client_class(adapter, scope, outcome['dataset_id'])
    client = cls(server['fractal_uri'], username=server['username'], password=server['password'],
                 verify=server['verify'], show_motd=False)
    info = client.get_server_information()
    if info.get('name') != 'CDS QCFractal Server' or info.get('version') != '0.70':
        raise ValueError('approved read-only bank deployment changed')
    return client


def passing_report(raw_audit,append_inputs,*,producer='grid400'):
    audit = json.loads(raw_audit)
    expected_status = ('PASS_QOMP_SERIAL_CASE_EVIDENCE' if producer=='qomp' else
                       'PASS_SELECTED_FRESH_SINGLETON_EVIDENCE' if append_inputs is None
                       else 'PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE')
    if (producer not in ('grid400','qomp') or producer=='qomp' and append_inputs is None
            or audit.get('status') != expected_status or audit.get('producer_profile') != producer
            or audit.get('campaign_scope') != ('qomp-serial' if producer=='qomp' else 'eligible-bank')
            or audit.get('dataset_mode') != ('fresh' if append_inputs is None else 'append')):
        raise ValueError('only a passing ordinary bank audit may publish an accepted checkpoint')
    return audit


def prepare_publication(client,*,pool,prepared,raw_audit,data,case_path,draft,append_inputs,output,
                        producer='grid400'):
    """Live double-snapshot confirmation around the pure, already-passing audit."""
    audit = passing_report(raw_audit,append_inputs,producer=producer)
    core, base, store = (helper(name) for name in ('audited_bank', 'dataset_guard', 'bank_store'))
    receipts = dict(audit=raw_audit,outcome=_raw(data/('qomp-outcome.json' if producer=='qomp' else 'pilot-outcome.json')),
                    case=_raw(case_path), envelope=_raw(data/'server-native-envelope.json'))
    tensor = _raw(data/'server-tensors.npz', 16*1024**2)
    snapshot = lambda: base.snapshot_dataset(client, dataset_id=audit['dataset_id'],
                                             max_entries=len(pool['eligible_ids']))
    before = snapshot()
    if append_inputs is None:
        initial = validate_initial_capture(data, json.loads(receipts['outcome']), draft)
        if base.digest(before) != base.digest(initial):
            raise ValueError('live initial bank differs from the captured audited singleton')
        context_raw, checkpoint = core.initialize(pool, prepared, receipts,
            core.digest(raw_audit), before, tensor)
        expected_proofs = json.loads(context_raw)['seed_receipts_sha256']
    else:
        views = captured(data)
        if base.digest(before) != base.digest(views['delivered']):
            raise ValueError('live bank differs from the captured audited append delivery')
        context_raw = append_inputs['context_raw']
        options=(helper('qomp_expansion').reducer_options(append_inputs['qomp_binding'])
                 if producer=='qomp' else {})
        checkpoint, _ = core.advance(append_inputs['checkpoint_raw'], append_inputs['checkpoint_sha256'],
            context_raw, append_inputs['context_sha256'], append_inputs['proof_blobs'],
            receipts=receipts, expected_audit_sha256=core.digest(raw_audit),
            captured=views,tensor_raw=tensor,**options)
        expected_proofs = checkpoint['ledger'][-1]['receipts_sha256']
    if base.digest(snapshot()) != base.digest(before):
        raise ValueError('live bank changed during ordinary audit checkpoint publication')
    checkpoint_raw = core.encode(checkpoint)
    context_path, checkpoint_path = output/'bank-context.json', output/'bank-checkpoint.json'
    result = dict(status='PASS_AUDITED_BANK_BOOTSTRAP' if append_inputs is None else 'PASS_AUDITED_BANK_ADVANCE',
        dataset_id=audit['dataset_id'], generation=checkpoint['generation'],
        context_path=str(context_path), context_sha256=core.digest(context_raw),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=core.digest(checkpoint_raw),
        audit_sha256=core.digest(raw_audit), audited_records=checkpoint['audited_records'],
        eligible_count=len(pool['eligible_ids']), refused_ids=list(pool['refused_ids']),
        whole_bank_complete=not pool['refused_ids'] and checkpoint['generation'] == 131,
        qualification='Recorded ordinary-OS audit proof and live-state agreement; no scientific accuracy claim.')
    return dict(receipts=receipts,expected_proofs=expected_proofs,
        context_raw=context_raw,checkpoint_raw=checkpoint_raw,result=result,producer=producer)


def commit_publication(publication,output):
    """No validation/native work here; QOMP caller runs ALL postseals first."""
    store=helper('bank_store')
    options=dict(producer='qomp') if publication['producer']=='qomp' else {}
    if store.store_receipts(publication['receipts'],**options)!=publication['expected_proofs']:
        raise ValueError('stored ordinary audit proof commitments differ')
    store._write_once(output/'bank-context.json',publication['context_raw'])
    store._write_once(output/'bank-checkpoint.json',publication['checkpoint_raw'])
    result=publication['result']
    store._write_once(output/'bank-advance.json',helper('audited_bank').encode(result))
    return result


def publish(client,*,pool,prepared,raw_audit,data,case_path,draft,append_inputs,output,producer='grid400'):
    publication=prepare_publication(client,pool=pool,prepared=prepared,raw_audit=raw_audit,data=data,
        case_path=case_path,draft=draft,append_inputs=append_inputs,output=output,producer=producer)
    return commit_publication(publication,output)


def advance_audit_checkpoint(*, args, raw_audit, data, draft, append_inputs, output, source):
    """Closed production adapter; controller must still verify audit job exit0."""
    passing_report(raw_audit, append_inputs)
    helper('native_controller').validate_source(source, args.commit)
    campaign = helper('bank_campaign')
    pool, raw_inputs = campaign.load_pool(args.inputs, args.build/'build-identity.json',
                                          args.qualification, args.descriptors)
    prepared = campaign.prepared_cases(pool, raw_inputs)
    outcome = _read(data/'pilot-outcome.json')
    client = approved_client(draft, outcome, append_inputs)
    result = publish(client, pool=pool, prepared=prepared, raw_audit=raw_audit, data=data,
        case_path=args.pilot, draft=draft, append_inputs=append_inputs, output=output)
    helper('native_controller').validate_source(source, args.commit)
    return result
