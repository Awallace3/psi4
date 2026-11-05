"""Pure ordinary-audit checkpoint reduction for a fresh eligible monomer bank.

Expected audit hashes and the bootstrap-context hash come from recorded,
positively verified independent jobs. Hashes bind ordinary-OS receipts, not
host attestation. This module never contacts QCArchive, writes, or retries.
"""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

PROOF_ROLES = frozenset(('audit', 'outcome', 'case', 'envelope'))
MAX_PROOF_BYTES = 512*1024**2
FILE_GATE_SHA256 = 'b70b0c621d766a16a9dae94d97ce251e70b9f9d5aeeabf949b452f977e8f64ec'
SCIENCE = '1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2'


def helper(name):
    if name not in ('dataset_guard', 'grid400_bank_policy', 'grid400_bank_case', 'phoenix_pilot'):
        raise ValueError('closed audited-bank helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_audited_bank_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _prefix(snapshot, entries, records):
    if (set(snapshot) != {'schema', 'metadata', 'specifications', 'entries', 'associations', 'records'}
            or snapshot.get('schema') != 'native-owned-dataset-snapshot/1'):
        raise ValueError('exact captured dataset snapshot structure required')
    try:
        return dict(snapshot, entries={name: snapshot['entries'][name] for name in entries},
            associations={name: snapshot['associations'][name] for name in entries},
            records={str(identifier): snapshot['records'][str(identifier)] for identifier in records})
    except KeyError as exc:
        raise ValueError('audited bank snapshot member missing') from exc


def _receipts(receipts):
    if (type(receipts) is not dict or set(receipts) != PROOF_ROLES
            or any(type(raw) is not bytes or len(raw) > 32*1024**2 for raw in receipts.values())):
        raise ValueError('exact bounded ordinary audit receipt bytes required')
    return {role: digest(raw) for role, raw in receipts.items()}


def normalize_specification(specification):
    """The same local metadata-model defaults used by the ordinary QCA capture."""
    from qcportal.singlepoint import QCSpecification
    return helper('dataset_guard').plain(QCSpecification(
        **helper('phoenix_pilot').execution_specification(specification)))


def _case_link(receipts, context, before, after, previous_sha256):
    links = _receipts(receipts)
    audit, outcome, case, envelope = (json.loads(receipts[k]) for k in ('audit', 'outcome', 'case', 'envelope'))
    base, policy = helper('dataset_guard'), helper('grid400_bank_policy')
    fresh = before is None
    mode = 'fresh' if fresh else 'append'
    identifier, mid = audit.get('record_id'), case.get('monomer_id')
    job = outcome.get('allocation', {}).get('JobId')
    numeric, artifact = audit.get('numeric', {}), envelope.get('artifacts', {}).get('tensors.npz', {})
    expected_status = ('PASS_SELECTED_FRESH_SINGLETON_EVIDENCE' if fresh
                       else 'PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE')
    raw_case = copy.deepcopy(case)
    if not fresh:
        raw_case['dataset_mode'] = 'awaiting-audited-bootstrap'
        for key in ('dataset_id', 'bootstrap_context_sha256', 'checkpoint_sha256'):
            raw_case.pop(key, None)
    if (mid not in context['eligible_ids']
            or digest(encode(raw_case)) != context['prepared_case_sha256'].get(mid)
            or type(identifier) is not int or identifier <= 0 or identifier in policy.PROTECTED_RECORDS
            or type(job) is not str or re.fullmatch('[1-9][0-9]*', job) is None
            or audit.get('status') != expected_status
            or any(item.get('producer_profile') != 'grid400' or item.get('dataset_mode') != mode
                   or item.get('campaign_scope') != 'eligible-bank' for item in (audit, outcome, case))
            or any(item.get('dataset_id') != context['dataset_id'] for item in (audit, outcome))
            or audit.get('monomer_id') != mid or outcome.get('monomer_id') != mid
            or outcome.get('record_id') != identifier or audit.get('case_job') != job
            or audit.get('case_outcome_sha256') != links['outcome']
            or audit.get('pilot_sha256') != links['case'] or outcome.get('pilot_sha256') != links['case']
            or audit.get('case_operational_commit') != outcome.get('operational_commit')
            or audit.get('scientific_build_commit') != SCIENCE
            or outcome.get('scientific_build_commit') != SCIENCE
            or case.get('producer') != context['producer']
            or case.get('eligibility_sha256') != context['eligibility_sha256']
            or audit.get('runtime_unchanged') is not True or outcome.get('runtime_unchanged') is not True
            or outcome.get('find_existing') is not False or outcome.get('maximum_records') != 1
            or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
            or outcome.get('runtime_input_sha256') != case.get('runtime_input_sha256')
            or outcome.get('compute_tag') != 'ccasp-native-grid400-bank-'+mid+'-'+job
            or audit.get('file_gate_sha256') != FILE_GATE_SHA256
            or audit.get('roles', {}).get('molecular_qcschema') != 1
            or not audit.get('traces') or any(t.get('status') != 'passed' for t in audit['traces'].values())
            or numeric.get('status') != 'PASS_NUMERIC_TRANSPORT_ONLY'
            or audit.get('envelope_identity', {}).get('status') != 'PASS_NATIVE_ENVELOPE_IDENTITY'
            or audit['envelope_identity'].get('policy') != 'exact-json-diagnostic-number/1'
            or audit['envelope_identity'].get('server_json_sha256') != digest(encode(envelope))
            or audit['envelope_identity'].get('epsilon_tolerance_used') is not False
            or envelope.get('schema') != 'native-monomer-artifacts/2'
            or envelope.get('request') != case['entry']['additional_keywords']['function_kwargs']['native_monomer']
            or envelope.get('implementation', {}).get('sha256') != context['producer']['implementation_sha256']
            or type(artifact.get('size')) is not int or not 0 < artifact['size'] <= 16*1024**2
            or not _sha(artifact.get('sha256')) or outcome.get('tensor_bytes') != artifact['size']
            or outcome.get('tensors_sha256') != artifact['sha256']
            or numeric.get('tensor_sha256') != artifact['sha256']):
        raise ValueError('ordinary audited bank case/provenance/artifact chain differs')
    name, spec = case['entry']['name'], context['specification_name']
    if fresh:
        if case.get('dataset_id') is not None or mid != context['eligible_ids'][0]:
            raise ValueError('only the first reconstructed eligible case can bootstrap')
    else:
        preservation = audit.get('captured_preservation', {})
        intent = outcome.get('dispatch_intent_sha256')
        if (case.get('dataset_id') != context['dataset_id']
                or case.get('bootstrap_context_sha256') != context['context_sha256']
                or case.get('checkpoint_sha256') != previous_sha256
                or outcome.get('checkpoint_sha256') != previous_sha256
                or not _sha(intent) or preservation.get('dispatch_intent_sha256') != intent
                or preservation.get('status') != 'PASS_CAPTURED_OWNED_DATASET_PRESERVATION'
                or preservation.get('checkpoint_sha256') != previous_sha256
                or preservation.get('before_sha256') != base.digest(before)
                or preservation.get('after_sha256') != base.digest(after)
                or preservation.get('audited_tensor_sha256') != artifact['sha256']
                or preservation.get('audited_native_envelope_sha256') != base.digest(envelope)):
            raise ValueError('ordinary bank append checkpoint/preservation/intent differs')
        base.assert_append_delta(before, after, entry_name=name, record_id=identifier,
                                 specification_name=spec)
        for prior in before['entries'].values():
            if (prior['attributes'].get('monomer_id') == mid or prior['attributes'].get(
                    'geometry_sha256') == case['entry']['attributes'].get('geometry_sha256')):
                raise ValueError('duplicate bank monomer/geometry cannot advance')
    base.validate_stored_entry(after['entries'][name], case['entry'])
    specification = normalize_specification(case['specification'])
    effective = copy.deepcopy(specification)
    effective['keywords'].update(case['entry']['additional_keywords'])
    record = after['records'][str(identifier)]
    if (after['associations'].get(name) != {spec: identifier}
            or specification != context['execution_specification']
            or after['specifications'].get(spec, {}).get('specification') != specification
            or record.get('specification') != effective
            or record.get('status') != 'complete' or len(record.get('history', [])) != 1
            or record['history'][0].get('manager_name') != outcome.get('manager')
            or record.get('native_envelope_sha256') != base.digest(envelope)
            or record.get('tensor_sha256') != artifact['sha256']):
        raise ValueError('audited bank record/history/artifacts changed')
    return dict(monomer_id=mid, entry_name=name, record_id=identifier, receipts_sha256=links,
        previous_checkpoint_sha256=previous_sha256, tensor_sha256=artifact['sha256'],
        tensor_bytes=artifact['size'], completed_record_sha256=base.digest(record))


def _tensor(receipts, tensor_raw):
    artifact = json.loads(receipts['envelope'])['artifacts']['tensors.npz']
    if (type(tensor_raw) is not bytes or len(tensor_raw) != artifact['size']
            or digest(tensor_raw) != artifact['sha256']
            or base64.b64decode(artifact['data'], validate=True) != tensor_raw):
        raise ValueError('actual audited tensor transport bytes differ')


def initialize(pool, prepared_cases, receipts, expected_audit_sha256, snapshot, tensor_raw):
    """Create a new context only from the recorded ordinary fresh audit."""
    links = _receipts(receipts)
    policy, cases, base = (helper(name) for name in ('grid400_bank_policy', 'grid400_bank_case', 'dataset_guard'))
    eligible = pool.get('eligible_ids', [])
    audit = json.loads(receipts['audit'])
    identifier = audit.get('dataset_id')
    if (pool.get('status') != 'VERIFIED_DESCRIPTOR_POOL_ELIGIBILITY_ONLY'
            or not eligible or eligible != sorted(set(eligible))
            or set(eligible) & set(pool.get('refused_ids', []))
            or set(eligible) | set(pool.get('refused_ids', [])) != {
                f'monomer-{i:04d}' for i in range(1, 132)}
            or set(prepared_cases) != set(eligible) or links['audit'] != expected_audit_sha256
            or not _sha(expected_audit_sha256)
            or type(identifier) is not int or identifier <= 0 or identifier in policy.PROTECTED_DATASETS):
        raise ValueError('recorded fresh audit and complete reconstructed eligible membership required')
    first = prepared_cases[eligible[0]]
    eligibility_sha = digest(encode(dict(eligible_ids=eligible, refused_ids=pool['refused_ids'],
                                        descriptor_authority=pool['authority'])))
    for mid, case in prepared_cases.items():
        if (case.get('monomer_id') != mid or case.get('producer_profile') != 'grid400'
                or case.get('campaign_scope') != 'eligible-bank' or case.get('campaign') != policy.CAMPAIGN
                or case.get('dataset_name') != cases.DATASET_NAME
                or case.get('specification_name') != cases.SPECIFICATION_NAME
                or case.get('eligibility_sha256') != eligibility_sha
                or case.get('producer') != first['producer']
                or case.get('dataset_mode') != ('fresh' if mid == eligible[0] else 'awaiting-audited-bootstrap')
                or 'dataset_id' in case):
            raise ValueError('complete reconstructed unbound bank case inventory required')
    if encode(first) != encode(json.loads(receipts['case'])):
        raise ValueError('fresh audit case differs from measured-pool reconstruction')
    metadata = snapshot['metadata']
    spec = snapshot['specifications'].get(cases.SPECIFICATION_NAME, {}).get('specification', {})
    normalized = normalize_specification(first['specification'])
    if (metadata.get('id') != identifier or metadata.get('name') != cases.DATASET_NAME
            or metadata.get('dataset_type') != 'singlepoint'
            or metadata.get('owner_user', metadata.get('creator_user')) != 'awallace43'
            or metadata.get('provenance') != dict(producer=first['producer'],
                                                descriptor_bank_sha256=first['descriptor_bank_sha256'])
            or set(snapshot['specifications']) != {cases.SPECIFICATION_NAME}
            or spec != normalized
            or spec.get('protocols', {}).get('error_correction', {}).get('default_policy') is not False
            or set(snapshot['entries']) != {first['entry']['name']}
            or set(snapshot['associations']) != {first['entry']['name']}
            or set(snapshot['records']) != {str(audit.get('record_id'))}):
        raise ValueError('exact new ordinary-audited singleton snapshot required')
    # Reject top-level extras/schema mismatches before constructing an anchor.
    _prefix(snapshot, [first['entry']['name']], [audit['record_id']])
    context = dict(schema='native-audited-bank-context/1', campaign=policy.CAMPAIGN,
        dataset_id=identifier, dataset_name=cases.DATASET_NAME, specification_name=cases.SPECIFICATION_NAME,
        producer=first['producer'], execution_specification=normalized,
        eligibility_sha256=first['eligibility_sha256'],
        eligible_ids=list(eligible), refused_ids=list(pool['refused_ids']),
        prepared_case_sha256={mid: digest(encode(case)) for mid, case in prepared_cases.items()},
        seed_receipts_sha256=links, seed_snapshot_sha256=base.digest(snapshot),
        first_record=audit['record_id'], first_entry=first['entry']['name'])
    context_raw = encode(context)
    bound = dict(context, context_sha256=digest(context_raw))
    _case_link(receipts, bound, None, snapshot, None)
    _tensor(receipts, tensor_raw)
    checkpoint = dict(schema='native-audited-bank-checkpoint/1', status='READ_ONLY_INITIAL_BASELINE',
        campaign=policy.CAMPAIGN, dataset_id=identifier, eligibility_sha256=first['eligibility_sha256'],
        bootstrap_context_sha256=digest(context_raw), generation=1, previous_checkpoint_sha256=None,
        audited_records=[audit['record_id']], ledger=[], snapshot=copy.deepcopy(snapshot),
        snapshot_sha256=base.digest(snapshot))
    proofs = {sha: receipts[role] for role, sha in links.items()}
    checkpoint_raw = encode(checkpoint)
    validate_checkpoint(checkpoint_raw, digest(checkpoint_raw), context_raw, digest(context_raw), proofs)
    return context_raw, checkpoint


def validate_checkpoint(raw, expected_sha256, context_raw, context_sha256, proofs):
    """Verify every seed/append receipt against exact final-snapshot prefixes."""
    if (type(raw) is not bytes or digest(raw) != expected_sha256
            or type(context_raw) is not bytes or digest(context_raw) != context_sha256):
        raise ValueError('exact recorded context/checkpoint bytes required')
    context, checkpoint, base = json.loads(context_raw), json.loads(raw), helper('dataset_guard')
    policy = helper('grid400_bank_policy')
    generation, ledger = checkpoint.get('generation'), checkpoint.get('ledger')
    if (context.get('schema') != 'native-audited-bank-context/1' or context.get('campaign') != policy.CAMPAIGN
            or context.get('dataset_id') in policy.PROTECTED_DATASETS
            or type(generation) is not int or not 1 <= generation <= len(context['eligible_ids'])
            or type(ledger) is not list or len(ledger) != generation-1
            or checkpoint.get('schema') != 'native-audited-bank-checkpoint/1'
            or checkpoint.get('campaign') != context['campaign']
            or checkpoint.get('dataset_id') != context['dataset_id']
            or checkpoint.get('eligibility_sha256') != context['eligibility_sha256']
            or checkpoint.get('bootstrap_context_sha256') != context_sha256
            or checkpoint.get('snapshot_sha256') != base.digest(checkpoint['snapshot'])):
        raise ValueError('bounded audited bank checkpoint identity differs')
    required = set(context['seed_receipts_sha256'].values())
    for link in ledger:
        if set(link.get('receipts_sha256', {})) != PROOF_ROLES:
            raise ValueError('complete append audit receipt roles required')
        required.update(link['receipts_sha256'].values())
    if (type(proofs) is not dict or set(proofs) != required
            or any(not _sha(sha) or type(blob) is not bytes or len(blob) > 32*1024**2
                   or digest(blob) != sha for sha, blob in proofs.items())
            or sum(len(blob) for blob in proofs.values()) > MAX_PROOF_BYTES):
        raise ValueError('complete exact bounded bank audit proof bytes required')
    bound = dict(context, context_sha256=context_sha256)
    entries, records = [context['first_entry']], [context['first_record']]
    prefix = _prefix(checkpoint['snapshot'], entries, records)
    if base.digest(prefix) != context['seed_snapshot_sha256']:
        raise ValueError('ordinary-audited bootstrap snapshot changed')
    seed = {role: proofs[sha] for role, sha in context['seed_receipts_sha256'].items()}
    _case_link(seed, bound, None, prefix, None)
    if (checkpoint.get('status') != ('READ_ONLY_INITIAL_BASELINE' if generation == 1 else 'READ_ONLY_AUDITED_ADVANCE')
            or generation == 1 and checkpoint.get('previous_checkpoint_sha256') is not None
            or generation > 1 and checkpoint.get('previous_checkpoint_sha256') != ledger[-1].get('previous_checkpoint_sha256')):
        raise ValueError('monotonic audited bank transition required')
    for index, link in enumerate(ledger, start=1):
        name, identifier = link['entry_name'], link['record_id']
        if (name in entries or identifier in records or link['monomer_id'] != context['eligible_ids'][index]
                or not _sha(link.get('previous_checkpoint_sha256'))):
            raise ValueError('exact next unique eligible bank member required')
        after = _prefix(checkpoint['snapshot'], entries+[name], records+[identifier])
        receipts = {role: proofs[sha] for role, sha in link['receipts_sha256'].items()}
        if _case_link(receipts, bound, prefix, after, link['previous_checkpoint_sha256']) != link:
            raise ValueError('audited bank ledger changed')
        entries.append(name)
        records.append(identifier)
        prefix = after
    if (checkpoint.get('audited_records') != records or len(records) != generation
            or set(checkpoint['snapshot']['entries']) != set(entries)
            or set(checkpoint['snapshot']['associations']) != set(entries)
            or set(checkpoint['snapshot']['records']) != set(map(str, records))):
        raise ValueError('checkpoint cannot bless unaudited bank extras')
    return checkpoint


def bind_next_case(prepared_case, previous_raw, previous_sha256, context_raw, context_sha256, proofs):
    checkpoint = validate_checkpoint(previous_raw, previous_sha256, context_raw, context_sha256, proofs)
    context = json.loads(context_raw)
    if checkpoint['generation'] >= len(context['eligible_ids']):
        raise ValueError('eligible bank is complete')
    mid = context['eligible_ids'][checkpoint['generation']]
    if (prepared_case.get('monomer_id') != mid
            or digest(encode(prepared_case)) != context['prepared_case_sha256'][mid]
            or prepared_case.get('dataset_mode') != 'awaiting-audited-bootstrap'):
        raise ValueError('exact reconstructed next eligible bank case required')
    return dict(copy.deepcopy(prepared_case), dataset_mode='append', dataset_id=context['dataset_id'],
        bootstrap_context_sha256=context_sha256, checkpoint_sha256=previous_sha256)


def advance(previous_raw, previous_sha256, context_raw, context_sha256, proofs,
            *, receipts, expected_audit_sha256, captured, tensor_raw):
    previous = validate_checkpoint(previous_raw, previous_sha256, context_raw, context_sha256, proofs)
    context, base = json.loads(context_raw), helper('dataset_guard')
    links = _receipts(receipts)
    if (links['audit'] != expected_audit_sha256 or not _sha(expected_audit_sha256)
            or previous['generation'] >= len(context['eligible_ids'])
            or set(captured) != {'before', 'entry', 'record', 'delivered'}
            or captured['before'] != previous['snapshot']):
        raise ValueError('recorded next audit and exact predecessor captures required')
    bound = dict(context, context_sha256=context_sha256)
    link = _case_link(receipts, bound, captured['before'], captured['delivered'], previous_sha256)
    if link['monomer_id'] != context['eligible_ids'][previous['generation']]:
        raise ValueError('only the immediate eligible successor may advance')
    case = json.loads(receipts['case'])
    for role, record_id in (('entry', None), ('record', link['record_id'])):
        base.assert_append_delta(captured['before'], captured[role],
            entry_name=link['entry_name'], record_id=record_id,
            specification_name=context['specification_name'])
        base.validate_stored_entry(captured[role]['entries'][link['entry_name']], case['entry'])
    base.validate_fresh_record(captured['record']['records'][str(link['record_id'])],
        tag=json.loads(receipts['outcome'])['compute_tag'],
        specification=previous['snapshot']['specifications'][context['specification_name']]['specification'],
        entry=case['entry'])
    _tensor(receipts, tensor_raw)
    updated_proofs = dict(proofs)
    updated_proofs.update({sha: receipts[role] for role, sha in links.items()})
    result = dict(copy.deepcopy(previous), status='READ_ONLY_AUDITED_ADVANCE',
        generation=previous['generation']+1, previous_checkpoint_sha256=previous_sha256,
        audited_records=previous['audited_records']+[link['record_id']],
        ledger=previous['ledger']+[link], snapshot=copy.deepcopy(captured['delivered']),
        snapshot_sha256=base.digest(captured['delivered']))
    validate_checkpoint(encode(result), digest(encode(result)), context_raw, context_sha256, updated_proofs)
    return result, updated_proofs
