"""Pure bounded-wave metadata reduction. No scheduler, solver, writes or network."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

PROTOCOL = 'registered-wave/1'
SCOPE = 'eligible-wave'
DATASET = 357
_MODULES = {}


def helper(name):
    if name not in ('audited_bank', 'dataset_guard', 'grid400_bank_policy',
                    'bank_campaign', 'bank_store', 'native_controller'):
        raise ValueError('closed wave helper required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_wave_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('wave SHA256 required')
    return value


def positive(value):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError('positive exact wave integer required')
    return value


def job(value):
    if type(value) is not str or re.fullmatch('[1-9][0-9]{0,19}', value) is None:
        raise ValueError('actual wave allocation ID required')
    return value


def decoded(raw, expected=None):
    if type(raw) is not bytes or len(raw) > 64*1024**2:
        raise ValueError('bounded actual wave bytes required')
    if expected is not None and digest(raw) != sha(expected):
        raise ValueError('wave bytes differ from recorded hash')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate wave JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _same(left, right):
    if encode(left) != encode(right):
        raise ValueError('exact wave metadata differs')


def _identity(value, plan):
    if (value.get('wave_id') != plan['wave_id'] or value.get('dataset_id') != DATASET
            or value.get('dispatch_protocol') != PROTOCOL):
        raise ValueError('wave identity differs')


def _serial_required(checkpoint, context):
    required = set(context['seed_receipts_sha256'].values())
    for link in checkpoint['ledger']:
        required.update(link['receipts_sha256'].values())
    return required


def validate_checkpoint(raw, expected_sha256, context_raw, context_sha256, proofs, *, _depth=0):
    """Validate actual serial anchor or linked whole waves; no projected prefixes."""
    checkpoint, context = decoded(raw, expected_sha256), decoded(context_raw, context_sha256)
    if _depth > 16:
        raise ValueError('bounded wave chain required')
    if checkpoint.get('schema') == 'native-audited-bank-checkpoint/1':
        required = _serial_required(checkpoint, context)
        return helper('audited_bank').validate_checkpoint(
            raw, expected_sha256, context_raw, context_sha256,
            {key: proofs[key] for key in required})
    if (checkpoint.get('schema') != 'native-audited-bank-wave-checkpoint/1'
            or checkpoint.get('status') != 'READ_ONLY_AUDITED_WAVE_ADVANCE'
            or checkpoint.get('bootstrap_context_sha256') != context_sha256
            or checkpoint.get('dataset_id') != DATASET
            or checkpoint.get('eligibility_sha256') != context['eligibility_sha256']
            or checkpoint.get('campaign') != context['campaign']):
        raise ValueError('whole-wave checkpoint identity required')
    required = checkpoint.get('proof_sha256')
    if (type(required) is not list or required != sorted(set(required)) or len(required) > 4096
            or not set(required) <= set(proofs)):
        raise ValueError('complete bounded wave proofs required')
    for key in required:
        if digest(proofs[key]) != sha(key):
            raise ValueError('changed wave proof')
    previous_raw = proofs[checkpoint['previous_checkpoint_sha256']]
    previous = validate_checkpoint(previous_raw, digest(previous_raw), context_raw, context_sha256,
                                   proofs, _depth=_depth+1)
    link = checkpoint['wave_link']
    result = _reduce_link(previous, context_raw, context_sha256, proofs, link)
    _same(checkpoint, result)
    return checkpoint


def validate_plan(plan_raw, plan_sha256, *, prepared_cases, checkpoint_raw,
                  checkpoint_sha256, context_raw, context_sha256, proof_blobs):
    plan = decoded(plan_raw, plan_sha256)
    context = decoded(context_raw, context_sha256)
    previous = validate_checkpoint(checkpoint_raw, checkpoint_sha256, context_raw,
                                   context_sha256, proof_blobs)
    g = previous['generation']
    eligible, refused = context['eligible_ids'], context['refused_ids']
    mids = eligible[g:g+8]
    if (not mids or plan.get('schema') != 'native-eligible-wave-plan/1'
            or plan.get('campaign_scope') != SCOPE or plan.get('dispatch_protocol') != PROTOCOL
            or plan.get('dataset_id') != DATASET or context['dataset_id'] != DATASET
            or plan.get('generation_before') != g or type(plan.get('generation_before')) is not int
            or plan.get('generation_after') != g+len(mids)
            or plan.get('wave_id') != 'wave-%04d-%04d' % (g, g+len(mids))
            or plan.get('eligible_ids') != eligible or plan.get('refused_ids') != refused
            or plan.get('eligibility_sha256') != context['eligibility_sha256']
            or plan.get('bootstrap_context_sha256') != context_sha256
            or plan.get('predecessor_sha256') != checkpoint_sha256
            or plan.get('predecessor_snapshot_sha256') != previous['snapshot_sha256']
            or plan.get('member_ids') != mids or set(plan.get('members', {})) != set(mids)
            or set(prepared_cases) != set(eligible)
            or eligible != sorted(set(eligible)) or refused != sorted(set(refused))
            or set(eligible) & set(refused)
            or set(eligible) | set(refused) != {f'monomer-{i:04d}' for i in range(1, 132)}
            or type(plan.get('source')) is not str or not Path(plan['source']).is_absolute()
            or re.fullmatch('[0-9a-f]{40}', plan.get('operational_commit', '')) is None):
        raise ValueError('exact next bounded eligible wave required')
    names, geometries = set(), set()
    for entry in previous['snapshot']['entries'].values():
        names.add(entry['name'])
        geometries.add(entry['attributes']['geometry_sha256'])
    for mid, prepared in prepared_cases.items():
        if digest(encode(prepared)) != context['prepared_case_sha256'][mid]:
            raise ValueError('original raw prepared inventory changed')
    for mid in mids:
        case = bind_member(prepared_cases[mid], plan=plan, monomer_id=mid)
        member = plan['members'][mid]
        if (member.get('prepared_case_sha256') != digest(encode(prepared_cases[mid]))
                or member.get('case_sha256') != digest(encode(case))
                or encode(member.get('execution_profile')) != encode(case['execution_profile'])):
            raise ValueError('entire reconstructed wave case differs')
        name, geometry = case['entry']['name'], case['entry']['attributes']['geometry_sha256']
        if name in names or geometry in geometries:
            raise ValueError('duplicate wave entry or geometry')
        names.add(name)
        geometries.add(geometry)
    return plan


def bind_member(prepared_case, *, plan, monomer_id):
    if (monomer_id not in plan['member_ids'] or prepared_case.get('monomer_id') != monomer_id
            or prepared_case.get('campaign_scope') != 'eligible-bank'
            or prepared_case.get('schema') != 'native-selected-molecular-pilot/1'
            or prepared_case.get('dataset_mode') != 'awaiting-audited-bootstrap'):
        raise ValueError('unmodified eligible prepared wave member required')
    return dict(copy.deepcopy(prepared_case), campaign_scope=SCOPE, dispatch_protocol=PROTOCOL,
        dataset_mode='append', dataset_id=DATASET, wave_id=plan['wave_id'],
        bootstrap_context_sha256=plan['bootstrap_context_sha256'],
        checkpoint_sha256=plan['predecessor_sha256'],
        predecessor_snapshot_sha256=plan['predecessor_snapshot_sha256'],
        prepared_case_sha256=digest(encode(prepared_case)))


def validate_registration(plan, registration_raw, registration_sha256, *,
                          predecessor_snapshot, cases, captures, registered_snapshot):
    reg = decoded(registration_raw, registration_sha256)
    _identity(reg, plan)
    base = helper('dataset_guard')
    if (reg.get('schema') != 'native-eligible-wave-registration/1'
            or reg.get('campaign_scope') != SCOPE or reg.get('member_ids') != plan['member_ids']
            or set(reg.get('members', {})) != set(plan['member_ids'])
            or set(captures) != set(plan['member_ids']) or set(cases) != set(plan['member_ids'])
            or reg.get('bootstrap_context_sha256') != plan['bootstrap_context_sha256']
            or reg.get('predecessor_sha256') != plan['predecessor_sha256']
            or reg.get('predecessor_snapshot_sha256') != base.digest(predecessor_snapshot)
            or base.digest(predecessor_snapshot) != plan['predecessor_snapshot_sha256']):
        raise ValueError('exact wave registration predecessor and membership required')
    sha(reg['plan_sha256'])
    sha(reg['dispatch_intent_sha256'])
    role = reg['registration_role']
    if role.get('kind') != 'allocation':
        raise ValueError('actual registration allocation required')
    helper('native_controller').admit_audit_allocation(role['allocation'], job(role['job_id']))
    before, jobs, records, tags = predecessor_snapshot, set(), set(), set()
    for mid in plan['member_ids']:
        slot, case, views = reg['members'][mid], cases[mid], captures[mid]
        name, spec = case['entry']['name'], case['specification_name']
        identifier, actual_job = positive(slot['record_id']), job(slot['science_job_id'])
        positive(slot['molecule_id'])
        tag = 'ccasp-native-grid400-bank-'+mid+'-'+actual_job
        if (set(views) != {'before', 'entry', 'record'}
                or slot['case_sha256'] != plan['members'][mid]['case_sha256']
                or slot['entry_name'] != name or slot['specification_name'] != spec
                or slot['compute_tag'] != tag or actual_job == role['job_id']
                or actual_job in jobs or identifier in records or tag in tags):
            raise ValueError('unique actual registered wave slot required')
        sha(slot['worker_preflight_sha256'])
        jobs.add(actual_job)
        records.add(identifier)
        tags.add(tag)
        _same(before, views['before'])
        for key in ('before', 'entry', 'record'):
            if slot['captures'][key]['snapshot_sha256'] != base.digest(views[key]):
                raise ValueError('actual registration capture commitment differs')
        base.assert_append_delta(before, views['entry'], entry_name=name, specification_name=spec)
        base.assert_append_delta(before, views['record'], entry_name=name, record_id=identifier,
                                 specification_name=spec)
        for view in (views['entry'], views['record']):
            base.validate_stored_entry(view['entries'][name], case['entry'])
        _same(views['entry']['entries'][name], views['record']['entries'][name])
        fresh = views['record']['records'][str(identifier)]
        base.validate_fresh_record(fresh, tag=tag,
            specification=helper('audited_bank').normalize_specification(case['specification']),
            entry=case['entry'])
        if fresh['molecule_id'] != slot['molecule_id']:
            raise ValueError('registered molecule identity differs')
        sha(fresh['wave_identity_sha256'])
        before = views['record']
    _same(before, registered_snapshot)
    if base.digest(registered_snapshot) != reg['registered_snapshot_sha256']:
        raise ValueError('actual all-waiting registration snapshot required')
    return reg


def permit_members(registration):
    return {mid: {key: registration['members'][mid][key] for key in (
        'science_job_id', 'compute_tag', 'record_id', 'case_sha256')}
        for mid in registration['member_ids']}


def validate_release_permit(plan, plan_sha256, registration, registration_sha256, permit_raw):
    permit = decoded(permit_raw)
    expected = dict(schema='native-eligible-wave-release-permit/1', dispatch_protocol=PROTOCOL,
        wave_id=plan['wave_id'], plan_sha256=plan_sha256,
        registration_sha256=registration_sha256, dataset_id=DATASET,
        predecessor_sha256=plan['predecessor_sha256'],
        dispatch_intent_sha256=registration['dispatch_intent_sha256'],
        members=permit_members(registration))
    _same(permit, expected)
    if registration['plan_sha256'] != plan_sha256:
        raise ValueError('registration must bind actual plan bytes')
    return permit


def validate_registered_slot(plan, registration, release_permit, draft, *, allocation=None):
    mid = draft['monomer_id']
    if mid not in plan['member_ids']:
        raise ValueError('undeclared wave member')
    slot = registration['members'][mid]
    _identity(draft, plan)
    if (draft.get('campaign_scope') != SCOPE or draft.get('dataset_mode') != 'append'
            or digest(encode(draft)) != slot['case_sha256']
            or release_permit['members'] != permit_members(registration)
            or draft['checkpoint_sha256'] != plan['predecessor_sha256']
            or draft['bootstrap_context_sha256'] != plan['bootstrap_context_sha256']):
        raise ValueError('exact registered wave case required')
    if allocation is not None:
        if allocation.get('JobId') != slot['science_job_id']:
            raise ValueError('SCI allocation differs from registered actual job')
        profile = helper('grid400_bank_policy').validate_profile(draft['execution_profile'])
        expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
            NumCPUs=str(profile['cores']), NumNodes='1',
            MinMemoryNode=str(profile['job_memory_GiB'])+'G', TimeLimit=profile['time_limit'])
        if any(allocation.get(key) != value for key, value in expected.items()):
            raise ValueError('actual finite SCI route differs')
    return copy.deepcopy(slot)


def validate_capture(plan, registration, *, predecessor_snapshot, registered_snapshot,
                     snapshot, member_id=None, require_complete=False):
    """Peers may compute, but immutable full-body identities may not change."""
    base = helper('dataset_guard')
    if set(snapshot) != set(registered_snapshot):
        raise ValueError('unknown snapshot fields')
    for key in ('schema', 'metadata', 'specifications', 'entries', 'associations'):
        _same(snapshot[key], registered_snapshot[key])
    if set(snapshot['records']) != set(registered_snapshot['records']):
        raise ValueError('unknown or missing wave record')
    for identifier, previous in predecessor_snapshot['records'].items():
        _same(snapshot['records'][identifier], previous)
    statuses = {}
    keys = {'id', 'molecule_id', 'status', 'body_sha256', 'native_envelope_sha256',
            'tensor_sha256', 'history', 'specification', 'task', 'wave_identity_sha256'}
    for mid in plan['member_ids']:
        slot = registration['members'][mid]
        identifier = str(slot['record_id'])
        prior, now = registered_snapshot['records'][identifier], snapshot['records'][identifier]
        if set(now) != keys or set(prior) != keys:
            raise ValueError('closed wave record snapshot required')
        for key in ('id', 'molecule_id', 'specification', 'wave_identity_sha256'):
            _same(now[key], prior[key])
        sha(now['body_sha256'])
        sha(now['wave_identity_sha256'])
        status, history = now['status'], now['history']
        if status not in ('waiting', 'running', 'complete'):
            raise ValueError('wave failure/retry state cannot be preserved')
        if require_complete and status != 'complete':
            raise ValueError('all wave members must be completed')
        if status == 'waiting':
            _same(now, prior)
        elif status == 'running':
            # A manager may create a running history before it delivers.
            if (len(history) > 1 or now['task'] is None
                    or now['tensor_sha256'] is not None
                    or now['native_envelope_sha256'] != base.digest(None)):
                raise ValueError('running wave lifecycle differs')
            _same(now['task'], prior['task'])
            if history and history[0].get('status') not in ('running', None):
                raise ValueError('non-running peer history')
        else:
            if (len(history) != 1 or not history[0].get('manager_name')
                    or history[0].get('status') not in ('complete', None)
                    or now['task'] is not None):
                raise ValueError('single completed wave history required')
            sha(now['tensor_sha256'])
            sha(now['native_envelope_sha256'])
        statuses[mid] = status
    if member_id is not None and member_id not in statuses:
        raise ValueError('undeclared capture owner')
    return dict(snapshot_sha256=base.digest(snapshot),
        preserved_records=len(predecessor_snapshot['records']), peer_statuses=statuses)


def completed_accounting(raw, actual_job):
    if type(raw) is not bytes or len(raw) > 1024**2:
        raise ValueError('bounded actual accounting response required')
    rows = [line.strip().split('|')[:3] for line in raw.decode().splitlines() if line.strip()]
    for suffix in ('', '.batch', '.0'):
        found = [row for row in rows if row[0] == job(actual_job)+suffix]
        if found != [[actual_job+suffix, 'COMPLETED', '0:0']]:
            raise ValueError('unique positive allocation/batch/.0 accounting required')
    if any(len(row) != 3 or row[0] not in {
            actual_job, actual_job+'.batch', actual_job+'.0', actual_job+'.extern'}
            or row[1:] != ['COMPLETED', '0:0'] for row in rows):
        raise ValueError('unknown or unsuccessful accounting row')


def _member_receipts(plan, reg, permit, barrier_sha, mid, receipts, delivered, final):
    """Scientific evidence checks are unchanged; authority is explicitly wave-only."""
    serial, base = helper('audited_bank'), helper('dataset_guard')
    if set(receipts) != {'audit', 'outcome', 'case', 'envelope'}:
        raise ValueError('complete member audit receipts required')
    audit, outcome, case, envelope = (decoded(receipts[key])
                                    for key in ('audit', 'outcome', 'case', 'envelope'))
    slot = validate_registered_slot(plan, reg, permit, case, allocation=outcome['allocation'])
    artifact = envelope.get('artifacts', {}).get('tensors.npz', {})
    tensor = base64.b64decode(artifact['data'], validate=True)
    if (len(tensor) != artifact['size'] or len(tensor) > 16*1024**2
            or digest(tensor) != artifact['sha256']):
        raise ValueError('actual audited member tensor bytes differ')
    common = dict(campaign_scope=SCOPE, dispatch_protocol=PROTOCOL, dataset_mode='append',
        producer_profile='grid400', wave_id=plan['wave_id'], dataset_id=DATASET,
        monomer_id=mid, record_id=slot['record_id'], compute_tag=slot['compute_tag'],
        bootstrap_context_sha256=plan['bootstrap_context_sha256'],
        checkpoint_sha256=plan['predecessor_sha256'],
        predecessor_snapshot_sha256=plan['predecessor_snapshot_sha256'],
        plan_sha256=reg['plan_sha256'], registration_sha256=permit['registration_sha256'],
        release_permit_sha256=digest(encode(permit)),
        dispatch_intent_sha256=reg['dispatch_intent_sha256'],
        runtime_unchanged=True, scientific_build_commit=serial.SCIENCE,
        pilot_sha256=digest(receipts['case']),
        worker_preflight_sha256=slot['worker_preflight_sha256'],
        execution_profile=case['execution_profile'])
    # Permit hash is an actual-byte commitment supplied on reports; callers
    # compare it to the stored permit raw, not a re-encoded projection.
    common.pop('release_permit_sha256')
    for item in (audit, outcome):
        if any(encode(item.get(key)) != encode(value) for key, value in common.items()):
            raise ValueError('member report wave bindings differ')
    preservation = audit.get('captured_preservation', {})
    if (audit.get('status') != 'PASS_SELECTED_WAVE_CASE_SINGLETON_EVIDENCE'
            or audit.get('case_job') != slot['science_job_id']
            or audit.get('case_outcome_sha256') != digest(receipts['outcome'])
            or audit.get('science_barrier_sha256') != barrier_sha
            or audit.get('audit_commit') != plan['operational_commit']
            or audit.get('case_operational_commit') != plan['operational_commit']
            or outcome.get('operational_commit') != plan['operational_commit']
            or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
            or outcome.get('maximum_records') != 1 or outcome.get('find_existing') is not False
            or outcome.get('qcengine_internal_retries') != 0
            or outcome.get('runtime_input_sha256') != case['runtime_input_sha256']
            or audit.get('file_gate_sha256') != serial.FILE_GATE_SHA256
            or audit.get('roles', {}).get('molecular_qcschema') != 1
            or not audit.get('traces')
            or any(t.get('status') != 'passed' for t in audit['traces'].values())
            or audit.get('numeric', {}).get('status') != 'PASS_NUMERIC_TRANSPORT_ONLY'
            or audit['numeric'].get('tensor_sha256') != artifact['sha256']
            or audit.get('envelope_identity', {}).get('status') != 'PASS_NATIVE_ENVELOPE_IDENTITY'
            or audit['envelope_identity'].get('policy') != 'exact-json-diagnostic-number/1'
            or audit['envelope_identity'].get('epsilon_tolerance_used') is not False
            or audit['envelope_identity'].get('server_json_sha256') != digest(encode(envelope))
            or envelope.get('schema') != 'native-monomer-artifacts/2'
            or envelope.get('request') != case['entry']['additional_keywords']['function_kwargs']['native_monomer']
            or envelope.get('implementation', {}).get('sha256') != case['producer']['implementation_sha256']
            or outcome.get('tensors_sha256') != artifact['sha256']
            or outcome.get('tensor_bytes') != artifact['size']
            or preservation.get('status') != 'PASS_CAPTURED_WAVE_DATASET_PRESERVATION'
            or preservation.get('after_sha256') != base.digest(delivered)
            or preservation.get('audited_native_envelope_sha256') != base.digest(envelope)
            or preservation.get('audited_tensor_sha256') != artifact['sha256']):
        raise ValueError('ordinary member audit/artifact evidence differs')
    record = delivered['records'][str(slot['record_id'])]
    if (record['status'] != 'complete' or len(record['history']) != 1
            or record['history'][0].get('manager_name') != outcome['manager']
            or record['native_envelope_sha256'] != base.digest(envelope)
            or record['tensor_sha256'] != artifact['sha256']):
        raise ValueError('audited member delivery differs')
    _same(record, final['records'][str(slot['record_id'])])


def _reduce_link(previous, context_raw, context_sha256, proofs, link):
    """Revalidate one entire wave from actual content-addressed observations."""
    context = decoded(context_raw)
    get = lambda key: proofs[link[key]]
    plan_raw, reg_raw, permit_raw = get('plan_sha256'), get('registration_sha256'), get('permit_sha256')
    plan, reg, permit = map(decoded, (plan_raw, reg_raw, permit_raw))
    # The whole original inventory is reconstructed by campaign at entry.
    # Here case commitments must match the unchanged bootstrap inventory.
    mids = context['eligible_ids'][previous['generation']:previous['generation']+8]
    if (plan['member_ids'] != mids or plan['generation_before'] != previous['generation']
            or plan['generation_after'] != previous['generation']+len(mids)
            or plan['predecessor_sha256'] != link['predecessor_sha256']
            or plan['bootstrap_context_sha256'] != context_sha256
            or plan['predecessor_snapshot_sha256'] != previous['snapshot_sha256']
            or plan['eligible_ids'] != context['eligible_ids'] or plan['refused_ids'] != context['refused_ids']):
        raise ValueError('whole-wave successor differs')
    cases, receipts, captures = {}, {}, {}
    for mid in mids:
        receipts[mid] = {role: proofs[key] for role, key in link['receipts_sha256'][mid].items()}
        cases[mid] = decoded(receipts[mid]['case'])
        raw_case = copy.deepcopy(cases[mid])
        for key in ('dispatch_protocol', 'dataset_id', 'wave_id', 'bootstrap_context_sha256',
                    'checkpoint_sha256', 'predecessor_snapshot_sha256', 'prepared_case_sha256'):
            raw_case.pop(key, None)
        raw_case.update(campaign_scope='eligible-bank', dataset_mode='awaiting-audited-bootstrap')
        if (digest(encode(raw_case)) != context['prepared_case_sha256'][mid]
                or encode(bind_member(raw_case, plan=plan, monomer_id=mid)) != encode(cases[mid])):
            raise ValueError('wave proof raw scientific case differs')
        captures[mid] = {role: decoded(proofs[key])
                         for role, key in link['captures_sha256'][mid].items()}
    registered = decoded(get('registered_snapshot_sha256'))
    validate_registration(plan, reg_raw, digest(reg_raw), predecessor_snapshot=previous['snapshot'],
        cases=cases, captures=captures, registered_snapshot=registered)
    validate_release_permit(plan, digest(plan_raw), reg, digest(reg_raw), permit_raw)
    barrier = decoded(get('barrier_sha256'))
    validate_barrier(plan, reg, permit_raw, get('barrier_sha256'),
                     outcomes={mid: receipts[mid]['outcome'] for mid in mids},
                     accounting={mid: proofs[link['science_accounting_sha256'][mid]] for mid in mids},
                     deliveries={mid: decoded(proofs[link['deliveries_sha256'][mid]]) for mid in mids})
    final = decoded(get('final_snapshot_sha256'))
    validate_capture(plan, reg, predecessor_snapshot=previous['snapshot'],
                     registered_snapshot=registered, snapshot=final, require_complete=True)
    for mid in mids:
        delivered = decoded(proofs[link['deliveries_sha256'][mid]])
        validate_capture(plan, reg, predecessor_snapshot=previous['snapshot'],
                         registered_snapshot=registered, snapshot=delivered, member_id=mid)
        _member_receipts(plan, reg, permit, link['barrier_sha256'], mid, receipts[mid], delivered, final)
        for role in ('audit', 'outcome'):
            if decoded(receipts[mid][role])['release_permit_sha256'] != digest(permit_raw):
                raise ValueError('actual permit bytes changed')
        report = decoded(receipts[mid]['audit'])
        helper('native_controller').admit_audit_allocation(report['allocation'], report['allocation']['JobId'])
        completed_accounting(proofs[link['audit_accounting_sha256'][mid]], report['allocation']['JobId'])
    inherited = (set(previous['proof_sha256']) if 'proof_sha256' in previous
                 else _serial_required(previous, context))
    required = set(inherited) | {link['predecessor_sha256']}
    for key in ('plan_sha256', 'registration_sha256', 'permit_sha256', 'barrier_sha256',
                'registered_snapshot_sha256', 'final_snapshot_sha256'):
        required.add(link[key])
    for field in ('science_accounting_sha256', 'audit_accounting_sha256', 'deliveries_sha256'):
        required.update(link[field].values())
    for field in ('receipts_sha256', 'captures_sha256'):
        for values in link[field].values():
            required.update(values.values())
    anchor = previous.get('serial_anchor_sha256', link['predecessor_sha256'])
    return dict(schema='native-audited-bank-wave-checkpoint/1', status='READ_ONLY_AUDITED_WAVE_ADVANCE',
        campaign=context['campaign'], dataset_id=DATASET,
        eligibility_sha256=context['eligibility_sha256'], bootstrap_context_sha256=context_sha256,
        serial_anchor_sha256=anchor, previous_checkpoint_sha256=link['predecessor_sha256'],
        wave_links=previous.get('wave_links', [])+[digest(encode(link))], wave_link=copy.deepcopy(link),
        generation=previous['generation']+len(mids),
        audited_records=previous['audited_records']+[reg['members'][mid]['record_id'] for mid in mids],
        snapshot=final, snapshot_sha256=helper('dataset_guard').digest(final),
        proof_sha256=sorted(required))


def reduce_wave(previous_raw, previous_sha256, context_raw, context_sha256, proofs, *, link):
    previous = validate_checkpoint(previous_raw, previous_sha256, context_raw, context_sha256, proofs)
    if link['predecessor_sha256'] != previous_sha256 or proofs[previous_sha256] != previous_raw:
        raise ValueError('actual predecessor bytes required')
    return _reduce_link(previous, context_raw, context_sha256, proofs, link)


def validate_barrier(plan, reg, permit_raw, barrier_raw, *, outcomes, accounting, deliveries):
    barrier, permit = decoded(barrier_raw), decoded(permit_raw)
    if (barrier.get('schema') != 'native-eligible-wave-science-barrier/1'
            or barrier.get('status') != 'ALL_SCIENCE_POSITIVELY_VERIFIED'
            or barrier.get('wave_id') != plan['wave_id']
            or barrier.get('member_ids') != plan['member_ids']
            or set(barrier.get('members', {})) != set(plan['member_ids'])
            or barrier.get('plan_sha256') != reg['plan_sha256']
            or barrier.get('registration_sha256') != permit['registration_sha256']
            or barrier.get('release_permit_sha256') != digest(permit_raw)):
        raise ValueError('all-SCI-positive barrier required')
    if any(set(values) != set(plan['member_ids']) for values in (outcomes, accounting, deliveries)):
        raise ValueError('exact all-member SCI proof inventory required')
    for mid in plan['member_ids']:
        slot, member = reg['members'][mid], barrier['members'][mid]
        outcome = decoded(outcomes[mid], member['outcome_sha256'])
        completed_accounting(accounting[mid], slot['science_job_id'])
        for key in ('science_job_id', 'record_id', 'compute_tag', 'case_sha256'):
            if member[key] != slot[key]:
                raise ValueError('SCI barrier member identity differs')
        if (digest(accounting[mid]) != member['accounting_sha256']
                or helper('dataset_guard').digest(deliveries[mid]) != member['delivery_snapshot_sha256']
                or outcome.get('allocation', {}).get('JobId') != slot['science_job_id']
                or outcome.get('dataset_id') != DATASET or outcome.get('monomer_id') != mid
                or outcome.get('campaign_scope') != SCOPE
                or outcome.get('dispatch_protocol') != PROTOCOL
                or outcome.get('producer_profile') != 'grid400'
                or outcome.get('dataset_mode') != 'append'
                or outcome.get('wave_id') != plan['wave_id']
                or outcome.get('bootstrap_context_sha256') != plan['bootstrap_context_sha256']
                or outcome.get('checkpoint_sha256') != plan['predecessor_sha256']
                or outcome.get('predecessor_snapshot_sha256') != plan['predecessor_snapshot_sha256']
                or outcome.get('dispatch_intent_sha256') != reg['dispatch_intent_sha256']
                or outcome.get('record_id') != slot['record_id']
                or outcome.get('compute_tag') != slot['compute_tag']
                or outcome.get('pilot_sha256') != slot['case_sha256']
                or outcome.get('worker_preflight_sha256') != slot['worker_preflight_sha256']
                or outcome.get('plan_sha256') != reg['plan_sha256']
                or outcome.get('registration_sha256') != permit['registration_sha256']
                or outcome.get('release_permit_sha256') != digest(permit_raw)
                or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
                or outcome.get('runtime_unchanged') is not True
                or outcome.get('find_existing') is not False
                or outcome.get('maximum_records') != 1
                or outcome.get('qcengine_internal_retries') != 0):
            raise ValueError('SCI barrier delivery proof differs')
        record = deliveries[mid]['records'][str(slot['record_id'])]
        if (record['status'] != 'complete' or len(record['history']) != 1
                or record['history'][0].get('manager_name') != outcome.get('manager')
                or record['tensor_sha256'] != outcome.get('tensors_sha256')
                or record['native_envelope_sha256'] == helper('dataset_guard').digest(None)):
            raise ValueError('SCI barrier requires own positively completed delivery')
    return barrier
