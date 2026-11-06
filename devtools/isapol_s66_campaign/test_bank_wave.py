"""Synthetic metadata only: no QCA, scheduler, quantum calculation or credentials."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_test_wave_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accounting(actual_job):
    return ''.join(actual_job+suffix+'|COMPLETED|0:0\n' for suffix in ('', '.batch', '.0')).encode()


def fixture(tmp_path):
    core, old = load('bank_wave'), load('test_audited_bank')
    serial = core.helper('audited_bank')
    factory, inputs, pool = old.load('test_grid400_bank_case').fixture()
    eligible = [f'monomer-{i:04d}' for i in (1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15)]
    pool.update(eligible_ids=eligible,
                refused_ids=[f'monomer-{i:04d}' for i in range(1, 132) if f'monomer-{i:04d}' not in eligible])
    pool['bank']['producer'] = dict(scientific_commit=serial.SCIENCE,
        implementation_sha256=dict(core='a'*64, isapol_python='b'*64), qualification='SYNTHETIC')
    entries = json.loads(inputs['entries'])
    for index, entry in enumerate(entries['entries']):
        entry['attributes']['geometry_sha256'] = core.digest(str(index).encode())
        # Finite non-overlapping toy coordinates for real QCPortal entry-model
        # validation, never a quantum calculation or scientific reference.
        geometry = [0., 0., 0., 3.+index/1000., 0., 0., 6.+index/1000., 0., 0.]
        entry['molecule']['geometry'] = geometry
        row = pool['bank']['monomers'][index]
        row['runtime_input']['molecule']['geometry'] = list(geometry)
        row['runtime_input_sha256'] = factory.helper('prepare_descriptor_bank_cb63305').json_digest(
            row['runtime_input'])
        pool['allocation']['reports'][index]['runtime_input_sha256'] = row['runtime_input_sha256']
        pool['allocation']['reports'][index]['status'] = (
            'PASS_DESCRIPTOR_PLAN_ONLY' if f'monomer-{index+1:04d}' in eligible else 'REFUSED_DESCRIPTOR_PLAN')
    inputs['entries'] = core.encode(entries)
    pool['bank']['input_sha256']['entries'] = core.digest(inputs['entries'])
    prepared = {mid: factory.prepare_case(inputs, pool, mid) for mid in eligible}
    seed_receipts, seed_views, tensor = old.delivered(serial, prepared[eligible[0]])
    for role in ('audit', 'outcome'):
        value = json.loads(seed_receipts[role])
        value['dataset_id'] = 357
        seed_receipts[role] = core.encode(value)
    audit = json.loads(seed_receipts['audit'])
    audit['case_outcome_sha256'] = core.digest(seed_receipts['outcome'])
    seed_receipts['audit'] = core.encode(audit)
    seed_views['delivered']['metadata']['id'] = 357
    context_raw, previous = serial.initialize(pool, prepared, seed_receipts,
        core.digest(seed_receipts['audit']), seed_views['delivered'], tensor)
    previous_raw = core.encode(previous)
    proofs = {core.digest(value): value for value in seed_receipts.values()}
    mids = eligible[1:9]
    directory = tmp_path/'wave-0001-0009'
    plan = dict(schema='native-eligible-wave-plan/1', campaign_scope='eligible-wave',
        dispatch_protocol=core.PROTOCOL, wave_id='wave-0001-0009', dataset_id=357,
        source='/S', operational_commit='c'*40,
        bootstrap_context_path=str(tmp_path/'serial/bank-context.json'),
        bootstrap_context_sha256=core.digest(context_raw),
        predecessor_path=str(tmp_path/'serial/bank-checkpoint.json'),
        predecessor_sha256=core.digest(previous_raw), predecessor_snapshot_sha256=previous['snapshot_sha256'],
        generation_before=1, generation_after=9, eligible_ids=eligible, refused_ids=pool['refused_ids'],
        eligibility_sha256=previous['eligibility_sha256'], member_ids=mids, members={})
    cases = {mid: core.bind_member(prepared[mid], plan=plan, monomer_id=mid) for mid in mids}
    for mid, case in cases.items():
        plan['members'][mid] = dict(case_path=str(directory/'cases'/(mid+'.json')),
            prepared_case_sha256=core.digest(core.encode(prepared[mid])),
            case_sha256=core.digest(core.encode(case)), execution_profile=case['execution_profile'])
    plan_raw = core.encode(plan)
    args = dict(prepared_cases=prepared, checkpoint_raw=previous_raw,
        checkpoint_sha256=core.digest(previous_raw), context_raw=context_raw,
        context_sha256=core.digest(context_raw), proof_blobs=proofs)
    core.validate_plan(plan_raw, core.digest(plan_raw), **args)
    before, captures, members, completions, receipts = previous['snapshot'], {}, {}, {}, {}
    role = dict(kind='allocation', job_id='8000', allocation=dict(JobId='8000',
        Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small', NumCPUs='4',
        NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00'))
    for mid in mids:
        member_receipts, views, _ = old.delivered(serial, cases[mid], before)
        identifier = json.loads(member_receipts['outcome'])['record_id']
        for key in ('record', 'delivered'):
            views[key]['records'][str(identifier)]['wave_identity_sha256'] = core.digest(mid.encode())
        captures[mid] = {key: views[key] for key in ('before', 'entry', 'record')}
        receipts[mid] = member_receipts
        completions[mid] = views['delivered']['records'][str(identifier)]
        actual_job = str(500+int(mid[-4:]))
        members[mid] = dict(case_sha256=plan['members'][mid]['case_sha256'],
            worker_preflight_sha256='f'*64, science_job_id=actual_job,
            compute_tag='ccasp-native-grid400-bank-'+mid+'-'+actual_job,
            entry_name=cases[mid]['entry']['name'], specification_name=cases[mid]['specification_name'],
            record_id=identifier, molecule_id=9000+int(mid[-4:]),
            captures={key: dict(path=str(directory/'registration'/mid/(
                'owned-'+{'before':'before-entry', 'entry':'after-entry', 'record':'after-record'}[key]+'.json')),
                snapshot_sha256=core.helper('dataset_guard').digest(views[key]))
                for key in captures[mid]})
        before = views['record']
    reg = dict(schema='native-eligible-wave-registration/1', campaign_scope='eligible-wave',
        dispatch_protocol=core.PROTOCOL, wave_id=plan['wave_id'], plan_sha256=core.digest(plan_raw),
        dataset_id=357, bootstrap_context_sha256=core.digest(context_raw),
        predecessor_sha256=core.digest(previous_raw), predecessor_snapshot_sha256=previous['snapshot_sha256'],
        registration_role=role, dispatch_intent_sha256='d'*64, member_ids=mids, members=members,
        registered_snapshot_path=str(directory/'registration/registered-snapshot.json'),
        registered_snapshot_sha256=core.helper('dataset_guard').digest(before))
    reg_raw = core.encode(reg)
    permit = dict(schema='native-eligible-wave-release-permit/1', dispatch_protocol=core.PROTOCOL,
        wave_id=plan['wave_id'], plan_sha256=core.digest(plan_raw), registration_sha256=core.digest(reg_raw),
        dataset_id=357, predecessor_sha256=core.digest(previous_raw), dispatch_intent_sha256='d'*64,
        members=core.permit_members(reg))
    permit_raw = core.encode(permit)
    binding = dict(plan_path=directory/'plan.json', plan_raw=plan_raw, plan_sha256=core.digest(plan_raw),
        plan=plan, context_raw=context_raw, context_sha256=core.digest(context_raw),
        checkpoint_raw=previous_raw, checkpoint_sha256=core.digest(previous_raw), proof_blobs=proofs,
        monomer_id=mids[0], registration_path=directory/'registration.json', registration_raw=reg_raw,
        registration_sha256=core.digest(reg_raw), registration=reg,
        release_permit_path=directory/'release-permit.json', release_permit_raw=permit_raw,
        release_permit_sha256=core.digest(permit_raw), release_permit=permit,
        captures=captures, registered_snapshot=before, science_barrier_path=None,
        science_barrier_raw=None, science_barrier_sha256=None, science_barrier=None)
    return NS(core=core, serial=serial, prepared=prepared, previous=previous, args=args, cases=cases,
        plan=plan, binding=binding, captures=captures, reg=reg, registered=before,
        completions=completions, receipts=receipts, directory=directory)


def test_membership_is_eligible_slice_not_numeric_range(tmp_path):
    f = fixture(tmp_path)
    assert f.plan['member_ids'] == ['monomer-%04d' % i for i in (2, 4, 5, 7, 8, 9, 10, 12)]
    for mid, case in f.cases.items():
        assert case['entry'] == f.prepared[mid]['entry']
        with pytest.raises(ValueError):
            load('bank_transport').RequestScope(case, case, f.reg['members'][mid]['compute_tag'])


@pytest.mark.parametrize('change', ['membership', 'duplicate', 'refused', 'geometry', 'context',
                                    'proof', 'generation', 'raw', 'profile'])
def test_plan_rejects_changed_authority(tmp_path, change):
    f = fixture(tmp_path)
    if change == 'membership':
        f.plan['member_ids'].reverse()
    elif change == 'duplicate':
        f.plan['member_ids'][-1] = f.plan['member_ids'][0]
    elif change == 'refused':
        f.plan['member_ids'][-1] = 'monomer-0003'
    elif change == 'geometry':
        mid = f.plan['member_ids'][0]
        f.args['prepared_cases'][mid]['entry']['attributes']['geometry_sha256'] = (
            f.previous['snapshot']['entries'][next(iter(f.previous['snapshot']['entries']))]['attributes']['geometry_sha256'])
    elif change == 'context':
        f.args['context_raw'] += b' '
    elif change == 'proof':
        f.args['proof_blobs'].pop(next(iter(f.args['proof_blobs'])))
    elif change == 'generation':
        f.plan['generation_after'] += 1
    elif change == 'raw':
        f.args['checkpoint_raw'] += b' '
    else:
        f.plan['members'][f.plan['member_ids'][0]]['execution_profile']['cores'] = 100
    raw = f.core.encode(f.plan)
    with pytest.raises((ValueError, KeyError)):
        f.core.validate_plan(raw, f.core.digest(raw), **f.args)


def registration_check(f):
    raw = f.core.encode(f.reg)
    return f.core.validate_registration(f.plan, raw, f.core.digest(raw),
        predecessor_snapshot=f.previous['snapshot'], cases=f.cases,
        captures=f.captures, registered_snapshot=f.registered)


def test_actual_registration_and_job_slot(tmp_path):
    f = fixture(tmp_path)
    assert registration_check(f) == f.reg
    permit = f.core.validate_release_permit(f.plan, f.binding['plan_sha256'], f.reg,
        f.binding['registration_sha256'], f.binding['release_permit_raw'])
    mid = f.plan['member_ids'][0]
    assert f.core.validate_registered_slot(f.plan, f.reg, permit, f.cases[mid]) == f.reg['members'][mid]
    with pytest.raises(ValueError, match='actual job'):
        f.core.validate_registered_slot(f.plan, f.reg, permit, f.cases[mid], allocation=dict(JobId='123'))


@pytest.mark.parametrize('change', ['job', 'tag', 'record', 'case', 'capture', 'molecule',
                                    'running', 'role', 'prefix'])
def test_registration_rejects_unobserved_or_changed_slots(tmp_path, change):
    f = fixture(tmp_path)
    mid = f.plan['member_ids'][-1]
    slot = f.reg['members'][mid]
    if change == 'job':
        slot['science_job_id'] = f.reg['members'][f.plan['member_ids'][0]]['science_job_id']
    elif change == 'tag':
        slot['compute_tag'] += '-wrong'
    elif change == 'record':
        slot['record_id'] += 1
    elif change == 'case':
        slot['case_sha256'] = 'a'*64
    elif change == 'capture':
        f.captures[mid]['entry']['metadata']['name'] = 'changed'
    elif change == 'molecule':
        slot['molecule_id'] += 1
    elif change == 'running':
        f.registered['records'][str(slot['record_id'])]['status'] = 'running'
    elif change == 'role':
        f.reg['registration_role']['allocation']['QOS'] = 'inferno'
    else:
        f.captures[mid]['before']['records'][str(f.previous['audited_records'][0])]['body_sha256'] = 'f'*64
    with pytest.raises((ValueError, KeyError)):
        registration_check(f)


@pytest.mark.parametrize('key', ['plan_sha256', 'registration_sha256', 'dataset_id',
                                'predecessor_sha256', 'members', 'dispatch_intent_sha256'])
def test_release_permit_is_exact(tmp_path, key):
    f = fixture(tmp_path)
    permit = copy.deepcopy(f.binding['release_permit'])
    permit[key] = {} if key == 'members' else 'wrong'
    with pytest.raises(ValueError):
        f.core.validate_release_permit(f.plan, f.binding['plan_sha256'], f.reg,
                                      f.binding['registration_sha256'], f.core.encode(permit))


def capture_check(f, snapshot, **kwargs):
    return f.core.validate_capture(f.plan, f.reg, predecessor_snapshot=f.previous['snapshot'],
        registered_snapshot=f.registered, snapshot=snapshot, **kwargs)


def test_peer_completion_is_allowed_but_not_whole_wave_acceptance(tmp_path):
    f = fixture(tmp_path)
    snapshot = copy.deepcopy(f.registered)
    mid = f.plan['member_ids'][1]
    snapshot['records'][str(f.reg['members'][mid]['record_id'])] = f.completions[mid]
    summary = capture_check(f, snapshot, member_id=f.plan['member_ids'][0])
    assert summary['peer_statuses'][mid] == 'complete' and summary['preserved_records'] == 1
    with pytest.raises(ValueError, match='all wave members'):
        capture_check(f, snapshot, require_complete=True)


@pytest.mark.parametrize('change', ['stable', 'extra', 'missing', 'prefix', 'spec', 'entry',
                                    'association', 'history', 'task', 'error', 'unknown_field'])
def test_peer_identity_is_closed(tmp_path, change):
    f = fixture(tmp_path)
    snapshot = copy.deepcopy(f.registered)
    mid = f.plan['member_ids'][1]
    identifier = str(f.reg['members'][mid]['record_id'])
    snapshot['records'][identifier] = copy.deepcopy(f.completions[mid])
    record = snapshot['records'][identifier]
    if change == 'stable':
        record['wave_identity_sha256'] = 'f'*64
    elif change == 'extra':
        snapshot['records']['999999'] = record
    elif change == 'missing':
        snapshot['records'].pop(identifier)
    elif change == 'prefix':
        snapshot['records'][str(f.previous['audited_records'][0])]['body_sha256'] = 'f'*64
    elif change == 'spec':
        record['specification']['method'] = 'wrong'
    elif change == 'entry':
        snapshot['entries'][f.reg['members'][mid]['entry_name']]['attributes']['reference'] = 'wrong'
    elif change == 'association':
        snapshot['associations'][f.reg['members'][mid]['entry_name']] = {}
    elif change == 'history':
        record['history'] *= 2
    elif change == 'task':
        record['task'] = {'compute_tag': 'wrong'}
    elif change == 'error':
        record['status'] = 'error'
    else:
        record['unknown'] = 1
    with pytest.raises(ValueError):
        capture_check(f, snapshot)


@pytest.mark.parametrize('suffix', ['', '.batch', '.0'])
@pytest.mark.parametrize('change', ['missing', 'failed', 'duplicate'])
def test_accounting_requires_positive_unique_rows(suffix, change):
    core = load('bank_wave')
    text = accounting('123').decode()
    row = '123'+suffix+'|COMPLETED|0:0\n'
    if change == 'missing':
        text = text.replace(row, '')
    elif change == 'failed':
        text = text.replace(row, '123'+suffix+'|FAILED|1:0\n')
    else:
        text += row
    with pytest.raises(ValueError):
        core.completed_accounting(text.encode(), '123')


def test_lifecycle_identity_preserves_unknown_immutable_fields():
    transport = load('bank_wave_transport')
    body = dict(id=123, name='immutable', status='waiting', extras={}, future_immutable='bound')
    changed = dict(body, status='complete', extras={'psi4:native_monomer': {}})
    assert transport.stable_record_identity(body) == transport.stable_record_identity(changed)
    for key in ('name', 'id', 'future_immutable'):
        changed = dict(body, **{key: 'changed'})
        assert transport.stable_record_identity(body) != transport.stable_record_identity(changed)


@pytest.mark.parametrize('wave_first', [False, True])
def test_serial_wave_reservations_collide(monkeypatch, tmp_path, wave_first):
    store = load('bank_store')
    monkeypatch.setattr(store, 'INTENTS', tmp_path/'intents')
    f = fixture(tmp_path)
    mid = f.plan['member_ids'][0]
    slot = f.reg['members'][mid]
    fields = load('bank_wave_campaign').intent_fields(f.binding, f.reg['members'],
                                                     f.reg['registration_role'])
    serial = dict(dataset_id=357, checkpoint_sha256=f.binding['checkpoint_sha256'],
        snapshot_sha256=f.plan['predecessor_snapshot_sha256'], context_sha256=f.binding['context_sha256'],
        case_sha256=slot['case_sha256'], eligibility_sha256=f.plan['eligibility_sha256'],
        monomer_id=mid, compute_tag=slot['compute_tag'])
    if wave_first:
        monkeypatch.setenv('SLURM_JOB_ID', '8000')
        saved = store._reserve_wave_intent(**fields)
        assert store._verify_wave_intent(saved, **fields) == saved['sha256']
        monkeypatch.setenv('SLURM_JOB_ID', slot['science_job_id'])
        with pytest.raises(FileExistsError):
            store.reserve_intent(**serial)
    else:
        monkeypatch.setenv('SLURM_JOB_ID', slot['science_job_id'])
        store.reserve_intent(**serial)
        monkeypatch.setenv('SLURM_JOB_ID', '8000')
        with pytest.raises(FileExistsError):
            store._reserve_wave_intent(**fields)


@pytest.mark.parametrize('used,new,allowed', [(2,8,True), (37,8,True), (38,8,False),
                                             (45,0,True), (46,0,False), (2,9,False)])
def test_global_headroom(used, new, allowed):
    controller = load('bank_wave_controller')
    if allowed:
        controller.admit_headroom(used, new)
    else:
        with pytest.raises(ValueError):
            controller.admit_headroom(used, new)


def test_controller_barriers_and_exact_argv(tmp_path):
    f, controller = fixture(tmp_path), load('bank_wave_controller')
    mid = f.plan['member_ids'][0]
    worker = controller.phase_command(f.binding, mid, 'worker')
    sci = controller.phase_command(f.binding, mid, 'science')
    assert worker[-4:] == ['--grid400-bank-wave', str(f.binding['plan_path']), f.binding['plan_sha256'], '-']
    assert '--hold' in sci and '--wave-registration-sha256' not in sci
    assert f.binding['registration_sha256'] not in sci
    with pytest.raises(ValueError, match='whole-SCI'):
        controller.phase_command(f.binding, mid, 'audit')
    f.binding['science_barrier_sha256'] = 'e'*64
    audit = controller.phase_command(f.binding, mid, 'audit')
    assert audit[-6:] == ['--wave-registration-sha256', f.binding['registration_sha256'],
                         '--wave-science-barrier', str(f.directory/'science-barrier.json'),
                         '--wave-science-barrier-sha256', 'e'*64]
    events = []
    def remote(action, state):
        events.append((action, state.get('phase')))
        if action == 'initialize':
            return dict(wave_plan='plan', wave_plan_sha256='h')
        if action == 'submit':
            return dict(jobs={} if state['phase'] == 'science' else {state['phase']: state['phase']})
        if action == 'science-jobs':
            return dict(jobs={'science': 'science'})
        return {}
    controller.run_wave(remote, lambda actual_job: events.append(('wait', actual_job)))
    assert events.index(('verify', 'registration')) < events.index(('release', None))
    assert events.index(('wait', 'science')) < events.index(('barrier', None))
    assert events.index(('barrier', None)) < events.index(('submit', 'audit'))
    assert events.index(('verify', 'audit')) < events.index(('submit', 'finalizer'))
    assert events.index(('verify', 'finalizer')) < events.index(('accept', None))
    assert controller.PRODUCTION_ENABLED is False


def complete(f):
    """Actual-shaped synthetic receipt/capture/accounting proof chain."""
    core, binding = f.core, f.binding
    base = core.helper('dataset_guard')
    final = copy.deepcopy(f.registered)
    for mid in f.plan['member_ids']:
        final['records'][str(f.reg['members'][mid]['record_id'])] = f.completions[mid]
    outcomes, deliveries, sci_accounting, audits, audit_accounting = {}, {}, {}, {}, {}
    fields = load('bank_wave_runtime').started_fields(binding)
    for mid in f.plan['member_ids']:
        slot, case = f.reg['members'][mid], f.cases[mid]
        outcome = core.decoded(f.receipts[mid]['outcome'])
        profile = case['execution_profile']
        outcome.update(fields, dataset_id=357, campaign=case['campaign'], execution_profile=profile,
            worker_preflight_sha256=slot['worker_preflight_sha256'], qcengine_internal_retries=0,
            allocation=dict(JobId=slot['science_job_id'], Account='gts-cs207-chemx',
                QOS='embers', Partition='cpu-small', NumCPUs=str(profile['cores']), NumNodes='1',
                MinMemoryNode=str(profile['job_memory_GiB'])+'G', TimeLimit=profile['time_limit']))
        outcomes[mid] = core.encode(outcome)
        delivered = copy.deepcopy(f.registered)
        delivered['records'][str(slot['record_id'])] = f.completions[mid]
        deliveries[mid] = delivered
        sci_accounting[mid] = accounting(slot['science_job_id'])
    barrier = dict(schema='native-eligible-wave-science-barrier/1',
        status='ALL_SCIENCE_POSITIVELY_VERIFIED', wave_id=f.plan['wave_id'],
        plan_sha256=binding['plan_sha256'], registration_sha256=binding['registration_sha256'],
        release_permit_sha256=binding['release_permit_sha256'], member_ids=f.plan['member_ids'], members={})
    for mid in f.plan['member_ids']:
        run = f.directory/'members'/mid/'science'
        slot = f.reg['members'][mid]
        barrier['members'][mid] = {key: slot[key] for key in (
            'science_job_id', 'record_id', 'compute_tag', 'case_sha256')}
        barrier['members'][mid].update(outcome_path=str(run/'payload/pilot-outcome.json'),
            outcome_sha256=core.digest(outcomes[mid]), accounting_path=str(run/'accounting.txt'),
            accounting_sha256=core.digest(sci_accounting[mid]),
            delivery_snapshot_sha256=base.digest(deliveries[mid]))
    barrier_raw = core.encode(barrier)
    binding.update(science_barrier_path=f.directory/'science-barrier.json', science_barrier=barrier,
                   science_barrier_raw=barrier_raw, science_barrier_sha256=core.digest(barrier_raw))
    for index, mid in enumerate(f.plan['member_ids']):
        slot = f.reg['members'][mid]
        audit = core.decoded(f.receipts[mid]['audit'])
        audit_job = str(9000+index)
        allocation = dict(f.reg['registration_role']['allocation'], JobId=audit_job)
        audit.update(fields, dataset_id=357, campaign=f.cases[mid]['campaign'],
            status='PASS_SELECTED_WAVE_CASE_SINGLETON_EVIDENCE', audit_commit=f.plan['operational_commit'],
            allocation=allocation, compute_tag=slot['compute_tag'],
            worker_preflight_sha256=slot['worker_preflight_sha256'],
            case_outcome_sha256=core.digest(outcomes[mid]), science_barrier_sha256=core.digest(barrier_raw),
            execution_profile=f.cases[mid]['execution_profile'])
        audit['captured_preservation'].update(fields, status='PASS_CAPTURED_WAVE_DATASET_PRESERVATION',
            before_sha256=base.digest(f.registered), after_sha256=base.digest(deliveries[mid]),
            preserved_records=len(f.previous['snapshot']['records']),
            peer_statuses={peer: 'complete' if peer == mid else 'waiting' for peer in f.plan['member_ids']})
        audits[mid], audit_accounting[mid] = core.encode(audit), accounting(audit_job)
    blobs = dict(binding['proof_blobs'])
    def add(raw):
        key = core.digest(raw)
        blobs[key] = raw
        return key
    link = dict(predecessor_sha256=add(binding['checkpoint_raw']),
        plan_sha256=add(binding['plan_raw']), registration_sha256=add(binding['registration_raw']),
        permit_sha256=add(binding['release_permit_raw']), barrier_sha256=add(barrier_raw),
        registered_snapshot_sha256=add(core.encode(f.registered)),
        final_snapshot_sha256=add(core.encode(final)), receipts_sha256={}, captures_sha256={},
        deliveries_sha256={}, science_accounting_sha256={}, audit_accounting_sha256={})
    for mid in f.plan['member_ids']:
        link['receipts_sha256'][mid] = dict(audit=add(audits[mid]), outcome=add(outcomes[mid]),
            case=add(core.encode(f.cases[mid])), envelope=add(f.receipts[mid]['envelope']))
        link['captures_sha256'][mid] = {role: add(core.encode(f.captures[mid][role]))
                                       for role in ('before', 'entry', 'record')}
        link['deliveries_sha256'][mid] = add(core.encode(deliveries[mid]))
        link['science_accounting_sha256'][mid] = add(sci_accounting[mid])
        link['audit_accounting_sha256'][mid] = add(audit_accounting[mid])
    return NS(blobs=blobs, link=link, final=final, outcomes=outcomes, deliveries=deliveries,
              sci_accounting=sci_accounting, audits=audits, audit_accounting=audit_accounting)


def reduce(f, finished):
    return f.core.reduce_wave(f.binding['checkpoint_raw'], f.binding['checkpoint_sha256'],
        f.binding['context_raw'], f.binding['context_sha256'], finished.blobs, link=finished.link)


def test_whole_wave_advances_eight_once_preserving_actual_serial_anchor(tmp_path):
    f = fixture(tmp_path)
    finished = complete(f)
    checkpoint = reduce(f, finished)
    assert checkpoint['generation'] == f.previous['generation']+8 == 9
    assert checkpoint['serial_anchor_sha256'] == f.binding['checkpoint_sha256']
    assert len(checkpoint['wave_links']) == 1
    assert len(checkpoint['audited_records']) == 9
    assert checkpoint['snapshot'] == finished.final
    assert finished.blobs[checkpoint['serial_anchor_sha256']] == f.binding['checkpoint_raw']
    raw = f.core.encode(checkpoint)
    assert f.core.validate_checkpoint(raw, f.core.digest(raw), f.binding['context_raw'],
        f.binding['context_sha256'], finished.blobs) == checkpoint


@pytest.mark.parametrize('change', ['missing_audit', 'failed_audit', 'bad_numeric', 'bad_trace',
    'bad_tensor', 'changed_prefix', 'pending_peer', 'audit_accounting', 'sci_accounting',
    'plan_scope', 'registration_job', 'audit_barrier', 'outcome_dataset', 'fake_generation'])
def test_no_member_or_partial_wave_can_advance(tmp_path, change):
    f = fixture(tmp_path)
    completed = complete(f)
    mid = f.plan['member_ids'][0]
    if change == 'missing_audit':
        completed.blobs.pop(completed.link['receipts_sha256'][mid]['audit'])
    elif change in ('failed_audit', 'bad_numeric', 'bad_trace', 'audit_barrier'):
        audit = f.core.decoded(completed.audits[mid])
        if change == 'failed_audit':
            audit['status'] = 'FAIL'
        elif change == 'bad_numeric':
            audit['numeric']['status'] = 'FAIL'
        elif change == 'bad_trace':
            audit['traces']['unit']['status'] = 'failed'
        else:
            audit['science_barrier_sha256'] = 'a'*64
        raw = f.core.encode(audit)
        completed.link['receipts_sha256'][mid]['audit'] = f.core.digest(raw)
        completed.blobs[f.core.digest(raw)] = raw
    elif change in ('changed_prefix', 'pending_peer'):
        final = copy.deepcopy(completed.final)
        if change == 'changed_prefix':
            final['records'][str(f.previous['audited_records'][0])]['body_sha256'] = 'a'*64
        else:
            identifier = str(f.reg['members'][mid]['record_id'])
            final['records'][identifier] = f.registered['records'][identifier]
        raw = f.core.encode(final)
        completed.link['final_snapshot_sha256'] = f.core.digest(raw)
        completed.blobs[f.core.digest(raw)] = raw
    elif change in ('audit_accounting', 'sci_accounting'):
        key = ('audit' if change == 'audit_accounting' else 'science')+'_accounting_sha256'
        raw = b'9000|FAILED|1:0\n'
        completed.link[key][mid] = f.core.digest(raw)
        completed.blobs[f.core.digest(raw)] = raw
    elif change in ('plan_scope', 'fake_generation'):
        plan = copy.deepcopy(f.plan)
        if change == 'plan_scope':
            plan['campaign_scope'] = 'eligible-bank'
        else:
            plan['generation_after'] = 2
        raw = f.core.encode(plan)
        completed.link['plan_sha256'] = f.core.digest(raw)
        completed.blobs[f.core.digest(raw)] = raw
    elif change == 'registration_job':
        reg = copy.deepcopy(f.reg)
        reg['members'][mid]['science_job_id'] = '123'
        raw = f.core.encode(reg)
        completed.link['registration_sha256'] = f.core.digest(raw)
        completed.blobs[f.core.digest(raw)] = raw
    elif change == 'outcome_dataset':
        outcome = f.core.decoded(completed.outcomes[mid])
        outcome['dataset_id'] = 354
        raw = f.core.encode(outcome)
        completed.link['receipts_sha256'][mid]['outcome'] = f.core.digest(raw)
        completed.blobs[f.core.digest(raw)] = raw
    else:
        envelope = f.core.decoded(f.receipts[mid]['envelope'])
        envelope['artifacts']['tensors.npz']['data'] = 'd3Jvbmc='
        raw = f.core.encode(envelope)
        completed.link['receipts_sha256'][mid]['envelope'] = f.core.digest(raw)
        completed.blobs[f.core.digest(raw)] = raw
    with pytest.raises((ValueError, KeyError)):
        reduce(f, completed)


@pytest.mark.parametrize('field', ['dataset_id', 'monomer_id', 'campaign_scope', 'dispatch_protocol',
    'bootstrap_context_sha256', 'checkpoint_sha256', 'predecessor_snapshot_sha256',
    'release_permit_sha256', 'worker_preflight_sha256'])
def test_barrier_independently_checks_every_outcome_identity(tmp_path, field):
    f = fixture(tmp_path)
    completed = complete(f)
    mid = f.plan['member_ids'][0]
    outcome = f.core.decoded(completed.outcomes[mid])
    outcome[field] = 'wrong'
    completed.outcomes[mid] = f.core.encode(outcome)
    barrier = copy.deepcopy(f.binding['science_barrier'])
    barrier['members'][mid]['outcome_sha256'] = f.core.digest(completed.outcomes[mid])
    with pytest.raises(ValueError):
        f.core.validate_barrier(f.plan, f.reg, f.binding['release_permit_raw'], f.core.encode(barrier),
            outcomes=completed.outcomes, accounting=completed.sci_accounting, deliveries=completed.deliveries)


def write_fixture(f, completed):
    controller = load('bank_wave_controller')
    f.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(value)
    save(f.directory/'registration/registered-snapshot.json', f.core.encode(f.registered))
    for mid in f.plan['member_ids']:
        run, audit_run = f.directory/'members'/mid/'science', f.directory/'members'/mid/'audit'
        for name, raw in (('pilot-outcome.json', completed.outcomes[mid]),
                          ('server-native-envelope.json', f.receipts[mid]['envelope']),
                          ('owned-after-delivery.json', f.core.encode(completed.deliveries[mid]))):
            save(run/'payload'/name, raw)
        save(run/'accounting.txt', completed.sci_accounting[mid])
        save(audit_run/'payload/audit-report.json', completed.audits[mid])
        report = f.core.decoded(completed.audits[mid])
        save(audit_run/'submission.json', f.core.encode(dict(job_id=report['allocation']['JobId'],
                                                          command=controller.phase_command(f.binding, mid, 'audit'))))
        save(audit_run/'accounting.txt', completed.audit_accounting[mid])
        save(f.directory/'cases'/(mid+'.json'), f.core.encode(f.cases[mid]))
        for role, meta in f.reg['members'][mid]['captures'].items():
            save(Path(meta['path']), f.core.encode(f.captures[mid][role]))


@pytest.mark.parametrize('changed_live', [False, True])
def test_finalizer_double_live_and_proofs_before_checkpoint(monkeypatch, tmp_path, changed_live):
    f = fixture(tmp_path)
    completed = complete(f)
    write_fixture(f, completed)
    controller = load('bank_wave_controller')
    monkeypatch.setattr(controller.helper('bank_wave_campaign'), 'revalidate_binding', lambda binding: None)
    transport = controller.helper('bank_wave_transport')
    monkeypatch.setattr(transport, 'require_readonly_client', lambda *args, **kwargs: None)
    second = copy.deepcopy(completed.final)
    if changed_live:
        second['metadata']['name'] = 'changed'
    snapshots = iter([completed.final, second])
    monkeypatch.setattr(transport, 'snapshot_dataset', lambda *args, **kwargs: next(snapshots))
    events = []
    monkeypatch.setattr(controller.helper('bank_store'), '_store_wave_proofs', lambda blobs: events.append('proofs'))
    allocation = dict(f.reg['registration_role']['allocation'], JobId='10000')
    if changed_live:
        with pytest.raises(ValueError, match='double-live'):
            controller.publish_wave(object(), f.binding, allocation=allocation,
                                    save=lambda name, value: events.append(name))
        assert events == ['final-snapshot.json']
    else:
        result = controller.publish_wave(object(), f.binding, allocation=allocation,
                                        save=lambda name, value: events.append(name))
        assert result['generation'] == 9
        assert events == ['final-snapshot.json', 'proofs', 'bank-checkpoint.json', 'wave-advance.json']


@pytest.mark.parametrize('failure', [None, 'finalizer_failed', 'live_disagrees', 'missing_audit'])
def test_controller_accepts_only_positive_finalizer_and_matching_live(monkeypatch, tmp_path, failure):
    f = fixture(tmp_path)
    completed = complete(f)
    write_fixture(f, completed)
    checkpoint = reduce(f, completed)
    controller = load('bank_wave_controller')
    production = controller.Production(NS())
    monkeypatch.setattr(production, 'binding', lambda **kwargs: f.binding)
    monkeypatch.setattr(production, 'client', lambda binding: object())
    monkeypatch.setattr(controller.helper('bank_store'), '_read_wave_proofs', lambda required: completed.blobs)
    run = f.directory/'finalizer'
    (run/'payload').mkdir(parents=True, mode=0o700)
    checkpoint_raw = f.core.encode(checkpoint)
    (run/'payload/bank-checkpoint.json').write_bytes(checkpoint_raw)
    result = dict(status='PASS_AUDITED_BANK_WAVE_ADVANCE',
        allocation=dict(f.reg['registration_role']['allocation'], JobId='10000'),
        checkpoint_path=str(run/'payload/bank-checkpoint.json'),
        checkpoint_sha256=f.core.digest(checkpoint_raw), generation=9,
        plan_sha256=f.binding['plan_sha256'], registration_sha256=f.binding['registration_sha256'],
        science_barrier_sha256=f.binding['science_barrier_sha256'])
    (run/'payload/wave-advance.json').write_bytes(f.core.encode(result))
    (run/'submission.json').write_bytes(f.core.encode(dict(job_id='10000',
        command=controller.phase_command(f.binding, None, 'finalizer'))))
    finalizer_accounting = accounting('10000')
    if failure == 'finalizer_failed':
        finalizer_accounting = finalizer_accounting.replace(b'COMPLETED', b'FAILED')
    (run/'accounting.txt').write_bytes(finalizer_accounting)
    (run/'verified.json').write_bytes(f.core.encode(dict(phase='finalizer', job_id='10000',
        accounting_sha256=f.core.digest(finalizer_accounting))))
    for mid in f.plan['member_ids']:
        audit_run = f.directory/'members'/mid/'audit'
        actual_job = f.core.decoded(completed.audits[mid])['allocation']['JobId']
        (audit_run/'verified.json').write_bytes(f.core.encode(dict(phase='audit', job_id=actual_job,
            accounting_sha256=f.core.digest(completed.audit_accounting[mid]))))
    if failure == 'missing_audit':
        (f.directory/'members'/f.plan['member_ids'][-1]/'audit/verified.json').unlink()
    live = copy.deepcopy(completed.final)
    if failure == 'live_disagrees':
        live['metadata']['name'] = 'changed'
    monkeypatch.setattr(controller.helper('bank_wave_transport'), 'snapshot_dataset',
                        lambda *args, **kwargs: live)
    if failure:
        with pytest.raises((ValueError, FileNotFoundError)):
            production.accept()
        assert not (f.directory/'accepted.json').exists()
    else:
        state = production.accept()
        assert state['generation'] == 9 and state['finalizer_job_id'] == '10000'
        assert f.core.decoded((f.directory/'accepted.json').read_bytes()) == state


def test_transitive_source_inventory_keeps_original_bank_and_accounting_helper():
    campaign = load('bank_wave_campaign')
    assert set(campaign.helper('bank_campaign').ADDITIONAL_SOURCES) <= set(campaign.source_names())
    assert {'native_controller.py', 'bank_wave_ops.sbatch', 'bank_wave_transport.py'} <= set(campaign.source_names())


def test_published_checkpoint_cannot_crosspair_two_valid_successors(monkeypatch, tmp_path):
    first, second = fixture(tmp_path/'first'), fixture(tmp_path/'second')
    a, b = complete(first), complete(second)
    checkpoint_a, checkpoint_b = reduce(first, a), reduce(second, b)
    assert checkpoint_a['previous_checkpoint_sha256'] == checkpoint_b['previous_checkpoint_sha256']
    assert checkpoint_a['generation'] == checkpoint_b['generation']
    assert checkpoint_a['wave_link']['plan_sha256'] != checkpoint_b['wave_link']['plan_sha256']
    # Independently validate both; only their pairing with controller authority
    # is wrong, not the checkpoint's internal proof chain.
    for f, completed, checkpoint in ((first, a, checkpoint_a), (second, b, checkpoint_b)):
        raw = f.core.encode(checkpoint)
        f.core.validate_checkpoint(raw, f.core.digest(raw), f.binding['context_raw'],
                                  f.binding['context_sha256'], completed.blobs)
    controller = load('bank_wave_controller')
    production = controller.Production(NS())
    path = first.directory/'finalizer/payload/bank-checkpoint.json'
    path.parent.mkdir(parents=True, mode=0o700)
    raw = first.core.encode(checkpoint_b)
    path.write_bytes(raw)
    result = dict(status='PASS_AUDITED_BANK_WAVE_ADVANCE', checkpoint_path=str(path),
        checkpoint_sha256=first.core.digest(raw), generation=9,
        plan_sha256=first.binding['plan_sha256'], registration_sha256=first.binding['registration_sha256'],
        science_barrier_sha256=first.binding['science_barrier_sha256'])
    monkeypatch.setattr(controller.helper('bank_store'), '_read_wave_proofs', lambda required: b.blobs)
    with pytest.raises(ValueError, match='this actual wave'):
        production._published(first.binding, result)
    assert not (first.directory/'accepted.json').exists()


@pytest.mark.parametrize('phase', ['registration', 'finalizer'])
def test_ops_can_start_before_sbatch_returns_without_submission_metadata(monkeypatch, tmp_path, phase):
    f = fixture(tmp_path)
    completed = complete(f)
    write_fixture(f, completed)
    controller = load('bank_wave_controller')
    run = f.directory/phase
    args = NS(run=run)
    production = controller.Production(args)
    monkeypatch.setattr(production, 'binding', lambda **kwargs: f.binding)
    monkeypatch.setattr(production, 'queue', lambda: 10)
    monkeypatch.setattr(production, 'held', lambda binding: f.reg['members'])
    monkeypatch.setattr(production, '_require_verified', lambda *args: None)
    monkeypatch.setattr(production, 'client', lambda *args, **kwargs: object())
    allocation = dict(f.reg['registration_role']['allocation'], JobId='8000')
    monkeypatch.setenv('SLURM_JOB_ID', '8000')
    monkeypatch.setattr(controller.helper('phoenix_installed_checks'), 'allocation', lambda: allocation)
    monkeypatch.setattr(controller.helper('bank_wave_transport'), 'WaveScope', lambda *args: object())
    events = []
    monkeypatch.setattr(controller, 'register_once',
        lambda *args, **kwargs: events.append('started') or dict(registration_role=allocation))
    monkeypatch.setattr(controller, 'publish_wave',
        lambda *args, **kwargs: events.append('started') or dict(allocation=allocation))
    def execute(command, **kwargs):
        if '--test-only' in command:
            return NS(returncode=0, stdout='', stderr='')
        assert command[0].endswith('/sbatch')
        assert '--hold' not in command  # No broader job-control approval.
        assert (run/'intent.json').exists() and not (run/'submission.json').exists()
        events.append('sbatch-enter')
        production.operations(phase)  # Eager scheduler start, before stdout.
        events.append('sbatch-return')
        return NS(returncode=0, stdout='8000\n', stderr='')
    monkeypatch.setattr(controller.subprocess, 'run', execute)
    result = production.submit(phase)
    assert result['jobs'][phase] == '8000'
    assert events == ['sbatch-enter', 'started', 'sbatch-return']
    assert controller.read(run/'submission.json')['job_id'] == '8000'
