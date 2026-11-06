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
