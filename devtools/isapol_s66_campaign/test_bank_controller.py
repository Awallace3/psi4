"""Bounded controller tests: metadata and injected scheduler only, never SCI."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_controller_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(monkeypatch, tmp_path):
    controller, fixtures = load('bank_controller'), load('test_audited_bank')
    core, pool, cases = fixtures.setup()
    campaign = controller.helper('bank_campaign')
    monkeypatch.setattr(campaign, 'CAMPAIGN_ROOT', tmp_path/'campaign')
    monkeypatch.setattr(campaign, 'CASE_ROOT', tmp_path/'campaign/cases')
    monkeypatch.setattr(campaign, 'prepared_cases', lambda p, inputs: copy.deepcopy(cases))
    monkeypatch.setattr(controller, 'pool_inputs', lambda: (copy.deepcopy(pool), {}))
    args = NS(source=Path('/unit/source'), commit='c'*40, adapter_seal='d'*64,
              monomer='monomer-0001', phase='worker')
    return controller, args, cases, pool, core


@pytest.mark.parametrize('profile', ['s20', 's40', 's60', 's80', 's120'])
@pytest.mark.parametrize('phase', ['worker', 'science', 'audit'])
@pytest.mark.parametrize('append', [False, True])
def test_commands_use_finite_embers_resources_and_exact_wrapper_signatures(
        monkeypatch, tmp_path, profile, phase, append):
    controller, args, cases, _, _ = setup(monkeypatch, tmp_path)
    campaign = controller.helper('bank_campaign')
    case = cases['monomer-0001']
    policy = controller.helper('phoenix_worker_cb63305').helper('grid400_bank_policy')
    case['execution_profile'] = policy.execution_profile(profile)
    case['dataset_mode'] = 'append' if append else 'fresh'
    state = dict(generation=1 if append else 0, bank_context=str(tmp_path/'bank-context.json'),
        bank_context_sha256='a'*64, bank_checkpoint=str(tmp_path/'bank-checkpoint.json'),
        bank_checkpoint_sha256='b'*64)
    command = controller.phase_command(args.source, args.commit, case, state, phase)
    assert '--qos=embers' in command and '--no-requeue' in command
    assert not any('afterok' in item or 'inferno' in item or 'accepted-' in item for item in command)
    expected = case['execution_profile']
    assert '--cpus-per-task='+str(4 if phase == 'audit' else expected['cores']) in command
    assert '--mem='+str(24 if phase == 'audit' else expected['job_memory_GiB'])+'G' in command
    assert '--time='+(expected['time_limit'] if phase == 'science' else '00:45:00') in command
    index = next(i for i, value in enumerate(command) if value.endswith('.sbatch'))
    argv = command[index+1:]
    assert argv[:3] == [str(args.source), args.commit, str(campaign.CASE_ROOT/'monomer-0001.json')]
    base = 4 if phase == 'worker' else 5 if phase == 'science' else 6
    if append:
        assert argv[base:] == ['--grid400-bank-append', state['bank_context'], 'a'*64,
                              state['bank_checkpoint'], 'b'*64]
    else:
        assert argv[base:] == ['--grid400-bank-fresh']


def test_initialize_select_and_partial_selection_cannot_be_replayed(monkeypatch, tmp_path):
    controller, args, _, pool, _ = setup(monkeypatch, tmp_path)
    initial = controller.initialize(args)
    assert initial['generation'] == 0 and initial['status'] == 'NO_PRODUCTION_MODEL_ACCEPTED'
    selected = controller.select(args)
    assert selected['monomer'] == pool['eligible_ids'][0] and selected['complete'] is False
    with pytest.raises(FileExistsError):
        controller.select(args)
    with pytest.raises(FileExistsError):
        controller.initialize(args)
    assert not list(controller.root().glob('*worker*'))


@pytest.mark.parametrize('field,value', [('commit', 'e'*40), ('adapter_sha256', 'e'*64),
    ('eligible_ids', ['monomer-0002']), ('prepared_sha256', {})])
def test_plan_tamper_fails_before_selection(monkeypatch, tmp_path, field, value):
    controller, args, _, _, _ = setup(monkeypatch, tmp_path)
    controller.initialize(args)
    path = controller.root()/'plan.json'
    plan = json.loads(path.read_bytes())
    plan[field] = value
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='plan or source'):
        controller.select(args)
    assert not list((controller.root()/'cases').iterdir())


@pytest.mark.parametrize('failure', ['preflight', 'ambiguous', 'queue', 'intent_sync'])
def test_submission_failure_never_retries_or_advances(monkeypatch, tmp_path, failure):
    controller, args, _, _, _ = setup(monkeypatch, tmp_path)
    controller.initialize(args)
    controller.select(args)
    calls = []
    monkeypatch.setattr(controller.subprocess, 'check_output',
        lambda *a, **kw: 'embers\n'*49 if failure == 'queue' else '')
    def run(command, **kwargs):
        calls.append(command)
        if '--test-only' in command:
            if failure == 'preflight':
                raise RuntimeError('synthetic preflight failure')
            return NS(returncode=0, stdout='', stderr='')
        return NS(returncode=1, stdout='possibly accepted', stderr='synthetic failure')
    monkeypatch.setattr(controller.subprocess, 'run', run)
    if failure == 'intent_sync':
        io = controller.helper('native_controller')
        original = io.write_once
        def write(path, value):
            original(path, value)
            if path.name == 'intent.json':
                raise OSError('synthetic fsync failure')
        monkeypatch.setattr(io, 'write_once', write)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        controller.submit(args)
    assert len([call for call in calls if '--test-only' not in call]) == (1 if failure == 'ambiguous' else 0)
    assert controller.context(args)[1]['generation'] == 0
    if failure != 'queue':
        with pytest.raises(FileExistsError):
            controller.submit(args)


def test_submission_is_exclusive_and_science_needs_positive_worker_proof(monkeypatch, tmp_path):
    controller, args, _, _, _ = setup(monkeypatch, tmp_path)
    controller.initialize(args)
    controller.select(args)
    calls = []
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw: '')
    monkeypatch.setattr(controller.subprocess, 'run', lambda command, **kw:
        calls.append(command) or NS(returncode=0, stdout='' if '--test-only' in command else '123\n', stderr=''))
    assert controller.submit(args)['job_id'] == '123'
    with pytest.raises(FileExistsError):
        controller.submit(args)
    assert len(calls) == 2
    args.phase = 'science'
    with pytest.raises(ValueError, match='bounded regular'):
        controller.submit(args)
    assert len(calls) == 2


def test_run_chunk_accepts_only_bounded_positive_audit_progress():
    controller = load('bank_controller')
    calls, waits = [], []
    generation = 0
    def remote(action, fields):
        nonlocal generation
        calls.append((action, fields))
        if action == 'select':
            return dict(complete=False, monomer=f'monomer-{generation+1:04d}')
        if action == 'submit':
            return dict(job_id=str(len(calls)))
        if fields['phase'] == 'audit':
            generation += 1
        return dict(generation=generation)
    result = controller.run_chunk(remote, waits.append, 2)
    assert result['generation'] == 2 and result['chunk_complete']
    assert len(waits) == 6
    assert [fields['phase'] for action, fields in calls if action == 'verify'] == list(controller.PHASES)*2
    for maximum in (0, 9, True):
        with pytest.raises(ValueError):
            controller.run_chunk(remote, waits.append, maximum)


@pytest.mark.parametrize('failed_phase', ['worker', 'science', 'audit'])
def test_any_failed_phase_stops_without_next_case_or_retry(failed_phase):
    controller, calls = load('bank_controller'), []
    def remote(action, fields):
        calls.append((action, fields))
        if action == 'select':
            return dict(complete=False, monomer='monomer-0001')
        if action == 'verify' and fields['phase'] == failed_phase:
            raise RuntimeError('synthetic phase failure')
        return dict(job_id='123')
    with pytest.raises(RuntimeError):
        controller.run_chunk(remote, lambda job: None, 8)
    assert len([1 for action, _ in calls if action == 'select']) == 1
    assert calls[-1] == ('verify', dict(monomer='monomer-0001', phase=failed_phase))


def test_full_eligible_subset_completion_is_not_whole_bank_completion():
    controller = load('bank_controller')
    result = controller.run_chunk(lambda action, state: dict(complete=True, generation=116,
        eligible_count=116, refused_ids=['monomer-0066'], whole_bank_complete=False),
        lambda job: pytest.fail('no jobs after eligible completion'), 1)
    assert result['complete'] and not result['whole_bank_complete']


def test_unsealed_remote_controller_cannot_reach_metadata_or_source_reads(monkeypatch):
    controller, campaign = load('bank_controller'), load('bank_campaign')
    monkeypatch.setattr(campaign, 'ALLOCATION_SHA256', None)
    calls = []
    def helper(name):
        calls.append(name)
        if name != 'bank_campaign':
            pytest.fail('unsealed controller reached '+name)
        return campaign
    monkeypatch.setattr(controller, 'helper', helper)
    monkeypatch.setattr('sys.argv', ['controller', 'remote', '--action', 'initialize',
        '--source', '/unit/source', '--commit', 'a'*40, '--adapter-seal', 'b'*64])
    with pytest.raises(ValueError, match='has not been sealed'):
        controller.main()
    assert calls == ['bank_campaign']


def first_state_case(monkeypatch, tmp_path, change):
    controller, args, cases, pool, core = setup(monkeypatch, tmp_path)
    fixtures = load('test_audited_bank')
    io, campaign = controller.helper('native_controller'), controller.helper('bank_campaign')
    controller.initialize(args)
    selection = controller.select(args)
    state = selection['predecessor']
    case = cases['monomer-0001']
    for phase in controller.PHASES:
        path = controller.phase_path(args.monomer, phase)
        path.mkdir()
        (path/'payload').mkdir()
    worker = controller.phase_path(args.monomer, 'worker')
    io.write_once(worker/'payload/worker-preflight.json', {})
    science = controller.phase_path(args.monomer, 'science')
    receipts, views, tensor = fixtures.delivered(core, case)
    # Bind the actual controller-written pretty JSON bytes, not fixture spelling.
    receipts['case'] = (campaign.CASE_ROOT/(args.monomer+'.json')).read_bytes()
    outcome = json.loads(receipts['outcome'])
    outcome['pilot_sha256'] = core.digest(receipts['case'])
    receipts['outcome'] = core.encode(outcome)
    job = '8001'
    report = json.loads(receipts['audit'])
    report.update(audit_commit=args.commit, pilot_sha256=core.digest(receipts['case']),
        case_outcome_sha256=core.digest(receipts['outcome']), allocation=dict(JobId=job,
            Account='gts-cs207-chemx', Partition='cpu-small', QOS='embers', NumCPUs='4',
            NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00'))
    receipts['audit'] = core.encode(report)
    io.write_once(science/'submission.json', dict(job_id=outcome['allocation']['JobId']))
    (science/'payload/pilot-outcome.json').write_bytes(receipts['outcome'])
    context_raw, checkpoint = core.initialize(pool, cases, receipts, core.digest(receipts['audit']),
                                              views['delivered'], tensor)
    audit_run = controller.phase_path(args.monomer, 'audit')
    (audit_run/'payload/audit-report.json').write_bytes(receipts['audit'])
    (audit_run/'payload/bank-context.json').write_bytes(context_raw)
    checkpoint_raw = core.encode(checkpoint)
    (audit_run/'payload/bank-checkpoint.json').write_bytes(checkpoint_raw)
    advanced = dict(context_path=str(audit_run/'payload/bank-context.json'),
        context_sha256=core.digest(context_raw), checkpoint_path=str(audit_run/'payload/bank-checkpoint.json'),
        checkpoint_sha256=core.digest(checkpoint_raw), audit_sha256=core.digest(receipts['audit']), generation=1)
    if change == 'wrong_audit_hash':
        advanced['audit_sha256'] = 'f'*64
    elif change == 'wrong_generation':
        advanced['generation'] = 2
    elif change == 'foreign_output':
        advanced['checkpoint_path'] = str(tmp_path/'bank-checkpoint.json')
    io.write_once(audit_run/'payload/bank-advance.json', advanced)
    args.phase = 'audit'
    io.write_once(audit_run/'submission.json', dict(selection={k: v for k, v in selection.items()
        if k != 'complete'}, phase='audit', command=controller.phase_command(
            args.source, args.commit, case, state, 'audit'), job_id=job))
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw:
        '\n'.join(job+suffix+'|'+('FAILED|1:0' if change == 'failed_accounting'
            else 'COMPLETED|0:0') for suffix in ('', '.batch', '.0')))
    proofs = {core.digest(value): value for value in receipts.values()}
    store = campaign.helper('bank_store')
    monkeypatch.setattr(store, 'read_proofs', lambda required: proofs)
    auditor = controller.helper('bank_audit')
    monkeypatch.setattr(auditor, 'approved_client', lambda *a: NS())
    live = copy.deepcopy(views['delivered'])
    if change == 'live_changed':
        live['metadata']['name'] = 'changed'
    elif change == 'live_signed_zero':
        geometry = live['entries'][case['entry']['name']]['molecule']['geometry']
        geometry[next(i for i, value in enumerate(geometry) if value == 0.0)] = -0.0
    monkeypatch.setattr(auditor.helper('dataset_guard'), 'snapshot_dataset', lambda *a, **kw: live)
    if change == 'none':
        result = controller.verify(args)
        assert result['generation'] == 1 and result['status'] == 'ORDINARY_AUDIT_ACCEPTED'
        # The next selection comes only from this actual ordinary proof chain.
        assert controller.context(args)[2]['monomer_id'] == 'monomer-0002'
        return controller, args, cases, pool, core
    else:
        with pytest.raises(ValueError):
            controller.verify(args)
        assert not (controller.root()/'states/001.json').exists()


@pytest.mark.parametrize('change', ['none', 'failed_accounting', 'wrong_audit_hash',
    'wrong_generation', 'foreign_output', 'live_changed', 'live_signed_zero'])
def test_first_state_requires_completed_matching_ordinary_audit_and_live_checkpoint(
        monkeypatch, tmp_path, change):
    first_state_case(monkeypatch, tmp_path, change)


@pytest.mark.parametrize('alternate_predecessor', [False, True])
def test_generation_two_requires_exact_recorded_predecessor_and_ordinary_append_proofs(
        monkeypatch, tmp_path, alternate_predecessor):
    controller, args, cases, _, core = (
        first_state_case(monkeypatch, tmp_path, 'none'))
    fixtures = load('test_audited_bank')
    campaign, io = controller.helper('bank_campaign'), controller.helper('native_controller')
    _, state, expected, bound = controller.context(args)
    args.monomer = expected['monomer_id']
    selected = controller.select(args)
    selection = {key: value for key, value in selected.items() if key != 'complete'}
    prior_raw = bound['checkpoint_raw']+(b'\n' if alternate_predecessor else b'')
    prior_sha = core.digest(prior_raw)
    case = core.bind_next_case(cases[args.monomer], prior_raw, prior_sha,
        bound['context_raw'], bound['context_sha256'], bound['proof_blobs'])
    for phase in controller.PHASES:
        path = controller.phase_path(args.monomer, phase)
        path.mkdir()
        (path/'payload').mkdir()
    worker = controller.phase_path(args.monomer, 'worker')
    io.write_once(worker/'payload/worker-preflight.json', {})
    science = controller.phase_path(args.monomer, 'science')
    receipts, views, tensor = fixtures.delivered(core, case, json.loads(prior_raw)['snapshot'])
    if not alternate_predecessor:
        receipts['case'] = (campaign.CASE_ROOT/(args.monomer+'.json')).read_bytes()
    outcome = json.loads(receipts['outcome'])
    outcome['pilot_sha256'] = core.digest(receipts['case'])
    receipts['outcome'] = core.encode(outcome)
    report = json.loads(receipts['audit'])
    report.update(audit_commit=args.commit, pilot_sha256=core.digest(receipts['case']),
        case_outcome_sha256=core.digest(receipts['outcome']), allocation=dict(JobId='8002',
            Account='gts-cs207-chemx', Partition='cpu-small', QOS='embers', NumCPUs='4',
            NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00'))
    receipts['audit'] = core.encode(report)
    io.write_once(science/'submission.json', dict(job_id=outcome['allocation']['JobId']))
    (science/'payload/pilot-outcome.json').write_bytes(receipts['outcome'])
    checkpoint, proofs = core.advance(prior_raw, prior_sha, bound['context_raw'],
        bound['context_sha256'], bound['proof_blobs'], receipts=receipts,
        expected_audit_sha256=core.digest(receipts['audit']), captured=views, tensor_raw=tensor)
    audit_run = controller.phase_path(args.monomer, 'audit')
    (audit_run/'payload/audit-report.json').write_bytes(receipts['audit'])
    (audit_run/'payload/bank-context.json').write_bytes(bound['context_raw'])
    checkpoint_raw = core.encode(checkpoint)
    (audit_run/'payload/bank-checkpoint.json').write_bytes(checkpoint_raw)
    advanced = dict(context_path=str(audit_run/'payload/bank-context.json'),
        context_sha256=bound['context_sha256'], checkpoint_path=str(audit_run/'payload/bank-checkpoint.json'),
        checkpoint_sha256=core.digest(checkpoint_raw), audit_sha256=core.digest(receipts['audit']), generation=2)
    io.write_once(audit_run/'payload/bank-advance.json', advanced)
    io.write_once(audit_run/'submission.json', dict(selection=selection, phase='audit',
        command=controller.phase_command(args.source, args.commit, expected, state, 'audit'), job_id='8002'))
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw:
        '\n'.join('8002'+suffix+'|COMPLETED|0:0' for suffix in ('', '.batch', '.0')))
    monkeypatch.setattr(campaign.helper('bank_store'), 'read_proofs',
                        lambda required: {sha: proofs[sha] for sha in required})
    auditor = controller.helper('bank_audit')
    monkeypatch.setattr(auditor.helper('dataset_guard'), 'snapshot_dataset',
                        lambda *a, **kw: views['delivered'])
    args.phase = 'audit'
    if alternate_predecessor:
        with pytest.raises(ValueError, match='unique immediate'):
            controller.verify(args)
        assert not (controller.root()/'states/002.json').exists()
    else:
        result = controller.verify(args)
        assert result['generation'] == 2
        assert result['bank_context_sha256'] == state['bank_context_sha256']
        assert controller.context(args)[2]['monomer_id'] == 'monomer-0003'
