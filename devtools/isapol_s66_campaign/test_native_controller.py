"""Orchestration tests only: no cluster, HTTP, Psi4 import or scientific call."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

path = Path(__file__).with_name('native_controller.py')
spec = importlib.util.spec_from_file_location('native_controller_unit', path)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)

CP_SHA = 'a'*64
COMMIT = 'b'*40
SOURCE = controller.ROOT/'gits/psi4.ccasp-native-s66-controller-unit'
CAMPAIGN = controller.RUNS/'ccasp-native-s66-ed561666fc-campaign001'


def checkpoint(*monomers):
    return dict(snapshot=dict(entries={
        value: dict(attributes=dict(monomer_id=value)) for value in monomers}))


def row(monomer='monomer-0001', profile='s20'):
    return dict(monomer_id=monomer, profile=profile, sha256='c'*64)


@pytest.mark.parametrize('profile,cores,memory,time', [
    ('s20', 8, 32, '02:00:00'), ('s40', 12, 64, '04:00:00'), ('s80', 24, 144, '08:00:00')])
@pytest.mark.parametrize('phase', controller.PHASES)
def test_all_explicit_finite_phase_commands(profile, cores, memory, time, phase):
    command = controller.phase_command(SOURCE, COMMIT, CAMPAIGN, row(profile=profile), phase,
                                       '/approved/checkpoint.json', CP_SHA)
    assert command[0] == '/opt/slurm/current/bin/sbatch'
    for value in ('--account=gts-cs207-chemx', '--partition=cpu-small', '--qos=embers',
                  '--no-requeue', '--nodes=1', '--ntasks=1'):
        assert value in command
    if phase == 'audit':
        cores, memory = 4, 24
    assert '--cpus-per-task='+str(cores) in command
    assert '--mem='+str(memory)+'G' in command
    assert '--time='+(time if phase == 'science' else '00:45:00') in command
    assert '--ed561666-remaining-case' in command
    assert not any('inferno' in value or 'afterok' in value or 'reset' in value for value in command)
    # Worker/bootstrap require this basename as well as the closed RUNS prefix.
    assert controller.phase_paths(CAMPAIGN, 'monomer-0001', phase).name.startswith('ccasp-native-s66-')
    if phase != 'worker':
        assert '--checkpoint-proofs' in command
        assert str(controller.PROOFS) in command and CP_SHA in command
        for role, source in controller.SEED.items():
            assert '--accepted-'+role in command and str(source) in command
    assert ('--advance-checkpoint' in command) is (phase == 'audit')


def test_audited_membership_not_server_status_selects_remaining():
    rows = [row('monomer-0006'), row('monomer-0002'), row('monomer-0001')]
    result = controller.remaining(rows, checkpoint('monomer-0047', 'monomer-0006'))
    assert [value['monomer_id'] for value in result] == ['monomer-0001', 'monomer-0002']
    with pytest.raises(ValueError, match='duplicate'):
        controller.remaining(rows+[rows[0]], checkpoint('monomer-0047'))
    with pytest.raises(ValueError, match='out-of-bank'):
        controller.remaining(rows, checkpoint('monomer-9999'))


@pytest.mark.parametrize('suffix', ['', '.batch', '.0'])
@pytest.mark.parametrize('replacement', ['missing', 'FAILED|1:0', 'PREEMPTED|0:0', 'COMPLETED|1:0'])
def test_every_accounting_row_must_positively_succeed(suffix, replacement):
    rows = {'': 'COMPLETED|0:0', '.batch': 'COMPLETED|0:0', '.0': 'COMPLETED|0:0'}
    if replacement == 'missing':
        del rows[suffix]
    else:
        rows[suffix] = replacement
    text = '\n'.join('123'+key+'|'+value+'|' for key, value in rows.items())
    with pytest.raises(ValueError, match='allocation, batch and step'):
        controller.completed_rows(text, '123')


def test_empty_accounting_is_not_success():
    with pytest.raises(ValueError):
        controller.completed_rows('', '123')
    controller.completed_rows('\n'.join('123'+s+'|COMPLETED|0:0|' for s in ('', '.batch', '.0')), '123')


@pytest.mark.parametrize('field', ['JobId', 'Account', 'Partition', 'QOS', 'NumCPUs',
                                 'NumNodes', 'MinMemoryNode', 'TimeLimit'])
def test_independent_audit_own_allocation_is_admitted(field):
    allocation = dict(JobId='123', Account='gts-cs207-chemx', Partition='cpu-small', QOS='embers',
                      NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00')
    controller.admit_audit_allocation(allocation, '123')
    allocation[field] = 'wrong'
    with pytest.raises(ValueError):
        controller.admit_audit_allocation(allocation, '123')


def test_metadata_is_exclusive_not_a_lock_repair(tmp_path):
    path = tmp_path/'intent.json'
    controller.write_once(path, dict(intent='SYNTHETIC'))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        controller.write_once(path, dict(intent='REPLACEMENT'))
    assert path.read_bytes() == original


def test_sequential_loop_waits_and_validates_before_next_phase():
    events = []
    def remote(action, state):
        events.append((action, copy.deepcopy(state)))
        if action == 'initialize':
            return dict(checkpoint='parent', checkpoint_sha256=CP_SHA, generation=129,
                        remaining=[row('monomer-0001'), row('monomer-0002')])
        if action == 'submit':
            return dict(job_id='123')
        if state['phase'] == 'audit':
            generation = 130 if state['monomer'] == 'monomer-0001' else 131
            return dict(checkpoint='checkpoint'+str(generation), checkpoint_sha256=str(generation)*21+'f',
                        generation=generation)
        return dict(status='VERIFIED_SYNTHETIC')
    def wait(job):
        events.append(('wait', job))
    result = controller.run_campaign(remote, wait, dict(checkpoint='parent', checkpoint_sha256=CP_SHA))
    assert result['generation'] == 131
    assert [event[0] for event in events] == ['initialize']+['submit', 'wait', 'verify']*6
    submits = [value for action, value in events if action == 'submit']
    assert [value['phase'] for value in submits] == list(controller.PHASES)*2
    assert all(value['checkpoint'] == 'parent' for value in submits[:3])
    assert all(value['checkpoint'] == 'checkpoint130' for value in submits[3:])


@pytest.mark.parametrize('failure', ['submit', 'wait', 'verify'])
def test_any_ambiguous_or_failed_operation_stops_without_retry(failure):
    counts = dict(submit=0, wait=0, verify=0)
    def remote(action, state):
        if action == 'initialize':
            return dict(checkpoint='parent', checkpoint_sha256=CP_SHA, generation=2,
                        remaining=[row()])
        counts[action] += 1
        if action == failure:
            raise RuntimeError('AMBIGUOUS_SYNTHETIC')
        return dict(job_id='123')
    def wait(job):
        counts['wait'] += 1
        if failure == 'wait':
            raise RuntimeError('FAILED_SYNTHETIC')
    with pytest.raises(RuntimeError):
        controller.run_campaign(remote, wait, dict(checkpoint='parent', checkpoint_sha256=CP_SHA))
    assert counts['submit'] == 1 and all(count <= 1 for count in counts.values())


def test_phase_reservation_refuses_second_writer_before_sbatch(monkeypatch, tmp_path):
    campaign = tmp_path/'campaign'
    campaign.mkdir()
    args = NS(campaign=campaign, monomer='monomer-0001', phase='worker',
              source=SOURCE, commit=COMMIT, checkpoint=tmp_path/'checkpoint.json',
              checkpoint_sha256=CP_SHA)
    run = controller.phase_paths(campaign, args.monomer, 'worker')
    run.mkdir()
    monkeypatch.setattr(controller, 'context', lambda args: (row(), {}, {}, {}))
    monkeypatch.setattr(controller, 'confirm_live', lambda cp: None)
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw: '')
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **kw: pytest.fail('must not submit'))
    with pytest.raises(FileExistsError):
        controller.submit(args)


@pytest.mark.parametrize('response', [
    NS(returncode=1, stdout='', stderr='lost response'),
    NS(returncode=0, stdout='', stderr=''),
    NS(returncode=0, stdout='123\n124\n', stderr='')])
def test_lost_submission_response_keeps_permanent_intent(monkeypatch, tmp_path, response):
    campaign = tmp_path/'campaign'
    campaign.mkdir()
    args = NS(campaign=campaign, monomer='monomer-0001', phase='worker',
              source=SOURCE, commit=COMMIT, checkpoint=tmp_path/'checkpoint.json',
              checkpoint_sha256=CP_SHA)
    monkeypatch.setattr(controller, 'context', lambda args: (row(), {}, {}, {}))
    monkeypatch.setattr(controller, 'confirm_live', lambda cp: None)
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw: '')
    calls = []
    def execute(command, **kw):
        calls.append(command)
        if '--test-only' in command:
            return NS(returncode=0, stdout='test only', stderr='')
        return response
    monkeypatch.setattr(controller.subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='ambiguous/failed'):
        controller.submit(args)
    run = controller.phase_paths(campaign, args.monomer, 'worker')
    assert read_json(run/'intent.json')['checkpoint_sha256'] == CP_SHA
    assert (run/'submission-response.json').exists() and not (run/'submission.json').exists()
    with pytest.raises(FileExistsError):
        controller.submit(args)
    assert len(calls) == 2  # one test-only, one actual send, no replay


def read_json(path):
    return json.loads(path.read_bytes())


def test_help_has_no_quantum_or_server_side_effects():
    import subprocess
    result = subprocess.run([sys_executable(), '-I', '-S', '-B', str(path), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0 and '--checkpoint-sha256' in result.stdout


@pytest.mark.parametrize('phase', controller.PHASES)
def test_complete_wrapper_positional_order(phase):
    command = controller.phase_command(SOURCE, COMMIT, CAMPAIGN, row(), phase, '/parent', CP_SHA)
    position = next(i for i, value in enumerate(command) if value.endswith('.sbatch'))
    case = str(controller.PREPARATION/'cases/monomer-0001.json')
    worker = str(controller.phase_paths(CAMPAIGN, 'monomer-0001', 'worker')/'payload/worker-preflight.json')
    run = str(controller.phase_paths(CAMPAIGN, 'monomer-0001', phase))
    if phase == 'worker':
        expected = [str(SOURCE), COMMIT, case, run, '--ed561666-remaining-case']
    elif phase == 'science':
        expected = [str(SOURCE), COMMIT, case, worker, run]+controller.append_flags('/parent', CP_SHA)
    else:
        expected = [str(SOURCE), COMMIT, case, worker,
                    str(controller.phase_paths(CAMPAIGN, 'monomer-0001', 'science')), run]
        expected += controller.append_flags('/parent', CP_SHA)+['--advance-checkpoint']
    assert command[position+1:] == expected


@pytest.mark.parametrize('phase', ['science', 'audit'])
@pytest.mark.parametrize('problem', ['missing', 'malformed', 'checkpoint', 'phase'])
def test_predecessor_guard_refuses_before_scheduler(monkeypatch, tmp_path, phase, problem):
    args = NS(campaign=tmp_path, monomer='monomer-0001', phase=phase, source=SOURCE,
              commit=COMMIT, checkpoint=tmp_path/'parent', checkpoint_sha256=CP_SHA)
    monkeypatch.setattr(controller, 'context', lambda args: (row(), {}, {}, {}))
    previous = controller.PHASES[controller.PHASES.index(phase)-1]
    directory = controller.phase_paths(tmp_path, args.monomer, previous)
    directory.mkdir()
    if problem != 'missing':
        value = dict(phase=previous, checkpoint_sha256=CP_SHA)
        if problem in value:
            value[problem] = 'wrong'
        if problem == 'checkpoint':
            value['checkpoint_sha256'] = 'wrong'
        (directory/'verified.json').write_text('!' if problem == 'malformed' else json.dumps(value))
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **k: pytest.fail('no scheduler'))
    with pytest.raises((ValueError, FileNotFoundError)):
        controller.submit(args)


def initial_example():
    cp = checkpoint('monomer-0047', 'monomer-0006')
    cp.update(generation=2, audited_records=[5234318, 5234319],
        operational_commit='6ddbbb8ef8f5be0b24657dc4656c933bfa9d4263',
        ledger=[dict(monomer_id='monomer-0006', record_id=5234319,
                     receipts_sha256=dict(audit='a', outcome='o'))])
    audit = dict(allocation=dict(JobId='14999621'), record_id=5234319, case_job='14999135',
                 audit_commit='7438d9b04ea87423080f5ce75e7a81862bccd8eb')
    outcome = dict(allocation=dict(JobId='14999135'), operational_commit=audit['audit_commit'])
    inputs = dict(checkpoint_proof_blobs={key: json.dumps(value).encode()
                                        for key, value in dict(a=audit, o=outcome).items()})
    return cp, inputs


@pytest.mark.parametrize('problem', ['generation', 'monomer', 'records', 'advance_revision',
                                   'science_job', 'audit_job', 'old_ops', 'missing_proof'])
def test_actual_initial_identity_and_proofs_required_before_reservation(monkeypatch, tmp_path, problem):
    cp, inputs = initial_example()
    if problem == 'generation':
        cp['generation'] = 3
    elif problem == 'monomer':
        cp['snapshot']['entries'].pop('monomer-0006')
    elif problem == 'records':
        cp['audited_records'] = [5234318, 99999]
    elif problem == 'advance_revision':
        cp['operational_commit'] = 'wrong'
    elif problem == 'missing_proof':
        inputs['checkpoint_proof_blobs'].pop('a')
    else:
        audit = json.loads(inputs['checkpoint_proof_blobs']['a'])
        if problem == 'science_job':
            audit['case_job'] = '14999136'
        elif problem == 'audit_job':
            audit['allocation']['JobId'] = '14999622'
        else:
            audit['audit_commit'] = 'wrong'
        inputs['checkpoint_proof_blobs']['a'] = json.dumps(audit).encode()
    monkeypatch.setattr(controller, 'checkpoint_inputs', lambda *a: (cp, inputs))
    monkeypatch.setattr(controller, 'input_rows', lambda: pytest.fail('stop before input/live reads'))
    args = NS(checkpoint=tmp_path/'cp', checkpoint_sha256=CP_SHA, campaign=tmp_path/'campaign',
              source=SOURCE, commit=COMMIT, adapter_seal='d'*64)
    with pytest.raises((ValueError, KeyError)):
        controller.initialize(args)
    assert not args.campaign.exists()


def test_new_directory_entry_is_synced_before_submission_metadata(monkeypatch, tmp_path):
    events = []
    original = controller.sync_directory
    def sync(path):
        events.append(Path(path))
        original(path)
    monkeypatch.setattr(controller, 'sync_directory', sync)
    campaign = tmp_path/'campaign'
    controller.mkdir_once(campaign)
    controller.mkdir_once(campaign/'states')
    controller.write_once(campaign/'states/002.json', dict(synthetic=True))
    assert events == [tmp_path, campaign, campaign/'states']


@pytest.mark.parametrize('file', ['owned_dataset_cb63305.py', 'runtime_audit_cb63305.py',
                                 'phoenix_pilot_cb63305.sbatch', 'validate_numeric.py'])
def test_hidden_modified_operational_sources_are_not_trusted(monkeypatch, tmp_path, file):
    root = tmp_path/'root'
    source = root/'gits/psi4.ccasp-native-s66-controller-unit'
    relative = 'devtools/isapol_s66_campaign/'
    names = {relative+name+'.py' for name in controller.HELPERS}
    names |= {relative+name for name in ('native_controller.py', 'phoenix_worker_cb63305.sbatch',
        'phoenix_pilot_cb63305.sbatch', 'runtime_audit_cb63305.sbatch', 'validate_numeric.py')}
    names.add('devtools/isapol_native_acceptance/trace_gate.py')
    inventory = {name: ('PINNED '+name).encode() for name in names}
    for name, raw in inventory.items():
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(controller, 'ROOT', root)
    monkeypatch.setattr(controller, '__file__', str(source/relative/'native_controller.py'))
    monkeypatch.setattr(controller, 'committed_code', lambda *a: inventory)
    def git(command, **kw):
        return COMMIT+'\n' if 'rev-parse' in command else ''  # even hidden changes must fail
    monkeypatch.setattr(controller.subprocess, 'check_output', git)
    controller.validate_source(source, COMMIT)
    (source/relative/file).write_bytes(b'HIDDEN_MODIFICATION')
    with pytest.raises(ValueError, match='policy bytes changed'):
        controller.validate_source(source, COMMIT)


def audit_verification_example(monkeypatch, tmp_path):
    args = NS(campaign=tmp_path, monomer='monomer-0001', phase='audit', source=SOURCE,
              commit=COMMIT, checkpoint=tmp_path/'parent', checkpoint_sha256=CP_SHA)
    previous = dict(generation=2, ledger=[dict(seed='SYNTHETIC')], audited_records=[1, 2])
    report = dict(status='PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE',
        allocation=dict(JobId='124', Account='gts-cs207-chemx', Partition='cpu-small', QOS='embers',
                        NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00'),
        audit_commit=COMMIT, case_operational_commit=COMMIT, monomer_id=args.monomer,
        dataset_id=355, record_id=3, case_job='123', roles=dict(molecular_qcschema=1),
        traces={'synthetic': dict(status='passed')})
    run = controller.phase_paths(tmp_path, args.monomer, 'audit')
    (run/'payload').mkdir(parents=True)
    report_raw = json.dumps(report).encode()
    (run/'payload/audit-report.json').write_bytes(report_raw)
    successor = dict(generation=3, previous_checkpoint_sha256=CP_SHA,
        audited_records=[1, 2, 3], ledger=previous['ledger']+[dict(
            monomer_id=args.monomer, receipts_sha256=dict(audit=controller.sha(report_raw)))])
    (run/'payload/next-checkpoint.json').write_text('SYNTHETIC')
    for phase, name, data in (
        ('worker', 'payload/worker-preflight.json', {}),
        ('science', 'submission.json', dict(job_id='123'))):
        path = controller.phase_paths(tmp_path, args.monomer, phase)/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    command = controller.phase_command(SOURCE, COMMIT, tmp_path, row(), 'audit', args.checkpoint, CP_SHA)
    (run/'submission.json').write_text(json.dumps(dict(
        job_id='124', checkpoint_sha256=CP_SHA, case_sha256=row()['sha256'], command=command)))
    monkeypatch.setattr(controller, 'context', lambda args: (row(), {}, previous, {}))
    monkeypatch.setattr(controller, 'helper', lambda name: NS())
    monkeypatch.setattr(controller, 'checkpoint_inputs', lambda *a: (successor, {}))
    monkeypatch.setattr(controller, 'confirm_live', lambda cp: None)
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **k:
                        '\n'.join('124'+s+'|COMPLETED|0:0|' for s in ('', '.batch', '.0')))
    (tmp_path/'states').mkdir()
    return args, previous, successor, run


@pytest.mark.parametrize('problem', ['generation', 'parent', 'prefix', 'monomer', 'audit_hash', 'records', 'live'])
def test_bad_successor_never_publishes_campaign_state(monkeypatch, tmp_path, problem):
    args, before, successor, run = audit_verification_example(monkeypatch, tmp_path)
    if problem == 'generation':
        successor['generation'] = 4
    elif problem == 'parent':
        successor['previous_checkpoint_sha256'] = 'foreign'
    elif problem == 'prefix':
        successor['ledger'][0] = dict(changed=True)
    elif problem == 'monomer':
        successor['ledger'][-1]['monomer_id'] = 'monomer-0002'
    elif problem == 'audit_hash':
        successor['ledger'][-1]['receipts_sha256']['audit'] = 'wrong'
    elif problem == 'records':
        successor['audited_records'] = [1, 2, 999]
    else:
        def changed(cp):
            raise ValueError('live changed')
        monkeypatch.setattr(controller, 'confirm_live', changed)
    with pytest.raises(ValueError):
        controller.verify(args)
    assert not (run/'verified.json').exists()
    assert not list((tmp_path/'states').iterdir())


def test_crash_after_audit_verification_is_fail_closed_not_repaired(monkeypatch, tmp_path):
    args, before, successor, run = audit_verification_example(monkeypatch, tmp_path)
    original = controller.write_once
    def fail_state(path, value):
        if Path(path).parent == tmp_path/'states':
            raise OSError('simulated storage failure')
        original(path, value)
    monkeypatch.setattr(controller, 'write_once', fail_state)
    with pytest.raises(OSError):
        controller.verify(args)
    assert (run/'verified.json').exists() and not list((tmp_path/'states').iterdir())
    with pytest.raises(FileExistsError):
        controller.verify(args)


@pytest.mark.parametrize('failure', ['returncode', 'missing_prefix', 'duplicate_prefix', 'json', 'timeout'])
def test_remote_adapter_never_replays_ambiguous_response(monkeypatch, failure):
    import subprocess
    args = NS(source=SOURCE, commit=COMMIT, campaign=CAMPAIGN)
    monkeypatch.setattr(controller.Remote, 'seal', lambda self: 'd'*64)
    remote = controller.Remote(args)
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 150)
        text = controller.PREFIX+'{}\n'
        if failure == 'missing_prefix':
            text = 'noise'
        elif failure == 'duplicate_prefix':
            text *= 2
        elif failure == 'json':
            text = controller.PREFIX+'!'
        return NS(returncode=1 if failure == 'returncode' else 0, stdout=text, stderr='synthetic')
    monkeypatch.setattr(controller.subprocess, 'run', execute)
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        remote.call('submit', dict(checkpoint='/parent', checkpoint_sha256=CP_SHA,
                                   monomer='monomer-0001', phase='science'))
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert kwargs['env']['PHX_ALIAS'] == 'phoenix2'
    import shlex
    parsed = shlex.split(command[1])
    assert parsed[-4:] == ['--monomer', 'monomer-0001', '--phase', 'science']


def test_watcher_adapter_uses_installed_waiter_once(monkeypatch):
    monkeypatch.setattr(controller.Remote, 'seal', lambda self: 'd'*64)
    remote = controller.Remote(NS())
    calls = []
    monkeypatch.setattr(controller.subprocess, 'run', lambda cmd, **kw: calls.append((cmd, kw)))
    remote.wait('123')
    assert calls == [([str(remote.watcher), '-i', '300', 'phoenix2', '123'], dict(check=True))]


def test_remote_environment_launcher_never_resolves_user_path_env(monkeypatch):
    import shlex
    monkeypatch.setattr(controller.Remote, 'seal', lambda self: 'd'*64)
    remote = controller.Remote(NS(source=SOURCE, commit=COMMIT, campaign=CAMPAIGN))
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return NS(returncode=0, stdout=controller.PREFIX+'{}\n', stderr='')
    monkeypatch.setattr(controller.subprocess, 'run', execute)
    remote.call('initialize', dict(checkpoint='/parent', checkpoint_sha256=CP_SHA))
    assert shlex.split(calls[0][1])[0] == '/usr/bin/env'


@pytest.mark.parametrize('problem', ['dirty', 'wrong_commit'])
def test_wrong_or_dirty_source_stops_before_code_inventory(monkeypatch, problem):
    def git(command, **kw):
        if 'rev-parse' in command:
            return ('f'*40 if problem == 'wrong_commit' else COMMIT)+'\n'
        return ' M hidden-helper.py\n' if problem == 'dirty' else ''
    monkeypatch.setattr(controller.subprocess, 'check_output', git)
    monkeypatch.setattr(controller, 'committed_code', lambda *a: pytest.fail('must stop before helpers'))
    with pytest.raises(ValueError, match='exact clean'):
        controller.validate_source(SOURCE, COMMIT)


def test_stale_checkpoint_cannot_authorize_another_phase(monkeypatch, tmp_path):
    args = NS(campaign=tmp_path, source=SOURCE, commit=COMMIT, adapter_seal='d'*64,
              checkpoint=Path('/old'), checkpoint_sha256=CP_SHA, monomer='monomer-0001')
    plan = dict(source=str(SOURCE), commit=COMMIT,
        controller_sha256=controller.sha(Path(controller.__file__).read_bytes()),
        local_adapter_sha256=args.adapter_seal, preparation_sha256=controller.INDEX_SHA, rows=[row()])
    monkeypatch.setattr(controller, 'read', lambda path: plan)
    monkeypatch.setattr(controller, 'input_rows', lambda: [row()])
    monkeypatch.setattr(controller, 'current_state',
                        lambda path: dict(checkpoint='/new', checkpoint_sha256='e'*64))
    monkeypatch.setattr(controller, 'checkpoint_inputs', lambda *a: pytest.fail('must stop before proof reads'))
    with pytest.raises(ValueError, match='stale'):
        controller.context(args)


def test_response_persistence_failure_after_submit_never_replays(monkeypatch, tmp_path):
    campaign = tmp_path/'campaign'
    campaign.mkdir()
    args = NS(campaign=campaign, monomer='monomer-0001', phase='worker',
              source=SOURCE, commit=COMMIT, checkpoint=tmp_path/'checkpoint.json',
              checkpoint_sha256=CP_SHA)
    monkeypatch.setattr(controller, 'context', lambda args: (row(), {}, {}, {}))
    monkeypatch.setattr(controller, 'confirm_live', lambda cp: None)
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw: '')
    actual_sends = []
    def execute(command, **kw):
        if '--test-only' not in command:
            actual_sends.append(command)
        return NS(returncode=0, stdout='123', stderr='')
    monkeypatch.setattr(controller.subprocess, 'run', execute)
    original = controller.write_once
    def persist(path, value):
        if Path(path).name == 'submission-response.json':
            raise OSError('simulated post-submit storage failure')
        original(path, value)
    monkeypatch.setattr(controller, 'write_once', persist)
    with pytest.raises(OSError):
        controller.submit(args)
    run = controller.phase_paths(campaign, args.monomer, 'worker')
    assert (run/'intent.json').exists() and not (run/'submission.json').exists()
    with pytest.raises(FileExistsError):
        controller.submit(args)
    assert len(actual_sends) == 1


def sys_executable():
    import sys
    return sys.executable
