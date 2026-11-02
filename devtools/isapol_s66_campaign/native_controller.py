"""Sequential owned355 orchestration, not a scientific implementation.

One immutable checkout; worker -> fresh QCArchive singleton -> independent
audit/advance. Any failed or ambiguous operation stops. There is deliberately
no resume, retry, reset, retag, cancellation or lock-clearing command.
Receipts assume ordinary OS trust, not execution attestation.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path('/storage/project/r-cs207-0/awallace43')
RUNS = ROOT/'runs'
MANAGER = ROOT/'miniconda/envs/p4qca'
PREPARATION = RUNS/'ccasp-native-s66-ed561666fc-remaining-preparation001'
PROOFS = RUNS/'ccasp-native-s66-ed561666fc-checkpoint-proofs'
INDEX_SHA = '0affce4cd1e450f4e7d0e97f72ffab4b1f09447798f00ddacf40a9c3387df7e3'
PREFIX = 'CCASP_CONTROLLER_RESULT='
PHASES = ('worker', 'science', 'audit')
SEED = dict(
    audit=RUNS/'ccasp-native-s66-ed561666fc-benzene-audit002/payload/audit-report.json',
    outcome=RUNS/'ccasp-native-s66-ed561666fc-benzene-pilot001/payload/pilot-outcome.json',
    pilot=RUNS/'ccasp-native-s66-ed561666fc-benzene-preparation001/pilot.json',
    envelope=RUNS/'ccasp-native-s66-ed561666fc-benzene-pilot001/payload/server-native-envelope.json')
PROFILES = dict(s20=(8, 32, '02:00:00'), s40=(12, 64, '04:00:00'),
                s80=(24, 144, '08:00:00'))
HELPERS = ('owned_dataset_cb63305', 'dataset_owned354_cb63305',
           'prepare_owned354_baseline_cb63305', 'phoenix_pilot',
           'phoenix_worker_cb63305', 'phoenix_pilot_cb63305', 'runtime_audit_cb63305')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def helper(name):
    if name not in HELPERS:
        raise ValueError('closed orchestration helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_native_controller_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir_once(path):
    Path(path).mkdir(mode=0o700, exist_ok=False)
    sync_directory(Path(path).parent)


def write_once(path, value):
    """Exclusive durable metadata; an existing/partial file is never repaired."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(Path(path).parent)


def require_hash(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('exact SHA256 required')
    return value


def require_monomer(value):
    if type(value) is not str or re.fullmatch(r'monomer-[0-9]{4}', value) is None:
        raise ValueError('closed bank monomer required')
    if not 1 <= int(value[-4:]) <= 131:
        raise ValueError('closed bank monomer required')
    return value


def remaining(rows, checkpoint):
    audited = {entry['attributes']['monomer_id']
               for entry in checkpoint['snapshot']['entries'].values()}
    identifiers = [row['monomer_id'] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('duplicate preparation representatives')
    if not audited <= set(identifiers) | {'monomer-0047'}:
        raise ValueError('checkpoint contains an out-of-bank representative')
    return sorted((row for row in rows if row['monomer_id'] not in audited),
                  key=lambda row: row['monomer_id'])


def phase_paths(campaign, monomer, phase):
    require_monomer(monomer)
    if phase not in PHASES:
        raise ValueError('closed phase required')
    return Path(campaign)/('ccasp-native-s66-ed561666fc-'+monomer+'-'+phase)


def append_flags(checkpoint, checkpoint_sha):
    result = ['--ed561666-remaining-case', '--owned-checkpoint', str(checkpoint),
              '--checkpoint-sha256', require_hash(checkpoint_sha), '--checkpoint-proofs', str(PROOFS)]
    for role, path in SEED.items():
        result += ['--accepted-'+role, str(path)]
    return result


def phase_command(source, commit, campaign, row, phase, checkpoint, checkpoint_sha):
    """One resource/argument policy, exercised without subprocesses by tests."""
    monomer = require_monomer(row['monomer_id'])
    if phase not in PHASES or row['profile'] not in PROFILES:
        raise ValueError('finite orchestration phase/profile required')
    cores, memory, science_time = PROFILES[row['profile']]
    if phase == 'audit':
        cores, memory = 4, 24
    time = science_time if phase == 'science' else '00:45:00'
    run = phase_paths(campaign, monomer, phase)
    case = PREPARATION/'cases'/(monomer+'.json')
    worker = phase_paths(campaign, monomer, 'worker')/'payload/worker-preflight.json'
    scripts = dict(worker='phoenix_worker_cb63305.sbatch',
                   science='phoenix_pilot_cb63305.sbatch', audit='runtime_audit_cb63305.sbatch')
    command = ['/opt/slurm/current/bin/sbatch', '--parsable', '--account=gts-cs207-chemx',
        '--partition=cpu-small', '--qos=embers', '--no-requeue', '--nodes=1', '--ntasks=1',
        '--cpus-per-task='+str(cores), '--mem='+str(memory)+'G', '--time='+time,
        '--job-name=ccasp-ed-'+monomer[-4:]+'-'+phase,
        '--output='+str(run/'logs/job-%j.out'), '--error='+str(run/'logs/job-%j.err'),
        str(Path(source)/'devtools/isapol_s66_campaign'/scripts[phase]),
        str(source), commit, str(case)]
    if phase == 'worker':
        command += [str(run), '--ed561666-remaining-case']
    elif phase == 'science':
        command += [str(worker), str(run)] + append_flags(checkpoint, checkpoint_sha)
    else:
        command += [str(worker), str(phase_paths(campaign, monomer, 'science')), str(run)]
        command += append_flags(checkpoint, checkpoint_sha) + ['--advance-checkpoint']
    return command


def completed_rows(text, job):
    rows = [line.strip().split('|')[:3] for line in text.splitlines()]
    if any([job+suffix, 'COMPLETED', '0:0'] not in rows for suffix in ('', '.batch', '.0')):
        raise ValueError('allocation, batch and step must all complete0:0')


def admit_audit_allocation(allocation, job):
    expected = dict(JobId=job, Account='gts-cs207-chemx', Partition='cpu-small', QOS='embers',
                    NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00')
    if any(allocation.get(key) != value for key, value in expected.items()):
        raise ValueError('exact independent audit allocation required')


def committed_code(source, commit):
    """Closed Git-tree inventory, including every transitive operational helper.

    Inspect archive bytes in memory, never extract or copy source files. This
    also catches assume-unchanged/skip-worktree changes hidden from git status.
    """
    scopes = ('devtools/isapol_s66_campaign', 'devtools/isapol_native_acceptance/trace_gate.py')
    raw = subprocess.check_output(['git', '-C', str(source), 'archive', commit, *scopes])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers()
                if member.isfile() and Path(member.name).suffix in ('.py', '.sh', '.sbatch')}


def validate_source(source, commit):
    source = Path(source)
    if (not re.fullmatch(r'[0-9a-f]{40}', commit)
            or source.parent != ROOT/'gits' or source.is_symlink()
            or not source.name.startswith('psi4.ccasp-native-s66-')):
        raise ValueError('private immutable campaign checkout required')
    git = lambda *args: subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
        raise ValueError('exact clean campaign checkout required')
    here = Path(__file__).resolve()
    if here != source/'devtools/isapol_s66_campaign/native_controller.py':
        raise ValueError('remote controller must execute from its pinned checkout')
    inventory = committed_code(source, commit)
    required = {'devtools/isapol_s66_campaign/'+name+'.py' for name in HELPERS}
    required |= {'devtools/isapol_s66_campaign/'+name for name in (
        'native_controller.py', 'phoenix_worker_cb63305.sbatch', 'phoenix_pilot_cb63305.sbatch',
        'runtime_audit_cb63305.sbatch')}
    required.add('devtools/isapol_native_acceptance/trace_gate.py')
    if not required <= set(inventory):
        raise ValueError('complete committed orchestration/helper inventory required')
    for relative, raw in inventory.items():
        if (source/relative).read_bytes() != raw:
            raise ValueError('tracked campaign policy bytes changed')


def checkpoint_inputs(path, expected_sha):
    raw = Path(path).read_bytes()
    if sha(raw) != require_hash(expected_sha):
        raise ValueError('exact checkpoint bytes required')
    append = helper('dataset_owned354_cb63305')
    proofs = append.read_checkpoint_proofs(raw, expected_sha, PROOFS)
    guard = helper('owned_dataset_cb63305')
    blobs = {role: value.read_bytes() for role, value in SEED.items()}
    accepted = guard.verify_first_acceptance(blobs, producer='ed561666')
    checkpoint = guard.validate_checkpoint_contents(json.loads(raw), accepted, proofs)
    return checkpoint, dict(first_receipt_blobs=blobs, checkpoint_raw=raw,
        checkpoint_sha256=expected_sha, checkpoint_proof_blobs=proofs)


def confirm_live(checkpoint):
    baseline, guard = helper('prepare_owned354_baseline_cb63305'), helper('owned_dataset_cb63305')
    client, _ = baseline.approved_client(producer='ed561666')
    first = guard.snapshot_dataset(client, producer='ed561666')
    if first != checkpoint['snapshot'] or guard.snapshot_dataset(client, producer='ed561666') != first:
        raise ValueError('live owned355 state differs from the audited checkpoint')


def input_rows():
    raw = (PREPARATION/'preparation.json').read_bytes()
    if sha(raw) != INDEX_SHA:
        raise ValueError('closed all130 raw-case preparation index changed')
    rows = json.loads(raw)['cases']
    if ({row['monomer_id'] for row in rows} !=
            {'monomer-'+str(i).zfill(4) for i in range(1, 132) if i != 47}
            or len(rows) != 130):
        raise ValueError('all130 raw representatives required')
    for row in rows:
        case_raw = (PREPARATION/'cases'/(row['monomer_id']+'.json')).read_bytes()
        case = json.loads(case_raw)
        if (sha(case_raw) != row['sha256'] or case['case_id'] != row['case_id']
                or case['monomer_id'] != row['monomer_id']
                or case['execution_profile']['name'] != row['profile']
                or case['runtime_input_sha256'] != row['runtime_input_sha256']):
            raise ValueError('staged raw case differs from the closed preparation')
    return rows


def current_state(campaign):
    files = sorted((Path(campaign)/'states').glob('[0-9][0-9][0-9].json'))
    if not files:
        raise ValueError('campaign has no positively recorded checkpoint')
    return read(files[-1])


def initialize(args):
    checkpoint, inputs = checkpoint_inputs(args.checkpoint, args.checkpoint_sha256)
    if (checkpoint['generation'] != 2 or
            {v['attributes']['monomer_id'] for v in checkpoint['snapshot']['entries'].values()}
            != {'monomer-0047', 'monomer-0006'}
            or checkpoint['audited_records'] != [5234318, 5234319]
            or checkpoint.get('operational_commit') != '6ddbbb8ef8f5be0b24657dc4656c933bfa9d4263'):
        raise ValueError('automation starts only from audited benzene plus fresh methylamine')
    link = checkpoint['ledger'][0]
    proofs = inputs['checkpoint_proof_blobs']
    audit, outcome = (json.loads(proofs[link['receipts_sha256'][role]]) for role in ('audit', 'outcome'))
    if (link['monomer_id'] != 'monomer-0006' or link['record_id'] != 5234319
            or audit['allocation']['JobId'] != '14999621' or audit['record_id'] != 5234319
            or audit['case_job'] != outcome['allocation']['JobId'] or audit['case_job'] != '14999135'
            or audit['audit_commit'] != '7438d9b04ea87423080f5ce75e7a81862bccd8eb'
            or outcome['operational_commit'] != audit['audit_commit']):
        raise ValueError('actual frozen methylamine science/audit history required')
    require_hash(args.adapter_seal)
    rows = input_rows()
    confirm_live(checkpoint)
    campaign = args.campaign
    mkdir_once(campaign)
    mkdir_once(campaign/'states')
    plan = dict(schema='native-sequential-campaign/1', source=str(args.source), commit=args.commit,
        controller_sha256=sha(Path(__file__).read_bytes()), preparation_sha256=INDEX_SHA,
        local_adapter_sha256=args.adapter_seal,
        starting_checkpoint=str(args.checkpoint), starting_sha256=args.checkpoint_sha256, rows=rows,
        qualification='Ordinary OS/approved receipt trust; no automatic retry or scientific accuracy claim.')
    write_once(campaign/'plan.json', plan)
    state = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256, generation=2)
    write_once(campaign/'states/002.json', state)
    return dict(state, remaining=remaining(rows, checkpoint))


def context(args):
    plan = read(args.campaign/'plan.json')
    if (plan['source'] != str(args.source) or plan['commit'] != args.commit
            or plan['controller_sha256'] != sha(Path(__file__).read_bytes())
            or plan['local_adapter_sha256'] != args.adapter_seal
            or plan['preparation_sha256'] != INDEX_SHA
            or plan['rows'] != input_rows()):
        raise ValueError('campaign source/plan changed')
    state = current_state(args.campaign)
    if state['checkpoint'] != str(args.checkpoint) or state['checkpoint_sha256'] != args.checkpoint_sha256:
        raise ValueError('stale campaign checkpoint; never resubmit an earlier transition')
    checkpoint, inputs = checkpoint_inputs(args.checkpoint, args.checkpoint_sha256)
    todo = remaining(plan['rows'], checkpoint)
    if not todo or todo[0]['monomer_id'] != args.monomer:
        raise ValueError('only the next unaudited representative may run')
    row = todo[0]
    raw = (PREPARATION/'cases'/(args.monomer+'.json')).read_bytes()
    if sha(raw) != row['sha256']:
        raise ValueError('selected raw case changed')
    return row, json.loads(raw), checkpoint, inputs


def submit(args):
    row, _, checkpoint, _ = context(args)
    phase = args.phase
    if phase == 'worker':
        confirm_live(checkpoint)
    else:
        previous = PHASES[PHASES.index(phase)-1]
        proof = read(phase_paths(args.campaign, args.monomer, previous)/'verified.json')
        if proof['checkpoint_sha256'] != args.checkpoint_sha256 or proof['phase'] != previous:
            raise ValueError('previous phase lacks matching positive verification')
    # A failed status query must never be interpreted as spare capacity.
    queue = subprocess.check_output(['/opt/slurm/current/bin/squeue', '-u', 'awallace43',
                                     '-r', '-h', '-o', '%q'], text=True)
    if sum(value.strip() == 'embers' for value in queue.splitlines()) >= 49:
        raise ValueError('insufficient embers submit headroom; stop without queuing')
    run = phase_paths(args.campaign, args.monomer, phase)
    mkdir_once(run)
    mkdir_once(run/'logs')
    command = phase_command(args.source, args.commit, args.campaign, row, phase,
                            args.checkpoint, args.checkpoint_sha256)
    # Preflight has no SCI or queue mutation, but still leaves the directory
    # permanently reserved if it fails.
    subprocess.run(command[:1]+['--test-only']+command[1:], check=True, capture_output=True, text=True)
    intent = dict(phase=phase, monomer=args.monomer, checkpoint_sha256=args.checkpoint_sha256,
                  case_sha256=row['sha256'], command=command)
    write_once(run/'intent.json', intent)
    response = subprocess.run(command, check=False, capture_output=True, text=True)
    write_once(run/'submission-response.json',
               dict(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
    job = response.stdout.strip()
    if response.returncode != 0 or re.fullmatch(r'[1-9][0-9]*', job) is None:
        raise RuntimeError('ambiguous/failed submission: stop; never automatically retry or clear intent')
    write_once(run/'submission.json', dict(intent, job_id=job))
    return dict(phase=phase, monomer=args.monomer, job_id=job)


def verify(args):
    row, case, checkpoint, inputs = context(args)
    run = phase_paths(args.campaign, args.monomer, args.phase)
    submitted = read(run/'submission.json')
    job = submitted['job_id']
    if (submitted['checkpoint_sha256'] != args.checkpoint_sha256
            or submitted['case_sha256'] != row['sha256']
            or submitted['command'] != phase_command(args.source, args.commit, args.campaign,
                row, args.phase, args.checkpoint, args.checkpoint_sha256)):
        raise ValueError('submission intent differs')
    completed_rows(subprocess.check_output(['/opt/slurm/current/bin/sacct', '-n', '-P',
        '-j', job, '-o', 'JobIDRaw,State,ExitCode'], text=True), job)
    worker = helper('phoenix_worker_cb63305')
    bootstrap = helper('phoenix_pilot_cb63305')
    worker_path = phase_paths(args.campaign, args.monomer, 'worker')/'payload/worker-preflight.json'
    preflight = read(worker_path)
    if args.phase == 'worker':
        worker.admit_allocation(preflight['allocation'], execution_profile=case['execution_profile'])
        if preflight['allocation']['JobId'] != job or preflight['operational_commit'] != args.commit:
            raise ValueError('worker job/source differs')
        bootstrap.validate_preflight(preflight, case,
            manifest_sha=sha((RUNS/'ccasp-native-s66-ed561666fc-build001/build-identity.json').read_bytes()),
            qualification_sha=sha((RUNS/'ccasp-native-s66-ed561666fc-checks001/payload/installed-validation.json').read_bytes()),
            pilot_sha=row['sha256'], worker_sources=worker.source_hashes('ed561666', remaining_case=True),
            transport=worker.helper('dataset_fresh_cb63305').transport_binding(producer='ed561666'),
            producer='ed561666', remaining_case=True)
        confirm_live(checkpoint)
    elif args.phase == 'science':
        outcome = read(run/'payload/pilot-outcome.json')
        bootstrap.admit_allocation(outcome['allocation'], execution_profile=case['execution_profile'])
        if (outcome['allocation']['JobId'] != job or outcome['operational_commit'] != args.commit
                or outcome['dataset_id'] != 355 or outcome['monomer_id'] != args.monomer
                or outcome['pilot_sha256'] != row['sha256']
                or outcome['worker_preflight_sha256'] != sha(worker_path.read_bytes())
                or outcome['status'] != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
                or outcome['find_existing'] is not False or outcome['qcengine_internal_retries'] != 0):
            raise ValueError('fresh scientific delivery differs')
        from qcportal.singlepoint import QCSpecification
        audit = helper('runtime_audit_cb63305')
        audit.validate_preservation(run/'payload', outcome, case, inputs,
            QCSpecification(**helper('phoenix_pilot').execution_specification(case['specification'])),
            producer='ed561666')
    else:
        report_path = run/'payload/audit-report.json'
        report = read(report_path)
        admit_audit_allocation(report['allocation'], job)
        if (report['status'] != 'PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE'
                or report['audit_commit'] != args.commit or report['case_operational_commit'] != args.commit
                or report['monomer_id'] != args.monomer or report['dataset_id'] != 355
                or report['case_job'] != read(phase_paths(
                    args.campaign, args.monomer, 'science')/'submission.json')['job_id']
                or report['roles']['molecular_qcschema'] != 1 or not report['traces']
                or any(value['status'] != 'passed' for value in report['traces'].values())):
            raise ValueError('independent audit did not pass')
        path = run/'payload/next-checkpoint.json'
        next_sha = sha(path.read_bytes())
        next_checkpoint, _ = checkpoint_inputs(path, next_sha)
        if (next_checkpoint['generation'] != checkpoint['generation']+1
                or next_checkpoint['previous_checkpoint_sha256'] != args.checkpoint_sha256
                or next_checkpoint['ledger'][:-1] != checkpoint.get('ledger', [])
                or next_checkpoint['ledger'][-1]['monomer_id'] != args.monomer
                or next_checkpoint['ledger'][-1]['receipts_sha256']['audit'] != sha(report_path.read_bytes())
                or next_checkpoint['audited_records'] != checkpoint['audited_records']+[report['record_id']]):
            raise ValueError('audit checkpoint is not the unique immediate accepted successor')
        confirm_live(next_checkpoint)
        state = dict(checkpoint=str(path), checkpoint_sha256=next_sha, generation=next_checkpoint['generation'])
        write_once(run/'verified.json', dict(state, phase=args.phase, job_id=job,
                   predecessor_sha256=args.checkpoint_sha256, audit_sha256=sha(report_path.read_bytes())))
        write_once(args.campaign/'states'/('%03d.json' % state['generation']), state)
        return state
    result = dict(phase=args.phase, job_id=job, monomer=args.monomer,
                  checkpoint_sha256=args.checkpoint_sha256)
    write_once(run/'verified.json', result)
    return result


def run_campaign(remote, wait, initial):
    """Small injectable orchestration seam; subprocesses live in the adapter."""
    state = remote('initialize', initial)
    for row in state.pop('remaining'):
        selected = dict(state, monomer=row['monomer_id'])
        for phase in PHASES:
            submitted = remote('submit', dict(selected, phase=phase))
            wait(submitted['job_id'])
            verified = remote('verify', dict(selected, phase=phase))
            if phase == 'audit':
                state = verified
    if state['generation'] != 131:
        raise ValueError('campaign stopped short of all131 audited representatives')
    return state


class Remote:
    def __init__(self, args):
        self.args = args
        skill = Path.home()/'.pi/agent/skills/phoenix-slurm-experiments/scripts'
        self.phx, self.watcher = skill/'phx', skill/'watch_jobs.sh'
        self.adapter_seal = self.seal()

    def seal(self):
        return sha(json.dumps({path.name: sha(path.read_bytes()) for path in (
            self.phx, self.watcher)}, sort_keys=True).encode())

    def call(self, action, state):
        args = self.args
        if self.seal() != self.adapter_seal:
            raise ValueError('installed orchestration adapters changed')
        command = [str(MANAGER/'bin/python'), '-I', '-B',
            str(args.source/'devtools/isapol_s66_campaign/native_controller.py'), 'remote',
            '--action', action, '--source', str(args.source), '--commit', args.commit,
            '--campaign', str(args.campaign), '--checkpoint', str(state['checkpoint']),
            '--checkpoint-sha256', state['checkpoint_sha256'], '--adapter-seal', self.adapter_seal]
        for key in ('monomer', 'phase'):
            if key in state:
                command += ['--'+key, state[key]]
        clean = ['/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', '-u', 'LD_PRELOAD',
                 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
                 'PATH='+str(MANAGER/'bin')+':/usr/bin:/bin', 'LD_LIBRARY_PATH='+str(MANAGER/'lib')]
        response = subprocess.run([str(self.phx), shlex.join(clean+command)],
            env=dict(os.environ, PHX_ALIAS='phoenix2'), text=True, capture_output=True, timeout=150)
        if response.returncode:
            raise RuntimeError('remote operation failed; no retry: '+response.stderr[-4000:])
        lines = [line[len(PREFIX):] for line in response.stdout.splitlines() if line.startswith(PREFIX)]
        if len(lines) != 1:
            raise RuntimeError('ambiguous remote response; no retry')
        result = json.loads(lines[0])
        print(json.dumps(dict(action=action, result=result), sort_keys=True), flush=True)
        return result

    def wait(self, job):
        if re.fullmatch(r'[1-9][0-9]*', job) is None:
            raise ValueError('positive recorded scheduler job required')
        if self.seal() != self.adapter_seal:
            raise ValueError('installed watcher/SSH adapter changed')
        subprocess.run([str(self.watcher), '-i', '300', 'phoenix2', job], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'remote'))
    for name in ('source', 'campaign', 'checkpoint'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--action', choices=('initialize', 'submit', 'verify'))
    parser.add_argument('--monomer')
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--adapter-seal')
    args = parser.parse_args()
    if (args.campaign.parent != RUNS or args.campaign.is_symlink()
            or re.fullmatch(r'ccasp-native-s66-ed561666fc-campaign[0-9]{3}', args.campaign.name) is None):
        raise ValueError('private exclusive campaign namespace required')
    require_hash(args.checkpoint_sha256)
    if args.mode == 'remote':
        validate_source(args.source, args.commit)
        require_hash(args.adapter_seal)
        if not args.action or (args.action != 'initialize' and (not args.monomer or not args.phase)):
            raise ValueError('explicit remote operation required')
        result = dict(initialize=initialize, submit=submit, verify=verify)[args.action](args)
        print(PREFIX+json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    else:
        if args.action or args.monomer or args.phase or args.adapter_seal:
            raise ValueError('remote-only arguments cannot be ignored')
        # The local adapter must be the same tracked bytes as the requested
        # frozen remote revision. Other local untracked research files are not
        # executed, copied or silently included.
        root = Path(__file__).resolve().parents[2]
        raw = subprocess.check_output(['git', '-C', str(root), 'show',
            args.commit+':devtools/isapol_s66_campaign/native_controller.py'])
        if raw != Path(__file__).read_bytes():
            raise ValueError('local controller differs from the requested frozen revision')
        remote = Remote(args)
        result = run_campaign(remote.call, remote.wait,
            dict(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256))
        print(PREFIX+json.dumps(dict(status='ALL_131_AUDITED_MODELS', **result)), flush=True)


if __name__ == '__main__':
    main()
