"""Bounded sequential eligible-bank dispatch; no scientific retry or old seed.

Each chunk starts only from a positively recorded state. An existing partial
selection/submission stops rather than repairing or replaying it. Scientific
acceptance still belongs to the separately allocated ordinary auditor.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace as NS

PREFIX = 'CCASP_BANK_CONTROLLER_RESULT='
PHASES = ('worker', 'science', 'audit')
_MODULES = {}


def helper(name):
    if name not in ('bank_campaign', 'bank_store', 'audited_bank', 'bank_audit',
                    'bank_dataset', 'native_controller', 'phoenix_worker_cb63305',
                    'phoenix_pilot_cb63305','runtime_audit_cb63305','qomp_expansion'):
        raise ValueError('closed eligible-bank controller helpers required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_bank_controller_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(path):
    return helper('bank_store')._read_regular(Path(path), maximum=64*1024**2)


def read(path):
    return json.loads(raw(path))


def campaign(args=None,*,producer='grid400'):
    producer=getattr(args,'producer',producer)
    if producer not in ('grid400','qomp'):
        raise ValueError('closed serial controller producer required')
    return helper('qomp_expansion' if producer=='qomp' else 'bank_campaign')


def root(args=None,*,producer='grid400'):
    return campaign(args,producer=producer).CAMPAIGN_ROOT


def phase_path(monomer,phase,*,producer='grid400'):
    if producer not in ('grid400','qomp'):
        raise ValueError('closed phase-path producer required')
    if producer=='qomp':
        return campaign(producer=producer).phase_path(monomer,phase)
    helper('native_controller').require_monomer(monomer)
    if phase not in PHASES:
        raise ValueError('closed bank phase required')
    return root()/('ccasp-native-s66-grid400-'+monomer+'-'+phase)


def checkpoint_args(state):
    return {name: (Path(state[name]) if name in ('bank_context', 'bank_checkpoint') else state[name])
            for name in ('bank_context', 'bank_context_sha256', 'bank_checkpoint', 'bank_checkpoint_sha256')}


def phase_command(source,commit,case,state,phase,*,producer='grid400',bootstrap_sha256=None):
    """Exact finite resources and wrapper signatures, independent of subprocess."""
    if phase not in PHASES:
        raise ValueError('closed bank phase required')
    if producer=='qomp':
        worker_sha=None
        if phase!='worker':
            w=phase_path(case['monomer_id'],'worker',producer='qomp')
            verified=read(w/'verified.json')
            worker_sha=verified['worker_preflight_sha256']
            if worker_sha!=sha(raw(w/'payload/worker-preflight.json')):
                raise ValueError('positively verified QOMP worker receipt changed')
        return campaign(producer='qomp').phase_command(source,commit,case,state,phase,
            bootstrap_sha256=bootstrap_sha256,worker_sha256=worker_sha)
    worker = helper('phoenix_worker_cb63305')
    worker.validate_campaign_scope(case['producer_profile'], case['campaign_scope'])
    profile = worker.helper('grid400_bank_policy').validate_profile(case['execution_profile'])
    mid = helper('native_controller').require_monomer(case['monomer_id'])
    cores, memory = (4, 24) if phase == 'audit' else (profile['cores'], profile['job_memory_GiB'])
    time = profile['time_limit'] if phase == 'science' else '00:45:00'
    run = phase_path(mid, phase)
    scripts = dict(worker='phoenix_worker_cb63305.sbatch', science='phoenix_pilot_cb63305.sbatch',
                   audit='runtime_audit_cb63305.sbatch')
    command = ['/opt/slurm/current/bin/sbatch', '--parsable', '--account=gts-cs207-chemx',
        '--partition=cpu-small', '--qos=embers', '--no-requeue', '--nodes=1', '--ntasks=1',
        '--cpus-per-task='+str(cores), '--mem='+str(memory)+'G', '--time='+time,
        '--job-name=ccasp-g400-'+mid[-4:]+'-'+phase,
        '--output='+str(run/'logs/job-%j.out'), '--error='+str(run/'logs/job-%j.err'),
        str(Path(source)/'devtools/isapol_s66_campaign'/scripts[phase]),
        str(source), commit, str(helper('bank_campaign').CASE_ROOT/(mid+'.json'))]
    preflight = phase_path(mid, 'worker')/'payload/worker-preflight.json'
    if phase != 'worker':
        command += [str(preflight)]
    if phase == 'audit':
        command += [str(phase_path(mid, 'science'))]
    command += [str(run)]
    if state['generation'] == 0:
        if case['dataset_mode'] != 'fresh':
            raise ValueError('initial bank dispatch requires its fresh bootstrap case')
        command += ['--grid400-bank-fresh']
    else:
        if case['dataset_mode'] != 'append':
            raise ValueError('subsequent bank dispatch requires its audited append case')
        bound = checkpoint_args(state)
        command += ['--grid400-bank-append', str(bound['bank_context']),
            bound['bank_context_sha256'], str(bound['bank_checkpoint']), bound['bank_checkpoint_sha256']]
    return command


def pool_inputs(args=None):
    c=campaign(args)
    if getattr(args,'producer','grid400')=='qomp':
        pool,inputs,cases,carry=c.actual_pool()
        return dict(pool,eligible_ids=sorted(cases),refused_ids=[],carried_original_ids=carry['accepted_ids']),inputs
    return c.load_pool(c.INPUTS,c.BUILD_IDENTITY,c.QUALIFICATION,c.DESCRIPTORS)


def initialize(args):
    if getattr(args,'producer','grid400')=='qomp':
        return campaign(args).initialize(args)
    pool, inputs = pool_inputs()  # No filesystem side effects until the actual pool admits.
    cases = helper('bank_campaign').prepared_cases(pool, inputs)
    core, io = helper('audited_bank'), helper('native_controller')
    io.mkdir_once(root())
    for name in ('states', 'cases', 'selections'):
        io.mkdir_once(root()/name)
    plan = dict(schema='native-eligible-bank-controller/1', source=str(args.source), commit=args.commit,
        controller_sha256=sha(Path(__file__).read_bytes()), adapter_sha256=args.adapter_seal,
        eligibility_sha256=cases[pool['eligible_ids'][0]]['eligibility_sha256'],
        eligible_ids=pool['eligible_ids'], refused_ids=pool['refused_ids'],
        prepared_sha256={mid: sha(core.encode(case)) for mid, case in cases.items()})
    io.write_once(root()/'plan.json', plan)
    state = dict(generation=0, status='NO_PRODUCTION_MODEL_ACCEPTED', bank_context=None,
        bank_context_sha256=None, bank_checkpoint=None, bank_checkpoint_sha256=None)
    io.write_once(root()/'states/000.json', state)
    return state


def context(args):
    c, core = campaign(args), helper('audited_bank')
    qomp=getattr(args,'producer','grid400')=='qomp'
    pool,inputs=pool_inputs(args) if qomp else pool_inputs()
    cases=(c.actual_pool()[2] if qomp else c.prepared_cases(pool,inputs))
    plan = read(root(args)/'plan.json')
    if qomp and plan.get('bootstrap_sha256')!=args.bootstrap_sha256:
        raise ValueError('parent-pinned QOMP bootstrap changed')
    if (plan['source'] != str(args.source) or plan['commit'] != args.commit
            or plan['controller_sha256'] != sha(Path(__file__).read_bytes())
            or plan['adapter_sha256'] != args.adapter_seal
            or plan['eligible_ids'] != pool['eligible_ids'] or plan['refused_ids'] != pool['refused_ids']
            or plan['eligibility_sha256'] != cases[pool['eligible_ids'][0]]['eligibility_sha256']
            or plan['prepared_sha256'] != {mid: sha(core.encode(case)) for mid, case in cases.items()}):
        raise ValueError('immutable eligible-bank plan or source changed')
    paths = sorted((root(args)/'states').glob('[0-9][0-9][0-9].json'))
    if not paths:
        raise ValueError('no positively recorded controller state')
    state = read(paths[-1])
    generation = state['generation']
    if (type(generation) is not int or not 0 <= generation <= len(pool['eligible_ids'])
            or paths[-1].name != '%03d.json' % generation):
        raise ValueError('exact monotonic eligible-bank generation required')
    bound = None
    if generation:
        bound = c.load_checkpoint(NS(**checkpoint_args(state),
            **(dict(bootstrap_sha256=args.bootstrap_sha256) if qomp else {})))
        checkpoint = json.loads(bound['checkpoint_raw'])
        context_data = json.loads(bound['context_raw'])
        if (checkpoint['generation'] != generation or context_data['eligible_ids'] != plan['eligible_ids']
                or context_data['prepared_case_sha256'] != plan['prepared_sha256']):
            raise ValueError('recorded audited checkpoint differs from measured eligible bank')
        # A state file alone cannot promote a failed or running audit allocation.
        previous_mid = pool['eligible_ids'][generation-1]
        if qomp and generation==1:
            c.validate_initial_state(state,bound)
        else:
            audit_run = phase_path(previous_mid,'audit',producer='qomp' if qomp else 'grid400')
            verification = read(audit_run/'verified.json')
            report = read(audit_run/'payload/audit-report.json')
            if (verification.get('state') != state or verification.get('phase') != 'audit'
                or verification.get('audit_sha256') != sha(raw(audit_run/'payload/audit-report.json'))
                or verification.get('job_id') != report['allocation']['JobId']):
                raise ValueError('accepted state lacks matching positive audit verification')
    elif state != dict(generation=0, status='NO_PRODUCTION_MODEL_ACCEPTED', bank_context=None,
            bank_context_sha256=None, bank_checkpoint=None, bank_checkpoint_sha256=None):
        raise ValueError('initial bank state cannot claim an accepted model')
    if qomp and generation==0:
        raise ValueError('QOMP expansion cannot invent a fresh seed')
    mid = pool['eligible_ids'][generation] if generation < len(pool['eligible_ids']) else None
    case = cases[mid] if mid is not None else None
    if bound is not None and mid is not None:
        case = core.bind_next_case(case, bound['checkpoint_raw'], bound['checkpoint_sha256'],
            bound['context_raw'],bound['context_sha256'],bound['proof_blobs'],
            **(c.reducer_options(bound['qomp_binding']) if qomp else {}))
    return pool, state, case, bound


def select(args):
    pool, state, case, _ = context(args)
    if case is None:
        if getattr(args,'producer','grid400')=='qomp':
            return dict(complete=True,generation=state['generation'],eligible_count=len(pool['eligible_ids']),
                refused_ids=[],carried_original_count=len(pool['carried_original_ids']),
                cohort_transport_complete=True,whole_bank_complete=False)
        return dict(complete=True, generation=state['generation'], eligible_count=len(pool['eligible_ids']),
                    refused_ids=pool['refused_ids'], whole_bank_complete=not pool['refused_ids'])
    mid, io = case['monomer_id'], helper('native_controller')
    # Existing partial selection is deliberately not resumable or automatically repaired.
    c=campaign(args)
    if getattr(args,'producer','grid400')=='qomp':
        helper('bank_store')._write_once(c.CASE_ROOT/(mid+'.json'),helper('audited_bank').encode(case))
    else:
        io.write_once(c.CASE_ROOT/(mid+'.json'),case)
    selection = dict(monomer=mid, predecessor=state,
        case_sha256=sha(raw(c.CASE_ROOT/(mid+'.json'))))
    io.write_once(root(args)/'selections'/(mid+'.json'),selection)
    return dict(complete=False, **selection)


def selected(args):
    pool, state, case, bound = context(args)
    if case is None or case['monomer_id'] != args.monomer:
        raise ValueError('only the next unaudited eligible monomer may dispatch')
    selection = read(root(args)/'selections'/(args.monomer+'.json'))
    path = campaign(args).CASE_ROOT/(args.monomer+'.json')
    if (selection != dict(monomer=args.monomer, predecessor=state, case_sha256=sha(raw(path)))
            or json.loads(raw(path)) != case):
        raise ValueError('selected case or predecessor changed')
    return pool, state, case, bound, selection


def submit(args):
    _, state, case, _, selection = selected(args)
    phase = args.phase
    producer=getattr(args,'producer','grid400')
    location=lambda p:phase_path(args.monomer,p,producer=producer)
    if phase not in PHASES:
        raise ValueError('closed bank phase required')
    if phase != 'worker':
        previous = PHASES[PHASES.index(phase)-1]
        verified = read(location(previous)/'verified.json')
        if (verified.get('selection') != selection or verified.get('phase') != previous
                or verified.get('job_id') != read(location(previous)/'submission.json')['job_id']):
            raise ValueError('matching positive predecessor phase required')
    queue = subprocess.check_output(['/opt/slurm/current/bin/squeue', '-u', 'awallace43',
                                     '-r', '-h', '-o', '%q'], text=True)
    if sum(line.strip() == 'embers' for line in queue.splitlines()) >= 49:
        raise ValueError('insufficient embers headroom; no submission')
    io, run = helper('native_controller'), location(phase)
    quota=None
    if producer=='qomp':
        quota=campaign(args).controller_quota(case)
        if phase=='worker':
            io.mkdir_once(run.parent)
    io.mkdir_once(run)
    io.mkdir_once(run/'logs')
    command = phase_command(args.source,args.commit,case,state,phase,
        **(dict(producer='qomp',bootstrap_sha256=args.bootstrap_sha256) if producer=='qomp' else {}))
    subprocess.run(command[:1]+['--test-only']+command[1:], check=True, capture_output=True, text=True)
    intent = dict(selection=selection, phase=phase, command=command)
    if producer=='qomp':
        intent['scratch_admission']=quota
    io.write_once(run/'intent.json', intent)
    if producer=='qomp':
        io.write_once(run/'pre-submit-quota.json',campaign(args).controller_quota(case))
    response = subprocess.run(command, check=False, capture_output=True, text=True)
    io.write_once(run/'submission-response.json',
                  dict(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
    job = response.stdout.strip()
    if response.returncode or re.fullmatch('[1-9][0-9]*', job) is None:
        raise RuntimeError('ambiguous/failed bank submission; never retry or clear the intent')
    io.write_once(run/'submission.json', dict(intent, job_id=job))
    return dict(phase=phase, monomer=args.monomer, job_id=job)


def verify(args):
    pool, state, case, bound, selection = selected(args)
    phase, io = args.phase, helper('native_controller')
    producer=getattr(args,'producer','grid400')
    qomp=producer=='qomp'
    location=lambda p:phase_path(args.monomer,p,producer=producer)
    run=location(phase)
    submission = read(run/'submission.json')
    job = submission['job_id']
    if (submission.get('selection') != selection or submission.get('phase') != phase
            or submission.get('command') != phase_command(args.source,args.commit,case,state,phase,
                **(dict(producer='qomp',bootstrap_sha256=args.bootstrap_sha256) if qomp else {}))):
        raise ValueError('recorded phase submission changed')
    accounting=subprocess.check_output(['/opt/slurm/current/bin/sacct','-n','-P',
        '-j',job,'-o','JobIDRaw,State,ExitCode'],text=True)
    (campaign(args).completed_rows if qomp else io.completed_rows)(accounting,job)
    worker, science = helper('phoenix_worker_cb63305'), helper('phoenix_pilot_cb63305')
    c=campaign(args)
    worker_path=location('worker')/'payload/worker-preflight.json'
    preflight = read(worker_path)
    scope=dict(producer=producer,campaign_scope='qomp-serial' if qomp else 'eligible-bank')
    if phase == 'worker':
        worker.admit_allocation(preflight['allocation'], execution_profile=case['execution_profile'], **scope)
        if preflight['allocation']['JobId'] != job or preflight['operational_commit'] != args.commit:
            raise ValueError('actual bank worker job/source differs')
        qualification=(c.helper('qomp_campaign').QUALIFICATION if qomp else c.QUALIFICATION)
        science.validate_preflight(preflight,case,
            manifest_sha=sha(raw(qualification if qomp else c.BUILD_IDENTITY)),
            qualification_sha=sha(raw(qualification)),pilot_sha=selection['case_sha256'],
            worker_sources=worker.source_hashes(remaining_case=bound is not None, **scope),
            transport=helper('bank_dataset').transport_binding(**(dict(producer='qomp') if qomp else {})),
            remaining_case=bound is not None,**scope,
            **(dict(operational_commit=args.commit,checkpoint_binding=bound) if qomp else {}))
    elif phase == 'science':
        outcome = read(run/'payload'/('qomp-outcome.json' if qomp else 'pilot-outcome.json'))
        science.admit_allocation(outcome['allocation'], execution_profile=case['execution_profile'], **scope)
        if (outcome['allocation']['JobId'] != job or outcome['operational_commit'] != args.commit
                or outcome['pilot_sha256'] != selection['case_sha256']
                or outcome['worker_preflight_sha256'] != sha(raw(worker_path))
                or outcome['monomer_id'] != args.monomer or outcome.get('campaign_scope') != scope['campaign_scope']
                or outcome['status'] != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
                or outcome['find_existing'] is not False or outcome['qcengine_internal_retries'] != 0):
            raise ValueError('actual fresh bank science delivery differs')
        auditor = helper('bank_audit')
        if bound is None:
            auditor.validate_initial_capture(run/'payload', outcome, case)
        else:
            auditor.validate_preservation(run/'payload',outcome,case,bound,
                helper('audited_bank').normalize_specification(case['specification']),
                **(dict(producer='qomp') if qomp else {}))
    else:
        report_path = run/'payload/audit-report.json'
        report, advanced = read(report_path), read(run/'payload/bank-advance.json')
        io.admit_audit_allocation(report['allocation'], job)
        helper('bank_audit').passing_report(raw(report_path),bound,
            **(dict(producer='qomp') if qomp else {}))
        science_run=location('science')
        outcome = read(science_run/'payload'/('qomp-outcome.json' if qomp else 'pilot-outcome.json'))
        if (report['audit_commit'] != args.commit or report['case_operational_commit'] != args.commit
                or report['monomer_id'] != args.monomer or report['dataset_id'] != outcome['dataset_id']
                or report['record_id'] != outcome['record_id']
                or report['case_job'] != read(science_run/'submission.json')['job_id']
                or report['roles'].get('molecular_qcschema') != 1 or not report['traces']
                or any(value['status'] != 'passed' for value in report['traces'].values())
                or advanced['audit_sha256'] != sha(raw(report_path))
                or advanced['generation'] != state['generation']+1):
            raise ValueError('positive independent bank audit/publication differs')
        new_state = dict(generation=advanced['generation'], status='ORDINARY_AUDIT_ACCEPTED',
            bank_context=advanced['context_path'], bank_context_sha256=advanced['context_sha256'],
            bank_checkpoint=advanced['checkpoint_path'], bank_checkpoint_sha256=advanced['checkpoint_sha256'])
        if (Path(new_state['bank_context']) != run/'payload/bank-context.json'
                or Path(new_state['bank_checkpoint']) != run/'payload/bank-checkpoint.json'):
            raise ValueError('audit checkpoint must come from this exact new audit allocation')
        new_bound=c.load_checkpoint(NS(**checkpoint_args(new_state),
            **(dict(bootstrap_sha256=args.bootstrap_sha256) if qomp else {})))
        checkpoint = json.loads(new_bound['checkpoint_raw'])
        context_data = json.loads(new_bound['context_raw'])
        proof = (context_data['seed_receipts_sha256'] if bound is None
                 else checkpoint['ledger'][-1]['receipts_sha256'])
        if (checkpoint['generation'] != new_state['generation']
                or proof['audit'] != sha(raw(report_path))
                or context_data['eligible_ids'] != pool['eligible_ids']
                or checkpoint['audited_records'][-1] != report['record_id']
                or (bound is not None and (new_bound['context_raw'] != bound['context_raw']
                    or checkpoint['previous_checkpoint_sha256'] != bound['checkpoint_sha256']))):
            raise ValueError('audit did not create the unique immediate ordinary accepted successor')
        client=helper('bank_audit').approved_client(case,outcome,bound,
            **(dict(producer='qomp') if qomp else {}))
        live = helper('bank_audit').helper('dataset_guard').snapshot_dataset(
            client, dataset_id=report['dataset_id'], max_entries=len(pool['eligible_ids']))
        if helper('bank_audit').helper('dataset_guard').digest(live) != checkpoint['snapshot_sha256']:
            raise ValueError('live bank differs after positive audit completion')
        if qomp:
            c.validate_source(args.source,args.commit)
        io.write_once(run/'verified.json', dict(phase=phase, selection=selection,
            job_id=job, state=new_state, audit_sha256=sha(raw(report_path))))
        io.write_once(root(args)/'states'/('%03d.json' % new_state['generation']),new_state)
        return new_state
    result = dict(phase=phase, job_id=job, selection=selection)
    if qomp and phase=='worker':
        result['worker_preflight_sha256']=sha(raw(worker_path))
    if qomp:
        c.validate_source(args.source,args.commit)
    io.write_once(run/'verified.json', result)
    return result


def run_chunk(remote, wait, maximum):
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError('one through eight sequential monomers per bounded chunk required')
    for _ in range(maximum):
        selection = remote('select', {})
        if selection['complete']:
            return selection
        for phase in PHASES:
            fields = dict(monomer=selection['monomer'], phase=phase)
            submitted = remote('submit', fields)
            wait(submitted['job_id'])
            state = remote('verify', fields)
    return dict(chunk_complete=True, generation=state['generation'],
                qualification='Only positively audited progress; refused monomers remain held.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('setup', 'run', 'remote'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--maximum', type=int, default=1)
    parser.add_argument('--action', choices=('initialize', 'select', 'submit', 'verify'))
    parser.add_argument('--monomer')
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--adapter-seal')
    parser.add_argument('--producer',choices=('grid400','qomp'),default='grid400')
    parser.add_argument('--bootstrap-sha256')
    args = parser.parse_args()
    if args.mode == 'remote':
        if args.producer=='qomp':
            campaign(args).load_binding(args.bootstrap_sha256)
            campaign(args).validate_source(args.source,args.commit)
        else:
            helper('bank_campaign').authority()
            helper('native_controller').validate_source(args.source,args.commit)
        helper('native_controller').require_hash(args.adapter_seal)
        if args.action in ('submit', 'verify') and (not args.monomer or not args.phase):
            raise ValueError('explicit selected monomer and phase required')
        result = dict(initialize=initialize, select=select, submit=submit, verify=verify)[args.action](args)
        if args.producer=='qomp':
            campaign(args).validate_source(args.source,args.commit)
        else:
            helper('native_controller').validate_source(args.source,args.commit)
        print(PREFIX+json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    else:
        native = helper('native_controller')
        class Remote(native.Remote):
            def call(self, action, state):
                if self.seal() != self.adapter_seal:
                    raise ValueError('installed SSH/watcher adapters changed')
                command = [str(native.MANAGER/'bin/python'), '-I', '-B',
                    str(args.source/'devtools/isapol_s66_campaign/bank_controller.py'), 'remote',
                    '--action', action, '--source', str(args.source), '--commit', args.commit,
                    '--adapter-seal', self.adapter_seal]
                if args.producer=='qomp':
                    command += ['--producer','qomp','--bootstrap-sha256',args.bootstrap_sha256]
                for key in ('monomer', 'phase'):
                    if key in state:
                        command += ['--'+key, state[key]]
                clean = ['/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', '-u', 'LD_PRELOAD',
                    'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
                    'PATH='+str(native.MANAGER/'bin')+':/usr/bin:/bin',
                    'LD_LIBRARY_PATH='+str(native.MANAGER/'lib')]
                if args.producer=='qomp':
                    clean=['/usr/bin/env','-i','USER=awallace43','LOGNAME=awallace43',
                        'HOME=/storage/home/hcoda1/3/awallace43','LANG=C.UTF-8','LC_ALL=C.UTF-8',
                        'GIT_OPTIONAL_LOCKS=0','PATH='+str(native.MANAGER/'bin')+':/usr/bin:/bin',
                        'LD_LIBRARY_PATH='+str(native.MANAGER/'lib')]
                response = subprocess.run([str(self.phx), shlex.join(clean+command)],
                    env=dict(os.environ, PHX_ALIAS='phoenix2'), text=True, capture_output=True, timeout=150)
                lines = [line[len(PREFIX):] for line in response.stdout.splitlines() if line.startswith(PREFIX)]
                if response.returncode or len(lines) != 1:
                    raise RuntimeError('failed/ambiguous remote bank operation; no automatic retry')
                result = json.loads(lines[0])
                print(json.dumps(dict(action=action, result=result), sort_keys=True), flush=True)
                return result
        remote = Remote(args)
        if args.mode == 'setup':
            remote.call('initialize', {})
        else:
            print(json.dumps(run_chunk(remote.call, remote.wait, args.maximum), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
