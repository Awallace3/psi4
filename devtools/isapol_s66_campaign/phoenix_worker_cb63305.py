"""Fresh request3 manager discovery preflight; no SCF or QCArchive writes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

RUNS = Path('/storage/project/r-cs207-0/awallace43/runs')
SCRATCH = Path('/storage/scratch1/3/awallace43')
BASE = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt')
MANAGER = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca')
SOURCES = (
    'phoenix_worker_cb63305.py', 'phoenix_worker_cb63305.sbatch', 'pilot_cb63305.py',
    'case_cb63305.py',
    'dataset_fresh_cb63305.py', 'dataset_append.py', 'prepare_descriptor_bank_cb63305.py',
    'qualified_cb63305.py', 'resource_plan_cb63305.py', 'descriptor_probe_cb63305.py',
    'descriptor_probe.py', 'descriptor_admission_cb63305.py', 'descriptor_scratch_cb63305.py',
    'descriptor_profiles.py',
    'phoenix_descriptor_bank_cb63305.py', 'phoenix_installed_checks.py',
    'phoenix_worker_probe.py', 'phoenix_pilot.py', 'single_claim.py', 'native_psi4.sh', 'term_wait.sh',
    'producer_paths.sh',
)
HELPERS = {
    'pilot_cb63305', 'phoenix_installed_checks', 'phoenix_worker_probe', 'phoenix_pilot',
    'descriptor_admission_cb63305', 'descriptor_scratch_cb63305',
    'phoenix_descriptor_bank_cb63305', 'dataset_fresh_cb63305', 'case_cb63305',
    'descriptor_profiles',
    'grid400_policy',
    'grid400_bank_policy', 'bank_campaign', 'bank_dataset', 'bank_store',
    'bank_wave_runtime',
    'qomp_campaign','qomp_execution',
}
_MODULES = {}


def helper(name):
    if name not in HELPERS:
        raise ValueError('closed expanded worker helpers required')
    if name == 'bank_wave_runtime' and name in _MODULES:
        return _MODULES[name]
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_expanded_worker_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    if name == 'bank_wave_runtime':
        _MODULES[name] = module
    return module


def selected_producer(args):
    producer = getattr(args, 'producer', 'cb63305')
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400','qomp'):
        raise ValueError('closed molecular producer selector required')
    return producer


def validate_campaign_scope(producer, scope):
    if producer=='qomp':
        if scope!='qomp-pilot':
            raise ValueError('qomp requires its explicit fresh serial pilot scope')
        return scope
    if (type(scope) is not str or scope not in ('default', 'eligible-bank', 'eligible-wave')
            or scope in ('eligible-bank', 'eligible-wave') and producer != 'grid400'):
        raise ValueError('eligible bank requires explicit grid400 producer and campaign scope')
    return scope


def selected_scope(args):
    scope = validate_campaign_scope(selected_producer(args), getattr(args, 'campaign_scope', 'default'))
    if scope != 'eligible-wave' and any(getattr(args, key, None) is not None for key in (
            'wave_plan', 'wave_plan_sha256', 'wave_registration', 'wave_registration_sha256',
            'wave_science_barrier', 'wave_science_barrier_sha256')):
        raise ValueError('wave inputs require explicit eligible-wave scope')
    return scope


def source_names(producer='cb63305', *, remaining_case=False, campaign_scope='default'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400','qomp'):
        raise ValueError('closed implemented worker producer required')
    if producer!='qomp':
        helper('pilot_cb63305').producer_pins(producer)
    validate_campaign_scope(producer, campaign_scope)
    if type(remaining_case) is not bool:
        raise ValueError('explicit worker case mode required')
    if producer=='qomp':
        if remaining_case:
            raise ValueError('qomp append/wave execution is not implemented in148A')
        return tuple(dict.fromkeys(SOURCES+helper('qomp_execution').SOURCE_NAMES))
    if producer == 'grid400' and remaining_case and campaign_scope == 'default':
        raise ValueError('grid400 is fresh singleton only, never a remaining case')
    extra = (('qualified_ed561666.py', 'resource_plan_ed561666.py',
              'prepare_ed561666_inputs.py') if producer == 'ed561666' else ())
    if producer == 'ed561666' and remaining_case:
        extra += ('owned_dataset_cb63305.py',)
    if producer == 'grid400':
        extra += ('qualified_grid400.py', 'resource_plan_ed561666.py',
                  'resource_plan_grid400.py', 'grid400_policy.py', 'grid400_cohort.py',
                  'envelope_identity.py')
    if campaign_scope == 'eligible-bank':
        extra += helper('bank_campaign').ADDITIONAL_SOURCES
    elif campaign_scope == 'eligible-wave':
        extra += helper('bank_wave_runtime').source_names()
    return tuple(dict.fromkeys(SOURCES + extra))


def source_hashes(producer='cb63305', *, remaining_case=False, campaign_scope='default'):
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in source_names(producer, remaining_case=remaining_case, campaign_scope=campaign_scope)}


def admit_allocation(allocation, *, execution_profile=None, producer='cb63305', campaign_scope='default'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400','qomp'):
        raise ValueError('closed worker allocation producer required')
    validate_campaign_scope(producer, campaign_scope)
    expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                    NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00')
    if producer=='qomp':
        profile=helper('qomp_execution').policy().validate_profile(execution_profile)
        expected.update(NumCPUs=str(profile['cores']),MinMemoryNode=str(profile['job_memory_GiB'])+'G')
    elif producer == 'grid400':
        profile = helper('grid400_bank_policy' if campaign_scope in ('eligible-bank', 'eligible-wave')
                         else 'grid400_policy').validate_profile(execution_profile)
        expected.update(NumCPUs=str(profile['cores']),
                        MinMemoryNode=str(profile['job_memory_GiB'])+'G')
    elif execution_profile is not None:
        cases = helper('case_cb63305')
        name = execution_profile.get('name')
        if name not in cases.PROFILES:
            raise ValueError('explicit finite remaining-case worker profile required')
        profile = dict(name=name, **cases.PROFILES[name], account='gts-cs207-chemx',
                       partition='cpu-small', qos='embers', concurrency=1)
        if cases.json_digest(profile) != cases.json_digest(execution_profile):
            raise ValueError('remaining-case worker profile changed')
        expected.update(NumCPUs=str(profile['cores']),
                        MinMemoryNode=str(profile['job_memory_GiB'])+'G')
    if any(allocation.get(k) != v for k, v in expected.items()):
        raise ValueError('exact effective expanded-worker embers allocation required')


def prepare_draft(args, manifest_path):
    """Reconstruct either the archived first-case shape or a full remaining case."""
    producer = selected_producer(args)
    if producer=='qomp':
        return helper('qomp_campaign').verify_prepared_from_paths(args)
    if selected_scope(args) == 'eligible-wave':
        return helper('bank_wave_runtime').prepare_draft(args, manifest_path)
    if selected_scope(args) == 'eligible-bank':
        return helper('bank_campaign').verify_prepared_from_paths(args, manifest_path)
    if producer == 'grid400' and args.remaining_case:
        raise ValueError('grid400 requires a new fresh singleton, not an append case')
    if args.remaining_case:
        return helper('case_cb63305').verify_prepared_from_paths(
            args.pilot, args.inputs, manifest_path, args.qualification, args.descriptors,
            **({} if producer == 'cb63305' else dict(producer=producer)))
    draft = helper('pilot_cb63305').prepare_from_paths(
        args.inputs, manifest_path, args.qualification, args.descriptors, producer=producer)
    if json.loads(args.pilot.read_bytes()) != draft:
        raise ValueError('pilot does not reproduce from admitted raw/descriptor/producer receipts')
    return draft


def validate_versions(qcengine, compute):
    if qcengine != 'v0.34.1+3.gd56a0598' or compute != '0.63':
        raise ValueError('reviewed deployed QCEngine and compute0.63 required')


def discover_manager_programs(environment, discover):
    """Plain discovery python3 children must not read ambient cached bytecode."""
    if (not environment.get('PYTHONPYCACHEPREFIX')
            or environment['PYTHONPYCACHEPREFIX'] != environment.get('CCASP_NATIVE_CACHE')):
        raise ValueError('fresh explicit discovery bytecode-cache prefix required')
    os.environ.clear()
    os.environ.update(environment)
    return discover(None)['qcengine.compute']['programs']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'qualification', 'inputs', 'descriptors', 'pilot', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666', 'grid400','qomp'), default='cb63305')
    parser.add_argument('--remaining-case', action='store_true',
                        help='Require a fully reconstructed remaining-bank case, not the first pilot')
    parser.add_argument('--campaign-scope', choices=('default', 'eligible-bank', 'eligible-wave','qomp-pilot'), default='default')
    parser.add_argument('--wave-plan', type=Path)
    parser.add_argument('--wave-plan-sha256')
    for name in ('bank-context', 'bank-checkpoint'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if getattr(args,'producer','cb63305')=='qomp':
        return helper('qomp_execution').run_phase('worker',args,perform)
    return perform(args)


def perform(args):
    producer = selected_producer(args)
    scope = selected_scope(args)
    qomp=producer=='qomp'
    q=helper('qomp_execution') if qomp else None
    if scope == 'eligible-bank':
        helper('bank_campaign').authority()
    if scope == 'eligible-wave':
        helper('bank_wave_runtime').validate_args(args)
    scope_args = {} if scope == 'default' else dict(campaign_scope=scope)
    pins = (helper('qomp_campaign').pins() if qomp else helper('bank_wave_runtime').pins() if scope == 'eligible-wave'
            else helper('bank_campaign').pins() if scope == 'eligible-bank'
            else helper('pilot_cb63305').producer_pins(producer))
    checks = helper('phoenix_installed_checks')
    git=q.git if qomp else checks.git
    allocation = checks.allocation()
    source = Path(__file__).resolve().parents[2]
    if (git(source, 'rev-parse', 'HEAD') != args.commit
            or git(source, 'status', '--porcelain')):
        raise ValueError('exact clean operational checkout required')
    build, run = args.build.resolve(), args.run.resolve()
    if (not build.is_relative_to(RUNS) or not run.is_relative_to(RUNS)
            or scope != 'eligible-wave' and not run.name.startswith('ccasp-native-s66-')
            or run.is_relative_to(build) or build.is_relative_to(run)):
        raise ValueError('disjoint private build and worker run required')
    if scope == 'eligible-wave':
        helper('bank_wave_runtime').validate_paths(args, role='worker')
    manifest_path = helper('qomp_campaign').QUALIFICATION if qomp else build/'build-identity.json'
    draft = prepare_draft(args, manifest_path)
    profile = draft.get('execution_profile')
    admit_allocation(allocation, execution_profile=profile,
                     **(dict(producer=producer, **scope_args) if producer in ('grid400','qomp') else {}))
    if scope == 'eligible-bank':
        helper('bank_store').prepare_storage()
    pilot_sha = checks.digest(args.pilot)
    manifest = q.context(args) if qomp else json.loads(manifest_path.read_bytes())
    identity = json.loads(args.qualification.with_name('imported-identity.json').read_bytes())
    changes = (q.source_changes(source,manifest['commit'],args.commit) if qomp else
               git(source, 'diff', '--name-only', manifest['commit'], args.commit).splitlines())
    if qomp:
        q.qualified().verify_descriptor_operations(source,changes)
    elif any(not p.startswith('devtools/isapol_s66_campaign/')
           and p not in checks.TEST_ONLY_CHANGES for p in changes):
        raise ValueError('worker operations change scientific/build sources')
    operations = source_hashes(producer, remaining_case=args.remaining_case, **scope_args)
    manager_sources = MANAGER/'lib/python3.13/site-packages/qcfractalcompute'
    manager_pins = helper('phoenix_pilot').MANAGER_FILES
    if any(checks.digest(manager_sources/name) != expected for name, expected in manager_pins.items()):
        raise ValueError('reviewed deployed manager sources changed')
    environment = build/'env'
    runtime_check=(q.runtime_check if qomp else lambda: checks.verify_runtime(environment,manifest))
    runtime_check()
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    for name in ('traces', 'launcher'):
        (payload/name).mkdir()
    (payload/'launcher/psi4').symlink_to(source/'devtools/isapol_s66_campaign'/
                                      ('qomp_launcher.sh' if qomp else 'native_psi4.sh'))
    scratch = SCRATCH/(pins['worker_namespace']+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir()
    common = helper('phoenix_descriptor_bank_cb63305')
    disk = (q.scratch_admission(scratch,profile) if qomp else
            common.quota_snapshot(scratch, payload, helper('descriptor_scratch_cb63305')))
    env = common.child_environment(environment, scratch)
    if profile is not None:
        env.update(OMP_NUM_THREADS=str(profile['cores']), MKL_NUM_THREADS=str(profile['cores']))
    if qomp:
        env.update(q.launch_binding(args,draft,payload,scratch,'worker'))
    probe = helper('phoenix_worker_probe')
    runtime = probe.runtime_input(draft)
    runtime_path = payload/'runtime-input.json'
    with runtime_path.open('x') as stream:
        json.dump(runtime, stream, sort_keys=True, allow_nan=False)
    if checks.digest(runtime_path) != draft['runtime_input_sha256']:
        raise ValueError('scientific worker runtime differs from admitted descriptor input')
    science_path = payload/'science-descriptors.json'
    command = checks.installed_python_command(environment/'bin/python', scratch/'cache',
        str(Path(__file__).with_name('descriptor_probe_cb63305.py')),
        '--input', str(runtime_path), '--output', str(science_path),
        '--memory-bytes', str(draft['intended_memory_bytes']), '--monomer-id', draft['monomer_id'],
        '--producer', producer)
    descriptor_scope = {} if scope == 'default' else dict(scope='full-bank')
    if descriptor_scope:
        command += ['--scope', 'full-bank']
    if qomp:
        command += ['--runtime-view',str(helper('qomp_campaign').VIEW)]
    subprocess.run(command, env=env, cwd=scratch/'cwd', stdin=subprocess.DEVNULL, check=True)
    science = json.loads(science_path.read_bytes())
    admitted_row = dict(monomer_id=draft['monomer_id'], runtime_input=runtime,
        runtime_input_sha256=draft['runtime_input_sha256'],
        intended_memory_bytes=draft['intended_memory_bytes'])
    result = helper('descriptor_admission_cb63305').verify_descriptor(science, admitted_row, identity,
        policy=science['source_policy_sha256'],
        operational_sources_sha256={key: operations[key]
            for key in helper('descriptor_profiles').profile(producer, **descriptor_scope)['child_sources']},
        runtime_sha256=draft['runtime_input_sha256'], producer=producer, **descriptor_scope)
    if result['status'] != 'PASS_DESCRIPTOR_PLAN_ONLY':
        raise ValueError('fresh worker descriptor plan refused; no automatic promotion')
    env.update(PATH=f'{payload}/launcher:{MANAGER}/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH=str(MANAGER/'lib'), CCASP_NATIVE_ENV=str(environment),
        CCASP_NATIVE_BASE=str(BASE), CCASP_NATIVE_CACHE=str(scratch/'cache'),
        PYTHONPYCACHEPREFIX=str(scratch/'cache'),
        CCASP_NATIVE_HOME=str(scratch/'home'), CCASP_NATIVE_TRACE=str(payload/'traces'))
    os.environ.clear()
    os.environ.update(env)
    if (Path(sys.executable).resolve() != (MANAGER/'bin/python').resolve()
            or shutil.which('python3') != str(MANAGER/'bin/python3')):
        raise ValueError('parent/discovery must keep the declared manager interpreter')
    import inspect
    import qcengine
    import qcfractalcompute
    from qcengine.config import get_config
    from qcfractalcompute.apps.app_manager import discover_programs_conda
    validate_versions(str(qcengine.__version__), str(qcfractalcompute.__version__))
    worker_shape = (profile['cores'], profile['science_memory_GiB']) if profile else (4, 20)
    task_config = get_config(task_config=dict(ncores=worker_shape[0], memory=float(worker_shape[1]),
                                             scratch_directory=str(scratch)))
    if (task_config.ncores != worker_shape[0] or task_config.memory != worker_shape[1]
            or task_config.retries != 0):
        raise ValueError('exact worker cores/scientific memory and zero internal retries required')
    transport_adapter = (q.dataset() if qomp else helper('bank_wave_runtime').helper('bank_wave_transport') if scope == 'eligible-wave'
                         else helper('bank_dataset' if scope == 'eligible-bank' else 'dataset_fresh_cb63305'))
    transport_args = dict(producer='qomp') if qomp else {} if scope != 'default' else dict(producer=producer)
    transport = transport_adapter.transport_binding(**transport_args)
    harness = qcengine.get_program('psi4')
    harness_path = Path(inspect.getfile(type(harness))).resolve()
    if harness_path != Path('/storage/project/r-cs207-0/awallace43/gits/qcengine/qcengine/programs/psi4.py'):
        raise ValueError('unexpected deployed QCEngine harness')
    harness_sha = checks.digest(harness_path)
    version = harness.get_version()
    programs = discover_manager_programs(env, discover_programs_conda)
    probe.validate_worker_discovery(programs, version, science['version'],
                                   scientific_imported='psi4' in sys.modules)
    runtime_check()
    if (source_hashes(producer, remaining_case=args.remaining_case, **scope_args) != operations
            or git(source, 'rev-parse', 'HEAD') != args.commit
            or git(source, 'status', '--porcelain')
            or checks.digest(args.pilot) != pilot_sha
            or checks.digest(harness_path) != harness_sha
            or transport_adapter.transport_binding(**transport_args) != transport
            or any(checks.digest(manager_sources/name) != expected for name, expected in manager_pins.items())
            or prepare_draft(args, manifest_path) != draft):
        raise ValueError('worker source, runtime, raw inputs or harness changed during discovery')
    report = dict(status=('PASS_SELECTED_WAVE_CASE_WORKER_PREFLIGHT_ONLY' if scope == 'eligible-wave'
                          else 'PASS_SELECTED_BANK_CASE_WORKER_PREFLIGHT_ONLY'
                          if (producer == 'ed561666' or scope == 'eligible-bank') and args.remaining_case
                          else 'PASS_SELECTED_FRESH_WORKER_PREFLIGHT_ONLY' if producer in ('ed561666', 'grid400')
                          else 'PASS_BANK_CASE_WORKER_PREFLIGHT_ONLY' if args.remaining_case
                          else 'PASS_EXPANDED_WORKER_PREFLIGHT_ONLY'), allocation=allocation,
        scientific_build_commit=manifest['commit'], operational_commit=args.commit,
        build_identity_sha256=checks.digest(manifest_path), pilot_sha256=pilot_sha,
        qualification_sha256=checks.digest(args.qualification), runtime_unchanged=True,
        runtime_input_sha256=draft['runtime_input_sha256'], monomer_id=draft['monomer_id'],
        execution_profile=profile,
        task_config=dict(ncores=task_config.ncores, memory_GiB=task_config.memory,
                         internal_retries=task_config.retries),
        science=science, descriptor_verification=result, scratch_snapshot=disk,
        operational_sources_sha256=operations, manager_sources_sha256=manager_pins,
        transport_binding=transport, manager_python=sys.executable,
        discovery_python=shutil.which('python3'), qcengine_version=str(qcengine.__version__),
        qcfractalcompute_version=str(qcfractalcompute.__version__),
        harness_path=str(harness_path), harness_sha256=harness_sha,
        discovered_psi4=programs['psi4'], psi4_cli=shutil.which('psi4'),
        trace_files=[p.name for p in sorted((payload/'traces').glob('*.strace'))],
        qualification='Installed descriptor/request and actual manager discovery only; '
                      'no SCF, molecular calculation, manager acquisition or server writes. '
                      'No independent trace, molecular, RSS or SCF scratch acceptance.')
    if producer in ('ed561666', 'grid400'):
        report.update(producer_profile=producer, dataset_mode='append' if args.remaining_case else 'fresh')
    if qomp:
        report.update(status='PASS_QOMP_FRESH_WORKER_PREFLIGHT_ONLY',producer_profile='qomp',
                      campaign_scope='qomp-pilot',dataset_mode='fresh',
                      producer_binding_sha256=report.pop('build_identity_sha256'))
        return payload/'worker-preflight.json',report
    if scope == 'eligible-bank':
        report['campaign_scope'] = scope
    if scope == 'eligible-wave':
        binding = helper('bank_wave_runtime').helper('bank_wave_campaign').load_binding(
            args, require_registration=False)
        report.update(campaign_scope=scope, dispatch_protocol='registered-wave/1',
            campaign=draft['campaign'], wave_id=draft['wave_id'],
            plan_sha256=binding['plan_sha256'],
            bootstrap_context_sha256=draft['bootstrap_context_sha256'],
            checkpoint_sha256=draft['checkpoint_sha256'],
            predecessor_snapshot_sha256=draft['predecessor_snapshot_sha256'])
    if scope == 'eligible-wave':
        helper('bank_wave_runtime').write_json_once(payload/'worker-preflight.json', report)
    else:
        with (payload/'worker-preflight.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], job=os.environ['SLURM_JOB_ID'])))


if __name__ == '__main__':
    main()
