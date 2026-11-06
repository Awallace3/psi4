"""Fresh cb/request3 QCArchive singleton; delivery is not independent audit."""
import argparse
import base64
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import importlib.util

RUNS = Path('/storage/project/r-cs207-0/awallace43/runs')
SCRATCH = Path('/storage/scratch1/3/awallace43')
BASE = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt')
MANAGER = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca')
HELPERS = {'pilot_cb63305', 'phoenix_worker_cb63305', 'phoenix_installed_checks',
    'phoenix_worker_probe', 'phoenix_pilot', 'dataset_fresh_cb63305',
    'single_record_cb63305', 'descriptor_scratch_cb63305', 'phoenix_descriptor_bank_cb63305',
    'case_cb63305', 'dataset_owned354_cb63305', 'owned_dataset_cb63305', 'grid400_policy',
    'envelope_identity', 'bank_campaign', 'bank_dataset', 'bank_append', 'bank_store',
    'grid400_bank_policy', 'dataset_guard', 'bank_wave_runtime'}
_MODULES = {}


def helper(name):
    if name not in HELPERS:
        raise ValueError('closed first expanded-pilot helpers required')
    if name == 'bank_wave_runtime' and name in _MODULES:
        return _MODULES[name]
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_expanded_bootstrap_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    if name == 'bank_wave_runtime':
        _MODULES[name] = module
    return module


def verify_delivery_envelope(local, server, request, implementation, *, producer):
    """The production post-delivery identity seam, before artifact publication."""
    if producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed delivery producer required')
    if (not isinstance(server, dict) or server.get('schema') != 'native-monomer-artifacts/2'
            or server.get('request') != request
            or server.get('implementation', {}).get('sha256') != implementation):
        raise RuntimeError('local/server request3 envelope or producer identity differs')
    if producer == 'grid400':
        return helper('envelope_identity').verify_native_envelope_identity(local, server)
    if server != local:
        raise RuntimeError('local/server request3 envelope or producer identity differs')
    return None


def admit_allocation(allocation, *, execution_profile=None, producer='cb63305', campaign_scope='default'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed molecular allocation producer required')
    helper('phoenix_worker_cb63305').validate_campaign_scope(producer, campaign_scope)
    expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                    NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='02:00:00')
    if producer == 'grid400':
        profile = helper('grid400_bank_policy' if campaign_scope in ('eligible-bank', 'eligible-wave')
                         else 'grid400_policy').validate_profile(execution_profile)
        expected.update(NumCPUs=str(profile['cores']),
                        MinMemoryNode=str(profile['job_memory_GiB'])+'G',
                        TimeLimit=profile['time_limit'])
    elif execution_profile is not None:
        cases = helper('case_cb63305')
        name = execution_profile.get('name')
        if name not in cases.PROFILES:
            raise ValueError('explicit finite remaining-case molecular profile required')
        profile = dict(name=name, **cases.PROFILES[name], account='gts-cs207-chemx',
                       partition='cpu-small', qos='embers', concurrency=1)
        if cases.json_digest(profile) != cases.json_digest(execution_profile):
            raise ValueError('remaining-case molecular profile changed')
        expected.update(NumCPUs=str(profile['cores']),
                        MinMemoryNode=str(profile['job_memory_GiB'])+'G',
                        TimeLimit=profile['time_limit'])
    if any(allocation.get(k) != v for k, v in expected.items()):
        raise ValueError('exact effective four-hour grid400 molecular embers allocation required'
            if producer == 'grid400' else 'exact effective two-hour expanded molecular embers allocation required')


def validate_preflight(preflight, draft, *, manifest_sha, qualification_sha, pilot_sha,
                       worker_sources, transport, producer='cb63305', remaining_case=False,
                       campaign_scope='default'):
    if (type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400')
            or type(remaining_case) is not bool):
        raise ValueError('closed molecular producer selector required')
    helper('phoenix_worker_cb63305').validate_campaign_scope(producer, campaign_scope)
    profile = draft.get('execution_profile')
    if campaign_scope in ('eligible-bank', 'eligible-wave'):
        policy = helper('grid400_bank_policy')
        for item in (preflight, draft):
            if (item.get('producer_profile') != 'grid400'
                    or item.get('campaign_scope') != campaign_scope
                    or item.get('dataset_mode') != ('append' if remaining_case else 'fresh')):
                raise ValueError('explicit reconstructed eligible-bank worker and molecular mode required')
            policy.validate_profile(item.get('execution_profile'))
        if campaign_scope == 'eligible-wave':
            if not remaining_case:
                raise ValueError('wave is append only')
            for key in ('dispatch_protocol', 'wave_id', 'bootstrap_context_sha256',
                        'checkpoint_sha256', 'predecessor_snapshot_sha256'):
                if preflight.get(key) != draft.get(key) or draft.get(key) is None:
                    raise ValueError('exact wave worker/case authority required')
            if draft['dispatch_protocol'] != 'registered-wave/1':
                raise ValueError('explicit registered-wave protocol required')
    elif producer == 'grid400':
        policy = helper('grid400_policy')
        if (remaining_case or preflight.get('producer_profile') != producer
                or preflight.get('dataset_mode') != 'fresh'
                or draft.get('producer_profile') != producer or draft.get('dataset_mode') != 'fresh'
                or draft.get('monomer_id') != policy.MONOMER):
            raise ValueError('explicit fresh grid400 worker mode required, never append')
        policy.validate_profile(profile)
        policy.validate_profile(preflight.get('execution_profile'))
    status = ('PASS_SELECTED_WAVE_CASE_WORKER_PREFLIGHT_ONLY' if campaign_scope == 'eligible-wave'
              else 'PASS_SELECTED_BANK_CASE_WORKER_PREFLIGHT_ONLY'
              if (producer == 'ed561666' or campaign_scope == 'eligible-bank') and remaining_case
              else 'PASS_SELECTED_FRESH_WORKER_PREFLIGHT_ONLY' if producer in ('ed561666', 'grid400')
              else 'PASS_BANK_CASE_WORKER_PREFLIGHT_ONLY' if profile
              else 'PASS_EXPANDED_WORKER_PREFLIGHT_ONLY')
    if producer == 'ed561666' and (
            preflight.get('producer_profile') != producer
            or preflight.get('dataset_mode') != ('append' if remaining_case else 'fresh')
            or draft.get('producer_profile') != producer
            or draft.get('dataset_mode') != ('append' if remaining_case else 'fresh')
            or not isinstance(profile, dict)
            or not remaining_case and profile.get('name') != 's40'):
        raise ValueError('explicit selected cap32 discovery mode/profile required')
    if producer == 'ed561666' and remaining_case:
        context = helper('owned_dataset_cb63305').context(producer)
        if (draft.get('dataset_id') != context['dataset_id']
                or draft.get('monomer_id') == context['first_monomer']
                or draft.get('required_first_pilot_audit_sha256') != context['accepted_sha256']['audit']):
            raise ValueError('reconstructed owned355 remaining case required')
    if (preflight.get('status') != status
            or preflight.get('runtime_unchanged') is not True
            or preflight.get('monomer_id') != draft['monomer_id']
            or preflight.get('scientific_build_commit') != draft['producer']['scientific_commit']
            or preflight.get('build_identity_sha256') != manifest_sha
            or preflight.get('qualification_sha256') != qualification_sha
            or preflight.get('pilot_sha256') != pilot_sha
            or preflight.get('runtime_input_sha256') != draft['runtime_input_sha256']
            or preflight.get('operational_sources_sha256') != worker_sources
            or preflight.get('transport_binding') != transport
            or preflight.get('science', {}).get('implementation', {}).get('sha256') !=
                draft['producer']['implementation_sha256']
            or preflight.get('descriptor_verification', {}).get('status') != 'PASS_DESCRIPTOR_PLAN_ONLY'):
        raise ValueError('exact expanded installed worker discovery and raw pilot binding required')
    if profile is not None and (
            preflight.get('execution_profile') != profile
            or preflight.get('task_config') != dict(ncores=profile['cores'],
                memory_GiB=profile['science_memory_GiB'], internal_retries=0)):
        raise ValueError('exact remaining-case scientific worker profile required')


def load_append_inputs(args):
    producer = getattr(args, 'producer', 'cb63305')
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed molecular producer selector required')
    scope = helper('phoenix_worker_cb63305').selected_scope(args)
    if scope == 'eligible-wave':
        return helper('bank_wave_runtime').load_binding(args)
    bank_fields = ('bank_context', 'bank_context_sha256', 'bank_checkpoint', 'bank_checkpoint_sha256')
    if scope == 'eligible-bank':
        campaign = helper('bank_campaign')
        campaign.authority()  # Refuse an unsealed pool before any checkpoint reads.
        if any(getattr(args, name, None) is not None for name in (
                'owned_checkpoint', 'checkpoint_sha256', 'checkpoint_proofs',
                'accepted_audit', 'accepted_outcome', 'accepted_pilot', 'accepted_envelope')):
            raise ValueError('historical acceptance/checkpoint inputs cannot enter the new bank campaign')
        if args.remaining_case:
            return campaign.load_checkpoint(args)
        if any(getattr(args, name, None) is not None for name in bank_fields):
            raise ValueError('fresh bank bootstrap cannot ignore prior checkpoint inputs')
        return None
    if any(getattr(args, name, None) is not None for name in bank_fields):
        raise ValueError('bank checkpoint inputs require the explicit bank campaign scope')
    if producer == 'grid400' and args.remaining_case:
        raise ValueError('grid400 is a fresh singleton only, never owned-dataset append')
    selection = {} if producer == 'cb63305' else dict(producer=producer)
    proof_path = getattr(args, 'checkpoint_proofs', None)
    roles = ('audit', 'outcome', 'pilot', 'envelope')
    paths = {role: getattr(args, 'accepted_'+role) for role in roles}
    if not args.remaining_case:
        if args.owned_checkpoint or args.checkpoint_sha256 or any(paths.values()) or proof_path is not None:
            raise ValueError('owned append inputs cannot be ignored in first-case mode')
        return None
    if not args.owned_checkpoint or not args.checkpoint_sha256 or not all(paths.values()):
        raise ValueError('remaining case requires all closed acceptance receipts and checkpoint')
    if producer != 'ed561666' and proof_path is not None:
        raise ValueError('checkpoint advancement proof directories are cap32-only')
    blobs = {role: path.read_bytes() for role, path in paths.items()}
    accepted = helper('owned_dataset_cb63305').verify_first_acceptance(blobs, **selection)
    raw = args.owned_checkpoint.read_bytes()
    append = helper('dataset_owned354_cb63305')
    proofs = (append.read_checkpoint_proofs(raw, args.checkpoint_sha256, proof_path)
              if proof_path is not None else {})
    append.validate_checkpoint(raw, args.checkpoint_sha256, accepted, **selection,
                               **(dict(proof_blobs=proofs) if proofs else {}))
    result = dict(first_receipt_blobs=blobs, checkpoint_raw=raw,
                  checkpoint_sha256=args.checkpoint_sha256)
    if proofs:
        result['checkpoint_proof_blobs'] = proofs
    return result


def validate_server_config(server):
    if (server.get('username') != 'awallace43'
            or server.get('fractal_uri') != 'http://128.61.254.96:7777'):
        raise ValueError('approved authenticated owner and deployment URI required')


def scratch_snapshot(scratch, payload, guard, *, execution_profile=None):
    """Conservative pilot headroom snapshot, not a certified SCF disk bound."""
    import shutil
    if scratch.stat().st_dev != SCRATCH.stat().st_dev:
        raise ValueError('pilot scratch is on another filesystem')
    quotas = {}
    for role, flag, owner in (('user', '-u', os.getuid()), ('group', '-g', scratch.stat().st_gid)):
        result = subprocess.run(['/usr/bin/lfs', 'quota', flag, str(owner), str(SCRATCH)],
                                capture_output=True, text=True, timeout=45, check=True)
        if len(result.stdout) > 65536:
            raise ValueError('bounded unscaled scratch quota required')
        quotas[role] = guard.parse_quota(result.stdout, str(SCRATCH))
        with (payload/('scratch-'+role+'-quota.txt')).open('x') as stream:
            stream.write(result.stdout)
    result = guard.admit_scratch(free_bytes=shutil.disk_usage(scratch).free,
        free_files=os.statvfs(scratch).f_favail, user_quota=quotas['user'], group_quota=quotas['group'],
        required_bytes=execution_profile['scratch_bytes'] if execution_profile else 64*1024**3,
        required_files=execution_profile['scratch_files'] if execution_profile else 65536)
    result['qualification'] = ('Conservative pilot free-space/quota snapshot only; not a reservation, '
                               'certified SCF disk bound or guarantee against concurrent usage.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'qualification', 'inputs', 'descriptors', 'pilot', 'worker-preflight', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666', 'grid400'), default='cb63305')
    parser.add_argument('--remaining-case', action='store_true')
    parser.add_argument('--campaign-scope', choices=('default', 'eligible-bank', 'eligible-wave'), default='default')
    parser.add_argument('--wave-plan', type=Path)
    parser.add_argument('--wave-plan-sha256')
    parser.add_argument('--wave-registration', type=Path)
    for name in ('bank-context', 'bank-checkpoint'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--owned-checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--checkpoint-proofs', type=Path)
    for role in ('audit', 'outcome', 'pilot', 'envelope'):
        parser.add_argument('--accepted-'+role, type=Path)
    args = parser.parse_args()
    append_inputs = load_append_inputs(args)
    producer = args.producer
    checks, worker = (helper(name) for name in (
        'phoenix_installed_checks', 'phoenix_worker_cb63305'))
    scope = worker.selected_scope(args)
    bank_mode = scope == 'eligible-bank'
    wave_mode = scope == 'eligible-wave'
    wave = helper('bank_wave_runtime') if wave_mode else None
    binding = append_inputs if wave_mode else None
    scope_args = dict(campaign_scope=scope) if bank_mode or wave_mode else {}
    pins = (wave.pins() if wave_mode else helper('bank_campaign').pins() if bank_mode
            else helper('pilot_cb63305').producer_pins(producer))
    transport_args = {} if bank_mode or wave_mode or producer == 'cb63305' else dict(producer=producer)
    allocation = checks.allocation()
    source = Path(__file__).resolve().parents[2]
    if (checks.git(source, 'rev-parse', 'HEAD') != args.commit
            or checks.git(source, 'status', '--porcelain')):
        raise ValueError('exact clean expanded bootstrap source required')
    build, run = args.build.resolve(), args.run.resolve()
    if (not build.is_relative_to(RUNS) or not run.is_relative_to(RUNS)
            or not wave_mode and not run.name.startswith('ccasp-native-s66-')
            or build.is_relative_to(run) or run.is_relative_to(build)):
        raise ValueError('disjoint private build and molecular run required')
    if wave_mode:
        wave.validate_paths(args, role='science')
    manifest_path = build/'build-identity.json'
    draft = worker.prepare_draft(args, manifest_path)
    if bank_mode:
        helper('bank_store').prepare_storage()
    profile = draft.get('execution_profile')
    admit_allocation(allocation, execution_profile=profile,
                     **(dict(producer=producer, **scope_args) if producer == 'grid400' else {}))
    pilot_sha = checks.digest(args.pilot)
    manifest = json.loads(manifest_path.read_bytes())
    manifest_sha = checks.digest(manifest_path)
    preflight = json.loads(args.worker_preflight.read_bytes())
    preflight_sha = checks.digest(args.worker_preflight)
    changes = checks.git(source, 'diff', '--name-only', manifest['commit'], args.commit).splitlines()
    if any(not p.startswith('devtools/isapol_s66_campaign/')
           and p not in checks.TEST_ONLY_CHANGES for p in changes):
        raise ValueError('bootstrap operations change scientific/build sources')
    operations = worker.source_hashes(producer, remaining_case=args.remaining_case, **scope_args)
    operations.update({name: checks.digest(Path(__file__).with_name(name)) for name in (
        'phoenix_pilot_cb63305.py', 'phoenix_pilot_cb63305.sbatch', 'single_record_cb63305.py')})
    if args.remaining_case and not bank_mode and not wave_mode:
        operations.update({name: checks.digest(Path(__file__).with_name(name)) for name in (
            'dataset_owned354_cb63305.py', 'owned_dataset_cb63305.py', 'dataset_guard.py')})
    worker_sources = {key: operations[key]
                      for key in worker.source_names(producer, remaining_case=args.remaining_case, **scope_args)}
    legacy = helper('phoenix_pilot')
    manager_sources = MANAGER/'lib/python3.13/site-packages/qcfractalcompute'
    if any(checks.digest(manager_sources/n) != v for n, v in legacy.MANAGER_FILES.items()):
        raise ValueError('reviewed manager source changed')
    rows = subprocess.check_output(['/opt/slurm/current/bin/sacct', '-n', '-P', '-j',
        preflight['allocation']['JobId'], '-o', 'JobIDRaw,State,ExitCode'], text=True).splitlines()
    for identifier in (preflight['allocation']['JobId']+suffix for suffix in ('', '.batch', '.0')):
        if not any(line.strip().split('|')[:3] == [identifier, 'COMPLETED', '0:0'] for line in rows):
            raise ValueError('worker preflight allocation/batch/step must all complete0:0')
    environment = build/'env'
    checks.verify_runtime(environment, manifest)
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    for name in ('traces', 'launcher'):
        (payload/name).mkdir()
    (payload/'launcher/psi4').symlink_to(source/'devtools/isapol_s66_campaign/native_psi4.sh')
    scratch = SCRATCH/(pins['namespace']+draft['monomer_id']+'-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir()
    disk = scratch_snapshot(scratch, payload, helper('descriptor_scratch_cb63305'),
                            execution_profile=profile)
    common = helper('phoenix_descriptor_bank_cb63305')
    env = common.child_environment(environment, scratch)
    worker_shape = (profile['cores'], profile['science_memory_GiB']) if profile else (4, 20)
    env.update(PATH=f'{payload}/launcher:{MANAGER}/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH=str(MANAGER/'lib'), PYTHONPYCACHEPREFIX=str(scratch/'cache'),
        CCASP_NATIVE_ENV=str(environment), CCASP_NATIVE_BASE=str(BASE),
        CCASP_NATIVE_CACHE=str(scratch/'cache'), CCASP_NATIVE_HOME=str(scratch/'home'),
        CCASP_NATIVE_TRACE=str(payload/'traces'), OMP_NUM_THREADS=str(worker_shape[0]),
        MKL_NUM_THREADS=str(worker_shape[0]))
    os.environ.clear()
    os.environ.update(env)
    if Path(sys.executable).resolve() != (MANAGER/'bin/python').resolve():
        raise ValueError('bootstrap must keep the reviewed p4qca manager interpreter')
    os.chdir(scratch/'cwd')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    import qcengine
    import qcfractalcompute
    import yaml
    from qcengine.config import get_config
    from qcfractalcompute.config import FractalComputeConfig
    from qcfractalcompute.apps.app_manager import discover_programs_conda
    worker.validate_versions(str(qcengine.__version__), str(qcfractalcompute.__version__))
    mutation = (wave.helper('bank_wave_transport') if wave_mode else
                helper('bank_dataset' if bank_mode else
                       'dataset_owned354_cb63305' if args.remaining_case else 'dataset_fresh_cb63305'))
    transport = mutation.transport_binding(**transport_args)
    validate_preflight(preflight, draft, manifest_sha=manifest_sha,
        qualification_sha=checks.digest(args.qualification), pilot_sha=pilot_sha,
        worker_sources=worker_sources, transport=transport, producer=producer,
        remaining_case=args.remaining_case, **scope_args)
    harness = qcengine.get_program('psi4')
    harness_path = Path(inspect.getfile(type(harness))).resolve()
    if (str(harness_path) != preflight['harness_path']
            or harness_path != Path('/storage/project/r-cs207-0/awallace43/gits/qcengine/qcengine/programs/psi4.py')
            or checks.digest(harness_path) != preflight['harness_sha256']
            or 'task_config' not in inspect.signature(qcengine.compute).parameters):
        raise ValueError('reviewed QCEngine harness changed')
    task_config = get_config(task_config=dict(memory=float(worker_shape[1]), ncores=worker_shape[0],
                                             scratch_directory=str(scratch)))
    if (task_config.retries != 0 or task_config.ncores != worker_shape[0]
            or task_config.memory != worker_shape[1]):
        raise ValueError('exact scientific worker shape and zero internal QCEngine retries required')
    version = harness.get_version()
    programs = worker.discover_manager_programs(env, discover_programs_conda)
    helper('phoenix_worker_probe').validate_worker_discovery(
        programs, version, preflight['science']['version'], scientific_imported='psi4' in sys.modules)
    runtime = helper('phoenix_worker_probe').runtime_input(draft)
    runtime['specification']['protocols'] = legacy.execution_specification(draft['specification'])['protocols']
    save = lambda name, data: (wave.write_json_once(payload/name, data) if wave_mode
                              else common.write_json_once(payload/name, data))
    save('expected-runtime.json', runtime)
    # Credentials stay on Phoenix in the approved config; never save config.
    server = yaml.safe_load(Path(
        '/storage/project/r-cs207-0/awallace43/qca/resources_losii_ac.yml').read_text())['server']
    validate_server_config(server)
    tag = pins['namespace']+draft['monomer_id']+'-'+os.environ['SLURM_JOB_ID']
    config_data = legacy.pilot_config(server, payload, scratch, tag)
    config_data['executors']['pilot'].update(cores_per_worker=worker_shape[0],
                                            memory_per_worker=float(worker_shape[1]))
    config = FractalComputeConfig(**config_data)
    helper('single_record_cb63305').validate_session_config(config, tag, worker_shape=worker_shape, **scope_args)
    transaction_scope = None
    if bank_mode:
        transaction_scope = mutation.prepare_transaction(draft, worker.prepare_draft(args, manifest_path), tag,
            checkpoint_raw=append_inputs['checkpoint_raw'] if args.remaining_case else None)
    client_type = (mutation.readonly_client_class(binding, monomer_id=binding['monomer_id'])
                   if wave_mode else mutation.client_class(transaction_scope) if bank_mode
                   else mutation.client_class(**transport_args))
    client = client_type(config.server.fractal_uri, username=config.server.username,
        password=config.server.password, verify=config.server.verify, show_motd=False)
    info = client.get_server_information()
    if info.get('name') != 'CDS QCFractal Server' or info.get('version') != '0.70':
        raise ValueError('approved deployment identity changed')
    managers = list(client.query_managers(status=['active'], limit=200))
    if len(managers) == 200 or any('*' in m.compute_tags or tag in m.compute_tags for m in managers):
        raise ValueError('singleton acquisition is not isolated from existing managers')
    started = dict(schema='native-expanded-pilot-outcome/1', status='STARTED_NOT_QUALIFIED',
        allocation=allocation, monomer_id=draft['monomer_id'], compute_tag=tag,
        scientific_build_commit=manifest['commit'], operational_commit=args.commit,
        build_identity_sha256=manifest_sha, pilot_sha256=pilot_sha,
        installed_qualification_sha256=checks.digest(args.qualification),
        worker_preflight_sha256=preflight_sha,
        descriptor_allocation_sha256=draft['descriptor_allocation_sha256'],
        runtime_input_sha256=draft['runtime_input_sha256'],
        operational_sources_sha256=operations, manager_sources=legacy.MANAGER_FILES,
        transport_binding=transport, scratch_path=str(scratch), scratch_snapshot=disk,
        maximum_records=1, find_existing=False, execution_protocols=runtime['specification']['protocols'],
        execution_profile=profile,
        checkpoint_sha256=(append_inputs['checkpoint_sha256'] if (bank_mode or wave_mode) and args.remaining_case
                           else args.checkpoint_sha256 if args.remaining_case else None),
        qcengine_internal_retries=0, dataset_name=draft['dataset_name'],
        specification_name=draft['specification_name'])
    if producer in ('ed561666', 'grid400'):
        started.update(producer_profile=producer, dataset_mode='append' if args.remaining_case else 'fresh')
    if bank_mode:
        started.update(campaign_scope=scope, campaign=draft['campaign'])
    if wave_mode:
        started.update(wave.started_fields(binding), campaign=draft['campaign'])
    save('started.json', started)
    if wave_mode:
        consumed = wave.registered_record(client, binding, draft, allocation, preflight, save)
        dataset, record, created = (consumed[key] for key in ('dataset', 'record', 'created'))
    elif bank_mode and args.remaining_case:
        campaign = helper('bank_campaign')
        pool, raw_inputs = campaign.load_pool(args.inputs, manifest_path, args.qualification, args.descriptors)
        prepared = campaign.prepared_cases(pool, raw_inputs)[draft['monomer_id']]
        created = helper('bank_append').append_once(client, transaction_scope, draft,
            dataset_adapter=mutation, prepared_case=prepared, **append_inputs,
            authenticated_owner=server['username'], save=save)
        dataset = client.get_dataset_by_id(draft['dataset_id'])
        if (dataset.id != draft['dataset_id'] or dataset.name != draft['dataset_name']
                or getattr(dataset, '_client', None) is not client):
            raise ValueError('new audited bank append returned another dataset/client identity')
    elif args.remaining_case:
        created = mutation.append_once(client, case_path=args.pilot, inputs=args.inputs,
            build_identity=manifest_path, qualification=args.qualification, descriptors=args.descriptors,
            **append_inputs, authenticated_owner=server['username'], compute_tag=tag, save=save,
            **transport_args)
        dataset = client.get_dataset_by_id(draft['dataset_id'])
        if (dataset.id != draft['dataset_id'] or dataset.name != draft['dataset_name']
                or getattr(dataset, '_client', None) is not client):
            raise ValueError('owned354 append returned another dataset/client identity')
    else:
        intent = None
        if bank_mode:
            intent = helper('bank_store').reserve_intent(dataset_id=None, checkpoint_sha256=None,
                snapshot_sha256=None, context_sha256=None, case_sha256=transaction_scope.case_sha256,
                eligibility_sha256=draft['eligibility_sha256'], monomer_id=draft['monomer_id'], compute_tag=tag)
            save('dispatch-intent.json', intent)
        dataset, _ = (mutation.register_once(client, transaction_scope, draft, save) if bank_mode
                      else mutation.register_once(client, draft, tag, save, **transport_args))
        if dataset.id in pins['protected_datasets']:
            raise ValueError('protected historical dataset cannot be a fresh pilot target')
        inserted = (mutation.submit_once(client, transaction_scope, dataset, draft, save) if bank_mode
                    else mutation.submit_once(client, dataset, draft, tag, save, **transport_args))
        created = dict(n_inserted=inserted.n_inserted, n_existing=inserted.n_existing)
        if intent is not None:
            created['dispatch_intent_sha256'] = intent['sha256']
    if not wave_mode:
        record = dataset.get_record(draft['entry']['name'], draft['specification_name'], force_refetch=True)
    protected = pins['protected_records']
    if producer == 'ed561666' and args.remaining_case:
        protected += (helper('owned_dataset_cb63305').context(producer)['first_record'],)
    if (record.id in protected
            or args.remaining_case and record.id != created['record_id']):
        raise ValueError('historical record cannot be a fresh expanded-pilot result')
    if not wave_mode:
        save('record-created.json', dict(dataset_id=dataset.id, record_id=record.id,
            molecule_id=record.molecule_id, compute_tag=tag,
            n_inserted=created['n_inserted'], n_existing=created['n_existing']))
    else:
        if created['compute_tag'] != tag:
            raise ValueError('registered wave tag differs from actual science job')
        wave.helper('bank_wave_campaign').revalidate_binding(binding)
    session = helper('single_record_cb63305').run_session(config, record_id=record.id,
        compute_tag=tag, runtime=runtime, payload=payload, save=save, worker_shape=worker_shape,
        **(dict(producer=producer, case=draft, campaign_scope=scope,
                wave_binding=binding, allocation=allocation) if wave_mode else
           dict(producer=producer, case=draft, campaign_scope=scope) if bank_mode else
           dict(transport_args, **(dict(case=draft) if producer == 'ed561666' and args.remaining_case else {}))))
    record = dataset.get_record(draft['entry']['name'], draft['specification_name'], force_refetch=True)
    history = record.compute_history
    save('server-record-outcome.json', dict(record_id=record.id, status=record.status.value,
        history_count=len(history), history_managers=[h.manager_name for h in history]))
    if record.status.value != 'complete' or len(history) != 1 or history[0].manager_name != session['manager']:
        raise RuntimeError('one complete result from the exact owned manager required')
    envelope = record.extras.get('psi4:native_monomer')
    request = runtime['specification']['keywords']['function_kwargs']['native_monomer']
    identity = verify_delivery_envelope(session['local_native'], envelope, request,
        draft['producer']['implementation_sha256'], producer=producer)
    if identity is not None:
        save('envelope-identity.json', identity)
    artifact = envelope['artifacts']['tensors.npz']
    if type(artifact['data']) is not str or len(artifact['data']) > 4*((16*1024**2+2)//3):
        raise RuntimeError('native tensor transport bound exceeded')
    raw = base64.b64decode(artifact['data'], validate=True)
    if len(raw) != artifact['size'] or hashlib.sha256(raw).hexdigest() != artifact['sha256']:
        raise RuntimeError('native tensor byte size/checksum mismatch')
    save('server-native-envelope.json', envelope)
    with (payload/'server-tensors.npz').open('xb') as stream:
        stream.write(raw)
    if wave_mode:
        wave.capture_delivery(client, binding, draft, record_id=record.id,
            manager=session['manager'], envelope=envelope, tensor_raw=raw, save=save)
        wave.helper('bank_wave_campaign').revalidate_binding(binding)
    elif bank_mode:
        base = helper('dataset_guard')
        after_delivery = base.snapshot_dataset(client, dataset_id=dataset.id,
                                               max_entries=draft['eligible_monomer_count'])
        if args.remaining_case:
            base.assert_append_delta(json.loads((payload/'owned-before-entry.json').read_bytes()),
                after_delivery, entry_name=draft['entry']['name'], record_id=record.id,
                specification_name=draft['specification_name'])
            save('owned-after-delivery.json', after_delivery)
        else:
            save('bank-initial-snapshot.json', after_delivery)
    elif args.remaining_case:
        owned = helper('owned_dataset_cb63305')
        after_delivery = owned.snapshot_dataset(client, **transport_args)
        owned.assert_append_delta(json.loads((payload/'owned-before-entry.json').read_bytes()),
            after_delivery, entry_name=draft['entry']['name'], record_id=record.id)
        save('owned-after-delivery.json', after_delivery)
    checks.verify_runtime(environment, manifest)
    if (any(checks.digest(Path(__file__).with_name(n)) != v for n, v in operations.items())
            or any(checks.digest(manager_sources/n) != v for n, v in legacy.MANAGER_FILES.items())
            or checks.digest(harness_path) != preflight['harness_sha256']
            or checks.digest(args.worker_preflight) != preflight_sha
            or checks.digest(args.pilot) != pilot_sha
            or mutation.transport_binding(**transport_args) != transport
            or checks.git(source, 'rev-parse', 'HEAD') != args.commit
            or checks.git(source, 'status', '--porcelain')
            or worker.prepare_draft(args, manifest_path) != draft
            or load_append_inputs(args) != append_inputs):
        raise RuntimeError('runtime, source, manager, transport or raw receipts changed')
    outcome = dict(started, status='DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION',
        dataset_id=dataset.id, record_id=record.id, task_id=session['task_id'], manager=session['manager'],
        runtime_unchanged=True, tensors_sha256=artifact['sha256'], tensor_bytes=len(raw),
        trace_files=[p.name for p in sorted((payload/'traces').glob('*.strace'))],
        qualification='Fresh single-record delivery only; independent numerical/export/OS-trace audit '
                      'is still required. No accuracy, solver, covariance or whole-set acceptance.')
    if bank_mode or wave_mode or producer == 'ed561666' and args.remaining_case:
        outcome['dispatch_intent_sha256'] = created['dispatch_intent_sha256']
    save('pilot-outcome.json', outcome)
    print(json.dumps(dict(status=outcome['status'], dataset_id=dataset.id, record_id=record.id)))


if __name__ == '__main__':
    main()
