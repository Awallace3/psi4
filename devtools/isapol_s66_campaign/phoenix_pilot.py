"""Dedicated one-record QCArchive pilot guards and operational bootstrap.

Only a later retrieved-artifact and runtime audit may qualify the molecular
result. A manager delivery acknowledgment alone is not scientific success.
"""
import copy
import json
import math
import re

_FAILURE_SNAPSHOT = None


def preserve_failure_snapshot():
    """Read-only best effort after an exception; never mask it or roll back."""
    if _FAILURE_SNAPSHOT is None:
        return
    client, guard, save = _FAILURE_SNAPSHOT
    try:
        save('dataset-failure-snapshot.json', guard.snapshot_dataset(client))
    except BaseException as error:
        try:
            save('dataset-failure-snapshot-error.json', dict(error_type=type(error).__name__))
        except BaseException:
            pass


def pilot_config(server, run, scratch, compute_tag):
    """Construct, rather than overlay, the single owned local executor."""
    if (server.get('fractal_uri') != 'http://128.61.254.96:7777'
            or set(server)-{'fractal_uri', 'username', 'password', 'verify'}
            or type(compute_tag) is not str
            or re.fullmatch('[a-z0-9][a-z0-9-]{0,100}', compute_tag) is None):
        raise ValueError('approved deployment and one exact dedicated tag required')
    environments = dict(use_manager_environment=True, conda=[], apptainer=[])
    return dict(
        base_folder=str(run), cluster=compute_tag, logfile=str(run)+'/manager.log',
        parsl_run_dir=str(run)+'/parsl', parsl_usage_tracking=0,
        # Omission uses None without compute0.63's duration pre-validator,
        # which incorrectly calls duration_to_seconds on explicit None.
        update_frequency=10., update_frequency_jitter=0.,
        server=copy.deepcopy(server), environments=copy.deepcopy(environments),
        executors=dict(pilot=dict(
            type='local', max_workers=1, compute_tags=[compute_tag],
            cores_per_worker=4, memory_per_worker=20.,
            scratch_directory=str(scratch), environments=copy.deepcopy(environments))))


def execution_specification(draft):
    """Tighten failure behavior only: never let a harness alter the model."""
    if draft.get('protocols'):
        raise ValueError('unexpected preexisting execution protocols')
    result = copy.deepcopy(draft)
    result['protocols'] = dict(error_correction=dict(default_policy=False))
    return result


def validate_claim(task, runtime, *, record_id, compute_tag):
    """Inspect the actual claimed QCEngine input before its app is submitted."""
    if (task.get('record_id') != record_id or task.get('compute_tag') != compute_tag
            or task.get('function') != 'qcengine.compute'
            or not {'psi4'} <= set(task.get('required_programs', ())) <= {'psi4', 'qcengine'}):
        raise ValueError('unexpected claimed dispatcher, record or compute tag')
    kwargs = task['function_kwargs']
    if set(kwargs) != {'input_data', 'program'} or kwargs['program'] != 'psi4':
        raise ValueError('claimed task is not the sole native Psi4 calculation')
    data = kwargs['input_data']
    if data.get('schema_name') != 'qcschema_input' or data.get('schema_version') != 1:
        raise ValueError('deployed compute0.63 pilot requires its inspected QCSchema1 task')
    specification = data
    expected = runtime['specification']
    if (specification.get('driver') != expected['driver']
            or specification.get('model') != expected['model']
            or specification.get('keywords') != expected['keywords']):
        raise ValueError('claimed scientific request differs from the immutable pilot')
    extras = specification.get('extras', {})
    if extras.get('psiapi') or extras.get('extra_infiles'):
        raise ValueError('claimed task changes the CLI-only or raw-input boundary')
    if specification.get('protocols', {}) != expected['protocols']:
        raise ValueError('claimed task changes the explicit no-correction protocols')
    molecule, raw = data['molecule'], runtime['molecule']
    if (molecule.get('symbols') != raw['symbols']
            or molecule.get('molecular_charge', 0.) != 0.
            or molecule.get('molecular_multiplicity', 1) != 1
            or molecule.get('fix_com') is not True or molecule.get('fix_orientation') is not True):
        raise ValueError('claimed molecular state/order differs from the raw pilot')
    geometry = molecule['geometry']
    # MessagePack may carry flat or nested plain arrays. Neither changes order.
    flat = [x for xyz in geometry for x in xyz] if geometry and isinstance(geometry[0], list) else geometry
    if (len(flat) != len(raw['geometry'])
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   or abs(x-y) > 1e-10 for x, y in zip(flat, raw['geometry']))):
        raise ValueError('claimed molecular coordinates differ from the raw pilot')
    if 'sapt_reference' in json.dumps(kwargs, allow_nan=False):
        raise ValueError('dataset reference attributes entered the executable task')


# Exact remotely inspected compute0.63 private-API boundary. These are ordinary
# manager sources, not scientific response/partition/model inputs.
MANAGER_FILES = {
    'compute_manager.py': '713110118195deaa9383c72de211b8b94c8041f42026b1ca7d976851d3203e7a',
    'config.py': '3c671b10054d91872d111c8ada5283746f5d33ba8ee959a1242500e281cc8f53',
    'executors.py': 'f344cbf886375c86dea69e11b55669792d3ea8b5fe19cc00d2b7772c7a11d77b',
    'apps/qcengine.py': '56f73eb48204e2f81272848fc4b04d19fa799769900239d0dca7d4fbb80e4196',
    'apps/app_manager.py': 'c6f95cb5228b0add9f180dafb3ae47f8afb5e438a42d3f8aa606e1172f8da8f6',
    'apps/helpers.py': '8b9a4414c221b863f89f82fda7d2e606757d46f2374bbd6331e3546566f29569',
    'run_scripts/qcengine_compute.py': '164ff06fd7ead53656414cd8abcc8e74616c208e3c13b190a29e8df3173713db',
    'run_scripts/qcengine_list.py': 'e5194cf24a5838edc0efe04cff2351e089256987a17e383ca382dd22ade17fe1',
}


def main():
    import argparse
    import base64
    import hashlib
    import importlib.util
    import inspect
    import logging
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'qualification', 'worker-preflight', 'pilot', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    append_spec = importlib.util.spec_from_file_location(
        'append_receipts', Path(__file__).with_name('append_receipts.py'))
    append = importlib.util.module_from_spec(append_spec)
    append_spec.loader.exec_module(append)
    append.add_arguments(parser)
    args = parser.parse_args()
    checkpoint = append.verify(args)
    spec = importlib.util.spec_from_file_location(
        'worker_probe', Path(__file__).with_name('phoenix_worker_probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    helper = probe.load_helper('phoenix_installed_checks')
    allocation = helper.allocation()
    if allocation['NumCPUs'] != '4' or allocation['MinMemoryNode'] != '24G':
        raise RuntimeError('pilot requires its declared four-core24GiB embers shape')
    source = Path(__file__).resolve().parents[2]
    if helper.git(source, 'rev-parse', 'HEAD') != args.commit or helper.git(source, 'status', '--porcelain'):
        raise RuntimeError('exact clean operational source required')
    prefix = Path('/storage/project/r-cs207-0/awallace43/runs')
    build, run = args.build.resolve(), args.run.resolve()
    if (not build.is_relative_to(prefix) or not run.is_relative_to(prefix)
            or not run.name.startswith('ccasp-native-s66-')
            or build.is_relative_to(run) or run.is_relative_to(build)):
        raise RuntimeError('unexpected pilot paths')
    manifest_path = build/'build-identity.json'
    manifest = json.loads(manifest_path.read_text())
    qualification = json.loads(args.qualification.read_text())
    preflight = json.loads(args.worker_preflight.read_text())
    draft = json.loads(args.pilot.read_text())
    manifest_hash = helper.digest(manifest_path)
    if (qualification['status'] != 'passed' or qualification['runtime_unchanged'] is not True
            or qualification['build_identity_sha256'] != manifest_hash
            or qualification['build_commit'] != manifest['commit']
            or preflight['status'] != 'PASS_WORKER_PREFLIGHT_ONLY'
            or preflight['runtime_unchanged'] is not True
            or preflight['build_identity_sha256'] != manifest_hash
            or preflight['qualification_sha256'] != helper.digest(args.qualification)
            or preflight['pilot_sha256'] != helper.digest(args.pilot)
            or draft['status'] != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
            or (checkpoint is None and draft['monomer_id'] != 'monomer-0001')
            or (checkpoint is not None and (
                preflight.get('append_checkpoint') != checkpoint
                or preflight['operational_commit'] != args.commit))):
        raise RuntimeError('exact installed qualification, worker preflight and pilot checkpoint required')
    helper.junit_summary(args.qualification.with_name('junit.xml'), expected=197)
    changes = helper.git(source, 'diff', '--name-only', manifest['commit'], args.commit).splitlines()
    if any(not p.startswith('devtools/isapol_s66_campaign/') and p not in helper.TEST_ONLY_CHANGES
           for p in changes):
        raise RuntimeError('operational revision changes scientific/build code')
    rows = subprocess.check_output([
        '/opt/slurm/current/bin/sacct', '-n', '-P', '-j', preflight['allocation']['JobId'],
        '-o', 'JobIDRaw,State,ExitCode'], text=True).splitlines()
    for job in (preflight['allocation']['JobId'], preflight['allocation']['JobId']+'.batch'):
        if not any(line.strip().split('|')[:3] == [job, 'COMPLETED', '0:0'] for line in rows):
            raise RuntimeError('worker preflight allocation/batch did not both complete0:0')
    manager_env = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca')
    manager_sources = manager_env/'lib/python3.13/site-packages/qcfractalcompute'
    for name, expected in MANAGER_FILES.items():
        if helper.digest(manager_sources/name) != expected:
            raise RuntimeError('deployed manager source changed: '+name)
    if helper.digest(preflight['harness_path']) != preflight['harness_sha256']:
        raise RuntimeError('deployed QCEngine harness changed since worker preflight')
    environment = build/'env'
    helper.verify_runtime(environment, manifest)
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    traces = payload/'traces'
    traces.mkdir()
    launcher = payload/'launcher'
    launcher.mkdir()
    (launcher/'psi4').symlink_to(source/'devtools/isapol_s66_campaign/native_psi4.sh')
    monomer = draft['monomer_id']
    scratch_label = 'water' if checkpoint is None else monomer
    scratch = Path('/storage/scratch1/3/awallace43')/('ccasp-native-'+scratch_label+'-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(exist_ok=False)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir()
    for key in list(os.environ):
        if key in ('PYTHONPATH', 'PYTHONHOME', 'PSIPATH', 'PSIDATADIR', 'LD_PRELOAD') or key.startswith('QCF_COMPUTE_'):
            os.environ.pop(key)
    os.environ.update(
        PATH=f'{launcher}:{manager_env}/bin:/usr/bin:/bin', LD_LIBRARY_PATH=str(manager_env/'lib'),
        HOME=str(scratch/'home'), TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(scratch/'cache'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        CCASP_NATIVE_ENV=str(environment),
        CCASP_NATIVE_BASE='/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt',
        CCASP_NATIVE_CACHE=str(scratch/'cache'), CCASP_NATIVE_HOME=str(scratch/'home'),
        CCASP_NATIVE_TRACE=str(traces))
    if Path(sys.executable).resolve() != (manager_env/'bin/python').resolve():
        raise RuntimeError('manager must use its declared p4qca interpreter')
    os.chdir(scratch/'cwd')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    import qcengine
    import qcfractalcompute
    import yaml
    from qcfractalcompute.apps.app_manager import discover_programs_conda
    from qcfractalcompute.compute_manager import ComputeManager
    from qcfractalcompute.config import FractalComputeConfig
    from qcengine.config import get_config
    from qcportal import PortalClient
    from qcportal.singlepoint import QCSpecification, SinglepointDatasetNewEntry
    if (str(qcengine.__version__) != preflight['qcengine_version']
            or str(qcfractalcompute.__version__) != '0.63'
            or 'task_config' not in inspect.signature(qcengine.compute).parameters):
        raise RuntimeError('deployed compute/QCEngine API changed')
    task_config = get_config(task_config=dict(memory=20., ncores=4, scratch_directory=str(scratch)))
    if task_config.retries != 0:
        raise RuntimeError('pilot requires zero internal QCEngine retries, not an ambient retry policy')
    programs = discover_programs_conda(None)['qcengine.compute']['programs']
    probe.validate_worker_discovery(
        programs, qcengine.get_program('psi4').get_version(), preflight['science']['version'],
        scientific_imported='psi4' in sys.modules)
    runtime = probe.runtime_input(draft)
    execution_spec = execution_specification(draft['specification'])
    runtime['specification']['protocols'] = execution_spec['protocols']
    (payload/'expected-runtime.json').write_text(json.dumps(runtime, allow_nan=False))
    # Read credentials in place; never serialize the config into run artifacts.
    config_path = Path('/storage/project/r-cs207-0/awallace43/qca/resources_losii_ac.yml')
    server = yaml.safe_load(config_path.read_text())['server']
    tag = 'ccasp-native-s66-'+scratch_label+'-'+os.environ['SLURM_JOB_ID']
    config = FractalComputeConfig(**pilot_config(server, payload, scratch, tag))
    mutation = append.helper('dataset_append') if checkpoint is not None else None
    transport_binding = mutation.transport_binding() if mutation is not None else None
    if checkpoint is not None and preflight.get('append_transport_binding') != transport_binding:
        raise RuntimeError('append transport changed since bound worker preflight')
    client_type = mutation.client_class() if mutation is not None else PortalClient
    client = client_type(config.server.fractal_uri, username=config.server.username,
                         password=config.server.password, verify=config.server.verify,
                         show_motd=False)
    info = client.get_server_information()
    if info['name'] != 'CDS QCFractal Server' or info['version'] != '0.70':
        raise RuntimeError('unexpected approved deployment identity')
    managers = list(client.query_managers(status=['active'], limit=200))
    if len(managers) == 200 or any('*' in m.compute_tags or tag in m.compute_tags for m in managers):
        raise RuntimeError('pilot acquisition is not isolated from existing managers')

    def save(name, data):
        with (payload/name).open('x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)

    started = dict(
        status='STARTED_NOT_QUALIFIED', allocation=allocation, compute_tag=tag,
        scientific_build_commit=manifest['commit'], operational_commit=args.commit,
        build_identity_sha256=manifest_hash, pilot_sha256=helper.digest(args.pilot),
        installed_qualification_sha256=helper.digest(args.qualification),
        worker_preflight_sha256=helper.digest(args.worker_preflight),
        monomer_id=monomer, scratch_path=str(scratch), append_checkpoint=checkpoint,
        append_transport_binding=transport_binding,
        manager_sources=MANAGER_FILES, maximum_records=1, find_existing=False,
        execution_protocols=execution_spec['protocols'], qcengine_internal_retries=0)
    save('started.json', started)
    dataset_name = 'S66x8-native-CCASP-'+manifest['commit'][:10]+'-v1'
    entry = SinglepointDatasetNewEntry(**draft['entry'])
    specification_name = 'pbe0-atz-native-request2'
    before = None
    if checkpoint is not None:
        guard = append.helper('dataset_guard')
        baseline, _ = append.load(args.append_baseline)
        before = guard.snapshot_dataset(client)
        if before != baseline['snapshot'] or server['username'] != 'awallace43':
            raise RuntimeError('owned server baseline changed before first append')
        guard.validate_owned(before, authenticated_owner=server['username'],
                             implementation=preflight['science']['implementation']['sha256'],
                             specification=QCSpecification(**execution_spec),
                             water_envelope_sha256=baseline['water_envelope_canonical_sha256'])
        guard.validate_new_entry(before, draft)
        save('dataset-before.json', before)
        global _FAILURE_SNAPSHOT
        _FAILURE_SNAPSHOT = (client, guard, save)
        save('entry-insert-attempt.json', dict(dataset_id=353, entry=entry.name, transport_retries=False))
        mutation.insert_entry_once(client, draft['entry'])
        after_entry = guard.snapshot_dataset(client)
        guard.assert_append_delta(before, after_entry, entry_name=entry.name)
        guard.validate_stored_entry(after_entry['entries'][entry.name], draft['entry'])
        save('dataset-after-entry.json', after_entry)
        dataset = client.get_dataset_by_id(353)
    else:
        save('dataset-create-attempt.json', dict(name=dataset_name, tag=tag, time=time.time()))
        dataset = client.add_dataset(
            'singlepoint', dataset_name, existing_ok=False, default_tag=tag,
            description='Fresh native PBE0/aug-cc-pVTZ experimental independent-site monomers. '
                        'First water pilot only; rank2 C10 partial(2,2); no accuracy/parity claim.',
            provenance=dict(scientific_build_commit=manifest['commit'],
                            implementation_sha256=preflight['science']['implementation']['sha256']))
        save('dataset-created.json', dict(dataset_id=dataset.id, name=dataset_name))
        if not dataset.add_entries([entry]).success:
            raise RuntimeError('pilot entry insertion failed')
        if not dataset.add_specification(specification_name, QCSpecification(**execution_spec)).success:
            raise RuntimeError('immutable pilot specification insertion failed')
        stored_entry = dataset.get_entry(entry.name, force_refetch=True)
        if (stored_entry.attributes != entry.attributes
                or stored_entry.additional_keywords != entry.additional_keywords
                or dataset.get_record(entry.name, specification_name) is not None):
            raise RuntimeError('stored entry changed or already has a computational record')
    save('record-submit-attempt.json', dict(
        dataset_id=dataset.id, entry=entry.name, specification=specification_name,
        compute_tag=tag, find_existing=False))
    if checkpoint is not None:
        if guard.snapshot_dataset(client) != after_entry:
            raise RuntimeError('owned dataset changed immediately before singleton submission')
        inserted = mutation.submit_record_once(client, entry_name=entry.name, compute_tag=tag)
    else:
        inserted = dataset.submit(entry_names=[entry.name], specification_names=[specification_name],
                                  compute_tag=tag, find_existing=False)
    if not inserted.success or inserted.n_inserted != 1 or inserted.n_existing != 0:
        raise RuntimeError('pilot did not create exactly one fresh record')
    record = dataset.get_record(entry.name, specification_name, force_refetch=True)
    record_id = record.id
    if checkpoint is not None:
        after_submit = guard.snapshot_dataset(client)
        guard.assert_append_delta(before, after_submit, entry_name=entry.name, record_id=record_id)
        guard.validate_stored_entry(after_submit['entries'][entry.name], draft['entry'])
        guard.validate_fresh_record(after_submit['records'][str(record_id)], tag=tag,
                                    specification=QCSpecification(**execution_spec), entry=draft['entry'])
        save('dataset-after-submit.json', after_submit)
    save('record-created.json', dict(
        dataset_id=dataset.id, record_id=record_id, molecule_id=record.molecule_id,
        compute_tag=tag, n_inserted=inserted.n_inserted, n_existing=inserted.n_existing))
    mixin = probe.load_helper('single_claim').SingleClaimMixin

    class PilotManager(mixin, ComputeManager):
        local_native = None

        def preprocess_new_tasks(self, tasks):
            task = tasks[0]
            values = dict(record_id=task.record_id, compute_tag=task.compute_tag,
                          function=task.function, required_programs=task.required_programs,
                          function_kwargs=task.function_kwargs)
            validate_claim(values, runtime, record_id=record_id, compute_tag=tag)
            save('claimed-input.json', values)
            save('claim.json', dict(task_id=task.id, record_id=task.record_id, manager=self.name))

        def postprocess_results(self, results):
            for task_id, result in results.items():
                with (payload/('local-result-'+str(task_id)+'.zstd')).open('xb') as stream:
                    stream.write(result.result_compressed)
                self.local_native = result.result.get('extras', {}).get('psi4:native_monomer')

    manager = PilotManager(config, target_record_id=record_id, compute_tag=tag, acquisition_seconds=300)
    save('manager-started.json', dict(
        name=manager.name, tag=tag, executor='local', max_workers=1, cores=4, memory_gib=20))
    interrupted = False

    def on_term(signum, frame):
        nonlocal interrupted
        interrupted = True
        manager.acquisition_closed = True
        manager.stop()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        manager.start()
    finally:
        # start() normally deactivates and clears this kernel itself. If a
        # callback raises, still deactivate only this owned manager and clean
        # its executor. Never reset/cancel records or touch another manager.
        if manager.dflow_kernel is not None:
            try:
                manager.client.deactivate(
                    active_tasks=manager.statistics.active_tasks,
                    active_cores=manager.statistics.active_cores,
                    active_memory=manager.statistics.active_memory,
                    total_cpu_hours=manager.statistics.total_cpu_hours)
            finally:
                manager.dflow_kernel.cleanup()
        save('manager-delivery.json', dict(
            manager=manager.name, record_id=record_id, task_id=manager.target_task_id,
            outcome=manager.delivery_outcome, interrupted=interrupted,
            claim_error_type=manager.claim_error_type,
            submission_error_type=manager.submission_error_type,
            local_statistics=manager.statistics.dict()))
    if interrupted or manager.delivery_outcome != 'accepted':
        raise RuntimeError('pilot result was not normally acknowledged; inspect delivery evidence')
    record = dataset.get_record(entry.name, specification_name, force_refetch=True)
    status = record.status.value
    history = record.compute_history
    save('server-record-outcome.json', dict(
        record_id=record_id, status=status, history_count=len(history),
        history_managers=[h.manager_name for h in history]))
    if status != 'complete' or len(history) != 1 or history[0].manager_name != manager.name:
        raise RuntimeError('server did not retain one complete result from this owned manager')
    envelope = record.extras.get('psi4:native_monomer')
    if (not isinstance(envelope, dict) or envelope != manager.local_native
            or envelope.get('schema') != 'native-monomer-artifacts/2'
            or envelope.get('request') != runtime['specification']['keywords']['function_kwargs']['native_monomer']):
        raise RuntimeError('server/local native envelope or immutable request differs')
    artifact = envelope['artifacts']['tensors.npz']
    if type(artifact['data']) is not str or len(artifact['data']) > 4*((16*1024**2+2)//3):
        raise RuntimeError('retrieved tensor transport exceeds its native bound')
    raw = base64.b64decode(artifact['data'], validate=True)
    if len(raw) != artifact['size'] or hashlib.sha256(raw).hexdigest() != artifact['sha256']:
        raise RuntimeError('server tensor checksum/size differs')
    save('server-native-envelope.json', envelope)
    with (payload/'server-tensors.npz').open('xb') as stream:
        stream.write(raw)
    if checkpoint is not None:
        after = guard.snapshot_dataset(client)
        guard.assert_append_delta(before, after, entry_name=entry.name, record_id=record_id)
        guard.validate_stored_entry(after['entries'][entry.name], draft['entry'])
        save('dataset-after-delivery.json', after)
        save('dataset-preservation.json', dict(
            status='PASS_EXISTING_DATASET_PRESERVED', dataset_id=353,
            before_sha256=guard.digest(before), after_sha256=guard.digest(after),
            added_entry=entry.name, added_record=record_id, previous_record_ids=sorted(before['records']),
            qualification='Fresh read-only snapshots, not an atomic server lock.'))
    helper.verify_runtime(environment, manifest)
    if (helper.digest(manifest_path) != manifest_hash
            or helper.git(source, 'rev-parse', 'HEAD') != args.commit
            or helper.git(source, 'status', '--porcelain')
            or any(helper.digest(manager_sources/n) != v for n, v in MANAGER_FILES.items())):
        raise RuntimeError('scientific/runtime or operational manager source changed')
    report = dict(started, status='DELIVERED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION',
                  dataset_id=dataset.id, record_id=record_id, task_id=manager.target_task_id,
                  manager=manager.name, server_status=status, runtime_unchanged=True,
                  tensors_sha256=artifact['sha256'], tensor_bytes=len(raw),
                  trace_files=[p.name for p in sorted(traces.glob('*.strace'))],
                  qualification='Fresh QCArchive result/checksum delivery only; '
                  'independent tensor/scientific/OS-trace qualification remains.')
    save('pilot-outcome.json', report)
    _FAILURE_SNAPSHOT = None
    print(json.dumps(dict(status=report['status'], record_id=record_id,
                          dataset_id=dataset.id, report=str(payload/'pilot-outcome.json'))))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        preserve_failure_snapshot()
        raise
