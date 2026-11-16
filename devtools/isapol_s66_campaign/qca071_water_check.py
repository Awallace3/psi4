"""Actual S66 water-dimer fragment/direct-vs-QCA compatibility check.

Separate diagnostic dataset only. This cannot select/advance the production
bank, change its specification, reset a record, or retry a calculation.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace as NS

P = Path('/storage/project/r-cs207-0/awallace43')
SOURCE = P/'gits/psi4.ccasp-native-qomp-production148-v4'
RUN = P/'runs/ccasp-native-s66-qca071-water-check001'
MANAGER = P/'miniconda/envs/p4qca'
SCRATCH = Path('/storage/scratch1/3/awallace43')
BOOTSTRAP_SHA256 = '091f87bbea8562787b476122df602991d6b2e9fbec23e1190876b6cbe14ae11a'
_MODULES = {}


def helper(name):
    if name not in ('qomp_execution', 'qomp_campaign', 'qca071_bridge',
                    'phoenix_installed_checks', 'phoenix_descriptor_bank_cb63305',
                    'phoenix_pilot', 'phoenix_pilot_cb63305', 'single_claim',
                    'validate_numeric', 'qomp_bank_policy', 'dataset_fresh_cb63305',
                    'phoenix_worker_cb63305'):
        raise ValueError('closed water-check helper required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_qca071_check_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def extract(envelope, directory):
    artifact = envelope['artifacts']['tensors.npz']
    raw = base64.b64decode(artifact['data'], validate=True)
    if (len(raw) > 16*1024**2 or len(raw) != artifact['size']
            or hashlib.sha256(raw).hexdigest() != artifact['sha256']):
        raise ValueError('bounded exact native tensor transport required')
    write(directory/'server-native-envelope.json', envelope)
    with (directory/'server-tensors.npz').open('xb') as stream:
        stream.write(raw)
    report = helper('validate_numeric').validate_numeric(envelope, raw)
    write(directory/'numeric-report.json', report)
    return raw


def environment(directory, mid, mode, profile, commit):
    q = helper('qomp_execution')
    c = helper('qomp_campaign')
    directory.mkdir(mode=0o700)
    for name in ('traces', 'launcher'):
        (directory/name).mkdir(mode=0o700)
    (directory/'launcher/psi4').symlink_to(
        SOURCE/'devtools/isapol_s66_campaign/qomp_launcher.sh')
    scratch = SCRATCH/f'ccasp-native-qca071-{os.environ["SLURM_JOB_ID"]}-{mid}-{mode}'
    scratch.mkdir(mode=0o700)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir(mode=0o700)
    admission = q.scratch_admission(scratch, profile)
    write(directory/'scratch-admission.json', admission)
    env = helper('phoenix_descriptor_bank_cb63305').child_environment(c.BUILD/'env', scratch)
    env.update(PATH=f'{directory}/launcher:{MANAGER}/bin:/usr/bin:/bin',
               LD_LIBRARY_PATH=str(MANAGER/'lib'),
               PYTHONPYCACHEPREFIX=str(scratch/'cache'),
               CCASP_NATIVE_ENV=str(c.BUILD/'env'),
               CCASP_NATIVE_BASE=str(q.qualified().runtime().BASE),
               CCASP_NATIVE_CACHE=str(scratch/'cache'),
               CCASP_NATIVE_HOME=str(scratch/'home'),
               CCASP_NATIVE_TRACE=str(directory/'traces'),
               OMP_NUM_THREADS=str(profile['cores']),
               MKL_NUM_THREADS=str(profile['cores']), OPENBLAS_NUM_THREADS='1')
    env.update(q.launch_binding(NS(commit=commit),
                               dict(execution_profile=profile), directory, scratch, 'science'))
    os.environ.clear()
    os.environ.update(env)
    os.chdir(scratch/'cwd')
    return scratch


def manager_session(config, record_id, tag, runtime, directory):
    from qcfractalcompute.compute_manager import ComputeManager
    mixin = helper('single_claim').SingleClaimMixin

    class WaterManager(mixin, ComputeManager):
        local_native = None

        def preprocess_new_tasks(self, tasks):
            task = tasks[0]
            claimed = dict(record_id=task.record_id, compute_tag=task.compute_tag,
                           function=task.function, required_programs=task.required_programs,
                           function_kwargs=task.function_kwargs)
            helper('phoenix_pilot').validate_claim(
                claimed, runtime, record_id=record_id, compute_tag=tag)
            write(directory/'claimed-input.json', claimed)

        def postprocess_results(self, results):
            for task_id, result in results.items():
                with (directory/f'local-result-{task_id}.zstd').open('xb') as stream:
                    stream.write(result.result_compressed)
                extras = result.result.get('extras')
                self.local_native = (extras.get('psi4:native_monomer')
                                     if isinstance(extras, dict) else None)

    with helper('dataset_fresh_cb63305').grid400_manager_client_factory() as client_type:
        manager = WaterManager(config, target_record_id=record_id, compute_tag=tag,
                               acquisition_seconds=300)
        if type(manager.client) is not client_type:
            raise ValueError('diagnostic manager must use existing one-shot transport')
    write(directory/'manager-started.json', dict(name=manager.name, tag=tag,
                                                record_id=record_id))
    old = {}

    def stop(number, frame):
        manager.acquisition_closed = True
        manager.stop()
        raise KeyboardInterrupt('water-check signal '+str(number))

    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, stop)
    try:
        manager.start()
    finally:
        try:
            if manager.dflow_kernel is not None:
                try:
                    manager.client.deactivate(
                        active_tasks=manager.statistics.active_tasks,
                        active_cores=manager.statistics.active_cores,
                        active_memory=manager.statistics.active_memory,
                        total_cpu_hours=manager.statistics.total_cpu_hours)
                finally:
                    manager.dflow_kernel.cleanup()
            write(directory/'manager-delivery.json',
                  dict(name=manager.name, outcome=manager.delivery_outcome,
                       target_task_id=manager.target_task_id,
                       claim_error_type=manager.claim_error_type,
                       submission_error_type=manager.submission_error_type))
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
    if manager.delivery_outcome != 'accepted' or manager.local_native is None:
        raise RuntimeError('water-check owned result was not accepted; no retry')
    return manager.name, manager.local_native


def compare(direct, transported, directory):
    import numpy as np
    validator = helper('validate_numeric')
    a = validator.tensor_arrays(extract(direct, directory/'direct'), 3)
    b = validator.tensor_arrays(extract(transported, directory/'qca'), 3)
    differences = {}
    for name in ('local', 'refined'):
        x, y = a[name], b[name]
        differences[name] = dict(max_abs=float(np.max(np.abs(x-y))),
                                 rms=float(np.sqrt(np.mean((x-y)**2))),
                                 equal=bool(np.array_equal(x, y)))
    ea, eb = direct['result'], transported['result']
    energy = abs(ea['energy_Eh']-eb['energy_Eh'])
    cn = {}
    for n in (6, 8, 10):
        left = [p['value'] for p in ea['isotropic_pairs'] if p['order'] == n]
        right = [p['value'] for p in eb['isotropic_pairs'] if p['order'] == n]
        cn[str(n)] = dict(max_abs=max(abs(x-y) for x, y in zip(left, right)),
                         max_relative=max(abs(x-y)/max(abs(x), abs(y), 1e-30)
                                          for x, y in zip(left, right)))
    if ea['frequencies'] != eb['frequencies'] or ea['cp_weights'] != eb['cp_weights']:
        raise ValueError('comparison frequency grid changed')
    # Cross-run recomputation is not bitwise deterministic (see the existing
    # water performance report). These diagnostic comparison limits do NOT
    # replace or relax validate_numeric's existing transport/residual gates.
    passed = (energy <= 1e-10
              and all(np.allclose(a[k], b[k], atol=1e-6, rtol=1e-6)
                      for k in ('local', 'refined'))
              and all(v['max_relative'] <= 1e-6 for v in cn.values()))
    report = dict(status='PASS_DIRECT_VS_QCA071' if passed else 'FAIL_DIRECT_VS_QCA071',
                  energy_absolute_difference_Eh=energy, tensors=differences,
                  isotropic_coefficients=cn,
                  comparison_limits=dict(energy_abs_Eh=1e-10, tensor_atol=1e-6,
                                         tensor_rtol=1e-6, cn_rtol=1e-6),
                  qualification='Two independent native recomputations; exact local/server '
                                'transport is checked separately. Not solver certification.')
    write(directory/'comparison.json', report)
    if not passed:
        raise ValueError('water direct/QCA numerical comparison failed')
    return report


def compare_dimer(rows, envelopes):
    """Compare undamped isotropic cross-fragment tails, not a SAPT total."""
    import numpy as np
    values = {}
    validator = helper('validate_numeric')
    for mode in ('direct', 'qca'):
        arrays = []
        for pair in envelopes:
            envelope = pair[mode]
            raw = base64.b64decode(envelope['artifacts']['tensors.npz']['data'], validate=True)
            refined = validator.tensor_arrays(raw, 3)['refined']
            arrays.append((np.trace(refined[:, :, 1:4, 1:4], axis1=2, axis2=3)/3.,
                           np.trace(refined[:, :, 4:9, 4:9], axis1=2, axis2=3)/5.))
        weights = np.asarray(envelopes[0][mode]['result']['cp_weights'])
        if envelopes[0][mode]['result']['cp_weights'] != envelopes[1][mode]['result']['cp_weights']:
            raise ValueError('water fragments must share the CP quadrature')
        xyz = [np.asarray(row['runtime_input']['molecule']['geometry']).reshape(3, 3)
               for row in rows]
        pairs, energy = [], {str(n): 0. for n in (6, 8, 10)}
        a1, a2 = arrays[0]
        b1, b2 = arrays[1]
        for i in range(3):
            for j in range(3):
                distance = float(np.linalg.norm(xyz[0][i]-xyz[1][j]))
                if distance <= 0:
                    raise ValueError('distinct original water-dimer sites required')
                for n, integrand in (
                        (6, 6*a1[:, i]*b1[:, j]),
                        (8, 15*(a1[:, i]*b2[:, j]+a2[:, i]*b1[:, j])),
                        (10, 70*a2[:, i]*b2[:, j])):
                    cn = math.fsum((weights*integrand).tolist())
                    pairs.append(dict(site_a=i, site_b=j, order=n, value=cn))
                    energy[str(n)] -= cn/distance**n
        values[mode] = dict(cross_fragment_coefficients=pairs,
                            undamped_isotropic_rank2_tail_Eh=energy)
    delta = {}
    for n in (6, 8, 10):
        a = [x['value'] for x in values['direct']['cross_fragment_coefficients'] if x['order'] == n]
        b = [x['value'] for x in values['qca']['cross_fragment_coefficients'] if x['order'] == n]
        delta[str(n)] = dict(max_abs=max(abs(x-y) for x, y in zip(a, b)),
                            max_relative=max(abs(x-y)/max(abs(x), abs(y), 1e-30)
                                             for x, y in zip(a, b)))
    if any(d['max_relative'] > 1e-6 for d in delta.values()):
        raise ValueError('water-dimer cross-fragment comparison failed')
    return dict(status='PASS_WATER_DIMER_ISOTROPIC_COMPARISON',
                direct=values['direct'], qca=values['qca'], differences=delta,
                qualification='Undamped isotropic rank2 tails only; C10 is partial. '
                              'Not a total interaction energy or SAPT comparison.')


def run(args):
    checks = helper('phoenix_installed_checks')
    allocation = checks.allocation()
    q, c = helper('qomp_execution'), helper('qomp_campaign')
    checks.qomp_engine().qomp_enter()
    os.environ.update(PATH=str(MANAGER/'bin')+':/opt/slurm/current/bin:/usr/bin:/bin',
                      LD_LIBRARY_PATH=str(MANAGER/'lib'),
                      PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    if (Path(__file__).resolve().parents[2] != SOURCE or args.run != RUN
            or q.git(SOURCE, 'rev-parse', 'HEAD') != args.commit
            or q.git(SOURCE, 'status', '--porcelain')):
        raise ValueError('exact clean isolated water-check source/run required')
    q.qualified().verify_descriptor_operations(
        SOURCE, q.source_changes(SOURCE, q.qualified().SCIENTIFIC_COMMIT, args.commit))
    q.runtime_check()
    manager_sources = MANAGER/'lib/python3.13/site-packages/qcfractalcompute'
    manager_pins = helper('phoenix_pilot').MANAGER_FILES
    bootstrap = P/'runs/ccasp-native-s66-qomp148-bank003/inputs/bootstrap'
    authority_raw = c.read(bootstrap/'authority.json')
    if c.sha(authority_raw) != BOOTSTRAP_SHA256:
        raise ValueError('parent-pinned installed-worker authority changed')
    authority = json.loads(authority_raw)
    worker_raw = c.read(bootstrap/'worker-preflight.json')
    if c.sha(worker_raw) != authority['worker_sha256']:
        raise ValueError('actual installed-worker receipt changed')
    worker = json.loads(worker_raw)

    def verify_manager():
        from importlib.metadata import version
        expected = dict(qcportal='0.63', qcfractalcompute='0.63',
                        qcengine='0.34.1+3.gd56a0598', qcelemental='0.30.1')
        if any(version(name) != pinned for name, pinned in expected.items()):
            raise ValueError('reviewed SDK/harness package versions changed')
        if any(checks.digest(manager_sources/name) != digest
               for name, digest in manager_pins.items()):
            raise ValueError('reviewed installed manager sources changed')
        if checks.digest(worker['harness_path']) != worker['harness_sha256']:
            raise ValueError('reviewed scientific QCEngine harness changed')

    verify_manager()
    bank_raw = c.read(c.INPUTS/'descriptor-bank.json')
    if c.sha(bank_raw) != c.BANK_SHA256:
        raise ValueError('existing measured QOMP input bank changed')
    bank = json.loads(bank_raw)
    rows = [next(x for x in bank['monomers'] if x['monomer_id'] == mid)
            for mid in ('monomer-0001', 'monomer-0002')]
    profile = helper('qomp_bank_policy').validate_execution_profile(rows[0])
    for row in rows:
        if (row['runtime_input']['molecule']['symbols'] != ['O', 'H', 'H']
                or helper('qomp_bank_policy').validate_execution_profile(row) != profile):
            raise ValueError('original water-dimer fragment request/shape required')
        if c.sha(c.encode(row['runtime_input'])) != row['runtime_input_sha256']:
            raise ValueError('immutable water fragment request changed')
    if (allocation.get('NumCPUs') != str(profile['cores'])
            or allocation.get('MinMemoryNode') != str(profile['job_memory_GiB'])+'G'):
        raise ValueError('water-check exact existing science resource shape required')
    payload = RUN/'payload'
    payload.mkdir(mode=0o700)
    write(payload/'started.json', dict(schema='qca071-water-check/1', allocation=allocation,
                                      commit=args.commit, bank_sha256=c.BANK_SHA256,
                                      monomers=[r['monomer_id'] for r in rows],
                                      execution_profile=profile))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    original_env = dict(os.environ)
    original_cwd = Path.cwd()
    bridge = helper('qca071_bridge')
    try:
        with bridge.forward(RUN/'tunnel') as uri:
            server = bridge.server_config(uri)
            from qcportal import PortalClient
            class DiagnosticClient(PortalClient):
                def _request(self, method, endpoint, **kwargs):
                    # No automatic replay of diagnostic creation/submission
                    # after an ambiguous HTTP response.
                    if method.lower() not in ('get', 'head'):
                        kwargs['allow_retries'] = False
                        kwargs['internal_retry'] = False
                    return super()._request(method, endpoint, **kwargs)
            client = DiagnosticClient(server['fractal_uri'], username=server['username'],
                                      password=server['password'], verify=server['verify'],
                                      show_motd=False)
            info = bridge.validate_information(client.get_server_information())
            write(payload/'server.json', info)
            # Dedicated diagnostic namespace; no production dataset is modified.
            name = 'CCASP-QCA071-water-dimer-check-'+os.environ['SLURM_JOB_ID']
            write(payload/'dataset-create-intent.json', dict(name=name, find_existing=False))
            dataset = client.add_dataset('singlepoint', name, description=(
                'Isolated water-dimer-fragment compatibility check; not production acceptance.'))
            if dataset.id in c.PROTECTED_DATASETS or dataset.id == 359:
                raise ValueError('diagnostic dataset must not be a production dataset')
            write(payload/'dataset-created.json', dict(id=dataset.id, name=dataset.name))
            from qcportal.singlepoint import SinglepointDatasetNewEntry
            from qcportal.dataset_models import DatasetSubmitBody
            from qcportal.metadata_models import InsertCountsMetadata
            summary = []
            envelopes = []
            for row in rows:
                mid = row['monomer_id']
                directory = payload/mid
                directory.mkdir(mode=0o700)
                runtime = copy.deepcopy(row['runtime_input'])
                runtime['specification']['protocols'] = (
                    helper('phoenix_pilot').execution_specification({})['protocols'])
                write(directory/'expected-runtime.json', runtime)
                spec_name = 'pbe0-atz-native-qomp-request3-water-check'
                if mid == rows[0]['monomer_id']:
                    meta = dataset.add_specification(
                        spec_name, dict(program='psi4', **runtime['specification']))
                    if not meta.success or meta.n_inserted != 1:
                        raise ValueError('one new diagnostic specification required')
                # Entry keywords retain each fragment's own declaration/frames.
                entry_keywords = copy.deepcopy(runtime['specification']['keywords'])
                meta = dataset.add_entries([SinglepointDatasetNewEntry(
                    name=row['entry_name'], molecule=runtime['molecule'],
                    additional_keywords=entry_keywords,
                    attributes=dict(diagnostic_only=True, monomer_id=mid))])
                if not meta.success or meta.n_inserted != 1 or meta.n_existing:
                    raise ValueError('one new diagnostic entry required')
                # Direct QCEngine and manager QCEngine both execute the same
                # sealed ordinary CLI, with no Python scientific overlay.
                direct_dir = directory/'direct'
                scratch = environment(direct_dir, mid, 'direct', profile, args.commit)
                # Import QCEngine only after replacing HOME/cache/loader state,
                # just as the production science entry point does.
                import qcengine
                import qcfractalcompute
                helper('phoenix_worker_cb63305').validate_versions(
                    str(qcengine.__version__), str(qcfractalcompute.__version__))
                input_data = dict(schema_name='qcschema_input', schema_version=1,
                                  molecule=runtime['molecule'], **runtime['specification'])
                result = qcengine.compute(input_data, 'psi4',
                    task_config=dict(ncores=profile['cores'],
                                     memory=float(profile['science_memory_GiB']),
                                     scratch_directory=str(scratch), retries=0),
                    return_dict=True)
                if result.get('success') is not True:
                    write(direct_dir/'failed-result.json', result)
                    raise RuntimeError('direct water native calculation failed')
                direct = result['extras']['psi4:native_monomer']
                q.runtime_check()
                verify_manager()
                os.environ.clear(); os.environ.update(original_env)
                os.chdir(original_cwd)
                qca_dir = directory/'qca'
                scratch = environment(qca_dir, mid, 'qca', profile, args.commit)
                tag = f'qca071-water-check-{allocation["JobId"]}-{mid}'
                write(qca_dir/'submit-intent.json',
                      dict(dataset_id=dataset.id, entry=row['entry_name'],
                           specification=spec_name, compute_tag=tag, find_existing=False))
                inserted = client.make_request('post',
                    f'api/v1/datasets/singlepoint/{dataset.id}/submit',
                    InsertCountsMetadata, body=DatasetSubmitBody(
                        entry_names=[row['entry_name']], specification_names=[spec_name],
                        compute_tag=tag, find_existing=False), allow_retries=False)
                if not inserted.success or inserted.n_inserted != 1 or inserted.n_existing:
                    raise ValueError('one fresh diagnostic record required; no retry')
                record = dataset.get_record(row['entry_name'], spec_name, force_refetch=True)
                write(qca_dir/'record-created.json',
                      dict(dataset_id=dataset.id, record_id=record.id, compute_tag=tag))
                managers = list(client.query_managers(status=['active'], limit=200))
                if len(managers) >= 200 or any(
                        any(t == tag or any(ch in t for ch in '*?[') for t in m.compute_tags)
                        for m in managers):
                    raise ValueError('diagnostic manager tag isolation not established')
                # Build using the existing reviewed manager constructor, then
                # change ONLY its diagnostic network destination in memory.
                original_server = dict(server, fractal_uri=bridge.ORIGINAL_URI)
                cfg = helper('phoenix_pilot').pilot_config(
                    original_server, qca_dir, scratch, tag)
                cfg['server']['fractal_uri'] = uri
                cfg['executors']['pilot'].update(
                    cores_per_worker=profile['cores'],
                    memory_per_worker=float(profile['science_memory_GiB']))
                from qcfractalcompute.config import FractalComputeConfig
                config = FractalComputeConfig(**cfg)
                manager, local = manager_session(
                    config, record.id, tag, runtime, qca_dir)
                record = dataset.get_record(row['entry_name'], spec_name, force_refetch=True)
                if (record.status.value != 'complete' or len(record.compute_history) != 1
                        or record.compute_history[0].manager_name != manager):
                    raise ValueError('one complete water result from its exact manager required')
                retrieved = record.extras['psi4:native_monomer']
                helper('phoenix_pilot_cb63305').verify_delivery_envelope(
                    local, retrieved,
                    runtime['specification']['keywords']['function_kwargs']['native_monomer'],
                    q.qualified().IMPLEMENTATION, producer='qomp')
                comparison = compare(direct, retrieved, directory)
                envelopes.append(dict(direct=direct, qca=retrieved))
                q.runtime_check()
                summary.append(dict(monomer_id=mid, record_id=record.id,
                                    comparison=comparison))
                os.environ.clear(); os.environ.update(original_env)
                os.chdir(original_cwd)
            q.runtime_check()
            verify_manager()
            if q.git(SOURCE, 'rev-parse', 'HEAD') != args.commit or q.git(
                    SOURCE, 'status', '--porcelain'):
                raise ValueError('water-check operational source changed')
            report = dict(status='PASS_QCA071_WATER_DIMER_FRAGMENTS',
                          allocation=allocation, commit=args.commit,
                          server=info, dataset_id=dataset.id, members=summary,
                          water_dimer=compare_dimer(rows, envelopes),
                          production_dataset_modified=False,
                          scientific_runtime_unchanged=True)
            write(payload/'comparison-report.json', report)
            print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        os.environ.clear(); os.environ.update(original_env)
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
