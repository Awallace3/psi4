"""Independent post-delivery cb/request3 tensor and decoded-FD trace audit.

No quantum calculation, server mutation, record reset or manager is started.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

RUNS = Path('/storage/project/r-cs207-0/awallace43/runs')
SCRATCH = Path('/storage/scratch1/3/awallace43')
BASE = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt')
MANAGER = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca')
HELPERS = {'pilot_cb63305', 'phoenix_pilot_cb63305', 'phoenix_worker_cb63305',
    'phoenix_installed_checks', 'phoenix_worker_probe', 'phoenix_pilot', 'runtime_audit',
    'validate_numeric', 'dataset_fresh_cb63305', 'phoenix_descriptor_bank_cb63305'}


def helper(name):
    if name not in HELPERS:
        raise ValueError('closed expanded runtime audit helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_expanded_audit_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def scratch_for_outcome(outcome):
    job = outcome['allocation']['JobId']
    if (type(job) is not str or re.fullmatch(r'[0-9]+', job) is None
            or outcome.get('monomer_id') != 'monomer-0006'):
        raise ValueError('first expanded-pilot job and monomer identity required')
    expected = SCRATCH/('ccasp-native-cb63305-monomer-0006-'+job)
    if (outcome.get('scratch_path') != str(expected)
            or outcome.get('compute_tag') != 'ccasp-native-cb63305-monomer-0006-'+job):
        raise ValueError('expanded molecular scratch/tag differs from actual case identity')
    return expected


def validate_delivery(outcome, delivery, created, claimed, server):
    if (outcome.get('schema') != 'native-expanded-pilot-outcome/1'
            or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
            or outcome.get('runtime_unchanged') is not True
            or outcome.get('find_existing') is not False or outcome.get('maximum_records') != 1
            or type(outcome.get('dataset_id')) is not int or outcome['dataset_id'] <= 0
            or outcome['dataset_id'] == 353
            or type(outcome.get('record_id')) is not int or outcome['record_id'] <= 0
            or outcome['record_id'] in (5234314, 5234315)):
        raise ValueError('fresh delivered expanded single-record case required')
    record, task, manager = outcome['record_id'], outcome['task_id'], outcome['manager']
    stats = delivery['local_statistics']
    if (created['dataset_id'] != outcome['dataset_id']
            or created['compute_tag'] != outcome['compute_tag']
            or created['n_inserted'] != 1 or created['n_existing'] != 0
            or any(item['record_id'] != record for item in (delivery, created, claimed, server))
            or claimed['task_id'] != task or delivery['task_id'] != task
            or claimed['manager'] != manager or delivery['manager'] != manager
            or delivery['outcome'] != 'accepted' or delivery['interrupted'] is not False
            or stats['total_successful_tasks'] != 1 or stats['total_failed_tasks'] != 0
            or stats['total_rejected_tasks'] != 0 or stats['active_tasks'] != 0
            or server['status'] != 'complete' or server['history_count'] != 1
            or server['history_managers'] != [manager]):
        raise ValueError('singleton claim, delivery or server history differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'case', 'worker-preflight', 'qualification', 'inputs', 'descriptors', 'pilot', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    checks = helper('phoenix_installed_checks')
    allocation = checks.allocation()
    source = Path(__file__).resolve().parents[2]
    if (checks.git(source, 'rev-parse', 'HEAD') != args.commit
            or checks.git(source, 'status', '--porcelain')):
        raise ValueError('exact clean independently pinned audit source required')
    build, case, run = (p.resolve() for p in (args.build, args.case, args.run))
    if (any(not p.is_relative_to(RUNS) for p in (build, case, run))
            or any(a.is_relative_to(b) for a in (build, case, run)
                   for b in (build, case, run) if a != b)
            or len({build, case, run}) != 3):
        raise ValueError('disjoint private build, molecular case and audit run required')
    data = case/'payload'
    read = lambda name: json.loads((data/name).read_bytes())
    outcome, delivery, created, claimed, server = (
        read(name) for name in ('pilot-outcome.json', 'manager-delivery.json',
                               'record-created.json', 'claim.json', 'server-record-outcome.json'))
    validate_delivery(outcome, delivery, created, claimed, server)
    scratch = scratch_for_outcome(outcome)
    bootstrap = helper('phoenix_pilot_cb63305')
    bootstrap.admit_allocation(outcome['allocation'])
    manifest_path = build/'build-identity.json'
    manifest = json.loads(manifest_path.read_bytes())
    prep = helper('pilot_cb63305')
    draft = prep.prepare_from_paths(args.inputs, manifest_path, args.qualification, args.descriptors)
    if json.loads(args.pilot.read_bytes()) != draft:
        raise ValueError('independent audit pilot does not reproduce from actual admitted raw inputs')
    worker, legacy = helper('phoenix_worker_cb63305'), helper('phoenix_pilot')
    preflight = json.loads(args.worker_preflight.read_bytes())
    transport = helper('dataset_fresh_cb63305').transport_binding()
    bootstrap.validate_preflight(preflight, draft, manifest_sha=checks.digest(manifest_path),
        qualification_sha=checks.digest(args.qualification), pilot_sha=checks.digest(args.pilot),
        worker_sources=worker.source_hashes(), transport=transport)
    if (outcome['scientific_build_commit'] != draft['producer']['scientific_commit']
            or outcome['build_identity_sha256'] != checks.digest(manifest_path)
            or outcome['pilot_sha256'] != checks.digest(args.pilot)
            or outcome['installed_qualification_sha256'] != checks.digest(args.qualification)
            or outcome['worker_preflight_sha256'] != checks.digest(args.worker_preflight)
            or outcome['descriptor_allocation_sha256'] != draft['descriptor_allocation_sha256']
            or outcome['runtime_input_sha256'] != draft['runtime_input_sha256']
            or outcome['transport_binding'] != transport
            or outcome['manager_sources'] != legacy.MANAGER_FILES):
        raise ValueError('new producer, raw pilot and installed discovery receipt chain differs')
    intended_dataset, submitted = read('dataset-created.json'), read('record-submit-attempt.json')
    if (intended_dataset != dict(dataset_id=outcome['dataset_id'], name=draft['dataset_name'])
            or submitted['dataset_id'] != outcome['dataset_id']
            or submitted['entry'] != draft['entry']['name']
            or submitted['specification'] != draft['specification_name']
            or submitted['compute_tag'] != outcome['compute_tag']
            or submitted['find_existing'] is not False or submitted['transport_retries'] is not False
            or set(outcome['operational_sources_sha256']) != set(worker.SOURCES) | {
                'phoenix_pilot_cb63305.py', 'phoenix_pilot_cb63305.sbatch', 'single_record_cb63305.py'}):
        raise ValueError('fresh dataset intent or complete execution-source inventory differs')
    expected = helper('phoenix_worker_probe').runtime_input(draft)
    expected['specification']['protocols'] = legacy.execution_specification(draft['specification'])['protocols']
    if expected != read('expected-runtime.json'):
        raise ValueError('expected molecular runtime was not independently reconstructed')
    raw_claim, envelope = read('claimed-input.json'), read('server-native-envelope.json')
    legacy.validate_claim(raw_claim, expected, record_id=outcome['record_id'],
                          compute_tag=outcome['compute_tag'])
    request = expected['specification']['keywords']['function_kwargs']['native_monomer']
    if (request['schema'] != 'native-monomer-request/3' or envelope['request'] != request
            or envelope['implementation']['sha256'] != draft['producer']['implementation_sha256']):
        raise ValueError('new request3 or qualified implementation differs across transport')
    construction = envelope['molecule_construction']
    if (construction['input_fix_symmetry'] != raw_claim['function_kwargs']['input_data']['molecule'].get('fix_symmetry')
            or construction['requested_point_group'] != 'c1'
            or construction['effective_point_group'] != 'c1'
            or not 0 <= construction['max_abs_coordinate_change_bohr'] <= 1e-10):
        raise ValueError('private point-group or original ordered coordinates changed')
    from qcportal.compression import decompress, CompressionEnum
    local = decompress((data/('local-result-'+str(outcome['task_id'])+'.zstd')).read_bytes(),
                       CompressionEnum.zstd)
    if not local['success'] or local['extras']['psi4:native_monomer'] != envelope:
        raise ValueError('independent local/server native envelope comparison failed')
    job = outcome['allocation']['JobId']
    rows = subprocess.check_output(['/opt/slurm/current/bin/sacct', '-n', '-P', '-j', job,
        '-o', 'JobIDRaw,State,ExitCode'], text=True).splitlines()
    for identifier in (job+suffix for suffix in ('', '.batch', '.0')):
        if not any(line.strip().split('|')[:3] == [identifier, 'COMPLETED', '0:0'] for line in rows):
            raise ValueError('case allocation/batch/step did not all complete0:0')
    # Bind every execution helper to both the actual case revision and the
    # current read-only audit checkout; never silently relax the old trace gate.
    for name, expected_hash in outcome['operational_sources_sha256'].items():
        relative = 'devtools/isapol_s66_campaign/'+name
        raw = subprocess.check_output(['git', '-C', str(source), 'show',
                                        outcome['operational_commit']+':'+relative])
        if checks.digest(Path(__file__).with_name(name)) != expected_hash or hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError('actual case execution helper changed: '+name)
    gate_relative = 'devtools/isapol_native_acceptance/trace_gate.py'
    gate_path = source/gate_relative
    for relative in (gate_relative, 'devtools/isapol_s66_campaign/validate_numeric.py',
                     'devtools/isapol_s66_campaign/runtime_audit.py'):
        raw = subprocess.check_output(['git', '-C', str(source), 'show',
                                       outcome['operational_commit']+':'+relative])
        if (source/relative).read_bytes() != raw:
            raise ValueError('independent numeric/trace policy changed since actual molecular run')
    spec = importlib.util.spec_from_file_location('_expanded_decoded_fd_gate', gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    environment = build/'env'
    checks.verify_runtime(environment, manifest)
    numeric = helper('validate_numeric').validate_numeric(envelope, (data/'server-tensors.npz').read_bytes())
    traces = sorted((data/'traces').glob('*.strace'))
    if not traces or [p.name for p in traces] != outcome['trace_files']:
        raise ValueError('actual scientific CLI trace inventory differs')
    roles, reports = Counter(), {}
    classifier = helper('runtime_audit').cli_role
    forbidden = ['/storage/project/r-cs207-0/awallace43/gits',
        '/storage/project/r-cs207-0/awallace43/qca', '/storage/project/r-cs207-0/awallace43/data',
        str(case/'inputs'), str(case/'payload'), str(args.inputs), str(args.descriptors)]
    for path in traces:
        report = gate.check_trace(path.read_text().splitlines(), env=environment, base=BASE,
            manifest_path=manifest_path, manifest=manifest, run=case, scratch=scratch, forbidden=forbidden)
        report['trace_sha256'] = checks.digest(path)
        if len(report['executions']) != 1:
            report['status'], report['role_error'] = 'failed', 'only the initial owned interpreter may execute'
        else:
            try:
                report['role'] = classifier(report['executions'][0], env=environment, scratch=scratch)
                roles[report['role']] += 1
            except ValueError as error:
                report['status'], report['role_error'] = 'failed', str(error)
        reports[path.name] = report
    checks.verify_runtime(environment, manifest)
    if (checks.digest(preflight['harness_path']) != preflight['harness_sha256']
            or any(checks.digest(MANAGER/'lib/python3.13/site-packages/qcfractalcompute'/n) != v
                   for n, v in legacy.MANAGER_FILES.items())
            or checks.git(source, 'rev-parse', 'HEAD') != args.commit
            or checks.git(source, 'status', '--porcelain')
            or prep.prepare_from_paths(args.inputs, manifest_path, args.qualification, args.descriptors) != draft):
        raise ValueError('runtime, manager, audit source or immutable raw receipts changed')
    passed = all(r['status'] == 'passed' for r in reports.values()) and roles['molecular_qcschema'] == 1
    output = run/'payload'
    output.mkdir(exist_ok=False)
    report = dict(schema='native-expanded-pilot-audit/1',
        status='PASS_EXPANDED_SINGLE_PILOT_EVIDENCE' if passed else 'FAIL_EXPANDED_RUNTIME_INPUT_AUDIT',
        allocation=allocation, monomer_id=draft['monomer_id'], case_job=job,
        dataset_id=outcome['dataset_id'], record_id=outcome['record_id'],
        scientific_build_commit=manifest['commit'], case_operational_commit=outcome['operational_commit'],
        audit_commit=args.commit, case_outcome_sha256=checks.digest(data/'pilot-outcome.json'),
        pilot_sha256=checks.digest(args.pilot), file_gate_sha256=checks.digest(gate_path),
        numeric=numeric, roles=dict(roles), traces=reports, runtime_unchanged=True,
        qualification='One fresh cb/request3 molecular pilot execution/export/decoded-FD evidence only; '
                      'ordinary OS trust, not hostile-host/TOCTOU or atomic-server-lock proof. '
                      'No full-C10, independent solver, covariance, accuracy/parity or whole-set claim.')
    with (output/'audit-report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], record_id=outcome['record_id'], roles=dict(roles))))
    if not passed:
        raise RuntimeError('expanded molecular runtime audit failed; do not advance models')


if __name__ == '__main__':
    main()
