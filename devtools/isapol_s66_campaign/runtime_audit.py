"""Post-run audit of one delivered native pilot; never runs quantum work."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def cli_role(execution, *, env, scratch, worker_shape=(4, 20.)):
    """Each trace must start exactly one owned isolated Psi4 CLI interpreter."""
    if (type(worker_shape) is not tuple or len(worker_shape) != 2
            or type(worker_shape[0]) is not int
            or type(worker_shape[1]) not in (int, float)
            or worker_shape not in ((4, 20.), (8, 20.), (12, 40.), (24, 80.), (16, 60.))):
        raise ValueError('explicit finite CLI worker shape required')
    env, scratch = str(env), str(scratch)
    match = re.fullmatch(r'execve\("([^"]+)", (\[.*\]), .*?\)\s+=\s+0',
                         execution['syscall'])
    if match is None:
        raise ValueError('exact successful initial execve required')
    argv = json.loads(match[2])
    prefix = [env+'/bin/python', '-I', '-B', '-X',
              'pycache_prefix='+scratch+'/cache', env+'/bin/psi4']
    if match[1] != prefix[0] or argv[:6] != prefix:
        raise ValueError('foreign interpreter, CLI or import-cache boundary')
    rest = argv[6:]
    if rest == ['--module']:
        return 'discovery_module'
    if rest == ['--version']:
        return 'discovery_version'
    if (len(rest) == 8 and rest[0] == '--scratch'
            and str(Path(rest[1]).resolve()).startswith(scratch+'/')
            and rest[2:] == ['--nthread', str(worker_shape[0]), '--memory',
                             str(float(worker_shape[1]))+'GiB',
                             '--qcschema', 'data.msgpack']):
        return 'molecular_qcschema'
    raise ValueError('foreign CLI execution mode, options or scratch input')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scratch_for_outcome(outcome):
    job = outcome['allocation']['JobId']
    checkpoint = outcome.get('append_checkpoint')
    label = 'water' if checkpoint is None else checkpoint['monomer_id']
    if checkpoint is not None and label != 'monomer-0004':
        raise ValueError('unsupported molecular checkpoint scratch identity')
    expected = Path('/storage/scratch1/3/awallace43')/('ccasp-native-'+label+'-'+job)
    if outcome.get('scratch_path', str(expected)) != str(expected):
        raise ValueError('scratch path differs from validated case identity')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'case', 'worker-preflight', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--pilot', type=Path)
    append = load(Path(__file__).with_name('append_receipts.py'), 'append_receipts')
    append.add_arguments(parser)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    helper = load(Path(__file__).with_name('phoenix_installed_checks.py'), 'installed_checks')
    numeric = load(Path(__file__).with_name('validate_numeric.py'), 'numeric')
    pilot = load(Path(__file__).with_name('phoenix_pilot.py'), 'pilot')
    allocation = helper.allocation()
    if helper.git(source, 'rev-parse', 'HEAD') != args.commit or helper.git(source, 'status', '--porcelain'):
        raise RuntimeError('pinned clean operational audit source required')
    build, case, run = (p.resolve() for p in (args.build, args.case, args.run))
    prefix = Path('/storage/project/r-cs207-0/awallace43/runs')
    if any(not p.is_relative_to(prefix) for p in (build, case, run)) or case == run:
        raise RuntimeError('separate owned case and audit output required')
    output = run/'payload'
    output.mkdir(exist_ok=False)
    data = case/'payload'
    read = lambda name: json.loads((data/name).read_text())
    outcome, delivery, created, claimed, server = map(
        read, ('pilot-outcome.json', 'manager-delivery.json', 'record-created.json',
               'claim.json', 'server-record-outcome.json'))
    expected = read('expected-runtime.json')
    raw_claim = read('claimed-input.json')
    envelope = read('server-native-envelope.json')
    manifest_path = build/'build-identity.json'
    manifest = json.loads(manifest_path.read_text())
    preflight = json.loads(args.worker_preflight.read_text())
    checkpoint = append.verify(args)
    preservation = None
    if checkpoint is not None:
        if (outcome.get('append_checkpoint') != checkpoint
                or preflight.get('append_checkpoint') != checkpoint
                or preflight['operational_commit'] != outcome['operational_commit']
                or outcome['dataset_id'] != 353
                or outcome.get('append_transport_binding') != preflight.get('append_transport_binding')
                or outcome.get('append_transport_binding') != append.helper('dataset_append').transport_binding()):
            raise RuntimeError('append receipt/operational chain differs')
        guard = append.helper('dataset_guard')
        probe = append.helper('phoenix_worker_probe')
        draft, _ = append.load(args.pilot)
        rebuilt = probe.runtime_input(draft)
        rebuilt['specification']['protocols'] = pilot.execution_specification(draft['specification'])['protocols']
        if expected != rebuilt:
            raise RuntimeError('expected runtime was not reconstructed from bound raw input')
        baseline, _ = append.load(args.append_baseline)
        before = read('dataset-before.json')
        if before != baseline['snapshot']:
            raise RuntimeError('case before-state differs from approved baseline')
        name = draft['entry']['name']
        guard.assert_append_delta(before, read('dataset-after-entry.json'), entry_name=name)
        for filename in ('dataset-after-submit.json', 'dataset-after-delivery.json'):
            snapshot = read(filename)
            guard.assert_append_delta(before, snapshot, entry_name=name, record_id=outcome['record_id'])
            guard.validate_stored_entry(snapshot['entries'][name], draft['entry'])
        preservation = read('dataset-preservation.json')
        if (preservation['status'] != 'PASS_EXISTING_DATASET_PRESERVED'
                or preservation['before_sha256'] != guard.digest(before)
                or preservation['after_sha256'] != guard.digest(read('dataset-after-delivery.json'))
                or preservation['added_entry'] != name
                or preservation['added_record'] != outcome['record_id']):
            raise RuntimeError('dataset-preservation receipt differs')
    elif outcome.get('append_checkpoint') is not None:
        raise RuntimeError('append case requires all bound input receipts for audit')
    if (outcome['status'] != 'DELIVERED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
            or not outcome['runtime_unchanged']
            or outcome['build_identity_sha256'] != helper.digest(manifest_path)
            or outcome['scientific_build_commit'] != manifest['commit']
            or outcome['worker_preflight_sha256'] != helper.digest(args.worker_preflight)
            or outcome['find_existing'] is not False):
        raise RuntimeError('complete matching delivered native case required')
    record_id, task_id, manager = outcome['record_id'], outcome['task_id'], outcome['manager']
    if (created['n_inserted'] != 1 or created['n_existing'] != 0
            or any(item['record_id'] != record_id for item in (delivery, created, claimed, server))
            or claimed['task_id'] != task_id or delivery['task_id'] != task_id
            or claimed['manager'] != manager or delivery['manager'] != manager
            or delivery['outcome'] != 'accepted' or delivery['interrupted']
            or delivery['local_statistics']['total_successful_tasks'] != 1
            or delivery['local_statistics']['total_failed_tasks'] != 0
            or delivery['local_statistics']['total_rejected_tasks'] != 0
            or delivery['local_statistics']['active_tasks'] != 0
            or server['status'] != 'complete' or server['history_count'] != 1
            or server['history_managers'] != [manager]):
        raise RuntimeError('fresh single-record claim/delivery/server provenance differs')
    pilot.validate_claim(raw_claim, expected, record_id=record_id, compute_tag=outcome['compute_tag'])
    request = expected['specification']['keywords']['function_kwargs']['native_monomer']
    if envelope['request'] != request or envelope['implementation']['sha256'] != request['implementation_sha256']:
        raise RuntimeError('native request/implementation changed across transport')
    construction = envelope['molecule_construction']
    if (construction['input_fix_symmetry'] != raw_claim['function_kwargs']['input_data']['molecule'].get('fix_symmetry')
            or construction['requested_point_group'] != 'c1'
            or construction['effective_point_group'] != 'c1'
            or not 0 <= construction['max_abs_coordinate_change_bohr'] <= 1e-10):
        raise RuntimeError('private computational symmetry or coordinates changed')
    from qcportal.compression import decompress, CompressionEnum
    local = decompress((data/('local-result-'+str(task_id)+'.zstd')).read_bytes(), CompressionEnum.zstd)
    if not local['success'] or local['extras']['psi4:native_monomer'] != envelope:
        raise RuntimeError('independent local/server envelope comparison failed')
    job = outcome['allocation']['JobId']
    rows = subprocess.check_output([
        '/opt/slurm/current/bin/sacct', '-n', '-P', '-j', job,
        '-o', 'JobIDRaw,State,ExitCode'], text=True).splitlines()
    for identifier in (job, job+'.batch', job+'.0'):
        if not any(line.strip().split('|')[:3] == [identifier, 'COMPLETED', '0:0'] for line in rows):
            raise RuntimeError('case allocation/batch/payload step did not all complete0:0')
    gate_relative = 'devtools/isapol_native_acceptance/trace_gate.py'
    gate_path = source/gate_relative
    baseline_gate = subprocess.check_output(
        ['git', 'show', outcome['operational_commit']+':'+gate_relative], cwd=source)
    if gate_path.read_bytes() != baseline_gate:
        raise RuntimeError('file/exec gate changed since the actual case; no silent policy relaxation')
    gate = load(gate_path, 'decoded_fd_gate')
    environment = build/'env'
    base = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt')
    scratch = scratch_for_outcome(outcome)
    helper.verify_runtime(environment, manifest)
    raw = (data/'server-tensors.npz').read_bytes()
    numeric_report = numeric.validate_numeric(envelope, raw)
    traces = sorted((data/'traces').glob('*.strace'))
    if [p.name for p in traces] != outcome['trace_files'] or not traces:
        raise RuntimeError('trace inventory is missing or changed')
    roles, reports = Counter(), {}
    forbidden = [
        '/storage/project/r-cs207-0/awallace43/gits',
        '/storage/project/r-cs207-0/awallace43/qca',
        '/storage/project/r-cs207-0/awallace43/data',
        # Dataset reference attributes and operational/result receipts are not
        # scientific inputs. Native CLI data/checkpoints belong in own scratch.
        str(case/'inputs'), str(case/'payload'),
    ]
    for path in traces:
        report = gate.check_trace(
            path.read_text().splitlines(), env=environment, base=base,
            manifest_path=manifest_path, manifest=manifest, run=case,
            scratch=scratch, forbidden=forbidden)
        report['trace_sha256'] = helper.digest(path)
        if len(report['executions']) != 1:
            report['status'] = 'failed'
            report['role_error'] = 'trace must contain only its initial owned interpreter execution'
        else:
            try:
                report['role'] = cli_role(report['executions'][0], env=environment, scratch=scratch)
                roles[report['role']] += 1
            except ValueError as exc:
                report['status'], report['role_error'] = 'failed', str(exc)
        reports[path.name] = report
    helper.verify_runtime(environment, manifest)
    manager_sources = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca/lib/python3.13/site-packages/qcfractalcompute')
    if (helper.digest(preflight['harness_path']) != preflight['harness_sha256']
            or any(helper.digest(manager_sources/n) != v for n, v in outcome['manager_sources'].items())
            or helper.digest(manifest_path) != outcome['build_identity_sha256']
            or helper.git(source, 'rev-parse', 'HEAD') != args.commit
            or helper.git(source, 'status', '--porcelain')):
        raise RuntimeError('scientific or manager/audit implementation changed')
    passed = all(r['status'] == 'passed' for r in reports.values()) and roles['molecular_qcschema'] == 1
    report = dict(
        status='PASS_SINGLE_PILOT_EVIDENCE' if passed else 'FAIL_RUNTIME_INPUT_AUDIT',
        allocation=allocation, case_job=job, record_id=record_id,
        scientific_build_commit=manifest['commit'], case_operational_commit=outcome['operational_commit'],
        audit_commit=args.commit, file_gate_sha256=helper.digest(gate_path),
        case_outcome_sha256=helper.digest(data/'pilot-outcome.json'), roles=dict(roles),
        append_checkpoint=checkpoint, dataset_preservation=preservation,
        numeric=numeric_report, traces=reports, runtime_unchanged=True,
        qualification='One fresh native/QCArchive pilot, numeric/export and decoded-FD input evidence only. '
                      'Ordinary OS trust, not hostile-host/TOCTOU proof. No full-C10, solver certification, '
                      'convergence, accuracy/parity or full-S66x8 qualification.')
    with (output/'audit-report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], record_id=record_id, roles=dict(roles),
                          trace_statuses={k: v['status'] for k, v in reports.items()},
                          report=str(output/'audit-report.json'))))
    if not passed:
        raise RuntimeError('runtime evidence gate failed; inspect audit-report.json, do not advance pilots')


if __name__ == '__main__':
    main()
