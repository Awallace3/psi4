"""Read-only component audit of the preserved 15090188 launcher incident.

Never emit a successful pilot outcome, amend the failed job, start a manager,
submit science, mutate a server record, or declare a model accepted.
"""
import argparse
import base64
from collections import Counter
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path('/storage/project/r-cs207-0/awallace43')
PREFIX = 'ccasp-native-s66-grid400-1aaa9e0a30'
CASE_COMMIT = '2c2c002de7e5da29ef0998e1c4236674b2a63e6b'
CASE = ROOT/('runs/'+PREFIX+'-monomer-0008-pilot001')
BUILD = ROOT/('runs/'+PREFIX+'-build001')
INPUT = ROOT/('runs/'+PREFIX+'-preparation001')
DESCRIPTORS = ROOT/('runs/'+PREFIX+'-descriptors001/payload')
QUAL = ROOT/('runs/'+PREFIX+'-installed001/payload/installed-validation.json')
WORKER = ROOT/('runs/'+PREFIX+'-monomer-0008-worker001/payload/worker-preflight.json')
LOCAL_SHA = '3a6cd61b2a2eb8d038edd2b5d7e6213179586e81de07881da7b0cf9a8db5ae06'
TENSOR_SHA = 'aeadec4badf574785f8fc9960d462bc26b26ce7519cca1551555e308b450267a'
DIAGNOSTIC_PATH = ('result', 'provenance', 'partition', 'diagnostics', 'drho_metric_condition')
FAILURE = 'RuntimeError: local/server request3 envelope or producer identity differs'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location('_grid400_forensic_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def bounded(path, limit=16*1024**2):
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    require(0 < len(raw) <= limit, 'bounded nonempty forensic evidence required')
    return raw


def compare_incident_envelopes(local, server):
    """Exact JSON-number proof at ONE measured path; no epsilon tolerance."""
    require(type(local) is dict and type(server) is dict, 'original envelope dictionaries required')
    a, b = copy.deepcopy(local), copy.deepcopy(server)
    for key in DIAGNOSTIC_PATH[:-1]:
        require(type(a.get(key)) is dict and type(b.get(key)) is dict,
                'measured diagnostic path missing')
        a, b = a[key], b[key]
    key = DIAGNOSTIC_PATH[-1]
    x, y = a.pop(key, None), b.pop(key, None)
    require(type(x) is float and type(y) is int and math.isfinite(x),
            'only the measured float-to-integer diagnostic incident is supported')
    require(repr(x) == '5.141176000843883e+16' and y == 51411760008438830,
            'incident diagnostic values differ')
    require(Decimal(json.dumps(x, allow_nan=False)) == Decimal(json.dumps(y, allow_nan=False)),
            'exact JSON decimal values differ')
    # Rebuild copies for the full comparison, deleting just the proved path.
    left, right = copy.deepcopy(local), copy.deepcopy(server)
    for envelope in (left, right):
        node = envelope
        for part in DIAGNOSTIC_PATH[:-1]:
            node = node[part]
        del node[key]
    # Canonical JSON compares every remaining value/structure exactly; no
    # float/int normalization, tolerance, or ignored metadata elsewhere.
    encode = lambda value: json.dumps(value, sort_keys=True, allow_nan=False).encode()
    require(encode(left) == encode(right), 'another envelope field differs')
    require(local != server, 'original strict comparison did not reproduce')
    return dict(status='EXACT_JSON_DECIMAL_EQUIVALENCE_AT_MEASURED_DIAGNOSTIC_ONLY',
        path='/'.join(DIAGNOSTIC_PATH), local_json=json.dumps(x), server_json=json.dumps(y),
        local_type='float', server_type='int', epsilon_tolerance_used=False,
        original_python_equality=False, every_other_field_exact=True)


def tensor_bytes(envelope):
    artifact = envelope['artifacts']['tensors.npz']
    require(type(artifact['data']) is str and len(artifact['data']) <= 4*((16*1024**2+2)//3),
            'bounded original tensor artifact required')
    raw = base64.b64decode(artifact['data'], validate=True)
    require(type(artifact['size']) is int and len(raw) == artifact['size'] == 323646
            and sha(raw) == artifact['sha256'] == TENSOR_SHA, 'exact incident tensor bytes required')
    return raw


def validate_chain(started, created, claim, delivery, captured_server, live_server):
    require(started.get('status') == 'STARTED_NOT_QUALIFIED'
        and started.get('producer_profile') == 'grid400' and started.get('dataset_mode') == 'fresh'
        and started.get('monomer_id') == 'monomer-0008'
        and started.get('operational_commit') == CASE_COMMIT
        and started.get('allocation', {}).get('JobId') == '15090188'
        and started.get('find_existing') is False and started.get('maximum_records') == 1
        and started.get('qcengine_internal_retries') == 0,
        'original fresh singleton start evidence required, not a reconstructed outcome')
    manager = claim['manager']
    require(created.get('dataset_id') == 356 and created.get('n_inserted') == 1
        and created.get('n_existing') == 0
        and created.get('compute_tag') == started.get('compute_tag')
        and all(row.get('record_id') == 5234327 for row in (
            created, claim, delivery, captured_server, live_server))
        and claim.get('task_id') == delivery.get('task_id') == 4722332
        and delivery.get('manager') == manager and delivery.get('outcome') == 'accepted'
        and delivery.get('interrupted') is False
        and delivery.get('claim_error_type') is None and delivery.get('submission_error_type') is None,
        'exact original creation, claim and acknowledged delivery required')
    stats = delivery['local_statistics']
    require(all(stats.get(key) == value for key, value in dict(
        total_successful_tasks=1, total_failed_tasks=0, total_rejected_tasks=0, active_tasks=0).items()),
        'one successful task without failures or active work required')
    for server in (captured_server, live_server):
        require(server.get('status') == 'complete' and server.get('history_count') == 1
            and server.get('history_managers') == [manager], 'exact singleton server history required')


def validate_intents(started, draft, dataset_created, submitted):
    require(started.get('dataset_name') == draft['dataset_name']
        and started.get('specification_name') == draft['specification_name']
        and dataset_created == dict(dataset_id=356, name=draft['dataset_name']),
        'exact original fresh dataset/specification creation intent required')
    require(submitted.get('dataset_id') == 356
        and submitted.get('entry') == draft['entry']['name']
        and submitted.get('specification') == draft['specification_name']
        and submitted.get('compute_tag') == started['compute_tag']
        and submitted.get('find_existing') is False
        and submitted.get('transport_retries') is False,
        'exact original one-shot entry/specification/tag submission intent required')


def readonly_client_class():
    from qcportal import PortalClient

    class ReadOnlyClient(PortalClient):
        def _request(self, method, endpoint, **kwargs):
            if method.lower() not in ('get', 'head'):
                require(method.lower() == 'post' and endpoint in (
                    'auth/v1/login', 'auth/v1/refresh', 'api/v1/records/bulkGet',
                    'api/v1/records/singlepoint/bulkGet'), 'forensic audit forbids server mutations')
                kwargs.update(allow_retries=False, internal_retry=False)
            return super()._request(method, endpoint, **kwargs)

    return ReadOnlyClient


def readonly_client():
    import yaml
    server = yaml.safe_load((ROOT/'qca/resources_losii_ac.yml').read_text())['server']
    require(server['username'] == 'awallace43'
        and server['fractal_uri'] == 'http://128.61.254.96:7777', 'approved read-only deployment required')
    client = readonly_client_class()(server['fractal_uri'], username=server['username'],
        password=server.get('password'), verify=server.get('verify', True), show_motd=False)
    info = client.get_server_information()
    require(info.get('name') == 'CDS QCFractal Server' and info.get('version') == '0.70',
            'approved server identity changed')
    return client


def server_snapshot(client):
    record = client.get_records(5234327, include=['*'])
    require(record.id == 5234327, 'only original completed record may be read')
    summary = dict(record_id=record.id, status=record.status.value,
        history_count=len(record.compute_history),
        history_managers=[h.manager_name for h in record.compute_history])
    return summary, record.extras.get('psi4:native_monomer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    modules = Path(__file__).parent
    checks = load(modules/'phoenix_installed_checks.py', 'checks')
    audit = load(modules/'runtime_audit_cb63305.py', 'audit')
    worker = load(modules/'phoenix_worker_cb63305.py', 'worker')
    bootstrap = load(modules/'phoenix_pilot_cb63305.py', 'bootstrap')
    numeric = load(modules/'validate_numeric.py', 'numeric')
    classifier = load(modules/'runtime_audit.py', 'classifier').cli_role
    allocation = checks.allocation()
    audit.admit_audit_allocation(allocation, producer='grid400')
    require(checks.git(source, 'rev-parse', 'HEAD') == args.commit
        and not checks.git(source, 'status', '--porcelain'), 'exact clean forensic source required')
    run = args.run.resolve()
    require(run == ROOT/('runs/'+PREFIX+'-monomer-0008-forensic001'),
            'separate fixed forensic output namespace required')
    output = run/'payload'
    output.mkdir(exist_ok=False)
    data = CASE/'payload'
    require(not (data/'pilot-outcome.json').exists(), 'incident must retain its missing outcome receipt')
    names = ('started.json', 'record-created.json', 'claim.json', 'manager-delivery.json',
             'server-record-outcome.json', 'expected-runtime.json', 'claimed-input.json',
             'local-result-4722332.zstd', 'dataset-created.json', 'record-submit-attempt.json')
    paths = {name: data/name for name in names}
    paths.update(pilot=INPUT/'pilot.json', worker=WORKER, qualification=QUAL,
                 build=BUILD/'build-identity.json', stderr=CASE/'logs/job-15090188.err')
    seals = {name: checks.digest(path) for name, path in paths.items()}
    require(seals['local-result-4722332.zstd'] == LOCAL_SHA
        and seals['pilot'] == 'debc4bec8923501df62ba8207c26208d863f0ab3ab9f7ae9d0fa6b80f3db4eac'
        and seals['build'] == 'd2b4049c2e5818e462ad3c6ac36f93d6a9623a6a205c701b885887ec2a9f3bfd'
        and seals['worker'] == '8562f746eab9b030b06a9202681b6bee467a028ade2d3b5e6ea4f0d482d17612'
        and seals['qualification'] == 'a9f7179ec4721a79b462a9516639d474f9ca952b7adbe89b4069438b4ed8e8dc',
        'exact original incident receipt hashes required')
    with paths['stderr'].open('rb') as stream:
        stream.seek(max(0, paths['stderr'].stat().st_size-65536))
        require(FAILURE in stream.read().decode(errors='replace'), 'original identity failure absent')
    read = lambda name: json.loads(bounded(data/name))
    started, created, claim, delivery, captured = map(read, names[:5])
    draft_args = argparse.Namespace(producer='grid400', remaining_case=False,
        inputs=INPUT, descriptors=DESCRIPTORS, qualification=QUAL, pilot=INPUT/'pilot.json')
    draft = worker.prepare_draft(draft_args, BUILD/'build-identity.json')
    validate_intents(started, draft, read('dataset-created.json'), read('record-submit-attempt.json'))
    preflight = json.loads(bounded(WORKER))
    transport = load(modules/'dataset_fresh_cb63305.py', 'transport').transport_binding(producer='grid400')
    bootstrap.validate_preflight(preflight, draft, manifest_sha=seals['build'],
        qualification_sha=seals['qualification'], pilot_sha=seals['pilot'],
        worker_sources=worker.source_hashes('grid400'), transport=transport,
        producer='grid400', remaining_case=False)
    require(started['build_identity_sha256'] == seals['build']
        and started['pilot_sha256'] == seals['pilot']
        and started['worker_preflight_sha256'] == seals['worker']
        and started['installed_qualification_sha256'] == seals['qualification']
        and started['runtime_input_sha256'] == draft['runtime_input_sha256']
        and started['descriptor_allocation_sha256'] == draft['descriptor_allocation_sha256']
        and started['transport_binding'] == transport,
        'original start identity chain differs')
    bootstrap.admit_allocation(started['allocation'], execution_profile=draft['execution_profile'],
                               producer='grid400')
    expected = worker.helper('phoenix_worker_probe').runtime_input(draft)
    legacy = worker.helper('phoenix_pilot')
    expected['specification']['protocols'] = legacy.execution_specification(draft['specification'])['protocols']
    require(expected == read('expected-runtime.json'), 'raw runtime reconstruction differs')
    require(started['execution_profile'] == draft['execution_profile']
        and started['execution_protocols'] == expected['specification']['protocols']
        and started['manager_sources'] == legacy.MANAGER_FILES,
        'original execution profile, protocols or manager inventory differs')
    legacy.validate_claim(read('claimed-input.json'), expected, record_id=5234327,
                          compute_tag=started['compute_tag'])
    scratch = audit.scratch_for_outcome(started, expected_monomer='monomer-0008', producer='grid400')
    # Original modules, numeric/FD policies, and execution helper inventory must
    # remain byte-for-byte those of the failed launcher. Only this NEW procedure
    # has its own revision and distinct nonacceptance report.
    expected_sources = set(worker.source_names('grid400')) | {
        'phoenix_pilot_cb63305.py', 'phoenix_pilot_cb63305.sbatch', 'single_record_cb63305.py'}
    require(set(started['operational_sources_sha256']) == expected_sources,
            'complete original execution-source inventory required')
    policies = ('devtools/isapol_native_acceptance/trace_gate.py',
                'devtools/isapol_s66_campaign/validate_numeric.py',
                'devtools/isapol_s66_campaign/runtime_audit.py',
                'devtools/isapol_s66_campaign/runtime_audit_cb63305.py')
    for relative in policies + tuple('devtools/isapol_s66_campaign/'+name for name in expected_sources):
        original = subprocess.check_output(['git', '-C', str(source), 'show', CASE_COMMIT+':'+relative])
        require((source/relative).read_bytes() == original, 'original policy/helper changed: '+relative)
        name = Path(relative).name
        if name in expected_sources:
            require(sha(original) == started['operational_sources_sha256'][name],
                    'original execution helper commitment differs')
    accounting = subprocess.check_output(['/opt/slurm/current/bin/sacct', '-n', '-P', '-j', '15090188',
        '-o', 'JobIDRaw,State,ExitCode'], text=True).splitlines()
    failed_rows = []
    for identifier in ('15090188', '15090188.batch', '15090188.0'):
        require(any(row.strip().split('|')[:3] == [identifier, 'FAILED', '1:0'] for row in accounting),
                'original failed accounting must remain explicit')
        failed_rows.append(dict(job_id=identifier, state='FAILED', exit_code='1:0'))
    manifest = json.loads(bounded(BUILD/'build-identity.json'))
    environment = BUILD/'env'
    checks.verify_runtime(environment, manifest)
    require(manifest['commit'] == started['scientific_build_commit']
        == '1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2', 'original scientific source differs')
    from qcportal.compression import decompress, CompressionEnum
    import zstandard
    local_raw = bounded(paths['local-result-4722332.zstd'], 32*1024**2)
    require(0 < zstandard.frame_content_size(local_raw) <= 256*1024**2, 'bounded local decode required')
    local_result = decompress(local_raw, CompressionEnum.zstd)
    require(local_result.get('success') is True, 'original task did not succeed')
    local = local_result['extras']['psi4:native_monomer']
    client = readonly_client()
    live, server = server_snapshot(client)
    validate_chain(started, created, claim, delivery, captured, live)
    identity = compare_incident_envelopes(local, server)
    request = expected['specification']['keywords']['function_kwargs']['native_monomer']
    for envelope in (local, server):
        require(envelope.get('schema') == 'native-monomer-artifacts/2'
            and envelope['request'] == request
            and envelope['implementation']['sha256'] == draft['producer']['implementation_sha256'],
            'exact original request/producer required')
        construction = envelope['molecule_construction']
        require(construction['input_fix_symmetry'] == read('claimed-input.json')[
            'function_kwargs']['input_data']['molecule'].get('fix_symmetry')
            and construction['requested_point_group'] == construction['effective_point_group'] == 'c1'
            and 0 <= construction['max_abs_coordinate_change_bohr'] <= 1e-10,
            'original ordered coordinates/computational symmetry differ')
    tensors = {role: tensor_bytes(envelope) for role, envelope in (('local', local), ('server', server))}
    require(tensors['local'] == tensors['server'], 'local/server tensor bytes differ')
    numeric_reports = {}
    for role, envelope in (('local', local), ('server', server)):
        with (output/(role+'-native-envelope.json')).open('x') as stream:
            json.dump(envelope, stream, indent=2, allow_nan=False)
        with (output/(role+'-tensors.npz')).open('xb') as stream:
            stream.write(tensors[role])
        try:
            numeric_reports[role] = numeric.validate_numeric(envelope, tensors[role])
        except (ValueError, KeyError, TypeError) as exc:
            numeric_reports[role] = dict(status='FAIL_NUMERIC_COMPONENT', error=str(exc)[:500])
    gate_path = source/policies[0]
    gate = load(gate_path, 'decoded_fd_gate')
    traces = sorted((data/'traces').glob('*.strace'))
    require(bool(traces), 'original trace inventory missing')
    trace_seals = {path.name: checks.digest(path) for path in traces}
    forbidden = [str(ROOT/'gits'), str(ROOT/'qca'), str(ROOT/'data'), str(CASE/'inputs'),
                 str(data), str(INPUT), str(DESCRIPTORS)]
    roles, reports = Counter(), {}
    for path in traces:
        report = gate.check_trace(path.read_text().splitlines(), env=environment,
            base=audit.BASE, manifest_path=BUILD/'build-identity.json', manifest=manifest,
            run=CASE, scratch=scratch, forbidden=forbidden)
        report['trace_sha256'] = trace_seals[path.name]
        if len(report['executions']) != 1:
            report.update(status='failed', role_error='only original owned interpreter may execute')
        else:
            try:
                report['role'] = classifier(report['executions'][0], env=environment, scratch=scratch,
                                            worker_shape=(16, 60.))
                roles[report['role']] += 1
            except ValueError as exc:
                report.update(status='failed', role_error=str(exc))
        reports[path.name] = report
    # Fresh checks now are forensic observations, not a replacement for the
    # original launcher's missing contemporaneous end-of-run checks.
    checks.verify_runtime(environment, manifest)
    live_after, server_after = server_snapshot(client)
    require(live_after == live and server_after == server, 'server evidence changed during analysis')
    require(all(checks.digest(path) == seals[name] for name, path in paths.items())
        and {path.name: checks.digest(path) for path in sorted((data/'traces').glob('*.strace'))} == trace_seals,
        'original evidence changed during analysis')
    require(checks.digest(preflight['harness_path']) == preflight['harness_sha256']
        and all(checks.digest(audit.MANAGER/'lib/python3.13/site-packages/qcfractalcompute'/name) == value
                for name, value in legacy.MANAGER_FILES.items())
        and checks.git(source, 'rev-parse', 'HEAD') == args.commit
        and not checks.git(source, 'status', '--porcelain')
        and worker.prepare_draft(draft_args, BUILD/'build-identity.json') == draft
        and not (data/'pilot-outcome.json').exists() and 'psi4' not in sys.modules,
        'runtime, manager, source or original missing-receipt state changed')
    passed = (all(row['status'] == 'PASS_NUMERIC_TRANSPORT_ONLY' for row in numeric_reports.values())
        and all(row['status'] == 'passed' for row in reports.values()) and roles['molecular_qcschema'] == 1)
    report = dict(schema='native-grid400-forensic-components/1',
        status=('FORENSIC_COMPONENT_CHECKS_PASSED_NOT_MODEL_ACCEPTANCE' if passed
                else 'FORENSIC_COMPONENT_CHECKS_FAILED_NOT_MODEL_ACCEPTANCE'),
        model_accepted=False, original_protocol_passed=False, original_completion_receipt_present=False,
        reconstructed_completion_receipt=False, original_accounting=failed_rows,
        audit_allocation=allocation, audit_commit=args.commit, original_case_commit=CASE_COMMIT,
        scientific_commit=manifest['commit'], record_id=5234327, dataset_id=356, task_id=4722332,
        original_failure=FAILURE, input_sha256=seals, identity=identity,
        tensor_sha256=TENSOR_SHA, numeric=numeric_reports, roles=dict(roles), traces=reports,
        trace_inventory_qualification='Post-run observed trace inventory; original completion inventory absent.',
        file_gate_sha256=checks.digest(gate_path), numeric_validator_sha256=checks.digest(modules/'validate_numeric.py'),
        runtime_rechecked_now=True, original_contemporaneous_end_check_present=False,
        qualification='Separately authorized read-only forensic component analysis only. Original launcher '
        'FAILED and missing completion receipt are NOT repaired or waived. Exact tensor bytes, request and '
        'producer; one measured exact decimal-JSON diagnostic representation incident, no epsilon tolerance. '
        'Original numeric/decoded-FD gates unchanged. Not ordinary pilot protocol PASS, model acceptance, '
        'full C10, independent solver certification, accuracy/parity or whole-set qualification.')
    with (output/'forensic-report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], record_id=5234327, roles=dict(roles),
        numeric={key: value['status'] for key, value in numeric_reports.items()},
        model_accepted=False, original_protocol_passed=False)))
    if not passed:
        raise RuntimeError('forensic component checks failed; result remains unqualified')


if __name__ == '__main__':
    main()
