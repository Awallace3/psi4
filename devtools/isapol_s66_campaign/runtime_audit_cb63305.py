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
    'validate_numeric', 'dataset_fresh_cb63305', 'phoenix_descriptor_bank_cb63305',
    'owned_dataset_cb63305', 'dataset_owned354_cb63305', 'prepare_owned354_baseline_cb63305',
    'envelope_identity', 'bank_campaign', 'bank_dataset', 'bank_audit', 'bank_runtime_gate',
    'bank_wave_runtime'}
_MODULES = {}


def helper(name):
    if name not in HELPERS:
        raise ValueError('closed expanded runtime audit helpers required')
    if name == 'bank_wave_runtime' and name in _MODULES:
        return _MODULES[name]
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_expanded_audit_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    if name == 'bank_wave_runtime':
        _MODULES[name] = module
    return module


def admit_audit_allocation(allocation, *, producer):
    """Enforce this fresh audit's own shape, not the science job's shape."""
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed audit allocation producer required')
    if producer == 'grid400':
        expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
            NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00')
        if any(allocation.get(key) != value for key, value in expected.items()):
            raise ValueError('exact grid400 audit allocation route and4CPU24GiB45min required')


def verify_delivery_identity(local_result, server, saved_witness, *, producer):
    """Independently reproduce the launcher's identity witness, never trust it."""
    if producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed audited delivery producer required')
    if not local_result['success']:
        raise ValueError('independent local/server native envelope comparison failed')
    local = local_result['extras']['psi4:native_monomer']
    if producer == 'grid400':
        identity = helper('envelope_identity').verify_native_envelope_identity(local, server)
        if type(saved_witness) is not dict or saved_witness != identity:
            raise ValueError('independently reproduced JSON envelope identity witness differs')
        return identity
    if local != server:
        raise ValueError('independent local/server native envelope comparison failed')
    return None


def scratch_for_outcome(outcome, *, expected_monomer='monomer-0006', producer='cb63305',
                        remaining_case=False, campaign_scope='default'):
    helper('phoenix_worker_cb63305').validate_campaign_scope(producer, campaign_scope)
    bank = campaign_scope in ('eligible-bank', 'eligible-wave')
    pins = (helper('bank_wave_runtime').pins() if campaign_scope == 'eligible-wave'
            else helper('bank_campaign').pins() if bank else helper('pilot_cb63305').producer_pins(producer))
    job = outcome['allocation']['JobId']
    if (type(job) is not str or re.fullmatch(r'[0-9]+', job) is None
            or type(expected_monomer) is not str
            or re.fullmatch(r'monomer-[0-9]{4}', expected_monomer) is None
            or not 1 <= int(expected_monomer[-4:]) <= 131
            or outcome.get('monomer_id') != expected_monomer):
        raise ValueError('exact expanded-pilot job and reconstructed monomer identity required')
    if producer == 'ed561666' and not remaining_case and expected_monomer != pins['monomer']:
        raise ValueError('fixed fresh cap32 monomer required')
    if bank and (outcome.get('campaign_scope') != campaign_scope
            or outcome.get('producer_profile') != 'grid400'
            or outcome.get('dataset_mode') != ('append' if remaining_case else 'fresh')):
        raise ValueError('explicit eligible bank scratch mode and producer required')
    if producer == 'grid400' and not bank and (
            remaining_case or expected_monomer != pins['monomer']
            or outcome.get('producer_profile') != producer or outcome.get('dataset_mode') != 'fresh'):
        raise ValueError('fixed fresh grid400 monomer and mode required')
    if producer == 'ed561666' and remaining_case and (
            expected_monomer == pins['monomer'] or outcome.get('dataset_mode') != 'append'):
        raise ValueError('selected remaining cap32 monomer required')
    tag = pins['namespace']+expected_monomer+'-'+job
    expected = SCRATCH/tag
    if (outcome.get('scratch_path') != str(expected)
            or outcome.get('compute_tag') != tag):
        raise ValueError('expanded molecular scratch/tag differs from actual case identity')
    return expected


def validate_delivery(outcome, delivery, created, claimed, server, *, owned354=False,
                      producer='cb63305', campaign_scope='default'):
    helper('phoenix_worker_cb63305').validate_campaign_scope(producer, campaign_scope)
    bank = campaign_scope in ('eligible-bank', 'eligible-wave')
    pins = (helper('bank_wave_runtime').pins() if campaign_scope == 'eligible-wave'
            else helper('bank_campaign').pins() if bank else helper('pilot_cb63305').producer_pins(producer))
    if bank and (outcome.get('campaign_scope') != campaign_scope
            or outcome.get('producer_profile') != 'grid400'
            or outcome.get('dataset_mode') != ('append' if owned354 else 'fresh')
            or outcome.get('dataset_name') != pins['dataset_name']
            or outcome.get('specification_name') != pins['specification_name']):
        raise ValueError('explicit eligible bank delivered mode and dataset/specification required')
    if producer == 'grid400' and not bank and (
            owned354 or outcome.get('producer_profile') != producer
            or outcome.get('dataset_mode') != 'fresh' or outcome.get('monomer_id') != pins['monomer']
            or outcome.get('dataset_name') != pins['dataset_name']
            or outcome.get('specification_name') != pins['specification_name']):
        raise ValueError('grid400 delivery must be a fresh singleton, never owned append')
    if producer == 'ed561666' and (
            outcome.get('producer_profile') != producer
            or outcome.get('dataset_mode') != ('append' if owned354 else 'fresh')):
        raise ValueError('selected cap32 delivery/append mode differs')
    if (outcome.get('schema') != 'native-expanded-pilot-outcome/1'
            or outcome.get('status') != 'DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
            or outcome.get('runtime_unchanged') is not True
            or outcome.get('find_existing') is not False or outcome.get('maximum_records') != 1
            or type(outcome.get('dataset_id')) is not int or outcome['dataset_id'] <= 0
            or outcome['dataset_id'] in pins['protected_datasets']
            or type(outcome.get('record_id')) is not int or outcome['record_id'] <= 0
            or outcome['record_id'] in pins['protected_records']):
        raise ValueError('fresh delivered expanded single-record case required')
    if owned354 and not bank:
        target, first_record = 354, 5234316
        if producer == 'ed561666':
            context = helper('owned_dataset_cb63305').context(producer)
            target, first_record = context['dataset_id'], context['first_record']
        if outcome['dataset_id'] != target or outcome['record_id'] == first_record:
            raise ValueError('one fresh remaining record in owned354/selected dataset required')
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


def validate_preservation(data, outcome, draft, append_inputs, specification, *, producer='cb63305'):
    """Independently reduce captured server snapshots; no live writes or manager."""
    guard = helper('owned_dataset_cb63305')
    selected = {} if producer == 'cb63305' else dict(producer=producer)
    accepted = guard.verify_first_acceptance(append_inputs['first_receipt_blobs'], **selected)
    checkpoint = helper('dataset_owned354_cb63305').validate_checkpoint(
        append_inputs['checkpoint_raw'], append_inputs['checkpoint_sha256'], accepted, **selected,
        **(dict(proof_blobs=append_inputs['checkpoint_proof_blobs'])
           if append_inputs.get('checkpoint_proof_blobs') else {}))
    read = lambda name: json.loads((data/name).read_bytes())
    before, entry, record, delivered = (read(name) for name in (
        'owned-before-entry.json', 'owned-after-entry.json',
        'owned-after-record.json', 'owned-after-delivery.json'))
    if (outcome.get('checkpoint_sha256') != append_inputs['checkpoint_sha256']
            or before != checkpoint['snapshot']):
        raise ValueError('actual append differs from sealed owned354 baseline')
    validate_owned = guard.validate_owned_dataset if selected else guard.validate_owned354
    validate_owned(before, authenticated_owner='awallace43',
                   specification=specification, accepted=accepted, **selected)
    guard.validate_new_case(before, draft, reproduced_case=draft, **selected)
    name, identifier = draft['entry']['name'], outcome['record_id']
    guard.assert_append_delta(before, entry, entry_name=name)
    for snapshot in (record, delivered):
        guard.assert_append_delta(before, snapshot, entry_name=name, record_id=identifier)
    for snapshot in (entry, record, delivered):
        guard.base().validate_stored_entry(snapshot['entries'][name], draft['entry'])
    guard.base().validate_fresh_record(record['records'][str(identifier)],
        tag=outcome['compute_tag'], specification=specification, entry=draft['entry'])
    validate_owned(delivered, authenticated_owner='awallace43',
                   specification=specification, accepted=accepted, **selected)
    finished = delivered['records'][str(identifier)]
    if (finished['status'] != 'complete' or len(finished['history']) != 1
            or finished['history'][0]['manager_name'] != outcome['manager']):
        raise ValueError('captured delivered record differs from actual owned singleton manager')
    envelope = read('server-native-envelope.json')
    envelope_sha = guard.base().digest(envelope)
    tensor_raw = (data/'server-tensors.npz').read_bytes()
    tensor_sha = hashlib.sha256(tensor_raw).hexdigest()
    if (finished.get('native_envelope_sha256') != envelope_sha
            or finished.get('tensor_sha256') != tensor_sha
            or envelope['artifacts']['tensors.npz']['sha256'] != tensor_sha
            or outcome.get('tensors_sha256') != tensor_sha
            or outcome.get('tensor_bytes') != len(tensor_raw)):
        raise ValueError('captured delivered record differs from audited envelope/tensor bytes')
    intent_sha = None
    if selected:
        intent_sha = helper('dataset_owned354_cb63305').verify_dispatch_intent(
            read('dispatch-intent.json'), checkpoint_sha256=append_inputs['checkpoint_sha256'],
            snapshot_sha256=checkpoint['snapshot_sha256'],
            case_sha256=outcome['pilot_sha256'], monomer_id=draft['monomer_id'],
            entry_name=draft['entry']['name'], compute_tag=outcome['compute_tag'])
        if intent_sha != outcome.get('dispatch_intent_sha256'):
            raise ValueError('audited owned dispatch intent differs from the delivered outcome')
    result = dict(status=('PASS_CAPTURED_OWNED_DATASET_PRESERVATION' if selected
                        else 'PASS_CAPTURED_OWNED354_PRESERVATION'),
        checkpoint_sha256=append_inputs['checkpoint_sha256'],
        before_sha256=guard.base().digest(before), after_sha256=guard.base().digest(delivered),
        audited_native_envelope_sha256=envelope_sha, audited_tensor_sha256=tensor_sha,
        preserved_records=len(before['records']),
        qualification='Reduction of captured server snapshots, not an atomic lock or fresh live query.')
    if intent_sha is not None:
        result['dispatch_intent_sha256'] = intent_sha
    return result


def advance_audit_checkpoint(*, raw_audit, data, case_path, append_inputs, output, operational_commit):
    """After a passing audit, confirm live state and emit one immutable checkpoint."""
    audit = json.loads(raw_audit)
    if (audit.get('status') != 'PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE'
            or audit.get('producer_profile') != 'ed561666' or audit.get('dataset_mode') != 'append'):
        raise ValueError('only a passing selected append audit may advance a checkpoint')
    baseline = helper('prepare_owned354_baseline_cb63305')
    append = helper('dataset_owned354_cb63305')
    from qcportal.singlepoint import QCSpecification
    seed = json.loads(append_inputs['first_receipt_blobs']['pilot'])
    specification = QCSpecification(**helper('phoenix_pilot').execution_specification(seed['specification']))
    receipts = dict(audit=raw_audit, outcome=(data/'pilot-outcome.json').read_bytes(),
                    case=case_path.read_bytes(), envelope=(data/'server-native-envelope.json').read_bytes())
    captured = {role: json.loads((data/name).read_bytes()) for role, name in dict(
        before='owned-before-entry.json', entry='owned-after-entry.json',
        record='owned-after-record.json', delivered='owned-after-delivery.json').items()}
    client, transport = baseline.approved_client(producer='ed561666')
    checkpoint = baseline.advance_baseline(client, blobs=append_inputs['first_receipt_blobs'],
        authenticated_owner=baseline.OWNER, specification=specification,
        previous_raw=append_inputs['checkpoint_raw'], previous_sha256=append_inputs['checkpoint_sha256'],
        receipts=receipts, expected_audit_sha256=hashlib.sha256(raw_audit).hexdigest(),
        captured=captured, tensor_raw=(data/'server-tensors.npz').read_bytes(),
        proof_blobs=append_inputs.get('checkpoint_proof_blobs'))
    links = append.store_checkpoint_proofs(receipts)
    if links != checkpoint['ledger'][-1]['receipts_sha256']:
        raise ValueError('stored proof commitments differ from the passing audit')
    checkpoint.update(operational_commit=operational_commit, transport_binding=transport,
        advancement_source_sha256=hashlib.sha256(Path(__file__).with_name(
            'prepare_owned354_baseline_cb63305.py').read_bytes()).hexdigest())
    path = output/'next-checkpoint.json'
    with path.open('x') as stream:
        json.dump(checkpoint, stream, indent=2, allow_nan=False)
    return dict(status='PASS_AUDITED_CHECKPOINT_ADVANCE', generation=checkpoint['generation'],
        dataset_id=355, audited_records=checkpoint['audited_records'],
        checkpoint_path=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'case', 'worker-preflight', 'qualification', 'inputs', 'descriptors', 'pilot', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666', 'grid400'), default='cb63305')
    parser.add_argument('--remaining-case', action='store_true')
    parser.add_argument('--campaign-scope', choices=('default', 'eligible-bank', 'eligible-wave'), default='default')
    for name in ('wave-plan', 'wave-registration', 'wave-science-barrier'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('bank-context', 'bank-checkpoint'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--owned-checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--checkpoint-proofs', type=Path)
    parser.add_argument('--advance-checkpoint', action='store_true')
    parser.add_argument('--advance-bank-checkpoint', action='store_true')
    for role in ('audit', 'outcome', 'pilot', 'envelope'):
        parser.add_argument('--accepted-'+role, type=Path)
    args = parser.parse_args()
    worker = helper('phoenix_worker_cb63305')
    scope = worker.selected_scope(args)
    bank = scope == 'eligible-bank'
    wave_mode = scope == 'eligible-wave'
    wave = helper('bank_wave_runtime') if wave_mode else None
    scope_args = dict(campaign_scope=scope) if bank or wave_mode else {}
    if wave_mode:
        wave.validate_args(args, registered=True)
    if bank:
        helper('bank_campaign').authority()
    if args.advance_bank_checkpoint and not bank:
        raise ValueError('bank checkpoint publication requires explicit eligible-bank scope')
    if bank and args.advance_checkpoint:
        raise ValueError('historical checkpoint publication cannot be used for the new bank')
    if args.advance_checkpoint and (args.producer != 'ed561666' or not args.remaining_case):
        raise ValueError('checkpoint advancement is only supported for selected cap32 append audits')
    bootstrap = helper('phoenix_pilot_cb63305')
    append_inputs = bootstrap.load_append_inputs(args)
    binding = append_inputs if wave_mode else None
    producer = args.producer
    checks = helper('phoenix_installed_checks')
    allocation = checks.allocation()
    admit_audit_allocation(allocation, producer=producer)
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
    if wave_mode:
        wave.validate_paths(args, role='audit')
    data = case/'payload'
    read = lambda name: json.loads((data/name).read_bytes())
    outcome, delivery, created, claimed, server = (
        read(name) for name in ('pilot-outcome.json', 'manager-delivery.json',
                               'record-created.json', 'claim.json', 'server-record-outcome.json'))
    validate_delivery(outcome, delivery, created, claimed, server, owned354=args.remaining_case,
                      producer=producer, **scope_args)
    manifest_path = build/'build-identity.json'
    manifest = json.loads(manifest_path.read_bytes())
    worker, legacy = helper('phoenix_worker_cb63305'), helper('phoenix_pilot')
    draft = worker.prepare_draft(args, manifest_path)
    profile = draft.get('execution_profile')
    scratch = scratch_for_outcome(outcome, expected_monomer=draft['monomer_id'], producer=producer,
                                  remaining_case=args.remaining_case, **scope_args)
    bootstrap.admit_allocation(outcome['allocation'], execution_profile=profile,
                               **(dict(producer=producer, **scope_args) if producer == 'grid400' else {}))
    if profile is not None and outcome.get('execution_profile') != profile:
        raise ValueError('actual molecular execution profile differs from reconstructed case')
    worker_shape = (profile['cores'], profile['science_memory_GiB']) if profile else (4, 20)
    preflight = json.loads(args.worker_preflight.read_bytes())
    transport = (wave.helper('bank_wave_transport').transport_binding() if wave_mode
                 else helper('bank_dataset').transport_binding() if bank
                 else helper('dataset_fresh_cb63305').transport_binding(producer=producer))
    bootstrap.validate_preflight(preflight, draft, manifest_sha=checks.digest(manifest_path),
        qualification_sha=checks.digest(args.qualification), pilot_sha=checks.digest(args.pilot),
        worker_sources=worker.source_hashes(producer, remaining_case=args.remaining_case, **scope_args),
        transport=transport, producer=producer, remaining_case=args.remaining_case, **scope_args)
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
    submitted = None if wave_mode else read('record-submit-attempt.json')
    sources = set(worker.source_names(producer, remaining_case=args.remaining_case, **scope_args)) | {
        'phoenix_pilot_cb63305.py', 'phoenix_pilot_cb63305.sbatch', 'single_record_cb63305.py'}
    preservation = None
    if wave_mode:
        preservation = wave.audit_preservation(data, outcome, draft, binding,
            wave.helper('audited_bank').normalize_specification(draft['specification']))
    elif bank and args.remaining_case:
        preservation = helper('bank_audit').validate_preservation(data, outcome, draft, append_inputs,
            helper('bank_audit').helper('audited_bank').normalize_specification(draft['specification']))
    elif args.remaining_case:
        from qcportal.singlepoint import QCSpecification
        sources |= {'dataset_owned354_cb63305.py', 'owned_dataset_cb63305.py', 'dataset_guard.py'}
        preservation = validate_preservation(data, outcome, draft, append_inputs,
            QCSpecification(**legacy.execution_specification(draft['specification'])),
            **({} if producer == 'cb63305' else dict(producer=producer)))
    elif read('dataset-created.json') != dict(dataset_id=outcome['dataset_id'], name=draft['dataset_name']):
        raise ValueError('fresh dataset intent differs')
    if bank and not args.remaining_case:
        helper('bank_audit').validate_initial_capture(data, outcome, draft)
    if not wave_mode and (submitted['dataset_id'] != outcome['dataset_id']
            or submitted['entry'] != draft['entry']['name']
            or submitted['specification'] != draft['specification_name']
            or submitted['compute_tag'] != outcome['compute_tag']
            or submitted['find_existing'] is not False or submitted['transport_retries'] is not False):
        raise ValueError('fresh dataset intent or complete execution-source inventory differs')
    if set(outcome['operational_sources_sha256']) != sources:
        raise ValueError('complete execution-source inventory differs')
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
    identity = verify_delivery_identity(local, envelope,
        read('envelope-identity.json') if producer == 'grid400' else None, producer=producer)
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
    classifier = helper('bank_runtime_gate' if bank or wave_mode else 'runtime_audit').cli_role
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
                report['role'] = classifier(report['executions'][0], env=environment, scratch=scratch,
                    **(dict(execution_profile=profile) if bank or wave_mode else dict(worker_shape=worker_shape)))
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
            or worker.prepare_draft(args, manifest_path) != draft
            or bootstrap.load_append_inputs(args) != append_inputs):
        raise ValueError('runtime, manager, audit source or immutable raw receipts changed')
    passed = all(r['status'] == 'passed' for r in reports.values()) and roles['molecular_qcschema'] == 1
    output = run/'payload'
    output.mkdir(exist_ok=False)
    if wave_mode:
        wave.helper('bank_wave_campaign').revalidate_binding(binding)
    verdict = ('PASS_SELECTED_WAVE_CASE_SINGLETON_EVIDENCE' if wave_mode
               else 'PASS_SELECTED_BANK_CASE_SINGLETON_EVIDENCE'
               if (producer == 'ed561666' or bank) and args.remaining_case
               else 'PASS_SELECTED_FRESH_SINGLETON_EVIDENCE' if producer in ('ed561666', 'grid400')
               else 'PASS_BANK_CASE_SINGLETON_EVIDENCE' if args.remaining_case
               else 'PASS_EXPANDED_SINGLE_PILOT_EVIDENCE')
    report = dict(schema='native-expanded-pilot-audit/1',
        status=verdict if passed else 'FAIL_EXPANDED_RUNTIME_INPUT_AUDIT',
        allocation=allocation, monomer_id=draft['monomer_id'], case_job=job,
        dataset_id=outcome['dataset_id'], record_id=outcome['record_id'],
        scientific_build_commit=manifest['commit'], case_operational_commit=outcome['operational_commit'],
        audit_commit=args.commit, case_outcome_sha256=checks.digest(data/'pilot-outcome.json'),
        pilot_sha256=checks.digest(args.pilot), file_gate_sha256=checks.digest(gate_path),
        numeric=numeric, envelope_identity=identity, roles=dict(roles), traces=reports, runtime_unchanged=True,
        execution_profile=profile, captured_preservation=preservation,
        qualification='One fresh request3 molecular pilot execution/export/decoded-FD evidence only; '
                      'ordinary OS trust, not hostile-host/TOCTOU or atomic-server-lock proof. '
                      'No full-C10, independent solver, covariance, accuracy/parity or whole-set claim.')
    if producer in ('ed561666', 'grid400'):
        report.update(producer_profile=producer, dataset_mode='append' if args.remaining_case else 'fresh')
    if bank:
        report['campaign_scope'] = scope
    if wave_mode:
        report.update(wave.started_fields(binding), campaign=draft['campaign'],
            compute_tag=outcome['compute_tag'],
            worker_preflight_sha256=checks.digest(args.worker_preflight),
            science_barrier_sha256=binding['science_barrier_sha256'])
    if wave_mode:
        wave.write_json_once(output/'audit-report.json', report)
    else:
        with (output/'audit-report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], record_id=outcome['record_id'], roles=dict(roles))))
    if not passed:
        raise RuntimeError('expanded molecular runtime audit failed; do not advance models')
    if args.advance_bank_checkpoint:
        advanced = helper('bank_audit').advance_audit_checkpoint(args=args,
            raw_audit=(output/'audit-report.json').read_bytes(), data=data, draft=draft,
            append_inputs=append_inputs, output=output, source=source)
        checks.verify_runtime(environment, manifest)
        if checks.git(source, 'rev-parse', 'HEAD') != args.commit or checks.git(source, 'status', '--porcelain'):
            raise ValueError('pinned audit source changed during bank checkpoint publication')
        print(json.dumps(advanced))
    if args.advance_checkpoint:
        relative = 'devtools/isapol_s66_campaign/prepare_owned354_baseline_cb63305.py'
        expected = subprocess.check_output(['git', '-C', str(source), 'show', args.commit+':'+relative])
        if (source/relative).read_bytes() != expected:
            raise ValueError('tracked read-only advancement helper differs from the pinned audit source')
        advanced = advance_audit_checkpoint(raw_audit=(output/'audit-report.json').read_bytes(),
            data=data, case_path=args.pilot, append_inputs=append_inputs,
            output=output, operational_commit=args.commit)
        checks.verify_runtime(environment, manifest)
        if checks.git(source, 'rev-parse', 'HEAD') != args.commit or checks.git(source, 'status', '--porcelain'):
            raise ValueError('pinned audit source changed during read-only checkpoint advancement')
        print(json.dumps(advanced))


if __name__ == '__main__':
    main()
