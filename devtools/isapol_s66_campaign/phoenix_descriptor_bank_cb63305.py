"""Sealed all131 installed descriptor allocation; no SCF or QCArchive writes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

BASE = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt')
RUNS = Path('/storage/project/r-cs207-0/awallace43/runs')
SCRATCH = Path('/storage/scratch1/3/awallace43')
OP_FILES = (
    'phoenix_descriptor_bank_cb63305.py', 'phoenix_descriptor_bank_cb63305.sbatch',
    'descriptor_probe_cb63305.py', 'descriptor_probe.py', 'descriptor_admission_cb63305.py',
    'descriptor_scratch_cb63305.py', 'prepare_descriptor_bank_cb63305.py',
    'propose_descriptor_resources_cb63305.py',
    'term_wait.sh',
    'qualified_cb63305.py', 'resource_plan_cb63305.py', 'phoenix_installed_checks.py',
)
HELPERS = {
    'phoenix_installed_checks', 'qualified_cb63305', 'resource_plan_cb63305',
    'qualified_ed561666', 'resource_plan_ed561666',
    'prepare_descriptor_bank_cb63305', 'descriptor_admission_cb63305', 'descriptor_scratch_cb63305',
}


def helper(name):
    if name not in HELPERS:
        raise ValueError('closed descriptor allocation helpers required')
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_descriptor_allocation_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def read_bounded(path, maximum=16*1024**2):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum+1)
    if not raw or len(raw) > maximum:
        raise ValueError('bounded nonempty receipt/input required')
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json_once(path, data):
    """Publish a complete JSON file without replacing an existing artifact."""
    temporary = path.with_name('.'+path.name+'.partial')
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def operation_hashes(*, producer='cb63305'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed descriptor source inventory producer required')
    files = OP_FILES+(
        ('qualified_ed561666.py', 'resource_plan_ed561666.py', 'prepare_ed561666_inputs.py')
        if producer == 'ed561666' else ())
    return {name: digest(Path(__file__).with_name(name).read_bytes()) for name in files}


def policy_binding(source, manifest, planner):
    if manifest['commit'] != planner.SCIENTIFIC_COMMIT:
        raise ValueError('exact selected scientific producer required')
    observed = {}
    for name, expected in planner.FILES.items():
        raw = subprocess.check_output(['git', '-C', str(source), 'show',
                                       manifest['commit']+':'+name])
        observed[name] = digest(raw)
        if observed[name] != expected:
            raise ValueError('scientific source policy binding differs: '+name)
    return observed


def admit_route(allocation):
    """Enforce the effective allocation, not merely the submission template."""
    expected = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                    NumCPUs='16', NumNodes='1', TimeLimit='00:45:00')
    if any(str(allocation.get(key)) != value for key, value in expected.items()):
        raise ValueError('effective descriptor route differs from the fixed embers allocation')
    return dict(account=expected['Account'], qos=expected['QOS'], partition=expected['Partition'],
                cpus=16, nodes=1, time_limit=expected['TimeLimit'])


def quota_snapshot(scratch, payload, guard):
    if scratch.stat().st_dev != SCRATCH.stat().st_dev:
        raise ValueError('private descriptor scratch is on another filesystem')
    results = {}
    for role, flag, owner in (('user', '-u', os.getuid()), ('group', '-g', scratch.stat().st_gid)):
        query = subprocess.run(['/usr/bin/lfs', 'quota', flag, str(owner), str(SCRATCH)],
                               capture_output=True, text=True, timeout=45)
        if query.returncode or len(query.stdout) > 65536:
            raise RuntimeError('descriptor scratch quota query failed: '+query.stderr[:512])
        results[role] = guard.parse_quota(query.stdout, str(SCRATCH))
        with (payload/('scratch-'+role+'-quota.txt')).open('x') as stream:
            stream.write(query.stdout)
    # Check a real, fresh writable file as well as quota/free-space snapshots.
    probe = scratch/'write-probe'
    with probe.open('xb') as stream:
        stream.write(b'descriptor-only scratch check\n')
        stream.flush()
        os.fsync(stream.fileno())
    probe.unlink()
    return guard.admit_scratch(free_bytes=shutil.disk_usage(scratch).free,
        free_files=os.statvfs(scratch).f_favail,
        user_quota=results['user'], group_quota=results['group'],
        required_bytes=2*1024**3, required_files=4096)


def child_environment(environment, scratch):
    """No inherited credentials, manager configuration or user Python paths."""
    env = {key: os.environ[key] for key in (
        'USER', 'SLURM_JOB_ID', 'SLURM_CLUSTER_NAME', 'SLURMD_NODENAME', 'SLURM_CPUS_PER_TASK')
        if key in os.environ}
    env.update(PATH=f'{environment}/bin:{BASE}/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH=f'{environment}/lib:{BASE}/lib', HOME=str(scratch/'home'),
        TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        LANG='C.UTF-8')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'qualification', 'bank', 'resources', 'inventory', 'entries',
                 'declarations', 'input-validation', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--expected-bank-sha256', required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666'), default='cb63305')
    args = parser.parse_args()
    if (re.fullmatch('[0-9a-f]{40}', args.commit) is None
            or re.fullmatch('[0-9a-f]{64}', args.expected_bank_sha256) is None):
        raise ValueError('exact operational commit and prepared bank hash required')
    checks = helper('phoenix_installed_checks')
    allocation = checks.allocation()
    admit_route(allocation)
    source = Path(__file__).resolve().parents[2]
    if checks.git(source, 'rev-parse', 'HEAD') != args.commit or checks.git(source, 'status', '--porcelain'):
        raise RuntimeError('exact clean immutable operational checkout required')
    build, run = args.build.resolve(), args.run.resolve()
    if (not build.is_relative_to(RUNS) or not run.is_relative_to(RUNS)
            or not run.name.startswith('ccasp-native-s66-')
            or build.is_relative_to(run) or run.is_relative_to(build)):
        raise RuntimeError('disjoint private build and descriptor run required')
    profile, planner = helper('qualified_'+args.producer), helper('resource_plan_'+args.producer)
    # Preserve the historical default interface; the new path is explicit.
    selection = {} if args.producer == 'cb63305' else dict(producer=args.producer)
    prepare, admission = helper('prepare_descriptor_bank_cb63305'), helper('descriptor_admission_cb63305')
    scratch_guard = helper('descriptor_scratch_cb63305')
    paths = dict(build_identity=build/'build-identity.json',
        installed_qualification=args.qualification,
        imported_identity=args.qualification.with_name('imported-identity.json'),
        junit=args.qualification.with_name('junit.xml'),
        inventory=args.inventory, entries=args.entries, declarations=args.declarations,
        input_validation=args.input_validation, resources=args.resources, bank=args.bank)
    if len({p.resolve() for p in paths.values()}) != len(paths):
        raise ValueError('distinct immutable receipt and input roles required')
    blobs = {key: read_bounded(path) for key, path in paths.items()}
    receipt_blobs = {key: blobs[key] for key in profile.RECEIPT_SHA256}
    producer = profile.verify_receipts(receipt_blobs)
    manifest, identity = (json.loads(blobs[key]) for key in ('build_identity', 'imported_identity'))
    if (Path(manifest['installed_package']).resolve() != build/'env/lib/python3.13/site-packages/psi4'
            or digest(blobs['bank']) != args.expected_bank_sha256):
        raise ValueError('prepared bank or private build path differs')
    bank = json.loads(blobs['bank'])
    regenerated = prepare.prepare_bank({key: blobs[key] for key in prepare.INPUT_SHA256},
                                      receipt_blobs, blobs['resources'], **selection)
    if bank != regenerated:
        raise ValueError('prepared bank does not reproduce from exact immutable raw inputs')
    changes = checks.git(source, 'diff', '--name-only', manifest['commit'], args.commit).splitlines()
    if any(not name.startswith('devtools/isapol_s66_campaign/')
           and name not in checks.TEST_ONLY_CHANGES for name in changes):
        raise ValueError('descriptor operations change scientific/build code')
    policy = policy_binding(source, manifest, planner)
    # Parent arithmetic uses the same verified pure source, without importing
    # Psi4. Operational checkout staging must include this fixed source file.
    planner.shared_pfit_plan()
    memory = admission.admit_memory(bank, allocation, nonnumeric_headroom_bytes=16*1024**3)
    operations = operation_hashes(**selection)
    environment = build/'env'
    checks.verify_runtime(environment, manifest)
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    scratch = SCRATCH/('ccasp-expanded-descriptors-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(mode=0o700, exist_ok=False)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir()
    disk = quota_snapshot(scratch, payload, scratch_guard)
    env = child_environment(environment, scratch)
    receipts = {str(p.resolve()): digest(blobs[key]) for key, p in paths.items()}
    write_json_once(payload/'started.json', dict(status='STARTED_DESCRIPTOR_ALLOCATION_ONLY',
        allocation=allocation, bank_sha256=args.expected_bank_sha256,
        scientific_commit=manifest['commit'], operational_commit=args.commit,
        producer=producer, policy_id=planner.POLICY, expected_monomers=131,
        memory_contract=memory, scratch_snapshot=disk, source_policy_sha256=policy,
        operational_sources_sha256=operations, input_receipts_sha256=receipts))
    reports = []
    try:
        for row in regenerated['monomers']:
            monomer = row['monomer_id']
            runtime = json.dumps(row['runtime_input'], sort_keys=True, allow_nan=False).encode()
            if digest(runtime) != row['runtime_input_sha256']:
                raise ValueError('canonical runtime input hash differs')
            runtime_path, report_path = (payload/(monomer+suffix) for suffix in (
                '-runtime.json', '-descriptors.json'))
            with runtime_path.open('xb') as stream:
                stream.write(runtime)
            command = checks.installed_python_command(environment/'bin/python', scratch/'cache',
                str(Path(__file__).with_name('descriptor_probe_cb63305.py')),
                '--input', str(runtime_path), '--output', str(report_path),
                '--memory-bytes', str(row['intended_memory_bytes']), '--monomer-id', monomer,
                '--producer', args.producer)
            with (payload/(monomer+'.stdout')).open('x') as out, (payload/(monomer+'.stderr')).open('x') as err:
                subprocess.run(command, cwd=scratch/'cwd', env=env, stdin=subprocess.DEVNULL,
                               stdout=out, stderr=err, check=True)
            raw = read_bounded(report_path)
            report = json.loads(raw)
            child_sources = ('descriptor_probe_cb63305.py', 'descriptor_probe.py',
                             'resource_plan_cb63305.py')
            if args.producer == 'ed561666':
                child_sources += ('resource_plan_ed561666.py',)
            result = admission.verify_descriptor(report, row, identity, policy=policy,
                operational_sources_sha256={key: operations[key] for key in child_sources},
                runtime_sha256=digest(runtime), **selection)
            reports.append(dict(result, descriptor_report=report_path.name,
                                descriptor_report_sha256=digest(raw), runtime_input_sha256=digest(runtime)))
        status = admission.verify_coverage(reports, {r['monomer_id'] for r in regenerated['monomers']})
        checks.verify_runtime(environment, manifest)
        if (any(checks.digest(Path(path)) != expected for path, expected in receipts.items())
                or checks.git(source, 'rev-parse', 'HEAD') != args.commit
                or checks.git(source, 'status', '--porcelain') or operation_hashes(**selection) != operations):
            raise ValueError('source, input or runtime receipts changed during descriptor allocation')
        write_json_once(payload/'descriptor-allocation.json', dict(
            schema='native-campaign-descriptor-allocation/2', status=status,
            allocation=allocation, scientific_build_commit=manifest['commit'],
            operational_commit=args.commit, policy_id=planner.POLICY,
            bank_sha256=args.expected_bank_sha256, producer=producer,
            source_policy_sha256=policy, operational_sources_sha256=operations,
            input_receipts_sha256=receipts, memory_contract=memory, scratch_snapshot=disk,
            runtime_unchanged=True, reports=reports,
            qualification='Actual installed basis descriptors and predicted native ledger only. '
                          'No SCF, ISA, response/fit, QCArchive writes, molecular results, '
                          'accuracy/covariance proof, total RSS or SCF scratch admission.'))
    except BaseException as error:
        write_json_once(payload/'descriptor-allocation-failure.json', dict(
            status='FAILED_DESCRIPTOR_ALLOCATION', completed_descriptor_reports=len(reports),
            reports=reports, error=str(error)[:1024],
            qualification='Partial or failed descriptor allocation is never all-bank admission.'))
        raise
    print(json.dumps(dict(status=status, monomers=len(reports),
                          report=str(payload/'descriptor-allocation.json'))))


if __name__ == '__main__':
    main()
