"""Execute one fresh installed native profile under independent Phoenix tracing."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_trace_gate', HERE/'trace_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def allocation():
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME') != 'phoenix-slurm'
            or os.environ.get('SLURMD_NODENAME', '').split('.')[0] != host
            or re.fullmatch(r'atl\d+(?:-\d+){5}', host) is None):
        raise RuntimeError('native science requires a Phoenix compute allocation')
    raw = subprocess.check_output([
        '/opt/slurm/current/bin/scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True)
    fields = dict(item.split('=', 1) for item in raw.split() if '=' in item)
    if (fields.get('QOS') != 'embers' or fields.get('Account') != 'gts-cs207-chemx'
            or fields.get('JobState') != 'RUNNING' or fields.get('BatchHost') != host
            or fields.get('UserId', '').split('(')[0] != 'awallace43'):
        raise RuntimeError('effective allocation is not the authorized running free-embers science job')
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--case', choices=('water-smoke', 'water-rank2', 'benzene'), required=True)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--geometry-sha256')
    args = parser.parse_args()
    allocated = allocation()
    build, run, base = (p.resolve() for p in (args.build, args.run, args.dependencies))
    prefix = Path('/storage/project/r-cs207-0/awallace43/runs').resolve()
    if (not build.is_relative_to(prefix) or not run.is_relative_to(prefix)
            or not build.name.startswith('build-')
            or base != Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt').resolve()):
        raise RuntimeError('unexpected private build/run/dependency root')
    manifest_path = build/'build-identity.json'
    manifest_before = gate.digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((build/'build-record.json').read_text())
    if (manifest['commit'] != args.commit or manifest['tree'] != args.tree
            or receipt['status'] != 'complete'
            or receipt['build_identity_sha256'] != manifest_before
            or not manifest['clean_source'] or not manifest['fresh_build']):
        raise RuntimeError('fresh matching build identity required')
    source = HERE.parents[1]
    for git_args, expected in (
            (['rev-parse', 'HEAD'], args.commit), (['rev-parse', 'HEAD^{tree}'], args.tree),
            (['status', '--porcelain'], '')):
        if subprocess.check_output(['git', *git_args], cwd=source, text=True).strip() != expected:
            raise RuntimeError('launcher source does not match the clean built snapshot')
    for name in ('phoenix_run.py', 'trace_gate.py'):
        built = Path(manifest['source_snapshot'])/'devtools/isapol_native_acceptance'/name
        if gate.digest(HERE/name) != gate.digest(built):
            raise RuntimeError('operational launcher differs from built source: '+name)
    # Native manifest existence alone is not successful SLURM completion.
    build_job = str(receipt['job'])
    accounting = subprocess.check_output([
        '/opt/slurm/current/bin/sacct', '-n', '-P', '-j', build_job,
        '-o', 'JobIDRaw,State,ExitCode'], text=True)
    rows = [line.strip().split('|') for line in accounting.splitlines() if line.strip()]
    for job in (build_job, build_job+'.batch'):
        if not any(r[:3] == [job, 'COMPLETED', '0:0'] for r in rows):
            raise RuntimeError('build allocation/batch not both COMPLETED0:0: '+job)
    env_root = build/'env'
    if Path(manifest['installed_package']).resolve() != env_root/'lib/python3.13/site-packages/psi4':
        raise RuntimeError('foreign installed package root')
    # Verify the sealed private and ordinary external dependencies before work.
    for name, expected in manifest['runtime_files'].items():
        path = env_root/name
        resolved = path.resolve()
        if ((not resolved.is_relative_to(env_root)
             and str(resolved) not in manifest['external_runtime_files'])
                or gate.digest(path) != expected):
            raise RuntimeError('private runtime changed: '+name)
    for name, expected in manifest['external_runtime_files'].items():
        if gate.digest(name) != expected:
            raise RuntimeError('external runtime changed: '+name)
    geometry = None
    if args.case == 'benzene':
        if (args.geometry is None or args.geometry_sha256 is None
                or gate.digest(args.geometry) != args.geometry_sha256):
            raise RuntimeError('exact attributed geometry input required')
        geometry = args.geometry.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run/'cwd').mkdir()
    (run/'empty-import-cache').mkdir()
    scratch = Path('/storage/scratch1/3/awallace43')/('ccasp-native-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(exist_ok=False)
    # This run owns the scratch directory; only its own fresh files can be read.
    # Pass only this newly created job-specific root to the decoded-fd gate.
    scratch_root = scratch.resolve()
    if os.statvfs(scratch_root).f_bavail*os.statvfs(scratch_root).f_frsize < 400*1024**3:
        raise RuntimeError('less than400GiB scratch free')
    forbidden = [
        Path('/storage/project/r-cs207-0/awallace43/gits').resolve(),
        Path(manifest['source_snapshot']).resolve(),
        env_root/'lib/python3.13/site-packages/psi4/tests',
        env_root/'share/psi4/samples',
    ]
    profile = 'smoke' if args.case == 'water-smoke' else 'rank2'
    module = ('psi4.driver.procrouting.isapol_benzene_acceptance'
              if args.case == 'benzene' else 'psi4.driver.procrouting.isapol_acceptance')
    command = [str(env_root/'bin/python'), '-I', '-B', '-X',
               'pycache_prefix='+str(run/'empty-import-cache'), '-m', module,
               '--build-manifest', str(manifest_path), '--output', str(run/'output')]
    inputs = {}
    if geometry is not None:
        inputs[str(geometry)] = args.geometry_sha256
        command += ['--geometry', str(geometry), '--geometry-sha256', args.geometry_sha256]
    else:
        command += ['--profile', profile]
    for path in forbidden:
        command += ['--forbid-read-prefix', str(path)]
    trace = run/'os-trace.log'
    traced = ['/usr/bin/strace', '-f', '-qq', '-yy', '-s', '0',
              '-e', 'trace=file,process', '-o', str(trace), *command]
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PSIPATH', 'PSIDATADIR', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(
        PATH=f'{env_root}/bin:{base}/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH=f'{env_root}/lib:{base}/lib',
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(scratch_root), SCRATCH=str(scratch_root), PSI_SCRATCH=str(scratch_root),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    started = time.time()
    (run/'started.json').write_text(json.dumps(dict(
        command=traced, commit=args.commit, tree=args.tree, allocation=allocated,
        geometry_sha256=args.geometry_sha256, started=started), indent=2))
    with (run/'stdout.log').open('w') as stream:
        result = subprocess.run(traced, cwd=run/'cwd', env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
    try:
        checked = gate.check_trace(
            trace.read_text().splitlines(), env=env_root, base=base,
            manifest_path=manifest_path, manifest=manifest, run=run,
            forbidden=forbidden, allowed_inputs=inputs, scratch=scratch_root)
    except (ValueError, OSError) as error:
        checked = dict(status='failed', error=str(error))
    record_path = run/'output/result.json'
    scientific = json.loads(record_path.read_text()) if record_path.exists() else {}
    errors = []
    for file, key in [('tensors.npz', 'tensors_sha256')]+(
            [('angular.json.gz', 'angular_sha256')] if profile == 'rank2' else []):
        path = run/'output'/file
        if not path.is_file() or gate.digest(path) != scientific.get(key):
            errors.append('missing/altered '+file)
    if profile == 'rank2' and not errors:
        with gzip.open(run/'output/angular.json.gz', 'rt') as stream:
            if json.load(stream).get('schema') != 'native-recoupled-dispersion/1':
                errors.append('wrong angular schema')
    expected_schema = 'installed-native-isa-'+('benzene' if geometry is not None else 'water')+'/1'
    passed = (
        result.returncode == 0 and checked['status'] == 'passed' and not errors
        and scientific.get('schema') == expected_schema and scientific.get('status') == 'complete'
        and scientific.get('source_commit') == args.commit and scientific.get('source_tree') == args.tree
        and scientific.get('profile') == profile and scientific.get('child_executables') == []
        and gate.digest(manifest_path) == manifest_before)
    if geometry is not None:
        passed = passed and scientific.get('geometry_sha256') == args.geometry_sha256
    acceptance = dict(
        schema='phoenix-installed-native-acceptance/1',
        status='passed' if passed else 'failed', case=args.case,
        source_commit=args.commit, source_tree=args.tree, allocation=allocated,
        exit_code=result.returncode, started=started, finished=time.time(),
        trace_gate=checked, artifact_errors=errors,
        build_identity_sha256=manifest_before,
        build_identity_unchanged=gate.digest(manifest_path) == manifest_before,
        geometry_sha256=args.geometry_sha256, trace_sha256=gate.digest(trace),
        result_sha256=gate.digest(record_path) if record_path.exists() else None,
        qualification='Declared native model, not grid convergence, unrestrictedC10, CamCASP parity or dimer-energy accuracy.')
    (run/'acceptance.json').write_text(json.dumps(acceptance, indent=2))
    print('PHOENIX_NATIVE_ACCEPTANCE', acceptance['status'], args.case, args.commit, flush=True)
    if not passed:
        print(json.dumps(dict(gate=checked.get('disallowed_accesses', checked.get('error')),
                              artifact_errors=errors))[:12000], flush=True)
        print((run/'stdout.log').read_text(errors='replace')[-12000:], flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
