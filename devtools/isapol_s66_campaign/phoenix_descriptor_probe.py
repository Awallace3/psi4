"""Compute allocation supervisor for sealed, SCF-free descriptor predictions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def policy_binding(source, manifest, planner):
    if manifest['commit'] != planner.SCIENTIFIC_COMMIT:
        raise RuntimeError('descriptor mirror requires exact acf54 scientific producer')
    observed = {}
    for path, expected in planner.FILES.items():
        raw = subprocess.check_output(
            ['git', '-C', str(source), 'show', manifest['commit']+':'+path])
        observed[path] = hashlib.sha256(raw).hexdigest()
        if observed[path] != expected:
            raise RuntimeError('source-bound planning policy differs: '+path)
    return observed


def disjoint_paths(build, run):
    return not (build.is_relative_to(run) or run.is_relative_to(build))


def verify_child(report, identity, *, policy, runtime_sha256, probe_sha256, planner_sha256,
                 intended_memory_bytes):
    """A pure calculator/unsigned child cannot produce a bound allocation pass."""
    planner = helper('resource_plan')
    if (not policy or policy != planner.FILES
            or report['implementation']['sha256'] != identity['implementation']['sha256']
            or Path(report['core']).resolve() != Path(identity['core']).resolve()
            or Path(report['package']).resolve() != Path(identity['package']).resolve()
            or report['runtime_input_sha256'] != runtime_sha256
            or report['source_policy_sha256'] != policy
            or report['descriptor_probe_sha256'] != probe_sha256
            or report['planner_sha256'] != planner_sha256
            or report['coordinate_change_bohr'] != 0
            or report['intended_memory_bytes'] != intended_memory_bytes
            or report['plan']['policy_scientific_commit'] != planner.SCIENTIFIC_COMMIT
            or report['plan']['resources']['max_bytes'] > report['intended_memory_bytes']
            or report['plan']['status'] not in ('WITHIN_PREDICTED_LEDGER_LIMITS', 'REFUSED')
            or (report['status'] == 'PASS_DESCRIPTOR_PLAN_ONLY'
                and (report['plan']['refusals']
                     or report['plan']['status'] != 'WITHIN_PREDICTED_LEDGER_LIMITS'))
            or report['status'] not in ('PASS_DESCRIPTOR_PLAN_ONLY', 'REFUSED_DESCRIPTOR_PLAN')):
        raise RuntimeError('descriptor implementation/input/policy differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'qualification', 'run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pilots', type=Path, nargs='+', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--memory-bytes', type=int, required=True)
    args = parser.parse_args()
    checks = helper('phoenix_installed_checks')
    planner = helper('resource_plan')
    worker = helper('phoenix_worker_probe')
    allocation = checks.allocation()
    source = Path(__file__).resolve().parents[2]
    if checks.git(source, 'rev-parse', 'HEAD') != args.commit or checks.git(source, 'status', '--porcelain'):
        raise RuntimeError('exact clean operational source required')
    prefix = Path('/storage/project/r-cs207-0/awallace43/runs')
    build, run = args.build.resolve(), args.run.resolve()
    if (not build.is_relative_to(prefix) or not run.is_relative_to(prefix)
            or not run.name.startswith('ccasp-native-s66-') or not disjoint_paths(build, run)):
        raise RuntimeError('unexpected immutable build/probe paths')
    if args.memory_bytes != 20*1024**3:
        raise RuntimeError('this24GiB allocation declares exactly20GiB scientific memory')
    manifest_path = build/'build-identity.json'
    manifest = json.loads(manifest_path.read_text())
    qualification = json.loads(args.qualification.read_text())
    identity_path = args.qualification.with_name('imported-identity.json')
    identity = json.loads(identity_path.read_text())
    junit_path = args.qualification.with_name('junit.xml')
    receipts = {str(p.resolve()): checks.digest(p)
                for p in (manifest_path, args.qualification, identity_path, junit_path, *args.pilots)}
    if len(receipts) != 4+len(args.pilots):
        raise RuntimeError('unique immutable input receipt files required')
    checks.junit_summary(junit_path, expected=197)
    if (qualification.get('status') != 'passed'
            or qualification.get('runtime_unchanged') is not True
            or qualification['build_commit'] != manifest['commit']
            or qualification['build_tree'] != manifest['tree']
            or qualification['build_identity_sha256'] != checks.digest(manifest_path)
            or qualification['imported_identity_sha256'] != checks.digest(identity_path)
            or qualification['junit_sha256'] != checks.digest(junit_path)):
        raise RuntimeError('sealed installed qualification differs')
    changes = checks.git(source, 'diff', '--name-only', manifest['commit'], args.commit).splitlines()
    if any(not p.startswith('devtools/isapol_s66_campaign/')
           and p not in checks.TEST_ONLY_CHANGES for p in changes):
        raise RuntimeError('operational source changes scientific/build code')
    policy = policy_binding(source, manifest, planner)
    environment = build/'env'
    base = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt')
    checks.verify_runtime(environment, manifest)
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    scratch = Path('/storage/scratch1/3/awallace43')/('ccasp-native-descriptors-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(exist_ok=False)
    for name in ('home', 'cache', 'cwd'):
        (scratch/name).mkdir()
    env = dict(os.environ)
    for key in list(env):
        if key in ('PYTHONPATH', 'PYTHONHOME', 'PSIPATH', 'PSIDATADIR', 'LD_PRELOAD') or key.startswith('QCF_COMPUTE_'):
            env.pop(key)
    env.update(PATH=f'{environment}/bin:{base}/bin:/usr/bin:/bin',
               LD_LIBRARY_PATH=f'{environment}/lib:{base}/lib',
               HOME=str(scratch/'home'), TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    reports, seen = [], set()
    for pilot_path in args.pilots:
        pilot = json.loads(pilot_path.read_text())
        monomer = pilot['monomer_id']
        if (monomer in seen or len(monomer) != 12 or not monomer.startswith('monomer-')
                or not monomer[8:].isdigit()):
            raise RuntimeError('unique bounded monomer identifier required')
        seen.add(monomer)
        if (pilot.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
                or pilot['scientific_build_commit'] != manifest['commit']
                or pilot['build_identity_sha256'] != checks.digest(manifest_path)
                or pilot['source_files_sha256']['installed_qualification'] != checks.digest(args.qualification)
                or pilot['submission'] != dict(find_existing=False, maximum_concurrency=1, qos='embers')):
            raise RuntimeError('exact unsubmitted preparation required')
        runtime = worker.runtime_input(pilot)
        runtime_path = payload/(monomer+'-runtime.json')
        runtime_path.write_text(json.dumps(runtime, allow_nan=False))
        if 'sapt_reference' in runtime_path.read_text():
            raise RuntimeError('entry-only reference attributes entered runtime')
        report_path = payload/(monomer+'-descriptors.json')
        command = checks.installed_python_command(
            environment/'bin/python', scratch/'cache',
            str(Path(__file__).with_name('descriptor_probe.py')),
            '--input', str(runtime_path), '--output', str(report_path),
            '--memory-bytes', str(args.memory_bytes))
        with (payload/(monomer+'.stdout')).open('x') as out, (payload/(monomer+'.stderr')).open('x') as err:
            subprocess.run(command, cwd=scratch/'cwd', env=env, stdout=out, stderr=err, check=True)
        report = json.loads(report_path.read_text())
        verify_child(report, identity, policy=policy, runtime_sha256=checks.digest(runtime_path),
                     probe_sha256=checks.digest(Path(__file__).with_name('descriptor_probe.py')),
                     planner_sha256=checks.digest(Path(__file__).with_name('resource_plan.py')),
                     intended_memory_bytes=args.memory_bytes)
        reports.append(dict(monomer_id=monomer, status=report['status'],
                            pilot_sha256=checks.digest(pilot_path),
                            runtime_input_sha256=checks.digest(runtime_path),
                            descriptor_report_sha256=checks.digest(report_path),
                            descriptor_report=report_path.name))
    checks.verify_runtime(environment, manifest)
    if (checks.digest(manifest_path) != qualification['build_identity_sha256']
            or any(checks.digest(Path(p)) != digest for p, digest in receipts.items())
            or checks.git(source, 'rev-parse', 'HEAD') != args.commit
            or checks.git(source, 'status', '--porcelain')):
        raise RuntimeError('sealed runtime or operational source changed')
    result = dict(
        schema='native-campaign-descriptor-allocation/1',
        status='PASS_DESCRIPTOR_PLANS_ONLY' if all(r['status'] == 'PASS_DESCRIPTOR_PLAN_ONLY'
                                                  for r in reports) else 'REFUSED_DESCRIPTOR_PLANS',
        allocation=allocation, scientific_build_commit=manifest['commit'],
        operational_commit=args.commit, build_identity_sha256=checks.digest(manifest_path),
        qualification_sha256=checks.digest(args.qualification), source_policy_sha256=policy,
        input_receipts_sha256=receipts,
        intended_memory_bytes=args.memory_bytes, runtime_unchanged=True, reports=reports,
        qualification='No SCF, ISA, response/fit, QCArchive writes, record claims or manager discovery. '
                      'Descriptor/ledger prediction only; no molecular/runtime independence acceptance.')
    (payload/'descriptor-allocation.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], report=str(payload/'descriptor-allocation.json'))))


if __name__ == '__main__':
    main()
