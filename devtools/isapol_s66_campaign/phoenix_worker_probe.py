"""Compute-node worker compatibility preflight: no SCF or QCArchive writes.

Verify the sealed scientific installation, its exact native request and
descriptor-only admission, then exercise the actual compute0.63 discovery
subprocess through the owned CLI boundary. This does not qualify a molecular
calculation or server result transport.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_input(pilot):
    """Separate executable inputs from dataset-only reference attributes."""
    entry = pilot['entry']
    return dict(
        molecule=entry['molecule'],
        specification=dict(driver=pilot['specification']['driver'],
                           model={k: pilot['specification'][k] for k in ('method', 'basis')},
                           keywords=dict(pilot['specification']['keywords'],
                                         **entry['additional_keywords'])))


def validate_worker_discovery(programs, version, scientific_version, *, scientific_imported):
    if scientific_imported:
        raise RuntimeError('manager interpreter imported a scientific Psi4 package')
    if programs.get('psi4') != version or version != scientific_version:
        raise RuntimeError('discovery/QCEngine CLI/private installation versions differ')


def scientific_probe(input_path):
    import psi4
    from psi4.driver.procrouting.isapol_schema import (
        admit_native_schema, native_molecule_schema, validate_native_request, check_native_geometry)
    from psi4.driver.procrouting.isapol_molecular import plan_molecular_refinement
    data = json.loads(input_path.read_text())
    specification = data['specification']
    request, geometry = admit_native_schema(specification, data['molecule'])
    psi4.set_memory('20 GiB', quiet=True)
    psi4.set_options({k: v for k, v in specification['keywords'].items() if k != 'function_kwargs'})
    molecule = psi4.core.Molecule.from_schema(native_molecule_schema(data['molecule'], request))
    request, identity = validate_native_request(
        request.as_dict(), method=specification['model']['method'],
        basis=specification['model']['basis'], molecule=molecule)
    plan = plan_molecular_refinement(request.declaration, request.resources,
                                    max_replay_work=request.max_replay_work)
    print(json.dumps(dict(
        package=psi4.__file__, core=psi4.core.__file__, version=str(psi4.__version__),
        implementation=identity, plan=plan,
        coordinate_change_bohr=check_native_geometry(molecule, geometry),
        qualification='Descriptor/request admission only; no SCF or molecular properties executed.')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scientific-probe', type=Path)
    for name in ('build', 'qualification', 'pilot', 'run'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--commit')
    append = load_helper('append_receipts')
    append.add_arguments(parser)
    args = parser.parse_args()
    if args.scientific_probe:
        scientific_probe(args.scientific_probe)
        return
    if any(getattr(args, name) is None for name in ('build', 'qualification', 'pilot', 'run', 'commit')):
        parser.error('all parent preflight paths and exact operational commit are required')
    checkpoint = append.verify(args)
    transport_binding = append.helper('dataset_append').transport_binding() if checkpoint is not None else None
    helper = load_helper('phoenix_installed_checks')
    allocation = helper.allocation()
    source = Path(__file__).resolve().parents[2]
    if helper.git(source, 'rev-parse', 'HEAD') != args.commit or helper.git(source, 'status', '--porcelain'):
        raise RuntimeError('exact clean operational source required')
    build = args.build.resolve()
    prefix = Path('/storage/project/r-cs207-0/awallace43/runs')
    run = args.run.resolve()
    if (not build.is_relative_to(prefix) or not run.is_relative_to(prefix)
            or not run.name.startswith('ccasp-native-s66-')):
        raise RuntimeError('unexpected build/preflight paths')
    manifest_path = build/'build-identity.json'
    manifest = json.loads(manifest_path.read_text())
    qualification = json.loads(args.qualification.read_text())
    identity_path = args.qualification.with_name('imported-identity.json')
    identity = json.loads(identity_path.read_text())
    junit_path = args.qualification.with_name('junit.xml')
    helper.junit_summary(junit_path, expected=197)
    if (qualification.get('status') != 'passed'
            or qualification.get('runtime_unchanged') is not True
            or qualification['build_commit'] != manifest['commit']
            or qualification['build_tree'] != manifest['tree']
            or qualification['build_identity_sha256'] != helper.digest(manifest_path)
            or qualification['imported_identity_sha256'] != helper.digest(identity_path)
            or qualification['junit_sha256'] != helper.digest(junit_path)):
        raise RuntimeError('sealed installed qualification differs from this scientific build')
    changes = helper.git(source, 'diff', '--name-only', manifest['commit'], args.commit).splitlines()
    if any(not p.startswith('devtools/isapol_s66_campaign/')
           and p not in helper.TEST_ONLY_CHANGES for p in changes):
        raise RuntimeError('operational revision changes scientific/build code')
    pilot = json.loads(args.pilot.read_text())
    if (pilot.get('status') != 'PREPARED_NOT_REGISTERED_OR_SUBMITTED'
            or pilot['scientific_build_commit'] != manifest['commit']
            or pilot['build_identity_sha256'] != helper.digest(manifest_path)
            or pilot['source_files_sha256']['installed_qualification'] != helper.digest(args.qualification)
            or pilot['submission'] != dict(find_existing=False, maximum_concurrency=1, qos='embers')):
        raise RuntimeError('exact unsubmitted qualified pilot preparation required')
    entry = pilot['entry']
    kwargs = entry['additional_keywords']
    if kwargs['function_kwargs']['native_monomer']['implementation_sha256'] != identity['implementation']['sha256']:
        raise RuntimeError('pilot request pins differ from qualified imported installation')
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    traces = payload/'traces'
    traces.mkdir()
    launcher = payload/'launcher'
    launcher.mkdir()
    (launcher/'psi4').symlink_to(source/'devtools/isapol_s66_campaign/native_psi4.sh')
    scratch = Path('/storage/scratch1/3/awallace43')/('ccasp-native-worker-probe-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(exist_ok=False)
    for name in ('cache', 'home', 'cwd'):
        (scratch/name).mkdir()
    runtime = runtime_input(pilot)
    runtime_path = payload/'runtime-input.json'
    runtime_path.write_text(json.dumps(runtime, allow_nan=False))
    if 'sapt_reference' in runtime_path.read_text():
        raise RuntimeError('reference metadata entered runtime input')
    environment = build/'env'
    base = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt')
    manager_env = Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca')
    helper.verify_runtime(environment, manifest)
    env = dict(os.environ)
    for key in list(env):
        if key in ('PYTHONPATH', 'PYTHONHOME', 'PSIPATH', 'PSIDATADIR', 'LD_PRELOAD') or key.startswith('QCF_COMPUTE_'):
            env.pop(key)
    env.update(
        PATH=f'{environment}/bin:{base}/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH=f'{environment}/lib:{base}/lib',
        HOME=str(scratch/'home'), TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(scratch/'cache'),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = helper.installed_python_command(
        environment/'bin/python', scratch/'cache', str(Path(__file__).resolve()),
        '--scientific-probe', str(runtime_path))
    science = json.loads(subprocess.check_output(command, env=env, cwd=scratch/'cwd', text=True))
    if (science['implementation']['sha256'] != identity['implementation']['sha256']
            or Path(science['core']).resolve() != Path(identity['core']).resolve()
            or Path(science['package']).resolve() != Path(identity['package']).resolve()):
        raise RuntimeError('scientific child imported a foreign implementation')
    # The actual manager and its discovery/compute children must keep p4qca's
    # Python. The launcher directory contains psi4 only, never python3.
    env.update(
        PATH=f'{launcher}:{manager_env}/bin:/usr/bin:/bin', LD_LIBRARY_PATH=str(manager_env/'lib'),
        CCASP_NATIVE_ENV=str(environment), CCASP_NATIVE_BASE=str(base),
        CCASP_NATIVE_CACHE=str(scratch/'cache'), CCASP_NATIVE_HOME=str(scratch/'home'),
        CCASP_NATIVE_TRACE=str(traces))
    os.environ.clear()
    os.environ.update(env)
    if Path(sys.executable).resolve() != (manager_env/'bin/python').resolve():
        raise RuntimeError('parent must run in the existing p4qca manager environment')
    if shutil.which('python3') != str(manager_env/'bin/python3'):
        raise RuntimeError('discovery Python is not the manager interpreter')
    import qcengine
    import qcfractalcompute
    from qcengine.programs import get_program
    from qcfractalcompute.apps.app_manager import discover_programs_conda
    import inspect
    if str(qcengine.__version__) != 'v0.34.1+3.gd56a0598':
        raise RuntimeError('deployed QCEngine version changed; inspect CLI compatibility before proceeding')
    if str(qcfractalcompute.__version__) != '0.63':
        raise RuntimeError('deployed compute manager version changed; private adapter requires re-review')
    harness = get_program('psi4')
    harness_path = Path(inspect.getfile(type(harness))).resolve()
    if harness_path != Path('/storage/project/r-cs207-0/awallace43/gits/qcengine/qcengine/programs/psi4.py'):
        raise RuntimeError('unexpected parent QCEngine harness source')
    if 'psi4' in sys.modules:
        raise RuntimeError('manager interpreter imported a scientific Psi4 package')
    version = harness.get_version()
    programs = discover_programs_conda(None)['qcengine.compute']['programs']
    validate_worker_discovery(programs, version, science['version'],
                              scientific_imported='psi4' in sys.modules)
    helper.verify_runtime(environment, manifest)
    if (helper.digest(manifest_path) != qualification['build_identity_sha256']
            or helper.git(source, 'rev-parse', 'HEAD') != args.commit
            or helper.git(source, 'status', '--porcelain')):
        raise RuntimeError('sealed build or operational source changed during the probe')
    report = dict(
        status='PASS_WORKER_PREFLIGHT_ONLY', allocation=allocation,
        scientific_build_commit=manifest['commit'], operational_commit=args.commit,
        build_identity_sha256=helper.digest(manifest_path),
        qualification_sha256=helper.digest(args.qualification), pilot_sha256=helper.digest(args.pilot),
        runtime_input_sha256=helper.digest(runtime_path), runtime_unchanged=True,
        science=science, manager_python=sys.executable, discovery_python=shutil.which('python3'),
        qcengine_version=str(qcengine.__version__), qcfractalcompute_version=str(qcfractalcompute.__version__),
        harness_path=str(harness_path), harness_sha256=helper.digest(harness_path),
        discovered_psi4=programs['psi4'], psi4_cli=shutil.which('psi4'),
        append_checkpoint=checkpoint,
        append_transport_binding=transport_binding,
        trace_files=[p.name for p in sorted(traces.glob('*.strace'))],
        qualification='No SCF, molecular calculation, manager acquisition or QCArchive writes; OS traces not yet an independence certificate.')
    (payload/'worker-preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(status=report['status'], job=os.environ['SLURM_JOB_ID'],
                          report=str(payload/'worker-preflight.json'))))


if __name__ == '__main__':
    main()
