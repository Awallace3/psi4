"""Installed-only tests in a private Phoenix embers allocation.

The scientific build pin and test/harness pin are separate. A newer revision
may change only explicitly allowed tests/operational files; it may not change
scientific or build code. No production modules are overlaid or reinstalled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import xml.etree.ElementTree as ET


TEST_FILES = (
    'test_isapol_schema.py', 'test_isapol_molecular_admission.py',
    'test_isapol_molecular_declaration.py', 'test_isapol_pfit_stream.py',
    'test_isapol_bounded_oeprop.py', 'test_isapol_bounded.py',
    'test_isapol_pfit_workspace.py', 'test_isapol_pfit_forwarding.py',
    'test_isapol_extended_resources.py', 'test_isapol_bounded_response.py',
    'test_isapol_native_partition.py',
)
# Preserve all 430 prior interface cases and add all 29 native partition cases,
# including the bounded Drho-C forwarding/refusal regression. No local QM run.
EXPECTED_TESTS = 459
INSTALLED_SUITES = {'legacy': EXPECTED_TESTS, 'grid400': EXPECTED_TESTS+2}
TEST_ONLY_CHANGES = frozenset('tests/pytests/'+name for name in TEST_FILES) | frozenset(
    'devtools/isapol_s66_campaign/'+name for name in (
        'phoenix_installed_checks.py', 'phoenix_installed_checks.sbatch',
        'test_installed_checks.py', 'README.md'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def junit_summary(path, *, expected=None):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
    result = {key: sum(int(suite.get(key, '0')) for suite in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if result['tests'] <= 0 or result['failures'] or result['errors'] or result['skipped']:
        raise ValueError('nonempty fully successful installed test run required')
    if expected is not None and result['tests'] != expected:
        raise ValueError(f'installed suite must run exactly{expected} cases')
    return result


def qualification_suite(name):
    """Closed expected counts, not filters; no caller-supplied count is accepted.

    Historical harnesses retain their459-case inventory. This snapshot adds two
    cases: selecting legacy here fails the count rather than deselecting them.
    A fresh dense-grid qualification must explicitly select grid400.
    """
    if type(name) is not str or name not in INSTALLED_SUITES:
        raise ValueError('closed installed qualification suite required')
    return INSTALLED_SUITES[name]


def verify_runtime(environment, manifest):
    """Check the complete sealed private runtime and ordinary dependencies."""
    environment = Path(environment).resolve()
    paths = list(environment.rglob('*'))
    if any(p.is_file() and p.suffix == '.pyc' and p.parent.name != '__pycache__' for p in paths):
        raise RuntimeError('unsealed standalone bytecode in runtime inventory')
    actual_files = {p.relative_to(environment).as_posix() for p in paths
                    if p.is_file() and p.suffix != '.pyc'}
    actual_directories = {'.'} | {p.relative_to(environment).as_posix() for p in paths if p.is_dir()}
    if (actual_files != set(manifest['runtime_files'])
            or actual_directories != set(manifest['runtime_directories'])):
        raise RuntimeError('private installed runtime inventory changed')
    # Build manifests intentionally exclude .pyc. Every child Python is given
    # a fresh explicit bytecode prefix, so excluded old caches cannot execute.
    for path in paths:
        if path.is_symlink() and path.is_dir() and not path.resolve().is_relative_to(environment):
            raise RuntimeError('runtime directory symlink leaves declared dependencies')
    external = manifest['external_runtime_files']
    for name, expected in manifest['runtime_files'].items():
        path = environment/name
        resolved = path.resolve()
        if (not resolved.is_relative_to(environment) and str(resolved) not in external):
            raise RuntimeError('runtime symlink leaves declared dependencies: '+name)
        if digest(path) != expected:
            raise RuntimeError('private installed runtime changed: '+name)
    for name, expected in external.items():
        if digest(name) != expected:
            raise RuntimeError('ordinary external runtime changed: '+name)


def test_environment(original):
    """No caller-selected pytest filtering, plugin injection or user config."""
    env = {key: value for key, value in original.items() if not key.startswith('PYTEST_')}
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    return env


def installed_python_command(python, cache, *arguments):
    return [str(python), '-I', '-B', '-X', 'pycache_prefix='+str(cache), *arguments]


def validate_test_revision(changed_paths):
    if set(changed_paths)-TEST_ONLY_CHANGES:
        raise ValueError('newer test revision changes scientific/build or undeclared files')


def git(source, *args):
    return subprocess.check_output(['git', *args], cwd=source, text=True).strip()


def allocation():
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('USER') != 'awallace43'
            or os.environ.get('SLURM_CLUSTER_NAME') != 'phoenix-slurm'
            or os.environ.get('SLURMD_NODENAME', '').split('.')[0] != host
            or re.fullmatch(r'atl\d+(?:-\d+){5}', host) is None):
        raise RuntimeError('installed validation requires an authorized Phoenix compute allocation')
    text = subprocess.check_output([
        '/opt/slurm/current/bin/scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True)
    fields = dict(item.split('=', 1) for item in text.split() if '=' in item)
    if (fields.get('QOS') != 'embers' or fields.get('Account') != 'gts-cs207-chemx'
            or fields.get('JobState') != 'RUNNING' or fields.get('BatchHost') != host
            or fields.get('UserId', '').split('(')[0] != 'awallace43'):
        raise RuntimeError('effective installed-test allocation is not free embers')
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'run', 'dependencies'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('commit', 'tree', 'harness-commit'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--suite', choices=tuple(INSTALLED_SUITES), default='legacy')
    args = parser.parse_args()
    expected_tests = qualification_suite(args.suite)
    allocated = allocation()
    build, run, base = (p.resolve() for p in (args.build, args.run, args.dependencies))
    prefix = Path('/storage/project/r-cs207-0/awallace43/runs')
    if (not build.is_relative_to(prefix) or not run.is_relative_to(prefix)
            or not run.name.startswith('ccasp-native-s66-')
            or base != Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt').resolve()):
        raise RuntimeError('unexpected private installed-test paths')
    harness = Path(__file__).resolve().parents[2]
    if git(harness, 'rev-parse', 'HEAD') != args.harness_commit or git(harness, 'status', '--porcelain'):
        raise RuntimeError('operational harness source is not the pinned clean snapshot')
    manifest_path = build/'build-identity.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((build/'build-record.json').read_text())
    if (manifest['commit'] != args.commit or manifest['tree'] != args.tree
            or not manifest['fresh_build'] or not manifest['clean_source']
            or receipt['status'] != 'complete'
            or receipt['build_identity_sha256'] != manifest_hash):
        raise RuntimeError('completed exact-source build receipt required')
    source = Path(manifest['source_snapshot']).resolve()
    for values, expected in ((('rev-parse', 'HEAD'), args.commit),
                             (('rev-parse', 'HEAD^{tree}'), args.tree),
                             (('status', '--porcelain'), '')):
        if git(source, *values) != expected:
            raise RuntimeError('original scientific source changed')
    changed = git(harness, 'diff', '--name-only', args.commit, args.harness_commit).splitlines()
    validate_test_revision(changed)
    rows = subprocess.check_output([
        '/opt/slurm/current/bin/sacct', '-n', '-P', '-j', str(receipt['job']),
        '-o', 'JobIDRaw,State,ExitCode'], text=True)
    rows = [line.strip().split('|') for line in rows.splitlines() if line.strip()]
    for job in (str(receipt['job']), str(receipt['job'])+'.batch'):
        if not any(row[:3] == [job, 'COMPLETED', '0:0'] for row in rows):
            raise RuntimeError('build allocation and batch must both be COMPLETED0:0')
    environment = build/'env'
    package = environment/'lib/python3.13/site-packages/psi4'
    if Path(manifest['installed_package']).resolve() != package:
        raise RuntimeError('foreign package path in manifest')
    run.mkdir(parents=True, exist_ok=True)
    payload = run/'payload'
    payload.mkdir(exist_ok=False)
    cwd = payload/'cwd'
    cwd.mkdir()
    cache = payload/'empty-import-cache'
    cache.mkdir()
    scratch = Path('/storage/scratch1/3/awallace43')/('ccasp-native-checks-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(exist_ok=False)
    env = test_environment(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PSIPATH', 'PSIDATADIR', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(
        PATH=f'{environment}/bin:{base}/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH=f'{environment}/lib:{base}/lib',
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPYCACHEPREFIX=str(cache),
        TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    started = dict(
        status='started', build_commit=args.commit, build_tree=args.tree,
        harness_commit=args.harness_commit, harness_sha256=digest(__file__),
        test_source_commit=args.harness_commit, test_source_path=str(harness/'tests/pytests'),
        test_only_revision_changes=changed, expected_test_count=expected_tests,
        build_identity_sha256=manifest_hash, allocation=allocated, started=time.time(),
        qualification='Installed interface/contract tests, not campaign molecular or QCArchive acceptance.')
    if args.suite != 'legacy':
        started['installed_suite'] = args.suite
    (payload/'started.json').write_text(json.dumps(started, indent=2))
    verify_runtime(environment, manifest)
    python = environment/'bin/python'
    probe = (
        'import json,psi4;from psi4.driver.procrouting.isapol_schema import implementation_identity;'
        'print(json.dumps(dict(package=psi4.__file__,core=psi4.core.__file__,'
        'version=str(psi4.__version__),implementation=implementation_identity())))')
    imported = json.loads(subprocess.check_output(
        installed_python_command(python, cache, '-c', probe), env=env, cwd=cwd, text=True))
    if (Path(imported['package']).resolve().parent != package
            or Path(imported['core']).resolve() != Path(manifest['installed_core']).resolve()):
        raise RuntimeError('test Python imported a foreign/source Psi4')
    for relative, value in imported['implementation']['python_files'].items():
        if manifest['installed_files'].get('driver/'+relative) != value:
            raise RuntimeError('native route identity is not in the sealed build: '+relative)
    if digest(imported['core']) != imported['implementation']['sha256']['core']:
        raise RuntimeError('core identity differs from imported native implementation')
    (payload/'imported-identity.json').write_text(json.dumps(imported, indent=2))
    tests = [harness/'tests/pytests'/name for name in TEST_FILES]
    if not all(path.is_file() for path in tests):
        raise RuntimeError('pinned installed-test sources missing')
    test_hashes = {path.name: digest(path) for path in tests}
    junit = payload/'junit.xml'
    command = installed_python_command(
        python, cache, '-m', 'pytest', '--noconftest', '-c', '/dev/null',
        '-p', 'no:cacheprovider', '--import-mode=importlib', '-q', '--tb=short',
        '--junitxml='+str(junit), *map(str, tests))
    with (payload/'pytest.log').open('w') as stream:
        result = subprocess.run(command, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('installed tests failed; inspect payload/pytest.log and junit.xml')
    summary = junit_summary(junit, expected=expected_tests)
    verify_runtime(environment, manifest)
    if (digest(manifest_path) != manifest_hash or git(source, 'status', '--porcelain')
            or git(harness, 'rev-parse', 'HEAD') != args.harness_commit
            or git(harness, 'status', '--porcelain')
            or any(digest(path) != test_hashes[path.name] for path in tests)):
        raise RuntimeError('build identity/source changed during installed tests')
    report = dict(started, status='passed', finished=time.time(), command=command,
                  test_summary=summary, imported_identity_sha256=digest(payload/'imported-identity.json'),
                  junit_sha256=digest(junit), runtime_unchanged=True, test_source_sha256=test_hashes)
    (payload/'installed-validation.json').write_text(json.dumps(report, indent=2))
    print('PHOENIX_INSTALLED_NATIVE_TESTS', json.dumps(dict(
        status='passed', build_commit=args.commit, job=os.environ['SLURM_JOB_ID'],
        tests=summary, report=str(payload/'installed-validation.json'))), flush=True)


def qomp_engine():
    """Load this ops child's shared build helpers, never native Psi4."""
    import importlib.util
    import sys
    path = Path(__file__).resolve().parents[2]/'devtools/isapol_native_acceptance/phoenix_build.py'
    spec = importlib.util.spec_from_file_location('_qomp_fresh_build_engine', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def qomp_build_gate(engine, lib, args):
    """No READY/failed/foreign receipt adoption; parent supplies actual digests."""
    root = engine.QOMP_BUILD
    for filename, attribute in (('build-identity.json', 'build_identity_sha256'),
                                ('build-record.json', 'build_record_sha256'),
                                ('collection.json', 'collection_sha256')):
        expected = getattr(args, attribute)
        engine.qomp_need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected)
                         and lib.sha(root/filename) == expected, 'parent build/collection digest differs')
    engine.qomp_need(not (root/'FAILED').exists()
                     and lib.read(root/'COMPLETED') == dict(exit_code=0, mode='build', acceptance=False),
                     'fresh normal build must have actual successful terminal marker')
    value = lib.read(root/'result.json')
    manifest, record = lib.read(root/'build-identity.json'), lib.read(root/'build-record.json')
    source = engine.qomp_source_gate(args.ops_commit, args.ops_tree)
    engine.qomp_need(
        value['schema'] == 'qomp-production-operation/1'
        and value['status'] == 'FRESH_BUILD_COMPLETE_NOT_QUALIFIED'
        and value['mode'] == 'build' and value['acceptance'] is False
        and value['production_adoption'] is False and value['source'] == source
        and value['old_and_base_unchanged'] is True
        and manifest['schema'] == 'fresh-native-build-identity/1'
        and manifest['fresh_build'] is True and manifest['clean_source'] is True
        and manifest['commit'] == record['commit'] == engine.QOMP_SCI
        and manifest['tree'] == record['tree'] == engine.QOMP_TREE
        and manifest['operations'] == record['operations'] == source
        and manifest['source_snapshot'] == record['source'] == str(root/'source')
        and manifest['installed_package'] == str(root/'env/lib/python3.13/site-packages/psi4')
        and record['root'] == str(root) and record['dependencies'] == str(engine.QOMP_BASE)
        and manifest['old_manifest_sha256'] == lib.OLD_MANIFEST
        and manifest['provisioning'] == record['provisioning'] == 'offline qualified1aaa dependency clone'
        and value['identity']['package'] == manifest['installed_package']+'/__init__.py'
        and value['identity']['core'] == manifest['installed_core']
        and Path(manifest['installed_core']).resolve().parent == Path(manifest['installed_package']).resolve()
        and value['identity']['implementation']['core'] == lib.sha(manifest['installed_core'])
        and str(value['allocation']['JobId']) == str(manifest['allocation']['JobId']) == str(record['job'])
        and record['status'] == 'complete'
        and manifest['original_configure_sha256'] == record['original_configure_sha256']
        == lib.APPROVED_RECORD_SHA
        and record['build_identity_sha256'] == args.build_identity_sha256
        and value['build_identity_sha256'] == args.build_identity_sha256
        and value['build_record_sha256'] == args.build_record_sha256
        and value['collection_sha256'] == args.collection_sha256, 'genuine exact fb11 build required')
    expected_configure = lib.configure(lib.approved_record(lib.OLD/'build-record.json'),
                                       root/'source', root)
    expected_build = list(map(str, [engine.QOMP_BASE/'bin/cmake', '--build',
                                    root/'build', '--target', 'install', '-j', '2']))
    engine.qomp_need(record['configure'] == expected_configure and record['build'] == expected_build,
                     'actual build commands differ from approved replay')
    for label, argv in (('configure', expected_configure), ('build-install', expected_build)):
        saved = lib.read(root/(label+'-result.json'))
        engine.qomp_need(saved == record['command_results'][label] and saved['argv'] == argv
                         and type(saved['returncode']) is int and saved['returncode'] == 0
                         and saved['interrupted'] is False, 'actual build command did not return0')
    allocation_rows = subprocess.check_output([
        '/opt/slurm/current/bin/sacct', '-n', '-P', '-j', str(record['job']),
        '-o', 'JobIDRaw,State,ExitCode'],
        env=engine.qomp_control_environment(os.environ), text=True)
    rows = [row.split('|')[:3] for row in allocation_rows.splitlines() if row.strip()]
    engine.qomp_need(all([str(record['job'])+suffix, 'COMPLETED', '0:0'] in rows
                         for suffix in ('', '.batch', '.0')), 'build job/batch/step not COMPLETED0')
    for argv, expected in ((('rev-parse', 'HEAD'), engine.QOMP_SCI),
                           (('rev-parse', 'HEAD^{tree}'), engine.QOMP_TREE),
                           (('status', '--porcelain', '--untracked-files=all'), '')):
        engine.qomp_need(engine.qomp_git(root/'source', *argv) == expected, 'compiled source changed')
    verify_runtime(root/'env', manifest)
    checks = lib.installed_python_gate(root/'source', root/'build',
                                       Path(manifest['installed_package']),
                                       engine.qomp_installed_names(manifest['installed_files']))
    engine.qomp_need(checks == lib.read(root/'installed-python-checks.json'),
                     'complete installed Python mapping changed')
    lib.dependency_gate(root/'env', lib.old_seal())
    collection = lib.read(root/'collection.json')
    counts = engine.qomp_nodes_gate(collection)
    command_result = lib.read(root/'collect-tests-result.json')
    engine.qomp_need(collection['source'] == dict(commit=engine.QOMP_SCI, tree=engine.QOMP_TREE,
                                                 path=str(engine.QOMP_SCI_SOURCE/'tests/pytests'))
                     and collection['group_counts'] == counts['groups']
                     and collection['focused_modules'] == counts['focused_modules']
                     and collection['targets'] == engine.qomp_targets()
                     and collection['test_source_sha256'] == {
                         Path(path.split('::')[0]).name: lib.sha(path.split('::')[0])
                         for path in engine.qomp_targets()}
                     and collection['command_sha256'] == lib.sha(root/'collect-tests-command.json')
                     and collection['command_result_sha256'] == lib.sha(root/'collect-tests-result.json'),
                     'collected test source/command binding differs')
    engine.qomp_need(type(command_result['returncode']) is int and command_result['returncode'] == 0
                     and command_result['interrupted'] is False
                     and command_result['argv'] == lib.read(root/'collect-tests-command.json'),
                     'actual collection command failed/interrupted')
    return manifest, collection, value['identity']


def qomp_junit_gate(path, nodes):
    """Cross-check actual testcase names against the reviewed collection union."""
    from collections import Counter
    engine = qomp_engine()
    summary = junit_summary(path, expected=556)
    cases = ET.parse(path).getroot().findall('.//testcase')
    files = {node.split('::')[0] for node in nodes}
    actual = []
    for case in cases:
        classname = case.get('classname', '').split('.')
        matched = [file for file in files if Path(file).stem in classname]
        engine.qomp_need(len(matched) == 1, 'unknown/ambiguous JUnit module')
        file = matched[0]
        classes = classname[classname.index(Path(file).stem)+1:]
        actual.append('::'.join([file]+classes+[case.get('name', '')]))
    engine.qomp_need(Counter(actual) == Counter(nodes), 'JUnit omitted/replaced/duplicated nodes')
    return summary


def qomp_qualify(runner, lib, result, args):
    engine = qomp_engine()
    manifest, collection, built_identity = qomp_build_gate(engine, lib, args)
    prefix = engine.QOMP_BUILD/'env'
    before = engine.qomp_inventory(prefix, lib)
    before_build = {name: lib.sha(engine.QOMP_BUILD/name) for name in (
        'result.json', 'COMPLETED', 'build-identity.json', 'build-record.json', 'collection.json',
        'identity-before.stdout', 'identity-after.stdout', 'installed-python-checks.json')}
    lib.write(runner.root/'build-before.json', before_build)
    lib.write(runner.root/'prefix-before.json', before)
    try:
        identity = engine.qomp_identity(runner, prefix, args.scratch, lib, 'identity-before')
        engine.qomp_need(identity == built_identity, 'new build runtime identity changed')
        lib.write(runner.root/'imported-identity.json', identity)
        executed = engine.qomp_pytest(runner, prefix, args.scratch, lib, collect=False)
        engine.qomp_need(executed['nodes'] == collection['nodes'], 'actual collected union changed')
        summary = qomp_junit_gate(runner.root/'junit.xml', collection['nodes'])
        engine.qomp_need(engine.qomp_identity(runner, prefix, args.scratch, lib, 'identity-after')
                         == identity, 'qualification runtime identity changed')
        runner.check()
        report = dict(
            status='passed', acceptance=False, production_adoption=False, installed_suite='qomp-fb11',
            build_commit=engine.QOMP_SCI, build_tree=engine.QOMP_TREE,
            harness_commit=args.ops_commit, harness_tree=args.ops_tree,
            harness_sha256=lib.sha(Path(__file__)), test_source_commit=engine.QOMP_SCI,
            test_source_path=str(engine.QOMP_SCI_SOURCE/'tests/pytests'),
            expected_test_count=556, test_summary=summary,
            collection_sha256=args.collection_sha256,
            build_identity_sha256=args.build_identity_sha256,
            build_record_sha256=args.build_record_sha256,
            imported_identity_sha256=lib.sha(runner.root/'imported-identity.json'),
            junit_sha256=lib.sha(runner.root/'junit.xml'), imported_identity=identity,
            source=result['source'], allocation=result['allocation'])
        # Publication happens only after the integrity finally below succeeds.
    finally:
        after = engine.qomp_inventory(prefix, lib)
        after_build = {name: lib.sha(engine.QOMP_BUILD/name) for name in before_build}
        lib.write(runner.root/'prefix-after.json', after)
        lib.write(runner.root/'build-after.json', after_build)
        engine.qomp_need(after == before and after_build == before_build,
                         'private compiled prefix/build evidence changed')
        verify_runtime(prefix, manifest)
    report['runtime_unchanged'] = True
    report['test_source_sha256'] = {
        Path(path.split('::')[0]).name: lib.sha(path.split('::')[0]) for path in engine.qomp_targets()}
    lib.write(runner.root/'installed-validation.json', report)
    result.update(status='PASSED_INSTALLED_INTERFACE_NOT_PRODUCTION_ACCEPTANCE',
                  identity=identity, tests=summary,
                  installed_validation_sha256=lib.sha(runner.root/'installed-validation.json'))


def qomp_main():
    import sys
    engine = qomp_engine()
    engine.qomp_enter()
    engine.qomp_need(sys.argv[2:3] == ['checks'], 'closed installed qomp operation required')
    args = engine.qomp_parser('checks')
    engine.qomp_perform(args, qomp_qualify)
    return 0


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['qomp']:
        sys.exit(qomp_main())
    main()
