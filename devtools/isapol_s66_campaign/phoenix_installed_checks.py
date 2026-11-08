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


# The following diagnostic path is deliberately NOT a qomp_build_gate variant.
# Build002's allocation FAILED; its successful component commands do not make
# it a successful normal build. These parent-observed anchors are immutable.
EXISTING_TEST_COMMIT = '311d17bf9177a972f3212c048e2f2177408b95d7'
EXISTING_TEST_TREE = '57ff961824c0da6d76cf48e4da63fc83979f4f91'
EXISTING_BASE_OPS = 'cef3c6cdecdefb7a6a356d2019a6d07d0a121cd2'
EXISTING_TEST_FILE = 'tests/pytests/test_isapol_isa_grid_accounting.py'
EXISTING_TEST_SHA = '75096d590cf161c76fd6e0f39db25270208fd9dd7d1be44b1490f1290a07751c'
EXISTING_PAIR = dict(
    core='f4a1b302d56a1904ddd807460082c1f21b2d8b418539e7c808e617b1515e5469',
    isapol_python='331bfcedbdfd0cb66656c6de5bdb84297d8645ff21d4ad6675832e3592dbbf8b')
EXISTING_ANCHORS = dict([
    ('configure-result.json', '9a0955b57550a45b8cc8a4cccf8270d309df4238e738e4a7b049f08af84cada5'),
    ('build-install-result.json', '113cb17b13aeec638c872cdf8436d5c70b64665de3ebc114a32583eb03b4cdf2'),
    ('identity-before-result.json', 'ccd8fe71f49d5f83ab0bf6ce98967330d88c1936ce2589416c063a15a4dd9e78'),
    ('identity-before.stdout', '681f66459978082e298530922ab8f7f4c55bb0d7d5f767f095d930f62597d474'),
    ('result.json', 'a67fe225027d06d2cb805b3c606b2f5f9a753670551c772ce5dd111eea5a0400'),
    ('FAILED', '3110514949ecad01f76fc3ae42b50bc7c49121dccd4161229d41a8dbefc08297'),
    ('owner.json', '04137b8c95c2a73dcf239bc4fa7d45fe1f0c37e09c12ec54b30c51bbcf69b6e6'),
    ('protected-before.json', '99558a98ede3f04c1c854b425e6a967e68a7b0579529725dda2cc04453918525'),
    ('protected-after.json', '99558a98ede3f04c1c854b425e6a967e68a7b0579529725dda2cc04453918525'),
    ('installed-python-checks.json', '58933ac4be1d9aa949dae3c159775aa3c51705c83d6c59823dae43cbaee1fe83'),
    ('clone-links.json', '40684d6b0395d747233f7e38bb4729fa202434b6f21bb3f4805bda361b71eaf5'),
    ('configure-command.json', 'c881ae092427d2251a20b9a932c77965062d062810c2993130a5d0b0a360a3a2'),
    ('build-install-command.json', '59ea64d5a6b8331fcaa234609b2446ffa7d2fcf3dcf1d6e8d72809829ae0c417'),
    ('collect-tests-command.json', '99988d8a1902b8005472af406d46f47c901a2ca149a6182c6fbed03a4c444570'),
    ('collect-tests-result.json', '04eeee7d00c9293b8e03637d6f26f6fa47665d43e780d449728b9177eb6021e1'),
    ('collect-tests.stdout', 'f871aeaa9c6d555e601aa706cd400cc9af5ecb560cfdad0b54dec66e36610b95'),
    ('collect-tests.stderr', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    ('collection-nodes.json', 'daa5ffafaa18dde4b07f6ac514051c06f4373bfc658c56e8e1cb717002004030'),
    ('configure.stdout', '402de041cb0625ffb746b63dbc24d76d99c87c7479667a3d78b982a8e47c1752'),
    ('configure.stderr', 'c0ee906d522832ad188c56b17098118971b9ec67f0a64066c6a42c5271c85d38'),
    ('build-install.stdout', 'c4c74cb727a7607ac7c8bab2bc81083c054ed1244d6dfd50c423159dde09a9ac'),
    ('build-install.stderr', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    ('build/CMakeCache.txt', 'b3ed41ce09a5d807e36b51adfb6cc9809fa33d4766943956a40193860a41936b'),
    ('build/install_manifest.txt', '32b33ba891fc532a74f088b25bb7957bd01a273f13b8bca73e8a3e219fbd90b5'),
    ('build/psi4-core-prefix/src/psi4-core-build/CMakeCache.txt',
     'ecae7bd958b49dba07a6aab5fdc7f4b45b4ce15fe4c6ea0e0122789f1dd0c3f2'),
    ('build/psi4-core-prefix/src/psi4-core-build/install_manifest.txt',
     'c77fa5fa6187f17b1be940e2d2482d420904f404f2f1b4b8c4661a0aa7f5f962'),
])


def existing_paths(engine):
    root = engine.QOMP_ROOT
    return dict(ops=root/'gits/psi4.ccasp-native-s66-qomp-ops004',
                tests=root/'gits/psi4.ccasp-native-s66-qomp-tests003',
                build=root/'runs/ccasp-native-s66-qomp-fb11-build002',
                collect=root/'runs/ccasp-native-s66-qomp-fb11-collection004',
                checks=root/'runs/ccasp-native-s66-qomp-fb11-installed004')


def existing_need(condition, message):
    if not condition:
        raise ValueError(message)


def existing_test_ast(original, corrected):
    """Prove only five identifier substitutions; never execute either source."""
    import ast
    old, new = ast.parse(original), ast.parse(corrected)
    counts = dict(argument=0, name=0, decorator=0)
    for node in ast.walk(old):
        if isinstance(node, ast.arg) and node.arg == 'request':
            node.arg = 'angular_request'
            counts['argument'] += 1
        if isinstance(node, ast.Name) and node.id == 'request':
            node.id = 'angular_request'
            counts['name'] += 1
        if isinstance(node, ast.Constant) and node.value == 'radial,request,angular':
            node.value = 'radial,angular_request,angular'
            counts['decorator'] += 1
    existing_need(counts == dict(argument=1, name=3, decorator=1)
                  and ast.dump(old) == ast.dump(new), 'test revision is not exact alpha-rename')
    # Independent literal count below covers all parametrized tests, regardless
    # of their function spelling; no arbitrary selector/count override.
    count = 0
    for node in new.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            n = 1
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == 'parametrize':
                    n *= len(ast.literal_eval(d.args[1]))
            count += n
    existing_need(count == 9, 'corrected grid module must retain all nine cases')


def existing_parametrize_lint(text):
    """Closed literal argname forms only; catches pytest's reserved request."""
    import ast
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'parametrize'):
            continue
        existing_need(bool(node.args), 'uninspectable parametrization argnames')
        try:
            names = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            raise ValueError('uninspectable parametrization argnames') from None
        if type(names) is str:
            names = [name.strip() for name in names.split(',')]
        existing_need(type(names) in (list, tuple) and bool(names)
                      and all(type(name) is str and name.isidentifier() for name in names)
                      and len(names) == len(set(names)) and 'request' not in names,
                      'reserved/invalid parametrization argnames')


def existing_source_gate(engine, args):
    paths = existing_paths(engine)
    existing_need(engine.SOURCE == paths['ops'], 'closed ops004 source required')
    existing_need(engine.qomp_git(paths['ops'], 'merge-base', EXISTING_BASE_OPS, args.ops_commit)
                  == EXISTING_BASE_OPS, 'ops must descend from reviewed cef3')
    changed = set(engine.qomp_git(paths['ops'], 'diff', '--name-only',
                                  EXISTING_BASE_OPS, args.ops_commit).splitlines())
    existing_need(changed <= engine.QOMP_OPS_FILES, 'ops changed paths outside allowlist')
    roles = dict(ops=(paths['ops'], args.ops_commit, args.ops_tree),
                 compiled=(paths['build']/'source', engine.QOMP_SCI, engine.QOMP_TREE),
                 science=(engine.QOMP_SCI_SOURCE, engine.QOMP_SCI, engine.QOMP_TREE),
                 tests=(paths['tests'], EXISTING_TEST_COMMIT, EXISTING_TEST_TREE))
    for role, (path, commit, tree) in roles.items():
        for argv, expected in ((('rev-parse', '--show-toplevel'), str(path)),
                               (('rev-parse', 'HEAD'), commit),
                               (('rev-parse', 'HEAD^{tree}'), tree),
                               (('status', '--porcelain', '--untracked-files=all'), '')):
            existing_need(engine.qomp_git(path, *argv) == expected, 'source role changed')
        if role != 'ops':
            detached = subprocess.run(
                [*engine.QOMP_GIT, '-C', str(path), 'symbolic-ref', '-q', 'HEAD'],
                env=engine.qomp_control_environment(os.environ),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            existing_need(detached.returncode == 1, 'compiled/science/test source must be detached')
    existing_need(engine.qomp_git(paths['tests'], 'diff', '--name-only', engine.QOMP_SCI,
                                  EXISTING_TEST_COMMIT) == EXISTING_TEST_FILE,
                  'test-only revision changed other files')
    corrected = paths['tests']/EXISTING_TEST_FILE
    existing_need(digest(corrected) == EXISTING_TEST_SHA, 'corrected test blob differs')
    existing_test_ast((engine.QOMP_SCI_SOURCE/EXISTING_TEST_FILE).read_text(), corrected.read_text())
    for target in engine.qomp_targets(paths['tests']):
        existing_parametrize_lint(Path(target.split('::')[0]).read_text())
    return {role: dict(path=str(path), commit=commit, tree=tree)
            for role, (path, commit, tree) in roles.items()}


def existing_manifest(path, prefix, entries, unique):
    """Keep raw bytes/hash in anchors; explicitly deduplicate only the view."""
    lines = path.read_text().splitlines()
    existing_need(len(lines) == entries and len(set(lines)) == unique,
                  'install manifest entry/unique counts differ')
    for line in lines:
        p = Path(line)
        existing_need(p.is_absolute() and '..' not in p.parts
                      and p.is_relative_to(prefix) and p != prefix,
                      'install manifest destination escapes prefix')
    return {str(Path(line).relative_to(prefix)) for line in lines}


EXISTING_PROTECTED_BYTES = 18257504
EXISTING_ADDR2LINE_SHA = 'c1f9abaad54a61ae6cb621566b4388f5265853ffb4de7b26c004dbe34d66966f'
# Exact unused BASE metadata, NOT a general "sealed broken link" allowance.
# Tuple columns: link, raw text, lexical target, currently absent anchor.
# Einsums is OFF in the approved configure; these packaged CMake-test sources
# and the malformed TOOLS= artifact are not invoked by collection/check modes.
EXISTING_BASE_METADATA = (
    ('libexec/gcc/x86_64-conda-linux-gnu/14.3.0/TOOLS=addr2line',
     '../../../../bin/x86_64-conda-linux-gnu-TOOLS=addr2line',
     'bin/x86_64-conda-linux-gnu-TOOLS=addr2line',
     'bin/x86_64-conda-linux-gnu-TOOLS=addr2line'),
    ('share/cmake/Einsums/tests/cxx23_static_call_operator.cu',
     '../../cmake/tests/cxx23_static_call_operator.cpp',
     'share/cmake/cmake/tests/cxx23_static_call_operator.cpp', 'share/cmake/cmake'),
    ('share/cmake/Einsums/tests/cxx23_static_call_operator.hip',
     '../../cmake/tests/cxx23_static_call_operator.cpp',
     'share/cmake/cmake/tests/cxx23_static_call_operator.cpp', 'share/cmake/cmake'),
)


def existing_base_metadata_proof(engine, root):
    """Fixed historical raw metadata pin, not historical target contents."""
    import stat
    base = engine.QOMP_BASE
    existing_need(Path(root) == base and base.is_absolute(),
                  'BASE metadata exception cannot apply to another root')
    path = existing_paths(engine)['build']/'protected-before.json'
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        info = os.fstat(stream.fileno())
        existing_need(stat.S_ISREG(info.st_mode) and info.st_size == EXISTING_PROTECTED_BYTES,
                      'original protected snapshot kind/size differs')
        raw = stream.read(EXISTING_PROTECTED_BYTES+1)
    existing_need(len(raw) == EXISTING_PROTECTED_BYTES and hashlib.sha256(raw).hexdigest()
                  == EXISTING_ANCHORS['protected-before.json'],
                  'parent-pinned original protected snapshot differs')
    sealed = json.loads(raw)['base']
    existing_need(type(sealed) is dict, 'invalid original BASE inventory')
    return base, sealed


def existing_base_metadata_case(base, sealed, case):
    """Only exact lstat ENOENT at this entry's declared anchor is permitted."""
    import errno
    import stat
    existing_need(case in EXISTING_BASE_METADATA, 'unreviewed BASE metadata case')
    name, raw, relative_target, relative_anchor = case
    link, target, anchor = base/name, base/relative_target, base/relative_anchor
    existing_need(sealed.get(name) == dict(kind='link', target=raw),
                  'original BASE metadata link row differs')
    existing_need(str(target) == os.path.normpath(str(link.parent/raw))
                  and target.is_relative_to(anchor)
                  and relative_anchor not in sealed and relative_target not in sealed,
                  'BASE metadata target/anchor inventory differs')

    def parents():
        required = {base}
        for start in (link.parent, anchor.parent):
            existing_need(start.is_relative_to(base), 'BASE metadata parent escapes')
            required.add(start)
            required.update(p for p in start.parents if p.is_relative_to(base))
        for parent in sorted(required, key=lambda p: len(p.parts)):
            existing_need(stat.S_ISDIR(parent.lstat().st_mode), 'BASE metadata parent is not a directory')
            if parent != base:
                existing_need(sealed.get(str(parent.relative_to(base))) == dict(kind='directory'),
                              'BASE metadata parent is not a sealed directory')
        existing_need(base.resolve(strict=True) == base, 'BASE metadata root is not canonical')

    parents()  # A missing/denied/linked ancestor is NOT the permitted absence.
    existing_need(stat.S_ISLNK(link.lstat().st_mode) and os.readlink(link) == raw,
                  'BASE metadata link kind/raw text changed')
    try:
        anchor.lstat()
    except FileNotFoundError as error:
        existing_need(error.errno == errno.ENOENT and error.filename == str(anchor),
                      'unexplained BASE metadata absence')
    else:
        raise ValueError('BASE metadata absent anchor appeared or was repaired')
    parents()  # Also catch an ancestor disappearing during the anchor check.
    existing_need(stat.S_ISLNK(link.lstat().st_mode) and os.readlink(link) == raw,
                  'BASE metadata link changed during absence check')
    return dict(kind='known-dangling-base-metadata-link', target=raw,
                expected_target=str(target), absent_anchor=str(anchor),
                target_absence='current-lstat-ENOENT',
                historical_evidence='pinned raw link metadata only; no target content/absence proof',
                baseline_sha256=EXISTING_ANCHORS['protected-before.json'])


def existing_base_metadata_rows(base, sealed):
    return {case[0]: existing_base_metadata_case(base, sealed, case)
            for case in EXISTING_BASE_METADATA}


def existing_base_metadata_preflight(engine):
    """Targeted three-case + normal addr2line check: no walk/native/execute."""
    import stat
    base, sealed = existing_base_metadata_proof(engine, engine.QOMP_BASE)
    rows = existing_base_metadata_rows(base, sealed)
    link, target = base/'bin/addr2line', base/'bin/x86_64-conda-linux-gnu-addr2line'
    original = sealed.get('bin/addr2line', {})
    existing_need(original.get('kind') == 'link' and stat.S_ISLNK(link.lstat().st_mode)
                  and os.readlink(link) == original['target']
                  and link.resolve(strict=True) == target
                  and sealed.get('bin/x86_64-conda-linux-gnu-addr2line') == dict(
                      kind='file', sha256=EXISTING_ADDR2LINE_SHA)
                  and stat.S_ISREG(target.lstat().st_mode)
                  and digest(target) == EXISTING_ADDR2LINE_SHA,
                  'required normal BASE addr2line binding changed')
    existing_need(existing_base_metadata_rows(base, sealed) == rows,
                  'BASE metadata absence changed during preflight')
    return dict(base=str(base), metadata=rows, normal_addr2line=dict(
        target=original['target'], resolved=str(target), sha256=EXISTING_ADDR2LINE_SHA))


def existing_snapshot(root, external, *, base_engine=None):
    """Full current namespace plus resolved link bindings, not a sandbox.

    External directory links are refused (the offline clone materializes BASE
    file links). Every external file must be in a verified dependency inventory.
    Internal directory links bind the complete underlying namespace via rows.
    Only the explicit BASE route admits the three pinned metadata cases above;
    every other root/link still requires strict resolution.
    """
    known = None
    if base_engine is not None:
        base, sealed = existing_base_metadata_proof(base_engine, root)
        known = existing_base_metadata_rows(base, sealed)
    root = root.resolve(strict=True)
    rows = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root).as_posix()
        if known is not None and rel in known:
            case = next(case for case in EXISTING_BASE_METADATA if case[0] == rel)
            rows[rel] = existing_base_metadata_case(base, sealed, case)
        elif path.is_symlink():
            target = path.resolve(strict=True)
            existing_need(target != root and target != path, 'cyclic/root directory link')
            if not target.is_relative_to(root):
                existing_need(target.is_file() and str(target) in external
                              and digest(target) == external[str(target)],
                              'unattributed escaping link')
            existing_need(target.is_file() or target.is_dir(), 'nonregular link target')
            rows[rel] = dict(kind='link', target=os.readlink(path), resolved=str(target),
                             target_kind='file' if target.is_file() else 'directory',
                             sha256=digest(target) if target.is_file() else None)
        elif path.is_file():
            rows[rel] = dict(kind='file', sha256=digest(path))
        elif path.is_dir():
            rows[rel] = dict(kind='directory')
        else:
            raise ValueError('nonregular current build object')
    for row in rows.values():
        if row['kind'] == 'link' and Path(row['resolved']).is_relative_to(root):
            name = str(Path(row['resolved']).relative_to(root))
            existing_need(name in rows and rows[name]['kind'] in ('file', 'directory'),
                          'internal link target absent from full namespace')
    if known is not None:
        existing_need(existing_base_metadata_rows(base, sealed) == known
                      and {name: rows.get(name) for name in known} == known,
                      'BASE metadata absence/namespace changed during snapshot')
    return rows


def existing_cmake_origins(build, prefix):
    """Read literal FILE/PROGRAM CMake install projections, never execute CMake.

    Only .cmake config files need this fallback. All other changed files have
    explicit source projections below. Refuse variable/unattributable origins.
    """
    origins = {}
    inner = build/'build/psi4-core-prefix/src/psi4-core-build'
    fixed = {
        'psi4PluginCache.cmake': inner/'psi4PluginCache.cmake',
        'psi4Config.cmake': inner/'share/cmake/psi4/psi4Config.cmake',
        'psi4ConfigVersion.cmake': inner/'share/cmake/psi4/psi4ConfigVersion.cmake',
    }
    fixed.update({name: build/'source/cmake'/name for name in (
        'psi4OptionsTools.cmake', 'custom_static_library.cmake', 'custom_cxxstandard.cmake',
        'xhost.cmake')})
    for script in sorted((build/'build').rglob('cmake_install.cmake')):
        if script.is_symlink() or not script.resolve().is_relative_to(build/'build'):
            raise ValueError('escaping install script')
        for body in re.findall(r'file\(INSTALL\s+(.*?)\)', script.read_text(), re.S):
            dest = re.match(r'DESTINATION\s+"([^"]+)"\s+TYPE\s+(?:FILE|PROGRAM)\s+', body)
            if not dest or 'FILES' not in body:
                continue
            directory = dest[1].replace('${CMAKE_INSTALL_PREFIX}', str(prefix))
            if '$' in directory or Path(directory) != prefix/'share/cmake/psi4':
                continue  # inner stage scripts use the independent stage prefix
            rest = body.split('FILES', 1)[1]
            renamed = re.search(r'\bRENAME\s+"([^"]+)"', body)
            sources = re.findall(r'"([^"]+)"', rest)
            if renamed:
                sources = sources[:1]
            for source in sources:
                source = Path(source)
                name = renamed[1] if renamed else source.name
                destination = Path(directory)/name
                if destination.suffix != '.cmake':
                    continue
                existing_need(source.is_absolute() and '$' not in str(source)
                              and '..' not in source.parts and source.is_file()
                              and source.resolve().is_relative_to(build),
                              'unattributed CMake install origin')
                rel = str(destination.relative_to(prefix))
                # A current generated install script alone is not an origin
                # authority for arbitrary additions. These are the only nine
                # Psi4 CMake exports in the frozen source's install recipe.
                permitted = (destination.parent == prefix/'share/cmake/psi4'
                             and ((name in fixed and source == fixed[name])
                                  or (name in ('psi4Targets.cmake', 'psi4Targets-release.cmake')
                                      and source.is_relative_to(inner/'src/CMakeFiles/Export')
                                      and source.name == name)))
                existing_need(permitted, 'unreviewed CMake install projection')
                origins.setdefault(rel, set()).add(source)
    return origins


def existing_attribution(prefix, build, original, installed, python_checks, cmake):
    """Attribute EVERY present file, not just the original dependency subset."""
    source, stage = build/'source', build/'build/stage'
    package = prefix/'lib/python3.13/site-packages/psi4'
    rows = {}
    for path in sorted(prefix.rglob('*')):
        if not path.is_file():
            continue
        name, actual = str(path.relative_to(prefix)), digest(path)
        if path.is_relative_to(package) and path.suffix == '.py':
            rel = str(path.relative_to(package))
            existing_need(rel in python_checks and actual == python_checks[rel]['sha256'],
                          'unmapped installed Python')
            rows[name] = dict(kind='compiled-python', sha256=actual)
            continue
        if original['runtime_files'].get(name) == actual:
            rows[name] = dict(kind='original-dependency-or-unchanged-data', sha256=actual)
            continue
        existing_need(name in installed, 'unexplained prefix addition/change: '+name)
        if path.parent == package and path.name.startswith('core.') and path.suffix == '.so':
            existing_need(actual == EXISTING_PAIR['core'], 'unapproved compiled core')
            rows[name] = dict(kind='parent-observed-compiled-core', sha256=actual)
            continue
        # No new arbitrary native dependency/bytecode is attributed to "build".
        existing_need(path.suffix not in ('.so', '.pyc') and '.so.' not in path.name,
                      'unattributed native dependency/bytecode: '+name)
        candidates = []
        for start, root in (('share/psi4/samples/', source/'samples'),
                            ('share/psi4/', source/'psi4/share/psi4'),
                            ('include/psi4/', source/'psi4/include/psi4')):
            if name.startswith(start):
                candidates.append(root/name.removeprefix(start))
                break
        if name == 'bin/psi4':
            candidates.append(build/'build/psi4-core-prefix/src/psi4-core-build/psi4')
        candidates += list(cmake.get(name, ()))
        matches = [p for p in candidates if p.is_file()
                   and p.resolve().is_relative_to(build) and digest(p) == actual]
        existing_need(bool(matches), 'no source/configured output attribution: '+name)
        staged = stage/name
        if staged.exists():
            existing_need(staged.is_file() and staged.resolve().is_relative_to(stage)
                          and digest(staged) == actual, 'stage/prefix bytes differ: '+name)
        rows[name] = dict(kind='source/configured-install', sha256=actual,
                          origins=sorted(map(str, matches)))
    # Directory additions also need attribution; parents of real attributed
    # files and sealed original empty directories are the complete allowance.
    allowed_dirs = set(original['runtime_directories'])
    for name in rows:
        allowed_dirs.update(str(p) for p in Path(name).parents)
    for path in prefix.rglob('*'):
        if path.is_dir():
            existing_need(str(path.relative_to(prefix)) in allowed_dirs,
                          'unexplained prefix directory addition')
    return rows


def existing_accounting(job, state, code, engine):
    text = subprocess.check_output([
        '/opt/slurm/current/bin/sacct', '-n', '-P', '-j', str(job),
        '-o', 'JobIDRaw,State,ExitCode'], text=True,
        env=engine.qomp_control_environment(os.environ))
    rows = [line.split('|')[:3] for line in text.splitlines() if line.strip()]
    expected = [[str(job)+s, state, code] for s in ('', '.batch', '.0')]
    existing_need(all(rows.count(row) == 1 for row in expected)
                  and all(row in expected or row == [str(job)+'.extern', 'COMPLETED', '0:0']
                          for row in rows), 'historical/new allocation accounting differs')
    return rows


def existing_historical(engine, lib):
    paths = existing_paths(engine)
    root = paths['build']
    for name, sha in EXISTING_ANCHORS.items():
        path = root/name
        existing_need(path.is_file() and not path.is_symlink() and digest(path) == sha,
                      'parent-observed historical artifact changed: '+name)
    existing_need(not any(os.path.lexists(root/name) for name in (
        'COMPLETED', 'build-identity.json', 'build-record.json', 'collection.json')),
        'failed002 must not acquire normal build receipts')
    record = lib.approved_record(lib.OLD/'build-record.json')
    for label, expected in (
            ('configure', lib.configure(record, root/'source', root)),
            ('build-install', list(map(str, [engine.QOMP_BASE/'bin/cmake', '--build',
                                            root/'build', '--target', 'install', '-j', '2'])))):
        value = lib.read(root/(label+'-result.json'))
        existing_need(value['argv'] == lib.read(root/(label+'-command.json')) == expected
                      and type(value['returncode']) is int and value['returncode'] == 0
                      and value['interrupted'] is False, 'component command did not return0')
    identity_result = lib.read(root/'identity-before-result.json')
    collection_result = lib.read(root/'collect-tests-result.json')
    existing_need(type(identity_result['returncode']) is int
                  and identity_result['returncode'] == 0
                  and identity_result['interrupted'] is False
                  and type(collection_result['returncode']) is int
                  and collection_result['returncode'] == 2
                  and collection_result['argv'] == lib.read(root/'collect-tests-command.json'),
                  'historical identity/collection result differs')
    # Retained identity marker, not raw json.loads(stdout with a non-JSON prefix).
    marker = [line.removeprefix('QOMP_IDENTITY=') for line in
              (root/'identity-before.stdout').read_text().splitlines()
              if line.startswith('QOMP_IDENTITY=')]
    existing_need(len(marker) == 1, 'missing/ambiguous retained identity marker')
    identity = json.loads(marker[0])
    existing_identity(identity, root/'env')
    rows = existing_accounting('15320675', 'FAILED', '1:0', engine)
    outer = existing_manifest(root/'build/install_manifest.txt', root/'env', 6348, 6344)
    inner = existing_manifest(root/'build/psi4-core-prefix/src/psi4-core-build/install_manifest.txt',
                              root/'build/stage', 4985, 4985)
    existing_need(inner <= outer, 'stage destinations absent from outer manifest')
    return identity, outer, rows


def existing_identity(value, prefix):
    package = prefix/'lib/python3.13/site-packages/psi4'
    existing_need(value['implementation'] == EXISTING_PAIR and value['version'] == '1.12a1.dev212'
                  and value['package'] == str(package/'__init__.py')
                  and Path(value['core']).parent == package
                  and digest(value['core']) == EXISTING_PAIR['core'],
                  'actual compiled002 identity differs')


def existing_current(engine, lib):
    """Complete current attribution BEFORE any native probe or pytest import."""
    paths = existing_paths(engine)
    root, prefix = paths['build'], paths['build']/'env'
    identity, installed, accounting = existing_historical(engine, lib)
    original = lib.old_seal()
    protected = engine.qomp_protected(lib)
    existing_need(protected == lib.read(root/'protected-before.json')
                  == lib.read(root/'protected-after.json'), 'OLD/BASE historical seal differs')
    external = dict(original['external_runtime_files'])
    external.update({str(engine.QOMP_BASE/name): row['sha256']
                     for name, row in protected['base'].items() if row['kind'] == 'file'})
    snapshot = existing_snapshot(root, external)
    bindings = dict(old=existing_snapshot(lib.OLD/'env', external),
                    base=existing_snapshot(engine.QOMP_BASE, external, base_engine=engine))
    checks = lib.installed_python_gate(root/'source', root/'build',
        prefix/'lib/python3.13/site-packages/psi4',
        engine.qomp_installed_names(original['installed_files']))
    existing_need(checks == lib.read(root/'installed-python-checks.json'),
                  'complete compiled-fb11 installed Python mapping differs')
    lib.dependency_gate(prefix, original)
    attribution = existing_attribution(prefix, root, original, installed, checks,
                                        existing_cmake_origins(root, prefix))
    return dict(snapshot=snapshot, protected=protected, protected_bindings=bindings, attribution=attribution,
                historical_identity=identity, historical_accounting=accounting,
                proof='Current attributed inventory, NOT a reconstructed historical full seal.')


def existing_collection_gate(engine, lib, args):
    paths = existing_paths(engine)
    root = paths['collect']
    for name, attr in (('verification.json', 'verification_sha256'),
                        ('collection.json', 'collection_sha256'),
                        ('result.json', 'collection_result_sha256')):
        expected = getattr(args, attr)
        existing_need(re.fullmatch('[0-9a-f]{64}', expected) is not None
                      and digest(root/name) == expected, 'parent collection digest differs')
    existing_need(not os.path.lexists(root/'FAILED') and lib.read(root/'COMPLETED') == dict(
        exit_code=0, mode='collect-existing', acceptance=False), 'new collection did not complete')
    result, receipt, collection = (lib.read(root/name) for name in (
        'result.json', 'verification.json', 'collection.json'))
    roles = existing_source_gate(engine, args)
    existing_need(result['status'] == 'VERIFIED_EXISTING_INSTALL_COLLECTED_NOT_QUALIFIED'
                  and result['source'] == roles and result['acceptance'] is False
                  and result['production_adoption'] is False
                  and result['preserved_current_build_and_old_base'] is True
                  and receipt['schema'] == 'existing-failed-build-install-verification/1'
                  and receipt['source'] == roles and receipt['historical_build_status'] == 'FAILED'
                  and receipt['historical_job'] == '15320675'
                  and receipt['anchors'] == EXISTING_ANCHORS
                  and receipt['acceptance'] is False and receipt['production_adoption'] is False,
                  'distinct existing-install verification receipt required')
    existing_need(receipt['current_before_sha256'] == digest(root/'current-before.json')
                  and receipt['current_after_sha256'] == digest(root/'current-after.json')
                  and receipt['identity'] == result['identity']
                  and receipt['allocation'] == result['allocation']
                  and receipt['reconstructed_historical_full_seal'] is False
                  and receipt['historical_component_returncodes'] == dict(
                      configure=0, install=0, identity=0, collection=2),
                  'verification inventory/identity/allocation binding differs')
    existing_accounting(result['allocation']['JobId'], 'COMPLETED', '0:0', engine)
    engine.qomp_nodes_gate(collection)
    expected_source = dict(commit=EXISTING_TEST_COMMIT, tree=EXISTING_TEST_TREE,
                           path=str(paths['tests']/'tests/pytests'))
    existing_need(collection['source'] == expected_source
                  and collection['targets'] == engine.qomp_targets(paths['tests'])
                  and collection['test_source_sha256'] == {
                      Path(p.split('::')[0]).name: digest(p.split('::')[0])
                      for p in engine.qomp_targets(paths['tests'])}
                  and collection['command_sha256'] == digest(root/'collect-tests-command.json')
                  and collection['command_result_sha256'] == digest(root/'collect-tests-result.json'),
                  'corrected collection source/argv binding differs')
    command = lib.read(root/'collect-tests-result.json')
    existing_need(command['argv'] == lib.read(root/'collect-tests-command.json')
                  and type(command['returncode']) is int and command['returncode'] == 0
                  and command['interrupted'] is False, 'collection command failed')
    return receipt, collection


def existing_parser(engine, mode):
    import sys
    import uuid
    parser = argparse.ArgumentParser(description='Closed read-only compiled002 diagnostic verification')
    for key in ('ops-commit', 'ops-tree', 'launch-token'):
        parser.add_argument('--'+key, required=True)
    if mode == 'checks-existing':
        for key in ('verification-sha256', 'collection-sha256', 'collection-result-sha256'):
            parser.add_argument('--'+key, required=True)
    if mode == 'finalize':
        parser.add_argument('--mode', choices=('collect-existing', 'checks-existing'), required=True)
        parser.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args(sys.argv[3:])
    existing_need(str(uuid.UUID(args.launch_token)) == args.launch_token, 'invalid launch token')
    existing_need(all(re.fullmatch('[0-9a-f]{40}', v) for v in (args.ops_commit, args.ops_tree)),
                  'exact ops Git IDs required')
    if mode != 'finalize':
        args.mode = mode
    paths = existing_paths(engine)
    args.root = paths['collect' if args.mode == 'collect-existing' else 'checks']
    args.scratch = engine.QOMP_SCRATCH/args.root.name
    return args


def existing_perform(engine, lib, args):
    import signal
    roles = existing_source_gate(engine, args)  # stdlib/Git only
    paths = existing_paths(engine)
    lib.fresh_paths(args.root, args.scratch, [p for p in paths.values() if p != args.root]
                    +[engine.QOMP_BASE, lib.OLD, engine.QOMP_SCI_SOURCE])
    allocated = allocation()
    lib.allocation('oracle')  # exact free8CPU/32G/2h allocation, not the old build mode
    args.root.mkdir(mode=0o700, parents=True)
    args.scratch.mkdir(mode=0o700)
    lib.write(args.root/'owner.json', dict(token=args.launch_token, mode=args.mode))
    cwd = args.root/'cwd'
    cwd.mkdir(mode=0o700)
    os.chdir(cwd)  # before ANY Psi4 import, including identity probes
    prefix = paths['build']/'env'
    runner = engine.qomp_runner(lib, args.root, cwd, lib.environment(prefix, args.scratch), 6900)
    handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = dict(schema='existing-install-operation/1', status='FAILED', acceptance=False,
                  production_adoption=False, source=roles, mode=args.mode, allocation=allocated,
                  launch_token=args.launch_token, historical_job='15320675',
                  historical_build_status='FAILED', reconstructed_historical_full_seal=False)
    before = None
    try:
        before = existing_current(engine, lib)
        lib.write(args.root/'current-before.json', before)
        collection = None
        if args.mode == 'checks-existing':
            receipt, collection = existing_collection_gate(engine, lib, args)
            existing_need(before == lib.read(paths['collect']/'current-before.json')
                          == lib.read(paths['collect']/'current-after.json'),
                          'current attributed install differs from successful collection')
        identity = engine.qomp_identity(runner, prefix, args.scratch, lib, 'identity-before')
        existing_identity(identity, prefix)
        existing_need(identity == before['historical_identity'], 'retained/runtime identity differs')
        lib.write(args.root/'imported-identity.json', identity)
        executed = engine.qomp_pytest(runner, prefix, args.scratch, lib,
            collect=args.mode == 'collect-existing', test_source=paths['tests'],
            test_identity=dict(commit=EXISTING_TEST_COMMIT, tree=EXISTING_TEST_TREE,
                               path=str(paths['tests']/'tests/pytests')))
        if collection is not None:
            existing_need(executed['nodes'] == collection['nodes'], 'executed union differs')
            result['tests'] = qomp_junit_gate(args.root/'junit.xml', collection['nodes'])
        existing_need(engine.qomp_identity(runner, prefix, args.scratch, lib, 'identity-after')
                      == identity, 'runtime identity changed')
        runner.check()
        result['identity'] = identity
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error)[:1000])
        raise
    finally:
        try:
            after = existing_current(engine, lib)
            lib.write(args.root/'current-after.json', after)
            existing_need(before is not None and before == after, 'current002/OLD/BASE changed')
            existing_need(existing_source_gate(engine, args) == roles, 'source roles changed')
            result['preserved_current_build_and_old_base'] = True
        except BaseException as error:
            result['integrity_error'] = dict(type=type(error).__name__, message=str(error)[:1000])
            raise
        finally:
            # Never serialize caught exception/traceback objects or claim a
            # successful operation until the integrity finally has succeeded.
            lib.write(args.root/'result.json', result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    runner.check()
    if args.mode == 'collect-existing':
        lib.write(args.root/'verification.json', dict(
            schema='existing-failed-build-install-verification/1', source=roles,
            historical_job='15320675', historical_build_status='FAILED', anchors=EXISTING_ANCHORS,
            historical_component_returncodes=dict(configure=0, install=0, identity=0, collection=2),
            acceptance=False, production_adoption=False, reconstructed_historical_full_seal=False,
            current_before_sha256=digest(args.root/'current-before.json'),
            current_after_sha256=digest(args.root/'current-after.json'),
            identity=identity, allocation=allocated))
        result['status'] = 'VERIFIED_EXISTING_INSTALL_COLLECTED_NOT_QUALIFIED'
    else:
        lib.write(args.root/'installed-validation.json', dict(
            schema='existing-install-qualification/1', source=roles, tests=result['tests'],
            historical_job='15320675', historical_build_status='FAILED',
            acceptance=False, production_adoption=False, imported_identity=identity,
            verification_sha256=args.verification_sha256,
            collection_sha256=args.collection_sha256,
            collection_result_sha256=args.collection_result_sha256,
            junit_sha256=digest(args.root/'junit.xml'), allocation=allocated))
        result['status'] = 'PASSED_EXISTING_INSTALL_INTERFACE_NOT_PRODUCTION_ACCEPTANCE'
    # result.json was exclusive-written FAILED in finally; update ONLY our fresh
    # owned receipt after all postchecks. No historical artifact is opened w.
    lib.write(args.root/'successful-result.json', result)
    os.replace(args.root/'successful-result.json', args.root/'result.json')
    runner.check()


def existing_finalize(engine, lib, args):
    status = args.exit_code
    try:
        existing_need(lib.read(args.root/'owner.json') == dict(token=args.launch_token, mode=args.mode)
                      and args.root.stat().st_uid == os.getuid(), 'not our fresh owned root')
    except Exception:
        return status or 1
    failed, completed = args.root/'FAILED', args.root/'COMPLETED'
    if os.path.lexists(failed):
        previous = lib.read(failed)
        existing_need(previous['mode'] == args.mode and previous['acceptance'] is False
                      and type(previous['exit_code']) is int and previous['exit_code'] != 0,
                      'invalid failure marker')
        status = status or previous['exit_code']
    if not status:
        try:
            existing_source_gate(engine, args)
            value = lib.read(args.root/'result.json')
            expected = ('VERIFIED_EXISTING_INSTALL_COLLECTED_NOT_QUALIFIED'
                        if args.mode == 'collect-existing' else
                        'PASSED_EXISTING_INSTALL_INTERFACE_NOT_PRODUCTION_ACCEPTANCE')
            existing_need(value['status'] == expected and value['launch_token'] == args.launch_token
                          and value['acceptance'] is False and value['production_adoption'] is False
                          and value['preserved_current_build_and_old_base'] is True,
                          'incomplete read-only diagnostic')
            existing_need(existing_current(engine, lib) == lib.read(args.root/'current-before.json')
                          == lib.read(args.root/'current-after.json'), 'final current seal differs')
        except Exception:
            status = 1
    if status and os.path.lexists(completed):
        if not os.path.lexists(args.root/'COMPLETED_INVALIDATED'):
            os.link(completed, args.root/'COMPLETED_INVALIDATED')
        completed.unlink()
    marker = failed if status else completed
    if not os.path.lexists(marker):
        lib.write(marker, dict(exit_code=status, mode=args.mode, acceptance=False))
    else:
        existing_need(lib.read(marker) == dict(exit_code=status, mode=args.mode, acceptance=False),
                      'invalid existing terminal marker')
    return status


def existing_main():
    import sys
    engine = qomp_engine()
    engine.qomp_enter()
    mode = sys.argv[2] if len(sys.argv) > 2 else ''
    existing_need(mode in ('collect-existing', 'checks-existing', 'finalize'), 'closed operation required')
    args = existing_parser(engine, mode)
    lib = engine.qomp_library()
    if mode == 'finalize':
        return existing_finalize(engine, lib, args)
    existing_perform(engine, lib, args)
    return 0


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['qomp']:
        sys.exit(qomp_main())
    if sys.argv[1:2] == ['qomp-existing']:
        sys.exit(existing_main())
    main()
