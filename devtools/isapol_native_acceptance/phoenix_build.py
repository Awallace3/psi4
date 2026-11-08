"""Fresh installed-only native acceptance build inside a Phoenix allocation.

This builds the exact clean source snapshot, never an old stage/core. Invoke
with the pinned expected commit/tree and a new private run root. No submission,
retry, external science engine or quantum calculation is performed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=SOURCE, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_source(repository, snapshot, commit, tree, env):
    """Create a private commit-derived build source; require real source files."""
    subprocess.run(['git', 'worktree', 'add', '--detach', str(snapshot), commit],
                   cwd=repository, env=env, check=True)
    # Git copies the sparse operational checkout's worktree configuration.
    # Materialize the NEW private build tree, without modifying the launcher.
    subprocess.run(['git', 'sparse-checkout', 'disable'], cwd=snapshot, env=env, check=True)
    required = ('CMakeLists.txt', 'cmake/psi4OptionsTools.cmake', 'psi4/src/export_isapol.cc',
                'psi4/driver/procrouting/isapol_benzene_acceptance.py')
    if any(not (snapshot/path).is_file() for path in required):
        raise RuntimeError('incomplete private build checkout')
    for args, expected in ((['rev-parse', 'HEAD'], commit), (['rev-parse', 'HEAD^{tree}'], tree),
                           (['status', '--porcelain'], '')):
        if subprocess.check_output(['git', *args], cwd=snapshot, env=env, text=True).strip() != expected:
            raise RuntimeError('private build source identity mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--tree', required=True)
    args = parser.parse_args()
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('USER') != 'awallace43'
            or os.environ.get('SLURMD_NODENAME', '').split('.')[0] != host
            or re.fullmatch(r'atl\d+(?:-\d+){5}', host) is None
            or os.environ.get('SLURM_CLUSTER_NAME') != 'phoenix-slurm'):
        raise RuntimeError('build requires the authorized Phoenix compute allocation')
    scontrol = '/opt/slurm/current/bin/scontrol'
    allocation = subprocess.check_output(
        [scontrol, 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True)
    fields = dict(item.split('=', 1) for item in allocation.split() if '=' in item)
    if (fields.get('QOS') != 'embers' or fields.get('Account') != 'gts-cs207-chemx'
            or fields.get('JobState') != 'RUNNING'
            or fields.get('UserId', '').split('(')[0] != 'awallace43'
            or fields.get('BatchHost') != host):
        raise RuntimeError('effective allocation is not the authorized running free-embers build')
    if git('rev-parse', 'HEAD') != args.commit or git('rev-parse', 'HEAD^{tree}') != args.tree:
        raise RuntimeError('source commit/tree mismatch')
    if git('status', '--porcelain'):
        raise RuntimeError('source is not clean')
    root, base = args.root.resolve(), args.dependencies.resolve()
    if (not str(root).startswith('/storage/project/r-cs207-0/awallace43/runs/ccasp-native-')
            or base != Path('/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt').resolve()):
        raise RuntimeError('private build/dependency root outside declared Phoenix locations')
    root.mkdir(parents=True, exist_ok=True)
    # Claim atomically; concurrent allocations may not share an install prefix.
    (root/'.build-claim').mkdir(mode=0o700, exist_ok=False)
    (root/'.build-claim/job.txt').write_text(os.environ['SLURM_JOB_ID']+'\n')
    (root/'allocation.json').write_text(json.dumps(fields, indent=2))
    for name in ('env', 'build'):
        if (root/name).exists():
            raise RuntimeError('refusing existing private build/install: '+name)
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith('PIP_'):
            env.pop(key)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PSIPATH', 'PSIDATADIR', 'LD_PRELOAD',
                'GCC_ROOT', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH',
                'LIBRARY_PATH', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS',
                'CMAKE_PREFIX_PATH', 'CMAKE_TOOLCHAIN_FILE'):
        env.pop(key, None)
    env.update(PATH=f'{base}/bin:/usr/bin:/bin', LD_LIBRARY_PATH=str(base/'lib'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.update(PIP_CONFIG_FILE='/dev/null', PIP_DISABLE_PIP_VERSION_CHECK='1')
    tmp = Path('/storage/scratch1/3/awallace43')/('ccasp-native-build-'+os.environ['SLURM_JOB_ID'])
    tmp.mkdir(exist_ok=False)
    env.update(TMPDIR=str(tmp), SCRATCH=str(tmp), PSI_SCRATCH=str(tmp))
    snapshot = root/'source'
    materialize_source(SOURCE, snapshot, args.commit, args.tree, env)
    for compiler, frontend in (('cc', 'cc1'), ('c++', 'cc1plus')):
        executable = base/('bin/x86_64-conda-linux-gnu-'+compiler)
        actual = subprocess.check_output(
            [str(executable), '-print-prog-name='+frontend], env=env, text=True).strip()
        if not Path(actual).is_file():
            raise RuntimeError('selected compiler frontend unavailable: '+actual)
    subprocess.run([str(base/'bin/python3.13'), '-I', '-B', '-m', 'venv', str(root/'env')],
                   env=env, check=True)
    python = root/'env/bin/python'
    env['PATH'] = f'{root}/env/bin:'+env['PATH']
    subprocess.run([
        str(python), '-I', '-B', '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
        'numpy==2.4.3', 'scipy==1.17.1', 'qcelemental==0.50.4',
        'qcengine==0.50.0', 'pytest==9.0.3', 'pybind11==3.0.2',
        'https://github.com/psi-rking/optking/archive/0.5.0.tar.gz',
    ], env=env, check=True)
    frozen = subprocess.check_output(
        [str(python), '-I', '-B', '-m', 'pip', 'freeze'], env=env, text=True)
    (root/'dependencies.freeze.txt').write_text(frozen)
    values = dict(
        CMAKE_BUILD_TYPE='Release', CMAKE_INSTALL_PREFIX=str(root/'env'),
        PYMOD_INSTALL_LIBDIR='/python3.13/site-packages',
        CMAKE_C_COMPILER=str(base/'bin/x86_64-conda-linux-gnu-cc'),
        CMAKE_CXX_COMPILER=str(base/'bin/x86_64-conda-linux-gnu-c++'),
        CMAKE_PREFIX_PATH=str(base), CMAKE_FIND_PACKAGE_PREFER_CONFIG='ON',
        CMAKE_FIND_USE_PACKAGE_REGISTRY='OFF', CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY='OFF',
        ENABLE_Einsums='OFF', ENABLE_adcc='OFF', ENABLE_CYTHONIZE='OFF',
        ENABLE_OPENMP='ON', ENABLE_XHOST='OFF',
        CMAKE_C_FLAGS='-march=x86-64-v3', CMAKE_CXX_FLAGS='-march=x86-64-v3',
        psi4_CMAKE_UNITY_BUILD='ON', psi4_CMAKE_UNITY_BUILD_BATCH_SIZE='8',
        psi4_ENABLE_PRECOMPILE_HEADERS='OFF', BUILD_TESTING='ON',
        BUILD_Libint2_GENERATOR='OFF', FC_SYMBOL='2', MAX_AM_ERI='5',
        BLAS_TYPE='MKL', BLAS_LIBRARIES=str(base/'lib/libmkl_rt.so'),
        LAPACK_LIBRARIES=str(base/'lib/libmkl_rt.so'),
        BLAS_INCLUDE_DIRS=str(base/'include'), LAPACK_INCLUDE_DIRS=str(base/'include'),
        Python_EXECUTABLE=str(python), Python_ROOT_DIR=str(base),
        Python_INCLUDE_DIR=str(base/'include/python3.13'),
        Python_LIBRARY=str(base/'lib/libpython3.13.so'),
        Python_NumPy_INCLUDE_DIR=str(root/'env/lib/python3.13/site-packages/numpy/_core/include'),
        Libint2_DIR=str(base/'lib/cmake/libint2'), Libxc_DIR=str(base/'lib/cmake/Libxc'),
        gau2grid_DIR=str(base/'share/cmake/gau2grid'),
        pybind11_DIR=str(root/'env/lib/python3.13/site-packages/pybind11/share/cmake/pybind11'))
    for package in ('pybind11', 'gau2grid', 'Libint2', 'Libxc',
                    'qcelemental', 'qcengine', 'optking'):
        values['CMAKE_INSIST_FIND_PACKAGE_'+package] = 'ON'
    cmake = str(base/'bin/cmake')
    configure = [cmake, '-S', str(snapshot), '-B', str(root/'build'), '-G', 'Ninja']
    configure += ['-D'+name+'='+value for name, value in values.items()]
    build = [cmake, '--build', str(root/'build'), '--target', 'install', '-j', '6']
    record = dict(commit=args.commit, tree=args.tree, source=str(snapshot),
                  launcher_source=str(SOURCE), root=str(root),
                  dependencies=str(base), configure=configure, build=build,
                  job=os.environ['SLURM_JOB_ID'], started=time.time(), status='started')
    (root/'build-record.json').write_text(json.dumps(record, indent=2))
    subprocess.run(configure, env=env, cwd=root, check=True)
    subprocess.run(build, env=env, cwd=root, check=True)
    for command, expected in ((['rev-parse', 'HEAD'], args.commit),
                              (['rev-parse', 'HEAD^{tree}'], args.tree),
                              (['status', '--porcelain'], '')):
        if subprocess.check_output(['git', *command], cwd=snapshot, text=True).strip() != expected:
            raise RuntimeError('private source snapshot changed during build')
    imported = json.loads(subprocess.check_output([
        str(python), '-I', '-B', '-c',
        'import json,psi4;print(json.dumps({"package":psi4.__file__,"core":psi4.core.__file__}))',
    ], cwd=root, env=env, text=True))
    package = Path(imported['package']).resolve().parent
    if package != root/'env/lib/python3.13/site-packages/psi4':
        raise RuntimeError('fallback/foreign Psi4 package imported')
    if not Path(imported['core']).resolve().is_relative_to(package):
        raise RuntimeError('foreign native core imported')
    files = {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.so')}
    sources = list((snapshot/'psi4/driver/procrouting').glob('isapol_*.py'))
    sources += [snapshot/'psi4/driver/p4util/util.py',
                snapshot/'psi4/driver/procrouting/scf_proc/_scf_seal.py',
                snapshot/'psi4/driver/procrouting/scf_proc/scf_iterator.py']
    for source in sources:
        relative = source.relative_to(snapshot/'psi4').as_posix()
        if files.get(relative) != sha(source):
            raise RuntimeError('production install mismatch: '+relative)
    runtime_files = {p.relative_to(root/'env').as_posix(): sha(p)
                     for p in sorted((root/'env').rglob('*'))
                     if p.is_file() and p.suffix != '.pyc'}
    directories = ['.'] + [p.relative_to(root/'env').as_posix()
                            for p in sorted((root/'env').rglob('*')) if p.is_dir()]
    # Seal ordinary external runtime dependencies separately from the private
    # venv; base site-packages are not scientific inputs and remain forbidden.
    external = {p.resolve() for p in (base/'lib').glob('*.so*') if p.is_file()}
    external.add((base/'bin/python3.13').resolve())
    external.add((base/'ssl/openssl.cnf').resolve())
    stdlib = base/'lib/python3.13'
    external.update(p.resolve() for p in stdlib.rglob('*')
                    if p.is_file() and p.suffix != '.pyc'
                    and not p.is_relative_to(stdlib/'site-packages'))
    external_runtime_files = {str(p): sha(p) for p in sorted(external)}
    identity = dict(
        schema='fresh-native-build-identity/1', clean_source=True, fresh_build=True,
        commit=args.commit, tree=args.tree, installed_package=str(package),
        installed_core=imported['core'], installed_files=files,
        runtime_files=runtime_files, runtime_directories=directories,
        external_runtime_files=external_runtime_files,
        source_snapshot=str(snapshot), allocation=fields,
        proof_limits='Private clean commit-derived source and sealed ordinary dependencies; normal host/OS trust, not hostile-host or TOCTOU proof.')
    with (root/'build-identity.json').open('x') as stream:
        json.dump(identity, stream, indent=2)
    record.update(status='complete', finished=time.time(),
                  build_identity_sha256=sha(root/'build-identity.json'))
    (root/'build-record.json').write_text(json.dumps(record, indent=2))
    print('PHOENIX_NATIVE_BUILD_COMPLETE', args.commit, root, flush=True)


# New closed mode. Historical main() and its producer/provisioning behavior
# above are deliberately unchanged; qomp never calls its online provisioning.
QOMP_SCI = 'fb11b78d1f44658d3a9d62365b77f3d217f53129'
QOMP_TREE = '3a1922c885498131b3b51b7214dcab429cac073e'
QOMP_ROOT = Path('/storage/project/r-cs207-0/awallace43')
QOMP_RUNS = QOMP_ROOT/'runs'
QOMP_SCI_SOURCE = QOMP_ROOT/'gits/psi4.ccasp-native-s66-qomp-fb11-science001'
QOMP_OPS_SOURCE = QOMP_ROOT/'gits/psi4.ccasp-native-s66-qomp-ops001'
QOMP_BUILD = QOMP_RUNS/'ccasp-native-s66-qomp-fb11-build001'
QOMP_CHECK = QOMP_RUNS/'ccasp-native-s66-qomp-fb11-installed001'
QOMP_SCRATCH = Path('/storage/scratch1/3/awallace43')
QOMP_BASE = QOMP_ROOT/'miniconda/envs/p4cuest_sapt'
QOMP_LIBRARY_COMMIT = '8573ded3233001b50d99c26bf9a577e13b68c956'
QOMP_LIBRARY_PATH = 'devtools/isapol_q_omp/ops.py'
QOMP_LIBRARY_SHA = 'f14a0e7be6cc73544de07b0427dd4dba6a5d8b5b137d0adc7586dd5e4bd269e3'
QOMP_OPS_FILES = frozenset((
    'devtools/isapol_native_acceptance/phoenix_build.py',
    'devtools/isapol_native_acceptance/qomp.sbatch',
    'devtools/isapol_s66_campaign/phoenix_installed_checks.py',
    'devtools/isapol_s66_campaign/test_qomp_production_ops.py',
    'devtools/isapol_s66_campaign/QOMP_QUALIFICATION.md',
))
QOMP_FOCUSED = {
    'test_isapol_q_parallel.py': 34, 'test_isapol_native_performance.py': 18,
    'test_isapol_partitioned_response.py': 32, 'test_isapol_isa_grid_accounting.py': 9,
}
QOMP_MOLECULAR = (
    'test_isapol_distribution.py::test_isa_nonconvergence_and_admission',
    'test_isapol_distribution.py::test_isa_plan_charges_drho_refinement_once_and_to_the_cap',
)


def qomp_need(ok, message):
    if not ok:
        raise ValueError(message)


def qomp_git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def qomp_module(raw, name, filename):
    import types
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    sys.modules[name] = module
    exec(compile(raw, str(filename), 'exec'), module.__dict__)
    return module


def qomp_library():
    """Execute exactly reviewed Git bytes, not a checkout/cache/search result."""
    raw = subprocess.check_output(['git', '-C', str(SOURCE), 'show',
                                   QOMP_LIBRARY_COMMIT+':'+QOMP_LIBRARY_PATH])
    qomp_need(hashlib.sha256(raw).hexdigest() == QOMP_LIBRARY_SHA, '8573 helper bytes differ')
    return qomp_module(raw, '_qomp_pinned_8573', QOMP_LIBRARY_PATH)


def qomp_checks_module():
    path = SOURCE/'devtools/isapol_s66_campaign/phoenix_installed_checks.py'
    return qomp_module(path.read_bytes(), '_qomp_installed_engine', path)


def qomp_source_gate(commit, tree):
    qomp_need(SOURCE == QOMP_OPS_SOURCE, 'unexpected closed ops source path')
    qomp_need(qomp_git(SOURCE, 'rev-parse', 'HEAD') == commit
              and qomp_git(SOURCE, 'rev-parse', 'HEAD^{tree}') == tree, 'ops source identity differs')
    qomp_need(qomp_git(SOURCE, 'merge-base', QOMP_SCI, commit) == QOMP_SCI,
              'ops must descend from exact reviewed fb11')
    changed = set(qomp_git(SOURCE, 'diff', '--name-only', QOMP_SCI, commit).splitlines())
    qomp_need(changed <= QOMP_OPS_FILES, 'ops changed files outside exact allowlist')
    for source, expected_commit, expected_tree in (
            (SOURCE, commit, tree), (QOMP_SCI_SOURCE, QOMP_SCI, QOMP_TREE)):
        qomp_need(qomp_git(source, 'rev-parse', '--show-toplevel') == str(source.resolve()),
                  'source must be Git root')
        qomp_need(qomp_git(source, 'rev-parse', 'HEAD') == expected_commit
                  and qomp_git(source, 'rev-parse', 'HEAD^{tree}') == expected_tree,
                  'scientific/test source identity differs')
        qomp_need(not qomp_git(source, 'status', '--porcelain', '--untracked-files=all'),
                  'source must remain clean')
    detached = subprocess.run(['git', '-C', str(QOMP_SCI_SOURCE), 'symbolic-ref', '-q', 'HEAD'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    qomp_need(detached.returncode == 1, 'scientific source must be detached exact fb11')
    return dict(scientific_commit=QOMP_SCI, scientific_tree=QOMP_TREE,
                scientific_source=str(QOMP_SCI_SOURCE), test_source_commit=QOMP_SCI,
                ops_source=str(SOURCE), ops_commit=commit, ops_tree=tree)


def qomp_inventory(root, lib):
    """Owned, fixed-prefix byte/namespace/link snapshot, not a write sandbox."""
    rows = {}
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows[name] = dict(kind='link', target=os.readlink(path))
        elif path.is_file():
            rows[name] = dict(kind='file', sha256=lib.sha(path))
        elif path.is_dir():
            rows[name] = dict(kind='directory')
        else:
            raise ValueError('unexpected object in fixed runtime/dependency inventory')
    return rows


def qomp_protected(lib):
    manifest = lib.old_seal()
    qomp_checks_module().verify_runtime(lib.OLD/'env', manifest)
    lib.approved_record(lib.OLD/'build-record.json')
    return dict(old=qomp_inventory(lib.OLD/'env', lib),
                base=qomp_inventory(QOMP_BASE, lib),
                manifest_sha256=lib.sha(lib.OLD/'build-identity.json'),
                record_sha256=lib.sha(lib.OLD/'build-record.json'))


def qomp_runner(lib, root, cwd, env, seconds):
    class RecordedRunner(lib.Runner):
        @property
        def child(self):
            return getattr(self, '_child', None)

        @child.setter
        def child(self, value):
            self._child = value
            if value is not None:
                self._last_child = value

        def run(self, label, command):
            self._last_child = None
            try:
                return super().run(label, command)
            finally:
                child = self._last_child
                lib.write(root/(label+'-result.json'), dict(
                    argv=list(map(str, command)),
                    returncode=None if child is None else child.returncode,
                    interrupted=self.interrupted))
                self._last_child = None
    return RecordedRunner(root, cwd, env, seconds)


def qomp_targets():
    legacy = qomp_checks_module().TEST_FILES
    qomp_need(len(legacy) == len(set(legacy)) == 11
              and not set(legacy) & set(QOMP_FOCUSED)
              and 'test_isapol_distribution.py' not in legacy, 'test inventory overlap')
    names = list(legacy)+list(QOMP_FOCUSED)+list(QOMP_MOLECULAR)
    return [str(QOMP_SCI_SOURCE/'tests/pytests'/name) for name in names]


def qomp_nodes_gate(value, *, executed=False):
    nodes = value['nodes']
    qomp_need(type(nodes) is list and len(nodes) == len(set(nodes)) == 556,
              'exact unique 556 collected nodes required')
    legacy = set(qomp_checks_module().TEST_FILES)
    groups = dict(legacy=0, focused=0, molecular=0)
    focused = dict.fromkeys(QOMP_FOCUSED, 0)
    seen_legacy = set()
    for node in nodes:
        qomp_need(type(node) is str and node.startswith('tests/pytests/'), 'foreign test node')
        file = node.split('::')[0].removeprefix('tests/pytests/')
        if file in legacy:
            groups['legacy'] += 1
            seen_legacy.add(file)
        elif file in QOMP_FOCUSED:
            focused[file] += 1
            groups['focused'] += 1
        else:
            qomp_need(node.removeprefix('tests/pytests/') in QOMP_MOLECULAR,
                      'unexpected molecular/test node')
            groups['molecular'] += 1
    qomp_need(seen_legacy == legacy and groups == dict(legacy=461, focused=93, molecular=2)
              and focused == QOMP_FOCUSED, 'test group/module counts differ')
    qomp_need(all(type(value[k]) is int and value[k] == 0
                  for k in ('returncode', 'collection_errors', 'collection_skips')),
              'collection failed/skipped')
    if executed:
        qomp_need(value['call_nodes'] == nodes
                  and all(type(value[k]) is int and value[k] == 0
                          for k in ('failed_reports', 'skipped_reports')),
                  'tests failed/skipped/reordered/missing')
    return dict(groups=groups, focused_modules=focused, tests=556)


def qomp_pytest(runner, prefix, scratch, lib, *, collect):
    """Fixed selectors, installed helper directory only; no source path fallback."""
    package = prefix/'lib/python3.13/site-packages/psi4'
    output = runner.root/('collection-nodes.json' if collect else 'executed-nodes.json')
    arguments = ['--noconftest', '-c', '/dev/null', '-p', 'no:cacheprovider',
                 '--import-mode=importlib', '--rootdir='+str(QOMP_SCI_SOURCE), '-q']
    arguments += ['--collect-only'] if collect else [
        '--basetemp='+str(scratch/'pytest'), '--junitxml='+str(runner.root/'junit.xml')]
    arguments += qomp_targets()
    # Assertion precedes even pytest/helper imports. Test source is fb11;
    # the ONLY added helper directory is the verified private installed tests.
    script = """import json,pathlib,sys,psi4
package=pathlib.Path(psi4.__file__).resolve().parent
assert package == pathlib.Path(PACKAGE)
assert pathlib.Path(psi4.core.__file__).resolve().parent == package
sys.path.insert(0,str(package/'tests'))
import pytest
class Inventory:
 def __init__(self):
  self.nodes=[]; self.call_nodes=[]; self.collection_errors=0
  self.collection_skips=0; self.failed_reports=0; self.skipped_reports=0
 def pytest_collection_finish(self,session):
  self.nodes=[item.nodeid for item in session.items]
 def pytest_collectreport(self,report):
  self.collection_errors+=int(report.failed); self.collection_skips+=int(report.skipped)
 def pytest_runtest_logreport(self,report):
  self.failed_reports+=int(report.failed); self.skipped_reports+=int(report.skipped)
  if report.when=='call' and report.passed: self.call_nodes.append(report.nodeid)
plugin=Inventory()
code=int(pytest.main(ARGUMENTS,plugins=[plugin]))
with open(OUTPUT,'x') as stream:
 json.dump(dict(vars(plugin),returncode=code),stream,indent=2)
sys.exit(code)
"""
    command = [prefix/'bin/python', '-I', '-B', '-X', 'pycache_prefix='+str(scratch/'cache'),
               '-c', 'PACKAGE='+repr(str(package))+'\nARGUMENTS='+repr(arguments)
               +'\nOUTPUT='+repr(str(output))+'\n'+script]
    runner.run('collect-tests' if collect else 'installed-tests', command)
    value = lib.read(output)
    counts = qomp_nodes_gate(value, executed=not collect)
    if collect:
        lib.write(runner.root/'collection.json', dict(
            value, group_counts=counts['groups'], focused_modules=counts['focused_modules'],
            source=dict(commit=QOMP_SCI, tree=QOMP_TREE, path=str(QOMP_SCI_SOURCE/'tests/pytests')),
            targets=qomp_targets(), test_source_sha256={
                Path(path.split('::')[0]).name: lib.sha(path.split('::')[0]) for path in qomp_targets()},
            command_sha256=lib.sha(runner.root/'collect-tests-command.json'),
            command_result_sha256=lib.sha(runner.root/'collect-tests-result.json')))
    return value


def qomp_identity(runner, prefix, scratch, lib, label):
    value = lib.identity(runner, prefix/'bin/python', prefix, scratch, label)
    package = prefix/'lib/python3.13/site-packages/psi4'
    qomp_need(value['package'] == str(package/'__init__.py')
              and Path(value['core']).parent == package, 'foreign installed identity path')
    pair = value['implementation']
    qomp_need(set(pair) == {'core', 'isapol_python'}
              and all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in pair.values())
              and pair['core'] == lib.sha(value['core']), 'invalid/changed implementation identity')
    qomp_need(all(pair[k] != lib.OLD_ID[k] for k in pair), 'old implementation is not the new build')
    return value


def qomp_build(runner, lib, result, args):
    root, prefix = runner.root, runner.root/'env'
    original = lib.old_seal()
    record = lib.approved_record(lib.OLD/'build-record.json')
    materialize_source(QOMP_SCI_SOURCE, root/'source', QOMP_SCI, QOMP_TREE, runner.env)
    lib.write(root/'clone-links.json', lib.clone_runtime(lib.OLD/'env', prefix, QOMP_BASE))
    lib.dependency_gate(prefix, original)
    for compiler, frontend in (('cc', 'cc1'), ('c++', 'cc1plus')):
        actual = runner.run('compiler-'+compiler.replace('+', 'x'), [
            QOMP_BASE/('bin/x86_64-conda-linux-gnu-'+compiler), '-print-prog-name='+frontend]).strip()
        qomp_need(Path(actual).is_file() and Path(actual).resolve().is_relative_to(QOMP_BASE),
                  'foreign/missing compiler frontend')
    command = lib.configure(record, root/'source', root)
    runner.run('configure', command)
    build = [QOMP_BASE/'bin/cmake', '--build', root/'build', '--target', 'install', '-j', '2']
    runner.run('build-install', build)
    package = prefix/'lib/python3.13/site-packages/psi4'
    checks = lib.installed_python_gate(root/'source', root/'build', package,
                                       original['installed_files'])
    lib.write(root/'installed-python-checks.json', checks)
    lib.dependency_gate(prefix, original)
    before = qomp_identity(runner, prefix, args.scratch, lib, 'identity-before')
    installed_before = qomp_inventory(prefix, lib)
    qomp_pytest(runner, prefix, args.scratch, lib, collect=True)
    qomp_need(qomp_identity(runner, prefix, args.scratch, lib, 'identity-after') == before
              and qomp_inventory(prefix, lib) == installed_before, 'runtime changed during collection')
    for values, expected in ((('rev-parse', 'HEAD'), QOMP_SCI),
                             (('rev-parse', 'HEAD^{tree}'), QOMP_TREE),
                             (('status', '--porcelain', '--untracked-files=all'), '')):
        qomp_need(qomp_git(root/'source', *values) == expected, 'compiled snapshot changed')
    files = {p.relative_to(package).as_posix(): lib.sha(p) for p in sorted(package.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.so')}
    runtime = {p.relative_to(prefix).as_posix(): lib.sha(p) for p in sorted(prefix.rglob('*'))
               if p.is_file() and p.suffix != '.pyc'}
    directories = ['.']+[p.relative_to(prefix).as_posix() for p in sorted(prefix.rglob('*')) if p.is_dir()]
    external = {name: lib.sha(name) for name in original['external_runtime_files']}
    qomp_need(external == original['external_runtime_files'], 'original dependencies changed')
    manifest = dict(schema='fresh-native-build-identity/1', clean_source=True, fresh_build=True,
                    commit=QOMP_SCI, tree=QOMP_TREE, installed_package=str(package),
                    installed_core=before['core'], installed_files=files, runtime_files=runtime,
                    runtime_directories=directories, external_runtime_files=external,
                    source_snapshot=str(root/'source'), allocation=result['allocation'],
                    operations=result['source'], provisioning='offline qualified1aaa dependency clone',
                    old_manifest_sha256=lib.OLD_MANIFEST,
                    original_configure_sha256=lib.APPROVED_RECORD_SHA,
                    proof_limits='Ordinary host trust, byte/namespace/link seals; not a write sandbox.')
    qomp_checks_module().verify_runtime(prefix, manifest)
    lib.write(root/'build-identity.json', manifest)
    lib.write(root/'build-record.json', dict(
        commit=QOMP_SCI, tree=QOMP_TREE, source=str(root/'source'),
        launcher_source=str(SOURCE), root=str(root), dependencies=str(QOMP_BASE),
        configure=command, build=list(map(str, build)), job=os.environ['SLURM_JOB_ID'],
        status='complete', build_identity_sha256=lib.sha(root/'build-identity.json'),
        operations=result['source'], original_configure_sha256=lib.APPROVED_RECORD_SHA,
        provisioning='offline qualified1aaa dependency clone',
        command_results={name: lib.read(root/(name+'-result.json'))
                         for name in ('configure', 'build-install')}))
    result.update(status='FRESH_BUILD_COMPLETE_NOT_QUALIFIED', identity=before,
                  build_identity_sha256=lib.sha(root/'build-identity.json'),
                  build_record_sha256=lib.sha(root/'build-record.json'),
                  collection_sha256=lib.sha(root/'collection.json'))


def qomp_parser(mode):
    import uuid
    parser = argparse.ArgumentParser(description='Closed fb11 offline build/installed qualification')
    parser.add_argument('--ops-commit', required=True)
    parser.add_argument('--ops-tree', required=True)
    parser.add_argument('--launch-token', required=True)
    if mode == 'checks':
        for name in ('build-identity-sha256', 'build-record-sha256', 'collection-sha256'):
            parser.add_argument('--'+name, required=True)
    if mode == 'finalize':
        parser.add_argument('--mode', choices=('build', 'checks'), required=True)
        parser.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args(sys.argv[3:])
    qomp_need(str(uuid.UUID(args.launch_token)) == args.launch_token, 'invalid launch token')
    for name in ('ops_commit', 'ops_tree'):
        qomp_need(re.fullmatch('[0-9a-f]{40}', getattr(args, name)), 'exact ops Git identity required')
    if mode != 'finalize':
        args.mode = mode
    args.root = QOMP_BUILD if args.mode == 'build' else QOMP_CHECK
    args.scratch = QOMP_SCRATCH/args.root.name
    return args


def qomp_perform(args, action):
    import signal
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    source = qomp_source_gate(args.ops_commit, args.ops_tree)
    lib = qomp_library()
    lib.fresh_paths(args.root, args.scratch, [SOURCE, QOMP_SCI_SOURCE, lib.OLD, QOMP_BASE]
                    +([QOMP_BUILD] if args.mode == 'checks' else []))
    allocation = qomp_checks_module().allocation()  # Running compute host/user/cluster, not login.
    lib.allocation('build' if args.mode == 'build' else 'oracle')  # Exact finite8CPU/32G route.
    args.root.mkdir(mode=0o700, parents=True)
    args.scratch.mkdir(mode=0o700)
    lib.write(args.root/'owner.json', dict(token=args.launch_token, mode=args.mode))
    cwd = args.root/'cwd'
    cwd.mkdir(mode=0o700)
    os.chdir(cwd)
    prefix = (args.root if args.mode == 'build' else QOMP_BUILD)/'env'
    runner = qomp_runner(lib, args.root, cwd, lib.environment(prefix, args.scratch),
                         28200 if args.mode == 'build' else 6900)
    handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = dict(schema='qomp-production-operation/1', status='FAILED', acceptance=False,
                  production_adoption=False, mode=args.mode, source=source,
                  allocation=allocation, launch_token=args.launch_token)
    protected = None
    try:
        protected = qomp_protected(lib)
        lib.write(args.root/'protected-before.json', protected)
        qomp_need(not any(k in os.environ for k in ('OMP_PROC_BIND', 'OMP_PLACES')),
                  'unexpected inherited affinity policy')
        action(runner, lib, result, args)
        runner.check()
    except BaseException as error:
        result.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error)[:1000]))
        raise
    finally:
        try:
            after = qomp_protected(lib)
            lib.write(args.root/'protected-after.json', after)
            qomp_need(protected is not None and after == protected, 'OLD/BASE changed')
            result['old_and_base_unchanged'] = True
            qomp_source_gate(args.ops_commit, args.ops_tree)
            runner.check()
        except BaseException as error:
            result.update(status='FAILED', integrity_error=dict(
                type=type(error).__name__, message=str(error)[:1000]))
            raise
        finally:
            lib.write(args.root/'result.json', result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def qomp_finalize(args):
    lib = qomp_library()
    status = args.exit_code
    try:
        qomp_need(lib.read(args.root/'owner.json') == dict(token=args.launch_token, mode=args.mode)
                  and args.root.stat().st_uid == os.getuid(), 'not our owned operation')
    except Exception:
        return status or 1
    failed, completed = args.root/'FAILED', args.root/'COMPLETED'
    if failed.exists():
        previous = lib.read(failed)
        qomp_need(type(previous['exit_code']) is int and previous['exit_code'] != 0
                  and previous['mode'] == args.mode and previous['acceptance'] is False,
                  'invalid existing failure marker')
        status = status or previous['exit_code']
    if status == 0:
        try:
            qomp_source_gate(args.ops_commit, args.ops_tree)
            result = lib.read(args.root/'result.json')
            expected = ('FRESH_BUILD_COMPLETE_NOT_QUALIFIED' if args.mode == 'build'
                        else 'PASSED_INSTALLED_INTERFACE_NOT_PRODUCTION_ACCEPTANCE')
            qomp_need(result['status'] == expected and result['old_and_base_unchanged'] is True
                      and result['launch_token'] == args.launch_token
                      and result['acceptance'] is False, 'incomplete operation')
        except Exception:
            status = 1
    if status and completed.exists():
        archived = args.root/'COMPLETED_INVALIDATED'
        if not archived.exists():
            os.link(completed, archived)
        completed.unlink()
    marker = failed if status else completed
    if not marker.exists():
        lib.write(marker, dict(exit_code=status, mode=args.mode, acceptance=False))
    elif not status:
        qomp_need(lib.read(marker) == dict(exit_code=0, mode=args.mode, acceptance=False),
                  'invalid completion marker')
    return status


def qomp_main():
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    operation = sys.argv[2] if len(sys.argv) > 2 else ''
    qomp_need(operation in ('build', 'finalize'), 'closed qomp build/finalize operation required')
    args = qomp_parser(operation)
    if operation == 'finalize':
        return qomp_finalize(args)
    qomp_perform(args, qomp_build)
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['qomp']:
        sys.exit(qomp_main())
    main()
