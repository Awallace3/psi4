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


if __name__ == '__main__':
    main()
