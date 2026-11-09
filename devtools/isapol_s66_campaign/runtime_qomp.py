"""Finite qomp runtime admission and decoded-FD projection. No native imports.

Normal host/OS trust, not an atomic filesystem or hostile-host sandbox.
Only the reviewed view can select runtime paths; never walk BUILD/OLD/BASE.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re
import stat

PROJECT = '/storage/project/r-cs207-0/awallace43'
BUILD = PROJECT+'/runs/ccasp-native-s66-qomp-fb11-build002'
PREFIX = BUILD+'/env'
BASE = PROJECT+'/miniconda/envs/p4cuest_sapt'
SCRATCH_ROOT = Path('/storage/scratch1/3/awallace43')
VIEW_SHA256 = '29d688df80fc3eb55a516042d86f4913100a8e845a3c8152acaf1a1eccf2ad6e'
CFG_SHA256 = '4a81843f74dd6d3e63107b38166deba856394189518a2fa658be85748446e9f9'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def helper(name):
    need(name in ('qualified_qomp', 'bank_runtime_gate'), 'closed runtime helper')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_qomp_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def relative(name):
    need(type(name) is str and name and not name.startswith('/') and '\x00' not in name
         and '..' not in PurePosixPath(name).parts and str(PurePosixPath(name)) == name,
         'unsafe runtime member name')


def validate_view(view):
    """Structural checks also used by pure tests; load_view owns byte authority."""
    q = helper('qualified_qomp')
    need(type(view) is dict, 'typed runtime view required')
    need(view['schema'] == 'candidate-conservative-qomp-runtime-view/1'
         and view['status'] == 'NOT_ACTIVE_PENDING_COVERAGE_REVIEW'
         and view['closure_approved'] is False and view['production_adoption'] is False
         and view['scientific_model_acceptance'] is False,
         'reviewed candidate flags must NOT be rewritten')
    need(view['inventory_sha256'] == q.INVENTORY_SHA256
         and view['identity']['implementation'] == q.IMPLEMENTATION
         and view['identity']['version'] == q.VERSION
         and view['qualification_job'] == '15326636' and view['historical_build']['status'] == 'FAILED',
         'runtime qualification identity differs')
    for role, expected in (
            ('compiled',(q.SCIENTIFIC_COMMIT,q.SCIENTIFIC_TREE)),
            ('science',(q.SCIENTIFIC_COMMIT,q.SCIENTIFIC_TREE)),
            ('tests',(q.TEST_COMMIT,q.TEST_TREE)), ('ops',(q.QUALIFYING_OPS,q.QUALIFYING_OPS_TREE))):
        need((view['source'][role]['commit'],view['source'][role]['tree']) == expected,
             'foreign runtime source role')
    need(set(view['scopes']) == {'prefix','base'}
         and view['scopes']['prefix']['root'] == PREFIX
         and view['scopes']['base']['root'] == BASE, 'foreign runtime roots')
    counts = Counter()
    for scope in view['scopes'].values():
        need(type(scope['entries']) is dict, 'typed runtime member table required')
        for name,row in scope['entries'].items():
            relative(name)
            need(type(row) is dict, 'typed runtime entry required')
            need(row['kind'] in ('file','directory','link'), 'unknown runtime kind')
            counts[row['kind']] += 1
            if row['kind'] == 'file' or row.get('target_kind') == 'file':
                need(type(row.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),
                     'invalid runtime content hash')
            if row['kind'] == 'link':
                bind = row['target_namespace_binding']
                need(type(bind) is dict and bind.get('complete_target_subtree') is True,
                     'complete runtime target namespace binding required')
                relative(bind['member'])
                target_scope = view['scopes'][bind['scope']]
                target = target_scope['entries'][bind['member']]
                need(target['kind'] == row['target_kind']
                     and row['resolved'] == str(Path(target_scope['root'])/bind['member'])
                     and (target['kind'] != 'file' or target['sha256'] == row['sha256']),
                     'unjoined runtime link binding')
    need(counts == dict(file=13247,directory=2062,link=548), 'exact runtime member counts required')
    need(view['absence_constraints'] == [dict(path=BASE+'/lib/python313.zip',
         evidence='absent in complete qualified protected_bindings.base namespace',
         inventory_sha256=q.INVENTORY_SHA256)], 'unknown runtime absence constraint')
    base_site = 'lib/python3.13/site-packages'
    need(not any(n == base_site or n.startswith(base_site+'/')
                 for n in view['scopes']['base']['entries']), 'BASE site cannot be a runtime member')


def load_view(raw):
    need(type(raw) is bytes and len(raw) == 6807878
         and hashlib.sha256(raw).hexdigest() == VIEW_SHA256, 'literal reviewed runtime view required')
    view = helper('qualified_qomp').decode(raw)
    validate_view(view)
    return view


def check_entry(root, name, row):
    """Finite exact type/raw-link/target/content check, no directory traversal."""
    relative(name)
    root, path = Path(root), Path(root)/name
    mode = path.lstat().st_mode
    if row['kind'] == 'link':
        need(stat.S_ISLNK(mode) and os.readlink(path) == row['target']
             and str(path.resolve(strict=True)) == row['resolved'], 'runtime link changed: '+name)
        mode = path.stat().st_mode
        need((stat.S_ISDIR(mode) if row['target_kind']=='directory' else stat.S_ISREG(mode)),
             'runtime link target type changed: '+name)
        if row['target_kind'] == 'file':
            need(file_sha(path) == row['sha256'], 'runtime link content changed: '+name)
    elif row['kind'] == 'directory':
        need(stat.S_ISDIR(mode) and str(path.resolve(strict=True)) == str(root/name),
             'runtime directory type/parent changed: '+name)
    else:
        need(row['kind'] == 'file' and stat.S_ISREG(mode)
             and str(path.resolve(strict=True)) == str(root/name), 'runtime file type/parent changed: '+name)
        need(file_sha(path) == row['sha256'], 'runtime file content changed: '+name)


def namespace(root, *, exclude_site=False, expected=None):
    """Only COMPLETE prefix or non-site stdlib; never walk whole BASE."""
    root = Path(root)
    need(stat.S_ISDIR(root.lstat().st_mode), 'runtime namespace root is not directory')
    out, pending = {'.'}, [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                p = Path(entry.path)
                rel = str(p.relative_to(root))
                if exclude_site and (rel == 'site-packages' or rel.startswith('site-packages/')):
                    continue  # ROOT and descendants excluded, never traversed
                if expected is not None:
                    need(rel in expected, 'unselected namespace entry: '+rel)
                out.add(rel)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(p)
    return out


def absent_zip(path):
    import errno
    try:
        Path(path).lstat()
    except FileNotFoundError as e:
        need(e.errno == errno.ENOENT and e.filename == str(path), 'unexplained zip absence')
    else:
        raise ValueError('shadow python313.zip exists, including dangling links')


def validate_cfg(raw):
    need(hashlib.sha256(raw).hexdigest() == CFG_SHA256, 'qualified pyvenv.cfg changed')
    fields = {}
    for line in raw.decode().splitlines():
        if '=' in line:
            k,v = (s.strip() for s in line.split('=',1))
            need(k not in fields, 'duplicate venv field')
            fields[k] = v
    need(fields.get('include-system-site-packages') == 'false'
         and fields.get('home') == BASE+'/bin'
         and fields.get('executable') == BASE+'/bin/python3.13'
         and fields.get('version') == '3.13.11', 'isolated venv semantics differ')


def validate_python_path(paths, *, cli=False):
    """Before/after Phi import in descriptor/API child; CLI proof also uses FD audit."""
    allowed = {BASE+'/lib/python3.13', BASE+'/lib/python3.13/lib-dynload',
               BASE+'/lib/python313.zip', PREFIX+'/lib/python3.13/site-packages'}
    if cli:
        allowed.add(PREFIX+'/bin')
    for name in paths:
        need(type(name) is str and name and Path(name).is_absolute()
             and name in allowed and str(Path(name).resolve()) == name,
             'unapproved Python search path')


def verify_runtime_view(raw):
    """Call before/after; no BUILD/object/OLD audit or whole-BASE walk."""
    view = load_view(raw)
    prefix = view['scopes']['prefix']['entries']
    base = view['scopes']['base']['entries']
    need(namespace(PREFIX, expected=prefix) == set(prefix), 'complete private runtime namespace changed')
    std = 'lib/python3.13'
    wanted = {'.' if n==std else n.removeprefix(std+'/') for n in base
              if n==std or n.startswith(std+'/')}
    need(namespace(BASE+'/'+std, exclude_site=True, expected=wanted) == wanted,
         'non-site stdlib namespace changed')
    need(not any(n.endswith('.pth') for n in prefix), 'private .pth forbidden before startup')
    absent_zip(BASE+'/lib/python313.zip')
    validate_cfg((Path(PREFIX)/'pyvenv.cfg').read_bytes())
    for scope in view['scopes'].values():
        for name,row in scope['entries'].items():
            check_entry(scope['root'], name, row)
    return view


@contextmanager
def runtime_admission(raw):
    """Caller owns child/TERM handling; post-seal runs on Python unwind too."""
    verify_runtime_view(raw)
    try:
        yield
    finally:
        verify_runtime_view(raw)


def fresh_directory(path, *, empty):
    path = Path(path)
    info = path.lstat()
    need(path.is_absolute() and stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
         and str(path.resolve()) == str(path), 'fresh owned canonical directory required')
    need(not (info.st_mode & 0o022), 'runtime directory cannot be group/world writable')
    if empty:
        with os.scandir(path) as entries:
            need(next(entries,None) is None, 'preexisting HOME/cache could shadow startup')


def cli_environment(home, cache, scratch, *, inherited_threads):
    """No inherited loader/Python/plugin/config state; keep existing thread policy."""
    scratch = Path(scratch)
    need(scratch.parent == SCRATCH_ROOT and scratch.name.startswith('ccasp-native-')
         and Path(home) == scratch/'home' and Path(cache) == scratch/'cache',
         'closed scientific scratch/HOME/cache layout required')
    fresh_directory(scratch,empty=False)
    for p in (home,cache):
        fresh_directory(p,empty=True)
    need(Path(home) != Path(cache) and str(Path(cache)) == str(Path(scratch)/'cache'),
         'exact private CLI cache layout required')
    need(set(inherited_threads) == {'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'}
         and all(type(v) is str and re.fullmatch('[1-9][0-9]*',v)
                 for v in inherited_threads.values())
         and inherited_threads['OPENBLAS_NUM_THREADS'] == '1', 'explicit existing thread policy required')
    return dict(inherited_threads, PATH=PREFIX+'/bin:'+BASE+'/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH=PREFIX+'/lib:'+BASE+'/lib', HOME=str(home),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(cache),
        TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
        LANG='C.UTF-8', LC_ALL='C.UTF-8')


def ordinary_cli_launch(raw, argv, *, scratch, execution_profile, inherited_threads, producer='grid400'):
    """Admit exact ORIGINAL -I -B CLI, not a -S replacement or Python wrapper."""
    import json
    scratch = Path(scratch)
    gate = helper('bank_runtime_gate')
    shape = gate.validate_execution_profile(execution_profile,producer=producer)
    need(inherited_threads.get('OMP_NUM_THREADS') == inherited_threads.get('MKL_NUM_THREADS')
         == str(shape['cores']), 'existing execution-shape thread contract differs')
    env = cli_environment(scratch/'home',scratch/'cache',scratch,
                          inherited_threads=inherited_threads)
    fresh_directory(scratch/'cwd',empty=False)
    verify_runtime_view(raw)  # Must precede interpreter startup/site processing.
    command = [PREFIX+'/bin/python','-I','-B','-X','pycache_prefix='+str(Path(scratch)/'cache'),
               PREFIX+'/bin/psi4', *argv]
    execution = dict(syscall='execve('+json.dumps(command[0])+', '+json.dumps(command)+', []) = 0')
    gate.cli_role(execution,env=PREFIX,scratch=scratch,execution_profile=execution_profile,producer=producer)
    return dict(command=command, environment=env, cwd=str(scratch/'cwd'))


def trace_projection(raw, lines, *, run, scratch, allowed_inputs, forbidden, execution_profile,
                     producer='grid400'):
    """Reuse decoded-FD and exact CLI roles; forbid BASE-site/shadow successful opens."""
    role_gate = helper('bank_runtime_gate')
    role_gate.validate_execution_profile(execution_profile,producer=producer)
    view = load_view(raw)
    run, scratch = Path(run), Path(scratch)
    lines = list(lines)
    allowed_inputs = {os.path.normpath(str(p)):v for p,v in allowed_inputs.items()}
    p = Path(__file__).parents[1]/'isapol_native_acceptance/trace_gate.py'
    spec = importlib.util.spec_from_file_location('_qomp_decoded_fd',p)
    gate = importlib.util.module_from_spec(spec)
    exec(compile(p.read_bytes(),str(p),'exec'),gate.__dict__)
    prefix, base = (view['scopes'][k]['entries'] for k in ('prefix','base'))
    selected = {}
    dirs = {PREFIX, BASE}
    for label, entries in ((PREFIX,prefix),(BASE,base)):
        for name,row in entries.items():
            path = str(Path(label)/name)
            if row['kind']=='file' or row.get('target_kind')=='file':
                selected[path] = row['sha256']
                if row['kind']=='link':
                    selected[row['resolved']] = row['sha256']
            elif row['kind']=='directory' or row.get('target_kind')=='directory':
                dirs.add(path)
    errors = []
    for pid,text in gate.assembled_records(lines):
        if not re.match(r'(?:open|openat|openat2|creat)\(',text):
            continue
        match = re.search(r'\)\s+=\s+(-?\d+)(?:<([^>]+)>)?',text)
        need(match is not None, 'unparseable open return')
        if int(match[1]) < 0:
            continue
        target = match[2]
        need(target and target.startswith('/') and '\\' not in target, 'missing decoded fd target')
        path = os.path.normpath(target)
        if path in allowed_inputs and file_sha(path) != allowed_inputs[path]:
            errors.append(path)  # Also check declared inputs inside owned run/scratch.
        if gate.beneath(path,BASE) or gate.beneath(path,PREFIX):
            if path not in selected and not (path in dirs and 'O_DIRECTORY' in text):
                errors.append(path)
        elif path not in allowed_inputs and (
                re.search(r'\.(?:py|pyc|gbs)$',path)
                or (('.so' in Path(path).name) and (
                    gate.beneath(path,str(run)) or gate.beneath(path,str(scratch))
                    or '/python' in path or '/site-packages/' in path))):
            errors.append(path)
    manifest = dict(
        runtime_files={n:r['sha256'] for n,r in prefix.items() if r['kind']=='file'},
        runtime_directories=[n for n,r in prefix.items() if r['kind']=='directory'
                             or r.get('target_kind')=='directory'],
        external_runtime_files={p:h for p,h in selected.items() if gate.beneath(p,BASE)})
    report = gate.check_trace(lines,env=PREFIX,base=BASE,manifest_path=run/'runtime-view.json',
        manifest=manifest,run=run,scratch=scratch,allowed_inputs=allowed_inputs,
        forbidden=tuple(forbidden)+(BASE+'/lib/python3.13/site-packages',))
    if errors:
        report['status']='failed'
        report['unselected_or_shadow_opens']=sorted(set(errors))
    if report['only_initial_python_executed']:
        report['role']=role_gate.cli_role(report['executions'][0],
            env=PREFIX,scratch=scratch,execution_profile=execution_profile,producer=producer)
    return report
