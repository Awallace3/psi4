"""Stdlib fakes only: never import Psi4 or inspect a live installed runtime."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_test_qomp_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


q = load('qualified_qomp')
rt = load('runtime_qomp')
PROFILE = load('grid400_bank_policy').execution_profile('s20')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def synthetic_receipts(monkeypatch, change=None):
    """Synthetic shape oracle; real digest authority is never synthesized."""
    roles = {
        'compiled': dict(commit=q.SCIENTIFIC_COMMIT, tree=q.SCIENTIFIC_TREE),
        'science': dict(commit=q.SCIENTIFIC_COMMIT, tree=q.SCIENTIFIC_TREE),
        'tests': dict(commit=q.TEST_COMMIT, tree=q.TEST_TREE),
        'ops': dict(commit=q.QUALIFYING_OPS, tree=q.QUALIFYING_OPS_TREE)}
    ident = dict(version=q.VERSION, implementation=q.IMPLEMENTATION,
                 package=rt.PREFIX+'/lib/python3.13/site-packages/psi4/__init__.py',
                 core=rt.PREFIX+'/lib/python3.13/site-packages/psi4/core.so')
    nodes = ['tests/test_a.py::test_'+str(i) for i in range(556)]
    docs = {
        'qualification': dict(schema='existing-install-qualification/1', tests=q.TEST_SUMMARY,
            historical_job='15320675', historical_build_status='FAILED',
            acceptance=False, production_adoption=False, source=roles,
            imported_identity=ident, allocation=dict(JobId='15326636')),
        'qualification_result': dict(
            status='PASSED_EXISTING_INSTALL_INTERFACE_NOT_PRODUCTION_ACCEPTANCE',
            preserved_current_build_and_old_base=True, source=roles,
            allocation=dict(JobId='15326636'), identity=ident),
        'identity': ident, 'collection': dict(nodes=list(nodes)),
        'executed': dict(nodes=list(nodes), call_nodes=list(nodes), returncode=0, collection_errors=0,
                         collection_skips=0, failed_reports=0, skipped_reports=0),
        'command': ['installed-tests'], 'command_result': dict(
            argv=['installed-tests'], returncode=0, interrupted=False),
        'historical_build_result': dict(status='FAILED'),
        'configure_result': dict(returncode=0), 'install_result': dict(returncode=0),
        'verification': {}, 'collection_result': {}, 'original_configure': {}}
    docs = copy.deepcopy(docs)
    if change:
        change(docs)
    xml = ET.Element('testsuites')
    suite = ET.SubElement(xml, 'testsuite')
    for i in range(556):
        ET.SubElement(suite, 'testcase', classname='tests.test_a', name='test_'+str(i))
    blobs = {key: json.dumps(value).encode() for key, value in docs.items()}
    blobs['junit'] = ET.tostring(xml)
    pins = {key: sha(value) for key, value in blobs.items()}
    for field, role in (('verification_sha256','verification'),
                        ('collection_sha256','collection'),
                        ('collection_result_sha256','collection_result'),
                        ('junit_sha256','junit')):
        docs['qualification'][field] = pins[role]
    blobs['qualification'] = json.dumps(docs['qualification']).encode()
    pins['qualification'] = sha(blobs['qualification'])
    monkeypatch.setattr(q, 'RECEIPT_SHA256', pins)
    return blobs


def test_distinct_receipts_preserve_failed_build_and_no_activation(monkeypatch):
    blobs = synthetic_receipts(monkeypatch)
    out = q.verify_receipts(blobs)
    assert out['production_adoption'] is False
    assert out['implementation_sha256'] == q.IMPLEMENTATION
    assert out['scientific_commit'] != out['installed_test_commit']
    assert 'build_identity' not in q.RECEIPT_PATHS
    with pytest.raises(ValueError, match='digest'):
        q.verify_receipts(blobs | {'qualification': blobs['qualification']+b' '})
    with pytest.raises(ValueError, match='roles'):
        q.verify_receipts({'build_identity': b'normal READY'})


@pytest.mark.parametrize('change', [
    lambda d: d['qualification'].update(historical_build_status='COMPLETED'),
    lambda d: d['qualification']['source']['compiled'].update(commit='foreign'),
    lambda d: d['qualification_result'].update(preserved_current_build_and_old_base=False),
    lambda d: d['qualification'].update(tests=dict(tests=555, failures=0, errors=0, skipped=0)),
    lambda d: d['qualification'].update(acceptance=True),
    lambda d: d['identity'].update(version='other'),
    lambda d: d['executed']['call_nodes'].reverse(),
    lambda d: d['executed'].update(collection_skips=1),
    lambda d: d['command_result'].update(interrupted=True),
    lambda d: d['install_result'].update(returncode=1),
    lambda d: d['historical_build_result'].update(status='READY_BUILD_NOT_QUALIFIED'),
])
def test_receipt_shapes_fail_even_with_synthetic_hash_authority(monkeypatch, change):
    with pytest.raises(ValueError):
        q.verify_receipts(synthetic_receipts(monkeypatch, change))


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{'])
def test_json_no_duplicate_nonfinite_or_malformed(raw):
    with pytest.raises(ValueError):
        q.decode(raw)


def test_view_authority_is_literal_not_current_files():
    with pytest.raises(ValueError, match='literal'):
        rt.load_view(b'{}')
    assert q.VIEW_SHA256 == rt.VIEW_SHA256
    assert q.RECEIPT_SHA256['qualification'] == (
        '6e60da7759f3e1ae53bc86b93303bd321fa17bab5daac9905cb440cef5bd3f48')
    with pytest.raises(ValueError, match='typed'):
        rt.validate_view([])


@pytest.mark.parametrize('name', ['../x', '/x', 'x/../y', 'x//y', 'x/./y', '', '\x00x'])
def test_unsafe_view_members(name):
    with pytest.raises(ValueError):
        rt.relative(name)


def test_finite_file_link_kind_raw_target_and_content(tmp_path):
    (tmp_path/'file').write_bytes(b'qualified')
    (tmp_path/'alias').symlink_to('file')
    row = dict(kind='file', sha256=sha(b'qualified'))
    link = dict(kind='link', target='file', resolved=str(tmp_path/'file'),
                target_kind='file', sha256=row['sha256'])
    rt.check_entry(tmp_path, 'file', row)
    rt.check_entry(tmp_path, 'alias', link)
    (tmp_path/'file').write_bytes(b'changed')
    for name, record in [('file', row), ('alias', link)]:
        with pytest.raises(ValueError, match='content'):
            rt.check_entry(tmp_path, name, record)
    (tmp_path/'alias').unlink()
    (tmp_path/'alias').symlink_to('./file')
    with pytest.raises(ValueError, match='link changed'):
        rt.check_entry(tmp_path, 'alias', link)
    with pytest.raises(ValueError, match='type'):
        rt.check_entry(tmp_path, '.', row)


def test_canonical_parent_and_directory_namespace_cannot_escape(tmp_path):
    (tmp_path/'actual').mkdir()
    (tmp_path/'actual'/'item').write_bytes(b'known')
    (tmp_path/'alias').symlink_to('actual',target_is_directory=True)
    with pytest.raises(ValueError, match='parent'):
        rt.check_entry(tmp_path,'alias/item',dict(kind='file',sha256=sha(b'known')))
    (tmp_path/'unknown').mkdir()
    (tmp_path/'unknown'/'do-not-traverse').symlink_to('/missing')
    with pytest.raises(ValueError, match='unselected'):
        rt.namespace(tmp_path, expected={'.','actual','actual/item','alias'})


@pytest.mark.parametrize('kind', ['file','directory','link'])
def test_zip_lstat_rejects_any_present_type(tmp_path, kind):
    p = tmp_path/'python313.zip'
    rt.absent_zip(p)
    if kind == 'file':
        p.write_bytes(b'')
    elif kind == 'directory':
        p.mkdir()
    else:
        p.symlink_to('missing')
    with pytest.raises(ValueError, match='shadow'):
        rt.absent_zip(p)


def test_complete_namespace_excludes_base_site_root_and_children(tmp_path):
    (tmp_path/'stdlib.py').write_bytes(b'')
    (tmp_path/'site-packages').mkdir()
    (tmp_path/'site-packages'/'NEVER_READ').symlink_to('/missing')
    assert rt.namespace(tmp_path, exclude_site=True) == {'.','stdlib.py'}
    assert 'site-packages/NEVER_READ' in rt.namespace(tmp_path)
    (tmp_path/'injected.pth').write_bytes(b'import hostile')
    assert 'injected.pth' in rt.namespace(tmp_path, exclude_site=True)


@pytest.mark.parametrize('path', ['', '.', '/tmp', rt.BASE+'/lib/python3.13/site-packages',
    rt.PREFIX+'/lib/python3.13/site-packages/psi4', rt.BUILD+'/source',
    rt.PREFIX+'/bin', rt.BASE+'/lib/python3.13/other', '/tmp/evil.zip'])
def test_search_path_rejects_source_site_arbitrary_descendants(path):
    with pytest.raises(ValueError, match='search path'):
        rt.validate_python_path([path])


def test_search_path_exact_sets_only():
    rt.validate_python_path([rt.BASE+'/lib/python3.13',
        rt.BASE+'/lib/python3.13/lib-dynload', rt.BASE+'/lib/python313.zip',
        rt.PREFIX+'/lib/python3.13/site-packages'])
    rt.validate_python_path([rt.PREFIX+'/bin'], cli=True)


def fake_runtime(monkeypatch, tmp_path):
    base, prefix = tmp_path/'base', tmp_path/'prefix'
    std = base/'lib/python3.13'
    std.mkdir(parents=True)
    prefix.mkdir()
    (prefix/'pyvenv.cfg').write_bytes(b'cfg')
    view = dict(scopes=dict(
        prefix=dict(root=str(prefix), entries={
            '.': dict(kind='directory'),
            'pyvenv.cfg': dict(kind='file', sha256=sha(b'cfg'))}),
        base=dict(root=str(base), entries={
            'lib/python3.13': dict(kind='directory')})))
    monkeypatch.setattr(rt, 'PREFIX', str(prefix))
    monkeypatch.setattr(rt, 'BASE', str(base))
    monkeypatch.setattr(rt, 'load_view', lambda _: view)
    monkeypatch.setattr(rt, 'validate_cfg', lambda raw: raw == b'cfg' or rt.need(False,'cfg'))
    return base, prefix, view


@pytest.mark.parametrize('change', ['private_pth', 'stdlib_extra', 'zip', 'cfg'])
def test_live_namespace_and_cfg_fail_closed(monkeypatch, tmp_path, change):
    base, prefix, _ = fake_runtime(monkeypatch, tmp_path)
    rt.verify_runtime_view(b'fake')
    if change == 'private_pth':
        (prefix/'inject.pth').write_bytes(b'import bad')
    elif change == 'stdlib_extra':
        (base/'lib/python3.13/evil.py').write_bytes(b'')
    elif change == 'zip':
        (base/'lib/python313.zip').symlink_to('missing')
    else:
        (prefix/'pyvenv.cfg').write_bytes(b'changed')
    with pytest.raises(ValueError):
        rt.verify_runtime_view(b'fake')


def test_runtime_environment_does_not_inherit_loader_or_python(monkeypatch, tmp_path):
    scratch = tmp_path/'ccasp-native-unit'
    scratch.mkdir(mode=0o700)
    for name in ('home','cache'):
        (scratch/name).mkdir(mode=0o700)
    monkeypatch.setattr(rt, 'SCRATCH_ROOT', tmp_path)
    monkeypatch.setenv('LD_PRELOAD','never.so')
    monkeypatch.setenv('PYTHONPATH','/hostile')
    env = rt.cli_environment(scratch/'home', scratch/'cache', scratch,
        inherited_threads=dict(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1'))
    assert 'LD_PRELOAD' not in env and 'PYTHONPATH' not in env
    assert env['LD_LIBRARY_PATH'] == rt.PREFIX+'/lib:'+rt.BASE+'/lib'
    assert env['OPENBLAS_NUM_THREADS'] == '1'
    (scratch/'cache'/'old.pyc').write_bytes(b'shadow')
    with pytest.raises(ValueError, match='preexisting'):
        rt.cli_environment(scratch/'home', scratch/'cache', scratch,
            inherited_threads=dict(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1'))


def test_admitted_launch_uses_exact_ordinary_cli_and_checks_before_start(monkeypatch, tmp_path):
    scratch = tmp_path/'ccasp-native-unit'
    scratch.mkdir(mode=0o700)
    for name in ('home','cache','cwd'):
        (scratch/name).mkdir(mode=0o700)
    monkeypatch.setattr(rt, 'SCRATCH_ROOT', tmp_path)
    seen = []
    monkeypatch.setattr(rt, 'verify_runtime_view', lambda raw: seen.append(raw))
    gate = rt.helper('bank_runtime_gate')
    monkeypatch.setattr(rt, 'helper', lambda _: gate)
    result = rt.ordinary_cli_launch(b'checked', ['--version'], scratch=scratch,
        execution_profile=PROFILE, inherited_threads=dict(
            OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1'))
    assert seen == [b'checked']
    assert result['command'][:4] == [rt.PREFIX+'/bin/python', '-I','-B','-X']
    assert result['command'][-2:] == [rt.PREFIX+'/bin/psi4','--version']
    assert result['cwd'] == str(scratch/'cwd')
    molecular = rt.ordinary_cli_launch(b'checked',
        ['--scratch',str(scratch/'psio'),'--nthread','8',
         '--memory','20.0GiB','--qcschema','data.msgpack'], scratch=scratch,
        execution_profile=PROFILE, inherited_threads=dict(
            OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1'))
    assert molecular['command'][-1] == 'data.msgpack'
    with pytest.raises(ValueError):
        rt.ordinary_cli_launch(b'checked', ['--evil'], scratch=scratch,
            execution_profile=PROFILE, inherited_threads=dict(
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1'))
    with pytest.raises(ValueError, match='thread contract'):
        rt.ordinary_cli_launch(b'checked', ['--version'], scratch=scratch,
            execution_profile=PROFILE, inherited_threads=dict(
                OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1'))


def test_descriptor_selection_new_enum_and_legacy_profiles_unchanged():
    profiles = load('descriptor_profiles')
    for producer in ('cb63305','ed561666','grid400','qomp'):
        result = profiles.profile(producer)
        assert result['integration_grid'] == ((400,2030) if producer in ('grid400','qomp') else (200,974))
    result = profiles.profile('qomp', scope='full-bank')
    assert result['selected_monomers'] is None and result['drho_refinement_iterations'] == 32
    assert 'runtime_qomp.py' in result['child_sources']
    for producer in ('ready','qomp-future',''):
        with pytest.raises(ValueError):
            profiles.profile(producer)


def test_accounting_adapter_preserves_exact_unqualified_math_and_source_guard():
    plan = load('resource_plan_qomp')
    assert plan.SCIENTIFIC_COMMIT == q.SCIENTIFIC_COMMIT
    assert plan.POLICY.endswith('-unqualified')
    assert plan.FILES == plan._inner.FILES
    assert plan.plan is plan._inner.plan
    assert plan.source_binding is plan._inner.source_binding
    with pytest.raises((ValueError, FileNotFoundError)):
        plan.source_binding(Path(__file__).resolve().parents[2]/'missing')


def test_cfg_false_system_site_and_no_forged_authority(monkeypatch):
    raw = ('home = '+rt.BASE+'/bin\ninclude-system-site-packages = false\n'
           'executable = '+rt.BASE+'/bin/python3.13\nversion = 3.13.11\n').encode()
    monkeypatch.setattr(rt, 'CFG_SHA256', sha(raw))
    rt.validate_cfg(raw)
    with pytest.raises(ValueError, match='changed'):
        rt.validate_cfg(raw.replace(b'false',b'true'))
    # Even granting a synthetic hash, semantic isolation remains required.
    bad = raw.replace(b'false',b'true')
    monkeypatch.setattr(rt, 'CFG_SHA256', sha(bad))
    with pytest.raises(ValueError, match='semantics'):
        rt.validate_cfg(bad)


def test_view_foreign_source_and_link_binding_refuse_before_filesystem():
    view = dict(schema='candidate-conservative-qomp-runtime-view/1',
        status='NOT_ACTIVE_PENDING_COVERAGE_REVIEW', closure_approved=False,
        production_adoption=False, scientific_model_acceptance=False,
        inventory_sha256=q.INVENTORY_SHA256,
        identity=dict(implementation=q.IMPLEMENTATION, version=q.VERSION),
        qualification_job='15326636', historical_build=dict(status='FAILED'),
        source=dict(compiled=dict(commit='foreign',tree=q.SCIENTIFIC_TREE)))
    with pytest.raises(ValueError, match='source'):
        rt.validate_view(view)
    view['source'] = {role:dict(commit=commit,tree=tree) for role,commit,tree in (
        ('compiled',q.SCIENTIFIC_COMMIT,q.SCIENTIFIC_TREE),
        ('science',q.SCIENTIFIC_COMMIT,q.SCIENTIFIC_TREE),
        ('tests',q.TEST_COMMIT,q.TEST_TREE),('ops',q.QUALIFYING_OPS,q.QUALIFYING_OPS_TREE))}
    view['scopes'] = dict(
        prefix=dict(root=rt.PREFIX,entries={
            'lib64':dict(kind='link',target='lib',resolved='/foreign',target_kind='directory',
                         target_namespace_binding=dict(scope='prefix',member='lib',
                                                       complete_target_subtree=True)),
            'lib':dict(kind='directory')}),
        base=dict(root=rt.BASE,entries={}))
    with pytest.raises(ValueError, match='binding'):
        rt.validate_view(view)


def trace_fixture(monkeypatch, tmp_path):
    base, prefix, view = fake_runtime(monkeypatch, tmp_path)
    run, scratch = tmp_path/'run', tmp_path/'scratch'
    run.mkdir()
    scratch.mkdir()
    command = [str(prefix/'bin/python'),'-I','-B','-X',
               'pycache_prefix='+str(scratch/'cache'),str(prefix/'bin/psi4'),'--version']
    lines = ['100 execve('+json.dumps(command[0])+', '+json.dumps(command)+', []) = 0']
    return run, scratch, base, prefix, lines


def test_decoded_fd_exact_role_selected_content_and_actual_input(monkeypatch, tmp_path):
    run, scratch, base, prefix, lines = trace_fixture(monkeypatch, tmp_path)
    lines.append('100 openat(AT_FDCWD, "alias", O_RDONLY) = 3<'+str(prefix/'pyvenv.cfg')+'>')
    def audit(trace, inputs=None):
        return rt.trace_projection(b'fake',trace,run=run,scratch=scratch,
            allowed_inputs=inputs or {},forbidden=(),execution_profile=PROFILE)
    assert audit(lines)['status'] == 'passed'
    data = scratch/'explicit-input.json'
    data.write_bytes(b'unchanged')
    opened = lines+['100 open("explicit-input.json", O_RDONLY) = 4<'+str(data)+'>']
    assert audit(opened, {str(data):sha(b'unchanged')})['status'] == 'passed'
    assert audit(opened, {str(data):sha(b'other')})['status'] == 'failed'
    (prefix/'pyvenv.cfg').write_bytes(b'tampered')
    assert audit(lines)['status'] == 'failed'
    with pytest.raises(ValueError):
        audit([lines[0].replace('--version','--evil')])


@pytest.mark.parametrize('relative_path', [
    'base/lib/python3.13/site-packages/foreign.so', 'base/unselected.so',
    'scratch/foreign.py', 'scratch/foreign.pyc', 'scratch/libshadow.so',
    'run/basis.gbs'])
def test_trace_foreign_shadow_and_excluded_site_refuse(monkeypatch, tmp_path, relative_path):
    run, scratch, _, _, lines = trace_fixture(monkeypatch, tmp_path)
    path = tmp_path/relative_path
    # Paths need not exist to exercise decoded-FD rejection (no host file hash).
    lines.append('100 openat(AT_FDCWD, "relative", O_RDONLY) = 3<'+str(path)+'>')
    out = rt.trace_projection(b'fake',lines,run=run,scratch=scratch,
        allowed_inputs={},forbidden=(),execution_profile=PROFILE)
    assert out['status'] == 'failed'
    assert out['unselected_or_shadow_opens'] == [str(path)]


def test_trace_missing_decoded_fd_or_incomplete_syscall_refuse(monkeypatch, tmp_path):
    run, scratch, _, _, lines = trace_fixture(monkeypatch, tmp_path)
    for tail in ('100 open("relative", O_RDONLY) = 3',
                 '100 open("relative", O_RDONLY <unfinished ...>'):
        with pytest.raises(ValueError):
            rt.trace_projection(b'fake',lines+[tail],run=run,scratch=scratch,
                allowed_inputs={},forbidden=(),execution_profile=PROFILE)


def test_new_descriptor_guard_runs_on_both_sides_of_mock_import(monkeypatch):
    f = load('test_descriptor_probe_cb63305')
    runtime, calls, _ = f.mock_boundaries(monkeypatch, producer='qomp')
    fake_psi = sys.modules['psi4']
    fake_psi.__file__ = rt.PREFIX+'/lib/python3.13/site-packages/psi4/__init__.py'
    fake_psi.core.__file__ = rt.PREFIX+'/lib/python3.13/site-packages/psi4/core.so'
    seen = []
    guard = SimpleNamespace(PREFIX=rt.PREFIX,
        load_view=lambda raw: seen.append(('view',raw)),
        validate_python_path=lambda paths: seen.append(('path',tuple(paths))))
    original = f.probe.helper
    monkeypatch.setattr(f.probe, 'helper', lambda name:
        SimpleNamespace(runtime=lambda: guard) if name == 'qualified_qomp' else original(name))
    result = f.probe.scientific_probe(runtime, memory_bytes=20*1024**3,
                                     producer='qomp', runtime_view=b'pinned')
    assert [kind for kind,_ in seen] == ['view','path','path']
    assert result['plan']['dimensions']['isa_grid_accounting_rows'] == 3*99*230
    assert result['plan']['dimensions']['integration_grid_rows'] == 3*399*2030
    assert result['plan']['dimensions']['drho_refinement_iterations'] == 32
    fake_psi.core.__file__ = '/foreign/core.so'
    with pytest.raises(ValueError, match='foreign'):
        f.probe.scientific_probe(runtime, memory_bytes=20*1024**3,
                                 producer='qomp', runtime_view=b'pinned')


def test_prepared_qomp_bank_is_draft_only_and_uses_distinct_pair(monkeypatch):
    f = load('test_prepare_descriptor_bank_cb63305')
    documents, contracts = f.fixture()
    raw = f.bind_inputs(monkeypatch, documents)
    original = f.prepare.helper
    def helper(name):
        module = original(name)
        if name == 'qualified_qomp':
            module.verify_receipts = lambda _: dict(implementation_sha256=q.IMPLEMENTATION,
                                                     production_adoption=False)
        return module
    monkeypatch.setattr(f.prepare,'helper',helper)
    resources = json.loads(f.resources(raw,contracts))
    resources['policy_id'] = load('resource_plan_qomp').POLICY
    out = f.prepare.prepare_qomp_full_bank(raw,{},f.encode(resources))
    assert out['status'] == 'PREPARED_FOR_DESCRIPTORS_ONLY'
    assert out['monomer_count'] == 131 and out['campaign_scope'] == 'full-bank'
    for row in out['monomers']:
        claim = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
        assert claim['implementation_sha256'] == q.IMPLEMENTATION
        assert claim['resources']['max_work'] == 6*10**12
    resources['policy_id'] = '1aaa9e0a30-reference-h2h1-grid400-ledger-v4'
    with pytest.raises(ValueError, match='declarations'):
        f.prepare.prepare_qomp_full_bank(raw,{},f.encode(resources))


def test_inherited_ops_exception_is_exact_970_not_generic_source_permission(tmp_path):
    root = Path(__file__).resolve().parents[2]
    names = list(q.INHERITED_OPS)
    q.verify_descriptor_operations(root,names+['devtools/isapol_s66_campaign/qualified_qomp.py'])
    for name in names:
        p = tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((root/name).read_bytes()+b'\n')
        with pytest.raises(ValueError, match='operations change'):
            q.verify_descriptor_operations(tmp_path,[name])
    for name in ['psi4/driver/procrouting/isapol_distribution.py',
                 'tests/pytests/test_isapol_isa_grid_accounting.py', '../outside']:
        with pytest.raises(ValueError, match='operations change'):
            q.verify_descriptor_operations(root,[name])


def test_qomp_descriptor_admission_direct_installed_pair_not_old_nested_pair():
    f = load('test_descriptor_admission_cb63305')
    report, row, identity, kwargs = f.fixture('qomp')
    identity['implementation'] = dict(q.IMPLEMENTATION)
    report['implementation'] = dict(sha256=dict(q.IMPLEMENTATION))
    result = f.guard.verify_descriptor(report,row,identity,producer='qomp',**kwargs)
    assert result['dimensions']['isa_grid_accounting_rows'] == 3*99*230
    identity['implementation'] = dict(sha256=dict(q.IMPLEMENTATION))
    with pytest.raises(ValueError, match='binding differs'):
        f.guard.verify_descriptor(report,row,identity,producer='qomp',**kwargs)


@pytest.mark.parametrize('error', [None, RuntimeError('child failed'), SystemExit(143), KeyboardInterrupt()])
def test_finite_postseal_on_success_failure_and_signal_unwind(monkeypatch, error):
    seen = []
    monkeypatch.setattr(rt,'verify_runtime_view',lambda raw: seen.append(raw))
    def use():
        with rt.runtime_admission(b'pinned'):
            seen.append('child')
            if error is not None:
                raise error
    if error is None:
        use()
    else:
        with pytest.raises(type(error)):
            use()
    assert seen == [b'pinned','child',b'pinned']


def test_postseal_failure_cannot_be_success(monkeypatch):
    seen = []
    def check(raw):
        if seen:
            raise ValueError('runtime changed')
        seen.append(raw)
    monkeypatch.setattr(rt,'verify_runtime_view',check)
    with pytest.raises(ValueError,match='runtime changed'):
        with rt.runtime_admission(b'pinned'):
            pass
