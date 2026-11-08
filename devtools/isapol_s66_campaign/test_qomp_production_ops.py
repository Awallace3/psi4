"""Fake/stdlib only. NEVER import Psi4 or execute a compiler/SLURM command."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


B = load(ROOT/'devtools/isapol_native_acceptance/phoenix_build.py', '_qomp_build_tests')
C = load(ROOT/'devtools/isapol_s66_campaign/phoenix_installed_checks.py', '_qomp_checks_tests')


def fake_nodes():
    legacy = C.TEST_FILES
    rows = ['tests/pytests/'+legacy[i % len(legacy)]+'::test_fake_'+str(i) for i in range(461)]
    for name, count in B.QOMP_FOCUSED.items():
        rows += ['tests/pytests/'+name+'::test_fake_'+str(i) for i in range(count)]
    rows += ['tests/pytests/'+name for name in B.QOMP_MOLECULAR]
    return rows


def collected():
    return dict(nodes=fake_nodes(), call_nodes=[], collection_errors=0,
                collection_skips=0, failed_reports=0, skipped_reports=0, returncode=0)


def config_fixture(root):
    """Owned Git metadata only; no scientific source/compile/server operation."""
    repo, selected = root/'repo', root/'selected'
    repo.mkdir()
    env = B.qomp_control_environment({})
    def git(path, *args):
        return subprocess.check_output([*B.QOMP_GIT, '-c', 'user.name=Pure test',
            '-c', 'user.email=test@example.invalid', '-C', str(path), *args],
            env=env, text=True).strip()
    git(repo, 'init', '-q')
    (repo/'fixture').write_text('inert Git fixture\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'fixture')
    git(repo, 'worktree', 'add', '--detach', str(selected), 'HEAD')
    git(repo, 'config', 'extensions.worktreeConfig', 'true')
    admin = Path(git(selected, 'rev-parse', '--absolute-git-dir'))
    return repo, selected, admin, env, git


class QompPureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = B.qomp_library()

    def test_library_exact_git_bytes_and_no_old_source_or_test_gate(self):
        self.assertEqual(self.lib.PATCH, '731fb3d52f3c301b28b22fae469e0deb3f7e805c')
        with patch.object(B, 'QOMP_LIBRARY_SHA', '0'*64):
            with self.assertRaisesRegex(ValueError, '8573 helper bytes'):
                B.qomp_library()
        source = (ROOT/'devtools/isapol_native_acceptance/phoenix_build.py').read_text().split(
            '# New closed mode.', 1)[1]
        self.assertNotIn('lib.source_gate(', source)
        self.assertNotIn('lib.junit_gate(', source)
        self.assertNotIn('lib.test_command(', source)
        self.assertNotIn('pip', source)

    def test_historical_functions_policies_and_wrappers_are_byte_exact(self):
        old = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            B.QOMP_SCI+':devtools/isapol_native_acceptance/phoenix_build.py'], text=True)
        current = (ROOT/'devtools/isapol_native_acceptance/phoenix_build.py').read_text()
        self.assertEqual(current.split('# New closed mode.', 1)[0],
                         old.split("if __name__ == '__main__':", 1)[0])
        old = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            B.QOMP_SCI+':devtools/isapol_s66_campaign/phoenix_installed_checks.py'], text=True)
        current = (ROOT/'devtools/isapol_s66_campaign/phoenix_installed_checks.py').read_text()
        self.assertEqual(current.split('def qomp_engine():', 1)[0],
                         old.split("if __name__ == '__main__':", 1)[0])
        for name in ('qualified_cb63305.py', 'qualified_grid400.py', 'descriptor_profiles.py',
                     'resource_plan_cb63305.py', 'resource_plan_ed561666.py',
                     'resource_plan_grid400.py', 'resource_plan_isa_grid_v1.py',
                     'phoenix_installed_checks.sbatch'):
            path = 'devtools/isapol_s66_campaign/'+name
            self.assertEqual((ROOT/path).read_bytes(), subprocess.check_output(
                ['git', '-C', str(ROOT), 'show', B.QOMP_SCI+':'+path]))

    def test_source_roles_wrong_tree_dirty_detachment_and_allowlist(self):
        def git(path, *args):
            sci = path == B.QOMP_SCI_SOURCE
            if args == ('rev-parse', 'HEAD'):
                return B.QOMP_SCI if sci else 'a'*40
            if args == ('rev-parse', 'HEAD^{tree}'):
                return B.QOMP_TREE if sci else 'b'*40
            if args == ('rev-parse', '--show-toplevel'):
                return str(path.resolve())
            if args[0] == 'merge-base':
                return B.QOMP_SCI
            if args[0] == 'diff':
                return 'devtools/isapol_native_acceptance/phoenix_build.py'
            return ''
        with patch.object(B, 'QOMP_OPS_SOURCE', B.SOURCE), patch.object(B, 'qomp_git', side_effect=git), \
             patch.object(B, 'qomp_config_gate'), \
             patch.object(B.subprocess, 'run', return_value=NS(returncode=1)):
            value = B.qomp_source_gate('a'*40, 'b'*40)
            self.assertEqual(value['scientific_commit'], B.QOMP_SCI)
            self.assertEqual(value['test_source_commit'], B.QOMP_SCI)
            self.assertNotEqual(value['ops_commit'], value['test_source_commit'])
            for request, response in ((('rev-parse', 'HEAD^{tree}'), '0'*40),
                                      (('status', '--porcelain', '--untracked-files=all'), '?? extra'),
                                      (('diff', '--name-only', B.QOMP_SCI, 'a'*40),
                                       'psi4/driver/procrouting/isapol_distribution.py')):
                with patch.object(B, 'qomp_git', side_effect=lambda p, *a:
                                  response if a == request else git(p, *a)):
                    with self.assertRaises(ValueError):
                        B.qomp_source_gate('a'*40, 'b'*40)
            with patch.object(B.subprocess, 'run', return_value=NS(returncode=0)):
                with self.assertRaisesRegex(ValueError, 'detached'):
                    B.qomp_source_gate('a'*40, 'b'*40)
        with self.assertRaisesRegex(ValueError, 'closed ops source path'):
            B.qomp_source_gate('a'*40, 'b'*40)

    def test_targets_closed_union_no_overlap_or_full_molecular_module(self):
        targets = B.qomp_targets()
        self.assertEqual(len(targets), 17)  #11+4 files +2 explicit nodes
        self.assertEqual(len(targets), len(set(targets)))
        self.assertTrue(all(p.startswith(str(B.QOMP_SCI_SOURCE/'tests/pytests')) for p in targets))
        self.assertNotIn(str(B.QOMP_SCI_SOURCE/'tests/pytests/test_isapol_distribution.py'), targets)
        required = B.qomp_installed_names({'driver/previous.py'})
        self.assertIn('driver/previous.py', required)
        for name in ('driver/procrouting/isapol_isa_grid_plan.py',
                     'tests/test_isapol_q_parallel.py', 'tests/test_isapol_isa_grid_accounting.py',
                     'tests/isapol_water_recipe.py'):
            self.assertIn(name, required)  # Even when absent from old/observed inventories.
        counts = B.qomp_nodes_gate(collected())
        self.assertEqual(counts['groups'], dict(legacy=461, focused=93, molecular=2))
        self.assertEqual(C.qualification_suite('grid400'), 461)
        self.assertEqual(C.qualification_suite('legacy'), 459)

    def test_nodes_refuse_old32_82_missing_duplicate_foreign_skip_and_failed(self):
        value = collected()
        mutations = [
            lambda v: v.update(nodes=v['nodes'][:32]),
            lambda v: v.update(nodes=v['nodes'][:82]),
            lambda v: v['nodes'].__setitem__(-1, v['nodes'][0]),
            lambda v: v['nodes'].__setitem__(0, 'other.py::test_injected'),
            lambda v: v.update(collection_errors=1),
            lambda v: v.update(collection_skips=1),
            lambda v: v.update(returncode=1),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.assertRaises(ValueError):
                B.qomp_nodes_gate(changed)
        value['call_nodes'] = list(value['nodes'])
        B.qomp_nodes_gate(value, executed=True)
        for key, replacement in (('call_nodes', value['nodes'][:-1]),
                                 ('failed_reports', 1), ('skipped_reports', 1)):
            with self.assertRaises(ValueError):
                B.qomp_nodes_gate(dict(value, **{key: replacement}), executed=True)

    def test_recorded_runner_actual_nonzero_and_signal_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = B.qomp_runner(self.lib, root, root, dict(os.environ), 20)
            runner.run('ok', [sys.executable, '-I', '-B', '-c', 'print("stdlib only")'])
            self.assertEqual(self.lib.read(root/'ok-result.json')['returncode'], 0)
            with self.assertRaisesRegex(ValueError, 'exited 7'):
                runner.run('nonzero', [sys.executable, '-I', '-B', '-c', 'raise SystemExit(7)'])
            self.assertEqual(self.lib.read(root/'nonzero-result.json')['returncode'], 7)
            # Runner deliberately latches stop on ANY failed command; a fresh
            # operation, not an automatic retry/resume of that failed runner.
            runner = B.qomp_runner(self.lib, root, root, dict(os.environ), 20)
            with self.assertRaises(ValueError):
                runner.run('signal', [sys.executable, '-I', '-B', '-c',
                                     'import os,signal;os.kill(os.getpid(),signal.SIGTERM)'])
            self.assertEqual(self.lib.read(root/'signal-result.json')['returncode'], -signal.SIGTERM)

    def test_collect_fake_full_seam_installed_import_first_no_science_source_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(B, 'QOMP_SCI_SOURCE', ROOT):
            root = Path(tmp)
            lib = self.lib
            lib.write(root/'installed-python-checks.json', {
                'tests/isapol_water_recipe.py': dict(sha256='0'*64)})
            class FakeRunner:
                def run(self, label, command):
                    self.command = command
                    lib.write(root/(label+'-command.json'), list(map(str, command)))
                    lib.write(root/(label+'-result.json'), dict(returncode=0))
                    lib.write(root/'collection-nodes.json', collected())
            runner = FakeRunner()
            runner.root = root
            B.qomp_pytest(runner, root/'env', root/'scratch', lib, collect=True)
            script = runner.command[-1]
            self.assertLess(script.index('assert package =='), script.index('import pytest'))
            self.assertLess(script.index('qomp_search_path(sys.path'), script.index('import psi4'))
            self.assertIn("'-S'", repr(runner.command))
            self.assertIn("sys.path.insert(0,str(package/'tests'))", script)
            self.assertNotIn('sys.path.insert(0,str(SOURCE', script)
            value = lib.read(root/'collection.json')
            self.assertEqual(value['source']['commit'], B.QOMP_SCI)
            self.assertEqual(value['group_counts'], dict(legacy=461, focused=93, molecular=2))
            self.assertEqual(value['test_source_sha256']['test_isapol_isa_grid_accounting.py'],
                             lib.sha(ROOT/'tests/pytests/test_isapol_isa_grid_accounting.py'))
            self.assertFalse('psi4' in sys.modules)

    def test_complete_mapping_generated_three_flat_sample_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, build, package = [root/name for name in ('source', 'build', 'package')]
            names = ['__init__.py', 'extras.py', 'metadata.py', 'driver/procrouting/isapol_isa_grid_plan.py',
                     'tests/isapol_water_recipe.py', 'tests/test_isapol_isa_grid_accounting.py',
                     'tests/aediis-1/test_input.py']
            for name in names:
                if name in ('__init__.py', 'extras.py', 'metadata.py'):
                    expected = build/'psi4-core-prefix/src/psi4-core-build'/name
                elif name.startswith('driver/'):
                    expected = source/'psi4'/name
                elif name.startswith('tests/aediis'):
                    expected = source/name
                else:
                    expected = source/'tests/pytests'/name.removeprefix('tests/')
                expected.parent.mkdir(parents=True, exist_ok=True)
                expected.write_text('# actual configured/source fixture\n')
                target = package/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(expected.read_bytes())
            checked = self.lib.installed_python_gate(source, build, package, names)
            self.assertEqual(len(checked), len(names))
            for name in names:
                path = package/name
                saved = path.read_bytes()
                path.write_bytes(saved+b'# changed\n')
                with self.assertRaises(ValueError):
                    self.lib.installed_python_gate(source, build, package, names)
                path.write_bytes(saved)
            (package/'unknown.py').write_text('# no known mapping')
            with self.assertRaisesRegex(ValueError, 'unknown'):
                self.lib.installed_python_gate(source, build, package, names)
            (package/'unknown.py').unlink()
            expected = source/'tests/pytests/isapol_water_recipe.py'
            raw = expected.read_bytes()
            expected.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                self.lib.installed_python_gate(source, build, package, names)
            expected.write_bytes(raw)
            ambiguous = source/'tests/pytests/aediis-1/test_input.py'
            ambiguous.parent.mkdir(parents=True)
            ambiguous.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                self.lib.installed_python_gate(source, build, package, names)

    def test_approved_replay_and_sanitized_offline_environment(self):
        record = dict(configure=list(self.lib.APPROVED_CONFIGURE))
        argv = self.lib.configure(record, Path('/new/source'), Path('/new/run'))
        self.assertEqual(argv[argv.index('-S')+1], '/new/source')
        self.assertIn('-DCMAKE_INSTALL_PREFIX=/new/run/env', argv)
        self.assertIn('-DPYMOD_INSTALL_LIBDIR=/python3.13/site-packages', argv)
        for appended in ('-DCMAKE_TOOLCHAIN_FILE=/tmp/bad', '-C/tmp/preload',
                         '-DCMAKE_INSTALL_PREFIX=/old', '-DENABLE_OPENMP=OFF'):
            with self.assertRaises(ValueError):
                self.lib.configure(dict(configure=record['configure']+[appended]),
                                   Path('/new/source'), Path('/new/run'))
        with patch.dict(os.environ, dict(DESTDIR='/bad', CMAKE_INSTALL_MODE='SYMLINK',
                                        GCC_ROOT='/bad', CPATH='/bad', PYTEST_ADDOPTS='-k skip',
                                        CMAKE_TOOLCHAIN_FILE='/bad')):
            env = self.lib.environment(Path('/new/env'), Path('/new/scratch'))
            for name in ('DESTDIR', 'CMAKE_INSTALL_MODE', 'GCC_ROOT', 'CPATH',
                         'PYTEST_ADDOPTS', 'CMAKE_TOOLCHAIN_FILE'):
                self.assertNotIn(name, env)
            self.assertEqual(env['PIP_NO_INDEX'], '1')
            self.assertEqual(env['OPENBLAS_NUM_THREADS'], '1')

    def test_full_fake_build_gate_then_stale_failure_source_and_identity_refusals(self):
        lib = self.lib
        with tempfile.TemporaryDirectory() as tmp, patch.object(B, 'QOMP_BUILD', Path(tmp)), \
             patch.object(B, 'QOMP_SCI_SOURCE', ROOT), patch.object(C, 'qomp_engine', return_value=B):
            root = Path(tmp)
            package = root/'env/lib/python3.13/site-packages/psi4'
            package.mkdir(parents=True)
            core = package/'core.fake.so'
            core.write_bytes(b'fake bytes, never imported')
            source = dict(scientific_commit=B.QOMP_SCI, scientific_tree=B.QOMP_TREE,
                          test_source_commit=B.QOMP_SCI, ops_commit='a'*40, ops_tree='b'*40)
            args = NS(ops_commit='a'*40, ops_tree='b'*40)
            identity = dict(package=str(package/'__init__.py'), core=str(core), version='fake-only',
                            implementation=dict(core=lib.sha(core), isapol_python='f'*64))
            original = dict(configure=list(lib.APPROVED_CONFIGURE))
            configure = lib.configure(original, root/'source', root)
            build = list(map(str, [B.QOMP_BASE/'bin/cmake', '--build', root/'build',
                                   '--target', 'install', '-j', '2']))
            for label, argv in (('configure', configure), ('build-install', build),
                                ('collect-tests', ['fake collection command'])):
                lib.write(root/(label+'-command.json'), argv)
                lib.write(root/(label+'-result.json'), dict(
                    argv=argv, returncode=0, interrupted=False))
            lib.write(root/'installed-python-checks.json', {'checked': 'fake'})
            lib.write(root/'COMPLETED', dict(exit_code=0, mode='build', acceptance=False))
            manifest = dict(
                schema='fresh-native-build-identity/1', fresh_build=True, clean_source=True,
                commit=B.QOMP_SCI, tree=B.QOMP_TREE, operations=source,
                source_snapshot=str(root/'source'), installed_package=str(package),
                installed_core=str(core), installed_files={},
                old_manifest_sha256=lib.OLD_MANIFEST,
                provisioning='offline qualified1aaa dependency clone', allocation=dict(JobId='123'),
                original_configure_sha256=lib.APPROVED_RECORD_SHA)
            record = dict(
                commit=B.QOMP_SCI, tree=B.QOMP_TREE, operations=source, source=str(root/'source'),
                root=str(root), dependencies=str(B.QOMP_BASE), status='complete',
                provisioning=manifest['provisioning'], job='123', configure=configure, build=build,
                original_configure_sha256=lib.APPROVED_RECORD_SHA,
                command_results={name: lib.read(root/(name+'-result.json'))
                                 for name in ('configure', 'build-install')})
            collection = dict(collected(), source=dict(
                commit=B.QOMP_SCI, tree=B.QOMP_TREE, path=str(ROOT/'tests/pytests')),
                group_counts=dict(legacy=461, focused=93, molecular=2),
                focused_modules=dict(B.QOMP_FOCUSED), targets=B.qomp_targets(),
                test_source_sha256={Path(p.split('::')[0]).name: lib.sha(p.split('::')[0])
                                    for p in B.qomp_targets()},
                command_sha256=lib.sha(root/'collect-tests-command.json'),
                command_result_sha256=lib.sha(root/'collect-tests-result.json'))
            value = dict(schema='qomp-production-operation/1',
                         status='FRESH_BUILD_COMPLETE_NOT_QUALIFIED', mode='build',
                         acceptance=False, production_adoption=False, source=source,
                         old_and_base_unchanged=True, identity=identity, allocation=dict(JobId='123'))

            def publish(v, m, r, c):
                # Only synthetic owned fixture bytes. Rebind literals to reach
                # semantic gates rather than stopping at the first hash check.
                for name, data in (('build-identity.json', m), ('collection.json', c)):
                    (root/name).write_text(json.dumps(data))
                args.build_identity_sha256 = lib.sha(root/'build-identity.json')
                args.collection_sha256 = lib.sha(root/'collection.json')
                r = dict(r, build_identity_sha256=args.build_identity_sha256)
                (root/'build-record.json').write_text(json.dumps(r))
                args.build_record_sha256 = lib.sha(root/'build-record.json')
                v = dict(v, build_identity_sha256=args.build_identity_sha256,
                         collection_sha256=args.collection_sha256,
                         build_record_sha256=args.build_record_sha256)
                (root/'result.json').write_text(json.dumps(v))

            def git(path, *argv):
                return B.QOMP_SCI if argv[-1] == 'HEAD' else (
                    B.QOMP_TREE if argv[-1] == 'HEAD^{tree}' else '')
            accounting = '\n'.join('123'+suffix+'|COMPLETED|0:0|' for suffix in ('', '.batch', '.0'))
            with patch.object(B, 'qomp_source_gate', return_value=source), \
                 patch.object(B, 'qomp_git', side_effect=git), \
                 patch.object(C.subprocess, 'check_output', return_value=accounting), \
                 patch.object(C, 'verify_runtime'), patch.object(lib, 'old_seal', return_value={}), \
                 patch.object(lib, 'approved_record', return_value=original), \
                 patch.object(lib, 'installed_python_gate', return_value={'checked': 'fake'}), \
                 patch.object(lib, 'dependency_gate'):
                publish(value, manifest, record, collection)
                got = C.qomp_build_gate(B, lib, args)
                self.assertEqual(got[2], identity)
                args.collection_sha256 = '0'*64
                with self.assertRaisesRegex(ValueError, 'digest'):
                    C.qomp_build_gate(B, lib, args)
                for target, key, replacement in (
                        ('value', 'status', 'READY_BUILD_NOT_QUALIFIED'),
                        ('value', 'old_and_base_unchanged', False),
                        ('manifest', 'commit', '2fe7a711dd915e102fe6976f44d22571064aa31e'),
                        ('manifest', 'installed_package', '/old/env/psi4'),
                        ('record', 'source', '/wrong/source'),
                        ('record', 'status', 'FAILED'),
                        ('collection', 'nodes', collection['nodes'][:82])):
                    data = dict(value=copy.deepcopy(value), manifest=copy.deepcopy(manifest),
                                record=copy.deepcopy(record), collection=copy.deepcopy(collection))
                    data[target][key] = replacement
                    publish(data['value'], data['manifest'], data['record'], data['collection'])
                    with self.assertRaises(ValueError):
                        C.qomp_build_gate(B, lib, args)
                publish(value, manifest, record, collection)
                (root/'FAILED').write_text('retained synthetic failure')
                with self.assertRaisesRegex(ValueError, 'terminal'):
                    C.qomp_build_gate(B, lib, args)
                (root/'FAILED').unlink()
                with patch.object(C.subprocess, 'check_output',
                                  return_value=accounting.replace('123.0|COMPLETED|0:0', '123.0|FAILED|1:0')):
                    with self.assertRaisesRegex(ValueError, 'job/batch/step'):
                        C.qomp_build_gate(B, lib, args)

    def test_offline_clone_retargets_links_rejects_external_and_never_writes_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new, base = root/'old', root/'new', root/'base'
            old.mkdir()
            base.mkdir()
            (old/'data').write_text('unchanged dependency')
            (old/'link').symlink_to(old/'data')
            before = B.qomp_inventory(old, self.lib)
            changes = self.lib.clone_runtime(old, new, base)
            self.assertTrue(changes)
            self.assertEqual((new/'link').resolve(), new/'data')
            self.assertEqual(B.qomp_inventory(old, self.lib), before)
            foreign = root/'foreign'
            foreign.write_text('not an approved dependency')
            (old/'outside').symlink_to(foreign)
            with self.assertRaisesRegex(ValueError, 'BASE dependencies'):
                self.lib.clone_runtime(old, root/'rejected', base)
            self.assertEqual((old/'outside').resolve(), foreign)

    def test_failure_and_integrity_finally_record_old_base_and_never_success(self):
        lib = self.lib
        saved_cwd = Path.cwd()
        for scenario in ('action', 'changed', 'interrupted', 'late'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = NS(root=root/'output', scratch=root/'scratch', mode='build',
                          ops_commit='a'*40, ops_tree='b'*40,
                          launch_token='00000000-0000-0000-0000-000000000000')
                class FakeRunner:
                    interrupted = False
                    def stop(self, *unused):
                        self.interrupted = True
                    def check(self):
                        if self.interrupted:
                            raise InterruptedError('fake TERM')
                runner = FakeRunner()
                def action(active, helpers, result, arguments):
                    self.assertEqual(Path.cwd(), args.root/'cwd')
                    if scenario == 'action':
                        raise RuntimeError('fake stage failure')
                    if scenario == 'interrupted':
                        runner.stop()
                    if scenario == 'late':
                        result['status'] = 'FRESH_BUILD_COMPLETE_NOT_QUALIFIED'
                original_write = lib.write
                def write(path, value):
                    original_write(path, value)
                    if scenario == 'late' and path.name == 'result.json':
                        runner.stop()
                after = dict(snapshot='changed' if scenario == 'changed' else 'unchanged')
                try:
                    with patch.object(B, 'qomp_source_gate', return_value={}), \
                         patch.dict(os.environ, {}, clear=True), \
                         patch.object(B, 'qomp_library', return_value=lib), \
                         patch.object(B, 'qomp_checks_module', return_value=NS(allocation=lambda: {})), \
                         patch.object(lib, 'allocation', return_value={}), \
                         patch.object(B, 'qomp_runner', return_value=runner), \
                         patch.object(lib, 'write', side_effect=write), \
                         patch.object(B, 'qomp_protected', side_effect=[dict(snapshot='unchanged'), after]):
                        with self.assertRaises((RuntimeError, ValueError, InterruptedError)):
                            B.qomp_perform(args, action)
                    value = lib.read(args.root/'result.json')
                    if scenario != 'late':
                        self.assertEqual(value['status'], 'FAILED')
                    self.assertFalse(value['acceptance'])
                    self.assertFalse((args.root/'COMPLETED').exists())
                    self.assertEqual(lib.read(args.root/'protected-after.json'), after)
                    if scenario == 'action':
                        self.assertEqual(value['error']['type'], 'RuntimeError')
                        self.assertTrue(value['old_and_base_unchanged'])
                finally:
                    os.chdir(saved_cwd)

    def test_finalizer_no_foreign_writes_failure_or_interruption_adoption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, dict(os.environ)):
            root = Path(tmp)
            token = '00000000-0000-0000-0000-000000000000'
            args = NS(root=root, launch_token=token, mode='build',
                      ops_commit='a'*40, ops_tree='b'*40, exit_code=0)
            def clean_library():
                self.assertNotIn('LD_PRELOAD', os.environ)
                self.assertNotIn('GIT_CONFIG_COUNT', os.environ)
                self.assertEqual(os.environ['PATH'], '/usr/bin:/bin')
                return self.lib
            os.environ.update(LD_PRELOAD='/never-loaded.so', LD_LIBRARY_PATH='/not-used',
                              GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='core.hooksPath',
                              GIT_CONFIG_VALUE_0='/not-used', PATH='/fake/bin')
            with patch.object(B, 'qomp_library', side_effect=clean_library), \
                 patch.object(B, 'qomp_source_gate', return_value={}):
                self.assertEqual(B.qomp_finalize(args), 1)
                self.assertEqual(list(root.iterdir()), [])
                self.lib.write(root/'owner.json', dict(token=token, mode='build'))
                self.lib.write(root/'result.json', dict(
                    status='FRESH_BUILD_COMPLETE_NOT_QUALIFIED', old_and_base_unchanged=True,
                    launch_token=token, acceptance=False))
                self.assertEqual(B.qomp_finalize(args), 0)
                args.exit_code = 143
                self.assertEqual(B.qomp_finalize(args), 143)
                self.assertFalse((root/'COMPLETED').exists())
                self.assertTrue((root/'COMPLETED_INVALIDATED').exists())
                args.exit_code = 0
                self.assertEqual(B.qomp_finalize(args), 143)

    def test_control_environment_removes_path_git_and_loader_injection(self):
        dirty = dict(os.environ, PATH='/fake/bin', GIT_DIR='/fake/repo',
                     GIT_COMMON_DIR='/fake/common', GIT_WORK_TREE='/fake/tree',
                     GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='core.hooksPath',
                     GIT_CONFIG_VALUE_0='/fake/hooks', GIT_CONFIG='/fake/config',
                     GIT_EXEC_PATH='/fake/exec', LD_PRELOAD='/never-loaded.so',
                     LD_LIBRARY_PATH='/fake/lib', BASH_ENV='/fake/startup',
                     PYTHONPATH='/fake/site', SLURM_JOB_ID='123')
        clean = B.qomp_control_environment(dirty)
        self.assertEqual(clean['PATH'], '/usr/bin:/bin')
        self.assertEqual(clean['SLURM_JOB_ID'], '123')
        for key in dirty:
            if key.startswith(('LD_', 'PYTHON')) or key.startswith('GIT_') and key not in clean:
                self.assertNotIn(key, clean)
        self.assertEqual(clean['GIT_CONFIG_GLOBAL'], '/dev/null')
        self.assertEqual(clean['GIT_CONFIG_SYSTEM'], '/dev/null')
        self.assertNotIn('BASH_ENV', clean)
        self.assertEqual(B.QOMP_GIT[0], '/usr/bin/git')
        # Pure mock inspects exec environment; never launch a loader with LD_*.
        with patch.dict(os.environ, dirty), patch.object(B, 'qomp_config_gate'), \
             patch.object(B.subprocess, 'check_output', return_value='ok') as process:
            self.assertEqual(B.qomp_git(Path('/fake/repo'), 'rev-parse', 'HEAD'), 'ok')
            self.assertEqual(process.call_args.args[0][0], '/usr/bin/git')
            self.assertNotIn('LD_PRELOAD', process.call_args.kwargs['env'])
            self.assertNotIn('GIT_CONFIG_COUNT', process.call_args.kwargs['env'])
        for key in ('include.path', 'core.fsmonitor', 'core.alternateRefsCommand',
                    'protocol.ssh.allow', 'filter.fake.process', 'credential.helper'):
            with patch.object(B.subprocess, 'check_output', return_value=key):
                with self.assertRaisesRegex(ValueError, 'Git configuration'):
                    B.qomp_config_gate(Path('/fake/repo'), clean)

    def test_enabled_optional_absence_reproduces_actual_git128_and_finalizes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, dict(os.environ)):
            root = Path(tmp)
            repo, selected, admin, env, git = config_fixture(root)
            self.assertFalse((admin/'config.worktree').exists())
            # Same semantics as actual Phoenix Git2.52.0, not an old-Git claim.
            raw = subprocess.run([*B.QOMP_GIT, '-C', str(selected), 'config', '--worktree',
                                  '--no-includes', '--name-only', '--list'],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(raw.returncode, 128)
            self.assertIn(str(admin/'config.worktree').encode(), raw.stderr)
            B.qomp_config_gate(selected, env)  # RED on a6e: optional file was unconditionally read.
            out = root/'out'
            out.mkdir()
            token = '00000000-0000-0000-0000-000000000000'
            args = NS(root=out, launch_token=token, mode='build',
                      ops_commit='a'*40, ops_tree='b'*40, exit_code=0)
            self.lib.write(out/'owner.json', dict(token=token, mode='build'))
            self.lib.write(out/'result.json', dict(
                status='FRESH_BUILD_COMPLETE_NOT_QUALIFIED', old_and_base_unchanged=True,
                launch_token=token, acceptance=False))
            def library():
                B.qomp_config_gate(selected, B.qomp_control_environment(os.environ))
                return self.lib
            with patch.object(B, 'qomp_library', side_effect=library), \
                 patch.object(B, 'qomp_source_gate', return_value={}):
                self.assertEqual(B.qomp_finalize(args), 0)
                self.assertTrue((out/'COMPLETED').exists())
                args.exit_code = 143
                self.assertEqual(B.qomp_finalize(args), 143)
                self.assertFalse((out/'COMPLETED').exists())
                (admin/'config.worktree').write_text('[core]\n hooksPath = /not-used\n')
                with self.assertRaisesRegex(ValueError, 'Git configuration'):
                    B.qomp_finalize(args)
                self.assertFalse((out/'COMPLETED').exists())

    def test_present_worktree_config_safe_unsafe_malformed_and_read_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, selected, admin, env, git = config_fixture(root)
            config = admin/'config.worktree'
            config.write_text('[core]\n sparseCheckout = false\n')
            B.qomp_config_gate(selected, env)
            real = subprocess.check_output
            def read_error(command, **kwargs):
                if '--worktree' in command:
                    raise subprocess.CalledProcessError(128, command, stderr=b'synthetic read failure')
                return real(command, **kwargs)
            with patch.object(B.subprocess, 'check_output', side_effect=read_error):
                with self.assertRaises(subprocess.CalledProcessError):
                    B.qomp_config_gate(selected, env)  # Never swallow a generic128.
            config.write_text('[core]\n hooksPath = /not-used\n')
            with self.assertRaisesRegex(ValueError, 'Git configuration'):
                B.qomp_config_gate(selected, env)
            config.write_text('[malformed\n')
            with self.assertRaises(subprocess.CalledProcessError):
                B.qomp_config_gate(selected, env)

    def test_optional_config_admin_selection_links_and_permission_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, selected, admin, env, git = config_fixture(root)
            config = admin/'config.worktree'
            foreign = root/'foreign'
            foreign.write_text('[core]\n sparseCheckout = false\n')
            config.symlink_to(foreign)
            with self.assertRaises((ValueError, OSError)):
                B.qomp_config_gate(selected, env)
            config.unlink()
            config.write_text('[core]\n sparseCheckout = false\n')
            real_open = os.open
            def unreadable(path, *args, **kwargs):
                if Path(path) == config:
                    raise PermissionError('synthetic unreadable optional file')
                return real_open(path, *args, **kwargs)
            with patch.object(B.os, 'open', side_effect=unreadable):
                with self.assertRaises(PermissionError):
                    B.qomp_config_gate(selected, env)
            real = subprocess.check_output
            def wrong_admin_path(command, **kwargs):
                if '--git-path' in command:
                    return str(root/'config.worktree')+'\n'
                return real(command, **kwargs)
            with patch.object(B.subprocess, 'check_output', side_effect=wrong_admin_path):
                with self.assertRaisesRegex(ValueError, 'administrative'):
                    B.qomp_config_gate(selected, env)

    def test_disabled_worktree_extension_does_not_read_optional_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, selected, admin, env, git = config_fixture(root)
            git(repo, 'config', 'extensions.worktreeConfig', 'false')
            # This file is not selected by Git when the extension is disabled.
            (admin/'config.worktree').write_text('[core]\n hooksPath = /not-used\n')
            real = subprocess.check_output
            def no_optional(command, **kwargs):
                self.assertNotIn('--worktree', command)
                return real(command, **kwargs)
            with patch.object(B.subprocess, 'check_output', side_effect=no_optional):
                B.qomp_config_gate(selected, env)

    def test_pinned_git_materialization_ignores_inherited_hook_and_fake_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, snapshot, hooks, fakebin = [root/name for name in ('repo', 'snapshot', 'hooks', 'bin')]
            for path in (repo, hooks, fakebin):
                path.mkdir()
            marker = root/'hook-or-fake-git-ran'
            hook = hooks/'post-checkout'
            hook.write_text('#!/bin/sh\n: > "'+str(marker)+'"\n')
            hook.chmod(0o700)
            fake = fakebin/'git'
            fake.write_text('#!/bin/sh\n: > "'+str(marker)+'"\nexit 99\n')
            fake.chmod(0o700)
            env = B.qomp_control_environment({})
            def git(*args):
                return subprocess.check_output([*B.QOMP_GIT, '-c', 'user.name=Pure test',
                    '-c', 'user.email=test@example.invalid', '-C', str(repo), *args],
                    env=env, text=True).strip()
            git('init', '-q')
            for name in ('CMakeLists.txt', 'cmake/psi4OptionsTools.cmake', 'psi4/src/export_isapol.cc',
                         'psi4/driver/procrouting/isapol_benzene_acceptance.py'):
                path = repo/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fake source, never compiled or imported\n')
            git('add', '.')
            git('commit', '-qm', 'pure fixture')
            commit, tree = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD^{tree}')
            with patch.dict(os.environ, dict(PATH=str(fakebin), GIT_CONFIG_COUNT='1',
                            GIT_CONFIG_KEY_0='core.hooksPath', GIT_CONFIG_VALUE_0=str(hooks),
                            GIT_DIR=str(root/'not-a-repo'))), \
                 patch.object(B, 'QOMP_SCI_SOURCE', repo), patch.object(B, 'QOMP_SCI', commit), \
                 patch.object(B, 'QOMP_TREE', tree):
                B.qomp_materialize(snapshot)
                self.assertFalse(marker.exists())
                self.assertEqual(B.qomp_git(snapshot, 'rev-parse', 'HEAD'), commit)
            git('config', 'core.hooksPath', str(hooks))
            with self.assertRaisesRegex(ValueError, 'Git configuration'):
                B.qomp_git(repo, 'status', '--porcelain')
            self.assertFalse(marker.exists())

    def test_resolved_search_paths_exact_allowlist_and_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix, base = root/'env', root/'base'
            forbidden = [root/'science', root/'ops', root/'build/source']
            valid = [str(base/'lib/python313.zip'), str(base/'lib/python3.13'),
                     str(base/'lib/python3.13/lib-dynload'), str(prefix/'lib/python3.13/site-packages')]
            B.qomp_search_path(valid, prefix, base, forbidden)
            for path in ('', '.', str(root/'foreign/site-packages'), str(forbidden[0]),
                         str(forbidden[1]/'psi4'), str(forbidden[2]/'driver'),
                         str(base/'lib/python3.13/site-packages'), str(root/'unknown.zip')):
                with self.assertRaises(ValueError):
                    B.qomp_search_path(valid+[path], prefix, base, forbidden)
            helpers = str(prefix/'lib/python3.13/site-packages/psi4/tests')
            with self.assertRaises(ValueError):
                B.qomp_search_path(valid+[helpers], prefix, base, forbidden)
            B.qomp_search_path(valid+[helpers], prefix, base, forbidden, helpers=True)
            link = root/'looks-safe'
            link.symlink_to(forbidden[0], target_is_directory=True)
            with self.assertRaises(ValueError):
                B.qomp_search_path(valid+[str(link)], prefix, base, forbidden)

    def test_bootstrap_rejects_before_psi4_and_after_its_path_addition(self):
        import builtins
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix, base = root/'env', root/'base'
            initial = [str(base/'lib/python3.13'), str(base/'lib/python3.13/lib-dynload')]
            imported = []
            original_import = builtins.__import__
            def fake_import(name, *args, **kwargs):
                if name == 'psi4':
                    imported.append(name)
                    sys.path.append(str(root/'foreign/site-packages'))
                    return NS()
                return original_import(name, *args, **kwargs)
            with patch.object(B, 'QOMP_BASE', base):
                script = B.qomp_bootstrap(prefix)
            with patch.object(sys, 'path', initial+[str(root/'bad')]), \
                 patch.object(builtins, '__import__', side_effect=fake_import):
                with self.assertRaisesRegex(ValueError, 'search path'):
                    exec(script, {})
            self.assertEqual(imported, [])
            with patch.object(sys, 'path', list(initial)), \
                 patch.object(builtins, '__import__', side_effect=fake_import):
                with self.assertRaisesRegex(ValueError, 'search path'):
                    exec(script, {})
            self.assertEqual(imported, ['psi4'])  # Fake only, no native import.

    def test_junit_checks_all_names_not_only_counts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(C, 'qomp_engine', return_value=B):
            path = Path(tmp)/'junit.xml'
            nodes = fake_nodes()
            root = ET.Element('testsuites')
            suite = ET.SubElement(root, 'testsuite', tests='556', failures='0', errors='0', skipped='0')
            for node in nodes:
                file, name = node.split('::')
                ET.SubElement(suite, 'testcase', classname='tests.pytests.'+Path(file).stem, name=name)
            ET.ElementTree(root).write(path)
            self.assertEqual(C.qomp_junit_gate(path, nodes)['tests'], 556)
            suite[-1].set('name', 'test_substituted')
            ET.ElementTree(root).write(path)
            with self.assertRaisesRegex(ValueError, 'omitted/replaced'):
                C.qomp_junit_gate(path, nodes)

    def test_wrapper_term_during_active_child_and_after_completion_publication(self):
        wrapper = (ROOT/'devtools/isapol_native_acceptance/qomp.sbatch').read_text()
        for mode in ('build', 'checks', 'collect-existing', 'checks-existing'):
            for window in ('payload', 'finalize'):
                with self.subTest(mode=mode, window=window), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    helper = root/'source/devtools/isapol_native_acceptance/phoenix_build.py'
                    helper.parent.mkdir(parents=True)
                    helper.write_text("""
import os,pathlib,signal,sys
r=pathlib.Path(__file__).resolve().parents[3]
assert os.environ['PATH']=='/usr/bin:/bin'
assert not any(k in os.environ for k in ('LD_PRELOAD','LD_LIBRARY_PATH','GIT_CONFIG_COUNT','GIT_DIR'))
if sys.argv[2]=='finalize':
 code=int(sys.argv[sys.argv.index('--exit-code')+1])
 if code:
  if (r/'COMPLETED').exists(): (r/'COMPLETED').rename(r/'INVALIDATED')
  (r/'FAILED').write_text(str(code));sys.exit(code)
 (r/'COMPLETED').write_text('0')
 print('FINAL_READY',flush=True);signal.pause()
else:
 if (r/'WINDOW').read_text()=='payload':
  print('PAYLOAD_READY',flush=True);signal.pause()
""")
                    target = root/'source/devtools/isapol_s66_campaign/phoenix_installed_checks.py'
                    target.parent.mkdir(parents=True)
                    target.write_bytes(helper.read_bytes())
                    srun = root/'srun'
                    srun.write_text("#!/usr/bin/python3\nimport os,sys\n"
                                    "i=sys.argv.index('/usr/bin/python3');os.execv(sys.argv[i],sys.argv[i:])\n")
                    srun.chmod(0o700)
                    script = root/'wrapper'
                    self.assertEqual(wrapper.count('/opt/slurm/current/bin/srun'), 1)
                    poison = ('export LD_PRELOAD=/not-loaded-test.so LD_LIBRARY_PATH=/not-used '
                              'GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath '
                              'GIT_CONFIG_VALUE_0=/not-used GIT_DIR=/not-used\n')
                    clean_assert = ('[[ ! -v LD_PRELOAD && ! -v LD_LIBRARY_PATH '
                                    '&& ! -v GIT_CONFIG_COUNT && ! -v GIT_DIR ]] || exit 97\n')
                    text = wrapper.replace('umask 077\n', 'umask 077\n'+poison)
                    text = text.replace('export GIT_OPTIONAL_LOCKS=0\n',
                                        'export GIT_OPTIONAL_LOCKS=0\n'+clean_assert)
                    script.write_text(text.replace('/opt/slurm/current/bin/srun', str(srun)))
                    (root/'WINDOW').write_text(window)
                    args = ['bash', str(script), mode, str(root/'source'), 'a'*40, 'b'*40]
                    if mode in ('checks', 'checks-existing'):
                        args += ['c'*64, 'd'*64, 'e'*64]
                    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               text=True, env=dict(os.environ))
                    try:
                        ready = select.select([process.stdout], [], [], 8)[0]
                        self.assertTrue(ready, 'fake child did not reach exact signal window')
                        self.assertIn('READY', process.stdout.readline())
                        process.send_signal(signal.SIGTERM)
                        out, err = process.communicate(timeout=8)
                        self.assertEqual(process.returncode, 143, (out, err))
                        self.assertEqual((root/'FAILED').read_text(), '143')
                        self.assertFalse((root/'COMPLETED').exists())
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.communicate()


class ExistingInstallPureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = B.qomp_library()

    def test_reserved_argname_red_and_exact_real_test_alpha_rename_green(self):
        original = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'show',
            B.QOMP_SCI+':'+C.EXISTING_TEST_FILE], text=True)
        corrected = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'show',
            C.EXISTING_TEST_COMMIT+':'+C.EXISTING_TEST_FILE], text=True)
        with self.assertRaisesRegex(ValueError, 'reserved'):
            C.existing_parametrize_lint(original)
        C.existing_parametrize_lint(corrected)
        C.existing_test_ast(original, corrected)
        for altered in (corrected.replace('(2, 1, 6)', '(2, 1, 14)'),
                        corrected.replace('<= envelope', '< envelope'),
                        corrected.replace('[1, 3, 17]', '[1, 3]')):
            with self.assertRaises(ValueError):
                C.existing_test_ast(original, altered)
        self.assertNotIn('psi4', sys.modules)

    def test_argname_literal_forms_and_uninspectable_refusal(self):
        for names in ('"a,b"', '["a", "b"]', '("a", "b")'):
            C.existing_parametrize_lint('@pytest.mark.parametrize('+names+',[(1,2)])\ndef t(): pass')
        for names in ('"request"', '["a", "request"]', '("request",)', 'ARGNAMES',
                      '"a,a"', '""', '["a", 3]'):
            with self.assertRaises(ValueError):
                C.existing_parametrize_lint('@pytest.mark.parametrize('+names+',[])\ndef t(): pass')
        # Lint all actual corrected fixed selectors from Git, no source import.
        for target in B.qomp_targets():
            name = Path(target.split('::')[0]).name
            text = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'show',
                C.EXISTING_TEST_COMMIT+':tests/pytests/'+name], text=True)
            C.existing_parametrize_lint(text)

    def test_normal_gates_byte_exact_and_existing_modes_cannot_build(self):
        import ast
        text = (ROOT/'devtools/isapol_s66_campaign/phoenix_installed_checks.py').read_text()
        previous = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'show',
            C.EXISTING_BASE_OPS+':devtools/isapol_s66_campaign/phoenix_installed_checks.py'], text=True)
        for name in ('qomp_build_gate', 'qomp_qualify', 'qomp_main', 'main', 'verify_runtime'):
            def extract(source):
                node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                            and n.name == name)
                return ast.get_source_segment(source, node)
            self.assertEqual(extract(text), extract(previous))
        part = text[text.index('# The following diagnostic path'):]
        for forbidden in ('qomp_build_gate(', 'clone_runtime(', 'qomp_materialize(',
                          'qomp_build(', 'subprocess.Popen(', 'build-identity.json\'), manifest'):
            self.assertNotIn(forbidden, part)

    def test_manifest_duplicate_view_preserves_raw_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix, manifest = root/'env', root/'manifest'
            manifest.write_text(f'{prefix}/x\n{prefix}/y\n{prefix}/x\n')
            before = manifest.read_bytes()
            self.assertEqual(C.existing_manifest(manifest, prefix, 3, 2), {'x', 'y'})
            self.assertEqual(manifest.read_bytes(), before)
            with self.assertRaises(ValueError):
                C.existing_manifest(manifest, prefix, 3, 3)
            for value in (str(root/'outside'), str(prefix/'../outside'), 'relative'):
                manifest.write_text(value+'\n')
                with self.assertRaises(ValueError):
                    C.existing_manifest(manifest, prefix, 1, 1)

    def test_complete_snapshot_namespace_and_resolved_target_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree, external = root/'build', root/'external'
            tree.mkdir()
            external.write_text('verified dependency')
            (tree/'owned').write_text('object')
            (tree/'inside').symlink_to(tree/'owned')
            (tree/'dependency').symlink_to(external)
            expected = {str(external): C.digest(external)}
            before = C.existing_snapshot(tree, expected)
            self.assertEqual(before['dependency']['resolved'], str(external))
            with self.assertRaisesRegex(ValueError, 'escaping'):
                C.existing_snapshot(tree, {})
            external.write_text('changed')
            with self.assertRaises(ValueError):
                C.existing_snapshot(tree, expected)
            (tree/'dependency').unlink()
            (tree/'extra').write_text('new metadata is visible')
            self.assertNotEqual(C.existing_snapshot(tree, {}), before)
            (tree/'directory').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                C.existing_snapshot(tree, {})
            (tree/'directory').unlink()
            (tree/'broken').symlink_to(root/'absent')
            with self.assertRaises(FileNotFoundError):
                C.existing_snapshot(tree, {})

    def test_all_prefix_attribution_rejects_unknown_changes_and_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            prefix = build/'env'
            prefix.mkdir()
            (prefix/'dependency').write_text('sealed')
            original = dict(runtime_files={'dependency': C.digest(prefix/'dependency')},
                            runtime_directories=['.'])
            def gate(installed=(), cmake=None):
                return C.existing_attribution(prefix, build, original, set(installed), {}, cmake or {})
            self.assertEqual(set(gate()), {'dependency'})
            (prefix/'arbitrary.so').write_text('unknown ELF')
            with self.assertRaisesRegex(ValueError, 'unexplained prefix'):
                gate()
            with self.assertRaisesRegex(ValueError, 'native dependency'):
                gate(['arbitrary.so'])
            (prefix/'arbitrary.so').unlink()
            (prefix/'empty').mkdir()
            with self.assertRaisesRegex(ValueError, 'directory'):
                gate()
            (prefix/'empty').rmdir()
            (prefix/'dependency').write_text('dependency tamper')
            with self.assertRaises(ValueError):
                gate(['dependency'])

    def test_installed_python_is_strict_before_old_hash_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            package = build/'env/lib/python3.13/site-packages/psi4'
            package.mkdir(parents=True)
            path = package/'metadata.py'
            path.write_text('# configured\n')
            original = dict(runtime_files={str(path.relative_to(build/'env')): C.digest(path)},
                            runtime_directories=['.'])
            with self.assertRaisesRegex(ValueError, 'unmapped installed Python'):
                C.existing_attribution(build/'env', build, original, set(), {}, {})
            checked = {'metadata.py': dict(sha256=C.digest(path))}
            self.assertEqual(len(C.existing_attribution(
                build/'env', build, original, set(), checked, {})), 1)
            path.write_text('# tamper\n')
            with self.assertRaises(ValueError):
                C.existing_attribution(build/'env', build, original, set(), checked, {})

    def test_literal_cmake_origin_not_arbitrary_installed_hash_or_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            inner = build/'build/psi4-core-prefix/src/psi4-core-build'
            prefix = build/'env'
            inner.mkdir(parents=True)
            configured = inner/'share/cmake/psi4/psi4Config.cmake'
            configured.parent.mkdir(parents=True)
            configured.write_text('# generated\n')
            script = inner/'cmake_install.cmake'
            def install(origin):
                script.write_text('file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/psi4" '
                                  'TYPE FILE FILES "'+str(origin)+'")\n')
            install(configured)
            mapping = C.existing_cmake_origins(build, prefix)
            destination = prefix/'share/cmake/psi4/psi4Config.cmake'
            destination.parent.mkdir(parents=True)
            destination.write_bytes(configured.read_bytes())
            original = dict(runtime_files={}, runtime_directories=['.'])
            attributed = C.existing_attribution(prefix, build, original,
                {str(destination.relative_to(prefix))}, {}, mapping)
            self.assertEqual(len(attributed), 1)
            for origin in (build.parent/'escape.cmake', inner/'${UNKNOWN}.cmake'):
                install(origin)
                with self.assertRaisesRegex(ValueError, 'CMake install origin'):
                    C.existing_cmake_origins(build, prefix)

    def test_accounting_requires_actual_three_rows_with_only_known_extern(self):
        rows = '15320675|FAILED|1:0\n15320675.batch|FAILED|1:0\n15320675.0|FAILED|1:0\n'
        with patch.object(C.subprocess, 'check_output', return_value=rows):
            self.assertEqual(len(C.existing_accounting('15320675', 'FAILED', '1:0', B)), 3)
        for text in (rows.replace('.0|FAILED', '.0|COMPLETED'), rows+rows.splitlines()[0]+'\n',
                     rows+'15320675.extra|FAILED|1:0\n', rows.replace('1:0', '0:0')):
            with patch.object(C.subprocess, 'check_output', return_value=text):
                with self.assertRaises(ValueError):
                    C.existing_accounting('15320675', 'FAILED', '1:0', B)
        with patch.object(C.subprocess, 'check_output', return_value=
                          rows+'15320675.extern|COMPLETED|0:0\n'):
            C.existing_accounting('15320675', 'FAILED', '1:0', B)

    def test_historical_literal_anchors_closed_paths_and_failed_receipts(self):
        paths = C.existing_paths(B)
        self.assertEqual(paths['build'], B.QOMP_BUILD)
        self.assertEqual(paths['ops'].name, 'psi4.ccasp-native-s66-qomp-ops004')
        self.assertEqual(paths['tests'].name, 'psi4.ccasp-native-s66-qomp-tests003')
        self.assertEqual(C.EXISTING_ANCHORS['result.json'],
                         'a67fe225027d06d2cb805b3c606b2f5f9a753670551c772ce5dd111eea5a0400')
        self.assertEqual(C.EXISTING_ANCHORS['protected-before.json'],
                         C.EXISTING_ANCHORS['protected-after.json'])
        for value in C.EXISTING_ANCHORS.values():
            self.assertRegex(value, r'^[0-9a-f]{64}$')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(C, 'existing_paths', return_value=dict(build=root)):
                with self.assertRaisesRegex(ValueError, 'historical artifact'):
                    C.existing_historical(B, self.lib)

    def test_new_collect_pytest_targets_corrected_source_without_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tests = root/'corrected'
            for target in B.qomp_targets(tests):
                p = Path(target.split('::')[0])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('# inert source, never imported\n')
            self.lib.write(root/'installed-python-checks.json', {
                'tests/isapol_water_recipe.py': dict(sha256='0'*64)})
            lib = self.lib
            class FakeRunner:
                def run(self, label, command):
                    self.command = command
                    lib.write(root/(label+'-command.json'), list(map(str, command)))
                    lib.write(root/(label+'-result.json'), dict(returncode=0))
                    lib.write(root/'collection-nodes.json', collected())
            runner = FakeRunner()
            runner.root = root
            identity = dict(commit=C.EXISTING_TEST_COMMIT, tree=C.EXISTING_TEST_TREE,
                            path=str(tests/'tests/pytests'))
            B.qomp_pytest(runner, root/'env', root/'scratch', lib, collect=True,
                         test_source=tests, test_identity=identity)
            saved = lib.read(root/'collection.json')
            self.assertEqual(saved['source'], identity)
            self.assertEqual(saved['targets'], B.qomp_targets(tests))
            script = runner.command[-1]
            self.assertIn("--rootdir="+str(tests), script)
            self.assertNotIn('sys.path.insert(0,'+repr(str(tests)), script)
            self.assertLess(script.index('import psi4'), script.index('import pytest'))
            self.assertNotIn('psi4', sys.modules)

    def test_new_operation_integrity_finally_on_native_exception_and_signal(self):
        # Exercise actual perform/finalize control flow with fake identities and
        # an inert Runner. No compiler/Psi4/allocation/server command is invoked.
        for error in (ValueError('native fake failure'), InterruptedError('fake TERM')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = {k: root/k for k in ('ops', 'tests', 'build', 'collect', 'checks')}
                args = NS(root=paths['collect'], scratch=root/'scratch', mode='collect-existing',
                          ops_commit='a'*40, ops_tree='b'*40, launch_token='test-token')
                baseline = dict(historical_identity={'fake': True})
                lib = self.lib
                runner = NS(stop=lambda *a: None, check=lambda: None)
                prior = Path.cwd()
                try:
                    with patch.object(C, 'existing_paths', return_value=paths), \
                         patch.object(C, 'existing_source_gate', return_value={'roles': 'fake'}), \
                         patch.object(C, 'allocation', return_value={'JobId': 'fake'}), \
                         patch.object(lib, 'allocation'), patch.object(B, 'qomp_runner', return_value=runner), \
                         patch.object(C, 'existing_current', return_value=baseline) as current, \
                         patch.object(B, 'qomp_identity', side_effect=error):
                        with self.assertRaises(type(error)):
                            C.existing_perform(B, lib, args)
                    self.assertEqual(current.call_count, 2)
                    result = lib.read(args.root/'result.json')
                    self.assertEqual(result['status'], 'FAILED')
                    self.assertTrue(result['preserved_current_build_and_old_base'])
                    self.assertFalse((args.root/'verification.json').exists())
                    self.assertFalse((args.root/'COMPLETED').exists())
                finally:
                    os.chdir(prior)

    def test_new_finalizer_failure_monotonic_and_no_historical_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = NS(root=root, mode='collect-existing', launch_token='token',
                      ops_commit='a'*40, ops_tree='b'*40, exit_code=143)
            self.lib.write(root/'owner.json', dict(token='token', mode=args.mode))
            self.lib.write(root/'COMPLETED', dict(exit_code=0, mode=args.mode, acceptance=False))
            self.assertEqual(C.existing_finalize(B, self.lib, args), 143)
            self.assertTrue((root/'COMPLETED_INVALIDATED').exists())
            self.assertFalse((root/'COMPLETED').exists())
            args.exit_code = 0
            self.assertEqual(C.existing_finalize(B, self.lib, args), 143)
            self.assertEqual(self.lib.read(root/'FAILED')['exit_code'], 143)

    def test_distinct_collection_digest_receipt_and_role_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = dict(collect=root/'collect', tests=root/'tests')
            paths['collect'].mkdir()
            for target in B.qomp_targets(paths['tests']):
                p = Path(target.split('::')[0])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('# test data only\n')
            roles, identity, allocated = {'roles': 'exact'}, {'identity': 'exact'}, {'JobId': '100'}
            collection = dict(collected(), source=dict(commit=C.EXISTING_TEST_COMMIT,
                tree=C.EXISTING_TEST_TREE, path=str(paths['tests']/'tests/pytests')),
                targets=B.qomp_targets(paths['tests']), test_source_sha256={
                    Path(p.split('::')[0]).name: C.digest(p.split('::')[0])
                    for p in B.qomp_targets(paths['tests'])})
            def save(name, value):
                (paths['collect']/name).write_text(json.dumps(value))
            save('collect-tests-command.json', ['fixed', 'argv'])
            save('collect-tests-result.json', dict(
                argv=['fixed', 'argv'], returncode=0, interrupted=False))
            for name in ('current-before.json', 'current-after.json'):
                save(name, dict(current='attributed'))
            collection.update(command_sha256=C.digest(paths['collect']/'collect-tests-command.json'),
                command_result_sha256=C.digest(paths['collect']/'collect-tests-result.json'))
            receipt = dict(schema='existing-failed-build-install-verification/1', source=roles,
                historical_build_status='FAILED', historical_job='15320675', anchors=C.EXISTING_ANCHORS,
                acceptance=False, production_adoption=False, identity=identity, allocation=allocated,
                current_before_sha256=C.digest(paths['collect']/'current-before.json'),
                current_after_sha256=C.digest(paths['collect']/'current-after.json'),
                reconstructed_historical_full_seal=False,
                historical_component_returncodes=dict(configure=0, install=0, identity=0, collection=2))
            result = dict(status='VERIFIED_EXISTING_INSTALL_COLLECTED_NOT_QUALIFIED',
                source=roles, acceptance=False, production_adoption=False, identity=identity,
                allocation=allocated, preserved_current_build_and_old_base=True)
            save('COMPLETED', dict(exit_code=0, mode='collect-existing', acceptance=False))
            def invoke(r=receipt, c=collection, v=result):
                for name, value in (('verification.json', r), ('collection.json', c), ('result.json', v)):
                    save(name, value)
                args = NS(ops_commit='a'*40, ops_tree='b'*40,
                    verification_sha256=C.digest(paths['collect']/'verification.json'),
                    collection_sha256=C.digest(paths['collect']/'collection.json'),
                    collection_result_sha256=C.digest(paths['collect']/'result.json'))
                return C.existing_collection_gate(B, self.lib, args)
            with patch.object(C, 'existing_paths', return_value=paths), \
                 patch.object(C, 'existing_source_gate', return_value=roles), \
                 patch.object(C, 'existing_accounting'):
                invoke()
                for changed in (dict(receipt, schema='fresh-native-build-identity/1'),
                                dict(receipt, historical_build_status='COMPLETED'),
                                dict(receipt, source={'wrong': 'ops/test/compiled'}),
                                dict(receipt, current_before_sha256='0'*64),
                                dict(receipt, anchors={}),
                                dict(receipt, identity={'stale': '8558'})):
                    with self.assertRaises(ValueError):
                        invoke(r=changed)
                for count in (32, 82, 547, 555):
                    with self.assertRaises(ValueError):
                        invoke(c=dict(collection, nodes=collection['nodes'][:count]))
                save('FAILED', dict(exit_code=1))
                with self.assertRaises(ValueError):
                    invoke()

    def test_success_publication_after_integrity_and_initial_gap_before_import(self):
        for state in ('success', 'initial-gap', 'postcheck-change'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = {k: root/k for k in ('ops', 'tests', 'build', 'collect', 'checks')}
                args = NS(root=paths['collect'], scratch=root/'scratch', mode='collect-existing',
                          ops_commit='a'*40, ops_tree='b'*40, launch_token='fake')
                identity = dict(fake='identity')
                baseline = dict(historical_identity=identity)
                runner = NS(stop=lambda *unused: None, check=lambda: None)
                def pytest(*unused, **kwargs):
                    self.assertTrue(kwargs['collect'])
                    self.assertEqual(kwargs['test_source'], paths['tests'])
                    self.lib.write(args.root/'collection.json', collected())
                    return collected()
                current = ([baseline, baseline] if state == 'success' else
                           [ValueError('attribution gap'), ValueError('attribution gap')]
                           if state == 'initial-gap' else [baseline, dict(changed=True)])
                prior = Path.cwd()
                try:
                    with patch.object(C, 'existing_paths', return_value=paths), \
                         patch.object(C, 'existing_source_gate', return_value={'roles': 'fake'}), \
                         patch.object(C, 'allocation', return_value={'JobId': 'fake'}), \
                         patch.object(self.lib, 'allocation'), \
                         patch.object(B, 'qomp_runner', return_value=runner), \
                         patch.object(C, 'existing_current', side_effect=current), \
                         patch.object(B, 'qomp_identity', return_value=identity) as imported, \
                         patch.object(C, 'existing_identity'), \
                         patch.object(B, 'qomp_pytest', side_effect=pytest):
                        if state == 'success':
                            C.existing_perform(B, self.lib, args)
                        else:
                            with self.assertRaises(ValueError):
                                C.existing_perform(B, self.lib, args)
                    result = self.lib.read(args.root/'result.json')
                    if state == 'success':
                        self.assertEqual(result['status'],
                            'VERIFIED_EXISTING_INSTALL_COLLECTED_NOT_QUALIFIED')
                        receipt = self.lib.read(args.root/'verification.json')
                        self.assertEqual(receipt['historical_build_status'], 'FAILED')
                        self.assertFalse(receipt['acceptance'])
                        self.assertFalse(receipt['reconstructed_historical_full_seal'])
                    else:
                        self.assertEqual(result['status'], 'FAILED')
                        self.assertFalse((args.root/'verification.json').exists())
                    if state == 'initial-gap':
                        imported.assert_not_called()
                finally:
                    os.chdir(prior)

    def test_existing_source_gate_separate_roles_and_exact_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {k: root/k for k in ('ops', 'tests', 'build', 'collect', 'checks')}
            science = root/'science'
            corrected = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'show',
                C.EXISTING_TEST_COMMIT+':'+C.EXISTING_TEST_FILE], text=True)
            original = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'show',
                B.QOMP_SCI+':'+C.EXISTING_TEST_FILE], text=True)
            for target in B.qomp_targets(paths['tests']):
                p = Path(target.split('::')[0])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(corrected if p.name == Path(C.EXISTING_TEST_FILE).name else '# inert\n')
            old = science/C.EXISTING_TEST_FILE
            old.parent.mkdir(parents=True)
            old.write_text(original)
            commits = {paths['ops']: ('a'*40, 'b'*40), science: (B.QOMP_SCI, B.QOMP_TREE),
                       paths['build']/'source': (B.QOMP_SCI, B.QOMP_TREE),
                       paths['tests']: (C.EXISTING_TEST_COMMIT, C.EXISTING_TEST_TREE)}
            def git(path, *args):
                if args[0] == 'merge-base':
                    return C.EXISTING_BASE_OPS
                if args[0] == 'diff':
                    return (C.EXISTING_TEST_FILE if path == paths['tests']
                            else 'devtools/isapol_s66_campaign/phoenix_installed_checks.py')
                if args == ('rev-parse', '--show-toplevel'):
                    return str(path)
                if args == ('rev-parse', 'HEAD'):
                    return commits[path][0]
                if args == ('rev-parse', 'HEAD^{tree}'):
                    return commits[path][1]
                return ''
            args = NS(ops_commit='a'*40, ops_tree='b'*40)
            with patch.object(C, 'existing_paths', return_value=paths), \
                 patch.object(B, 'SOURCE', paths['ops']), patch.object(B, 'QOMP_SCI_SOURCE', science), \
                 patch.object(B, 'qomp_checks_module', return_value=C), \
                 patch.object(C.subprocess, 'run', return_value=NS(returncode=1)), \
                 patch.object(B, 'qomp_git', side_effect=git):
                value = C.existing_source_gate(B, args)
                self.assertEqual(value['compiled']['commit'], B.QOMP_SCI)
                self.assertEqual(value['tests']['commit'], C.EXISTING_TEST_COMMIT)
                self.assertNotEqual(value['compiled'], value['tests'])
                with patch.object(C.subprocess, 'run', return_value=NS(returncode=0)):
                    with self.assertRaisesRegex(ValueError, 'must be detached'):
                        C.existing_source_gate(B, args)
                for path in commits:
                    previous = commits[path]
                    commits[path] = ('0'*40, previous[1])
                    with self.assertRaisesRegex(ValueError, 'source role'):
                        C.existing_source_gate(B, args)
                    commits[path] = previous
                with patch.object(B, 'qomp_git', side_effect=lambda p, *a:
                    'psi4/driver/forbidden.py' if a[0] == 'diff' else git(p, *a)):
                    with self.assertRaisesRegex(ValueError, 'allowlist'):
                        C.existing_source_gate(B, args)

    def test_current_identity_prefix_pair_version_and_real_core_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            package = prefix/'lib/python3.13/site-packages/psi4'
            package.mkdir(parents=True)
            core = package/'core.test.so'
            core.write_bytes(b'inert NOT a native library')
            pair = dict(core=C.digest(core), isapol_python='1'*64)
            value = dict(implementation=pair, version='1.12a1.dev212',
                         package=str(package/'__init__.py'), core=str(core))
            with patch.object(C, 'EXISTING_PAIR', pair):
                C.existing_identity(value, prefix)
                for replacement in (dict(value, version='1.12a1.dev213'),
                                    dict(value, package='/foreign/__init__.py'),
                                    dict(value, implementation={'old': '8558'}),
                                    dict(value, core=str(prefix/'foreign.so'))):
                    with self.assertRaises(ValueError):
                        C.existing_identity(replacement, prefix)
                core.write_bytes(b'tamper')
                with self.assertRaises(ValueError):
                    C.existing_identity(value, prefix)


class BaseLink152Tests(unittest.TestCase):
    NAME = 'libexec/gcc/x86_64-conda-linux-gnu/14.3.0/TOOLS=addr2line'
    RAW = '../../../../bin/x86_64-conda-linux-gnu-TOOLS=addr2line'
    TARGET = 'bin/x86_64-conda-linux-gnu-TOOLS=addr2line'
    CMAKE = 'share/cmake/Einsums/tests/cxx23_static_call_operator.'
    CMAKE_RAW = '../../cmake/tests/cxx23_static_call_operator.cpp'

    def fixture(self, root):
        base, build, old = root/'base', root/'build', root/'old'
        (base/Path(self.NAME).parent).mkdir(parents=True)
        (base/'bin').mkdir()
        (base/self.NAME).symlink_to(self.RAW)
        (base/Path(self.CMAKE).parent).mkdir(parents=True)
        for suffix in ('cu', 'hip'):
            (base/(self.CMAKE+suffix)).symlink_to(self.CMAKE_RAW)
        normal = base/'bin/x86_64-conda-linux-gnu-addr2line'
        normal.write_bytes(b'inert sealed normal addr2line fixture, never executed')
        (base/'bin/addr2line').symlink_to(normal.name)
        (build/'env').mkdir(parents=True)
        (old/'env').mkdir(parents=True)
        protected = dict(base=B.qomp_inventory(base, NS(sha=C.digest)), old={},
                         manifest_sha256='0'*64, record_sha256='1'*64)
        for name in ('protected-before.json', 'protected-after.json'):
            (build/name).write_text(json.dumps(protected))
        (build/'installed-python-checks.json').write_text('{}')
        original = dict(runtime_files={}, runtime_directories=['.'], installed_files={},
                        external_runtime_files={})
        lib = NS(OLD=old, read=lambda p: json.loads(Path(p).read_text()),
                 old_seal=lambda: original, installed_python_gate=lambda *a: {},
                 dependency_gate=lambda *a: None)
        engine = NS(QOMP_BASE=base, qomp_protected=lambda lib: protected,
                    qomp_installed_names=lambda names: names)
        return NS(base=base, build=build, old=old, protected=protected, lib=lib, engine=engine,
                  baseline_sha=C.digest(build/'protected-before.json'),
                  baseline_bytes=(build/'protected-before.json').stat().st_size,
                  normal_sha=C.digest(normal))

    def proof(self, f):
        """Only ephemeral fixture pins, never observed shared-BASE selfapproval."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(C, 'existing_paths', return_value=dict(build=f.build)))
        stack.enter_context(patch.dict(C.EXISTING_ANCHORS, {
            'protected-before.json': f.baseline_sha, 'protected-after.json': f.baseline_sha}))
        stack.enter_context(patch.object(C, 'EXISTING_PROTECTED_BYTES', f.baseline_bytes))
        stack.enter_context(patch.object(C, 'EXISTING_ADDR2LINE_SHA', f.normal_sha))
        return stack

    def test_exact_sealed_base_case_through_complete_current_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            anchors = dict(C.EXISTING_ANCHORS,
                           **{'protected-before.json': f.baseline_sha,
                              'protected-after.json': f.baseline_sha})
            with patch.object(C, 'existing_paths', return_value=dict(build=f.build)), \
                 patch.object(C, 'existing_historical', return_value=({}, set(), [])), \
                 patch.dict(C.EXISTING_ANCHORS, anchors), \
                 patch.object(C, 'EXISTING_PROTECTED_BYTES', f.baseline_bytes, create=True):
                before = C.existing_current(f.engine, f.lib)
                after = C.existing_current(f.engine, f.lib)
            self.assertEqual(before, after)
            marker = before['protected_bindings']['base'][self.NAME]
            self.assertEqual(marker['kind'], 'known-dangling-base-metadata-link')
            self.assertEqual(marker['target_absence'], 'current-lstat-ENOENT')
            self.assertEqual(marker['target'], self.RAW)
            self.assertNotIn('sha256', marker)
            # The original raw BASE snapshot remains an unchanged link row.
            self.assertEqual(before['protected']['base'][self.NAME],
                             dict(kind='link', target=self.RAW))
            for suffix in ('cu', 'hip'):
                cmake = before['protected_bindings']['base'][self.CMAKE+suffix]
                self.assertEqual(cmake['kind'], 'known-dangling-base-metadata-link')
                self.assertEqual(cmake['absent_anchor'], str(f.base/'share/cmake/cmake'))
                self.assertNotIn('sha256', cmake)

    def test_parent_literals_and_exact_three_entry_table(self):
        self.assertEqual(C.EXISTING_PROTECTED_BYTES, 18257504)
        self.assertEqual(C.EXISTING_ANCHORS['protected-before.json'],
                         '99558a98ede3f04c1c854b425e6a967e68a7b0579529725dda2cc04453918525')
        self.assertEqual(C.EXISTING_ADDR2LINE_SHA,
                         'c1f9abaad54a61ae6cb621566b4388f5265853ffb4de7b26c004dbe34d66966f')
        self.assertEqual(C.EXISTING_BASE_METADATA, (
            (self.NAME, self.RAW, self.TARGET, self.TARGET),
            (self.CMAKE+'cu', self.CMAKE_RAW,
             'share/cmake/cmake/tests/cxx23_static_call_operator.cpp', 'share/cmake/cmake'),
            (self.CMAKE+'hip', self.CMAKE_RAW,
             'share/cmake/cmake/tests/cxx23_static_call_operator.cpp', 'share/cmake/cmake')))
        self.assertEqual(C.EXISTING_BASE_OPS, 'cef3c6cdecdefb7a6a356d2019a6d07d0a121cd2')
        self.assertEqual(C.existing_paths(B)['collect'].name, 'ccasp-native-s66-qomp-fb11-collection004')
        self.assertEqual(C.existing_paths(B)['checks'].name, 'ccasp-native-s66-qomp-fb11-installed004')
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            C.existing_base_metadata_case(Path('/not-accessed'), {}, ('unknown', 'x', 'x', 'x'))

    def test_targeted_preflight_no_walk_execute_or_native_and_valid_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            with self.proof(f), patch.object(Path, 'rglob', side_effect=AssertionError('no tree walk')), \
                 patch.object(os, 'scandir', side_effect=AssertionError('no scandir')), \
                 patch.object(subprocess, 'run', side_effect=AssertionError('no execute')), \
                 patch.object(subprocess, 'check_output', side_effect=AssertionError('no execute')):
                value = C.existing_base_metadata_preflight(f.engine)
            self.assertEqual(len(value['metadata']), 3)
            self.assertEqual(value['normal_addr2line']['sha256'], f.normal_sha)
            self.assertEqual(value['normal_addr2line']['resolved'],
                             str(f.base/'bin/x86_64-conda-linux-gnu-addr2line'))
            self.assertNotIn('psi4', sys.modules)
            with self.proof(f):
                before = C.existing_snapshot(f.base, {}, base_engine=f.engine)
                normal = before['bin/addr2line']
                self.assertEqual(normal['kind'], 'link')
                self.assertEqual(normal['sha256'], f.normal_sha)
                (f.base/'bin/x86_64-conda-linux-gnu-addr2line').write_bytes(b'changed normal binary')
                with self.assertRaisesRegex(ValueError, 'normal BASE addr2line'):
                    C.existing_base_metadata_preflight(f.engine)
                after = C.existing_snapshot(f.base, {}, base_engine=f.engine)
                self.assertNotEqual(normal, after['bin/addr2line'])

    def test_deny_each_changed_raw_type_missing_link_or_loop(self):
        for case in C.EXISTING_BASE_METADATA:
            for mutation in ('raw', 'file', 'directory', 'missing', 'loop'):
                with self.subTest(name=case[0], mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                    f = self.fixture(Path(tmp))
                    link = f.base/case[0]
                    link.unlink()
                    if mutation == 'raw':
                        link.symlink_to('another-missing-target')
                    elif mutation == 'file':
                        link.write_text('replaced')
                    elif mutation == 'directory':
                        link.mkdir()
                    elif mutation == 'loop':
                        link.symlink_to(link.name)
                    with self.proof(f), self.assertRaises((ValueError, FileNotFoundError)):
                        C.existing_base_metadata_preflight(f.engine)

    def test_deny_anchor_appearance_even_when_cpp_target_still_missing(self):
        for case in C.EXISTING_BASE_METADATA:
            for kind in ('file', 'directory', 'symlink-loop'):
                with self.subTest(name=case[0], kind=kind), tempfile.TemporaryDirectory() as tmp:
                    f = self.fixture(Path(tmp))
                    anchor = f.base/case[3]
                    if kind == 'file':
                        anchor.write_text('new target/ancestor')
                    elif kind == 'directory':
                        anchor.mkdir()
                    else:
                        anchor.symlink_to(anchor.name)
                    with self.proof(f), self.assertRaisesRegex(ValueError, 'anchor appeared'):
                        C.existing_base_metadata_preflight(f.engine)

    def test_deny_other_root_prefix_and_unknown_even_historically_sealed_broken_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            with self.proof(f):
                with self.assertRaisesRegex(ValueError, 'another root'):
                    C.existing_snapshot(f.build/'env', {}, base_engine=f.engine)
                # Same spelling in any prefix has NO metadata exception.
                name = f.build/'env'/self.NAME
                name.parent.mkdir(parents=True)
                name.symlink_to(self.RAW)
                with self.assertRaises(FileNotFoundError):
                    C.existing_snapshot(f.build/'env', {})
                with self.assertRaises(FileNotFoundError):
                    C.existing_snapshot(f.base, {})  # Explicit BASE routing required.
            unknown = f.base/'bin/new-unknown-broken-link'
            unknown.symlink_to('absent')
            f.protected['base'][str(unknown.relative_to(f.base))] = dict(kind='link', target='absent')
            # Even an independently sealed hypothetical broken link is not an
            # allowance: the production table is EXACTLY these three entries.
            for name in ('protected-before.json', 'protected-after.json'):
                (f.build/name).write_text(json.dumps(f.protected))
            f.baseline_sha = C.digest(f.build/'protected-before.json')
            f.baseline_bytes = (f.build/'protected-before.json').stat().st_size
            with self.proof(f), self.assertRaises(FileNotFoundError):
                C.existing_snapshot(f.base, {}, base_engine=f.engine)
            unknown.unlink()
            unknown.symlink_to(unknown.name)
            with self.proof(f), self.assertRaises((RuntimeError, OSError)):
                C.existing_snapshot(f.base, {}, base_engine=f.engine)

    def test_missing_or_linked_ancestor_and_noncanonical_base_fail(self):
        for parent in ('bin', 'libexec/gcc', 'share/cmake', '.'):
            for kind in ('missing', 'symlink'):
                with self.subTest(parent=parent, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    f = self.fixture(root)
                    path = f.base/parent
                    saved = root/'saved'
                    path.rename(saved)
                    if kind == 'symlink':
                        path.symlink_to(saved, target_is_directory=True)
                    with self.proof(f), self.assertRaises((ValueError, FileNotFoundError)):
                        C.existing_base_metadata_preflight(f.engine)

    def test_only_specific_lstat_enoent_not_permissions_loop_or_other_filename(self):
        import errno
        real_lstat = Path.lstat
        for case in C.EXISTING_BASE_METADATA:
            for which in ('permission', 'loop', 'wrong-filename', 'ancestor-permission'):
                with self.subTest(name=case[0], which=which), tempfile.TemporaryDirectory() as tmp:
                    f = self.fixture(Path(tmp))
                    anchor = f.base/case[3]
                    selected = anchor.parent if which == 'ancestor-permission' else anchor
                    def lstat(path, *args, **kwargs):
                        if path == selected:
                            if 'permission' in which:
                                raise PermissionError(errno.EACCES, 'injected permission', str(path))
                            if which == 'loop':
                                raise OSError(errno.ELOOP, 'injected loop', str(path))
                            raise FileNotFoundError(errno.ENOENT, 'wrong missing path',
                                                    str(f.base/'unrelated'))
                        return real_lstat(path, *args, **kwargs)
                    with self.proof(f), patch.object(Path, 'lstat', lstat), \
                         self.assertRaises((ValueError, OSError)):
                        C.existing_base_metadata_preflight(f.engine)

    def test_target_appearing_after_initial_and_row_checks_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            actual = C.existing_base_metadata_case
            calls = {}
            def changing(base, sealed, case):
                row = actual(base, sealed, case)
                calls[case[0]] = calls.get(case[0], 0)+1
                if case[0] == self.CMAKE+'hip' and calls[case[0]] == 2:
                    (base/'share/cmake/cmake').mkdir()  # After the last row's check.
                return row
            with self.proof(f), patch.object(C, 'existing_base_metadata_case', side_effect=changing), \
                 self.assertRaisesRegex(ValueError, 'anchor appeared'):
                C.existing_snapshot(f.base, {}, base_engine=f.engine)

    def test_original_baseline_pin_rows_and_required_normal_link_stay_failclosed(self):
        for mutation in ('sha', 'bytes', 'raw-row', 'anchor-row', 'normal-raw', 'normal-kind',
                         'normal-target', 'baseline-symlink'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                f = self.fixture(Path(tmp))
                with self.proof(f):
                    if mutation == 'sha':
                        with patch.dict(C.EXISTING_ANCHORS, {'protected-before.json': '0'*64}), \
                             self.assertRaisesRegex(ValueError, 'parent-pinned'):
                            C.existing_base_metadata_preflight(f.engine)
                        continue
                    if mutation == 'bytes':
                        (f.build/'protected-before.json').write_bytes(b'changed same-owner metadata')
                    elif mutation in ('raw-row', 'anchor-row'):
                        sealed = f.protected['base']
                        if mutation == 'raw-row':
                            sealed[self.NAME]['target'] = 'unapproved'
                        else:
                            sealed[self.TARGET] = dict(kind='directory')
                        with patch.object(C, 'existing_base_metadata_proof', return_value=(f.base, sealed)), \
                             self.assertRaises(ValueError):
                            C.existing_base_metadata_preflight(f.engine)
                        continue
                    elif mutation in ('normal-raw', 'normal-kind'):
                        link = f.base/'bin/addr2line'
                        link.unlink()
                        if mutation == 'normal-raw':
                            link.symlink_to('./x86_64-conda-linux-gnu-addr2line')
                        else:
                            link.write_text('not a symlink')
                    elif mutation == 'normal-target':
                        target = f.base/'bin/x86_64-conda-linux-gnu-addr2line'
                        target.unlink()
                        target.symlink_to('absent')
                    else:
                        before = f.build/'protected-before.json'
                        before.rename(f.build/'actual-proof')
                        before.symlink_to('actual-proof')
                    with self.assertRaises((ValueError, OSError)):
                        C.existing_base_metadata_preflight(f.engine)


if __name__ == '__main__':
    unittest.main(verbosity=2)
