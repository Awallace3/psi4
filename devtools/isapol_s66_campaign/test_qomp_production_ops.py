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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = '00000000-0000-0000-0000-000000000000'
            args = NS(root=root, launch_token=token, mode='build',
                      ops_commit='a'*40, ops_tree='b'*40, exit_code=0)
            with patch.object(B, 'qomp_library', return_value=self.lib), \
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
        for mode in ('build', 'checks'):
            for window in ('payload', 'finalize'):
                with self.subTest(mode=mode, window=window), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    helper = root/'source/devtools/isapol_native_acceptance/phoenix_build.py'
                    helper.parent.mkdir(parents=True)
                    helper.write_text("""
import os,pathlib,signal,sys
r=pathlib.Path(os.environ['FAKE_ROOT'])
if sys.argv[2]=='finalize':
 code=int(sys.argv[sys.argv.index('--exit-code')+1])
 if code:
  if (r/'COMPLETED').exists(): (r/'COMPLETED').rename(r/'INVALIDATED')
  (r/'FAILED').write_text(str(code));sys.exit(code)
 (r/'COMPLETED').write_text('0')
 print('FINAL_READY',flush=True);signal.pause()
else:
 if os.environ['WINDOW']=='payload':
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
                    script.write_text(wrapper.replace('/opt/slurm/current/bin/srun', str(srun)))
                    args = ['bash', str(script), mode, str(root/'source'), 'a'*40, 'b'*40]
                    if mode == 'checks':
                        args += ['c'*64, 'd'*64, 'e'*64]
                    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               text=True, env=dict(os.environ, FAKE_ROOT=str(root),
                                                                   WINDOW=window))
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
