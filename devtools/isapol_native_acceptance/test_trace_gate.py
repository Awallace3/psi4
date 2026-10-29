"""Synthetic decoded-fd tests; no molecular calculations or cluster calls."""
import tempfile
from pathlib import Path
import unittest

from trace_gate import check_trace, digest


class TraceGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env, self.run, self.base = (self.root/n for n in ('env', 'run', 'base'))
        for path in (self.env, self.run, self.base):
            path.mkdir()
        self.good = self.env/'module.py'
        self.good.write_text('installed module')
        self.manifest = dict(runtime_files={'module.py': digest(self.good)},
                             runtime_directories=['.'], external_runtime_files={})
        self.kw = dict(env=self.env, base=self.base, run=self.run,
                       manifest_path=self.root/'manifest.json',
                       manifest=self.manifest, forbidden=['/research'])
        self.initial = f'1 execve("{self.env}/bin/python", ["python"], 0x0) = 0'

    def check(self, *lines, **kwargs):
        return check_trace([self.initial, *lines], **dict(self.kw, **kwargs))

    def test_installed_and_fresh_own_output(self):
        result = self.check(
            f'1 openat(AT_FDCWD, "module.py", O_RDONLY) = 3<{self.good}>',
            f'1 openat(AT_FDCWD, "fresh.dat", O_RDWR|O_CREAT, 0600) = 4<{self.run}/fresh.dat>')
        self.assertEqual(result['status'], 'passed')

    def test_relative_dirfd_and_symlink_targets_cannot_hide_research(self):
        for spelling in ('"../../research/old.npz"', '"alias"', '"./old.npz"'):
            self.assertEqual(self.check(
                f'1 openat(3</tmp>, {spelling}, O_RDONLY) = 4</research/old.npz>'
            )['status'], 'failed')

    def test_absolute_alias_is_normalized_even_on_failure(self):
        self.assertEqual(self.check(
            '1 openat(AT_FDCWD, "/research/./old.npz", O_RDONLY) = -1 ENOENT'
        )['status'], 'failed')

    def test_unlisted_historical_and_unsealed_runtime_inputs_refuse(self):
        for path in (self.root/'old.npz', self.env/'old.npz'):
            self.assertEqual(self.check(
                f'1 openat(AT_FDCWD, "old.npz", O_RDONLY) = 3<{path}>'
            )['status'], 'failed')

    def test_runtime_mutation_is_refused(self):
        self.good.write_text('different bytes')
        self.assertEqual(self.check(
            f'1 openat(AT_FDCWD, "module.py", O_RDONLY) = 3<{self.good}>'
        )['status'], 'failed')

    def test_resumed_child_exec_is_refused(self):
        self.assertEqual(self.check(
            '2 execve("/usr/bin/helper", ["helper"], 0x0 <unfinished ...>',
            '2 <... execve resumed>) = 0',
        )['status'], 'failed')

    def test_resumed_open_is_checked(self):
        self.assertEqual(self.check(
            '1 openat(AT_FDCWD, "old.npz", O_RDONLY <unfinished ...>',
            '1 <... openat resumed>) = 3</research/old.npz>',
        )['status'], 'failed')

    def test_ambiguous_incomplete_trace_fails_closed(self):
        for lines in (
            ['1 openat(AT_FDCWD, "x", O_RDONLY) = 3'],
            ['1 openat(AT_FDCWD, "x", O_RDONLY <unfinished ...>'],
            ['1 <... openat resumed>) = 3</x>'],
        ):
            with self.assertRaises(ValueError):
                self.check(*lines)

    def test_only_fresh_own_pid_openmp_registration(self):
        path = '/dev/shm/__KMP_REGISTERED_LIB_1_123'
        create = f'1 openat(AT_FDCWD, "{path}", O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW, 0600) = 3<{path}>'
        read = f'1 openat(AT_FDCWD, "{path}", O_RDONLY|O_NOFOLLOW) = 3<{path}>'
        self.assertEqual(self.check(create, read)['status'], 'passed')
        self.assertEqual(self.check(read)['status'], 'failed')
        self.assertEqual(self.check(create.replace('LIB_1_', 'LIB_2_'))['status'], 'failed')

    def test_external_dependencies_must_be_sealed(self):
        path = self.base/'external.so'
        path.write_text('ordinary external runtime')
        syscall = f'1 openat(AT_FDCWD, "external.so", O_RDONLY) = 3<{path}>'
        self.assertEqual(self.check(syscall)['status'], 'failed')
        self.manifest['external_runtime_files'][str(path)] = digest(path)
        self.assertEqual(self.check(syscall)['status'], 'passed')
        path.write_text('mutated')
        self.assertEqual(self.check(syscall)['status'], 'failed')

    def test_only_explicit_unchanged_geometry_input(self):
        path = self.root/'geometry.json'
        path.write_text('new raw geometry declaration')
        syscall = f'1 openat(AT_FDCWD, "geometry.json", O_RDONLY) = 3<{path}>'
        self.assertEqual(self.check(syscall)['status'], 'failed')
        allowed = {str(path): digest(path)}
        self.assertEqual(self.check(syscall, allowed_inputs=allowed)['status'], 'passed')
        path.write_text('different bytes')
        self.assertEqual(self.check(syscall, allowed_inputs=allowed)['status'], 'failed')

    def test_only_explicit_fresh_scratch_root(self):
        scratch = self.root/'fresh-scratch'
        scratch.mkdir()
        path = scratch/'own-checkpoint.npy'
        syscall = f'1 openat(AT_FDCWD, "own-checkpoint.npy", O_RDWR|O_CREAT, 0600) = 3<{path}>'
        self.assertEqual(self.check(syscall)['status'], 'failed')
        self.assertEqual(self.check(syscall, scratch=scratch)['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
