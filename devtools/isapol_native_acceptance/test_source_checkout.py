"""Small Git-only regression: sparse staging must not produce sparse builds."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from phoenix_build import materialize_source


def git(directory, *args):
    return subprocess.check_output(['git', *args], cwd=directory, text=True,
                                   stderr=subprocess.PIPE).strip()


class SourceCheckoutTests(unittest.TestCase):
    def test_sparse_launcher_produces_complete_private_build_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, launcher, source = (root/name for name in ('repo', 'launcher', 'build-source'))
            repository.mkdir()
            git(repository, 'init', '-q')
            git(repository, 'config', 'user.name', 'Synthetic checkout test')
            git(repository, 'config', 'user.email', 'checkout-test@example.invalid')
            required = (
                'CMakeLists.txt', 'cmake/psi4OptionsTools.cmake', 'psi4/src/export_isapol.cc',
                'psi4/driver/procrouting/isapol_benzene_acceptance.py',
                'devtools/isapol_native_acceptance/placeholder.txt')
            for name in required:
                path = repository/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic source fixture; no engine\n')
            git(repository, 'add', '.')
            git(repository, 'commit', '-qm', 'Synthetic complete source')
            commit, tree = git(repository, 'rev-parse', 'HEAD'), git(repository, 'rev-parse', 'HEAD^{tree}')
            git(root, 'clone', '--shared', '--no-checkout', str(repository), str(launcher))
            git(launcher, 'sparse-checkout', 'init', '--cone')
            git(launcher, 'sparse-checkout', 'set', 'devtools/isapol_native_acceptance')
            git(launcher, 'checkout', '--detach', commit)
            self.assertFalse((launcher/'cmake/psi4OptionsTools.cmake').exists())
            materialize_source(launcher, source, commit, tree, dict(os.environ))
            for name in required:
                self.assertTrue((source/name).is_file(), name)
            self.assertEqual(git(source, 'status', '--porcelain'), '')
            self.assertEqual(git(source, 'rev-parse', 'HEAD^{tree}'), tree)
            # Materialization must not mutate the sparse operational checkout.
            self.assertFalse((launcher/'cmake/psi4OptionsTools.cmake').exists())
            self.assertEqual(git(launcher, 'config', '--get', 'core.sparseCheckout'), 'true')
            self.assertEqual(git(launcher, 'status', '--porcelain'), '')


if __name__ == '__main__':
    unittest.main()
