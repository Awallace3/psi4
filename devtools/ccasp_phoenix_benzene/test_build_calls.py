# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Stdlib-only regression tests for build command recording, with no child execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "ccasp_build_smoke", Path(__file__).with_name("build_smoke.py")
)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class BuildCallTests(unittest.TestCase):
    def exercise(self, returncode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir()
            (root / "logs").mkdir()
            command = [Path("/private/bin/cmake"), "-S", Path("/private/source")]
            expected = list(map(str, command))

            def run(argv, **kwargs):
                self.assertEqual(argv, expected)
                self.assertTrue(all(isinstance(arg, str) for arg in argv))
                self.assertEqual(kwargs["cwd"], root)
                return subprocess.CompletedProcess(argv, returncode)

            with mock.patch.object(BUILD, "RUN", root), mock.patch.object(
                BUILD.subprocess, "run", side_effect=run
            ) as child:
                if returncode:
                    with self.assertRaises(subprocess.CalledProcessError) as error:
                        BUILD.call("configure", command)
                    self.assertEqual(error.exception.returncode, returncode)
                else:
                    BUILD.call("configure", command)
                child.assert_called_once()

            self.assertEqual(
                json.loads((root / "metadata/configure-argv.json").read_text()),
                expected,
            )
            self.assertEqual(
                json.loads((root / "metadata/configure-exit.json").read_text()),
                {"returncode": returncode},
            )
            self.assertTrue((root / "logs/configure.log").is_file())

    def test_mixed_path_arguments_are_logged_and_executed_identically(self):
        self.exercise(0)

    def test_child_failure_preserves_arguments_and_exit_code(self):
        self.exercise(7)


if __name__ == "__main__":
    unittest.main()
