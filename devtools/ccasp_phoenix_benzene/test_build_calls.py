# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Stdlib-only regression tests for build command recording, with no child execution."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
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

    def test_python_prefix_exposes_only_exact_packages_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "old"
            destination = Path(tmp) / "isolated"
            for package, version in BUILD.PLAN["python_dependencies"].items():
                module = source / "lib" / package
                module.mkdir(parents=True)
                (module / "__init__.py").write_text("raise RuntimeError('must not import')\n")
                info = source / "lib" / f"{package}-{version}.dist-info"
                info.mkdir()
                (info / "METADATA").write_text(f"Name: {package}\nVersion: {version}\n")
            (source / "lib/psi4").mkdir()
            links = BUILD.stage_python_dependencies(source, destination)
            expected = {
                name
                for package, version in BUILD.PLAN["python_dependencies"].items()
                for name in (package, f"{package}-{version}.dist-info")
            }
            self.assertEqual({p.name for p in destination.iterdir()}, expected)
            self.assertEqual(len(links), 6)
            self.assertFalse((destination / "psi4").exists())
            self.assertTrue(all(Path(dst).resolve() == Path(src) for dst, src in links.items()))

    def test_python_prefix_refuses_wrong_metadata_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "old"
            (source / "lib/qcelemental").mkdir(parents=True)
            info = source / "lib/qcelemental-0.50.4.dist-info"
            info.mkdir()
            (info / "METADATA").write_text("Name: qcelemental\nVersion: 0.30.1\n")
            with self.assertRaisesRegex(ValueError, "qcelemental"):
                BUILD.stage_python_dependencies(source, Path(tmp) / "isolated")

    def test_dependency_import_gate_refuses_non_compute_execution(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
            with self.assertRaisesRegex(ValueError, "compute allocation only"):
                runpy.run_path(
                    str(Path(__file__).with_name("python_dependency_check.py")),
                    run_name="__main__",
                )


if __name__ == "__main__":
    unittest.main()
