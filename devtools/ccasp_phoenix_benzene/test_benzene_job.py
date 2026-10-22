# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Stdlib-only checks; no scientific imports or calculations."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "benzene_job", Path(__file__).with_name("benzene_job.py"))
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)


class BenzeneJobTests(unittest.TestCase):
    def mapped(self):
        return {
            str(JOB.OLD / "lib/libEinsums.so.1.1.3"): "recorded",
            str(JOB.ENV / "lib/libomp.so"): "recorded",
            str(JOB.ENV / "lib/libmkl_gnu_thread.so.2"): "recorded",
        }

    def test_reviewed_runtime_role_is_accepted(self):
        JOB.validate_libraries(self.mapped())

    def test_known_wrong_einsums_is_rejected(self):
        loaded = self.mapped()
        loaded[str(JOB.ENV / "lib/libEinsums.so.1.1.2")] = "known but rejected"
        with self.assertRaisesRegex(ValueError, "Einsums"):
            JOB.validate_libraries(loaded)

    def test_extra_openmp_or_intel_backend_is_rejected(self):
        for extra in ("libgomp.so", "libmkl_intel_thread.so.2"):
            with self.subTest(extra=extra):
                loaded = self.mapped()
                loaded[str(JOB.ENV / "lib" / extra)] = "rejected"
                with self.assertRaises(ValueError):
                    JOB.validate_libraries(loaded)

    def test_outside_environment_library_is_rejected(self):
        loaded = self.mapped()
        loaded["/other/environment/lib/libmkl_rt.so.2"] = "rejected"
        with self.assertRaisesRegex(ValueError, "outside"):
            JOB.validate_libraries(loaded)

    def test_driver_uses_accepted_build_and_fresh_output(self):
        argv = JOB.pilot_arguments("12345")
        options = dict(zip(argv[1::2], argv[2::2]))
        self.assertEqual(options["--commit"], JOB.BUILD_COMMIT)
        self.assertEqual(options["--core-sha256"], JOB.CORE_SHA)
        self.assertEqual(options["--stage"], str(JOB.STAGE))
        self.assertEqual(options["--out"], str(JOB.RUN / "results"))
        self.assertEqual(options["--scratch"], str(JOB.SCRATCH / "P-12345"))

    def test_driver_preflight_exits_produce_nonzero_diagnostic_receipts(self):
        for message in ("preflight refusal", 1, 0):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "metadata").mkdir()
                pilot = SimpleNamespace(main=mock.Mock(side_effect=SystemExit(message)))
                stderr = io.StringIO()
                with mock.patch.object(JOB, "RUN", root), \
                        mock.patch.object(JOB, "run",
                                          side_effect=lambda *_: JOB.invoke_pilot(pilot)), \
                        mock.patch.object(sys, "argv",
                                          ["job", "run", "--commit", "test", "--tree", "test"]), \
                        contextlib.redirect_stderr(stderr):
                    self.assertEqual(JOB.main(), 1)
                record = json.loads((root / "metadata/bootstrap-failure.json").read_text())
                self.assertIn("pilot preflight exit", record["traceback"])
                self.assertIn(str(message), stderr.getvalue())
                self.assertFalse((root / "results").exists())

    def test_zero_without_scientific_completion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without scientific completion"):
            JOB.check_driver_result(0, {})
        JOB.check_driver_result(0, {"science_pass": True})
        JOB.check_driver_result(1, {"science_pass": False})


if __name__ == "__main__":
    unittest.main()
