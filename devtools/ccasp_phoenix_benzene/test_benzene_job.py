# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Stdlib-only checks; no scientific imports or calculations."""
import importlib.util
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
