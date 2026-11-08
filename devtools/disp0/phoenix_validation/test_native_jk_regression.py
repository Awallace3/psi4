"""Offline launcher contracts; these mocks are not native/GPU evidence."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from native_jk_regression import EXPECTED, REQUIRED, junit_summary
from scf_repeat_probe import schedule


class NativeRegressionContracts(unittest.TestCase):
    def xml(self, path, names=None, child=None, **attributes):
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite", **attributes)
        for name in sorted(EXPECTED) if names is None else names:
            case = ET.SubElement(suite, "testcase", name=name)
            if child and name == REQUIRED:
                ET.SubElement(case, child)
        ET.ElementTree(root).write(path)

    def test_exact_six_tests_with_cutoff_regression_and_no_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"junit.xml"
            self.xml(path)
            self.assertEqual(junit_summary(path), {
                "tests":6, "skipped":0, "failures":0, "errors":0,
                "required_test_executed":True})

    def test_deselected_duplicate_missing_or_unknown_tests_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"junit.xml"
            for names in (
                sorted(EXPECTED-{REQUIRED}), [REQUIRED]*6,
                sorted(EXPECTED)+["unknown"], sorted(EXPECTED-{
                    "test_cuest_jk_symmetric_exchange_matches_cpu"})+["unknown"],
            ):
                self.xml(path, names)
                with self.assertRaises(RuntimeError):
                    junit_summary(path)

    def test_case_and_suite_failures_errors_or_skips_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"junit.xml"
            for child in ("skipped", "failure", "error"):
                self.xml(path, child=child)
                with self.assertRaises(RuntimeError):
                    junit_summary(path)
            for attribute in ("skipped", "failures", "errors"):
                self.xml(path, **{attribute:"1"})
                with self.assertRaises(RuntimeError):
                    junit_summary(path)

    def test_common_schedule_preserves_embers_h200_and_frozen_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"kind":"gpu", "mode":"native-tests", "outer_pin":"a"*40}
            (root/"manifest.json").write_text(json.dumps(manifest))
            config = {"dependency_environment":"/frozen/env", "threads":8}
            completed = SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
            with patch("scf_repeat_probe.subprocess.run", return_value=completed) as run, \
                    patch.dict(os.environ, {"SBATCH_QOS":"inferno", "SLURM_JOB_ID":"old"}):
                schedule(root, manifest, config, Path("/frozen/outer"))
            self.assertEqual(run.call_count, 2)
            trial, actual = run.call_args_list
            self.assertIn("--test-only", trial.args[0])
            self.assertNotIn("--test-only", actual.args[0])
            for flag in ("--qos=embers", "--constraint=H200", "--gres=gpu:h200:1",
                         "--cpus-per-task=8", "--time=00:30:00", "--signal=TERM@120"):
                self.assertIn(flag, actual.args[0])
            env = actual.kwargs["env"]
            self.assertNotIn("SBATCH_QOS", env)
            self.assertNotIn("SLURM_JOB_ID", env)
            self.assertEqual(env["DISP0_EXECUTE_DRIVER"], "scf_repeat_probe.py")
            self.assertEqual((root/"job-id.txt").read_text(), "12345\n")


if __name__ == "__main__":
    unittest.main()
