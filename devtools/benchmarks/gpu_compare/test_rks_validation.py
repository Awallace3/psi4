"""Stdlib-only checks that comparison prerequisites are explicitly RKS-only."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

SPEC = importlib.util.spec_from_file_location(
    "direct_validation", Path(__file__).parent.parent / "cuest_direct/validate.py")
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class RKSValidationTests(unittest.TestCase):
    def cases(self):
        return [ET.Element("testcase", name=name) for name in validation.RKS_CASES]

    def test_rks_targets_exclude_uks_and_df(self):
        targets = validation.test_targets(Path("/repo"), "rks")
        self.assertEqual(len(targets), 9)
        self.assertTrue(all("test_cuest_direct_jk.py::" in t for t in targets))
        self.assertFalse(any("uks" in t or "/test_cuest_jk.py" in t for t in targets))
        self.assertIn("/repo/tests/pytests/test_cuest_direct_jk.py::test_cuest_direct_with_gpu_xc",
                      targets)
        self.assertEqual(len(validation.test_targets(Path("/repo"), "full")), 2)

    def test_rks_requires_exact_complete_case_set(self):
        cases = self.cases()
        self.assertTrue(validation.validation_success(0, cases, "rks"))
        self.assertFalse(validation.validation_success(0, cases[:-1], "rks"))
        cases[-1] = cases[0]
        self.assertFalse(validation.validation_success(0, cases, "rks"))

    def test_rks_failure_error_skip_and_exit_are_fatal(self):
        for tag in ("failure", "error", "skipped"):
            cases = self.cases()
            ET.SubElement(cases[0], tag)
            self.assertFalse(validation.validation_success(0, cases, "rks"))
        self.assertFalse(validation.validation_success(1, self.cases(), "rks"))
        self.assertFalse(validation.validation_success(
            0, self.cases() + [ET.Element("testcase", name="test_cuest_direct_uks_energy")], "rks"))


if __name__ == "__main__":
    unittest.main()
