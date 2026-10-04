"""Fast harness-only guards; no Psi4/CUDA import or scientific calculations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_receipt
from lifecycle_campaign import compare_arms, scratch_status
from saptdft_cuest_grac import atomic_json


class TestReceipt(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.package = self.source / "build/stage/lib/psi4"
        self.package.mkdir(parents=True)
        (self.package / "core.test.so").write_bytes(b"test binary")
        (self.package.parents[2] / "CMakeCache.txt").write_text("ENABLE_cuEST=ON")
        (self.package / "metadata.py").write_text(
            "__version_long = '1.0+abcdef0'\n__version_is_clean = 'True'\n")
        self.mock = patch.object(build_receipt, "git", return_value="tree")
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def receipt(self):
        return build_receipt.capture(self.package, self.source, "abcdef0123456789")

    def test_unchanged_artifact_passes(self):
        receipt = self.receipt()
        self.assertEqual(build_receipt.verify(receipt), receipt)

    def test_modified_binary_fails(self):
        receipt = self.receipt()
        (self.package / "core.test.so").write_bytes(b"rebuilt")
        with self.assertRaises(AssertionError):
            build_receipt.verify(receipt)

    def test_wrong_source_version_fails(self):
        with self.assertRaises(AssertionError):
            build_receipt.capture(self.package, self.source, "9999999999999999")

    def test_nonfinite_json_rejected(self):
        target = Path(self.temp.name) / "result.json"
        with self.assertRaises(ValueError):
            atomic_json(target, {"shift_B": float("nan")})
        self.assertFalse(target.exists())
        atomic_json(target, {"shift_B": 0.1})
        self.assertEqual(json.loads(target.read_text()), {"shift_B": 0.1})

    def test_missing_scratch_is_recorded(self):
        missing = Path(self.temp.name) / "deleted-scratch"
        self.assertFalse(scratch_status(missing)["exists"])
        self.assertIn("error", scratch_status(missing))
        existing = scratch_status(Path(self.temp.name))
        self.assertTrue(existing["exists"])
        self.assertGreater(existing["free_bytes"], 0)


class TestAccuracy(unittest.TestCase):
    def paired(self):
        def result(shift):
            return {"components_hartree": {"SAPT TOTAL ENERGY": -0.01},
                    "grac_shifts_hartree": {"A": shift, "B": shift}}
        return {arm: result(0.075 if "gpu" in arm else 0.07512) for arm in
                ("old-cpu", "old-gpu", "new-cpu", "new-gpu-cpu-sad", "new-gpu-gpu-sad")}

    def test_baseline_gap_remains_visible(self):
        checks = compare_arms(self.paired())
        self.assertFalse(checks[0]["within_tolerance"])
        self.assertEqual(checks[0]["kind"], "baseline-cross-backend")
        self.assertTrue(all(x["within_tolerance"] for x in checks[1:]))

    def test_new_gpu_shift_regression_fails(self):
        paired = self.paired()
        paired["new-gpu-gpu-sad"]["grac_shifts_hartree"]["B"] += 2.e-6
        check = compare_arms(paired)[-1]
        self.assertEqual(check["kind"], "regression")
        self.assertFalse(check["within_tolerance"])

    def test_new_cpu_component_regression_fails(self):
        paired = self.paired()
        paired["new-cpu"]["components_hartree"]["SAPT TOTAL ENERGY"] += 2.e-6
        self.assertFalse(compare_arms(paired)[1]["within_tolerance"])

    def test_nan_shift_b_cannot_hide_behind_finite_a(self):
        paired = self.paired()
        paired["new-gpu-gpu-sad"]["grac_shifts_hartree"]["B"] = float("nan")
        with self.assertRaises(ValueError):
            compare_arms(paired)


if __name__ == "__main__":
    unittest.main()
