"""Fast harness-only guards; no Psi4/CUDA import or scientific calculations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_receipt
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


if __name__ == "__main__":
    unittest.main()
