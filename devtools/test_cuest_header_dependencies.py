"""Offline actual-header consumption guards, including a real shadow-header probe."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cuest_sdk_contract", ROOT / "cmake/cuest_sdk_contract.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
with patch.dict(sys.modules, {"cuest_sdk_contract": C}):
    SPEC = importlib.util.spec_from_file_location("deps", ROOT / "cmake/cuest_header_dependencies.py")
    D = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(D)


class Headers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sdk = self.root / "sdk"
        self.include = self.sdk / "include"
        self.include.mkdir(parents=True)
        self.umbrella = self.include / "cuest.h"
        self.umbrella.write_text("#define ADMITTED_MARK 1\n")
        self.addCleanup(patch.stopall)
        patch.object(D, "HEADERS_SHA256", C.header_digest(self.include)).start()

    def test_canonical_consumed_header(self):
        self.assertEqual(D.admitted_headers([self.umbrella], self.sdk),
                         {str(self.umbrella): C.sha256(self.umbrella)})

    def test_missing_umbrella_proof_rejected(self):
        with self.assertRaisesRegex(ValueError, "no consumed"):
            D.admitted_headers([], self.sdk)

    def test_shadow_umbrella_rejected(self):
        old = self.root / "old/cuest.h"
        old.parent.mkdir()
        old.write_bytes(b"shadow")
        with self.assertRaisesRegex(ValueError, "unadmitted"):
            D.admitted_headers([old], self.sdk)

    def test_shadow_subheader_rejected(self):
        old = self.root / "old/cuest/context/context_api.h"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"shadow")
        with self.assertRaisesRegex(ValueError, "unadmitted"):
            D.admitted_headers([self.umbrella, old], self.sdk)

    def test_modified_canonical_header_rejected(self):
        self.umbrella.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed during compilation"):
            D.admitted_headers([self.umbrella], self.sdk)

    def test_real_preprocessor_exposes_earlier_shadow(self):
        old = self.root / "old"
        old.mkdir()
        shadow = old / "cuest.h"
        shadow.write_text("#define ADMITTED_MARK 2\n")
        code = self.root / "probe.c"
        code.write_text("#include <cuest.h>\nint x=ADMITTED_MARK;\n")
        result = subprocess.run(["cc", "-M", "-I", str(old), "-I", str(self.include), str(code)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = result.stdout.replace("\\\n", " ").split()[1:]
        self.assertIn(str(shadow), paths)
        with self.assertRaisesRegex(ValueError, "unadmitted"):
            D.admitted_headers(paths, self.sdk)

    def test_unsupported_generator_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported generator"):
            D.dependencies(self.root, "unknown")


if __name__ == "__main__":
    unittest.main()
