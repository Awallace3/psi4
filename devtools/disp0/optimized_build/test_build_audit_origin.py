"""Build-metadata regression only: no compilation or library import."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("audit_origin_fixture", HERE.parent / "build_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OriginAudit(unittest.TestCase):
    def test_actual_escaped_ninja_origin_is_literal_not_an_environment_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit.safe_path(r"\$ORIGIN/../", root, [root])
            audit.audit_tokens([str(root / "never-executed-c++"), r"-Wl,-rpath,\$ORIGIN/../"], root, [root])

    def test_only_exact_origin_forms_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for value in ("$ORIGIN", "${ORIGIN}/..", r"\${ORIGIN}/../lib"):
                audit.safe_path(value, root, [root])
            for value in ("$ORIGIN_OTHER/lib", "${ORIGIN_OTHER}/lib", "$HOME/lib",
                          r"\$HOME/lib", "$ORIGIN/$OTHER", "$ORIGIN/../../cuda-12.6/lib"):
                with self.assertRaises(RuntimeError):
                    audit.safe_path(value, root, [root])

    def test_external_new_audit_output_preserves_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "failed-run"
            previous = run / "effective-build-audit"
            previous.mkdir(parents=True)
            marker = previous / "original.txt"
            marker.write_bytes(b"ORIGINAL_FAILURE_EVIDENCE")
            destination = root / "new-validation-step"
            self.assertEqual(audit.audit_output_directory(run, destination), destination)
            self.assertEqual(marker.read_bytes(), b"ORIGINAL_FAILURE_EVIDENCE")
            with self.assertRaises(FileExistsError):
                audit.audit_output_directory(run, destination)
            with self.assertRaises(RuntimeError):
                audit.audit_output_directory(run, run / "new-output")


if __name__ == "__main__":
    unittest.main()
