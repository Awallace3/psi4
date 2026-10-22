"""Gate-only acceptance must never imply benchmark completion."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "gate_recovery", ROOT / "devtools/benchmarks/cuest_direct/recovery.py")
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class GateOnly(unittest.TestCase):
    def test_valid_preflight_is_separate_from_panel_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preflight").mkdir()
            (root / "preflight/COMPLETED.json").write_text('{"complete": true}')
            RECOVERY.record_gate_completion(root, "native")
            r = json.loads((root / "GATES_COMPLETE.json").read_text())
            self.assertIs(r["complete"], True)
            self.assertIs(r["benchmarks_run"], False)
            self.assertFalse((root / "COMPLETE.json").exists())
            self.assertFalse((root / "panel").exists())

    def test_failed_missing_or_invalid_solver_never_accepted(self):
        for content in ({}, {"complete": False}, {"complete": "true"}):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "preflight").mkdir()
                (root / "preflight/COMPLETED.json").write_text(json.dumps(content))
                with self.assertRaisesRegex(RuntimeError, "incomplete GPU gates"):
                    RECOVERY.record_gate_completion(root, "native")
                self.assertFalse((root / "GATES_COMPLETE.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preflight").mkdir()
            (root / "preflight/COMPLETED.json").write_text('{"complete": true}')
            with self.assertRaisesRegex(RuntimeError, "incomplete GPU gates"):
                RECOVERY.record_gate_completion(root, "cpu")

    def test_explicit_alternative_stays_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preflight").mkdir()
            (root / "preflight/COMPLETED.json").write_text('{"complete": true}')
            RECOVERY.record_gate_completion(root, "cupy-canonical")
            r = json.loads((root / "GATES_COMPLETE.json").read_text())
            self.assertIs(r["unmodified_gpu4pyscf"], False)


if __name__ == "__main__":
    unittest.main()
