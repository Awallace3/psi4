import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

spec = importlib.util.spec_from_file_location("recovery", Path(__file__).with_name("recovery.py"))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def test_probe_gate_rejects_missing_wrong_mode_or_partial_bases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            self.assertFalse(recovery.probe_passed(path, "native"))
            r = dict(passed=True, config=dict(pyscf_eigensolver="native"),
                     basis_cases=[dict(basis=b, passed=True)
                                  for b in ("cc-pvtz", "aug-cc-pvtz")])
            path.write_text(json.dumps(r))
            self.assertTrue(recovery.probe_passed(path, "native"))
            self.assertFalse(recovery.probe_passed(path, "cupy-canonical"))
            r["basis_cases"].pop()
            path.write_text(json.dumps(r))
            self.assertFalse(recovery.probe_passed(path, "native"))
            path.write_text("{}")
            self.assertFalse(recovery.probe_passed(path, "native"))


if __name__ == "__main__":
    unittest.main()
