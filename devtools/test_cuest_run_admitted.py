"""Offline worker sequencing tests; no CUDA, Psi4 import, SDK or remote calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys

SPEC = importlib.util.spec_from_file_location(
    "launcher", Path(__file__).with_name("cuest_run_admitted.py"))
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


class Worker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = self.root / "identity.json"
        self.identity.write_text("{}")
        self.receipt = self.root / "runtime.json"
        self.events = []
        self.core = SimpleNamespace(__file__="/pinned/core.so", __name__=__name__,
                                    _cuest_runtime_version=self.version)
        self.contract = SimpleNamespace(admit_runtime=self.admit)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"psi4": SimpleNamespace(core=self.core), "numpy": SimpleNamespace()}).start()
        patch.object(W.importlib.util, "module_from_spec", return_value=self.contract).start()
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        patch.object(W.importlib.util, "spec_from_file_location", return_value=spec).start()
        self.payload = patch.object(W.runpy, "run_module", side_effect=self.execute).start()
        patch.object(W.inspect, "isbuiltin", return_value=True).start()

    def version(self):
        self.events.append("native-after-payload")
        return (0, 2, 50)

    def admit(self, *args):
        self.events.append("maps")
        return {"checked": True}

    def execute(self, *args, **kwargs):
        self.events.append("payload")

    def run_worker(self):
        return W.run("pytest", ["-x"], self.receipt, self.identity, "contract.py")

    def test_sequence_never_initializes_hidden_handle(self):
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.events, ["maps", "payload", "native-after-payload", "maps"])
        self.assertEqual(json.loads(self.receipt.read_text())["status"], "passed")

    def test_absent_old_core_provider_rejected_before_payload(self):
        del self.core._cuest_runtime_version
        self.assertEqual(self.run_worker(), 1)
        self.payload.assert_not_called()

    def test_wrong_native_version_rejected(self):
        self.core._cuest_runtime_version = lambda: (0, 2, 2)
        self.assertEqual(self.run_worker(), 1)
        self.assertIn("native cuEST", json.loads(self.receipt.read_text())["admission_error"])

    def test_list_receipt_not_accepted_as_tuple(self):
        self.core._cuest_runtime_version = lambda: [0, 2, 50]
        self.assertEqual(self.run_worker(), 1)

    def test_bool_is_not_version_integer(self):
        self.core._cuest_runtime_version = lambda: (False, 2, 50)
        self.assertEqual(self.run_worker(), 1)

    def test_vendor_status_error_rejected(self):
        self.core._cuest_runtime_version = Mock(side_effect=RuntimeError("vendor status"), __module__=__name__)
        self.assertEqual(self.run_worker(), 1)
        self.assertIn("vendor status", self.receipt.read_text())

    def test_null_handle_after_payload_rejected(self):
        self.core._cuest_runtime_version = Mock(side_effect=RuntimeError("not initialized"), __module__=__name__)
        self.assertEqual(self.run_worker(), 1)

    def test_payload_failure_not_converted_to_pass(self):
        self.payload.side_effect = SystemExit(2)
        self.assertEqual(self.run_worker(), 2)
        self.assertEqual(json.loads(self.receipt.read_text())["status"], "failed")

    def test_payload_failure_survives_second_admission_error(self):
        self.payload.side_effect = SystemExit(3)
        self.core._cuest_runtime_version = Mock(side_effect=RuntimeError("not initialized"), __module__=__name__)
        self.assertEqual(self.run_worker(), 3)

    def test_map_mismatch_rejected_before_payload(self):
        self.contract.admit_runtime = Mock(side_effect=ValueError("map mismatch"))
        self.assertEqual(self.run_worker(), 1)
        self.payload.assert_not_called()

    def test_map_mismatch_rejected_after_payload(self):
        self.contract.admit_runtime = Mock(side_effect=[{}, ValueError("map changed")])
        self.assertEqual(self.run_worker(), 1)
        self.assertIn("map changed", self.receipt.read_text())

    def test_no_receipt_overwrite(self):
        self.receipt.write_text("previous")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.run_worker()
        self.assertEqual(self.receipt.read_text(), "previous")

    def test_python_substitute_provider_rejected_before_payload(self):
        with patch.object(W.inspect, "isbuiltin", return_value=False):
            self.assertEqual(self.run_worker(), 1)
        self.payload.assert_not_called()

    def test_script_payload_admitted_in_same_process(self):
        with patch.object(W.runpy, "run_path", side_effect=self.execute) as script:
            status = W.run("driver.py", [], self.receipt, self.identity, "contract.py", script=True)
        self.assertEqual(status, 0)
        script.assert_called_once_with("driver.py", run_name="__main__")
        self.payload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
