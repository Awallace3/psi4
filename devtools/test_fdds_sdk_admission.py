"""Offline current benchmark admission tests; historical runner pins unchanged."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).parent / "benchmarks/fdds_benchmark.py")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class BenchmarkAdmission(unittest.TestCase):
    def setUp(self):
        self.psi = SimpleNamespace(core=SimpleNamespace(__file__="/pinned/core.so", cuest_chain_gemm=True))
        self.bridge = SimpleNamespace(check_provider=Mock(), native_receipt=Mock(return_value=(0, 2, 50)))
        self.contract = SimpleNamespace(admit_runtime=Mock(return_value={"checked": True}))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.identity = Path(self.tmp.name) / "core.json"
        self.identity.write_text(json.dumps({"pinned": True}))
        self.addCleanup(patch.stopall)
        patch.dict(B.os.environ, {"CUEST_CORE_IDENTITY": str(self.identity), "CUEST_CONTRACT": "/contract.py"}).start()
        patch.object(B.importlib.util, "module_from_spec", side_effect=[self.bridge, self.contract]).start()
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        patch.object(B.importlib.util, "spec_from_file_location", return_value=spec).start()

    def test_gpu_admits_before_and_after_native_route(self):
        before = B.sdk_admit_before(self.psi, "gpu")
        self.bridge.native_receipt.assert_not_called()
        after = B.sdk_admit_after(self.psi, "gpu", before)
        self.assertEqual(after["native_version"], (0, 2, 50))
        self.assertEqual(self.contract.admit_runtime.call_count, 2)

    def test_gpu_capable_cpu_control_does_not_initialize(self):
        before = B.sdk_admit_before(self.psi, "cpu")
        after = B.sdk_admit_after(self.psi, "cpu", before)
        self.assertIsNone(after["native_version"])
        self.bridge.native_receipt.assert_not_called()
        self.assertEqual(self.contract.admit_runtime.call_count, 2)

    def test_pure_cpu_build_requires_no_sdk(self):
        del self.psi.core.cuest_chain_gemm
        self.assertIsNone(B.sdk_admit_before(self.psi, "cpu"))
        self.assertIsNone(B.sdk_admit_after(self.psi, "cpu", None))
        self.bridge.check_provider.assert_not_called()

    def test_old_gpu_core_rejected_before_work(self):
        self.bridge.check_provider.side_effect = ValueError("provider absent")
        with self.assertRaisesRegex(ValueError, "provider absent"):
            B.sdk_admit_before(self.psi, "gpu")
        self.contract.admit_runtime.assert_not_called()

    def test_wrong_runtime_maps_rejected(self):
        self.contract.admit_runtime.side_effect = ValueError("wrong maps")
        with self.assertRaisesRegex(ValueError, "wrong maps"):
            B.sdk_admit_before(self.psi, "gpu")

    def test_failed_native_receipt_rejected(self):
        before = B.sdk_admit_before(self.psi, "gpu")
        self.bridge.native_receipt.side_effect = ValueError("wrong native version")
        with self.assertRaisesRegex(ValueError, "wrong native version"):
            B.sdk_admit_after(self.psi, "gpu", before)


if __name__ == "__main__":
    unittest.main()
