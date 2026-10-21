"""Offline regressions for post-science failures in fresh grid-sweep workers."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("sweep", Path(__file__).with_name("saptdft_grid_sweep.py"))
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class GridAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        (self.directory / "result.json").write_text('{"status":"ok","science_completed":true}')
        self.complete = SimpleNamespace(returncode=0, stderr="")
        self.receipt = {"status": "passed", "payload_exit": 0, "native_version": [0, 2, 50],
                        "before": {"pinned": True}, "after": {"pinned": True}}
        self.identity = self.directory / "identity.json"
        self.identity.write_text('{"pinned":true}')
        self.addCleanup(patch.stopall)
        patch.dict(S.os.environ, {"CUEST_CORE_IDENTITY": str(self.identity)}).start()
        self.write()

    def write(self):
        (self.directory / "runtime-admission.json").write_text(json.dumps(self.receipt))

    def result(self, gpu=True):
        return S.worker_result(self.complete, self.directory, (99, 590), None, gpu)

    def test_valid_science_and_admission(self):
        self.assertEqual(self.result()["status"], "ok")

    def test_nonzero_worker_rejects_previously_successful_science(self):
        self.complete.returncode = 1
        self.assertEqual(self.result()["status"], "failed")

    def test_failed_post_science_receipt_cannot_pass(self):
        self.receipt["status"] = "failed"
        self.write()
        self.assertEqual(self.result()["status"], "failed")

    def test_missing_receipt_rejected(self):
        (self.directory / "runtime-admission.json").unlink()
        self.assertEqual(self.result()["status"], "failed")

    def test_wrong_native_version_rejected(self):
        self.receipt["native_version"] = [0, 2, 2]
        self.write()
        self.assertEqual(self.result()["status"], "failed")

    def test_bool_version_rejected(self):
        self.receipt["native_version"] = [False, 2, 50]
        self.write()
        self.assertEqual(self.result()["status"], "failed")

    def test_changed_post_maps_rejected(self):
        self.receipt["after"] = {"pinned": False}
        self.write()
        self.assertEqual(self.result()["status"], "failed")

    def test_missing_result_rejected(self):
        (self.directory / "result.json").unlink()
        self.assertEqual(self.result()["status"], "failed")

    def test_cpu_worker_requires_no_gpu_receipt(self):
        (self.directory / "runtime-admission.json").unlink()
        self.assertEqual(self.result(gpu=False)["status"], "ok")

    def test_bool_payload_exit_rejected(self):
        self.receipt["payload_exit"] = False
        self.write()
        self.assertEqual(self.result()["status"], "failed")

    def test_self_consistent_but_unpinned_receipt_rejected(self):
        self.receipt["before"] = self.receipt["after"] = {"unrelated": True}
        self.write()
        self.assertEqual(self.result()["status"], "failed")


if __name__ == "__main__":
    unittest.main()
