from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

import tz_control


class RecoveryGateTests(unittest.TestCase):
    def test_energy_audit_refuses_nonfinite_missing_and_disagreement(self):
        c = dict(energy_review_threshold_eh=1e-4)
        pair = [dict(engine=e, ok=True, energy_hartree=v)
                for e, v in (("psi4", -10.), ("pyscf", -10.000001))]
        self.assertTrue(tz_control.energy_audit(pair, c)["verified"])
        pair[1]["energy_hartree"] = -11.
        self.assertFalse(tz_control.energy_audit(pair, c)["verified"])
        pair[1]["energy_hartree"] = float("nan")
        self.assertFalse(tz_control.energy_audit(pair, c)["verified"])
        self.assertFalse(tz_control.energy_audit(pair[:1], c)["verified"])

    def test_strict_gate_refuses_failure_or_missing_cleanup(self):
        for r in ({}, dict(ok=True), dict(ok=False, worker_cleanup=dict(verified=True))):
            with self.assertRaisesRegex(RuntimeError, "subsequent workers withheld"):
                tz_control.require_worker_success(r, True, "test")
        tz_control.require_worker_success(dict(ok=True, worker_cleanup=dict(verified=True)),
                                         True, "test")

    def test_cleanup_rejects_live_detached_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "x.usage.json").write_text(json.dumps(
                dict(state="EXITED", child_returncode=0)))
            with patch("containment.tree", return_value={123}), \
                    patch.object(tz_control.subprocess, "check_output", return_value=""):
                proof = tz_control.worker_cleanup(base, "x", "GPU-test")
            self.assertFalse(proof["verified"])
            self.assertEqual(proof["descendants"], [123])
