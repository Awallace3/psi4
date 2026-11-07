import json
from pathlib import Path
import tempfile
import unittest

from qualify_stage_scaling import completed, worker
from stage_scaling import SCIENCE_PIN, digest


class QualificationTests(unittest.TestCase):
    def test_allocation_exit_cleanup_and_retirement_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = "PASS"
            d = {"status": status, "source_pin": SCIENCE_PIN,
                 "scratch_cleanup": {"complete": True},
                 "processes": [{"returncode": 0, "owned_group_reaped": True,
                                "retirement_reason": None}]}
            (root/"job-status.env").write_text("exit_code=0\nterm_received=0\n")
            (root/"stage-result.json").write_text(json.dumps(d))
            self.assertEqual(completed(root, status), d)
            d["processes"][0]["retirement_reason"] = {"reason": "late retirement"}
            (root/"stage-result.json").write_text(json.dumps(d))
            with self.assertRaises(RuntimeError):
                completed(root, status)
            (root/"job-status.env").write_text("exit_code=143\nterm_received=1\n")
            with self.assertRaises(RuntimeError):
                completed(root, status)

    def test_worker_requires_final_result_timer_stdout_and_array_seals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = {"status": "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                 "source_pin": SCIENCE_PIN, "benchmark": True,
                 "benchmark_whole_energy_seconds": 3.,
                 "artifact_sha256": {}, "output_file_sha256": {}}
            (root/"result.json").write_text(json.dumps(d))
            (root/"timer.dat").write_text("timer")
            (root/"psi4.out").write_text("out")
            row = {"path": str(root), "energy_seconds": 3.,
                   "result_sha256": digest(root/"result.json"),
                   "timer_sha256": digest(root/"timer.dat"),
                   "psi4_out_sha256": digest(root/"psi4.out")}
            self.assertEqual(worker(row, root), d)
            (root/"psi4.out").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                worker(row, root)


if __name__ == "__main__":
    unittest.main()
