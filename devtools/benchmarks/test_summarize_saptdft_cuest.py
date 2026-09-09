"""CPU-only unit tests for benchmark reporting (no Psi4 import required)."""
import json
from pathlib import Path
import tempfile
import unittest

from summarize_saptdft_cuest import summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        entries = []
        for repeat in range(1, 4):
            for mode, wall in (("cpu", 10.0 + repeat), ("gpu", 2.0 + repeat)):
                name = f"water-cc-pvdz-{mode}-{repeat}"
                directory = self.root / name
                directory.mkdir()
                result = {"ok": True, "system": "water", "basis": "cc-pvdz", "mode": mode,
                          "wall_s": wall, "nbf": 48, "threads": 8,
                          "geometry": "same", "psi4_version": "test",
                          "options": {"USE_CUEST": mode == "gpu", "shift": 0.136},
                          "components_hartree": {"SAPT TOTAL ENERGY": -0.01 + (1e-8 if mode == "gpu" else 0)}}
                (directory / "result.json").write_text(json.dumps(result))
                entries.append({"name": name, "returncode": 0})
        (self.root / "campaign.json").write_text(json.dumps({"repeats": 3, "records": entries}))
        (self.root / "COMPLETE.json").write_text('{"ok": true, "count": 6}')

    def test_median_speedup_and_paired_accuracy(self):
        result = summarize(self.root)
        self.assertTrue(result["all_pass"])
        row = result["rows"][0]
        self.assertEqual(row["speedup"], 3.0)
        self.assertEqual(row["paired_repeats"], 3)
        self.assertAlmostEqual(row["max_abs_delta_hartree"], 1e-8)

    def test_incomplete_not_pass(self):
        (self.root / "COMPLETE.json").unlink()
        self.assertFalse(summarize(self.root)["all_pass"])

    def test_failed_completion_marker_not_pass(self):
        (self.root / "COMPLETE.json").write_text('{"ok": false, "count": 6}')
        self.assertFalse(summarize(self.root)["all_pass"])

    def test_missing_planned_cases_not_pass(self):
        path = self.root / "campaign.json"
        campaign = json.loads(path.read_text())
        campaign["cases"] = [["water", ["cc-pvdz", "aug-cc-pvdz"]]]
        path.write_text(json.dumps(campaign))
        self.assertFalse(summarize(self.root)["all_pass"])

    def test_missing_record_not_hidden(self):
        (self.root / "water-cc-pvdz-gpu-2" / "result.json").unlink()
        result = summarize(self.root)
        self.assertFalse(result["all_pass"])
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(result["rows"][0]["paired_repeats"], 2)

    def test_component_failure(self):
        path = self.root / "water-cc-pvdz-gpu-2" / "result.json"
        record = json.loads(path.read_text())
        record["components_hartree"]["SAPT TOTAL ENERGY"] += 1e-4
        path.write_text(json.dumps(record))
        result = summarize(self.root)
        self.assertFalse(result["all_pass"])
        self.assertFalse(result["rows"][0]["accuracy_pass"])

    def test_mismatched_options_rejected(self):
        path = self.root / "water-cc-pvdz-gpu-2" / "result.json"
        record = json.loads(path.read_text())
        record["options"]["shift"] = 0.0
        path.write_text(json.dumps(record))
        with self.assertRaises(AssertionError):
            summarize(self.root)


if __name__ == "__main__":
    unittest.main()
