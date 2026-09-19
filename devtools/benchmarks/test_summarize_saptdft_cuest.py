"""CPU-only unit tests for benchmark reporting (no Psi4 import required)."""
import json
from pathlib import Path
import tempfile
import unittest

from summarize_saptdft_cuest import markdown, summarize


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
                          "components_hartree": {"SAPT TOTAL ENERGY": -0.01 + (1e-8 if mode == "gpu" else 0)},
                          "host_memory": {"peak_rss_mib": (2000.0 if mode == "cpu" else 1400.0) + repeat,
                                          "rss_before_mib": 300.0, "rss_after_mib": 320.0,
                                          "peak_covers_timed_region_only": True}}
                if mode == "gpu":
                    result["device_memory"] = {"peak_mib": 5000.0 + repeat, "source": "per-process",
                                               "sample_interval_s": 0.5, "samples": 40}
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

    def test_memory_medians_reported_for_both_arms(self):
        memory = summarize(self.root)["rows"][0]["memory"]
        self.assertEqual(memory["host_peak_rss_mib"]["cpu"]["peak_rss_mib"]["median"], 2002.0)
        self.assertEqual(memory["host_peak_rss_mib"]["gpu"]["peak_rss_mib"]["median"], 1402.0)
        self.assertEqual(memory["device_peak_mib"], {"median": 5002.0, "min": 5001.0, "max": 5003.0})
        self.assertEqual(memory["device_source"], "per-process")
        self.assertTrue(memory["host_peak_rss_mib"]["gpu"]["peak_covers_timed_region_only"])

    def test_partially_recorded_memory_is_not_averaged(self):
        # Half-instrumented repeats must not be pooled into a median that reads
        # like a measurement of the whole arm.
        path = self.root / "water-cc-pvdz-gpu-2" / "result.json"
        record = json.loads(path.read_text())
        del record["device_memory"]
        path.write_text(json.dumps(record))
        memory = summarize(self.root)["rows"][0]["memory"]
        self.assertIsNone(memory["device_peak_mib"]["median"])

    def test_mixed_device_accounting_is_flagged_not_collapsed(self):
        # Device-wide use counts other tenants; per-process does not. A median
        # across both would compare two different quantities.
        path = self.root / "water-cc-pvdz-gpu-2" / "result.json"
        record = json.loads(path.read_text())
        record["device_memory"]["source"] = "device-wide"
        path.write_text(json.dumps(record))
        self.assertEqual(summarize(self.root)["rows"][0]["memory"]["device_source"],
                         ["device-wide", "per-process"])

    def test_unreset_host_peak_is_reported_as_whole_process(self):
        path = self.root / "water-cc-pvdz-gpu-2" / "result.json"
        record = json.loads(path.read_text())
        record["host_memory"]["peak_covers_timed_region_only"] = False
        path.write_text(json.dumps(record))
        summary = summarize(self.root)
        self.assertFalse(summary["rows"][0]["memory"]["host_peak_rss_mib"]["gpu"]
                         ["peak_covers_timed_region_only"])
        self.assertIn("whole process", markdown(summary))

    def test_results_recorded_before_memory_instrumentation_still_summarize(self):
        for path in self.root.glob("water-cc-pvdz-*/result.json"):
            record = json.loads(path.read_text())
            record.pop("host_memory", None)
            record.pop("device_memory", None)
            path.write_text(json.dumps(record))
        summary = summarize(self.root)
        self.assertTrue(summary["all_pass"])
        self.assertIsNone(summary["rows"][0]["memory"]["device_peak_mib"]["median"])
        self.assertIn("—", markdown(summary))

    def test_memory_table_rendered(self):
        text = markdown(summarize(self.root))
        self.assertIn("## Memory", text)
        self.assertIn("5002 [5001–5003]", text)
        self.assertIn("2002 [2001–2003]", text)

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
