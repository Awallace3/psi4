from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from memory_report import aggregate, campaign_rows


class MemoryReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.save("manifest.json", {"config": {"basis": "aug-cc-pvtz"},
                                    "gpu_identity": {"uuid": "gpu"}})
        stamp = datetime(2026, 10, 13, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        self.record = dict(system="protein157", functional="pbe", engine="psi4", repeat=0,
                           profile=False, ok=False, process_returncode=1,
                           started_unix_s=stamp, ended_unix_s=stamp + 2)
        self.save("records.json", [self.record])
        (self.base / "gpu-samples.csv").write_text(
            "timestamp, uuid, memory.used [MiB]\n"
            "2026/10/13 17:00:01.000, gpu, 2000 MiB\n"
            "2026/10/13 17:00:03.000, gpu, 9000 MiB\n")

    def save(self, name, value):
        (self.base / name).write_text(json.dumps(value))

    def test_historical_crash_rss_is_missing_not_zero(self):
        rows, hashes = campaign_rows(self.base)
        self.assertIsNone(rows[0]["peak_rss_gib"])
        self.assertFalse(rows[0]["ok"])
        self.assertEqual(rows[0]["sampled_gpu_device_peak_mib"], 2000)
        self.assertEqual(rows[0]["gpu_sample_count"], 1)
        self.assertEqual(len(hashes), 3)
        self.assertEqual(aggregate(rows)[0]["rss_observed"], 0)

    def test_supervisor_rss_and_process_memory_are_reported(self):
        self.record["memory"] = {"peak_rss_kib": 1024**2, "peak_rss_source": "supervisor",
                                 "sampled_gpu_device_peak_mib": 6000,
                                 "sampled_gpu_process_peak_mib": 5000, "samples": 9,
                                 "metric_sample_counts": {"gpu_device_mib": 4, "gpu_process_mib": 2}}
        self.save("records.json", [self.record])
        rows, _ = campaign_rows(self.base)
        self.assertEqual(rows[0]["peak_rss_gib"], 1)
        self.assertEqual(rows[0]["sampled_gpu_process_peak_mib"], 5000)
        self.assertEqual(rows[0]["sampled_gpu_device_peak_mib"], 6000)
        self.assertEqual(rows[0]["gpu_sample_count"], 4)
        self.assertEqual(rows[0]["gpu_process_sample_count"], 2)

    def test_fallback_gpu_count_and_rss_errors_are_preserved(self):
        self.record["memory"] = {"samples": 99, "sampled_gpu_device_peak_mib": None,
                                 "peak_rss_error": "Supervisor was killed",
                                 "metric_sample_counts": {"gpu_device_mib": 0},
                                 "gpu_query_errors": 99}
        self.save("records.json", [self.record])
        rows, _ = campaign_rows(self.base)
        self.assertEqual(rows[0]["gpu_sample_count"], 1)
        self.assertIn("Archived", rows[0]["gpu_device_sample_source"])
        self.assertEqual(rows[0]["peak_rss_error"], "Supervisor was killed")

    def test_profiles_excluded_from_aggregate(self):
        self.record["maxrss_kib"] = 1024**2
        self.save("records.json", [self.record, dict(self.record, profile=True, maxrss_kib=99*1024**2)])
        rows, _ = campaign_rows(self.base)
        summary = aggregate(rows)[0]
        self.assertEqual(summary["attempts"], 1)
        self.assertEqual(summary["rss_peak_gib"], 1)

    def test_wrong_uuid_is_rejected(self):
        path = self.base / "gpu-samples.csv"
        path.write_text(path.read_text().replace(", gpu,", ", wrong,"))
        with self.assertRaises(ValueError):
            campaign_rows(self.base)


if __name__ == "__main__":
    unittest.main()
