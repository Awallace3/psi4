import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.save("config.json", {})
        config_hash = hashlib.sha256((self.base / "config.json").read_bytes()).hexdigest()
        self.records, self.rows = [], []
        for system in ("peptide", "nanotube", "protein83"):
            for functional in ("svwn", "pbe", "m06-l", "b3lyp", "wb97x"):
                engine_summary = {"calculation_wall_s": {"median": 2, "min": 1, "max": 3},
                                  "process_wall_s": {"median": 4},
                                  "scf_iterations": {"median": 12}}
                self.rows.append(dict(system=system, functional=functional, complete=True,
                                      comparison_eligible=True,
                                      psi4_over_pyscf_calculation_time=.5,
                                      energy_delta_pyscf_minus_psi4_eh=0,
                                      psi4=copy.deepcopy(engine_summary),
                                      pyscf=copy.deepcopy(engine_summary)))
                for engine in ("psi4", "pyscf"):
                    for repeat, profile in [(0, False), (1, False), (2, False), ("profile", True)]:
                        timers = {name: {"timer_name": name, "wall_time": 1, "n_calls": 2}
                                  for name in ("JK: JK", "RV: Form V",
                                               "cuEST XC: CUDA LibXC Functional")}
                        self.records.append(dict(system=system, functional=functional,
                            engine=engine, repeat=repeat, profile=profile, ok=True,
                            process_returncode=0, gpu={"uuid": "gpu"}, config_sha256=config_hash,
                            maxrss_kib=1024**2, started_unix_s=0, ended_unix_s=2e9,
                            timer_records=timers, diagnostic_timers={
                                name: {"wall_s": 1, "calls": 2} for name in ("jk", "xc", "df_setup")}))
        self.save("records.json", self.records)
        self.save("summary.json", self.rows)
        self.save("COMPLETE.json", {"complete": True, "final_build_hashes_unchanged": True})
        self.save("manifest.json", {"systems": ["peptide", "nanotube", "protein83"],
                                   "config": {}, "gpu": "name, uuid\nH200, gpu\n"})
        (self.base / "gpu-samples.csv").write_text(
            "timestamp, uuid, memory.used [MiB]\n2026/10/13 09:29:32.185, gpu, 100 MiB\n")
        self.mock = patch.object(report, "summarize", return_value=copy.deepcopy(self.rows[::-1]))
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def save(self, name, obj):
        (self.base / name).write_text(json.dumps(obj))

    def test_replay_ignores_order_and_preserves_all_rows(self):
        data = report.audit(self.base)
        self.assertEqual(len(data["rows"]), 15)
        self.assertEqual(data["rows"][0]["psi4_speedup"], 2)
        self.assertEqual(data["rows"][0]["psi4"]["sampled_whole_device_max_mib"], 100)
        self.assertEqual(report.html_report(data).count("data-system="), 15)
        self.assertIn("Protein83 profiling", report.markdown(data))

    def test_failed_process_is_rejected(self):
        self.records[0]["process_returncode"] = 1
        self.save("records.json", self.records)
        with self.assertRaisesRegex(ValueError, "failed workers"):
            report.audit(self.base)

    def test_duplicate_worker_is_rejected(self):
        self.records[-1] = self.records[0]
        self.save("records.json", self.records)
        with self.assertRaisesRegex(ValueError, "Duplicate workers"):
            report.audit(self.base)

    def test_config_mismatch_is_rejected(self):
        self.records[0]["config_sha256"] = "invalid"
        self.save("records.json", self.records)
        with self.assertRaisesRegex(ValueError, "Config hash"):
            report.audit(self.base)

    def test_manifest_config_mismatch_is_rejected(self):
        self.save("config.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Manifest config"):
            report.audit(self.base)

    def test_summary_changes_are_rejected(self):
        self.rows[0]["psi4_over_pyscf_calculation_time"] = 99
        self.save("summary.json", self.rows)
        with self.assertRaisesRegex(ValueError, "Summary replay"):
            report.audit(self.base)

    def test_sample_uuid_mismatch_is_rejected(self):
        sample = self.base / "gpu-samples.csv"
        sample.write_text(sample.read_text().replace(", gpu,", ", other,"))
        with self.assertRaisesRegex(ValueError, "Sample GPU"):
            report.audit(self.base)

    def test_duplicate_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate summary"):
            report.require_summary_equal(self.rows, self.rows + [self.rows[0]])


if __name__ == "__main__":
    unittest.main()
