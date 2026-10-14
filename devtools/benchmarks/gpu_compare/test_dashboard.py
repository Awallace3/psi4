import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        records = [
            dict(system="protein157", functional="pbe", engine="psi4", profile=False,
                 ok=False, process_returncode=1, secret="DO_NOT_PUBLISH"),
            dict(system="protein157", functional="pbe", engine="pyscf", profile=False,
                 ok=True, process_returncode=0, calculation_wall_s=161.3),
        ]
        for name, value in (
                ("records.json", records),
                ("config.json", {"basis": "aug-cc-pvtz"}),
                ("summary.json", [{"system": "protein157", "functional": "pbe", "rung": "gga"}])):
            (self.base / name).write_text(json.dumps(value))
        memory = [dict(system="protein157", functional="pbe", engine=engine, profile=False,
                       peak_rss_gib=None if engine == "psi4" else 1.27,
                       sampled_gpu_device_peak_mib=1024)
                  for engine in ("psi4", "pyscf")]
        self.mock = patch.object(dashboard, "campaign_rows", return_value=(memory, {}))
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_failed_arm_has_no_timing_and_missing_rss(self):
        row = dashboard.data_for([self.base])[0]
        self.assertIsNone(row["psi4"]["calculation_s"])
        self.assertIsNone(row["psi4"]["peak_rss_gib"])
        self.assertEqual(row["pyscf"]["calculation_s"], 161.3)

    def test_share_safe_output_has_no_raw_paths_or_extra_record_fields(self):
        page = dashboard.render(dashboard.data_for([self.base]))
        self.assertNotIn("DO_NOT_PUBLISH", page)
        self.assertNotIn(str(self.base), page)
        self.assertNotIn("GPU-f824", page)
        self.assertIn("failed arm", page)
        self.assertIn("not a SLURM poll", page)


if __name__ == "__main__":
    unittest.main()
