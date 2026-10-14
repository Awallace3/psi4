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
        self.assertIn("all ratios are provisional", page)
        self.assertIn("PySCF stored grid points", page)
        self.assertIn("Grid-controlled comparison pending", page)

    def test_default_dark_theme_and_larger_type(self):
        page = dashboard.render(dashboard.data_for([self.base]))
        self.assertIn('<html lang="en" class="dark" data-theme="dark">', page)
        self.assertIn('aria-pressed="true">Light theme', page)
        self.assertIn('font:20px/1.6', page)
        self.assertIn('font-size:17px', page)

    def test_measured_plots_with_embedded_data_and_no_external_assets(self):
        page = dashboard.render(dashboard.data_for([self.base]))
        for identifier in ("time-plot", "rss-plot", "gpu-plot"):
            self.assertIn(f'id="{identifier}"', page)
        self.assertNotIn("cdn.jsdelivr.net", page)
        self.assertIn('id="functional"', page)
        self.assertIn('value=\'pbe\' selected', page)
        self.assertIn('tag measured', page)
        text = page.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
        data = json.loads(text)
        self.assertIsNone(data[0]["psi4"]["calculation_s"])
        self.assertIsNone(data[0]["psi4"]["peak_rss_gib"])
        self.assertEqual(data[0]["pyscf"]["calculation_s"], 161.3)
        self.assertNotIn("secret", text)

    def test_template_demo_content_is_not_published(self):
        page = dashboard.render(dashboard.data_for([self.base]))
        self.assertNotIn("renderDemo()", page)
        self.assertNotIn("WORKTREE@SHA", page)
        self.assertNotIn('id="fig-demo"', page)

    def test_failed_grid_count_can_be_recovered_from_constructed_grid_log(self):
        directory = self.base / "results" / "failed"
        directory.mkdir(parents=True)
        (directory / "pyscf.out").write_text("tot grids = 4848128\n")
        count, source = dashboard.grid_count(
            {"engine": "pyscf", "output": "/remote/campaign/results/failed"}, self.base)
        self.assertEqual(count, 4848128)
        self.assertEqual(source, "constructed grid (log)")

    def test_host_zero_grid_cannot_be_recovered_as_actual_cuest_grid(self):
        directory = self.base / "results" / "psi4"
        directory.mkdir(parents=True)
        (directory / "psi4.out").write_text("Total Points = 0\n")
        count, source = dashboard.grid_count(
            {"engine": "psi4", "output": "/remote/results/psi4"}, self.base)
        self.assertIsNone(count)


if __name__ == "__main__":
    unittest.main()
