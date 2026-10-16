import json
from pathlib import Path
import tempfile
import unittest

from control_report import data_for_control, render


class ControlReportTests(unittest.TestCase):
    def test_full_process_not_internal_clock_and_measured_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "protein83-panel"
            campaign = base / "protein83/pbe/stored-df/aug-cc-pvtz/campaign"
            campaign.mkdir(parents=True)
            values = dict(succeeded=3, full_process_wall_s=dict(median=12., min=11., max=13.),
                          calculation_wall_s=dict(median=1.),
                          memory=dict(peak_rss_kib=dict(max=1024**2),
                                      sampled_gpu_device_peak_mib=dict(max=2048),
                                      sampled_gpu_process_peak_mib=dict(max=1536)))
            summary = dict(complete=True, comparison_eligible=True, actual_grid_audit_verified=True,
                           energy_delta_pyscf_minus_psi4_eh=1e-9, psi4=values, pyscf=values)
            case = dict(system="protein83", functional="pbe", df_mode="stored-df",
                        source_grid=dict(nonzero_weight_points=90),
                        basis_runs={"aug-cc-pvtz": summary})
            (base / "control-results.json").write_text(json.dumps([case]))
            for engine in ("psi4", "pyscf"):
                (campaign / f"{engine}.record.json").write_text(json.dumps(
                    dict(engine=engine, nbf=50, nmo=49, grid_points=100 if engine == "psi4" else 104)))
            data = data_for_control(root)
            row = data["rows"][0]
            self.assertEqual(row["psi4"]["full_process_s"], 12.)
            self.assertEqual(row["psi4"]["peak_rss_gib"], 1.)
            self.assertEqual(row["pyscf"]["sampled_gpu_device_gib"], 2.)
            self.assertEqual(row["psi4"]["nbf"], 50)
            self.assertEqual(row["pyscf"]["stored_grid_points"], 104)
            self.assertEqual(row["nonzero_grid_points"], 90)
            self.assertTrue(row["eligible"])
            # Successful diagnostics without the four-arm audit cannot become a timed pair.
            summary["actual_grid_audit_verified"] = False
            (base / "control-results.json").write_text(json.dumps([case]))
            failed = data_for_control(root)["rows"][0]
            self.assertFalse(failed["eligible"])
            self.assertIsNone(failed["psi4"]["full_process_s"])
            self.assertIsNone(failed["energy_delta_eh"])

    def test_template_data_is_embedded_and_demo_is_removed(self):
        template = ('<title>TOPIC</title><style></style><main>demo</main>'
                    '<script type="application/json" id="data">{}</script><script>'
                    '/* demo figure: */ throw Error("demo"); /* code panes: */</script>')
        page = render(dict(rows=[], timing="t", memory="m"), template, "abcdef")
        self.assertNotIn('throw Error("demo")', page)
        self.assertIn('"rows":[]', page)
        self.assertIn("abcdef", page)
        self.assertIn("full_process_s", page)
        self.assertNotIn("ANALYSIS_REVISION", page)


if __name__ == "__main__":
    unittest.main()
