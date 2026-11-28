import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("phoenix_timing")
suite = load("phoenix_suite_timing")
submit = load("submit_phoenix_suite")
worker = load("suite_fsapt_worker")


class PhoenixSuiteContracts(unittest.TestCase):
    def test_single_job_embers_8h_command(self):
        argv = submit.command(Path("/source"), Path("/run"), "15960394")
        for flag in (
            "--job-name=fsapt-suite-h200",
            "--account=gts-cs207-chemx",
            "--qos=embers",
            "--partition=gpu-h200",
            "--constraint=H200",
            "--gres=gpu:h200:1",
            "--cpus-per-task=8",
            "--mem=160G",
            "--time=08:00:00",
            "--dependency=afterany:15960394",
            "--kill-on-invalid-dep=yes",
            "--no-requeue",
            "--signal=TERM@120",
        ):
            self.assertIn(flag, argv)
        joined = " ".join(argv)
        self.assertNotIn("afterok", joined)
        self.assertNotIn("inferno", joined)
        with self.assertRaises(ValueError):
            submit.command(Path("/source"), Path("/run"), "15960394:999")

    def test_completed_build_omits_purged_dependency(self):
        argv = submit.command(Path("/source"), Path("/run"), None)
        joined = " ".join(argv)
        self.assertNotIn("--dependency", joined)
        self.assertNotIn("--kill-on-invalid-dep", joined)
        self.assertIn("--qos=embers", argv)
        self.assertIn("--partition=gpu-h200", argv)
        self.assertIn("--time=08:00:00", argv)

    def test_blackwell_inferno_command(self):
        argv = submit.command(
            Path("/source"), Path("/run"), None,
            device_family="RTX_PRO_BLACKWELL", qos="inferno", account="gts-cs207", hours=8,
        )
        joined = " ".join(argv)
        self.assertIn("--job-name=fsapt-suite-rtx6000", argv)
        self.assertIn("--account=gts-cs207", argv)
        self.assertIn("--qos=inferno", argv)
        self.assertIn("--partition=gpu-rtxpro-blackwell", argv)
        self.assertIn("--gres=gpu:rtx_pro_6000_blackwell:1", argv)
        self.assertNotIn("--constraint=", joined)
        self.assertNotIn("--dependency", joined)

    def test_summarize_stage_status_preserves_skipped(self):
        summary = suite.summarize_stage_status({
            "stage1_small_gate": {"status": "PASS"},
            "stage6_protein157_gpu_probe": {"status": "SKIPPED_BY_MANIFEST"},
        })
        self.assertEqual(summary["stage1_small_gate"], "PASS")
        self.assertEqual(summary["stage6_protein157_gpu_probe"], "SKIPPED_BY_MANIFEST")

    def test_compare_fsapt_tables_passes_and_rejects_drift(self):
        base = {
            "charge": {
                "full": {"Elst": {"OH": {"OH": -8.736, "PH": 2.473}}, "Disp": {"OH": {"OH": -1.115, "PH": -0.659}}},
                "reduced": {"Elst": {"OH": {"OH": -8.282, "PH": 1.307}}},
            },
            "links5050": {
                "full": {"Total": {"OH": {"OH": -5.721, "PH": 2.408}}},
                "reduced": {"Total": {"OH": {"OH": -5.450, "PH": 1.159}}},
            },
        }
        near = {
            "charge": {
                "full": {"Elst": {"OH": {"OH": -8.736 + 1e-6, "PH": 2.473}}, "Disp": {"OH": {"OH": -1.115, "PH": -0.659}}},
                "reduced": {"Elst": {"OH": {"OH": -8.282, "PH": 1.307}}},
            },
            "links5050": {
                "full": {"Total": {"OH": {"OH": -5.721, "PH": 2.408}}},
                "reduced": {"Total": {"OH": {"OH": -5.450, "PH": 1.159}}},
            },
        }
        summary = suite.compare_fsapt_tables(base, near)
        self.assertEqual(summary["cells_checked"], 10)
        self.assertLess(summary["max_abs_diff_kcal_mol"], 2e-6)
        self.assertLess(summary["max_abs_diff_Eh"], 1e-8)

        drifted = {
            **near,
            "links5050": {
                "full": {"Total": {"OH": {"OH": -5.721 + 0.01, "PH": 2.408}}},
                "reduced": {"Total": {"OH": {"OH": -5.450, "PH": 1.159}}},
            },
        }
        with self.assertRaises(AssertionError):
            suite.compare_fsapt_tables(base, drifted)

    def test_fsapt1_geometry_and_groups(self):
        atom_lines = [
            line for line in worker.PHENOL_DIMER.splitlines()
            if line.strip() and line.strip()[0] in ("C", "H", "O") and len(line.split()) == 4
        ]
        self.assertEqual(len(atom_lines), 26)
        self.assertEqual(worker.PHENOL_DIMENSIONS["atoms"], 26)
        self.assertEqual(worker.PHENOL_DIMENSIONS["ao"], 312)
        self.assertIn("OH 1 2", worker.PHENOL_FA)
        self.assertIn("PH 16 17 18", worker.PHENOL_FB)

    def test_extract_step_timers_optional(self):
        before = {"SAPT(DFT):Dimer SCF": {"wall_time": 1.0, "n_calls": 1}}
        after = {
            "SAPT(DFT):Dimer SCF": {"wall_time": 3.5, "n_calls": 2},
            "SAPT(DFT) Energy;SAPT(DFT): F-SAPT Dispersion": {"wall_time": 0.4, "n_calls": 1},
        }
        out = worker.extract_step_timers(before, after)
        self.assertAlmostEqual(out["scf_dimer"], 2.5)
        self.assertAlmostEqual(out["fragment_disp"], 0.4)
        self.assertIsNone(out["gpu_fsapt_total"])

    def test_suite_main_imports_owned_process_from_science_source(self):
        text = (HERE / "phoenix_suite_timing.py").read_text()
        self.assertIn(
            'sys.path.insert(0, str(Path(manifest["science_source"]) / "devtools/disp0/phoenix_validation"))',
            text,
        )
        self.assertNotIn(
            'sys.path.insert(0, str(Path(manifest["helper_source"]) / "devtools/disp0/phoenix_validation"))',
            text,
        )


if __name__ == "__main__":
    unittest.main()
